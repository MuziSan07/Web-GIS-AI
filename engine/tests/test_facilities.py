"""Tests for facility siting and catchment coverage."""

from __future__ import annotations

import pytest

from planeng.allocate import allocate_land
from planeng.analysis import analyse_constraints, compute_suitability
from planeng.brief import DensityMix, PlanningBrief, derive_programme
from planeng.facilities import site_facilities
from planeng.facilities.siting import (
    CATEGORY_ZONE,
    MIN_SEPARATION_FRACTION,
    PERIPHERAL,
    _carve,
)
from planeng.roads import build_blocks, generate_network
from planeng.site import generate_site
from planeng.standards import load_standards
from shapely.geometry import box


@pytest.fixture(scope="module")
def std():
    return load_standards("botswana")


@pytest.fixture(scope="module")
def ctx(std):
    site = generate_site(target_area_ha=600, cell_size=15, seed=3)
    brief = PlanningBrief(
        project_name="Facilities", site_area_ha=600, target_population=15000,
        density_mix=DensityMix(low=0.05, medium=0.35, high=0.60),
    )
    prog = derive_programme(brief, std)
    cons = analyse_constraints(site, std)
    suit = compute_suitability(site, std, cons)
    alloc = allocate_land(site, prog, suit, cons)
    net = generate_network(site, std, cons, alloc, prog)
    blocks = build_blocks(site, std, alloc, net, prog)
    fp = site_facilities(site, std, prog, blocks, alloc)
    return dict(site=site, prog=prog, cons=cons, alloc=alloc,
                net=net, blocks=blocks, fp=fp)


# -- placement --------------------------------------------------------------

def test_facilities_are_actually_sited(ctx):
    assert ctx["fp"].facilities
    assert ctx["fp"].summary()["sited_total"] > 0


def test_never_sites_more_than_the_schedule_requires(ctx):
    """Over-provision would quietly inflate the land budget."""
    required = {f.key: f.count for f in ctx["prog"].facilities}
    for row in ctx["fp"].summary()["by_type"]:
        assert row["sited"] <= required.get(row["key"], 0), row["key"]


def test_sited_plus_unplaced_equals_the_requirement(ctx):
    """Nothing may be silently dropped from the schedule."""
    required = {f.key: f.count for f in ctx["prog"].facilities}
    fp = ctx["fp"]
    for key, count in required.items():
        if count <= 0:
            continue
        sited = len(fp.of_type(key))
        assert sited + fp.unplaced.get(key, 0) == count, key


def test_facilities_land_on_non_residential_blocks(ctx):
    """A school sited on top of housing plots is double-counted land."""
    fp, blocks = ctx["fp"], ctx["blocks"]
    residential = [b.geometry for b in blocks.for_use("residential")]
    for f in fp.facilities:
        overlap = sum(f.geometry.intersection(g).area for g in residential)
        assert overlap < f.geometry.area * 0.05, f.facility_id


def test_facilities_do_not_overlap_each_other(ctx):
    facs = ctx["fp"].facilities
    for i, a in enumerate(facs):
        for b in facs[i + 1:]:
            overlap = a.geometry.intersection(b.geometry).area
            assert overlap < min(a.geometry.area, b.geometry.area) * 0.02


def test_same_type_facilities_are_separated(ctx):
    """Two clinics side by side serve one place twice."""
    fp = ctx["fp"]
    for row in fp.summary()["by_type"]:
        facs = fp.of_type(row["key"])
        if len(facs) < 2 or not row["catchment_m"]:
            continue
        minimum = row["catchment_m"] * MIN_SEPARATION_FRACTION
        for i, a in enumerate(facs):
            for b in facs[i + 1:]:
                d = a.geometry.centroid.distance(b.geometry.centroid)
                assert d >= minimum * 0.95, f"{row['key']}: {d:.0f} m apart"


def test_cemetery_is_pushed_to_the_periphery(ctx):
    """Peripheral uses must not be sited in the middle of the settlement."""
    fp, site = ctx["fp"], ctx["site"]
    cemeteries = fp.of_type("cemetery")
    if not cemeteries:
        pytest.skip("no cemetery sited")
    assert "cemetery" in PERIPHERAL

    centre = site.boundary.centroid
    others = [f for f in fp.facilities if f.key != "cemetery"]
    if not others:
        pytest.skip("nothing to compare against")

    cem_d = min(f.geometry.centroid.distance(centre) for f in cemeteries)
    median_d = sorted(f.geometry.centroid.distance(centre) for f in others)[len(others) // 2]
    assert cem_d > median_d


def test_large_facilities_are_placed_before_small_ones(ctx):
    """A 5 ha school has few candidate sites; small ones would fragment them."""
    fp = ctx["fp"]
    schools = fp.of_type("secondary_school")
    posts = fp.of_type("health_post")
    if schools and posts:
        assert schools[0].geometry.area > posts[0].geometry.area


# -- coverage ---------------------------------------------------------------

def test_coverage_is_a_percentage(ctx):
    for key, pct in ctx["fp"].coverage.items():
        assert 0.0 <= pct <= 100.0, (key, pct)


def test_facilities_without_a_catchment_report_no_coverage(ctx):
    """Regression: a substation has no catchment, so 0% read as a failure."""
    fp = ctx["fp"]
    for row in fp.summary()["by_type"]:
        if not row["catchment_m"]:
            assert row["coverage_pct"] is None, row["key"]
        elif row["sited"]:
            assert row["coverage_pct"] is not None, row["key"]


def test_wider_catchments_cover_more(ctx):
    """A sanity check on the coverage metric itself."""
    rows = [r for r in ctx["fp"].summary()["by_type"]
            if r["coverage_pct"] is not None and r["sited"] >= 1]
    tight = [r for r in rows if r["catchment_m"] <= 800]
    wide = [r for r in rows if r["catchment_m"] >= 3000]
    if tight and wide:
        assert (sum(r["coverage_pct"] for r in wide) / len(wide)
                >= sum(r["coverage_pct"] for r in tight) / len(tight))


def test_poor_coverage_is_declared(ctx):
    """Meeting the count while failing the distribution must be surfaced."""
    fp = ctx["fp"]
    poor = [k for k, pct in fp.coverage.items() if pct < 70]
    for key in poor:
        assert any(key.replace("_", " ") in w and "catchment" in w
                   for w in fp.warnings), key


def test_unplaced_facilities_are_declared(ctx):
    fp = ctx["fp"]
    for key in fp.unplaced:
        assert any(key.replace("_", " ") in w and "could not be placed" in w
                   for w in fp.warnings), key


# -- carving ----------------------------------------------------------------

def test_carve_returns_the_requested_area():
    block = box(0, 0, 400, 400)          # 16 ha
    site = _carve(block, 20_000)         # want 2 ha
    assert site.area == pytest.approx(20_000, rel=0.15)
    assert site.within(block.buffer(0.01))


def test_carve_returns_the_whole_block_when_it_is_small_enough():
    block = box(0, 0, 100, 100)          # 1 ha
    assert _carve(block, 9_500).equals(block)


def test_carve_stays_inside_an_irregular_block():
    block = box(0, 0, 300, 300).difference(box(200, 200, 300, 300))
    site = _carve(block, 10_000)
    assert site.within(block.buffer(0.01))
    assert site.geom_type == "Polygon"


def test_every_category_maps_to_a_zone(ctx):
    categories = {f.category for f in ctx["prog"].facilities}
    for c in categories:
        assert c in CATEGORY_ZONE, c


# -- integration ------------------------------------------------------------

def test_facilities_reach_the_geojson(ctx):
    from planeng.export.geojson import plan_to_geojson
    fc = plan_to_geojson(ctx["site"], ctx["cons"], ctx["alloc"],
                         network=ctx["net"], blocks=ctx["blocks"],
                         facility_plan=ctx["fp"])
    feats = [f for f in fc["features"] if f["properties"]["layer"] == "facility"]
    assert len(feats) == len(ctx["fp"].facilities)
    assert all(f["properties"]["area_ha"] > 0 for f in feats)
    assert all(f["properties"]["key"] for f in feats)
