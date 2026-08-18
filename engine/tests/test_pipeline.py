"""Tests for the spatial pipeline: site, constraints, suitability, allocation."""

from __future__ import annotations

import numpy as np
import pytest

from planeng.allocate.zones import UNASSIGNED, allocate_land
from planeng.analysis.constraints import analyse_constraints, buildable_area_ha
from planeng.analysis.suitability import compute_suitability
from planeng.analysis.terrain import slope_percent
from planeng.brief import DensityMix, PlanningBrief, derive_programme
from planeng.crs import require_projected, utm_crs_for_longitude
from planeng.export.geojson import plan_to_geojson
from planeng.pipeline import generate_master_plan
from planeng.site import generate_site
from planeng.standards import load_standards


@pytest.fixture(scope="module")
def std():
    return load_standards("botswana")


@pytest.fixture(scope="module")
def site():
    return generate_site(target_area_ha=600.0, cell_size=20.0, seed=7)


@pytest.fixture(scope="module")
def brief():
    return PlanningBrief(
        project_name="Test Town", site_area_ha=600.0, target_population=15000,
        density_mix=DensityMix(low=0.05, medium=0.35, high=0.60),
    )


# -- CRS --------------------------------------------------------------------

def test_geographic_crs_is_rejected():
    """Degree-based area is the classic silent GIS failure. Fail loudly instead."""
    with pytest.raises(ValueError, match="projected CRS"):
        require_projected("EPSG:4326")


def test_utm_zone_selection_for_botswana():
    assert utm_crs_for_longitude(25.9) == "EPSG:32735"   # Gaborone, zone 35S
    assert utm_crs_for_longitude(22.0) == "EPSG:32734"   # western Botswana, 34S


# -- site -------------------------------------------------------------------

def test_site_area_matches_request(site):
    assert site.area_ha == pytest.approx(600.0, rel=0.001)


def test_site_is_projected_and_metric(site):
    assert require_projected(site.crs).is_projected


def test_site_is_deterministic():
    a = generate_site(target_area_ha=600, cell_size=20, seed=99)
    b = generate_site(target_area_ha=600, cell_size=20, seed=99)
    assert np.array_equal(a.dem, b.dem)
    assert a.boundary.equals(b.boundary)


def test_cell_area_conversions_round_trip(site):
    assert site.cells_to_ha(site.ha_to_cells(25.0)) == pytest.approx(25.0, abs=0.5)


def test_dem_shape_mismatch_is_rejected(site):
    from planeng.site.model import Site
    with pytest.raises(ValueError, match="does not match"):
        Site(name="bad", boundary=site.boundary, crs=site.crs,
             transform=site.transform, shape=site.shape,
             dem=np.zeros((3, 3)))


# -- terrain ----------------------------------------------------------------

def test_slope_is_non_negative_and_finite(site):
    s = slope_percent(site)
    assert s.shape == site.shape
    assert np.all(s >= 0) and np.all(np.isfinite(s))


def test_slope_of_a_plane_matches_its_gradient():
    """A 5 m rise over 100 m must read as 5%."""
    s = generate_site(target_area_ha=100, cell_size=10, seed=1)
    rows, cols = s.shape
    xx = np.arange(cols) * s.cell_size
    s.dem = np.tile(xx * 0.05, (rows, 1))     # exactly 5% east-facing
    computed = slope_percent(s)
    assert computed[2:-2, 2:-2].mean() == pytest.approx(5.0, abs=0.01)


# -- constraints ------------------------------------------------------------

def test_constraints_reduce_buildable_land(site, std):
    res = analyse_constraints(site, std)
    buildable = buildable_area_ha(site, res)
    assert 0 < buildable < site.area_ha


def test_every_exclusion_cites_a_rule(site, std):
    """A planner must be able to ask why any cell was excluded."""
    res = analyse_constraints(site, std)
    for key in res.exclusions:
        assert res.reasons[key], f"{key} has no rule citation"
        assert "=" in res.reasons[key]


def test_buildable_never_extends_outside_the_site(site, std):
    res = analyse_constraints(site, std)
    assert not (res.buildable & ~site.mask()).any()


def test_wetland_buffer_widens_with_the_standard(site, std):
    """Raising the buffer standard must exclude more land, not less."""
    narrow = analyse_constraints(site, std)
    narrow_ha = narrow.area_ha(site, "wetland")

    std.data["environment"]["buffers"]["wetland_m"] = 200
    wide = analyse_constraints(site, std)
    wide_ha = wide.area_ha(site, "wetland")
    std.data["environment"]["buffers"]["wetland_m"] = 50   # restore

    assert wide_ha > narrow_ha


# -- suitability ------------------------------------------------------------

def test_suitability_is_bounded_zero_to_one(site, std):
    cons = analyse_constraints(site, std)
    suit = compute_suitability(site, std, cons)
    for use, surface in suit.surfaces.items():
        assert surface.min() >= 0.0 and surface.max() <= 1.0, use


def test_constrained_land_is_never_suitable(site, std):
    """Unavailable land must score zero, not merely low."""
    cons = analyse_constraints(site, std)
    suit = compute_suitability(site, std, cons)
    for use, surface in suit.surfaces.items():
        assert np.all(surface[~cons.buildable] == 0.0), use


def test_every_land_use_has_a_surface(site, std):
    cons = analyse_constraints(site, std)
    suit = compute_suitability(site, std, cons)
    for use in ("residential", "commercial", "industrial",
                "institutional", "recreation_open_space"):
        assert use in suit.surfaces


# -- allocation -------------------------------------------------------------

def test_allocation_respects_constraints(site, std, brief):
    prog = derive_programme(brief, std)
    cons = analyse_constraints(site, std)
    suit = compute_suitability(site, std, cons)
    alloc = allocate_land(site, prog, suit, cons)
    assigned = alloc.labels != UNASSIGNED
    assert not (assigned & ~cons.buildable).any(), "allocated onto constrained land"


def test_allocation_hits_targets_closely(site, std, brief):
    prog = derive_programme(brief, std)
    cons = analyse_constraints(site, std)
    suit = compute_suitability(site, std, cons)
    alloc = allocate_land(site, prog, suit, cons)
    for row in alloc.summary():
        assert abs(row["variance_pct"]) < 10, row


def test_compactness_preserves_area(site, std, brief):
    """The compactness pass swaps cells in pairs, so areas must not drift."""
    prog = derive_programme(brief, std)
    cons = analyse_constraints(site, std)
    suit = compute_suitability(site, std, cons)
    raw = allocate_land(site, prog, suit, cons, compactness_passes=0)
    smoothed = allocate_land(site, prog, suit, cons, compactness_passes=3)
    for use in raw.uses:
        assert raw.achieved_ha[use] == pytest.approx(smoothed.achieved_ha[use], abs=0.01)


def test_allocation_prefers_suitable_land(site, std, brief):
    """Each use should land on ground that suits it.

    A use that occupies most of the site cannot beat the site average by
    arithmetic - it *is* most of the average. So the strict test applies to
    uses taking a minority of the buildable land, and the dominant use is only
    required not to be pushed onto materially worse ground than average.
    """
    prog = derive_programme(brief, std)
    cons = analyse_constraints(site, std)
    suit = compute_suitability(site, std, cons)
    alloc = allocate_land(site, prog, suit, cons)

    buildable_ha = site.cells_to_ha(int(cons.buildable.sum()))
    for use in alloc.uses:
        share = alloc.achieved_ha[use] / buildable_ha
        site_mean = suit.surfaces[use][cons.buildable].mean()
        allocated = alloc.mean_suitability[use]
        if share < 0.5:
            assert allocated > site_mean, f"{use} (share {share:.0%})"
        else:
            assert allocated > site_mean * 0.95, f"{use} (share {share:.0%})"


def test_no_buildable_land_raises(site, std, brief):
    prog = derive_programme(brief, std)
    cons = analyse_constraints(site, std)
    suit = compute_suitability(site, std, cons)
    cons.buildable = np.zeros_like(cons.buildable)
    with pytest.raises(ValueError, match="No buildable land"):
        allocate_land(site, prog, suit, cons)


# -- export -----------------------------------------------------------------

def test_geojson_is_wgs84_and_near_botswana(site, std, brief):
    plan = generate_master_plan(brief, site=site, standards=std)
    fc = plan_to_geojson(site, plan.constraints, plan.allocation)
    boundary = next(f for f in fc["features"] if f["properties"]["layer"] == "boundary")
    lon, lat = boundary["geometry"]["coordinates"][0][0]
    assert 20 < lon < 30 and -27 < lat < -17, (lon, lat)


def test_geojson_areas_computed_before_reprojection(site, std, brief):
    """Areas must come from projected metres, never from degrees."""
    plan = generate_master_plan(brief, site=site, standards=std)
    fc = plan_to_geojson(site, plan.constraints, plan.allocation)
    zones = [f for f in fc["features"] if f["properties"]["layer"] == "zone"]
    assert zones
    total = sum(f["properties"]["area_ha"] for f in zones)
    assert 100 < total < 700, total


# -- pipeline ---------------------------------------------------------------

def test_pipeline_runs_end_to_end(brief, site, std):
    plan = generate_master_plan(brief, site=site, standards=std)
    r = plan.report()
    assert r["total_plots"] > 0
    assert r["land"]["buildable_ha"] < r["land"]["gross_site_ha"]
    assert r["allocation"] and r["facilities"]
    assert not r["ruleset"]["validated"]


def test_pipeline_warns_when_constraints_bite(std):
    """Scaling targets to fit must be disclosed, not silent."""
    b = PlanningBrief(project_name="Dense", site_area_ha=600,
                      target_population=25000,
                      density_mix=DensityMix(low=0.4, medium=0.4, high=0.2))
    plan = generate_master_plan(b, cell_size=25.0, standards=std)
    assert any("DOES NOT FIT" in w or "scaled down" in w for w in plan.warnings)


def test_unvalidated_ruleset_reaches_the_report(brief, site, std):
    plan = generate_master_plan(brief, site=site, standards=std)
    assert any("UNVALIDATED RULESET" in w for w in plan.report()["warnings"])
