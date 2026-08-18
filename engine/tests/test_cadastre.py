"""Tests for cadastral subdivision.

The two rules that make a plot registrable - road frontage, and dimensions from
the ruleset - are enforced here rather than assumed.
"""

from __future__ import annotations

import pytest

from planeng.allocate import allocate_land
from planeng.analysis import analyse_constraints, compute_suitability
from planeng.brief import DensityMix, PlanningBrief, derive_programme
from planeng.cadastre import subdivide
from planeng.cadastre.parcels import FRONTAGE_TOLERANCE_M, _assign_bands
from planeng.roads import build_blocks, generate_network
from planeng.roads.network import _target_plot_depth
from planeng.site import generate_site
from planeng.standards import load_standards


@pytest.fixture(scope="module")
def std():
    return load_standards("botswana")


@pytest.fixture(scope="module")
def ctx(std):
    site = generate_site(target_area_ha=600, cell_size=15, seed=5)
    brief = PlanningBrief(
        project_name="Cadastre", site_area_ha=600, target_population=12000,
        density_mix=DensityMix(low=0.10, medium=0.40, high=0.50),
    )
    prog = derive_programme(brief, std)
    cons = analyse_constraints(site, std)
    suit = compute_suitability(site, std, cons)
    alloc = allocate_land(site, prog, suit, cons)
    net = generate_network(site, std, cons, alloc, prog)
    blocks = build_blocks(site, std, alloc, net, prog)
    cad = subdivide(site, std, blocks, net, prog)
    return dict(site=site, std=std, prog=prog, cons=cons, alloc=alloc,
                net=net, blocks=blocks, cad=cad)


# -- the registrability rules ----------------------------------------------

def test_every_plot_fronts_a_road(ctx):
    """A landlocked plot cannot be registered, serviced or sold."""
    net, cad = ctx["net"], ctx["cad"]
    assert cad.parcels
    landlocked = [p for p in cad.parcels
                  if p.geometry.distance(net.reserve_polygon) > FRONTAGE_TOLERANCE_M]
    assert not landlocked, f"{len(landlocked)} plots have no road frontage"


def test_no_plot_is_below_its_minimum_area(ctx):
    """A plot under the band minimum is a compliance failure, not a small plot."""
    cad, std = ctx["cad"], ctx["std"]
    for p in cad.parcels:
        minimum = std.get(f"residential.densities.{p.band}.plot_area_min_m2")
        assert p.area_m2 >= minimum - 0.5, f"{p.plot_number}: {p.area_m2:.0f} m2"


def test_mean_plot_size_tracks_the_standard(ctx):
    """Regression: stretching rows to fill deep blocks inflated every plot.

    High-density plots came out at 677 m2 against a 350 m2 standard, so the
    layout silently breached the ruleset it was generated from.
    """
    cad, std = ctx["cad"], ctx["std"]
    for row in cad.summary()["by_band"]:
        typical = std.get(
            f"residential.densities.{row['band']}.plot_area_typical_m2")
        assert row["mean_plot_m2"] == pytest.approx(typical, rel=0.20), row


def test_plots_do_not_overlap_road_reserves(ctx):
    net, cad = ctx["net"], ctx["cad"]
    for p in cad.parcels[:250]:
        assert p.geometry.intersection(net.reserve_polygon).area < p.area_m2 * 0.02


def test_plots_do_not_overlap_each_other(ctx):
    """Overlapping parcels cannot be registered as separate title."""
    cad = ctx["cad"]
    by_block: dict[str, list] = {}
    for p in cad.parcels:
        by_block.setdefault(p.block_id, []).append(p)
    checked = 0
    for parcels in list(by_block.values())[:40]:
        for i, a in enumerate(parcels):
            for b in parcels[i + 1:]:
                overlap = a.geometry.intersection(b.geometry).area
                assert overlap < min(a.area_m2, b.area_m2) * 0.01
                checked += 1
    assert checked > 0


def test_plot_numbers_are_unique(ctx):
    numbers = [p.plot_number for p in ctx["cad"].parcels]
    assert len(numbers) == len(set(numbers))


# -- band assignment --------------------------------------------------------

def test_density_gradient_puts_denser_plots_nearer_the_centre(ctx):
    """Conventional practice, and what keeps the schedule deliverable."""
    site, cad = ctx["site"], ctx["cad"]
    centre = site.boundary.centroid
    dists: dict[str, list[float]] = {}
    for p in cad.parcels:
        dists.setdefault(p.band, []).append(p.geometry.centroid.distance(centre))
    means = {b: sum(v) / len(v) for b, v in dists.items() if v}
    if "high" in means and "low" in means:
        assert means["high"] < means["low"]


def test_shortfall_is_shared_across_bands_not_dumped_on_the_last(ctx, std):
    """Regression: sequential filling wiped out low-density housing.

    Bands were filled high-first until each quota was exhausted, so any
    capacity shortfall landed entirely on the last band - low density fell to
    18 plots against a target of 225.
    """
    cad, prog = ctx["cad"], ctx["prog"]
    target = {d.band: d.dwellings for d in prog.dwellings}
    actual = {r["band"]: r["plots"] for r in cad.summary()["by_band"]}

    if target.get("low", 0) > 0 and cad.total_plots > 0:
        target_share = target["low"] / sum(target.values())
        actual_share = actual.get("low", 0) / cad.total_plots
        # Low density must retain a recognisable share, not be eliminated.
        assert actual_share > target_share * 0.25, (target_share, actual_share)


def test_every_assigned_block_gets_a_valid_band(ctx, std):
    bands = set(ctx["cad"].block_bands.values())
    assert bands <= {"low", "medium", "high"}


def test_band_assignment_is_empty_without_quotas(ctx, std):
    assert _assign_bands(ctx["site"], ctx["blocks"].for_use("residential"),
                         {}, std) == {}


# -- street spacing ---------------------------------------------------------

def test_street_spacing_follows_the_brief_not_the_default(std):
    """Regression: block depth came from a constant sized for low density.

    Applying it to a high-density brief produced blocks far deeper than two
    rows of plots, so interiors were unreachable and yield collapsed.
    """
    dense = derive_programme(PlanningBrief(
        project_name="d", site_area_ha=600, target_population=12000,
        density_mix=DensityMix(low=0.0, medium=0.0, high=1.0)), std)
    sparse = derive_programme(PlanningBrief(
        project_name="s", site_area_ha=600, target_population=12000,
        density_mix=DensityMix(low=1.0, medium=0.0, high=0.0)), std)

    assert _target_plot_depth(std, dense) < _target_plot_depth(std, sparse)
    # A high-density plot of typical area at minimum frontage is this deep.
    expected = (std.get("residential.densities.high.plot_area_typical_m2")
                / std.get("residential.densities.high.frontage_min_m"))
    assert _target_plot_depth(std, dense) == pytest.approx(expected)


def test_plot_depth_falls_back_without_a_programme(std):
    assert _target_plot_depth(std, None) > 0


# -- reconciliation and known limits ---------------------------------------

def test_deep_blocks_are_reported_not_silently_left_unusable(ctx):
    warnings = ctx["cad"].warnings
    if any("deeper than two plot rows" in w for w in warnings):
        assert "internal access streets" in " ".join(warnings)


def test_yield_shortfall_is_declared(ctx):
    """The engine must state when it cannot deliver the schedule."""
    cad, prog = ctx["cad"], ctx["prog"]
    variance = (cad.total_plots - prog.total_dwellings) / prog.total_dwellings * 100
    if abs(variance) > 15:
        assert any("against a dwelling schedule" in w for w in cad.warnings)


def test_known_limitation_net_to_gross_efficiency(ctx):
    """Documents a real, unresolved gap rather than hiding it.

    The ruleset assumes 72% of residential gross becomes saleable plots. The
    current layout achieves closer to half, the losses being block cores too
    deep to serve from a frontage and irregular edge fragments. Recovering
    this needs internal access streets inside deep blocks.

    The bound is deliberately loose: it exists to catch a regression, and to
    fail loudly if a future change finally closes the gap so this test - and
    the limitation it records - can be retired.
    """
    cad, alloc = ctx["cad"], ctx["alloc"]
    plot_ha = sum(p.area_m2 for p in cad.parcels) / 10_000
    residential_ha = alloc.achieved_ha["residential"]
    efficiency = plot_ha / residential_ha
    assert 0.35 < efficiency < 0.72, f"net-to-gross now {efficiency:.0%}"


# -- integration ------------------------------------------------------------

def test_parcels_reach_the_geojson(ctx):
    from planeng.export.geojson import plan_to_geojson
    fc = plan_to_geojson(ctx["site"], ctx["cons"], ctx["alloc"],
                         network=ctx["net"], blocks=ctx["blocks"],
                         cadastre=ctx["cad"])
    parcels = [f for f in fc["features"] if f["properties"]["layer"] == "parcel"]
    assert len(parcels) == ctx["cad"].total_plots
    assert all(f["properties"]["area_m2"] > 0 for f in parcels)
    assert all(f["geometry"]["type"] == "Polygon" for f in parcels)
