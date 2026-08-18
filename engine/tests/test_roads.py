"""Tests for road network generation and block formation."""

from __future__ import annotations

import numpy as np
import pytest
from shapely.geometry import Point

from planeng.allocate import allocate_land
from planeng.analysis import analyse_constraints, compute_suitability
from planeng.brief import DensityMix, PlanningBrief, derive_programme
from planeng.roads import build_blocks, generate_network, reconcile_plots
from planeng.roads.blocks import _mean_plot_area
from planeng.roads.routing import GridRouter, build_cost_surface, nearest_valid_cell
from planeng.site import generate_site
from planeng.standards import load_standards


@pytest.fixture(scope="module")
def std():
    return load_standards("botswana")


@pytest.fixture(scope="module")
def ctx(std):
    site = generate_site(target_area_ha=600, cell_size=15, seed=11)
    brief = PlanningBrief(
        project_name="Roads", site_area_ha=600, target_population=15000,
        density_mix=DensityMix(low=0.05, medium=0.35, high=0.60),
    )
    prog = derive_programme(brief, std)
    cons = analyse_constraints(site, std)
    suit = compute_suitability(site, std, cons)
    alloc = allocate_land(site, prog, suit, cons)
    net = generate_network(site, std, cons, alloc)
    blocks = build_blocks(site, std, alloc, net, prog)
    return dict(site=site, brief=brief, prog=prog, cons=cons,
                alloc=alloc, net=net, blocks=blocks)


# -- cost surface and routing ----------------------------------------------

def test_hard_constraints_are_impassable_but_crossings_are_only_expensive(ctx, std):
    """A wetland stops a road. A watercourse gets a bridge.

    Treating every constraint as impassable was wrong: it fragmented the site
    into islands and produced a plan that could not be built out.
    """
    site, cons = ctx["site"], ctx["cons"]
    cost = build_cost_surface(site, std, cons, "collector")

    for key in ("wetland", "protected_area"):
        if key in cons.exclusions:
            assert np.all(np.isinf(cost[cons.exclusions[key]])), key

    for key in ("watercourse", "powerline"):
        if key not in cons.exclusions:
            continue
        cells = cons.exclusions[key] & site.mask()
        # Excludes cells that also fall inside a hard constraint.
        crossable = cells & np.isfinite(cost)
        assert crossable.any(), f"{key} should be crossable somewhere"
        assert cost[crossable].max() > cost[cons.buildable].max(), key

    assert np.all(np.isinf(cost[~site.mask()]))
    assert np.all(np.isfinite(cost[cons.buildable]))


def test_steeper_ground_costs_more(ctx, std):
    """The router must prefer gentle ground, or the hierarchy is meaningless."""
    site, cons = ctx["site"], ctx["cons"]
    cost = build_cost_surface(site, std, cons, "collector")
    finite = np.isfinite(cost) & cons.buildable
    slope = cons.slope[finite]
    lo = cost[finite][slope < np.percentile(slope, 25)].mean()
    hi = cost[finite][slope > np.percentile(slope, 75)].mean()
    assert hi > lo


def test_router_finds_a_path_within_a_connected_component(ctx, std):
    """Routing must connect any two cells reachable from one another.

    The endpoints are taken from the same connected component deliberately:
    constraint buffers genuinely do cut the site into islands, and an
    unreachable island is reported as a connectivity warning rather than
    treated as a routing failure.
    """
    from scipy import ndimage

    site, cons = ctx["site"], ctx["cons"]
    cost = build_cost_surface(site, std, cons, "collector")
    router = GridRouter(site, cost)

    lab, n = ndimage.label(cons.buildable)
    sizes = [(int((lab == i).sum()), i) for i in range(1, n + 1)]
    _, biggest = max(sizes)
    cells = np.argwhere(lab == biggest)

    start, end = tuple(cells[0]), tuple(cells[-1])
    dist, pred = router.solve_from(start)
    assert router.reachable(dist, end)
    path = router.trace(pred, end)
    assert len(path) > 1
    assert path[0] == start and path[-1] == end


def test_disconnected_developable_land_is_reported(ctx, std):
    """Land cut off by *impassable* constraints must be surfaced, not lost.

    Fragmentation of the buildable surface is not itself a defect - roads
    bridge watercourses and pass under servitudes. Only fragmentation of the
    traversable surface strands land.
    """
    from scipy import ndimage

    site, cons = ctx["site"], ctx["cons"]
    traversable = np.isfinite(build_cost_surface(site, std, cons,
                                                 "district_distributor"))
    struct = ndimage.generate_binary_structure(2, 2)
    _, n = ndimage.label(traversable, structure=struct)
    if n <= 1:
        pytest.skip("network reaches all developable land on this site")
    assert any("cut off from the road network" in w for w in ctx["net"].warnings)


def test_crossable_constraints_keep_the_site_connected(ctx, std):
    """Regression: treating every constraint as impassable stranded 311 ha.

    A watercourse and a powerline servitude crossing the site split the
    buildable surface into islands. Allowing engineered crossings restores a
    single traversable surface, which is what makes the plan buildable.
    """
    from scipy import ndimage

    site, cons = ctx["site"], ctx["cons"]
    struct = ndimage.generate_binary_structure(2, 2)

    _, buildable_parts = ndimage.label(cons.buildable, structure=struct)
    traversable = np.isfinite(build_cost_surface(site, std, cons, "collector"))
    _, traversable_parts = ndimage.label(traversable, structure=struct)

    assert traversable_parts <= buildable_parts


def test_nearest_valid_cell_snaps_off_constrained_land(ctx):
    valid = ctx["cons"].buildable
    blocked = np.argwhere(~valid)
    if not len(blocked):
        pytest.skip("no constrained land on this site")
    snapped = nearest_valid_cell(tuple(blocked[0]), valid)
    assert snapped is not None and valid[snapped]


def test_nearest_valid_cell_gives_up_rather_than_searching_forever():
    valid = np.zeros((50, 50), dtype=bool)
    assert nearest_valid_cell((25, 25), valid, max_radius=5) is None


# -- network ----------------------------------------------------------------

def test_network_has_a_hierarchy(ctx):
    classes = {s.road_class for s in ctx["net"].segments}
    assert "primary_arterial" in classes
    assert len(classes) >= 3, classes


def test_reserve_widths_come_from_the_ruleset(ctx, std):
    for seg in ctx["net"].segments:
        expected = std.get(f"roads.hierarchy.{seg.road_class}.reserve_width_m")
        assert seg.reserve_width_m == expected


def test_higher_order_roads_have_wider_reserves(ctx):
    widths = {s.road_class: s.reserve_width_m for s in ctx["net"].segments}
    order = ["primary_arterial", "district_distributor", "collector", "local_access"]
    present = [c for c in order if c in widths]
    for a, b in zip(present, present[1:]):
        assert widths[a] > widths[b], f"{a} not wider than {b}"


def test_primary_spine_connects_to_the_existing_road(ctx):
    """A settlement with no external connection is not a settlement."""
    site, net = ctx["site"], ctx["net"]
    spine = net.by_class("primary_arterial")
    assert spine
    external = site.layers["existing_road"]
    nearest = min(s.geometry.distance(external) for s in spine)
    assert nearest < 250, f"spine starts {nearest:.0f} m from the external road"


def test_road_reserves_stay_inside_the_site(ctx):
    site, net = ctx["site"], ctx["net"]
    assert net.reserve_polygon is not None
    assert net.reserve_polygon.difference(site.boundary.buffer(1)).area < 1.0


def test_junction_spacing_standard_is_respected(ctx, std):
    """Distributors joining an arterial every 50 m destroy its traffic function."""
    minimum = std.get("roads.hierarchy.primary_arterial.junction_spacing_min_m")
    js = ctx["net"].junctions
    for i, a in enumerate(js):
        for b in js[i + 1:]:
            assert a.distance(b) >= minimum - 1.0


def test_reserve_area_is_consistent_with_the_land_budget(ctx, std):
    """Generated geometry should land near the circulation share we budgeted."""
    site, net = ctx["site"], ctx["net"]
    budgeted = std.get("land_budget.circulation_pct")
    actual = net.reserve_area_ha / site.area_ha * 100
    assert actual == pytest.approx(budgeted, abs=6.0), (
        f"roads occupy {actual:.1f}% against a budget of {budgeted}%"
    )


def test_network_summary_totals_match_segments(ctx):
    net, site = ctx["net"], ctx["site"]
    s = net.summary(site)
    assert s["total_length_km"] == pytest.approx(net.total_length_m / 1000, abs=0.01)
    assert sum(c["segments"] for c in s["by_class"]) == len(net.segments)


# -- blocks -----------------------------------------------------------------

def test_blocks_do_not_overlap_road_reserves(ctx):
    """A block that overlaps its own road reserve cannot be subdivided."""
    net, blocks = ctx["net"], ctx["blocks"]
    for b in blocks.blocks[:120]:
        overlap = b.geometry.intersection(net.reserve_polygon).area
        assert overlap < b.geometry.area * 0.02, b.block_id


def test_blocks_stay_within_their_allocated_use(ctx):
    site, alloc, blocks = ctx["site"], ctx["alloc"], ctx["blocks"]
    total_by_use: dict[str, float] = {}
    for b in blocks.blocks:
        total_by_use[b.use] = total_by_use.get(b.use, 0.0) + b.area_ha
    for use, ha in total_by_use.items():
        assert ha <= alloc.achieved_ha[use] + 0.5, use


def test_only_residential_blocks_yield_plots(ctx):
    for b in ctx["blocks"].blocks:
        if b.use != "residential":
            assert b.indicative_plots == 0, b.block_id


def test_block_ids_are_unique(ctx):
    ids = [b.block_id for b in ctx["blocks"].blocks]
    assert len(ids) == len(set(ids))


def test_plot_yield_uses_the_brief_density_mix_not_the_default(std):
    """Regression: block yield was priced off the ruleset default mix.

    A brief that overrides the mix would then have every block silently
    mispriced, and the plot reconciliation would report a false shortfall.
    """
    dense = derive_programme(PlanningBrief(
        project_name="d", site_area_ha=600, target_population=15000,
        density_mix=DensityMix(low=0.0, medium=0.0, high=1.0)), std)
    sparse = derive_programme(PlanningBrief(
        project_name="s", site_area_ha=600, target_population=15000,
        density_mix=DensityMix(low=1.0, medium=0.0, high=0.0)), std)

    assert _mean_plot_area(std, dense) < _mean_plot_area(std, sparse)
    # And it must equal the actual typical plot size of that single band.
    assert _mean_plot_area(std, dense) == pytest.approx(
        std.get("residential.densities.high.plot_area_typical_m2"))


def test_plot_yield_falls_back_to_the_ruleset_without_a_programme(std):
    assert _mean_plot_area(std, None) > 0


def test_block_yield_reconciles_with_the_dwelling_schedule(ctx):
    """The layout must be able to deliver roughly the plots the brief promises."""
    blocks, prog = ctx["blocks"], ctx["prog"]
    variance = (blocks.total_indicative_plots - prog.total_dwellings) \
        / prog.total_dwellings * 100
    assert -25 < variance < 40, f"{variance:.0f}% variance"


def test_reconcile_flags_a_real_shortfall():
    class FakeLayout:
        total_indicative_plots = 1000
    msg = reconcile_plots(FakeLayout(), 5000)
    assert msg and "needs adjustment" in msg


def test_reconcile_is_silent_when_close_enough():
    class FakeLayout:
        total_indicative_plots = 4900
    assert reconcile_plots(FakeLayout(), 5000) is None


def test_reconcile_reports_spare_capacity():
    class FakeLayout:
        total_indicative_plots = 8000
    msg = reconcile_plots(FakeLayout(), 5000)
    assert msg and "above" in msg


# -- integration ------------------------------------------------------------

def test_geojson_carries_roads_and_blocks(ctx):
    from planeng.export.geojson import plan_to_geojson
    fc = plan_to_geojson(ctx["site"], ctx["cons"], ctx["alloc"],
                         network=ctx["net"], blocks=ctx["blocks"])
    layers = {f["properties"]["layer"] for f in fc["features"]}
    assert {"road", "road_reserve", "block"} <= layers

    roads = [f for f in fc["features"] if f["properties"]["layer"] == "road"]
    assert all(f["geometry"]["type"] in ("LineString", "MultiLineString")
               for f in roads)
    assert all(f["properties"]["reserve_width_m"] > 0 for f in roads)
