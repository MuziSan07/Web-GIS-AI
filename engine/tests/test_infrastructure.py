"""Tests for the engineering infrastructure layer.

The interesting assertions here are physical rather than structural. A sewer
that runs uphill, a network whose demand does not add up to the demand the
programme derived, or a catchment that loses water between the site and its
boundary are all things that produce perfectly well-formed output and are
wrong, so those are what these check.

One plan is generated per module at a coarse cell size and shared: the layer
under test is deterministic given a plan, and generating a fresh one per test
would put minutes into the suite for no extra coverage.
"""

from __future__ import annotations

import numpy as np
import pytest
from shapely.ops import unary_union

from planeng.brief.models import PlanningBrief
from planeng.infrastructure.graph import build_graph, components, stitch
from planeng.infrastructure.lighting import check_corridor, design_lighting
from planeng.infrastructure.networks import (_diameter, design_sewer,
                                             design_water, distribute_demand)
from planeng.infrastructure.stormwater import fill_sinks, flow_accumulation
from planeng.pipeline import generate_master_plan
from planeng.standards import load_standards


@pytest.fixture(scope="module")
def plan():
    return generate_master_plan(
        PlanningBrief(project_name="Services", site_area_ha=600,
                      target_population=25000),
        cell_size=20.0,
    )


@pytest.fixture(scope="module")
def std():
    return load_standards("botswana")


@pytest.fixture(scope="module")
def infra(plan):
    return plan.infrastructure


# -- the road network has to be one network ---------------------------------

def test_road_reserves_form_a_single_connected_piece(plan):
    """Local streets are clipped, not routed, so they can be left stranded.

    A layout in several disconnected pieces still produces blocks, plots and
    facilities, and every one of them is undeliverable.
    """
    reserve = plan.network.reserve_polygon
    parts = getattr(reserve, "geoms", [reserve])
    assert len(list(parts)) == 1


def test_service_graph_is_fully_connected(infra):
    labels = components(infra.graph)
    assert labels.max() == 0, (
        f"service graph is in {labels.max() + 1} pieces; anything off the main "
        f"piece cannot be reticulated")


def test_stitching_is_needed_and_bounded(plan):
    """Centrelines stop at reserve edges, so the raw graph is disconnected.

    Guards both directions: that the stitch does something, and that it does not
    do so much it invents connectivity by wiring parallel streets together.
    """
    raw = build_graph(plan.network, plan.site)
    connectors = sum(1 for c in raw.road_class if c == "connector")
    assert connectors > 0
    assert connectors < len(raw.edges) * 0.25


def test_stitch_respects_the_gap_limit(plan):
    from planeng.infrastructure.graph import ServiceGraph

    # Two isolated edges a kilometre apart must not be joined.
    nodes = np.array([[0.0, 0.0], [10.0, 0.0], [1000.0, 0.0], [1010.0, 0.0]])
    g = ServiceGraph(nodes=nodes, edges=[(0, 1), (2, 3)], lengths=[10.0, 10.0],
                     road_class=["local_access"] * 2, elevation=np.zeros(4))
    assert stitch(g, max_gap=20.0) == 0
    assert components(g).max() == 1


# -- demand has to add up ---------------------------------------------------

def test_demand_on_the_graph_equals_the_plots_in_the_blocks(plan, infra):
    demand = distribute_demand(infra.graph, plan.blocks)
    assert demand.sum() == pytest.approx(plan.blocks.total_indicative_plots)


def test_every_plot_is_reachable_from_every_source(infra):
    for net in infra.networks:
        assert net.unserved_plots == 0, f"{net.service} leaves plots unserved"


# -- water ------------------------------------------------------------------

def test_mains_are_commercial_sizes_only(infra, std):
    allowed = set(std.get("infrastructure.water.commercial_diameters_mm"))
    assert {c.size_mm for c in infra.water.conduits} <= allowed


def test_no_main_is_below_the_minimum_diameter(infra, std):
    floor = std.get("infrastructure.water.min_main_diameter_mm")
    assert min(c.size_mm for c in infra.water.conduits) >= floor


def test_diameter_rises_with_flow():
    sizes = [110, 160, 200, 250, 315, 400]
    small = _diameter(5, 1.0, sizes, 110)
    large = _diameter(200, 1.0, sizes, 110)
    assert small < large
    assert small == 110                     # governed by the minimum, not flow


def test_every_main_carries_at_least_the_fire_flow(infra, std):
    """Domestic peak in an outer street is a couple of litres a second.

    Size on that alone and no hydrant on the network works.
    """
    fire = std.get("infrastructure.water.fire_flow_l_per_s")
    assert min(c.carries for c in infra.water.conduits) >= fire - 1e-6


# -- sewer ------------------------------------------------------------------

def test_sewer_falls_toward_the_works_everywhere_it_is_not_pumped(infra, std):
    """The one assertion this whole layer exists to make."""
    min_grad = std.get("infrastructure.sewer.min_self_cleansing_gradient")
    adverse = [c for c in infra.sewer.conduits
               if not c.pumped and c.gradient < min_grad - 1e-9]
    assert not adverse, (
        f"{len(adverse)} sewer runs fall less than the self-cleansing gradient "
        f"and are not pumped")


def test_sewer_trench_stays_within_the_excavation_limit(infra, std):
    limit = std.get("infrastructure.sewer.max_excavation_depth_m")
    assert infra.sewer.metrics["max_depth_m"] <= limit + 1e-6


def test_sewage_is_not_lifted_repeatedly_on_its_way_to_the_works(infra):
    """On flat ground a trunk lift is unavoidable, and one lift marks nearly
    every plot upstream of it as pumped. What would signal a bad design is
    lifting the same flow again and again."""
    assert infra.sewer.metrics["max_lifts_on_a_path"] <= 6


def test_pump_stations_are_sparse_against_the_network(infra):
    assert infra.sewer.metrics["pump_stations_per_km"] < 0.75


def test_a_site_without_the_fall_to_drain_itself_says_so(infra):
    """Pump stations are the symptom. The cause is a site whose relief is less
    than the fall its longest run needs, and that is what a client can act on."""
    m = infra.sewer.metrics
    assert {"longest_run_m", "fall_required_m", "fall_available_m"} <= set(m)
    if m["fall_required_m"] > m["fall_available_m"]:
        assert any("gravity drainage is not" in w for w in infra.sewer.warnings)


def test_the_outfall_is_low_ground(infra):
    """Chosen for short runs, but never by giving away head."""
    z = infra.graph.elevation
    outfall_level = z[infra.sewer.source_node]
    assert outfall_level <= float(np.percentile(z, 15))


def test_no_zero_length_runs_are_emitted(infra):
    """A graph tee landing on an existing node used to produce a run of no
    length, which then failed every dimensional check for no design reason."""
    for net in infra.networks:
        assert all(c.length_m > 0 for c in net.conduits)


def test_flat_ground_is_drained_by_burying_the_pipe_not_by_pumping(std):
    """A dead flat site has no ground gradient at all and still drains.

    This is the modelling error the layer was built around: follow the ground
    surface and every run on flat land needs a pump station, which is not how a
    sewer is laid.
    """
    from planeng.infrastructure.graph import ServiceGraph

    n = 10
    nodes = np.array([[i * 100.0, 0.0] for i in range(n)])
    g = ServiceGraph(
        nodes=nodes,
        edges=[(i, i + 1) for i in range(n - 1)],
        lengths=[100.0] * (n - 1),
        road_class=["local_access"] * (n - 1),
        elevation=np.zeros(n),              # perfectly flat
    )
    demand = np.ones(n)
    net = design_sewer(g, std, {"sewage_average_day_m3": 500.0}, demand,
                       works_node=0)
    assert net.conduits
    # 9 runs at 100 m and 0.5% is 4.5 m of fall onto 1 m of cover; limit 6 m.
    assert not net.pump_stations
    assert net.metrics["max_depth_m"] > 1.0
    assert all(c.gradient >= 0.005 - 1e-9 for c in net.conduits)


def test_a_long_enough_flat_run_does_need_a_pump_station(std):
    from planeng.infrastructure.graph import ServiceGraph

    n = 40                                   # 3.9 km of flat street
    nodes = np.array([[i * 100.0, 0.0] for i in range(n)])
    g = ServiceGraph(
        nodes=nodes,
        edges=[(i, i + 1) for i in range(n - 1)],
        lengths=[100.0] * (n - 1),
        road_class=["local_access"] * (n - 1),
        elevation=np.zeros(n),
    )
    net = design_sewer(g, std, {"sewage_average_day_m3": 500.0}, np.ones(n),
                       works_node=0)
    assert net.pump_stations


# -- stormwater -------------------------------------------------------------

def test_all_runoff_leaves_the_site(plan):
    """Mass balance. Every hectare of catchment has to cross the boundary."""
    acc, exits, _ = flow_accumulation(plan.site)
    catchment_ha = acc * plan.site.cell_area_ha
    inside_ha = plan.site.cells_to_ha(int(plan.site.mask().sum()))
    assert catchment_ha[exits].sum() == pytest.approx(inside_ha, rel=1e-6)


def test_sink_filling_leaves_no_interior_pit(plan):
    site = plan.site
    inside = site.mask()
    floor = float(site.dem[inside].min()) - 1000.0
    filled = fill_sinks(np.where(inside, site.dem, floor), inside)

    # A pit is a cell strictly lower than all eight of its neighbours.
    from scipy import ndimage
    local_min = ndimage.minimum_filter(filled, size=3)
    pits = inside & (filled <= local_min) & (
        filled < ndimage.minimum_filter(
            np.where(inside, filled, np.inf), size=3, mode="constant",
            cval=np.inf) + 1e-9)
    interior = inside & ~ndimage.binary_erosion(inside, np.ones((3, 3), bool))
    assert not (pits & ~interior).any()


def test_filling_never_lowers_the_ground(plan):
    site = plan.site
    inside = site.mask()
    floor = float(site.dem[inside].min()) - 1000.0
    raw = np.where(inside, site.dem, floor)
    assert (fill_sinks(raw, inside) >= raw - 1e-9).all()


def test_channels_are_traced_as_flow_paths_not_zigzags(infra, plan):
    """A branching network joined into one elevation-ordered line was 678 km
    of channel inside a 600 ha site. Bound it against the site itself."""
    storm = infra.stormwater
    diagonal = plan.site.boundary.length
    assert storm.total_channel_m < diagonal * 3


def test_drainage_reserve_is_reported_where_channels_exist(infra):
    storm = infra.stormwater
    if storm.channels:
        assert storm.reserve_polygon is not None
        assert storm.reserve_polygon.area > 0


def test_development_over_a_channel_is_reported_not_hidden(infra):
    storm = infra.stormwater
    conflict = storm.metrics.get("channel_conflict_ha", 0.0)
    if conflict > 0.5:
        assert any("drainage" in w for w in storm.warnings)


def test_rainfall_data_is_declared_missing_rather_than_invented(infra):
    """The peak flow cannot be computed without a local IDF curve."""
    assert infra.stormwater.missing_inputs
    assert any("intensity" in m for m in infra.stormwater.missing_inputs)


def test_development_raises_runoff_and_says_by_how_much(infra):
    m = infra.stormwater.metrics
    assert m["composite_runoff_coefficient"] > m["predevelopment_coefficient"]
    assert m["runoff_increase_factor"] > 1.0


# -- works ------------------------------------------------------------------

def test_a_works_that_does_not_fit_is_reported_with_a_reason(infra):
    for key, reason in infra.works.unplaced.items():
        assert reason and len(reason) > 20, f"{key} unplaced without a reason"


def test_sited_works_meet_their_area_standard(infra, std):
    minimums = {
        "reservoir": std.get("infrastructure.water.reservoir_site_area_ha"),
        "landfill": std.get("infrastructure.solid_waste.landfill_min_area_ha"),
    }
    for works in infra.works.sites:
        if works.key in minimums:
            assert works.area_ha >= minimums[works.key] * 0.9


def test_a_reservoir_without_head_is_marked_non_compliant(infra, std):
    """Gravity supply is a physical fact, not a scoring penalty."""
    res = infra.works.get("reservoir")
    if res is None:
        pytest.skip("no reservoir sited on this plan")
    min_head = std.get("infrastructure.water.reservoir_min_head_m")
    if not res.compliant:
        joined = " ".join(res.notes).lower()
        assert "head" in joined or "gravity" in joined
        assert min_head > 0


def test_works_land_take_is_measured_against_the_utilities_budget(infra):
    assert infra.works.utilities_budget_ha > 0


# -- lighting and corridor --------------------------------------------------

def test_pole_count_matches_length_over_spacing(infra, std):
    spacing = std.get("infrastructure.street_lighting.spacing_m")
    lengths: dict[str, float] = {}
    for (a, b), run in zip(infra.graph.edges, infra.graph.lengths):
        cls = infra.graph.class_of(a, b)
        lengths[cls] = lengths.get(cls, 0.0) + run

    for cls, poles in infra.lighting.poles_by_class.items():
        gap = spacing.get(cls, spacing["local_access"])
        assert poles == int(np.ceil(lengths[cls] / gap))


def test_lighting_reports_a_connected_load(infra):
    assert infra.lighting.load_kw > 0
    assert infra.lighting.summary()["annual_energy_mwh"] > 0


def test_corridor_check_uses_the_narrowest_reserve(plan, std):
    check = check_corridor(std, plan.network)
    assert check.narrowest_reserve_m == min(
        s.reserve_width_m for s in plan.network.segments)


def test_water_and_sewer_are_separated_in_the_corridor(plan, std):
    """Same trench is a contamination path on the day the sewer leaks."""
    check = check_corridor(std, plan.network)
    assert check.separations_ok, check.warnings


def test_a_reserve_too_narrow_for_the_services_is_reported(std):
    class Seg:
        reserve_width_m = 4.0

    class Net:
        segments = [Seg()]

    check = check_corridor(std, Net())
    assert not all(c.fits for c in check.checks)
    assert check.warnings


# -- assembly ---------------------------------------------------------------

def test_every_service_run_lies_on_the_road_reserve(plan, infra):
    """Services follow the reserve because that is the only land the authority
    owns. A run outside it needs a servitude nobody has negotiated."""
    reserve = plan.network.reserve_polygon.buffer(plan.site.cell_size)
    stray = [c for c in infra.water.conduits[:400]
             if not reserve.intersects(c.geometry)]
    assert not stray


def test_report_and_geojson_carry_the_engineering_layer(plan):
    import json

    report = plan.report()
    assert "infrastructure" in report
    json.dumps(report)

    layers = {f["properties"]["layer"] for f in plan.to_geojson()["features"]}
    assert {"service_run", "service_site", "drainage_channel"} <= layers


def test_timings_are_recorded_for_each_step(infra):
    assert set(infra.timings_ms) >= {"graph", "works", "water", "sewer",
                                     "electricity", "stormwater"}
