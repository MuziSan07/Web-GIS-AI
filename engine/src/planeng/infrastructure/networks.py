"""Reticulation design: water, sewer and electrical.

Three networks, one graph, three different problems.

**Water** is a pressure network. It can be pushed uphill, so the route is simply
the shortest path from the reservoir to each consumer, and the engineering
content is in the sizing: every main carries the peak flow of everything
downstream of it, so diameters are set by accumulating demand up the tree.

**Sewer** is the one network the ground dictates. It runs by gravity, so a run
is only valid if it falls at least the self-cleansing gradient toward the works.
That makes it a *directed* routing problem: the same street is usable in one
direction and not the other. Where no falling route exists, a pump station is
required - and a pump station is a capital item, an operating cost and a failure
mode, so it is counted and located rather than assumed away.

**Electrical** is a pressure network again in all but name, but it is limited by
distance rather than by head: a distribution transformer can only serve so far
before the low-voltage drop is unacceptable, so the design output is a
transformer count and the share of plots inside a transformer radius.

Demand is distributed to graph nodes in proportion to the plots each street
serves, and the totals are held to the programme figures. A network whose
demand does not add up to the demand the programme derived is not a design of
that settlement.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field

import numpy as np
from shapely.geometry import LineString

from ..standards.loader import Standards
from .graph import ServiceGraph

# Runs shorter than this are graph artefacts, not pipe.
MIN_RUN_M = 0.05


@dataclass
class Conduit:
    geometry: LineString
    service: str
    size_mm: float                  # nominal diameter, or 0 for cable
    length_m: float
    carries: float                  # l/s for pipes, kVA for cable
    road_class: str = "local_access"
    pumped: bool = False
    gradient: float = 0.0

    def summary(self) -> dict:
        return {
            "service": self.service,
            "size_mm": self.size_mm,
            "length_m": round(self.length_m, 1),
            "carries": round(self.carries, 2),
            "pumped": self.pumped,
        }


@dataclass
class UtilityNetwork:
    service: str
    conduits: list[Conduit] = field(default_factory=list)
    source_node: int = -1
    pump_stations: list[tuple[float, float]] = field(default_factory=list)
    unserved_plots: int = 0
    metrics: dict = field(default_factory=dict)
    warnings: list[str] = field(default_factory=list)

    @property
    def total_length_m(self) -> float:
        return sum(c.length_m for c in self.conduits)

    def length_by_size(self) -> dict[float, float]:
        out: dict[float, float] = {}
        for c in self.conduits:
            out[c.size_mm] = out.get(c.size_mm, 0.0) + c.length_m
        return out

    def summary(self) -> dict:
        return {
            "service": self.service,
            "total_length_km": round(self.total_length_m / 1000, 2),
            "runs": len(self.conduits),
            "pump_stations": len(self.pump_stations),
            "unserved_plots": self.unserved_plots,
            "by_size": [
                {"size_mm": int(k), "length_km": round(v / 1000, 2)}
                for k, v in sorted(self.length_by_size().items(), reverse=True)
                if k
            ],
            "metrics": {k: (round(v, 2) if isinstance(v, float) else v)
                        for k, v in self.metrics.items()},
            "warnings": self.warnings,
        }


# ---------------------------------------------------------------------------
# demand
# ---------------------------------------------------------------------------

def distribute_demand(graph: ServiceGraph, blocks) -> np.ndarray:
    """Plots served from each graph node.

    A block connects to the nearest point on the network, which is the same
    assumption the plot layout already makes when it gives every parcel road
    frontage.
    """
    demand = np.zeros(graph.n_nodes)
    if graph.n_nodes == 0:
        return demand
    for b in blocks.for_use("residential"):
        if b.indicative_plots <= 0:
            continue
        demand[graph.nearest_node(b.geometry)] += b.indicative_plots
    return demand


def _accumulate(graph: ServiceGraph, pred: np.ndarray, dist: np.ndarray,
                demand: np.ndarray) -> np.ndarray:
    """Total demand downstream of each node, toward the source.

    Nodes are processed furthest-first, so a node's own subtree total is
    complete before it is added to its parent. That is one pass, against the
    repeated tree walks the naive version would do.
    """
    total = demand.copy()
    order = np.argsort(-np.where(np.isfinite(dist), dist, -1.0))
    for v in order:
        p = int(pred[v])
        if p >= 0:
            total[p] += total[v]
    return total


# ---------------------------------------------------------------------------
# water
# ---------------------------------------------------------------------------

def design_water(graph: ServiceGraph, std: Standards, infra: dict,
                 demand: np.ndarray, source_node: int) -> UtilityNetwork:
    """Size a branched distribution network from the reservoir."""
    net = UtilityNetwork(service="water", source_node=source_node)
    if graph.n_nodes == 0 or source_node < 0:
        net.warnings.append("No network to lay water mains along.")
        return net

    velocity = std.get("infrastructure.water.design_velocity_m_per_s")
    sizes = std.get("infrastructure.water.commercial_diameters_mm")
    min_dia = std.get("infrastructure.water.min_main_diameter_mm")
    fire_flow = std.get("infrastructure.water.fire_flow_l_per_s")

    dist, pred = graph.tree_from(source_node)
    served = np.isfinite(dist)
    downstream = _accumulate(graph, pred, dist, demand * served)

    total_plots = demand.sum() or 1.0
    peak_l_s = float(infra.get("water_peak_day_m3", 0.0)) * 1000 / 86_400

    for v in range(graph.n_nodes):
        p = int(pred[v])
        if p < 0:
            continue
        if _edge_length(graph, p, v) <= MIN_RUN_M:
            continue
        share = downstream[v] / total_plots
        # Every main also has to pass the fire flow at its far end, which is
        # what governs size in the outer streets: domestic peak there is a
        # couple of litres a second, a hydrant is thirty.
        flow = max(peak_l_s * share, fire_flow)
        dia = _diameter(flow, velocity, sizes, min_dia)
        net.conduits.append(Conduit(
            geometry=_edge_line(graph, p, v), service="water", size_mm=dia,
            length_m=_edge_length(graph, p, v), carries=flow,
            road_class=_edge_class(graph, p, v),
        ))

    net.unserved_plots = int(demand[~served].sum())
    net.metrics = {
        "peak_flow_l_per_s": peak_l_s,
        "largest_main_mm": max((c.size_mm for c in net.conduits), default=0),
        "max_run_from_reservoir_m": float(dist[served].max()) if served.any() else 0.0,
        "fire_flow_governed_pct": round(
            100 * sum(c.length_m for c in net.conduits
                      if c.carries <= fire_flow + 1e-9)
            / max(net.total_length_m, 1e-9), 1),
    }
    if net.unserved_plots:
        net.warnings.append(
            f"{net.unserved_plots:,} plots are not reachable from the reservoir "
            f"along the road network.")
    return net


# ---------------------------------------------------------------------------
# sewer
# ---------------------------------------------------------------------------

def design_sewer(graph: ServiceGraph, std: Standards, infra: dict,
                 demand: np.ndarray, works_node: int) -> UtilityNetwork:
    """Route a gravity sewer to the works and set its invert levels.

    A sewer does not follow the ground. It is laid to a gradient by varying how
    deep it is buried, so a dead flat street still drains - the pipe simply gets
    deeper along it. What limits the design is excavation: past about six metres
    the trench needs shoring and dewatering, and a pump station becomes the
    cheaper answer.

    So the design runs in two passes. First route to the works, preferring
    ground that falls. Then set invert levels from the head of each run down to
    the works, holding the self-cleansing gradient, and record where the trench
    would go deeper than the ruleset allows. Those points, and only those, are
    the pump stations.
    """
    net = UtilityNetwork(service="sewer", source_node=works_node)
    if graph.n_nodes == 0 or works_node < 0:
        net.warnings.append("No treatment works to fall toward; no sewer laid.")
        return net

    min_grad = std.get("infrastructure.sewer.min_self_cleansing_gradient")
    velocity = std.get("infrastructure.sewer.design_velocity_m_per_s")
    sizes = std.get("infrastructure.sewer.commercial_diameters_mm")
    min_dia = std.get("infrastructure.sewer.min_sewer_diameter_mm")
    manhole_spacing = std.get("infrastructure.sewer.max_manhole_spacing_m")
    min_cover = std.get("infrastructure.sewer.min_cover_m")
    max_depth = std.get("infrastructure.sewer.max_excavation_depth_m")

    z = graph.elevation
    dist, pred = _gravity_tree(graph, works_node)
    served = np.isfinite(dist)

    downstream = _accumulate(graph, pred, dist, demand * served)
    total_plots = demand.sum() or 1.0
    peak_l_s = (float(infra.get("sewage_average_day_m3", 0.0)) * 1000 / 86_400
                * std.get("infrastructure.sewer.peak_factor"))

    arrive, leave, depth, lifts = _set_inverts(
        graph, pred, dist, z, min_grad, min_cover, max_depth)

    for v in range(graph.n_nodes):
        p = int(pred[v])
        if p < 0:
            continue
        length = _edge_length(graph, v, p)
        if length <= MIN_RUN_M:
            continue
        # Measured departure-to-arrival: the run leaves v at its outgoing invert
        # and reaches p at p's incoming one. Comparing against p's outgoing
        # invert reads every run into a pump station as though it climbed.
        grad = (leave[v] - arrive[p]) / length if length else 0.0
        flow = peak_l_s * downstream[v] / total_plots
        net.conduits.append(Conduit(
            geometry=_edge_line(graph, v, p), service="sewer",
            size_mm=_diameter(flow, velocity, sizes, min_dia),
            length_m=length, carries=flow, gradient=grad,
            pumped=v in lifts,          # rising main out of a wet well
            road_class=_edge_class(graph, v, p),
        ))

    net.pump_stations = _thin(
        [(float(graph.nodes[v][0]), float(graph.nodes[v][1])) for v in lifts],
        min_gap=250.0)

    # Which plots have their sewage lifted at least once on the way to the
    # works. That is the operational figure - a pump station is a permanent
    # cost and a failure mode for everything upstream of it - and it means more
    # than a share of pipe length, since all of this network runs by gravity.
    lift_set = set(lifts)
    behind_pump = np.zeros(graph.n_nodes, dtype=bool)
    lifts_on_path = np.zeros(graph.n_nodes, dtype=int)
    for v in np.argsort(np.where(np.isfinite(dist), dist, np.inf)):
        p = int(pred[v])
        if p >= 0:
            behind_pump[v] = behind_pump[p] or (v in lift_set)
            lifts_on_path[v] = lifts_on_path[p] + (1 if v in lift_set else 0)

    served_depth = depth[served]
    total_plots_all = demand.sum() or 1.0
    net.unserved_plots = int(demand[~served].sum())
    net.metrics = {
        "peak_flow_l_per_s": peak_l_s,
        "plots_behind_a_pump_pct": round(
            100 * demand[behind_pump].sum() / total_plots_all, 1),
        "manholes": int(math.ceil(net.total_length_m / manhole_spacing)),
        "mean_depth_m": float(served_depth.mean()) if served_depth.size else 0.0,
        "max_depth_m": float(served_depth.max()) if served_depth.size else 0.0,
        "depth_limit_m": max_depth,
        "works_inlet_depth_m": float(z[works_node] - arrive[works_node]),
        # How many times the worst-placed plot has its sewage lifted. This is
        # the figure that separates a flat site needing one trunk lift from a
        # layout that pumps repeatedly; the percentage above cannot, because a
        # single lift near the works marks nearly every plot upstream of it.
        "max_lifts_on_a_path": int(lifts_on_path[served].max())
        if served.any() else 0,
        "pump_stations_per_km": round(
            len(net.pump_stations) / max(net.total_length_m / 1000, 1e-9), 2),
    }
    if net.pump_stations:
        net.warnings.append(
            f"{len(net.pump_stations)} sewer pump station(s) are required where "
            f"the trench would otherwise pass {max_depth:.0f} m deep. "
            f"{net.metrics['plots_behind_a_pump_pct']:.0f}% of plots drain "
            f"through at least one, and the worst-placed is lifted "
            f"{net.metrics['max_lifts_on_a_path']} time(s) on its way. On "
            f"ground this flat a trunk lift is hard to avoid, but every station "
            f"is a permanent operating cost and a failure point that a gravity "
            f"system does not have.")
    if net.unserved_plots:
        net.warnings.append(
            f"{net.unserved_plots:,} plots cannot be drained to the works.")

    # The finding behind the pump count: whether this land has the fall to
    # drain itself at all. A pump station is a symptom; this is the cause, and
    # it is the number a client can act on - by moving the outfall, accepting
    # the pumping, or reconsidering the extent of the settlement.
    longest = float(dist[served].max()) if served.any() else 0.0
    needed = longest * min_grad
    relief = float(z[served].max() - z[works_node]) if served.any() else 0.0
    net.metrics["longest_run_m"] = longest
    net.metrics["fall_required_m"] = needed
    net.metrics["fall_available_m"] = relief
    if needed > relief:
        net.warnings.append(
            f"The longest sewer run is {longest / 1000:,.1f} km and needs "
            f"{needed:,.1f} m of fall at the {min_grad * 100:.1f}% "
            f"self-cleansing gradient. The site provides {relief:,.1f} m "
            f"between that point and the outfall. Full gravity drainage is not "
            f"physically available on this land: the shortfall is made up by "
            f"pumping, and no change to the layout removes it.")
    return net


def _set_inverts(graph: ServiceGraph, pred: np.ndarray, dist: np.ndarray,
                 z: np.ndarray, min_grad: float, min_cover: float,
                 max_depth: float):
    """Invert levels down the flow tree, and the runs that have to be pumped.

    Two levels per node, not one. Sewage arrives at a node at whatever depth the
    runs above it dictate, and where a pump lifts it, it leaves shallower than
    it arrived. Collapsing the two into one value makes every run into a pump
    station look as though it flows uphill.

    A run becomes a rising main when gravity would put the next manhole below
    the excavation limit - which happens where the ground climbs between two
    junctions, since the invert has to keep falling while the surface rises. The
    pump sits at the upstream end and pushes to minimum cover at the downstream
    one. Deepening the trench instead is what the limit exists to prevent, and
    it is why the limit holds here by construction rather than by hope.

    One pass, in order of decreasing distance from the works: that order is
    topological on the flow tree, so a node's arrival level is final when it is
    reached. Deciding every lift up front and applying them afterwards installs
    pumps for depths that the lifts upstream had already removed.

    The works itself is never a lift. A treatment works inlet is deep because
    everything drains to it; that is the design, not a failure of it.
    """
    arrive = z - min_cover                  # invert as flow reaches the node
    leave = arrive.copy()                   # invert as flow leaves it
    lifts: list[int] = []

    order = np.argsort(-np.where(np.isfinite(dist), dist, -1.0))
    for v in order:
        p = int(pred[v])
        if p < 0:
            continue                        # the works: nothing leaves it

        gravity = arrive[v] - min_grad * _edge_length(graph, v, p)
        too_deep_here = z[v] - arrive[v] > max_depth
        too_deep_there = z[p] - gravity > max_depth

        if too_deep_here or too_deep_there:
            lifts.append(int(v))
            leave[v] = z[v] - min_cover     # pumped up to cover at the wet well
            delivered = z[p] - min_cover    # rising main, so no fall required
        else:
            leave[v] = arrive[v]
            delivered = gravity

        if delivered < arrive[p]:
            arrive[p] = delivered

    # Depth is reported over the network, not at the works inlet.
    depth = z - arrive
    depth[np.flatnonzero(pred < 0)] = min_cover

    return arrive, leave, depth, lifts


def _gravity_tree(graph: ServiceGraph, works: int):
    """Flow paths to the works, by pipe length.

    Length alone, with no penalty for climbing, and that is not a
    simplification - it is the answer.

    Work out the trench depth at the end of a run. The invert has to fall at the
    self-cleansing gradient the whole way, so it drops by `gradient x length`,
    while the ground does whatever the ground does. Depth at the far end is
    therefore

        cover + gradient x length - (ground at the start - ground at the end)

    and that last term telescopes: between two fixed points the total change in
    ground level is the same whichever way round it goes. Every hill climbed is
    a hill descended again. So depth depends on the *length* of the route and
    nothing else, and the cheapest route by depth is simply the shortest one.

    An earlier version of this priced climbing at two hundred metres of pipe per
    metre of rise, which is exactly the break-even rate - a hundred metre detour
    to dodge half a metre of climb costs half a metre of fall - so the cost
    function was degenerate and the router wandered. Median runs came out at six
    kilometres on a site two and a half kilometres across.
    """
    dist, pred = graph.tree_from(works)
    return dist, pred


# ---------------------------------------------------------------------------
# electrical
# ---------------------------------------------------------------------------

def design_electricity(graph: ServiceGraph, std: Standards, infra: dict,
                       demand: np.ndarray, source_node: int) -> UtilityNetwork:
    """Medium-voltage reticulation from the intake substation."""
    net = UtilityNetwork(service="electricity", source_node=source_node)
    if graph.n_nodes == 0 or source_node < 0:
        net.warnings.append("No substation to reticulate from.")
        return net

    tx_kva = std.get("infrastructure.electricity.distribution_transformer_kva")
    lv_radius = std.get("infrastructure.electricity.transformer_max_lv_radius_m")
    kv = std.get("infrastructure.electricity.mv_voltage_kv")

    dist, pred = graph.tree_from(source_node)
    served = np.isfinite(dist)
    downstream = _accumulate(graph, pred, dist, demand * served)

    total_plots = demand.sum() or 1.0
    total_kva = float(infra.get("electrical_demand_mva", 0.0)) * 1000

    for v in range(graph.n_nodes):
        p = int(pred[v])
        if p < 0:
            continue
        if _edge_length(graph, p, v) <= MIN_RUN_M:
            continue
        kva = total_kva * downstream[v] / total_plots
        net.conduits.append(Conduit(
            geometry=_edge_line(graph, p, v), service="electricity",
            size_mm=0.0, length_m=_edge_length(graph, p, v), carries=kva,
            road_class=_edge_class(graph, p, v),
        ))

    transformers = int(math.ceil(total_kva / tx_kva)) if tx_kva else 0
    net.unserved_plots = int(demand[~served].sum())
    net.metrics = {
        "mv_voltage_kv": kv,
        "total_load_kva": total_kva,
        "distribution_transformers": transformers,
        "transformer_rating_kva": tx_kva,
        "max_feeder_run_m": float(dist[served].max()) if served.any() else 0.0,
        "plots_per_transformer": round(total_plots / transformers, 1)
        if transformers else 0.0,
        "lv_radius_m": lv_radius,
    }
    if net.unserved_plots:
        net.warnings.append(
            f"{net.unserved_plots:,} plots have no route back to the "
            f"substation.")
    return net


# ---------------------------------------------------------------------------

def _diameter(flow_l_s: float, velocity: float, sizes, min_mm: float) -> float:
    """Smallest commercial diameter that carries the flow at design velocity."""
    if flow_l_s <= 0 or velocity <= 0:
        return float(min_mm)
    area_m2 = (flow_l_s / 1000.0) / velocity
    needed_mm = math.sqrt(4 * area_m2 / math.pi) * 1000
    for s in sorted(sizes):
        if s >= max(needed_mm, min_mm):
            return float(s)
    return float(max(sizes))


def _edge_line(graph: ServiceGraph, a: int, b: int) -> LineString:
    return LineString([tuple(graph.nodes[a]), tuple(graph.nodes[b])])


def _edge_length(graph: ServiceGraph, a: int, b: int) -> float:
    return float(np.hypot(*(graph.nodes[a] - graph.nodes[b])))


def _edge_class(graph: ServiceGraph, a: int, b: int) -> str:
    return graph.class_of(a, b)


def _thin(points: list[tuple[float, float]], min_gap: float
          ) -> list[tuple[float, float]]:
    """Collapse clustered points to one representative each."""
    kept: list[tuple[float, float]] = []
    for p in points:
        if all(math.hypot(p[0] - q[0], p[1] - q[1]) >= min_gap for q in kept):
            kept.append(p)
    return kept
