"""Assembling the engineering layer.

Order matters here and it is not arbitrary. The works are sited first, because
a network needs a source and a destination before it can be routed; then water,
sewer and electrical are laid over the road graph; then stormwater, which comes
from the terrain rather than the roads and so does not depend on any of it; then
lighting and the corridor check, which are about what fits in the reserve once
everything else is in it.

Where a works could not be sited, the network that depends on it is still
designed, from the best available substitute, and the substitution is stated.
That is deliberate. A plan that silently omits the sewer because the treatment
works did not fit tells a planner nothing; a plan that lays the sewer to the
site low point and says plainly that the flow has to leave the site from there
tells them what to negotiate.
"""

from __future__ import annotations

import time
from dataclasses import dataclass, field

import numpy as np

from ..standards.loader import Standards
from .graph import ServiceGraph, build_graph, components
from .lighting import CorridorPlan, LightingPlan, check_corridor, design_lighting
from .networks import (UtilityNetwork, design_electricity, design_sewer,
                       design_water, distribute_demand)
from .sites import ServiceSitePlan, site_works
from .stormwater import StormwaterPlan, design_stormwater


@dataclass
class InfrastructurePlan:
    graph: ServiceGraph
    works: ServiceSitePlan
    water: UtilityNetwork
    sewer: UtilityNetwork
    electricity: UtilityNetwork
    stormwater: StormwaterPlan
    lighting: LightingPlan
    corridor: CorridorPlan
    timings_ms: dict[str, float] = field(default_factory=dict)
    warnings: list[str] = field(default_factory=list)

    @property
    def networks(self) -> list[UtilityNetwork]:
        return [self.water, self.sewer, self.electricity]

    def summary(self) -> dict:
        return {
            "graph": self.graph.summary(),
            "works": self.works.summary(),
            "water": self.water.summary(),
            "sewer": self.sewer.summary(),
            "electricity": self.electricity.summary(),
            "stormwater": self.stormwater.summary(),
            "lighting": self.lighting.summary(),
            "corridor": self.corridor.summary(),
            "totals": {
                "reticulation_km": round(
                    sum(n.total_length_m for n in self.networks) / 1000, 2),
                "pump_stations": len(self.sewer.pump_stations),
                "street_light_poles": self.lighting.total_poles,
            },
            "timings_ms": {k: round(v, 1) for k, v in self.timings_ms.items()},
            "warnings": self.warnings,
        }


def design_infrastructure(site, std: Standards, programme, allocation, blocks,
                          network, facility_plan, constraints
                          ) -> InfrastructurePlan:
    timings: dict[str, float] = {}
    warnings: list[str] = []
    infra = programme.infrastructure

    def timed(name, fn):
        t0 = time.perf_counter()
        out = fn()
        timings[name] = (time.perf_counter() - t0) * 1000
        return out

    graph = timed("graph", lambda: build_graph(network, site))

    if graph.n_nodes:
        labels = components(graph)
        n_comp = int(labels.max()) + 1
        if n_comp > 1:
            biggest = int(np.bincount(labels).max())
            warnings.append(
                f"The road network is in {n_comp} disconnected pieces; the "
                f"largest holds {biggest} of {graph.n_nodes} junctions. Every "
                f"service follows the reserve, so anything off the main piece "
                f"cannot be reticulated.")

    works = timed("works", lambda: site_works(
        site, std, programme, allocation, blocks, constraints))
    warnings.extend(works.warnings)

    demand = distribute_demand(graph, blocks)

    # -- sources -----------------------------------------------------------
    reservoir = works.get("reservoir")
    water_source = graph.nearest_node(reservoir.geometry) if (
        reservoir and graph.n_nodes) else -1

    wwtw = works.get("wastewater_treatment")
    if wwtw and graph.n_nodes:
        sewer_outfall = graph.nearest_node(wwtw.geometry)
    elif graph.n_nodes:
        # No works fits on the site, so the sewer is laid to an outfall instead.
        sewer_outfall = _pick_outfall(graph, demand)
        warnings.append(
            "No treatment works could be sited, so the sewer is laid to an "
            "outfall on the network. That point is the connection to a regional "
            "works, and the works itself is outside this plan.")
    else:
        sewer_outfall = -1

    subs = facility_plan.of_type("electrical_substation") if facility_plan else []
    if subs and graph.n_nodes:
        power_source = graph.nearest_node(subs[0].geometry)
    else:
        power_source = water_source if water_source >= 0 else 0
        if graph.n_nodes:
            warnings.append(
                "No electrical substation was sited, so reticulation is "
                "measured from an assumed intake point. Feeder lengths are "
                "indicative until the intake is fixed with the utility.")

    water = timed("water", lambda: design_water(
        graph, std, infra, demand, water_source))
    sewer = timed("sewer", lambda: design_sewer(
        graph, std, infra, demand, sewer_outfall))
    electricity = timed("electricity", lambda: design_electricity(
        graph, std, infra, demand, power_source))

    storm = timed("stormwater", lambda: design_stormwater(
        site, std, allocation, network, blocks))
    lighting = timed("lighting", lambda: design_lighting(graph, std))

    trunk_m = max((n.total_length_m for n in (water, sewer)), default=0.0)
    corridor = timed("corridor", lambda: check_corridor(std, network, trunk_m))

    for net in (water, sewer, electricity):
        warnings.extend(net.warnings)
    warnings.extend(storm.warnings)
    warnings.extend(corridor.warnings)

    return InfrastructurePlan(
        graph=graph, works=works, water=water, sewer=sewer,
        electricity=electricity, stormwater=storm, lighting=lighting,
        corridor=corridor, timings_ms=timings, warnings=warnings,
    )


def _pick_outfall(graph, demand):
    """Choose where the sewer leaves: low ground, but not in a far corner.

    The lowest point on the network is the obvious answer and often the wrong
    one. Depth accumulates with the length of a run, so an outfall tucked into a
    corner is reached by long trunk sewers from everywhere else and needs
    lifting on the way, while a slightly higher point nearer the middle of the
    demand is reached by shorter ones and needs none.

    So candidates are the lowest tenth of the network by ground level - anything
    higher gives away head that cannot be recovered - and among those the one
    with the shortest plot-weighted mean run wins. That is the trade a designer
    makes by eye, made explicit and measured.
    """
    z = graph.elevation
    if z is None or graph.n_nodes == 0:
        return -1

    cutoff = float(np.percentile(z, 10))
    candidates = np.flatnonzero(z <= cutoff)
    if candidates.size == 0:
        return int(np.argmin(z))
    if candidates.size > 40:
        # Cap the search: each candidate costs a Dijkstra, and the lowest forty
        # cover the plausible outfalls on any site this size.
        candidates = candidates[np.argsort(z[candidates])[:40]]

    weights = demand if demand is not None and demand.sum() > 0 else None
    best, best_cost = int(candidates[0]), float("inf")
    for node in candidates:
        dist, _ = graph.tree_from(int(node))
        reachable = np.isfinite(dist)
        if not reachable.any():
            continue
        if weights is None:
            cost = float(dist[reachable].mean())
        else:
            w = weights[reachable]
            cost = (float(np.average(dist[reachable], weights=w))
                    if w.sum() > 0 else float(dist[reachable].mean()))
        if cost < best_cost:
            best, best_cost = int(node), cost
    return best
