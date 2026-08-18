"""Stormwater: where the water already goes, and what development does to it.

The other three networks are built from the road layout. This one is not. Water
ran off this site in a particular pattern before anyone drew a plan, and the
cheapest, most robust drainage system is the one that keeps using that pattern:
the natural channels become the drainage reserves, and development stays out of
them. A layout that fills a natural watercourse has not removed the flow, it has
moved the flood into whichever street it built there instead.

So this works from the terrain first. It derives flow direction and
accumulation from the DEM, extracts the channels that carry real catchment,
finds where they leave the site - those are the outfalls, and they are fixed by
geography, not by design - and then reports what the plan does to them: every
place a road crosses a channel needs a culvert, and every channel under
development is a problem.

What is *not* computed here is the design peak flow. That needs a local
rainfall intensity-duration-frequency relationship, and this ruleset does not
carry one. Detention volume is given on the ruleset's per-hectare rule instead,
and the missing input is stated rather than filled with a plausible number.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
from scipy import ndimage
from shapely.geometry import LineString, Point
from shapely.ops import unary_union

from ..geom import cells_to_line
from ..site.model import Site
from ..standards.loader import Standards

# Catchment at which a flow path is a watercourse worth reserving land for.
# This is a policy number, not a physical one: every cell on a hillside is
# technically a flow path, and set too low this marks the whole site as
# drainage. Twenty hectares is about the point where a channel carries enough
# to need a crossing structure rather than a kerb inlet.
CHANNEL_MIN_CATCHMENT_HA = 20.0

# Tilt applied across filled depressions so they drain rather than pond.
FILL_EPSILON_M = 0.001


@dataclass
class DrainageLine:
    geometry: LineString
    catchment_ha: float
    reserve_width_m: float

    @property
    def length_m(self) -> float:
        return self.geometry.length


@dataclass
class StormwaterPlan:
    channels: list[DrainageLine] = field(default_factory=list)
    outfalls: list[Point] = field(default_factory=list)
    culverts: list[Point] = field(default_factory=list)
    reserve_polygon: object = None
    metrics: dict = field(default_factory=dict)
    warnings: list[str] = field(default_factory=list)
    missing_inputs: list[str] = field(default_factory=list)

    @property
    def total_channel_m(self) -> float:
        return sum(c.length_m for c in self.channels)

    def summary(self) -> dict:
        return {
            "channels": len(self.channels),
            "channel_length_km": round(self.total_channel_m / 1000, 2),
            "outfalls": len(self.outfalls),
            "culverts": len(self.culverts),
            "reserve_ha": round(
                self.reserve_polygon.area / 10_000, 2)
            if self.reserve_polygon else 0.0,
            "metrics": {k: (round(v, 2) if isinstance(v, float) else v)
                        for k, v in self.metrics.items()},
            "missing_inputs": self.missing_inputs,
            "warnings": self.warnings,
        }


def design_stormwater(site: Site, std: Standards, allocation, network,
                      blocks) -> StormwaterPlan:
    plan = StormwaterPlan()
    mask = site.mask()
    if not mask.any():
        return plan

    reserve_w = std.get("infrastructure.stormwater.min_drainage_reserve_m")

    acc, exits, target = flow_accumulation(site)
    catchment_ha = acc * site.cell_area_ha
    channel = mask & (catchment_ha >= CHANNEL_MIN_CATCHMENT_HA)

    plan.channels = _trace_channels(site, channel, catchment_ha, target,
                                    reserve_w)
    plan.outfalls = _outfalls(exits, catchment_ha, site)

    if plan.channels:
        plan.reserve_polygon = unary_union(
            [c.geometry.buffer(c.reserve_width_m / 2, cap_style=2)
             for c in plan.channels]).intersection(site.boundary)

    # -- what the layout does to the natural drainage ----------------------
    plan.culverts = _culverts(plan.channels, network)

    developed = _developed_mask(site, blocks)
    conflict_cells = int((channel & developed).sum())
    conflict_ha = site.cells_to_ha(conflict_cells)

    # -- runoff -------------------------------------------------------------
    coeffs = {
        "residential": std.get("infrastructure.stormwater.runoff_coefficient_residential"),
        "commercial": std.get("infrastructure.stormwater.runoff_coefficient_commercial"),
        "industrial": std.get("infrastructure.stormwater.runoff_coefficient_commercial"),
        "institutional": std.get("infrastructure.stormwater.runoff_coefficient_commercial"),
        "recreation_open_space": std.get("infrastructure.stormwater.runoff_coefficient_open_space"),
    }
    areas = {b: 0.0 for b in coeffs}
    for block in blocks.blocks:
        if block.use in areas:
            areas[block.use] += block.area_ha
    road_ha = (network.reserve_area_ha if network else 0.0)

    developed_ha = sum(areas.values()) + road_ha
    weighted = sum(areas[u] * coeffs[u] for u in areas) + road_ha * 0.95
    composite_c = weighted / developed_ha if developed_ha else 0.0
    pre_c = std.get("infrastructure.stormwater.runoff_coefficient_open_space")

    impervious_ha = weighted            # C-weighted area is the impervious equivalent
    detention_m3 = impervious_ha * std.get(
        "infrastructure.stormwater.detention_m3_per_impervious_ha")

    plan.metrics = {
        "composite_runoff_coefficient": composite_c,
        "predevelopment_coefficient": pre_c,
        "runoff_increase_factor": round(composite_c / pre_c, 2) if pre_c else 0.0,
        "impervious_equivalent_ha": impervious_ha,
        "detention_storage_m3": detention_m3,
        "design_storm_years": std.get("infrastructure.stormwater.design_storm_years"),
        "channel_conflict_ha": conflict_ha,
    }

    plan.missing_inputs.append(
        "Rainfall intensity-duration-frequency data for the district. Without "
        "it, pipe and channel capacities cannot be sized, and the detention "
        "figure above is the ruleset per-hectare rule rather than a routed "
        "storm.")

    if conflict_ha > 0.5:
        plan.warnings.append(
            f"{conflict_ha:.1f} ha of development sits on natural drainage "
            f"channels. Either the channel is culverted and the plan carries "
            f"that cost, or those blocks move: filling a channel relocates the "
            f"flood into the streets built over it.")
    if plan.culverts:
        plan.warnings.append(
            f"{len(plan.culverts)} road crossings of natural channels require "
            f"culverts or bridges. These are the crossings the road cost "
            f"surface priced, now located.")
    if not plan.outfalls:
        plan.warnings.append(
            "No stormwater outfall was found on the site boundary. Discharge "
            "arrangements must be confirmed with the downstream landowner.")
    return plan


# ---------------------------------------------------------------------------

def flow_accumulation(site: Site):
    """D8 flow accumulation.

    Returns the accumulated cell count, the cells that discharge off site, and
    the downstream neighbour of every cell.

    Each cell sends all its water to its steepest downslope neighbour, and
    totals are pushed downhill highest-first so every contributing cell is
    resolved before the cell it drains into. Single-direction rather than
    multiple-direction flow because what is wanted here is a channel network to
    keep development out of, and D8 yields discrete channels where
    multiple-direction routing yields a smear.

    Land outside the boundary is held far below the site rather than excluded.
    Excluding it makes the boundary a wall: water reaches the site edge and has
    nowhere to go, so it ponds at an interior low point and the plan reports no
    outfalls at all. Water does leave this site, and where it leaves is the one
    part of the drainage design nobody gets to choose.
    """
    inside = site.mask()
    floor = float(np.nanmin(site.dem[inside])) - 1000.0 if inside.any() else -1000.0
    dem = fill_sinks(np.where(inside, site.dem, floor), inside)

    rows, cols = site.shape
    # Seeded just below zero so a dead-flat spill path still picks a
    # direction: filling depressions creates flats by construction, and a
    # flat with no outflow direction is a pit again.
    best_drop = np.full(site.shape, -1e-9)
    target = np.full(site.shape, -1, dtype=np.int64)
    flat = np.arange(rows * cols).reshape(site.shape)

    for dr in (-1, 0, 1):
        for dc in (-1, 0, 1):
            if dr == 0 and dc == 0:
                continue
            shifted = np.full(site.shape, np.nan)
            idx = np.full(site.shape, -1, dtype=np.int64)
            r0, r1 = max(0, -dr), rows - max(0, dr)
            c0, c1 = max(0, -dc), cols - max(0, dc)
            shifted[r0:r1, c0:c1] = dem[r0 + dr:r1 + dr, c0 + dc:c1 + dc]
            idx[r0:r1, c0:c1] = flat[r0 + dr:r1 + dr, c0 + dc:c1 + dc]

            dist = np.hypot(dr, dc) * site.cell_size
            with np.errstate(invalid="ignore"):
                drop = (dem - shifted) / dist
            better = np.isfinite(drop) & (drop > best_drop + 1e-12)
            best_drop = np.where(better, drop, best_drop)
            target = np.where(better, idx, target)

    # Only site cells generate runoff; off-site cells exist to receive it.
    acc_flat = inside.astype(float).ravel()
    target_flat = target.ravel()
    inside_flat = inside.ravel()
    exits = np.zeros(rows * cols, dtype=bool)

    # Push totals downstream in topological order of the flow graph, not in
    # order of elevation. Two cells at the same level are ordered arbitrarily by
    # elevation, so on any flat the receiving cell can be processed before the
    # cell feeding it - and everything upstream of that point is silently
    # dropped. Draining each cell only once its own contributors are all in
    # makes the result independent of ties.
    indegree = np.zeros(rows * cols, dtype=np.int32)
    has_target = target_flat >= 0
    np.add.at(indegree, target_flat[has_target], 1)

    queue = list(np.flatnonzero(indegree == 0))
    while queue:
        cell = queue.pop()
        t_ = target_flat[cell]
        if t_ < 0:
            continue
        acc_flat[t_] += acc_flat[cell]
        if inside_flat[cell] and not inside_flat[t_]:
            exits[cell] = True
        indegree[t_] -= 1
        if indegree[t_] == 0:
            queue.append(t_)

    return (acc_flat.reshape(site.shape), exits.reshape(site.shape),
            target_flat.reshape(site.shape))


def fill_sinks(dem: np.ndarray, inside: np.ndarray) -> np.ndarray:
    """Raise closed depressions to their spill level.

    A raw surface is full of pits - some real, most of them survey noise or the
    artefact of a grid. Routing flow across an unfilled surface makes every pit
    a terminus: water arrives and stops, channels break into disconnected
    fragments, and the site appears to have no outfall at all, which is never
    true of real land.

    Priority-flood: flood inward from the boundary, always from the lowest cell
    reached so far, and raise each cell to the highest point the water had to
    climb to arrive. Every cell then has a downhill path off the site, which is
    what the accumulation pass assumes.
    """
    import heapq

    rows, cols = dem.shape
    filled = np.full(dem.shape, np.inf)
    seen = np.zeros(dem.shape, dtype=bool)
    heap: list[tuple[float, int, int]] = []

    # Seed from everything off site: that is where water is free to leave.
    edge = ~inside
    if not edge.any():
        edge = np.zeros(dem.shape, dtype=bool)
        edge[0, :] = edge[-1, :] = edge[:, 0] = edge[:, -1] = True

    for r, c in np.argwhere(edge):
        filled[r, c] = dem[r, c]
        seen[r, c] = True
        heap.append((float(dem[r, c]), int(r), int(c)))
    heapq.heapify(heap)

    while heap:
        level, r, c = heapq.heappop(heap)
        for dr in (-1, 0, 1):
            for dc in (-1, 0, 1):
                if dr == 0 and dc == 0:
                    continue
                rr, cc = r + dr, c + dc
                if not (0 <= rr < rows and 0 <= cc < cols) or seen[rr, cc]:
                    continue
                seen[rr, cc] = True
                # A cell cannot drain below the lowest lip it has to cross. The
                # epsilon matters: filling to exactly the spill level leaves a
                # dead flat surface with no drainage direction anywhere on it,
                # so the pit is replaced by a lake. A millimetre of tilt back
                # toward the spill point costs nothing and gives every cell a
                # direction.
                filled[rr, cc] = max(dem[rr, cc], level + FILL_EPSILON_M)
                heapq.heappush(heap, (float(filled[rr, cc]), rr, cc))

    return np.where(np.isfinite(filled), filled, dem)


def _trace_channels(site: Site, channel: np.ndarray, catchment_ha: np.ndarray,
                    target: np.ndarray, reserve_w: float) -> list[DrainageLine]:
    """Trace the channel network as flow paths, from each head downstream.

    Following flow direction, not sorting cells by height. A watercourse network
    branches, and a branching set of cells ordered by elevation and joined into
    one line is not a watercourse - it is a zigzag through every tributary in
    turn, hundreds of kilometres long, crossing the roads it never crosses on
    the ground.

    Each trace runs from a head - a channel cell nothing upstream flows into -
    down to wherever it meets a path already traced. The result is one line per
    tributary, meeting at confluences, which is what a drainage plan draws.
    """
    rows, cols = site.shape
    flat_channel = channel.ravel()
    flat_target = target.ravel()

    # A head is a channel cell with no channel cell draining into it.
    upstream = np.zeros(rows * cols, dtype=np.int32)
    idx = np.flatnonzero(flat_channel)
    for cell in idx:
        t_ = flat_target[cell]
        if t_ >= 0 and flat_channel[t_]:
            upstream[t_] += 1
    heads = [int(c) for c in idx if upstream[c] == 0]

    # Longest tributaries first, so the trunk is traced whole rather than in
    # pieces interrupted by whichever branch happened to be walked first.
    heads.sort(key=lambda c: -catchment_ha.ravel()[c])

    visited = np.zeros(rows * cols, dtype=bool)
    out: list[DrainageLine] = []

    for head in heads:
        cells: list[tuple[int, int]] = []
        cur = head
        guard = 0
        while cur >= 0 and flat_channel[cur] and guard < rows * cols:
            cells.append(divmod(int(cur), cols))
            if visited[cur]:
                break                      # joined an existing path: confluence
            visited[cur] = True
            cur = int(flat_target[cur])
            guard += 1

        if len(cells) < 3:
            continue
        line = cells_to_line(cells, site)
        if line is None or line.is_empty or line.length < site.cell_size * 2:
            continue
        out.append(DrainageLine(
            geometry=line,
            catchment_ha=float(max(catchment_ha[r, c] for r, c in cells)),
            reserve_width_m=reserve_w,
        ))
    return out


def _outfalls(exits: np.ndarray, catchment_ha: np.ndarray, site: Site
              ) -> list[Point]:
    """Where flow crosses the site boundary, largest catchment first.

    Clustered exits are collapsed: a channel two cells wide leaves through two
    cells and is still one outfall, and one outfall is one structure to build
    and one downstream landowner to negotiate with.
    """
    from shapely.geometry import Point as _P

    found = []
    for r, c in np.argwhere(exits):
        x, y = site.transform * (c + 0.5, r + 0.5)
        found.append((float(catchment_ha[r, c]), _P(x, y)))
    found.sort(key=lambda t_: -t_[0])

    kept: list[Point] = []
    for area, p in found:
        if area < CHANNEL_MIN_CATCHMENT_HA:
            continue
        if all(p.distance(q) > 100 for q in kept):
            kept.append(p)
    return kept


def _culverts(channels: list[DrainageLine], network) -> list[Point]:
    """Where the road network crosses a natural channel."""
    if not channels or network is None:
        return []
    roads = unary_union([s.geometry for s in network.segments])
    pts: list[Point] = []
    for ch in channels:
        hit = ch.geometry.intersection(roads)
        if hit.is_empty:
            continue
        for g in getattr(hit, "geoms", [hit]):
            p = g if g.geom_type == "Point" else g.centroid
            if all(p.distance(q) > 30 for q in pts):
                pts.append(p)
    return pts


def _developed_mask(site: Site, blocks) -> np.ndarray:
    geoms = [b.geometry for b in blocks.blocks]
    if not geoms:
        return np.zeros(site.shape, dtype=bool)
    return site.rasterize(unary_union(geoms))
