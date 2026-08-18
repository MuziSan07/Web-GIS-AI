"""Make the road network one network.

The higher tiers are routed, so they connect by construction: the spine starts
at the external access point and every distributor is traced back to it. Local
access streets are not routed - they are grid lines clipped to the shape of a
residential zone. Nothing in that process obliges a street to reach a collector,
and on a site where a zone patch is separated from the rest by slope or a
watercourse, none of them do.

The result is a plan that looks like a town and is not one: blocks, plots and
facilities are all laid out against streets that cannot be driven to. It also
makes every utility undeliverable, since water, sewer and electrical all follow
the reserve.

This pass finds the road components the generator left separate and routes a
link road from each back to the main network over the same cost surface the
rest of the hierarchy used, so a link crosses steep ground or a watercourse only
where it is genuinely the cheapest option.

Fragments too short to be a street are removed rather than linked. A thirty
metre stub connected to nothing is a clipping artefact, and spending three
hundred metres of road reaching it would be worse than deleting it.
"""

from __future__ import annotations

import numpy as np

from ..geom import cells_to_line
from ..site.model import Site
from ..standards.loader import Standards
from .routing import GridRouter, build_cost_surface

# An isolated run shorter than this is a clipping artefact, not a street.
MIN_ISLAND_LENGTH_M = 150.0

# A link longer than this is worth telling the planner about: it means a chunk
# of the layout sits behind something the road hierarchy did not want to cross.
LONG_LINK_M = 300.0


def connect_islands(net, site: Site, std: Standards, constraints) -> None:
    """Link stranded road components to the main network, in place."""
    groups = _components(net)
    if len(groups) <= 1:
        return

    # The main network is the one with the most road in it, not the largest
    # area: reserve width varies by class, so area would favour a short
    # arterial stub over a whole neighbourhood of local streets.
    groups.sort(key=lambda g: -sum(net.segments[i].length_m for i in g))
    main, islands = groups[0], groups[1:]

    width = std.get("roads.hierarchy.local_access.reserve_width_m")
    cost = build_cost_surface(site, std, constraints, "local_access")

    keep_out: set[int] = set()
    dropped_m = 0.0
    to_link = []

    for island in islands:
        length = sum(net.segments[i].length_m for i in island)
        if length < MIN_ISLAND_LENGTH_M:
            keep_out.update(island)
            dropped_m += length
        else:
            to_link.append(island)

    if keep_out:
        net.segments[:] = [s for i, s in enumerate(net.segments)
                           if i not in keep_out]
        net.warnings.append(
            f"Removed {len(keep_out)} isolated street fragment(s) totalling "
            f"{dropped_m:,.0f} m that connected to no other road. These are "
            f"artefacts of clipping the street grid to the zone boundary."
        )
        # Segment indices just shifted; recompute the surviving groups.
        if not to_link:
            net.rebuild_reserves(site)
            return
        groups = _components(net)
        groups.sort(key=lambda g: -sum(net.segments[i].length_m for i in g))
        main, to_link = groups[0], groups[1:]

    if not to_link:
        net.rebuild_reserves(site)
        return

    main_mask = _mask_of(net, site, main)
    try:
        router = GridRouter(site, cost)
        dist, pred = router.solve_from_cells(main_mask)
    except ValueError as exc:
        net.warnings.append(f"Could not route link roads: {exc}")
        return

    linked = 0
    unreachable = 0
    added_m = 0.0

    for island in to_link:
        island_mask = _mask_of(net, site, island)
        cells = np.flatnonzero(island_mask.ravel())
        reach = dist[cells]
        if not np.isfinite(reach).any():
            unreachable += 1
            continue

        end = np.unravel_index(int(cells[int(np.argmin(reach))]), site.shape)
        path = router.trace(pred, (int(end[0]), int(end[1])))
        if len(path) < 2:
            unreachable += 1
            continue

        line = cells_to_line(path, site)
        if line is None or line.is_empty or line.length <= 0:
            unreachable += 1
            continue

        from .network import RoadSegment
        net.segments.append(RoadSegment(line, "local_access", width))
        linked += 1
        added_m += line.length
        if line.length > LONG_LINK_M:
            net.warnings.append(
                f"A {line.length:,.0f} m link road was needed to reach an "
                f"otherwise stranded part of the layout. Confirm this is "
                f"cheaper than moving the development it serves."
            )

    if linked:
        net.warnings.append(
            f"Added {linked} link road(s) totalling {added_m:,.0f} m to join "
            f"street grids the layout had left unconnected to the network."
        )
    if unreachable:
        net.warnings.append(
            f"{unreachable} part(s) of the street layout could not be connected "
            f"to the road network at all: no route exists that does not cross "
            f"an impassable constraint. The development they serve is not "
            f"deliverable as drawn."
        )

    net.rebuild_reserves(site)


# ---------------------------------------------------------------------------

def _components(net) -> list[list[int]]:
    """Group segment indices whose reserves touch.

    Reserves, not centrelines: a local street that stops at the edge of a forty
    metre arterial reserve is connected in every sense that matters, even though
    its centreline stops twenty metres short of the arterial centreline.
    """
    from shapely.strtree import STRtree

    n = len(net.segments)
    if n == 0:
        return []

    buffers = [s.geometry.buffer(s.reserve_width_m / 2 + 0.5, cap_style=2)
               for s in net.segments]
    tree = STRtree(buffers)

    parent = list(range(n))

    def find(a: int) -> int:
        while parent[a] != a:
            parent[a] = parent[parent[a]]
            a = parent[a]
        return a

    def union(a: int, b: int) -> None:
        ra, rb = find(a), find(b)
        if ra != rb:
            parent[rb] = ra

    for i, buf in enumerate(buffers):
        for j in tree.query(buf):
            j = int(j)
            if j != i and buf.intersects(buffers[j]):
                union(i, j)

    groups: dict[int, list[int]] = {}
    for i in range(n):
        groups.setdefault(find(i), []).append(i)
    return list(groups.values())


def _mask_of(net, site: Site, indices) -> np.ndarray:
    """Raster mask of the reserves of a set of segments."""
    from shapely.ops import unary_union

    geoms = [net.segments[i].geometry.buffer(
        net.segments[i].reserve_width_m / 2, cap_style=2) for i in indices]
    mask = site.rasterize(unary_union(geoms))
    if not mask.any():
        # Very short segments can fall between cell centres. Burn their
        # endpoints so the component is still routable.
        for i in indices:
            for x, y in net.segments[i].geometry.coords:
                cell = _cell_of(site, x, y)
                if cell is not None:
                    mask[cell] = True
    return mask


def _cell_of(site: Site, x: float, y: float):
    col, row = ~site.transform * (x, y)
    r, c = int(round(row)), int(round(col))
    if 0 <= r < site.shape[0] and 0 <= c < site.shape[1]:
        return r, c
    return None
