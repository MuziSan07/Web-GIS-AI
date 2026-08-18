"""Internal access streets for over-deep blocks.

A block deeper than two plot rows cannot be subdivided from its frontages: the
middle is unreachable, and on the reference site that core was swallowing about
a third of the residential plot yield. The remedy is the one a designer would
apply - run a street down the block - and it has to happen after the blocks
exist, because only then is it known which are too deep.

Streets are laid parallel to the block's long axis, which is the direction the
perimeter streets already run, so the resulting plots keep a consistent
orientation instead of meeting the existing grid at an angle.
"""

from __future__ import annotations

import math

from shapely.geometry import LineString, MultiLineString

from ..site.model import Site
from ..standards.loader import Standards
from .blocks import BlockLayout
from .network import RoadNetwork, RoadSegment, _target_plot_depth

# A block is left alone unless it exceeds the two-row depth by this margin.
# Splitting a block only marginally too deep buys a sliver of frontage and
# costs a full road reserve.
DEPTH_TOLERANCE = 1.25


def add_internal_streets(
    site: Site,
    std: Standards,
    network: RoadNetwork,
    blocks: BlockLayout,
    programme=None,
) -> list[RoadSegment]:
    """Return internal streets needed to make deep blocks subdividable."""
    depth = _target_plot_depth(std, programme)
    width = std.get("roads.hierarchy.local_access.reserve_width_m")
    frontage_span = 2 * depth + width          # two plot rows plus the street

    added: list[RoadSegment] = []

    for block in blocks.for_use("residential"):
        geom = block.geometry
        metrics = _rect_metrics(geom)
        if metrics is None:
            continue
        short, long_len, ux, uy, cx, cy = metrics

        if short <= frontage_span * DEPTH_TOLERANCE:
            continue

        # How many streets are needed so no resulting strip exceeds two rows.
        strips = max(2, int(math.ceil(short / frontage_span)))
        n_streets = strips - 1
        if n_streets <= 0:
            continue

        # Perpendicular to the long axis, which is the offset direction.
        px, py = -uy, ux
        reach = long_len / 2 + width

        for k in range(1, n_streets + 1):
            offset = -short / 2 + k * short / strips
            mx, my = cx + px * offset, cy + py * offset
            line = LineString([
                (mx - ux * reach, my - uy * reach),
                (mx + ux * reach, my + uy * reach),
            ])
            clipped = line.intersection(geom)
            if clipped.is_empty:
                continue

            parts = (list(clipped.geoms)
                     if isinstance(clipped, MultiLineString) else [clipped])
            for part in parts:
                # A stub serving fewer than a couple of plots is not worth a
                # road reserve.
                if part.geom_type == "LineString" and part.length >= depth * 2:
                    added.append(RoadSegment(part, "local_access", width))

    network.segments.extend(added)
    return added


def _rect_metrics(geom):
    """Short side, long side, long-axis unit vector and centre of a block."""
    rect = geom.minimum_rotated_rectangle
    if rect.geom_type != "Polygon":
        return None
    coords = list(rect.exterior.coords)[:4]
    if len(coords) < 4:
        return None

    (x0, y0), (x1, y1), (x2, y2) = coords[0], coords[1], coords[2]
    e1 = math.hypot(x1 - x0, y1 - y0)
    e2 = math.hypot(x2 - x1, y2 - y1)
    if min(e1, e2) < 1e-6:
        return None

    if e1 >= e2:
        long_len, short, vx, vy = e1, e2, x1 - x0, y1 - y0
    else:
        long_len, short, vx, vy = e2, e1, x2 - x1, y2 - y1

    norm = math.hypot(vx, vy)
    c = rect.centroid
    return short, long_len, vx / norm, vy / norm, c.x, c.y
