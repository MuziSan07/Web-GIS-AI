"""Shared raster/vector conversion helpers.

Kept in one place so the map, the exports and the block generator all derive
polygons the same way. Divergent polygonisation is a subtle source of
"the areas don't tie up" bugs.
"""

from __future__ import annotations

import numpy as np
from rasterio import features
from shapely import make_valid, set_precision
from shapely.errors import GEOSException
from shapely.geometry import LineString, MultiPolygon, Polygon, shape
from shapely.ops import unary_union

from .site.model import Site


def polygonise(arr: np.ndarray, site: Site, min_area_m2: float = 0.0) -> list[Polygon]:
    """Boolean raster to polygons in the site CRS.

    Every polygon is repaired before it leaves this function. Raster regions
    that pinch to a single cell produce rings that touch themselves, which are
    invalid; they survive rotation and buffering unnoticed and then blow up
    deep inside an unrelated module as a GEOS side-location conflict. Repairing
    at the source is the only place that fixes it once.
    """
    out: list[Polygon] = []
    if not arr.any():
        return out
    for geom, val in features.shapes(
        arr.astype(np.uint8), mask=arr, transform=site.transform
    ):
        if val != 1:
            continue
        g = shape(geom)
        if not g.is_valid:
            g = g.buffer(0)
        # A repair can split a pinched region into several polygons.
        for part in (g.geoms if isinstance(g, MultiPolygon) else [g]):
            if part.geom_type == "Polygon" and part.area >= min_area_m2:
                out.append(part)
    return out


def cells_to_line(cells: list[tuple[int, int]], site: Site,
                  simplify_m: float | None = None) -> LineString | None:
    """Convert a raster path to a centreline in projected coordinates.

    Raster paths are staircases. Simplifying is not cosmetic: an unsimplified
    centreline buffered to a road reserve produces a serrated edge that breaks
    downstream block and parcel geometry.
    """
    if len(cells) < 2:
        return None
    pts = [site.transform * (c + 0.5, r + 0.5) for r, c in cells]
    line = LineString(pts)
    tol = simplify_m if simplify_m is not None else site.cell_size * 1.5
    return line.simplify(tol, preserve_topology=False)


def clean(geom):
    """Repair a geometry, returning a Polygon/MultiPolygon."""
    if geom is None or geom.is_empty:
        return geom
    if not geom.is_valid:
        geom = make_valid(geom)
    return geom


def safe_intersection(a, b):
    """Intersect two geometries, surviving GEOS robustness failures.

    A geometry can report itself valid and still raise a side location conflict
    during an overlay: the ring is well formed but two edges are closer than
    floating point can resolve. This is a known class of GEOS failure and the
    remedy is escalation - repair, then snap to a coarse grid - rather than
    trying to guess which input is at fault. Losing a millimetre of precision
    is irrelevant at planning scale; crashing the pipeline is not.
    """
    try:
        return a.intersection(b)
    except GEOSException:
        pass

    a2, b2 = make_valid(a), make_valid(b)
    try:
        return a2.intersection(b2)
    except GEOSException:
        # 1 mm grid. Planning geometry carries no meaning below this.
        return set_precision(a2, 0.001).intersection(set_precision(b2, 0.001))


def largest_parts(geom, min_area_m2: float) -> list[Polygon]:
    """Explode to polygon parts above a minimum area."""
    if geom is None or geom.is_empty:
        return []
    parts = list(geom.geoms) if isinstance(geom, MultiPolygon) else [geom]
    return [p for p in parts if p.area >= min_area_m2]


def union(geoms) -> object:
    return unary_union([g for g in geoms if g is not None and not g.is_empty])
