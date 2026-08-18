"""Shared raster/vector conversion helpers.

Kept in one place so the map, the exports and the block generator all derive
polygons the same way. Divergent polygonisation is a subtle source of
"the areas don't tie up" bugs.
"""

from __future__ import annotations

import numpy as np
from rasterio import features
from shapely.geometry import LineString, MultiPolygon, Polygon, shape
from shapely.ops import unary_union

from .site.model import Site


def polygonise(arr: np.ndarray, site: Site, min_area_m2: float = 0.0) -> list[Polygon]:
    """Boolean raster to polygons in the site CRS."""
    out: list[Polygon] = []
    if not arr.any():
        return out
    for geom, val in features.shapes(
        arr.astype(np.uint8), mask=arr, transform=site.transform
    ):
        if val != 1:
            continue
        g = shape(geom)
        if g.area >= min_area_m2:
            out.append(g)
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
    """Repair a geometry and drop slivers, returning a Polygon/MultiPolygon."""
    if geom is None or geom.is_empty:
        return geom
    if not geom.is_valid:
        geom = geom.buffer(0)
    return geom


def largest_parts(geom, min_area_m2: float) -> list[Polygon]:
    """Explode to polygon parts above a minimum area."""
    if geom is None or geom.is_empty:
        return []
    parts = list(geom.geoms) if isinstance(geom, MultiPolygon) else [geom]
    return [p for p in parts if p.area >= min_area_m2]


def union(geoms) -> object:
    return unary_union([g for g in geoms if g is not None and not g.is_empty])
