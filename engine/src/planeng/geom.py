"""Shared raster/vector conversion helpers.

Kept in one place so the map, the exports and the block generator all derive
polygons the same way. Divergent polygonisation is a subtle source of
"the areas don't tie up" bugs.
"""

from __future__ import annotations

import numpy as np
from rasterio import features
from shapely import coverage_simplify, make_valid, set_precision
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
            if part.geom_type != "Polygon" or part.area < min_area_m2:
                continue
            out.append(part)
    return out


def destaircase(geoms: list, cell_size: float) -> list:
    """Remove the raster staircase from a layer of polygons.

    Every boundary traced from a grid is a flight of steps one cell high. Left
    alone the plan looks pixelated, and a stepped parcel boundary carries dozens
    of vertices that mean nothing on the ground.

    The whole layer is simplified as a coverage, not polygon by polygon.
    Simplifying each in isolation moves shared boundaries independently, so
    neighbouring parcels drift apart or overlap by metres - which is a
    cadastral defect, not a cosmetic one. Coverage simplification moves a
    shared edge once, for both sides.

    This is presentation only. The geometry the plan is computed from stays
    exact; only what is drawn is smoothed.
    """
    if not geoms:
        return geoms
    tolerance = cell_size * 0.75
    try:
        simplified = list(coverage_simplify(geoms, tolerance=tolerance))
    except Exception:
        # Not a clean coverage (overlaps, or a GEOS refusal). Fall back to
        # per-geometry simplification, which is imperfect but still better
        # than emitting raw staircases.
        simplified = [g.simplify(tolerance, preserve_topology=True) for g in geoms]

    out = []
    for original, s in zip(geoms, simplified):
        s = clean(s)
        # A simplification that loses a tenth of the area has cut a corner it
        # should not have; keep the original rather than misreport the extent.
        if (s.is_empty or s.geom_type not in ("Polygon", "MultiPolygon")
                or abs(s.area - original.area) > original.area * 0.10):
            out.append(original)
        else:
            out.append(s)
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


def safe_difference(a, b):
    """Subtract b from a, surviving GEOS robustness failures.

    The same escalation as `safe_intersection`, and needed for the same reason:
    a geometry repeatedly carved by earlier operations reports itself valid and
    still raises a side location conflict on the next overlay.
    """
    try:
        return a.difference(b)
    except GEOSException:
        pass

    a2, b2 = make_valid(a), make_valid(b)
    try:
        return a2.difference(b2)
    except GEOSException:
        return set_precision(a2, 0.001).difference(set_precision(b2, 0.001))


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
