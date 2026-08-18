"""Synthetic site generator.

Produces a plausible 600 ha greenfield site so the full pipeline runs end to end
before survey data arrives. The terrain is not random noise: it has a drainage
line, a ridge and a consistent regional fall, because a planner reviewing the
output must be able to tell whether the engine is responding to landform
sensibly or just producing patterns.

When real data lands this module is replaced by an ingest step. Nothing
downstream changes.
"""

from __future__ import annotations

import numpy as np
from affine import Affine
from scipy import ndimage
from shapely.geometry import LineString, Point, Polygon

from ..crs import UTM_35S
from .model import Site

# A location east of Gaborone, in UTM 35S metres. Arbitrary but realistic:
# it puts the site in the right projection with the right coordinate magnitudes.
ORIGIN_E = 400_000.0
ORIGIN_N = 7_240_000.0


def _boundary(width: float, height: float, seed: int) -> Polygon:
    """An irregular site boundary rather than a rectangle.

    Real sites are bounded by roads, rivers and neighbouring holdings. A
    rectangular test site hides edge-effect bugs in the layout engine.
    """
    rng = np.random.default_rng(seed)
    cx, cy = ORIGIN_E + width / 2, ORIGIN_N + height / 2
    n = 14
    angles = np.linspace(0, 2 * np.pi, n, endpoint=False)
    # Radius varies smoothly so the outline undulates instead of spiking.
    base = min(width, height) / 2
    wobble = 1 + 0.18 * np.sin(3 * angles + rng.uniform(0, 2 * np.pi)) \
               + 0.09 * np.cos(5 * angles + rng.uniform(0, 2 * np.pi))
    xs = cx + base * wobble * np.cos(angles) * (width / min(width, height))
    ys = cy + base * wobble * np.sin(angles) * (height / min(width, height))
    return Polygon(zip(xs, ys)).buffer(0)


def generate_site(
    name: str = "Synthetic Greenfield Site",
    target_area_ha: float = 600.0,
    cell_size: float = 10.0,
    seed: int = 42,
) -> Site:
    """Build a synthetic site of approximately ``target_area_ha`` hectares."""
    rng = np.random.default_rng(seed)

    # Size the bounding box so the irregular boundary lands near the target area.
    side = np.sqrt(target_area_ha * 10_000)
    width = height = side * 1.30

    boundary = _boundary(width, height, seed)
    # Scale the boundary to hit the requested area precisely.
    scale = np.sqrt(target_area_ha * 10_000 / boundary.area)
    boundary = Polygon(
        [((x - boundary.centroid.x) * scale + boundary.centroid.x,
          (y - boundary.centroid.y) * scale + boundary.centroid.y)
         for x, y in boundary.exterior.coords]
    )

    minx, miny, maxx, maxy = boundary.bounds
    pad = cell_size * 4
    minx, miny, maxx, maxy = minx - pad, miny - pad, maxx + pad, maxy + pad

    cols = int(np.ceil((maxx - minx) / cell_size))
    rows = int(np.ceil((maxy - miny) / cell_size))
    transform = Affine.translation(minx, maxy) * Affine.scale(cell_size, -cell_size)

    # -- terrain --------------------------------------------------------
    yy, xx = np.mgrid[0:rows, 0:cols].astype(float)
    nx = xx / cols
    ny = yy / rows

    # Regional fall to the south-east, a ridge, and a broad valley.
    dem = (
        1010.0
        - 22.0 * nx - 16.0 * ny                       # regional gradient
        + 14.0 * np.exp(-((nx - 0.25) ** 2 + (ny - 0.30) ** 2) / 0.035)   # ridge
        - 9.0 * np.exp(-((nx - 0.72) ** 2) / 0.02)    # valley aligned N-S
        + 3.5 * np.sin(nx * 9.0) * np.cos(ny * 7.0)   # local undulation
    )
    # Mild correlated roughness, smoothed so slopes stay physical.
    rough = ndimage.uniform_filter(rng.normal(0, 1.1, size=(rows, cols)), size=5)
    dem += rough

    site = Site(
        name=name, boundary=boundary, crs=UTM_35S,
        transform=transform, shape=(rows, cols), dem=dem,
    )

    # -- constraint layers ----------------------------------------------
    # Watercourse follows the valley floor.
    valley_col = 0.72 * cols
    pts = []
    for r in range(0, rows, max(1, rows // 30)):
        c = valley_col + 12 * np.sin(r / rows * 5.0)
        x, y = transform * (c, r)
        pts.append((x, y))
    watercourse = LineString(pts)
    site.layers["watercourse"] = watercourse

    # A wetland pan in the lower reach of the valley.
    wx, wy = transform * (valley_col + 6, rows * 0.78)
    site.layers["wetland"] = Point(wx, wy).buffer(190).intersection(boundary)

    # A protected woodland block on the ridge.
    px, py = transform * (cols * 0.22, rows * 0.26)
    site.layers["protected_area"] = Point(px, py).buffer(230).intersection(boundary)

    # An existing regional road clipping the northern edge, which the road
    # network must connect to.
    ax, ay = transform * (0, rows * 0.14)
    bx, by = transform * (cols, rows * 0.06)
    site.layers["existing_road"] = LineString([(ax, ay), (bx, by)])

    # An existing powerline servitude crossing the site.
    cx1, cy1 = transform * (cols * 0.05, rows)
    cx2, cy2 = transform * (cols * 0.95, 0)
    site.layers["powerline"] = LineString([(cx1, cy1), (cx2, cy2)])

    return site
