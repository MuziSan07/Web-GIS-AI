"""Multi-criteria land suitability.

Each land use gets a 0-1 suitability surface built from weighted, normalised
criteria. The weights live in this module rather than the YAML ruleset because
they express planning judgement rather than statutory standards, and a planner
will want to tune them per project.

The criteria are deliberately explicit and inspectable. When a planner asks why
industrial land was pushed to the south-east, the answer is a list of named
criteria and weights, not a black box.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from scipy import ndimage

from ..site.model import Site
from ..standards.loader import Standards
from .constraints import ConstraintResult
from .terrain import relative_elevation

LandUse = str

# weight, higher-is-better flag per criterion
CRITERIA_WEIGHTS: dict[LandUse, dict[str, float]] = {
    "residential": {
        "gentle_slope": 0.30,
        "road_access": 0.20,
        "centrality": 0.15,
        "amenity_proximity": 0.15,
        "elevated_ground": 0.10,
        "industrial_separation": 0.10,
    },
    "commercial": {
        "road_access": 0.40,
        "centrality": 0.30,
        "gentle_slope": 0.20,
        "elevated_ground": 0.10,
    },
    "industrial": {
        "gentle_slope": 0.30,
        "road_access": 0.30,
        "edge_location": 0.25,
        "amenity_separation": 0.15,
    },
    "institutional": {
        "centrality": 0.35,
        "road_access": 0.25,
        "gentle_slope": 0.25,
        "industrial_separation": 0.15,
    },
    "recreation_open_space": {
        "amenity_proximity": 0.45,
        "centrality": 0.20,
        "steep_tolerance": 0.20,
        "road_access": 0.15,
    },
}


@dataclass
class SuitabilityResult:
    surfaces: dict[LandUse, np.ndarray]
    criteria: dict[str, np.ndarray]

    def best_use(self) -> tuple[np.ndarray, list[LandUse]]:
        """Argmax land use per cell, for a quick diagnostic view."""
        uses = sorted(self.surfaces)
        stack = np.stack([self.surfaces[u] for u in uses])
        return stack.argmax(axis=0), uses


def _norm(a: np.ndarray, mask: np.ndarray, invert: bool = False) -> np.ndarray:
    """Scale to 0-1 using the range inside the site only."""
    vals = a[mask]
    if vals.size == 0:
        return np.zeros_like(a, dtype=float)
    lo, hi = float(np.nanmin(vals)), float(np.nanmax(vals))
    if hi - lo < 1e-12:
        out = np.full_like(a, 0.5, dtype=float)
    else:
        out = (a - lo) / (hi - lo)
    out = np.clip(out, 0, 1)
    return 1.0 - out if invert else out


def _distance_to(site: Site, arr_true: np.ndarray) -> np.ndarray:
    """Euclidean distance in metres from every cell to the nearest True cell."""
    if not arr_true.any():
        return np.full(site.shape, np.inf)
    return ndimage.distance_transform_edt(~arr_true) * site.cell_size


def compute_suitability(
    site: Site, std: Standards, constraints: ConstraintResult
) -> SuitabilityResult:
    mask = site.mask()
    slope = constraints.slope
    crit: dict[str, np.ndarray] = {}

    # -- slope-derived ---------------------------------------------------
    crit["gentle_slope"] = _norm(slope, mask, invert=True)
    crit["steep_tolerance"] = _norm(slope, mask)          # recreation can take slope

    # -- access ----------------------------------------------------------
    road = site.layers.get("existing_road")
    if road is not None:
        d_road = _distance_to(site, site.rasterize(road.buffer(site.cell_size)))
    else:
        d_road = np.zeros(site.shape)
    crit["road_access"] = _norm(d_road, mask, invert=True)

    # -- centrality ------------------------------------------------------
    cx, cy = site.boundary.centroid.x, site.boundary.centroid.y
    rows, cols = site.shape
    yy, xx = np.mgrid[0:rows, 0:cols]
    wx, wy = site.transform * (xx + 0.5, yy + 0.5)
    d_centre = np.hypot(wx - cx, wy - cy)
    crit["centrality"] = _norm(d_centre, mask, invert=True)

    # -- edge ------------------------------------------------------------
    crit["edge_location"] = _norm(_distance_to(site, ~mask), mask, invert=True)

    # -- amenity (water and protected land are attractors for housing and
    #    open space, and repellents for industry) -------------------------
    amenity = np.zeros(site.shape, dtype=bool)
    for key in ("watercourse", "wetland", "protected_area"):
        geom = site.layers.get(key)
        if geom is not None and not geom.is_empty:
            amenity |= site.rasterize(geom)
    d_amenity = _distance_to(site, amenity)
    crit["amenity_proximity"] = _norm(d_amenity, mask, invert=True)
    crit["amenity_separation"] = _norm(d_amenity, mask)

    # -- elevation -------------------------------------------------------
    crit["elevated_ground"] = relative_elevation(site)

    # Industrial separation is resolved after the industrial surface exists, so
    # seed it neutrally and refine below.
    crit["industrial_separation"] = np.full(site.shape, 0.5, dtype=float)

    surfaces: dict[LandUse, np.ndarray] = {}
    for use, weights in CRITERIA_WEIGHTS.items():
        surfaces[use] = _weighted(crit, weights, mask, constraints)

    # Refine: push housing and institutions away from the most industrially
    # suitable land, then recompute the two surfaces that depend on it.
    ind = surfaces["industrial"]
    likely_industrial = ind > np.quantile(ind[mask], 0.85) if mask.any() else ind > 1
    d_ind = _distance_to(site, likely_industrial & mask)
    crit["industrial_separation"] = _norm(np.minimum(d_ind, 1500.0), mask)
    for use in ("residential", "institutional"):
        surfaces[use] = _weighted(crit, CRITERIA_WEIGHTS[use], mask, constraints)

    return SuitabilityResult(surfaces=surfaces, criteria=crit)


def _weighted(
    crit: dict[str, np.ndarray],
    weights: dict[str, float],
    mask: np.ndarray,
    constraints: ConstraintResult,
) -> np.ndarray:
    total = sum(weights.values())
    out = np.zeros(mask.shape, dtype=float)
    for name, w in weights.items():
        out += crit[name] * (w / total)
    # Constrained land is not merely unsuitable, it is unavailable.
    out[~constraints.buildable] = 0.0
    return out
