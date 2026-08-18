"""Terrain derivatives.

Slope drives almost every siting decision downstream: what can be built on, what
roads can climb, where stormwater goes. It is computed once and reused.
"""

from __future__ import annotations

import numpy as np

from ..site.model import Site


def slope_percent(site: Site) -> np.ndarray:
    """Slope as a percentage, by the Horn method used in standard GIS packages.

    Percent rather than degrees because planning and engineering standards are
    written in percent and gradients (1:8, 6%), and converting at every
    comparison invites errors.
    """
    z = site.dem
    cs = site.cell_size
    # Horn's 3x3 kernel, edge-padded so the result keeps the grid shape.
    p = np.pad(z, 1, mode="edge")
    dzdx = ((p[:-2, 2:] + 2 * p[1:-1, 2:] + p[2:, 2:])
            - (p[:-2, :-2] + 2 * p[1:-1, :-2] + p[2:, :-2])) / (8 * cs)
    dzdy = ((p[2:, :-2] + 2 * p[2:, 1:-1] + p[2:, 2:])
            - (p[:-2, :-2] + 2 * p[:-2, 1:-1] + p[:-2, 2:])) / (8 * cs)
    return np.hypot(dzdx, dzdy) * 100.0


def aspect_degrees(site: Site) -> np.ndarray:
    """Downslope direction in degrees clockwise from north. Flat cells return -1."""
    z = site.dem
    cs = site.cell_size
    p = np.pad(z, 1, mode="edge")
    dzdx = ((p[:-2, 2:] + 2 * p[1:-1, 2:] + p[2:, 2:])
            - (p[:-2, :-2] + 2 * p[1:-1, :-2] + p[2:, :-2])) / (8 * cs)
    dzdy = ((p[2:, :-2] + 2 * p[2:, 1:-1] + p[2:, 2:])
            - (p[:-2, :-2] + 2 * p[:-2, 1:-1] + p[:-2, 2:])) / (8 * cs)
    asp = np.degrees(np.arctan2(dzdy, -dzdx))
    asp = (90.0 - asp) % 360.0
    asp[np.hypot(dzdx, dzdy) < 1e-9] = -1.0
    return asp


def relative_elevation(site: Site) -> np.ndarray:
    """Height above the local minimum, normalised 0-1 within the site.

    A cheap proxy for flood exposure and drainage position when no hydrological
    model is available. Replaced by real floodlines as soon as the client
    supplies them.
    """
    m = site.mask()
    z = np.where(m, site.dem, np.nan)
    lo = np.nanmin(z)
    hi = np.nanmax(z)
    if hi - lo < 1e-9:
        return np.zeros_like(site.dem)
    return np.nan_to_num((site.dem - lo) / (hi - lo), nan=0.0)
