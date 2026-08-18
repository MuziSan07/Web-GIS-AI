"""Vectorise raster results and export GeoJSON for the web client.

Web clients expect WGS84, while all computation happens in projected metres, so
this module is the single place the reprojection happens. Areas are computed
before reprojection and carried as attributes, because area computed from
degrees is wrong.
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
from rasterio import features
from shapely.geometry import mapping, shape
from shapely.ops import transform as shapely_transform
from pyproj import Transformer

from ..allocate.zones import UNASSIGNED, AllocationResult
from ..analysis.constraints import ConstraintResult
from ..site.model import Site

# Colours are defined once, here, so the map legend, the 3D view and the printed
# plan set cannot drift apart.
USE_COLOURS = {
    "residential": "#E8B84B",
    "commercial": "#D1495B",
    "industrial": "#8B5E83",
    "institutional": "#4A7BA7",
    "recreation_open_space": "#5C9E62",
    "circulation": "#6B6B6B",
    "utilities": "#B07D3A",
}

CONSTRAINT_COLOURS = {
    "wetland": "#2E7D9A",
    "watercourse": "#3D8FB0",
    "protected_area": "#2F6B3F",
    "powerline": "#9A8C2E",
    "steep_slope": "#8A6A5C",
}


def _to_wgs84(geom, src_crs: str):
    tf = Transformer.from_crs(src_crs, "EPSG:4326", always_xy=True)
    return shapely_transform(lambda x, y, z=None: tf.transform(x, y), geom)


def _polygonise(arr: np.ndarray, site: Site, min_area_m2: float = 0.0):
    """Turn a boolean raster into polygons in the site CRS."""
    out = []
    for geom, val in features.shapes(
        arr.astype(np.uint8), mask=arr, transform=site.transform
    ):
        if val != 1:
            continue
        g = shape(geom)
        if g.area >= min_area_m2:
            out.append(g)
    return out


def plan_to_geojson(
    site: Site,
    constraints: ConstraintResult,
    allocation: AllocationResult | None = None,
    min_polygon_m2: float = 400.0,
) -> dict:
    """Assemble the whole plan as one FeatureCollection ready for MapLibre."""
    features_out: list[dict] = []

    # -- site boundary ----------------------------------------------------
    features_out.append({
        "type": "Feature",
        "geometry": mapping(_to_wgs84(site.boundary, site.crs)),
        "properties": {
            "layer": "boundary",
            "name": site.name,
            "area_ha": round(site.area_ha, 2),
        },
    })

    # -- constraints ------------------------------------------------------
    for key, arr in constraints.exclusions.items():
        for g in _polygonise(arr, site, min_polygon_m2):
            features_out.append({
                "type": "Feature",
                "geometry": mapping(_to_wgs84(g, site.crs)),
                "properties": {
                    "layer": "constraint",
                    "constraint": key,
                    "colour": CONSTRAINT_COLOURS.get(key, "#999999"),
                    "area_ha": round(g.area / 10_000, 3),
                    "rule": constraints.reasons.get(key, ""),
                },
            })

    # -- allocated zones --------------------------------------------------
    if allocation is not None:
        for i, use in enumerate(allocation.uses):
            arr = allocation.labels == i
            for g in _polygonise(arr, site, min_polygon_m2):
                features_out.append({
                    "type": "Feature",
                    "geometry": mapping(_to_wgs84(g, site.crs)),
                    "properties": {
                        "layer": "zone",
                        "use": use,
                        "colour": USE_COLOURS.get(use, "#CCCCCC"),
                        "area_ha": round(g.area / 10_000, 3),
                        "mean_suitability": allocation.mean_suitability.get(use),
                    },
                })

    # -- existing features the plan must respond to -----------------------
    for key in ("existing_road", "powerline", "watercourse"):
        geom = site.layers.get(key)
        if geom is None:
            continue
        clipped = geom.intersection(site.boundary.buffer(50))
        if clipped.is_empty:
            continue
        features_out.append({
            "type": "Feature",
            "geometry": mapping(_to_wgs84(clipped, site.crs)),
            "properties": {"layer": "existing", "feature": key},
        })

    return {"type": "FeatureCollection", "features": features_out}


def write_geojson(fc: dict, path: str | Path) -> Path:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as fh:
        json.dump(fc, fh)
    return path
