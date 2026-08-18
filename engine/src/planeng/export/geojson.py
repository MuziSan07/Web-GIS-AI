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
from shapely.geometry import Point, mapping, shape
from shapely.ops import transform as shapely_transform
from pyproj import Transformer

from ..allocate.zones import UNASSIGNED, AllocationResult
from ..analysis.constraints import ConstraintResult
from ..geom import destaircase
from ..geom import polygonise as _polygonise
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

ROAD_COLOURS = {
    "primary_arterial": "#F2F2F2",
    "district_distributor": "#C9CDD6",
    "collector": "#9BA3B0",
    "local_access": "#6E7684",
}

ROAD_WIDTHS_PX = {
    "primary_arterial": 3.2,
    "district_distributor": 2.2,
    "collector": 1.5,
    "local_access": 0.8,
}

FACILITY_COLOURS = {
    "education": "#3E7CB1",
    "health": "#C24E6B",
    "safety": "#B5533C",
    "community": "#7A6BB5",
    "civic": "#8A7A4E",
    "recreation": "#3F8F55",
    "utilities": "#B07D3A",
}

BAND_COLOURS = {
    "low": "#F0D79B",
    "medium": "#E8B84B",
    "high": "#D89A2E",
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


def plan_to_geojson(
    site: Site,
    constraints: ConstraintResult,
    allocation: AllocationResult | None = None,
    network=None,
    blocks=None,
    cadastre=None,
    facility_plan=None,
    controls=None,
    infrastructure=None,
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

    # -- road reserves ----------------------------------------------------
    # Reserves are drawn as polygons because their width is a planning
    # instrument, not a rendering choice: a 40 m reserve must occupy 40 m.
    if network is not None and network.reserve_polygon is not None             and not network.reserve_polygon.is_empty:
        features_out.append({
            "type": "Feature",
            "geometry": mapping(_to_wgs84(network.reserve_polygon, site.crs)),
            "properties": {
                "layer": "road_reserve",
                "colour": "#3A4150",
                "area_ha": round(network.reserve_area_ha, 2),
            },
        })

    # -- road centrelines -------------------------------------------------
    if network is not None:
        for seg in network.segments:
            features_out.append({
                "type": "Feature",
                "geometry": mapping(_to_wgs84(seg.geometry, site.crs)),
                "properties": {
                    "layer": "road",
                    "road_class": seg.road_class,
                    "colour": ROAD_COLOURS.get(seg.road_class, "#AAAAAA"),
                    "width_px": ROAD_WIDTHS_PX.get(seg.road_class, 1.0),
                    "reserve_width_m": seg.reserve_width_m,
                    "length_m": round(seg.length_m, 1),
                },
            })

    # -- development blocks ----------------------------------------------
    if blocks is not None:
        for b in blocks.blocks:
            features_out.append({
                "type": "Feature",
                "geometry": mapping(_to_wgs84(b.geometry, site.crs)),
                "properties": {
                    "layer": "block",
                    "block_id": b.block_id,
                    "use": b.use,
                    "colour": USE_COLOURS.get(b.use, "#CCCCCC"),
                    "area_ha": round(b.area_ha, 3),
                    "indicative_plots": b.indicative_plots,
                },
            })

    # -- cadastral parcels ------------------------------------------------
    if cadastre is not None:
        for p in cadastre.parcels:
            features_out.append({
                "type": "Feature",
                "geometry": mapping(_to_wgs84(p.geometry, site.crs)),
                "properties": {
                    "layer": "parcel",
                    "plot_number": p.plot_number,
                    "block_id": p.block_id,
                    "band": p.band,
                    "colour": BAND_COLOURS.get(p.band, "#E8B84B"),
                    "area_m2": round(p.area_m2, 1),
                    "frontage_m": p.frontage_m,
                },
            })

    # -- sited facilities -------------------------------------------------
    if facility_plan is not None:
        for f in facility_plan.facilities:
            features_out.append({
                "type": "Feature",
                "geometry": mapping(_to_wgs84(f.geometry, site.crs)),
                "properties": {
                    "layer": "facility",
                    "facility_id": f.facility_id,
                    "key": f.key,
                    "category": f.category,
                    "colour": FACILITY_COLOURS.get(f.category, "#4A7BA7"),
                    "area_ha": round(f.area_ha, 3),
                    "catchment_m": f.catchment_m,
                },
            })

    # -- building envelopes -----------------------------------------------
    # What a developer may actually build to, after setbacks. Drawn inside the
    # plots so the two read together at high zoom.
    if controls is not None:
        for c in controls.controls:
            if c.envelope is None or c.envelope.is_empty:
                continue
            features_out.append({
                "type": "Feature",
                "geometry": mapping(_to_wgs84(c.envelope, site.crs)),
                "properties": {
                    "layer": "envelope",
                    "plot_number": c.plot_number,
                    "band": c.band,
                    "colour": "#8C6E2A",
                    "envelope_m2": c.envelope_area_m2,
                    "max_footprint_m2": c.max_footprint_m2,
                    "max_floor_area_m2": c.max_floor_area_m2,
                    "max_storeys": c.max_storeys,
                    "compliant": c.compliant,
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

    _smooth_layers(features_out, site.cell_size)
    # -- engineering services ---------------------------------------------
    if infrastructure is not None:
        features_out.extend(_infrastructure_features(infrastructure, site))

    return {"type": "FeatureCollection", "features": features_out}


def _smooth_layers(features_out: list[dict], cell_size: float) -> None:
    """De-staircase each polygon layer as a coverage, in place.

    Done here rather than in the engine because it is a drawing decision. Each
    layer is smoothed independently and as a whole, so shared boundaries within
    a layer move once instead of drifting apart.
    """
    by_layer: dict[str, list[int]] = {}
    for i, f in enumerate(features_out):
        if f["geometry"]["type"] in ("Polygon", "MultiPolygon"):
            by_layer.setdefault(f["properties"]["layer"], []).append(i)

    for layer, indices in by_layer.items():
        if layer == "boundary":
            continue          # the site boundary is surveyed, not rasterised
        geoms = [shape(features_out[i]["geometry"]) for i in indices]
        for i, g in zip(indices, destaircase(geoms, cell_size)):
            features_out[i]["geometry"] = mapping(g)


def write_geojson(fc: dict, path: str | Path) -> Path:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as fh:
        json.dump(fc, fh)
    return path


# ---------------------------------------------------------------------------

# Services are drawn as lines over the plan, so they carry their own colours
# rather than borrowing the land use palette.
SERVICE_COLOURS = {
    "water": "#1f78b4",
    "sewer": "#8b5a2b",
    "electricity": "#e6ab02",
    "stormwater": "#4daf4a",
}


def _infrastructure_features(infra, site: Site) -> list[dict]:
    """Every service run, works site and structure as its own feature.

    Runs are emitted individually rather than merged per service: a planner
    reviewing this needs to click a main and see its diameter and the flow it
    carries, and a merged multiline carries neither.
    """
    out: list[dict] = []

    for works in infra.works.sites:
        out.append({
            "type": "Feature",
            "geometry": mapping(_to_wgs84(works.geometry, site.crs)),
            "properties": {
                "layer": "service_site",
                "key": works.key,
                "label": works.label,
                "area_ha": round(works.area_ha, 2),
                "level_m": round(works.level_m, 1),
                "compliant": works.compliant,
                "notes": " ".join(works.notes),
                "colour": "#984ea3",
            },
        })

    for net in infra.networks:
        colour = SERVICE_COLOURS.get(net.service, "#777777")
        for c in net.conduits:
            props = {
                "layer": "service_run",
                "service": net.service,
                "length_m": round(c.length_m, 1),
                "road_class": c.road_class,
                "colour": colour,
            }
            if net.service == "electricity":
                props["load_kva"] = round(c.carries, 1)
            else:
                props["size_mm"] = int(c.size_mm)
                props["flow_l_per_s"] = round(c.carries, 2)
            if net.service == "sewer":
                props["gradient_pct"] = round(c.gradient * 100, 2)
                props["pumped"] = c.pumped
            out.append({
                "type": "Feature",
                "geometry": mapping(_to_wgs84(c.geometry, site.crs)),
                "properties": props,
            })

    for x, y in infra.sewer.pump_stations:
        out.append({
            "type": "Feature",
            "geometry": mapping(_to_wgs84(Point(x, y), site.crs)),
            "properties": {"layer": "pump_station", "service": "sewer",
                           "colour": "#8b5a2b"},
        })

    storm = infra.stormwater
    for ch in storm.channels:
        out.append({
            "type": "Feature",
            "geometry": mapping(_to_wgs84(ch.geometry, site.crs)),
            "properties": {
                "layer": "drainage_channel",
                "catchment_ha": round(ch.catchment_ha, 1),
                "reserve_width_m": ch.reserve_width_m,
                "colour": SERVICE_COLOURS["stormwater"],
            },
        })
    for pt in storm.outfalls:
        out.append({
            "type": "Feature",
            "geometry": mapping(_to_wgs84(pt, site.crs)),
            "properties": {"layer": "outfall",
                           "colour": SERVICE_COLOURS["stormwater"]},
        })
    for pt in storm.culverts:
        out.append({
            "type": "Feature",
            "geometry": mapping(_to_wgs84(pt, site.crs)),
            "properties": {"layer": "culvert", "colour": "#666666"},
        })

    return out
