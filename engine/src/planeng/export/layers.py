"""The plan as a set of named layers.

Every export format - GeoPackage, shapefile, file geodatabase, DXF, IFC, the PDF
plan set - is a rendering of the same thing, so the decision about what a layer
*is* and what attributes it carries belongs in one place. Otherwise the DXF and
the GeoPackage drift apart and the plot numbers on a drawing stop matching the
plot numbers in the data, which is the sort of discrepancy that is discovered by
a conveyancer two years later.

Attribute *names* are kept to ten characters throughout, and that is a shapefile
constraint rather than a preference. Shapefiles silently truncate longer names,
so two fields called `envelope_area_m2` and `envelope_area_pct` both become
`envelope_a` and one of them is lost. Every other format here would take the
long names happily; they are short so that one canonical schema survives export
to the worst format on the list, rather than each format getting its own.

That limit is on names only. Values in a DBF run to 254 characters, and
clipping them to ten as well corrupts the data: `recreation_open_space` becomes
`recreation`, `residential` becomes `residentia`, and every consumer that
matches on a land use string - the plan set legend included - silently stops
matching. Values are written whole.

Geometry is kept in the site projection. Everything downstream measures areas
and distances, and a CAD drawing or a coordinate schedule in degrees is not
merely awkward, it is wrong.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import geopandas as gpd
from shapely.geometry import Point

# What each layer is for, carried through to the GeoPackage table description
# and the PDF legend so a recipient does not have to guess.
LAYER_PURPOSE = {
    "site_boundary": "Extent of the planning area.",
    "constraints": "Land excluded from development, with the rule that excludes it.",
    "zones": "Allocated land use.",
    "road_centreline": "Road centrelines by class.",
    "road_reserve": "Road reserve, the land to be transferred to the authority.",
    "blocks": "Development blocks bounded by the road network.",
    "parcels": "Cadastral plots for registration.",
    "building_envelopes": "Buildable area of each plot after setbacks.",
    "facilities": "Social and community facility sites.",
    "service_sites": "Engineering works: reservoir, treatment, landfill.",
    "water_mains": "Potable water distribution, sized on peak flow.",
    "sewers": "Foul sewer, with gradient and pumping status.",
    "electrical": "Medium voltage reticulation.",
    "pump_stations": "Sewer pump stations.",
    "drainage_channels": "Natural watercourses and their reserves.",
    "outfalls": "Points where stormwater leaves the site.",
    "culverts": "Road crossings of natural channels.",
}


@dataclass
class PlanLayers:
    """Named GeoDataFrames in the site CRS, ready for any writer."""

    crs: str
    name: str
    layers: dict[str, gpd.GeoDataFrame] = field(default_factory=dict)
    warnings: list[str] = field(default_factory=list)

    def __iter__(self):
        return iter(self.layers.items())

    def __len__(self) -> int:
        return len(self.layers)

    @property
    def feature_count(self) -> int:
        return sum(len(gdf) for gdf in self.layers.values())

    def non_empty(self) -> dict[str, gpd.GeoDataFrame]:
        return {k: v for k, v in self.layers.items() if len(v)}

    def summary(self) -> list[dict]:
        return [
            {
                "layer": name,
                "features": len(gdf),
                "geometry": (gdf.geom_type.iloc[0] if len(gdf) else None),
                "purpose": LAYER_PURPOSE.get(name, ""),
            }
            for name, gdf in self.layers.items()
        ]


# Which geometry class each layer holds. A shapefile is one class per file, and
# a mixed layer is nearly always a sign that an overlay returned a collection
# rather than a deliberate choice, so it is worth failing loudly on.
_EXPECTED_TYPE = {
    "road_centreline": "LineString", "water_mains": "LineString",
    "sewers": "LineString", "electrical": "LineString",
    "drainage_channels": "LineString",
    "pump_stations": "Point", "outfalls": "Point", "culverts": "Point",
}


def _single_type(name: str, gdf, out: PlanLayers):
    """Reduce a layer to one geometry class, keeping the dominant one.

    Every writer downstream wants a single class per layer and a shapefile
    refuses outright. Collections are exploded to their largest matching part
    rather than dropped, so a facility that came back as a polygon plus its own
    shared boundary is still exported as the polygon.
    """
    want = _EXPECTED_TYPE.get(name, "Polygon")
    types = set(gdf.geom_type)
    if types == {want}:
        return gdf

    fixed, dropped = [], 0
    for geom in gdf.geometry:
        if geom is None or geom.is_empty:
            fixed.append(None)
            dropped += 1
            continue
        if geom.geom_type in (want, f"Multi{want}"):
            fixed.append(geom)
            continue
        parts = [g for g in getattr(geom, "geoms", [])
                 if g.geom_type in (want, f"Multi{want}")]
        if parts:
            fixed.append(max(parts, key=lambda g: g.area if want == "Polygon"
                             else g.length))
        else:
            fixed.append(None)
            dropped += 1

    gdf = gdf.set_geometry(fixed, crs=gdf.crs)
    gdf = gdf[gdf.geometry.notna()].reset_index(drop=True)
    if dropped:
        out.warnings.append(
            f"{dropped} feature(s) in {name} had no {want} part and were "
            f"dropped from the export.")
    return gdf


def build_layers(plan) -> PlanLayers:
    """Turn a master plan into export-ready layers."""
    crs = plan.site.crs
    out = PlanLayers(crs=crs, name=plan.brief.project_name)

    def add(name: str, geoms, records: list[dict]) -> None:
        if not geoms:
            return
        gdf = gpd.GeoDataFrame(records, geometry=list(geoms), crs=crs)
        out.layers[name] = _single_type(name, gdf, out)

    # -- site and constraints ---------------------------------------------
    add("site_boundary", [plan.site.boundary],
        [{"name": plan.site.name[:60], "area_ha": round(plan.site.area_ha, 2),
          "crs": crs}])

    from ..geom import polygonise
    c_geoms, c_recs = [], []
    for key, arr in plan.constraints.exclusions.items():
        for g in polygonise(arr & plan.site.mask(), plan.site, 400.0):
            c_geoms.append(g)
            c_recs.append({"constraint": key,
                           "area_ha": round(g.area / 10_000, 3),
                           "rule": plan.constraints.reasons.get(key, "")[:250]})
    add("constraints", c_geoms, c_recs)

    # -- zones -------------------------------------------------------------
    z_geoms, z_recs = [], []
    for use in plan.allocation.uses:
        for g in polygonise(plan.allocation.mask_for(use), plan.site, 400.0):
            z_geoms.append(g)
            z_recs.append({"land_use": use,
                           "area_ha": round(g.area / 10_000, 3)})
    add("zones", z_geoms, z_recs)

    # -- roads -------------------------------------------------------------
    add("road_centreline",
        [s.geometry for s in plan.network.segments],
        [{"road_class": s.road_class,
          "reserve_m": s.reserve_width_m,
          "length_m": round(s.length_m, 1)} for s in plan.network.segments])

    if plan.network.reserve_polygon is not None:
        add("road_reserve", [plan.network.reserve_polygon],
            [{"area_ha": round(plan.network.reserve_area_ha, 2)}])

    # -- blocks and parcels -------------------------------------------------
    add("blocks", [b.geometry for b in plan.blocks.blocks],
        [{"block_id": b.block_id, "land_use": b.use,
          "area_ha": round(b.area_ha, 3), "plots": b.indicative_plots}
         for b in plan.blocks.blocks])

    add("parcels", [p.geometry for p in plan.cadastre.parcels],
        [{"plot_no": p.plot_number, "block_id": p.block_id,
          "band": p.band, "land_use": p.use,
          "area_m2": round(p.area_m2, 1),
          "frontage_m": round(p.frontage_m, 2)}
         for p in plan.cadastre.parcels])

    envs = [c for c in plan.controls.controls if c.envelope is not None
            and not c.envelope.is_empty]
    add("building_envelopes", [c.envelope for c in envs],
        [{"plot_no": c.plot_number, "band": c.band,
          "env_m2": round(c.envelope_area_m2, 1),
          "max_foot": round(c.max_footprint_m2, 1),
          "max_floor": round(c.max_floor_area_m2, 1),
          "storeys": c.max_storeys,
          "compliant": bool(c.compliant),
          # A violation list is the reason a plot is non compliant, and it is
          # the first thing anybody opening this file will look for.
          "violation": "; ".join(c.violations)[:250]} for c in envs])

    # -- facilities and works ----------------------------------------------
    fac = plan.facility_plan.facilities
    add("facilities", [f.geometry for f in fac],
        [{"fac_id": f.facility_id, "fac_type": f.key,
          "category": f.category, "area_ha": round(f.area_ha, 3),
          "catch_m": f.catchment_m} for f in fac])

    works = plan.infrastructure.works.sites
    add("service_sites", [w.geometry for w in works],
        [{"works": w.key, "label": w.label[:60],
          "area_ha": round(w.area_ha, 3), "level_m": round(w.level_m, 2),
          "compliant": bool(w.compliant),
          "notes": " ".join(w.notes)[:250]} for w in works])

    # -- services -----------------------------------------------------------
    infra = plan.infrastructure
    add("water_mains", [c.geometry for c in infra.water.conduits],
        [{"size_mm": int(c.size_mm), "flow_l_s": round(c.carries, 2),
          "length_m": round(c.length_m, 1),
          "road_class": c.road_class} for c in infra.water.conduits])

    add("sewers", [c.geometry for c in infra.sewer.conduits],
        [{"size_mm": int(c.size_mm), "flow_l_s": round(c.carries, 2),
          "length_m": round(c.length_m, 1),
          "grade_pct": round(c.gradient * 100, 3),
          "rising": bool(c.pumped)} for c in infra.sewer.conduits])

    add("electrical", [c.geometry for c in infra.electricity.conduits],
        [{"load_kva": round(c.carries, 1), "length_m": round(c.length_m, 1),
          "road_class": c.road_class}
         for c in infra.electricity.conduits])

    add("pump_stations", [Point(x, y) for x, y in infra.sewer.pump_stations],
        [{"service": "sewer"} for _ in infra.sewer.pump_stations])

    storm = infra.stormwater
    add("drainage_channels", [c.geometry for c in storm.channels],
        [{"catch_ha": round(c.catchment_ha, 2),
          "reserve_m": c.reserve_width_m,
          "length_m": round(c.length_m, 1)} for c in storm.channels])
    add("outfalls", list(storm.outfalls), [{"type": "outfall"}
                                           for _ in storm.outfalls])
    add("culverts", list(storm.culverts), [{"type": "culvert"}
                                           for _ in storm.culverts])

    return out
