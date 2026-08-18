"""IFC export: the site as a massing model.

Being clear about what this is matters more than the code. A master plan is not
a building model. There are no walls here, no slabs, no doors, and inventing
them would produce a file that looks like a design and is not one.

What a master plan *can* hand a project architect is the thing they otherwise
spend a week rebuilding: the site, its plots, the permitted envelope on each
plot, and where the roads and services run. That is a massing and context model,
and it is genuinely useful - an architect can open it, drop a building into an
envelope, and immediately see whether it breaks the setbacks or the storey
limit, because the envelope carries those as properties.

So each plot becomes an IfcSpatialZone carrying its planning attributes, and its
building envelope becomes a solid extruded to the permitted height. The height is
the control, not a proposal: a block sitting at exactly the maximum is what the
plot allows, not what anyone intends to build.

Coordinates are the site projection, shifted to a local origin. IFC uses single
precision floats in many viewers, and a coordinate of 7,242,000 loses
centimetres to rounding, so the true origin is written into IfcSite as the
georeferencing offset and the geometry is built near zero.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path

import ifcopenshell

from .layers import PlanLayers, build_layers

# Storey height used to turn a storey limit into a height. A planning control
# limits storeys; a solid needs metres.
STOREY_HEIGHT_M = 3.0


@dataclass
class BimResult:
    path: Path
    zones: int = 0
    envelopes: int = 0
    schema: str = "IFC4"
    origin: tuple[float, float] = (0.0, 0.0)
    warnings: list[str] = field(default_factory=list)

    def summary(self) -> dict:
        return {
            "format": "IFC",
            "schema": self.schema,
            "path": str(self.path),
            "zones": self.zones,
            "envelopes": self.envelopes,
            "local_origin": [round(v, 3) for v in self.origin],
            "warnings": self.warnings,
        }


def write_ifc(plan, path: str | Path, max_plots: int | None = None
              ) -> BimResult:
    """Write plots and their permitted envelopes as an IFC massing model."""
    layers = build_layers(plan)
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)

    minx, miny, _, _ = _origin(layers)
    result = BimResult(path=path, origin=(minx, miny))

    f = ifcopenshell.file(schema="IFC4")
    ctx, body, site, storey = _skeleton(f, plan, minx, miny)

    parcels = layers.layers.get("parcels")
    controls = {c.plot_number: c for c in plan.controls.controls}

    if parcels is not None:
        rows = parcels if max_plots is None else parcels.iloc[:max_plots]
        for plot_no, geom, band, area in zip(
                rows["plot_no"], rows.geometry, rows["band"], rows["area_m2"]):
            control = controls.get(plot_no)
            zone = _spatial_zone(f, body, site, geom, plot_no, band, area,
                                 minx, miny)
            if zone is not None:
                result.zones += 1
            if control is not None and control.envelope is not None \
                    and not control.envelope.is_empty:
                height = max(control.max_storeys, 1) * STOREY_HEIGHT_M
                if _envelope_solid(f, body, storey, control, height,
                                   minx, miny):
                    result.envelopes += 1

    f.write(str(path))

    result.warnings.append(
        "This is a massing and context model, not a building model. Plots carry "
        "their planning controls as properties and each solid is the permitted "
        "envelope, not a proposed building.")
    result.warnings.append(
        f"Geometry is built about a local origin at "
        f"{minx:,.1f}E {miny:,.1f}N in {plan.site.crs}, recorded on IfcSite. "
        f"Add it back before comparing against surveyed coordinates.")
    if max_plots is not None:
        result.warnings.append(
            f"Limited to the first {max_plots:,} plots.")
    return result


# ---------------------------------------------------------------------------

def _origin(layers: PlanLayers):
    boundary = layers.layers.get("site_boundary")
    gdf = boundary if boundary is not None and len(boundary) else next(
        iter(layers.non_empty().values()))
    return tuple(gdf.total_bounds)


def _skeleton(f, plan, minx, miny):
    """Project, units, geometric context and the spatial tree."""
    length_unit = f.create_entity("IfcSIUnit", UnitType="LENGTHUNIT",
                                  Name="METRE")
    area_unit = f.create_entity("IfcSIUnit", UnitType="AREAUNIT",
                                Name="SQUARE_METRE")
    units = f.create_entity("IfcUnitAssignment", Units=[length_unit, area_unit])

    origin = f.create_entity("IfcCartesianPoint", Coordinates=(0.0, 0.0, 0.0))
    axis = f.create_entity("IfcAxis2Placement3D", Location=origin)
    ctx = f.create_entity(
        "IfcGeometricRepresentationContext", ContextType="Model",
        CoordinateSpaceDimension=3, Precision=1e-5, WorldCoordinateSystem=axis)
    body = f.create_entity(
        "IfcGeometricRepresentationSubContext", ContextIdentifier="Body",
        ContextType="Model", ParentContext=ctx, TargetView="MODEL_VIEW")

    project = f.create_entity(
        "IfcProject", GlobalId=ifcopenshell.guid.new(),
        Name=plan.brief.project_name, UnitsInContext=units,
        RepresentationContexts=[ctx])

    placement = f.create_entity("IfcLocalPlacement", RelativePlacement=axis)
    site = f.create_entity(
        "IfcSite", GlobalId=ifcopenshell.guid.new(), Name=plan.site.name,
        ObjectPlacement=placement, CompositionType="ELEMENT",
        # The true position of the local origin, so the model can be put back.
        RefElevation=float(plan.site.dem[plan.site.mask()].min()),
        Description=f"Local origin {minx:.3f}E {miny:.3f}N {plan.site.crs}")
    f.create_entity("IfcRelAggregates", GlobalId=ifcopenshell.guid.new(),
                    RelatingObject=project, RelatedObjects=[site])

    building = f.create_entity(
        "IfcBuilding", GlobalId=ifcopenshell.guid.new(),
        Name=f"{plan.brief.project_name} development",
        ObjectPlacement=placement, CompositionType="ELEMENT")
    f.create_entity("IfcRelAggregates", GlobalId=ifcopenshell.guid.new(),
                    RelatingObject=site, RelatedObjects=[building])

    storey = f.create_entity(
        "IfcBuildingStorey", GlobalId=ifcopenshell.guid.new(),
        Name="Ground", ObjectPlacement=placement, CompositionType="ELEMENT",
        Elevation=0.0)
    f.create_entity("IfcRelAggregates", GlobalId=ifcopenshell.guid.new(),
                    RelatingObject=building, RelatedObjects=[storey])

    return ctx, body, site, storey


def _polyline(f, ring, minx, miny):
    """A closed IFC polyline from a shapely ring, shifted to local origin."""
    coords = [(x - minx, y - miny) for x, y in ring][:-1]
    if len(coords) < 3:
        return None
    points = [f.create_entity("IfcCartesianPoint", Coordinates=(float(x),
                                                                float(y)))
              for x, y in coords]
    points.append(points[0])
    return f.create_entity("IfcPolyline", Points=points)


def _extrude(f, body, geom, height: float, minx, miny):
    """An extruded solid from a polygon footprint."""
    poly = geom if geom.geom_type == "Polygon" else max(
        geom.geoms, key=lambda g: g.area)
    outer = _polyline(f, list(poly.exterior.coords), minx, miny)
    if outer is None:
        return None

    profile = f.create_entity("IfcArbitraryClosedProfileDef",
                              ProfileType="AREA", OuterCurve=outer)
    position = f.create_entity(
        "IfcAxis2Placement3D",
        Location=f.create_entity("IfcCartesianPoint",
                                 Coordinates=(0.0, 0.0, 0.0)))
    direction = f.create_entity("IfcDirection", DirectionRatios=(0.0, 0.0, 1.0))
    solid = f.create_entity(
        "IfcExtrudedAreaSolid", SweptArea=profile, Position=position,
        ExtrudedDirection=direction, Depth=float(height))
    shape = f.create_entity(
        "IfcShapeRepresentation", ContextOfItems=body,
        RepresentationIdentifier="Body", RepresentationType="SweptSolid",
        Items=[solid])
    return f.create_entity("IfcProductDefinitionShape", Representations=[shape])


def _spatial_zone(f, body, site, geom, plot_no, band, area, minx, miny):
    """A plot as a spatial zone carrying its planning attributes.

    IfcSpatialZone rather than IfcSpace: a plot is a planning designation over
    land, not a room, and a zone is what the schema provides for exactly that.
    """
    shape = _extrude(f, body, geom, 0.05, minx, miny)   # a thin land parcel
    if shape is None:
        return None
    zone = f.create_entity(
        "IfcSpatialZone", GlobalId=ifcopenshell.guid.new(), Name=str(plot_no),
        LongName=f"Plot {plot_no}", Representation=shape,
        PredefinedType="USERDEFINED", ObjectType=f"Planning plot ({band})")
    f.create_entity("IfcRelAggregates", GlobalId=ifcopenshell.guid.new(),
                    RelatingObject=site, RelatedObjects=[zone])
    _properties(f, zone, "Pset_PlanningPlot", {
        "PlotNumber": str(plot_no),
        "DensityBand": str(band),
        "PlotAreaM2": float(area),
    })
    return zone


def _envelope_solid(f, body, storey, control, height: float, minx, miny):
    """The permitted envelope as a solid, with the controls that produced it."""
    shape = _extrude(f, body, control.envelope, height, minx, miny)
    if shape is None:
        return None
    proxy = f.create_entity(
        "IfcBuildingElementProxy", GlobalId=ifcopenshell.guid.new(),
        Name=f"Envelope {control.plot_number}", Representation=shape,
        PredefinedType="USERDEFINED",
        ObjectType="Permitted building envelope")
    f.create_entity(
        "IfcRelContainedInSpatialStructure",
        GlobalId=ifcopenshell.guid.new(), RelatingStructure=storey,
        RelatedElements=[proxy])
    _properties(f, proxy, "Pset_DevelopmentControl", {
        "PlotNumber": str(control.plot_number),
        "MaxStoreys": int(control.max_storeys),
        "MaxFootprintM2": float(control.max_footprint_m2),
        "MaxFloorAreaM2": float(control.max_floor_area_m2),
        "EnvelopeAreaM2": float(control.envelope_area_m2),
        "Compliant": bool(control.compliant),
        "Violations": "; ".join(control.violations),
    })
    return proxy


def _properties(f, product, pset_name: str, values: dict) -> None:
    """Attach a property set. This is what makes the model worth opening."""
    props = []
    for key, value in values.items():
        if isinstance(value, bool):
            wrapped = f.create_entity("IfcBoolean", value)
        elif isinstance(value, int):
            wrapped = f.create_entity("IfcInteger", value)
        elif isinstance(value, float):
            wrapped = f.create_entity("IfcReal", value)
        else:
            wrapped = f.create_entity("IfcText", str(value))
        props.append(f.create_entity(
            "IfcPropertySingleValue", Name=key, NominalValue=wrapped))

    pset = f.create_entity(
        "IfcPropertySet", GlobalId=ifcopenshell.guid.new(), Name=pset_name,
        HasProperties=props)
    f.create_entity(
        "IfcRelDefinesByProperties", GlobalId=ifcopenshell.guid.new(),
        RelatedObjects=[product], RelatingPropertyDefinition=pset)
