"""DXF export, for the consultants who will actually detail this.

A master plan leaves the planner and goes to a civil engineer, a land surveyor
and a road designer, and all three work in CAD. What they need is not a picture
of the plan - it is the plan as geometry they can snap to, on layers they can
freeze, at full coordinate precision.

Two things matter more than anything else here.

**Coordinates stay projected.** A DXF has no coordinate reference system: it is
just numbers. Export a plan in degrees and it opens as a drawing two units
across, every dimension reads 0.00, and snapping is meaningless. Written in
metres on the site projection, the drawing lands in the right place in the
surveyor's world file and every distance is real. The projection is written into
the drawing as text for the same reason - the file cannot carry it any other way.

**Layer names match the GIS export.** Same names, same features, same plot
numbers. A drawing whose layers are named differently from the data it came from
is a reconciliation exercise for whoever receives both.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path

import ezdxf
from ezdxf.enums import TextEntityAlignment

from .layers import PlanLayers, build_layers

# AutoCAD Colour Index per layer. Chosen for a white background, which is what a
# drawing is plotted on, rather than for the dark map in the browser.
LAYER_COLOUR = {
    "site_boundary": 7,          # black/white
    "constraints": 8,            # grey
    "zones": 253,                # light grey, background wash
    "road_centreline": 1,        # red
    "road_reserve": 30,          # orange
    "blocks": 8,
    "parcels": 3,                # green
    "building_envelopes": 4,     # cyan
    "facilities": 6,             # magenta
    "service_sites": 200,
    "water_mains": 5,            # blue
    "sewers": 32,                # brown
    "electrical": 2,             # yellow
    "pump_stations": 32,
    "drainage_channels": 4,
    "outfalls": 4,
    "culverts": 8,
}

# Layers off by default: useful to have, too dense to plot.
FROZEN = {"building_envelopes", "zones", "blocks"}


@dataclass
class CadResult:
    path: Path
    layers: int = 0
    entities: int = 0
    labels: int = 0
    warnings: list[str] = field(default_factory=list)

    def summary(self) -> dict:
        return {
            "format": "DXF",
            "path": str(self.path),
            "layers": self.layers,
            "entities": self.entities,
            "labels": self.labels,
            "warnings": self.warnings,
        }


def write_dxf(plan_or_layers, path: str | Path, label_plots: bool = True,
              dxf_version: str = "R2010") -> CadResult:
    """Write the plan as a layered DXF in site coordinates."""
    layers = (plan_or_layers if isinstance(plan_or_layers, PlanLayers)
              else build_layers(plan_or_layers))
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)

    doc = ezdxf.new(dxf_version, setup=True)
    doc.header["$INSUNITS"] = 6          # metres, so a CAD user gets real units
    doc.header["$MEASUREMENT"] = 1       # metric
    msp = doc.modelspace()

    result = CadResult(path=path)

    for name, gdf in layers.non_empty().items():
        layer = doc.layers.add(name)
        layer.color = LAYER_COLOUR.get(name, 7)
        if name in FROZEN:
            layer.off = True
        result.layers += 1

        for geom in gdf.geometry:
            result.entities += _draw(msp, geom, name)

    # Plot numbers are the whole point of a cadastral drawing.
    if label_plots and "parcels" in layers.layers:
        doc.layers.add("parcel_numbers").color = 3
        parcels = layers.layers["parcels"]
        for plot_no, geom in zip(parcels["plot_no"], parcels.geometry):
            point = geom.representative_point()
            msp.add_text(
                str(plot_no),
                dxfattribs={"layer": "parcel_numbers", "height": 3.0},
            ).set_placement((point.x, point.y),
                            align=TextEntityAlignment.MIDDLE_CENTER)
            result.labels += 1

    # A DXF cannot carry a CRS, so it is stated on the drawing.
    doc.layers.add("drawing_notes").color = 7
    minx, miny, maxx, maxy = _bounds(layers)
    notes = [
        f"{layers.name}",
        f"Coordinate reference system: {layers.crs}",
        "Units: metres. Coordinates are full projected values, not shifted.",
        "Generated plan - not a surveyed cadastral diagram.",
    ]
    for i, text in enumerate(notes):
        msp.add_text(
            text, dxfattribs={"layer": "drawing_notes", "height": 8.0},
        ).set_placement((minx, miny - 30 - i * 14),
                        align=TextEntityAlignment.LEFT)

    doc.saveas(path)

    result.warnings.append(
        f"Coordinates are in {layers.crs}, in metres. Do not re-project this "
        f"drawing on import: a DXF carries no reference system, so the "
        f"receiving software will apply whatever it is told.")
    return result


# ---------------------------------------------------------------------------

def _draw(msp, geom, layer: str) -> int:
    """Draw one geometry, returning the number of entities added."""
    if geom is None or geom.is_empty:
        return 0

    kind = geom.geom_type
    if kind == "Point":
        msp.add_circle((geom.x, geom.y), radius=3.0, dxfattribs={"layer": layer})
        return 1
    if kind == "LineString":
        msp.add_lwpolyline(list(geom.coords), dxfattribs={"layer": layer})
        return 1
    if kind == "Polygon":
        count = 0
        msp.add_lwpolyline(list(geom.exterior.coords), close=True,
                           dxfattribs={"layer": layer})
        count += 1
        # Holes are drawn as their own closed polylines. A plan with a hole cut
        # for a wetland has to show the wetland, not a filled polygon over it.
        for ring in geom.interiors:
            msp.add_lwpolyline(list(ring.coords), close=True,
                               dxfattribs={"layer": layer})
            count += 1
        return count
    if kind.startswith("Multi") or kind == "GeometryCollection":
        return sum(_draw(msp, part, layer) for part in geom.geoms)
    return 0


def _bounds(layers: PlanLayers):
    boundary = layers.layers.get("site_boundary")
    if boundary is not None and len(boundary):
        return boundary.total_bounds
    for gdf in layers.non_empty().values():
        return gdf.total_bounds
    return (0.0, 0.0, 0.0, 0.0)
