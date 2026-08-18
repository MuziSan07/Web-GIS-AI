"""GIS exports: GeoPackage, shapefile, file geodatabase.

Three formats because clients have three kinds of desk. GeoPackage is what
should be used - one file, one CRS, long field names, real types, no size
limit. Shapefile is what most people ask for, and it is a 1998 format wearing
five files per layer with a ten character field name limit and no way to store a
null. File geodatabase is what an ArcGIS shop wants, and GDAL now writes it.

The differences are not stylistic and the caller is told about them rather than
left to find out: a shapefile export reports exactly which of its limitations
were hit on this plan, instead of truncating a field name silently and letting
the mismatch surface downstream.
"""

from __future__ import annotations

import shutil
from dataclasses import dataclass, field
from pathlib import Path

from .layers import LAYER_PURPOSE, PlanLayers, build_layers

# Shapefiles hold one geometry type per file and cannot mix. Anything already
# split by layer here is fine, but the limit is worth stating.
SHAPEFILE_FIELD_LIMIT = 10


@dataclass
class ExportResult:
    format: str
    path: Path
    layers: int = 0
    features: int = 0
    files: list[str] = field(default_factory=list)
    warnings: list[str] = field(default_factory=list)

    def summary(self) -> dict:
        return {
            "format": self.format,
            "path": str(self.path),
            "layers": self.layers,
            "features": self.features,
            "files": self.files,
            "warnings": self.warnings,
        }


def write_geopackage(plan_or_layers, path: str | Path) -> ExportResult:
    """One file, every layer, full field names and types.

    The format to hand over unless the recipient has said otherwise.
    """
    layers = _as_layers(plan_or_layers)
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.exists():
        path.unlink()

    result = ExportResult(format="GPKG", path=path)
    for name, gdf in layers.non_empty().items():
        gdf.to_file(path, layer=name, driver="GPKG")
        result.layers += 1
        result.features += len(gdf)
    result.files = [path.name]
    return result


def write_shapefile(plan_or_layers, directory: str | Path) -> ExportResult:
    """One shapefile set per layer, in a directory.

    Reports what the format cost rather than truncating in silence.
    """
    layers = _as_layers(plan_or_layers)
    directory = Path(directory)
    if directory.exists():
        shutil.rmtree(directory)
    directory.mkdir(parents=True, exist_ok=True)

    result = ExportResult(format="ESRI Shapefile", path=directory)
    truncated: set[str] = set()

    for name, gdf in layers.non_empty().items():
        for column in gdf.columns:
            if column != "geometry" and len(column) > SHAPEFILE_FIELD_LIMIT:
                truncated.add(column)
        target = directory / f"{name}.shp"
        gdf.to_file(target, driver="ESRI Shapefile")
        result.layers += 1
        result.features += len(gdf)
        result.files.extend(
            f"{name}{ext}" for ext in (".shp", ".shx", ".dbf", ".prj"))

    if truncated:
        result.warnings.append(
            f"Shapefile truncates field names to {SHAPEFILE_FIELD_LIMIT} "
            f"characters: {', '.join(sorted(truncated))}.")
    result.warnings.append(
        "Shapefile cannot store a null value, so an empty number reads as "
        "zero, and it has no way to record what a layer is for. The GeoPackage "
        "carries both.")
    return result


def write_filegdb(plan_or_layers, path: str | Path) -> ExportResult:
    """An Esri file geodatabase, for a client working in ArcGIS Pro."""
    layers = _as_layers(plan_or_layers)
    path = Path(path)
    if path.exists():
        shutil.rmtree(path)
    path.parent.mkdir(parents=True, exist_ok=True)

    result = ExportResult(format="OpenFileGDB", path=path)
    for name, gdf in layers.non_empty().items():
        gdf.to_file(path, layer=name, driver="OpenFileGDB")
        result.layers += 1
        result.features += len(gdf)
    result.files = [path.name]
    result.warnings.append(
        "Written with GDAL's OpenFileGDB driver, which produces a geodatabase "
        "ArcGIS Pro opens directly. It does not write Esri domains, subtypes or "
        "relationship classes; those are set up in ArcGIS against the imported "
        "feature classes.")
    return result


def _as_layers(plan_or_layers) -> PlanLayers:
    return (plan_or_layers if isinstance(plan_or_layers, PlanLayers)
            else build_layers(plan_or_layers))


def layer_manifest(layers: PlanLayers) -> list[dict]:
    """What is in the delivery, for a README or a data dictionary."""
    return [
        {
            "layer": name,
            "features": len(gdf),
            "geometry": gdf.geom_type.iloc[0] if len(gdf) else None,
            "fields": [c for c in gdf.columns if c != "geometry"],
            "purpose": LAYER_PURPOSE.get(name, ""),
        }
        for name, gdf in layers.non_empty().items()
    ]
