"""The delivery: every format, plus the note that explains them.

A client receives one archive, not eight downloads, and the archive contains a
README because a folder of .gpkg, .dxf and .ifc files with no explanation is a
support ticket. The README states the coordinate reference system once and
loudly - it is the single most common cause of a delivery being unusable - along
with the status of the ruleset the plan was generated from.
"""

from __future__ import annotations

import json
import zipfile
from dataclasses import dataclass, field
from pathlib import Path

from .bim import write_ifc
from .cad import write_dxf
from .layers import build_layers
from .plans import write_plan_set
from .schedules import write_schedules
from .vector import (layer_manifest, write_filegdb, write_geopackage,
                     write_shapefile)

# Every format, and whether it is on by default. Shapefile and file geodatabase
# are off unless asked for: they are large, lossy and duplicate the GeoPackage,
# and a delivery is better for not containing three copies of the same data.
FORMATS = {
    "geopackage": True,
    "plan_set": True,
    "dxf": True,
    "schedules": True,
    "ifc": True,
    "shapefile": False,
    "filegdb": False,
}

# Full IFC for three thousand plots is a large file and slow to open. The cap
# is a default, not a limit.
IFC_PLOT_CAP = 500


@dataclass
class BundleResult:
    directory: Path
    archive: Path | None = None
    outputs: list[dict] = field(default_factory=list)
    warnings: list[str] = field(default_factory=list)

    def summary(self) -> dict:
        return {
            "directory": str(self.directory),
            "archive": str(self.archive) if self.archive else None,
            "outputs": self.outputs,
            "warnings": self.warnings,
        }


def export_all(plan, directory: str | Path, formats: dict | None = None,
               archive: bool = True, ifc_plot_cap: int | None = IFC_PLOT_CAP
               ) -> BundleResult:
    """Write every requested format into one directory, and zip it."""
    wanted = {**FORMATS, **(formats or {})}
    directory = Path(directory)
    directory.mkdir(parents=True, exist_ok=True)

    layers = build_layers(plan)
    result = BundleResult(directory=directory)
    result.warnings.extend(layers.warnings)

    def record(res) -> None:
        summary = res.summary()
        result.outputs.append(summary)
        result.warnings.extend(summary.get("warnings", []))

    if wanted.get("geopackage"):
        record(write_geopackage(layers, directory / "plan.gpkg"))
    if wanted.get("shapefile"):
        record(write_shapefile(layers, directory / "shapefile"))
    if wanted.get("filegdb"):
        record(write_filegdb(layers, directory / "plan.gdb"))
    if wanted.get("dxf"):
        record(write_dxf(layers, directory / "plan.dxf"))
    if wanted.get("plan_set"):
        record(write_plan_set(plan, directory / "plan_set.pdf"))
    if wanted.get("schedules"):
        record(write_schedules(plan, directory / "schedules"))
    if wanted.get("ifc"):
        record(write_ifc(plan, directory / "plan_model.ifc",
                         max_plots=ifc_plot_cap))

    (directory / "layer_manifest.json").write_text(
        json.dumps(layer_manifest(layers), indent=2), encoding="utf-8")
    (directory / "README.txt").write_text(
        _readme(plan, layers, result), encoding="utf-8")

    if archive:
        result.archive = _zip(directory)
    return result


def _readme(plan, layers, result: BundleResult) -> str:
    std = plan.standards
    lines = [
        f"{plan.brief.project_name} - development plan data",
        "=" * 60,
        "",
        "COORDINATE REFERENCE SYSTEM",
        f"  {layers.crs} - all coordinates are metres on this projection.",
        "  The DXF and IFC files carry no reference system of their own. Do",
        "  not re-project them on import; set the receiving drawing to the",
        "  CRS above instead.",
        "",
        "STATUS",
    ]
    if std.is_validated:
        lines.append(f"  Ruleset {std.data['meta']['jurisdiction']} "
                     f"v{std.data['meta']['ruleset_version']}, validated.")
    else:
        lines += [
            "  NOT FOR STATUTORY USE.",
            f"  Generated from ruleset {std.data['meta']['jurisdiction']} "
            f"v{std.data['meta']['ruleset_version']}, which is placeholder",
            "  data. The standards behind every dimension in this delivery",
            "  have not been confirmed against Botswana planning legislation",
            "  by a registered planner.",
        ]
    lines += [
        "  Plot corners are computed, not surveyed. A diagram for lodgement",
        "  with the Deeds Registry must be prepared by a licensed surveyor.",
        "",
        "CONTENTS",
    ]
    for out in result.outputs:
        name = Path(out.get("path", out.get("directory", ""))).name
        lines.append(f"  {name:<24} {out.get('format', 'CSV schedules')}")
    lines += [
        "  layer_manifest.json      Field-by-field data dictionary",
        "",
        "LAYERS",
    ]
    for row in layers.summary():
        if row["features"]:
            lines.append(f"  {row['layer']:<22} {row['features']:>6}  "
                         f"{row['purpose']}")

    if result.warnings:
        lines += ["", "NOTES"]
        lines.extend(f"  - {w}" for w in dict.fromkeys(result.warnings))
    return "\n".join(lines) + "\n"


def _zip(directory: Path) -> Path:
    archive = directory.with_suffix(".zip")
    if archive.exists():
        archive.unlink()
    with zipfile.ZipFile(archive, "w", zipfile.ZIP_DEFLATED) as zf:
        for path in sorted(directory.rglob("*")):
            if path.is_file():
                zf.write(path, path.relative_to(directory))
    return archive
