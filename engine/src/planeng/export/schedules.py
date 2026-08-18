"""Schedules: the tables a plan is actually approved and registered from.

A drawing shows where things are. A schedule says what they are, and it is the
schedule that gets checked line by line at approval, priced by a quantity
surveyor and lodged with the deeds registry.

The beacon coordinate schedule is the one that matters most and the one most
easily got wrong. A plot is registered by the coordinates of its corners, not by
its picture, so the schedule has to list every beacon of every plot in the site
projection with the projection named on the sheet. It is also where this engine
has to be honest about what it is: these are computed corner positions, not
surveyed beacons. A registrable diagram comes from a licensed surveyor pegging
the ground, and saying so on the schedule is the difference between a useful
working document and a misleading one.
"""

from __future__ import annotations

import csv
from dataclasses import dataclass, field
from pathlib import Path

from .layers import PlanLayers, build_layers


@dataclass
class ScheduleResult:
    directory: Path
    files: list[str] = field(default_factory=list)
    rows: dict[str, int] = field(default_factory=dict)
    warnings: list[str] = field(default_factory=list)

    def summary(self) -> dict:
        return {
            "directory": str(self.directory),
            "files": self.files,
            "rows": self.rows,
            "warnings": self.warnings,
        }


def write_schedules(plan, directory: str | Path) -> ScheduleResult:
    """Write every schedule as CSV, one file each."""
    directory = Path(directory)
    directory.mkdir(parents=True, exist_ok=True)
    layers = build_layers(plan)
    result = ScheduleResult(directory=directory)

    _write(result, directory / "plot_schedule.csv", plot_schedule(plan))
    _write(result, directory / "beacon_coordinates.csv",
           beacon_schedule(layers))
    _write(result, directory / "facility_schedule.csv", facility_schedule(plan))
    _write(result, directory / "land_budget.csv", land_budget(plan))
    _write(result, directory / "services_schedule.csv", services_schedule(plan))

    result.warnings.append(
        "Beacon coordinates are computed plot corners, not surveyed beacons. "
        "A diagram for lodgement with the Deeds Registry must be prepared and "
        "signed by a licensed land surveyor who has pegged the ground.")
    return result


def plot_schedule(plan) -> list[dict]:
    """Every plot with its area, frontage, band and development control."""
    controls = {c.plot_number: c for c in plan.controls.controls}
    rows = []
    for p in plan.cadastre.parcels:
        c = controls.get(p.plot_number)
        rows.append({
            "plot_number": p.plot_number,
            "block_id": p.block_id,
            "land_use": p.use,
            "density_band": p.band,
            "area_m2": round(p.area_m2, 1),
            "frontage_m": round(p.frontage_m, 2),
            "max_footprint_m2": round(c.max_footprint_m2, 1) if c else "",
            "max_floor_area_m2": round(c.max_floor_area_m2, 1) if c else "",
            "max_storeys": c.max_storeys if c else "",
            "compliant": (c.compliant if c else ""),
            "violations": "; ".join(c.violations) if c else "",
        })
    return rows


def beacon_schedule(plan_or_layers) -> list[dict]:
    """Corner coordinates of every plot, in the site projection.

    Beacons are numbered clockwise from the north-westmost corner, which is the
    convention a surveyor expects and, more practically, makes the schedule
    reproducible: a ring can start anywhere and wind either way, and two exports
    of the same plot that number its corners differently are impossible to check
    against each other.
    """
    layers = (plan_or_layers if isinstance(plan_or_layers, PlanLayers)
              else build_layers(plan_or_layers))
    parcels = layers.layers.get("parcels")
    if parcels is None:
        return []

    rows = []
    for plot_no, geom in zip(parcels["plot_no"], parcels.geometry):
        if geom is None or geom.is_empty:
            continue
        poly = geom if geom.geom_type == "Polygon" else max(
            geom.geoms, key=lambda g: g.area)
        # Drop the closing duplicate, then normalise start point and direction.
        ring = list(poly.exterior.coords)[:-1]
        if not ring:
            continue
        if _is_counterclockwise(ring):
            ring = ring[::-1]
        start = min(range(len(ring)), key=lambda i: (-ring[i][1], ring[i][0]))
        ring = ring[start:] + ring[:start]

        for i, (x, y) in enumerate(ring, start=1):
            rows.append({
                "plot_number": plot_no,
                "beacon": f"{plot_no}-{i:02d}",
                "easting_m": round(x, 3),
                "northing_m": round(y, 3),
                "crs": layers.crs,
            })
    return rows


def facility_schedule(plan) -> list[dict]:
    rows = []
    for f in plan.facility_plan.facilities:
        rows.append({
            "facility_id": f.facility_id,
            "type": f.key,
            "category": f.category,
            "area_ha": round(f.area_ha, 3),
            "catchment_m": f.catchment_m,
            "coverage_pct": plan.facility_plan.coverage.get(f.key, ""),
        })
    for key, count in sorted(plan.facility_plan.unplaced.items()):
        rows.append({"facility_id": "", "type": key, "category": "",
                     "area_ha": "", "catchment_m": "",
                     "coverage_pct": f"NOT PLACED ({count})"})
    return rows


def land_budget(plan) -> list[dict]:
    rows = []
    for line in plan.allocation.summary():
        rows.append({
            "land_use": line["use"],
            "target_ha": line["target_ha"],
            "achieved_ha": line["achieved_ha"],
            "variance_ha": line["variance_ha"],
            "variance_pct": line["variance_pct"],
        })
    net = plan.network
    rows.append({"land_use": "road_reserve", "target_ha": "",
                 "achieved_ha": round(net.reserve_area_ha, 2),
                 "variance_ha": "", "variance_pct": ""})
    works = plan.infrastructure.works
    rows.append({"land_use": "engineering_works",
                 "target_ha": round(works.utilities_budget_ha, 2),
                 "achieved_ha": round(works.total_area_ha, 2),
                 "variance_ha": round(works.total_area_ha
                                      - works.utilities_budget_ha, 2),
                 "variance_pct": ""})
    return rows


def services_schedule(plan) -> list[dict]:
    """Quantities per service, which is what a cost plan is built from."""
    rows = []
    infra = plan.infrastructure
    for net in infra.networks:
        by_size = net.length_by_size()
        for size, length in sorted(by_size.items(), reverse=True):
            rows.append({
                "service": net.service,
                "size_mm": int(size) if size else "",
                "length_m": round(length, 1),
                "unit": "m",
            })
    rows.append({"service": "sewer", "size_mm": "", "unit": "no",
                 "length_m": len(infra.sewer.pump_stations),
                 })
    rows.append({"service": "sewer_manholes", "size_mm": "", "unit": "no",
                 "length_m": infra.sewer.metrics.get("manholes", 0)})
    rows.append({"service": "street_lighting", "size_mm": "", "unit": "no",
                 "length_m": infra.lighting.total_poles})
    rows.append({"service": "stormwater_culverts", "size_mm": "", "unit": "no",
                 "length_m": len(infra.stormwater.culverts)})
    return rows


# ---------------------------------------------------------------------------

def _write(result: ScheduleResult, path: Path, rows: list[dict]) -> None:
    if not rows:
        return
    with path.open("w", newline="", encoding="utf-8") as fh:
        writer = csv.DictWriter(fh, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)
    result.files.append(path.name)
    result.rows[path.stem] = len(rows)


def _is_counterclockwise(ring) -> bool:
    """Shoelace sign. Positive area means counterclockwise."""
    total = 0.0
    for (x1, y1), (x2, y2) in zip(ring, ring[1:] + ring[:1]):
        total += x1 * y2 - x2 * y1
    return total > 0
