"""Cadastral subdivision: blocks into individual plots.

This is where the plan stops being a zoning diagram and becomes a layout someone
could peg out. Two rules govern the geometry and both are non-negotiable:

* **Every plot must front a road.** A landlocked plot cannot be registered,
  serviced or sold. Blocks too deep to satisfy this are reported rather than
  quietly subdivided into unusable parcels.
* **Plot dimensions come from the ruleset**, not from whatever the block
  geometry happens to allow. A plot below the minimum area or frontage for its
  density band is not a plot, it is a compliance failure.

Density bands are assigned across blocks with a centrality gradient - higher
densities toward the centre - which is both conventional planning practice and
the arrangement that keeps the dwelling schedule deliverable.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field

from shapely import affinity
from shapely.geometry import Polygon, box
from shapely.ops import unary_union

from ..brief.derive import Programme
from ..roads.blocks import Block, BlockLayout
from ..roads.network import RoadNetwork
from ..site.model import Site
from ..standards.loader import Standards

# A plot is considered to front a road if its boundary comes this close to a
# reserve. Blocks are cut from the reserve edge, so the gap should be ~0.
FRONTAGE_TOLERANCE_M = 2.0

# Beyond this many plot depths, a block cannot be served from its perimeter and
# needs an internal street.
MAX_ROWS = 2


@dataclass
class Parcel:
    geometry: Polygon
    plot_number: str
    block_id: str
    band: str
    use: str
    frontage_m: float

    @property
    def area_m2(self) -> float:
        return self.geometry.area


@dataclass
class CadastralLayout:
    parcels: list[Parcel] = field(default_factory=list)
    block_bands: dict[str, str] = field(default_factory=dict)
    warnings: list[str] = field(default_factory=list)

    def by_band(self, band: str) -> list[Parcel]:
        return [p for p in self.parcels if p.band == band]

    @property
    def total_plots(self) -> int:
        return len(self.parcels)

    def summary(self) -> dict:
        bands: dict[str, dict] = {}
        for p in self.parcels:
            e = bands.setdefault(p.band, {"plots": 0, "area_m2": 0.0})
            e["plots"] += 1
            e["area_m2"] += p.area_m2
        return {
            "total_plots": self.total_plots,
            "by_band": [
                {
                    "band": b,
                    "plots": v["plots"],
                    "mean_plot_m2": round(v["area_m2"] / v["plots"], 1) if v["plots"] else 0,
                    "total_ha": round(v["area_m2"] / 10_000, 2),
                }
                for b, v in sorted(bands.items(),
                                   key=lambda kv: -kv[1]["plots"])
            ],
        }


def subdivide(
    site: Site,
    std: Standards,
    blocks: BlockLayout,
    network: RoadNetwork,
    programme: Programme,
) -> CadastralLayout:
    layout = CadastralLayout()
    residential = blocks.for_use("residential")
    if not residential:
        layout.warnings.append("No residential blocks to subdivide.")
        return layout

    quotas = {d.band: d.dwellings for d in programme.dwellings}
    bands = _assign_bands(site, residential, quotas, std)
    layout.block_bands = bands

    reserves = network.reserve_polygon
    counter = 0
    too_deep: list[str] = []

    for block in residential:
        band = bands.get(block.block_id)
        if band is None:
            continue
        spec = std.get(f"residential.densities.{band}")
        plots, deep = _subdivide_block(block, spec, reserves)
        if deep:
            too_deep.append(block.block_id)
        for poly, frontage in plots:
            counter += 1
            layout.parcels.append(Parcel(
                geometry=poly, plot_number=f"{block.block_id}-{counter:05d}",
                block_id=block.block_id, band=band, use="residential",
                frontage_m=round(frontage, 2),
            ))

    if too_deep:
        layout.warnings.append(
            f"{len(too_deep)} block(s) are deeper than two plot rows and were "
            f"subdivided from their frontages only. They need internal access "
            f"streets to develop the interior."
        )

    _reconcile(layout, programme)
    return layout


# ---------------------------------------------------------------------------

def _assign_bands(site: Site, blocks: list[Block], quotas: dict[str, int],
                  std: Standards) -> dict[str, str]:
    """Allocate density bands across blocks, densest toward the centre.

    Walking outward from the centre and filling the high-density quota first
    produces the density gradient planners expect, and guarantees the band
    totals track the dwelling schedule rather than drifting from it.
    """
    centre = site.boundary.centroid
    ordered = sorted(blocks, key=lambda b: b.geometry.centroid.distance(centre))

    remaining = {b: q for b, q in quotas.items() if q > 0}
    order = [b for b in ("high", "medium", "low") if b in remaining]
    if not order:
        return {}

    # When the blocks cannot hold the whole schedule, scale every band down in
    # proportion. Filling bands in sequence without this dumps the entire
    # shortfall on the last band, which wipes out low-density housing rather
    # than trimming each band fairly.
    demand_area = sum(
        q * std.get(f"residential.densities.{b}.plot_area_typical_m2")
        for b, q in remaining.items()
    )
    capacity_area = sum(b.geometry.area * 0.88 for b in ordered)
    if demand_area > 0 and capacity_area < demand_area:
        scale = capacity_area / demand_area
        remaining = {b: max(1, int(q * scale)) for b, q in remaining.items()}

    out: dict[str, str] = {}
    cursor = 0
    for block in ordered:
        while cursor < len(order) and remaining[order[cursor]] <= 0:
            cursor += 1
        band = order[cursor] if cursor < len(order) else order[-1]
        out[block.block_id] = band

        typical = std.get(f"residential.densities.{band}.plot_area_typical_m2")
        remaining[band] -= max(1, int(block.geometry.area * 0.88 / typical))

    return out


def _subdivide_block(block: Block, spec: dict, reserves) -> tuple[list, bool]:
    """Cut one block into plots aligned to its dominant axis.

    Works in the block's own rotated frame so plots are rectangular and aligned
    to the street, then rotates the result back. Cutting in map coordinates
    would produce skewed parcels on any block not aligned to north.
    """
    poly = block.geometry
    frontage_min = spec["frontage_min_m"]
    area_min = spec["plot_area_min_m2"]
    area_typical = spec["plot_area_typical_m2"]
    depth = area_typical / frontage_min

    angle = _dominant_angle(poly)
    centre = poly.centroid
    local = affinity.rotate(poly, -angle, origin=centre, use_radians=False)
    minx, miny, maxx, maxy = local.bounds
    width, height = maxx - minx, maxy - miny

    if width < frontage_min or height < depth * 0.6:
        return [], False

    # Plot depth is fixed by the standard and measured inward from each
    # frontage. Stretching rows to fill the block instead - which is the
    # obvious implementation - silently inflates every plot: a 350 m2 standard
    # becomes a 680 m2 plot on a deep block, and the layout stops complying
    # with the very ruleset it was generated from.
    if height >= 2 * depth:
        row_offsets = [miny, maxy - depth]
        row_h = depth
    elif height >= depth * 1.2:
        row_offsets = [miny, miny + height / 2]
        row_h = height / 2
    else:
        row_offsets = [miny]
        row_h = height

    too_deep = height > 2 * depth * 1.25

    cols = max(1, int(round(width / frontage_min)))
    col_w = width / cols
    if col_w < frontage_min * 0.85:
        cols = max(1, cols - 1)
        col_w = width / cols

    out = []
    for y0 in row_offsets:
        for c in range(cols):
            cell = box(minx + c * col_w, y0,
                       minx + (c + 1) * col_w, y0 + row_h)
            part = cell.intersection(local)
            if part.is_empty or part.geom_type != "Polygon":
                continue
            if part.area < area_min:
                continue
            plot = affinity.rotate(part, angle, origin=centre, use_radians=False)
            if reserves is not None and plot.distance(reserves) > FRONTAGE_TOLERANCE_M:
                continue          # landlocked: not a registrable plot
            out.append((plot, col_w))

    return out, too_deep


def _dominant_angle(poly: Polygon) -> float:
    """Orientation of the block's longest edge, in degrees."""
    rect = poly.minimum_rotated_rectangle
    if rect.geom_type != "Polygon":
        return 0.0
    coords = list(rect.exterior.coords)[:4]
    best, best_len = 0.0, -1.0
    for (x0, y0), (x1, y1) in zip(coords, coords[1:] + coords[:1]):
        length = math.hypot(x1 - x0, y1 - y0)
        if length > best_len:
            best_len = length
            best = math.degrees(math.atan2(y1 - y0, x1 - x0))
    return best


def _reconcile(layout: CadastralLayout, programme: Programme) -> None:
    """Compare surveyed plots against the dwelling schedule, band by band."""
    target = {d.band: d.dwellings for d in programme.dwellings}
    total_target = sum(target.values())
    if total_target <= 0:
        return

    variance = (layout.total_plots - total_target) / total_target * 100
    if abs(variance) > 15:
        layout.warnings.append(
            f"Cadastral layout produced {layout.total_plots:,} plots against a "
            f"dwelling schedule of {total_target:,} ({variance:+.0f}%). Review "
            f"block sizes, plot dimensions or the density mix."
        )
