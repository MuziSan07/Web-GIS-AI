"""Development blocks.

A block is what remains of developable land once road reserves are removed. It
is the unit a cadastral layout subdivides into plots, so blocks are the bridge
between the zoning plan and the parcel plan.

Blocks carry their land use and an indicative plot yield, which lets the plan
be checked against the dwelling schedule before any parcel geometry exists.
"""

from __future__ import annotations

from dataclasses import dataclass, field

from shapely.geometry import Polygon

from ..allocate.zones import AllocationResult
from ..geom import polygonise
from ..site.model import Site
from ..standards.loader import Standards
from .network import RoadNetwork


@dataclass
class Block:
    geometry: Polygon
    use: str
    block_id: str
    indicative_plots: int = 0

    @property
    def area_ha(self) -> float:
        return self.geometry.area / 10_000


@dataclass
class BlockLayout:
    blocks: list[Block] = field(default_factory=list)
    warnings: list[str] = field(default_factory=list)

    def for_use(self, use: str) -> list[Block]:
        return [b for b in self.blocks if b.use == use]

    @property
    def total_indicative_plots(self) -> int:
        return sum(b.indicative_plots for b in self.blocks)

    def summary(self) -> dict:
        by_use: dict[str, dict] = {}
        for b in self.blocks:
            entry = by_use.setdefault(b.use, {"blocks": 0, "area_ha": 0.0, "plots": 0})
            entry["blocks"] += 1
            entry["area_ha"] += b.area_ha
            entry["plots"] += b.indicative_plots
        return {
            "block_count": len(self.blocks),
            "indicative_plots": self.total_indicative_plots,
            "by_use": [
                {"use": u, "blocks": v["blocks"],
                 "area_ha": round(v["area_ha"], 2), "indicative_plots": v["plots"]}
                for u, v in sorted(by_use.items(), key=lambda kv: -kv[1]["area_ha"])
            ],
        }


def build_blocks(
    site: Site,
    std: Standards,
    allocation: AllocationResult,
    network: RoadNetwork,
    programme=None,
    min_block_m2: float = 800.0,
) -> BlockLayout:
    layout = BlockLayout()
    reserves = network.reserve_polygon

    # Plot yield uses the area-weighted mean plot size, since blocks are not yet
    # assigned to a specific density band. It must come from the dwelling
    # schedule actually derived for this brief - falling back to the ruleset
    # default mix silently misprices every block whenever the brief overrides it.
    mean_plot_m2 = _mean_plot_area(std, programme)
    # Blocks still lose land to internal verges and irregular geometry.
    yield_efficiency = 0.88

    counter = 0
    for use in allocation.uses:
        for poly in polygonise(allocation.mask_for(use), site, min_area_m2=min_block_m2):
            remaining = poly.difference(reserves) if reserves is not None else poly
            if remaining.is_empty:
                continue
            parts = (list(remaining.geoms)
                     if remaining.geom_type == "MultiPolygon" else [remaining])
            for part in parts:
                if part.area < min_block_m2 or part.geom_type != "Polygon":
                    continue
                counter += 1
                plots = 0
                if use == "residential":
                    plots = int(part.area * yield_efficiency / mean_plot_m2)
                layout.blocks.append(Block(
                    geometry=part, use=use,
                    block_id=f"B{counter:04d}", indicative_plots=plots,
                ))

    if not layout.blocks:
        layout.warnings.append(
            "Road reserves consumed all developable land; no blocks were formed. "
            "Check block spacing and reserve widths in the ruleset."
        )
    return layout


def _mean_plot_area(std: Standards, programme) -> float:
    """Area-weighted mean plot size for this brief, not for the default mix."""
    if programme is not None and getattr(programme, "dwellings", None):
        total = sum(d.dwellings for d in programme.dwellings)
        if total > 0:
            return sum(d.dwellings * d.plot_area_typical_m2
                       for d in programme.dwellings) / total

    mix = {k: v for k, v in std.get("residential.default_density_mix").items()
           if isinstance(v, (int, float))}
    return sum(
        std.get(f"residential.densities.{band}.plot_area_typical_m2") * share
        for band, share in mix.items()
    )


def reconcile_plots(layout: BlockLayout, target_plots: int) -> str | None:
    """Compare block capacity against the dwelling schedule.

    Catches the case where the layout physically cannot deliver the plots the
    brief promises, which is otherwise only discovered at parcel stage.
    """
    actual = layout.total_indicative_plots
    if target_plots <= 0:
        return None
    variance = (actual - target_plots) / target_plots * 100
    if variance < -10:
        return (
            f"Block layout yields about {actual:,} plots against a target of "
            f"{target_plots:,} ({variance:.0f}%). The road layout or density mix "
            f"needs adjustment to deliver the brief."
        )
    if variance > 25:
        return (
            f"Block layout could hold about {actual:,} plots, well above the "
            f"{target_plots:,} required ({variance:+.0f}%). There is capacity for "
            f"lower densities, larger plots or phased expansion."
        )
    return None
