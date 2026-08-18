"""Siting the engineering works.

A reservoir, a treatment works and a landfill are not allocated the way housing
is. Housing goes where a weighted suitability surface says it should; these go
where physics and public health say they must, and a multi-criteria score that
merely prefers high ground for a reservoir is not good enough - a reservoir
below the plots it serves does not supply them at low pressure, it does not
supply them at all.

So each works is sited against a hard rule with a stated reason:

* Reservoir - the highest available ground, and it must clear the highest plot
  it serves by the minimum head. If no site clears it, that is reported as a
  pumped supply, not hidden by picking the best of a bad set.
* Treatment works - the lowest ground, so the trunk sewer can reach it by
  gravity, and beyond the odour buffer from housing.
* Landfill - beyond its own, longer buffer, on the least suitable land, and
  large enough to be licensable.
* Transfer station - only above the population threshold, near the industrial
  area rather than out at the landfill.

Works land is taken from whatever the allocator left over, then from open
space. The ruleset budgets a utilities share for exactly this, but that budget
line is never zoned, because utilities has no suitability weighting for the
allocator to work from. The shortfall is reported rather than quietly absorbed.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
from scipy import ndimage
from shapely.geometry import Polygon

from ..allocate.zones import UNASSIGNED, AllocationResult
from ..brief.derive import Programme
from ..geom import destaircase, largest_parts, polygonise
from ..roads.blocks import BlockLayout
from ..site.model import Site
from ..standards.loader import Standards


@dataclass
class ServiceSite:
    geometry: Polygon
    key: str
    label: str
    area_ha: float
    level_m: float
    notes: list[str] = field(default_factory=list)
    compliant: bool = True

    def summary(self) -> dict:
        return {
            "key": self.key,
            "label": self.label,
            "area_ha": round(self.area_ha, 2),
            "level_m": round(self.level_m, 1),
            "compliant": self.compliant,
            "notes": self.notes,
        }


@dataclass
class ServiceSitePlan:
    sites: list[ServiceSite] = field(default_factory=list)
    unplaced: dict[str, str] = field(default_factory=dict)
    warnings: list[str] = field(default_factory=list)
    utilities_budget_ha: float = 0.0

    def get(self, key: str) -> ServiceSite | None:
        return next((s for s in self.sites if s.key == key), None)

    @property
    def total_area_ha(self) -> float:
        return sum(s.area_ha for s in self.sites)

    def summary(self) -> dict:
        return {
            "sited": [s.summary() for s in self.sites],
            "unplaced": [{"key": k, "reason": v}
                         for k, v in sorted(self.unplaced.items())],
            "total_area_ha": round(self.total_area_ha, 2),
            "utilities_budget_ha": round(self.utilities_budget_ha, 2),
            "warnings": self.warnings,
        }


def site_works(
    site: Site,
    std: Standards,
    programme: Programme,
    allocation: AllocationResult,
    blocks: BlockLayout,
    constraints,
) -> ServiceSitePlan:
    plan = ServiceSitePlan()
    plan.utilities_budget_ha = next(
        (line.target_ha for line in programme.land_budget
         if line.use == "utilities"), 0.0)

    infra = programme.infrastructure
    dem = site.dem
    mask = site.mask()

    # Land the allocator did not use, which is where a works belongs: taking a
    # treatment works out of housing land is a worse trade than using the
    # leftover, and the leftover exists because the budget never zoned it.
    spare = mask & constraints.buildable & (allocation.labels == UNASSIGNED)
    open_space = (allocation.mask_for("recreation_open_space")
                  if "recreation_open_space" in allocation.uses
                  else np.zeros(site.shape, dtype=bool))

    residential = _use_mask(site, blocks, "residential")
    industrial = _use_mask(site, blocks, "industrial")
    dist_res = ndimage.distance_transform_edt(
        ~residential, sampling=site.cell_size) if residential.any() else None

    # -- reservoir --------------------------------------------------------
    storage_m3 = float(infra.get("reservoir_storage_m3") or 0.0)
    min_head = std.get("infrastructure.water.reservoir_min_head_m")
    res_area = std.get("infrastructure.water.reservoir_site_area_ha")
    highest_served = float(dem[residential].max()) if residential.any() else float(
        dem[mask].max())

    reservoir = _place(
        site, spare | open_space, res_area,
        score=np.where(mask, dem, -np.inf),      # highest ground wins
        key="reservoir", label="Service reservoir",
    )
    if reservoir is None:
        plan.unplaced["reservoir"] = (
            f"No site of {res_area} ha was available on unallocated or open "
            f"space land.")
    else:
        head = reservoir.level_m - highest_served
        reservoir.notes.append(
            f"Stores {storage_m3:,.0f} m3, {std.get('infrastructure.water.reservoir_storage_hours')} "
            f"hours at average demand.")
        if head < min_head:
            reservoir.compliant = False
            reservoir.notes.append(
                f"Only {head:.1f} m above the highest plot it serves, against a "
                f"{min_head} m minimum. Gravity supply is not achievable from "
                f"this site: an elevated tower or a booster pump station is "
                f"required, which is an operating cost the plan should carry.")
            plan.warnings.append(
                f"The reservoir site gives {head:.1f} m of head against a "
                f"{min_head} m standard. The site has {float(dem[mask].max()) - highest_served:.1f} m "
                f"of relief above the highest housing, so no location on this "
                f"land supplies the top plots by gravity.")
        else:
            reservoir.notes.append(
                f"{head:.1f} m above the highest plot served, clearing the "
                f"{min_head} m minimum head.")
        plan.sites.append(reservoir)

    # -- wastewater treatment works ---------------------------------------
    buffer_m = std.get("infrastructure.sewer.treatment_works_buffer_m")
    wwtw_ha = (std.get("infrastructure.sewer.treatment_works_area_ha_per_1000_pop")
               * float(infra.get("design_population", 0)) / 1000)
    allowed = (spare | open_space)
    if dist_res is not None:
        allowed = allowed & (dist_res >= buffer_m)

    wwtw = _place(
        site, allowed, wwtw_ha,
        score=np.where(mask, -dem, -np.inf),     # lowest ground wins
        key="wastewater_treatment", label="Wastewater treatment works",
    )
    if wwtw is None:
        plan.unplaced["wastewater_treatment"] = (
            f"No site of {wwtw_ha:.1f} ha exists that is both unallocated and "
            f"{buffer_m} m clear of housing.")
        plan.warnings.append(
            f"The wastewater treatment works ({wwtw_ha:.1f} ha at "
            f"{buffer_m} m from housing) does not fit on this site. Treatment "
            f"must be off site, or a regional works must take the flow: "
            f"{infra.get('sewage_average_day_m3', 0):,.0f} m3/day.")
    else:
        wwtw.notes.append(
            f"Treats {infra.get('sewage_average_day_m3', 0):,.0f} m3/day average "
            f"dry weather flow.")
        wwtw.notes.append(
            f"Lowest available ground at {wwtw.level_m:.1f} m, so the trunk "
            f"sewer reaches it by gravity.")
        plan.sites.append(wwtw)

    # -- landfill ----------------------------------------------------------
    lf_buffer = std.get("infrastructure.solid_waste.landfill_buffer_from_residential_m")
    lf_area = std.get("infrastructure.solid_waste.landfill_min_area_ha")
    lf_allowed = spare.copy()
    if dist_res is not None:
        lf_allowed &= dist_res >= lf_buffer

    landfill = _place(
        site, lf_allowed, lf_area,
        score=np.where(mask, dist_res if dist_res is not None else 0.0, -np.inf),
        key="landfill", label="Landfill",
    )
    if landfill is None:
        plan.unplaced["landfill"] = (
            f"No {lf_area} ha site exists {lf_buffer} m clear of housing.")
        plan.warnings.append(
            f"No landfill can be sited here: a {lf_area:.0f} ha cell with a "
            f"{lf_buffer} m buffer from housing does not fit inside a "
            f"{site.area_ha:,.0f} ha settlement. "
            f"{infra.get('solid_waste_tonnes_per_day', 0):,.1f} tonnes/day must "
            f"go to a regional facility, and the plan needs a transfer station "
            f"and a haulage cost rather than a landfill.")
    else:
        landfill.notes.append(
            f"Receives {infra.get('solid_waste_tonnes_per_day', 0):,.1f} "
            f"tonnes/day, {infra.get('solid_waste_tonnes_per_year', 0):,.0f} "
            f"tonnes/year.")
        plan.sites.append(landfill)

    # -- refuse transfer station ------------------------------------------
    threshold = std.get(
        "infrastructure.solid_waste.transfer_station_threshold_population")
    pop = float(infra.get("design_population", 0))
    if pop >= threshold:
        near_industry = (ndimage.distance_transform_edt(
            ~industrial, sampling=site.cell_size)
            if industrial.any() else np.zeros(site.shape))
        ts = _place(
            site, spare | open_space, 0.5,
            score=np.where(mask, -near_industry, -np.inf),
            key="transfer_station", label="Refuse transfer station",
        )
        if ts is None:
            plan.unplaced["transfer_station"] = "No 0.5 ha site was available."
        else:
            ts.notes.append(
                f"Required above {threshold:,.0f} population; design population "
                f"is {pop:,.0f}.")
            if "landfill" in plan.unplaced:
                ts.notes.append(
                    "Carries the whole waste stream, because no landfill could "
                    "be sited on this land.")
            plan.sites.append(ts)

    # -- did the works fit the budget the ruleset set aside? ---------------
    if plan.utilities_budget_ha and plan.total_area_ha > plan.utilities_budget_ha:
        plan.warnings.append(
            f"Engineering works take {plan.total_area_ha:.1f} ha against a "
            f"{plan.utilities_budget_ha:.1f} ha utilities budget. The allocator "
            f"never zones that budget - utilities has no suitability weighting - "
            f"so the works are standing on unallocated and open space land.")

    return plan


# ---------------------------------------------------------------------------

def _use_mask(site: Site, blocks: BlockLayout, use: str) -> np.ndarray:
    """Raster mask of the blocks of one land use."""
    from shapely.ops import unary_union
    geoms = [b.geometry for b in blocks.for_use(use)]
    if not geoms:
        return np.zeros(site.shape, dtype=bool)
    return site.rasterize(unary_union(geoms))


def _place(site: Site, allowed: np.ndarray, area_ha: float,
           score: np.ndarray, key: str, label: str) -> ServiceSite | None:
    """Grow a compact site of the required area from the best allowed cell.

    Region growing rather than picking the best N cells outright: a works needs
    one contiguous piece of land it can be fenced and licensed on, and the
    highest-scoring cells on a site are usually scattered across every ridge.
    """
    need = max(site.ha_to_cells(area_ha), 1)
    if allowed.sum() < need:
        return None

    field_ = np.where(allowed, score, -np.inf)
    if not np.isfinite(field_).any():
        return None

    # Seed at the best cell, then take the best-scoring cells adjacent to what
    # has been taken so far.
    seed = np.unravel_index(int(np.argmax(field_)), site.shape)
    taken = np.zeros(site.shape, dtype=bool)
    taken[seed] = True
    structure = np.ones((3, 3), dtype=bool)

    target_m2 = area_ha * 10_000
    geom = None
    exhausted = False

    # Grow until the *polygon* meets the area, not until the cell count does.
    # Smoothing the raster boundary loses roughly a cell around the perimeter,
    # which on a site this small is a third of it - a reservoir platform cut
    # thirty percent under standard is not a reservoir platform.
    while True:
        while taken.sum() < need:
            frontier = ndimage.binary_dilation(taken, structure) & allowed & ~taken
            if not frontier.any():
                exhausted = True
                break
            cand = np.where(frontier, field_, -np.inf)
            best = np.unravel_index(int(np.argmax(cand)), site.shape)
            if not np.isfinite(cand[best]):
                exhausted = True
                break
            taken[best] = True

        polys = largest_parts(_to_polygon(taken, site), 100.0)
        geom = max(polys, key=lambda g: g.area) if polys else None
        if geom is not None and geom.area >= target_m2:
            break
        if exhausted:
            break
        # Ask for the shortfall in cells, plus a margin for the next round of
        # perimeter loss.
        short = target_m2 - (geom.area if geom is not None else 0.0)
        need += max(int(short / site.cell_area_m2 * 1.15), 1)
        if need > allowed.sum():
            need = int(allowed.sum())
            exhausted = True

    if geom is None or geom.area < target_m2 * 0.9:
        return None
    return ServiceSite(
        geometry=geom, key=key, label=label,
        area_ha=geom.area / 10_000,
        level_m=float(site.dem[taken].mean()),
    )


def _to_polygon(taken: np.ndarray, site: Site):
    from shapely.ops import unary_union
    parts = polygonise(taken, site)
    if not parts:
        return Polygon()
    return unary_union(destaircase(parts, site.cell_size))
