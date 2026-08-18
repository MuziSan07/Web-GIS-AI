"""Derive the quantified planning programme from a brief.

Turns "25,000 people on 600 hectares" into a dwelling schedule, a facility
schedule and a land budget, entirely from the jurisdiction ruleset. This runs
before any spatial work: it establishes whether the brief is even deliverable on
the site, which is a question worth answering in milliseconds rather than after
an hour of geoprocessing.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field

from ..standards.loader import Standards
from .models import PlanningBrief


@dataclass
class DwellingSchedule:
    band: str
    dwellings: int
    plot_area_typical_m2: float
    net_area_ha: float
    gross_area_ha: float


@dataclass
class FacilityRequirement:
    key: str
    category: str
    count: int
    site_area_ha: float
    total_area_ha: float
    catchment_m: float
    driver: str  # how the count was arrived at


@dataclass
class LandBudgetLine:
    use: str
    target_ha: float
    target_pct: float


@dataclass
class Programme:
    """The full quantified programme derived from a brief."""

    brief: PlanningBrief
    dwellings: list[DwellingSchedule] = field(default_factory=list)
    facilities: list[FacilityRequirement] = field(default_factory=list)
    land_budget: list[LandBudgetLine] = field(default_factory=list)
    infrastructure: dict[str, float] = field(default_factory=dict)
    warnings: list[str] = field(default_factory=list)

    @property
    def total_dwellings(self) -> int:
        return sum(d.dwellings for d in self.dwellings)

    @property
    def residential_gross_ha(self) -> float:
        return sum(d.gross_area_ha for d in self.dwellings)

    @property
    def facilities_ha(self) -> float:
        return sum(f.total_area_ha for f in self.facilities)

    @property
    def allocated_ha(self) -> float:
        return sum(line.target_ha for line in self.land_budget)

    def facility(self, key: str) -> FacilityRequirement | None:
        return next((f for f in self.facilities if f.key == key), None)


def derive_programme(brief: PlanningBrief, std: Standards) -> Programme:
    prog = Programme(brief=brief)

    if (warning := std.provenance_warning()) is not None:
        prog.warnings.append(warning)

    _derive_dwellings(brief, std, prog)
    _derive_facilities(brief, std, prog)
    _derive_infrastructure(brief, std, prog)
    _derive_land_budget(brief, std, prog)
    _check_feasibility(brief, prog)

    return prog


# ---------------------------------------------------------------------------

def _derive_dwellings(brief: PlanningBrief, std: Standards, prog: Programme) -> None:
    pph = std.get("demographics.persons_per_household")
    allowance = std.get("demographics.household_growth_allowance")
    net_to_gross = std.get("residential.net_to_gross_factor")

    if brief.target_residential_plots is not None:
        total_dwellings = brief.target_residential_plots
        housed = total_dwellings * pph / allowance
        if housed < brief.target_population * 0.9:
            prog.warnings.append(
                f"Brief requests {total_dwellings:,} plots, which houses about "
                f"{housed:,.0f} people at {pph} persons/household - short of the "
                f"{brief.target_population:,} target. Plot count governs."
            )
    else:
        households = brief.target_population / pph
        total_dwellings = int(math.ceil(households * allowance))

    mix = (brief.density_mix.as_dict() if brief.density_mix
           else {k: v for k, v in std.get("residential.default_density_mix").items()
                 if isinstance(v, (int, float))})

    # Apportion by mix, then push any rounding remainder into the largest band so
    # the schedule always totals exactly.
    counts = {band: int(total_dwellings * share) for band, share in mix.items()}
    remainder = total_dwellings - sum(counts.values())
    if remainder:
        counts[max(counts, key=lambda b: counts[b])] += remainder

    for band, n in counts.items():
        if n <= 0:
            continue
        typical = std.get(f"residential.densities.{band}.plot_area_typical_m2")
        net_ha = n * typical / 10_000
        prog.dwellings.append(DwellingSchedule(
            band=band,
            dwellings=n,
            plot_area_typical_m2=typical,
            net_area_ha=round(net_ha, 2),
            gross_area_ha=round(net_ha / net_to_gross, 2),
        ))

    prog.dwellings.sort(key=lambda d: ["low", "medium", "high"].index(d.band))


def _derive_facilities(brief: PlanningBrief, std: Standards, prog: Programme) -> None:
    pop = brief.design_population
    facilities = std.get("facilities")

    for category, entries in facilities.items():
        if not isinstance(entries, dict):
            continue
        for key, spec in entries.items():
            if not isinstance(spec, dict):
                continue

            # An explicit requirement in the brief always wins over the ratio.
            if key in brief.required_facilities:
                count = brief.required_facilities[key]
                driver = "specified in brief"
            elif "ha_per_1000_population" in spec:
                # Area-driven rather than count-driven (cemeteries, for example).
                area = spec["ha_per_1000_population"] * pop / 1000
                min_area = spec.get("min_site_area_ha", 0)
                total = max(area, min_area)
                prog.facilities.append(FacilityRequirement(
                    key=key, category=category, count=1,
                    site_area_ha=round(total, 2), total_area_ha=round(total, 2),
                    catchment_m=spec.get("catchment_m", 0),
                    driver=f"{spec['ha_per_1000_population']} ha per 1,000 people",
                ))
                continue
            else:
                threshold = spec.get("threshold_population")
                if not threshold:
                    continue
                count = max(
                    int(math.ceil(pop / threshold)),
                    spec.get("min_count", 0),
                )
                driver = f"1 per {threshold:,} people (design pop {pop:,})"

            if count <= 0:
                continue

            site = spec.get("site_area_ha", 0)
            prog.facilities.append(FacilityRequirement(
                key=key, category=category, count=count,
                site_area_ha=site, total_area_ha=round(count * site, 2),
                catchment_m=spec.get("catchment_m", 0), driver=driver,
            ))

    # Population-driven utility sites.
    sub_threshold = std.get("infrastructure.electricity.substation_threshold_population")
    sub_area = std.get("infrastructure.electricity.substation_site_area_ha")
    sub_count = max(1, int(math.ceil(pop / sub_threshold)))
    prog.facilities.append(FacilityRequirement(
        key="electrical_substation", category="utilities", count=sub_count,
        site_area_ha=sub_area, total_area_ha=round(sub_count * sub_area, 2),
        catchment_m=0, driver=f"1 per {sub_threshold:,} people",
    ))


def _derive_infrastructure(brief: PlanningBrief, std: Standards, prog: Programme) -> None:
    pop = brief.design_population
    lpcd = std.get("infrastructure.water.domestic_demand_lpcd")
    peak = std.get("infrastructure.water.peak_factor")
    storage_h = std.get("infrastructure.water.reservoir_storage_hours")
    return_factor = std.get("infrastructure.sewer.return_factor")
    admd = std.get("infrastructure.electricity.admd_kva_per_household")
    pph = std.get("demographics.persons_per_household")
    waste_kg = std.get("infrastructure.solid_waste.generation_kg_per_capita_day")

    avg_daily_m3 = pop * lpcd / 1000
    prog.infrastructure = {
        "design_population": pop,
        "water_average_day_m3": round(avg_daily_m3, 0),
        "water_peak_day_m3": round(avg_daily_m3 * peak, 0),
        "reservoir_storage_m3": round(avg_daily_m3 * storage_h / 24, 0),
        "sewage_average_day_m3": round(avg_daily_m3 * return_factor, 0),
        "electrical_demand_mva": round(prog.total_dwellings * admd / 1000, 2),
        "solid_waste_tonnes_per_day": round(pop * waste_kg / 1000, 1),
        "solid_waste_tonnes_per_year": round(pop * waste_kg * 365 / 1000, 0),
    }
    prog.infrastructure["households"] = round(pop / pph)


def _derive_land_budget(brief: PlanningBrief, std: Standards, prog: Programme) -> None:
    """Build the land budget from actual demand, not just the indicative shares.

    Residential and institutional areas come from the schedules derived above.
    The remaining uses fall back to the ruleset's indicative percentages.
    """
    gross = brief.site_area_ha
    budget = std.get("land_budget")

    residential_ha = prog.residential_gross_ha

    # Institutional land is what the facility schedule actually demands, excluding
    # recreation (budgeted separately) and utilities.
    institutional_ha = sum(
        f.total_area_ha for f in prog.facilities
        if f.category in ("education", "health", "safety", "community", "civic")
    )
    utilities_ha = sum(
        f.total_area_ha for f in prog.facilities if f.category == "utilities"
    )
    recreation_ha = sum(
        f.total_area_ha for f in prog.facilities if f.category == "recreation"
    )

    # Recreation and utilities have policy minimums as well as demand.
    recreation_ha = max(recreation_ha, gross * budget["recreation_open_space_pct"] / 100)
    utilities_ha = max(utilities_ha, gross * budget["utilities_pct"] / 100)

    commercial_ha = gross * budget["commercial_pct"] / 100
    industrial_ha = gross * budget["industrial_pct"] / 100 if brief.require_industrial else 0.0
    circulation_ha = gross * budget["circulation_pct"] / 100

    lines = [
        ("residential", residential_ha),
        ("commercial", commercial_ha),
        ("industrial", industrial_ha),
        ("institutional", institutional_ha),
        ("recreation_open_space", recreation_ha),
        ("circulation", circulation_ha),
        ("utilities", utilities_ha),
    ]

    for use, ha in lines:
        if ha <= 0:
            continue
        prog.land_budget.append(LandBudgetLine(
            use=use, target_ha=round(ha, 2), target_pct=round(ha / gross * 100, 2),
        ))


def _check_feasibility(brief: PlanningBrief, prog: Programme) -> None:
    """State plainly whether the brief fits the site."""
    gross = brief.site_area_ha
    allocated = prog.allocated_ha
    balance = gross - allocated

    if balance < 0:
        prog.warnings.append(
            f"BRIEF DOES NOT FIT: the programme requires {allocated:,.1f} ha but "
            f"the site is {gross:,.1f} ha - an overrun of {abs(balance):,.1f} ha "
            f"({abs(balance) / gross * 100:.1f}%). Reduce population, increase "
            f"density, or enlarge the site."
        )
    elif balance < gross * 0.05:
        prog.warnings.append(
            f"TIGHT FIT: only {balance:,.1f} ha ({balance / gross * 100:.1f}%) "
            f"remains unallocated. This leaves little room for constrained land, "
            f"road geometry losses or future expansion."
        )
