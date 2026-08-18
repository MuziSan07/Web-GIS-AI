"""The master planning pipeline.

One function, one plan. This is the seam the AI agent layer calls: agents build
a PlanningBrief and invoke `generate_master_plan`, and everything from here on
is deterministic. An agent cannot reach past this boundary to draw geometry
directly, which is what keeps the output reviewable.
"""

from __future__ import annotations

import time
from dataclasses import dataclass, field

from .allocate.zones import AllocationResult, allocate_land
from .analysis.constraints import ConstraintResult, analyse_constraints, buildable_area_ha
from .analysis.suitability import SuitabilityResult, compute_suitability
from .brief.derive import Programme, derive_programme
from .brief.models import PlanningBrief
from .export.geojson import plan_to_geojson
from .roads.blocks import BlockLayout, build_blocks, reconcile_plots
from .cadastre.parcels import CadastralLayout, subdivide
from .facilities.siting import FacilityPlan, site_facilities
from .roads.network import RoadNetwork, generate_network
from .site.model import Site
from .site.synthetic import generate_site
from .standards.loader import Standards, load_standards


@dataclass
class MasterPlan:
    brief: PlanningBrief
    site: Site
    programme: Programme
    constraints: ConstraintResult
    suitability: SuitabilityResult
    allocation: AllocationResult
    network: RoadNetwork
    blocks: BlockLayout
    cadastre: CadastralLayout
    facility_plan: FacilityPlan
    standards: Standards
    timings_ms: dict[str, float] = field(default_factory=dict)
    warnings: list[str] = field(default_factory=list)

    def to_geojson(self) -> dict:
        return plan_to_geojson(self.site, self.constraints, self.allocation,
                               network=self.network, blocks=self.blocks,
                               cadastre=self.cadastre,
                               facility_plan=self.facility_plan)

    def report(self) -> dict:
        """The machine-readable plan summary, used by the API and the UI."""
        buildable_ha = buildable_area_ha(self.site, self.constraints)
        return {
            "project_name": self.brief.project_name,
            "site": self.site.summary(),
            "brief": {
                "site_area_ha": self.brief.site_area_ha,
                "target_population": self.brief.target_population,
                "design_population": self.brief.design_population,
                "gross_density_pph": round(self.brief.gross_density_pph, 2),
                "settlement_form": self.brief.settlement_form.value,
                "plan_horizon_years": self.brief.plan_horizon_years,
            },
            "land": {
                "gross_site_ha": round(self.site.area_ha, 2),
                "buildable_ha": round(buildable_ha, 2),
                "constrained_ha": round(self.site.area_ha - buildable_ha, 2),
                "constrained_pct": round(
                    (self.site.area_ha - buildable_ha) / self.site.area_ha * 100, 1
                ),
            },
            "constraints": self.constraints.summary(self.site),
            "dwellings": [
                {
                    "band": d.band,
                    "plots": d.dwellings,
                    "plot_area_typical_m2": d.plot_area_typical_m2,
                    "net_ha": d.net_area_ha,
                    "gross_ha": d.gross_area_ha,
                }
                for d in self.programme.dwellings
            ],
            "total_plots": self.programme.total_dwellings,
            "facilities": [
                {
                    "key": f.key, "category": f.category, "count": f.count,
                    "site_area_ha": f.site_area_ha, "total_area_ha": f.total_area_ha,
                    "catchment_m": f.catchment_m, "driver": f.driver,
                }
                for f in sorted(self.programme.facilities,
                                key=lambda x: (x.category, x.key))
            ],
            "allocation": self.allocation.summary(),
            "roads": self.network.summary(self.site),
            "blocks": self.blocks.summary(),
            "cadastre": self.cadastre.summary(),
            "facility_siting": self.facility_plan.summary(),
            "infrastructure": self.programme.infrastructure,
            "warnings": self.warnings,
            "ruleset": {
                "jurisdiction": self.standards.jurisdiction,
                "version": self.standards.data["meta"]["ruleset_version"],
                "validated": self.standards.is_validated,
                "placeholder_count": len(self.standards.placeholder_paths()),
            },
            "timings_ms": {k: round(v, 1) for k, v in self.timings_ms.items()},
        }


def generate_master_plan(
    brief: PlanningBrief,
    site: Site | None = None,
    standards: Standards | None = None,
    cell_size: float = 10.0,
) -> MasterPlan:
    """Run the full pipeline from brief to allocated plan."""
    timings: dict[str, float] = {}

    def step(name: str, fn):
        t0 = time.perf_counter()
        result = fn()
        timings[name] = (time.perf_counter() - t0) * 1000
        return result

    std = standards or step("load_standards",
                            lambda: load_standards(brief.jurisdiction))

    if site is None:
        site = step("generate_site", lambda: generate_site(
            name=brief.project_name,
            target_area_ha=brief.site_area_ha,
            cell_size=cell_size,
        ))

    programme = step("derive_programme", lambda: derive_programme(brief, std))
    constraints = step("analyse_constraints", lambda: analyse_constraints(site, std))
    suitability = step("compute_suitability",
                       lambda: compute_suitability(site, std, constraints))
    allocation = step("allocate_land",
                      lambda: allocate_land(site, programme, suitability, constraints))
    network = step("generate_roads",
                   lambda: generate_network(site, std, constraints, allocation,
                                            programme))
    blocks = step("build_blocks",
                  lambda: build_blocks(site, std, allocation, network, programme))
    cadastre = step("subdivide",
                    lambda: subdivide(site, std, blocks, network, programme))
    facility_plan = step("site_facilities",
                         lambda: site_facilities(site, std, programme, blocks,
                                                 allocation))

    warnings = list(programme.warnings)
    warnings.extend(network.warnings)
    warnings.extend(blocks.warnings)
    warnings.extend(cadastre.warnings)
    warnings.extend(facility_plan.warnings)
    if (msg := reconcile_plots(blocks, programme.total_dwellings)) is not None:
        warnings.append(msg)

    # The programme is sized against gross site area, but constraints remove
    # land. Say so plainly rather than letting the quota scaling hide it.
    buildable_ha = buildable_area_ha(site, constraints)
    demanded = sum(allocation.target_ha.values())
    if demanded > buildable_ha * 1.001:
        warnings.append(
            f"Constraints reduce developable land to {buildable_ha:,.1f} ha of "
            f"{site.area_ha:,.1f} ha gross. Land use targets were scaled down "
            f"proportionally to fit; the full programme needs "
            f"{demanded:,.1f} ha."
        )

    return MasterPlan(
        brief=brief, site=site, programme=programme, constraints=constraints,
        suitability=suitability, allocation=allocation, network=network,
        blocks=blocks, cadastre=cadastre, facility_plan=facility_plan,
        standards=std, timings_ms=timings, warnings=warnings,
    )
