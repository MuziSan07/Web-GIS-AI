"""Street lighting and the utility corridor cross-section.

Two small modules that belong together, because both are about what fits inside
the road reserve.

**Lighting** is a schedule, not a routing problem: poles at the class spacing
along every run, extra luminaires at junctions where the conflict points are.
The output that matters to a client is the connected load, because street
lighting runs every night forever and is one of the larger permanent line items
in a municipal budget.

**The corridor** is the check nobody runs until it is too late. Four services
share one reserve, and they cannot share a trench: a water main below a sewer is
a contamination path on the day the sewer leaks, and a cable over a water main
turns every repair into an outage. The ruleset fixes a lateral offset for each
service; this verifies that those offsets fit inside the reserve that is
actually provided, and that the separations hold.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field

from ..standards.loader import Standards
from .graph import ServiceGraph


@dataclass
class LightingPlan:
    poles_by_class: dict[str, int] = field(default_factory=dict)
    junction_luminaires: int = 0
    load_kw: float = 0.0
    warnings: list[str] = field(default_factory=list)

    @property
    def total_poles(self) -> int:
        return sum(self.poles_by_class.values())

    def summary(self) -> dict:
        return {
            "total_poles": self.total_poles,
            "junction_luminaires": self.junction_luminaires,
            "connected_load_kw": round(self.load_kw, 1),
            "annual_energy_mwh": round(self.load_kw * 4380 / 1000, 1),
            "by_class": [
                {"road_class": k, "poles": v}
                for k, v in sorted(self.poles_by_class.items(),
                                   key=lambda kv: -kv[1])
            ],
            "warnings": self.warnings,
        }


def design_lighting(graph: ServiceGraph, std: Standards) -> LightingPlan:
    plan = LightingPlan()
    spacing = std.get("infrastructure.street_lighting.spacing_m")
    watts = std.get("infrastructure.street_lighting.load_w_per_luminaire")
    per_junction = std.get("infrastructure.street_lighting.junction_luminaires")

    # Length by class first, then poles: spacing a pole every N metres of each
    # short edge separately would round up once per edge and overcount badly on
    # a network of eight hundred runs.
    length: dict[str, float] = {}
    for (a, b), run in zip(graph.edges, graph.lengths):
        cls = graph.class_of(a, b)
        length[cls] = length.get(cls, 0.0) + run

    default = spacing.get("local_access", 30)
    for cls, run in length.items():
        gap = spacing.get(cls, default)
        plan.poles_by_class[cls] = int(math.ceil(run / gap)) if gap else 0

    degree: dict[int, int] = {}
    for a, b in graph.edges:
        degree[a] = degree.get(a, 0) + 1
        degree[b] = degree.get(b, 0) + 1
    junctions = sum(1 for d in degree.values() if d >= 3)
    plan.junction_luminaires = junctions * per_junction

    plan.load_kw = (plan.total_poles + plan.junction_luminaires) * watts / 1000
    return plan


# ---------------------------------------------------------------------------


@dataclass
class CorridorCheck:
    service: str
    offset_m: float
    fits: bool
    note: str = ""


@dataclass
class CorridorPlan:
    checks: list[CorridorCheck] = field(default_factory=list)
    narrowest_reserve_m: float = 0.0
    required_width_m: float = 0.0
    separations_ok: bool = True
    servitude_ha: float = 0.0
    warnings: list[str] = field(default_factory=list)

    def summary(self) -> dict:
        return {
            "narrowest_reserve_m": self.narrowest_reserve_m,
            "required_corridor_width_m": round(self.required_width_m, 2),
            "separations_ok": self.separations_ok,
            "servitude_ha": round(self.servitude_ha, 2),
            "services": [
                {"service": c.service, "offset_m": c.offset_m,
                 "fits": c.fits, "note": c.note}
                for c in self.checks
            ],
            "warnings": self.warnings,
        }


def check_corridor(std: Standards, network, trunk_length_m: float = 0.0
                   ) -> CorridorPlan:
    """Verify the four services fit the narrowest reserve they must share."""
    plan = CorridorPlan()
    offsets = std.get("utility_corridor.offsets_from_reserve_edge_m")
    separations = std.get("utility_corridor.min_horizontal_separation_m")
    servitudes = std.get("utility_corridor.servitude_width_m")

    widths = [s.reserve_width_m for s in network.segments] if network else []
    plan.narrowest_reserve_m = min(widths) if widths else 0.0

    # Services are laid from both edges inward, so the corridor has to hold the
    # two deepest offsets plus clearance between them.
    ordered = sorted(offsets.items(), key=lambda kv: kv[1])
    plan.required_width_m = (ordered[-1][1] + ordered[-2][1]
                             if len(ordered) >= 2 else 0.0)

    for service, offset in ordered:
        fits = offset * 2 <= plan.narrowest_reserve_m
        plan.checks.append(CorridorCheck(
            service=service, offset_m=offset, fits=fits,
            note="" if fits else
            f"{offset} m from each edge does not fit a "
            f"{plan.narrowest_reserve_m:.0f} m reserve.",
        ))

    gap = abs(offsets.get("water", 0) - offsets.get("sewer", 0))
    need = separations.get("water_to_sewer", 0)
    if gap < need:
        plan.separations_ok = False
        plan.warnings.append(
            f"Water and sewer are set {gap:.1f} m apart against a {need:.1f} m "
            f"minimum separation. A sewer leak beside a water main is a "
            f"contamination path, and this offset does not clear it.")

    if plan.required_width_m > plan.narrowest_reserve_m:
        plan.warnings.append(
            f"The four services need {plan.required_width_m:.1f} m of corridor "
            f"and the narrowest reserve is {plan.narrowest_reserve_m:.0f} m. "
            f"Either the local street reserve widens or a service moves out of "
            f"the road into a servitude across private land.")

    trunk_width = max(servitudes.values()) if servitudes else 0.0
    plan.servitude_ha = trunk_length_m * trunk_width / 10_000
    return plan
