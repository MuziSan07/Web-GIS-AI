"""Scoring a generated plan.

Every metric here is measured off the plan that was actually produced. Nothing
is asserted, estimated, or carried over from the brief — if a number appears in
a scorecard, some geometry or schedule in the plan produced it.

Scores are reported two ways on purpose. The **raw** value is the engineering
quantity a planner can argue with (metres of road per plot, percent of dwellings
within a school catchment). The **0-100 score** is only that raw value placed on
a stated scale, so scenarios can be ranked. Where a scale bound is a judgement
call it is named and justified here rather than buried in a constant.
"""

from __future__ import annotations

from dataclasses import dataclass, field

from ..pipeline import MasterPlan

# Scale anchors. Each pair is (raw value scoring 0, raw value scoring 100).
# These set what "good" means and are the most contestable numbers in this
# module, so they are declared in one place for a planner to challenge.
SCALES = {
    # Metres of road reserve per residential plot. Around 12 m/plot is typical
    # for a conventional grid layout; 30 m/plot is sprawling.
    "infrastructure_efficiency": (30.0, 12.0),
    # Share of dwellings inside the catchment of the facilities that have one.
    # Anchored at 70/98 rather than 40/95: measured output sits between 94% and
    # 98%, and a scale topping out at 95 scored every form 100 and told a
    # planner nothing. Below 70% the plan already raises a coverage warning.
    "accessibility": (70.0, 98.0),
    # Mean suitability of the land each use was actually given, 0-1.
    "environmental_suitability": (0.35, 0.70),
    # Mean absolute deviation from the land budget, as a share of the site.
    # In practice every form scores ~100 here, because the allocator enforces
    # the budget with hard quotas whichever form is chosen. It is kept because
    # it is a real conformance check — a future allocator that trades balance
    # for another objective would show up here — but it is weighted low, since
    # a criterion nothing can fail cannot rank anything.
    "land_use_balance": (2.0, 0.0),
}

WEIGHTS = {
    "infrastructure_efficiency": 0.30,
    "accessibility": 0.30,
    "environmental_suitability": 0.30,
    # Low: measured, this metric does not discriminate between forms (see the
    # note on its scale). Weighting it like the others would dilute the three
    # that do.
    "land_use_balance": 0.10,
}


@dataclass
class Metric:
    name: str
    raw: float
    unit: str
    score: float
    explanation: str


@dataclass
class Scorecard:
    form: str
    metrics: list[Metric] = field(default_factory=list)
    plots: int = 0
    road_km: float = 0.0
    facilities_sited: int = 0
    facilities_unplaced: int = 0
    warnings: list[str] = field(default_factory=list)

    @property
    def overall(self) -> float:
        """Weighted mean of the four metrics."""
        if not self.metrics:
            return 0.0
        total = sum(WEIGHTS[m.name] for m in self.metrics)
        return sum(m.score * WEIGHTS[m.name] for m in self.metrics) / total

    def metric(self, name: str) -> Metric | None:
        return next((m for m in self.metrics if m.name == name), None)

    def summary(self) -> dict:
        return {
            "form": self.form,
            "overall": round(self.overall, 1),
            "plots": self.plots,
            "road_km": round(self.road_km, 2),
            "facilities_sited": self.facilities_sited,
            "facilities_unplaced": self.facilities_unplaced,
            "metrics": [
                {
                    "name": m.name, "raw": round(m.raw, 2), "unit": m.unit,
                    "score": round(m.score, 1), "explanation": m.explanation,
                }
                for m in self.metrics
            ],
            "warnings": self.warnings,
        }


def _scale(name: str, raw: float) -> float:
    """Place a raw value on its 0-100 scale, clamped."""
    zero, hundred = SCALES[name]
    if abs(hundred - zero) < 1e-12:
        return 50.0
    return max(0.0, min(100.0, (raw - zero) / (hundred - zero) * 100.0))


def score(plan: MasterPlan) -> Scorecard:
    """Score one generated plan on the four criteria the client asked for."""
    card = Scorecard(
        form=plan.brief.settlement_form.value,
        plots=plan.cadastre.total_plots,
        road_km=plan.network.total_length_m / 1000,
        facilities_sited=len(plan.facility_plan.facilities),
        facilities_unplaced=sum(plan.facility_plan.unplaced.values()),
    )

    card.metrics.append(_infrastructure_efficiency(plan))
    card.metrics.append(_accessibility(plan))
    card.metrics.append(_environmental_suitability(plan))
    card.metrics.append(_land_use_balance(plan))

    if card.plots == 0:
        card.warnings.append("No plots were produced; scores are not meaningful.")
    if card.facilities_unplaced:
        card.warnings.append(
            f"{card.facilities_unplaced} required facilities could not be sited."
        )
    return card


# ---------------------------------------------------------------------------

def _infrastructure_efficiency(plan: MasterPlan) -> Metric:
    """Metres of road reserve per plot served.

    Road length per plot, not total road length: a scenario that houses fewer
    people on less road has not been more efficient, it has done less.
    """
    plots = plan.cadastre.total_plots
    road_m = plan.network.total_length_m
    raw = road_m / plots if plots else float("inf")
    if plots == 0:
        return Metric("infrastructure_efficiency", 0.0, "m/plot", 0.0,
                      "No plots produced, so road length serves nobody.")
    return Metric(
        "infrastructure_efficiency", raw, "m of road per plot",
        _scale("infrastructure_efficiency", raw),
        f"{road_m / 1000:,.1f} km of road serving {plots:,} plots.",
    )


def _accessibility(plan: MasterPlan) -> Metric:
    """Mean catchment coverage across facilities that have a catchment.

    Coverage already measures the share of dwellings within the walking or
    service distance the ruleset requires, so averaging it answers "can people
    reach the things provided for them?".
    """
    coverage = plan.facility_plan.coverage
    if not coverage:
        return Metric("accessibility", 0.0, "% of dwellings", 0.0,
                      "No facilities with a catchment were sited.")
    raw = sum(coverage.values()) / len(coverage)
    worst = min(coverage.items(), key=lambda kv: kv[1])
    return Metric(
        "accessibility", raw, "% of dwellings within catchment",
        _scale("accessibility", raw),
        f"Mean over {len(coverage)} facility types. Weakest: "
        f"{worst[0].replace('_', ' ')} at {worst[1]:.0f}%.",
    )


def _environmental_suitability(plan: MasterPlan) -> Metric:
    """Area-weighted mean suitability of the land each use was given.

    Weighted by area because a scenario that puts a hectare of commercial on
    perfect ground and 300 hectares of housing on poor ground has not made a
    good environmental choice.
    """
    alloc = plan.allocation
    total_ha = sum(alloc.achieved_ha.values())
    if total_ha <= 0:
        return Metric("environmental_suitability", 0.0, "mean suitability", 0.0,
                      "No land was allocated.")
    raw = sum(
        alloc.mean_suitability.get(use, 0.0) * alloc.achieved_ha.get(use, 0.0)
        for use in alloc.uses
    ) / total_ha
    return Metric(
        "environmental_suitability", raw, "mean suitability (0-1)",
        _scale("environmental_suitability", raw),
        f"Area-weighted across {len(alloc.uses)} land uses on "
        f"{total_ha:,.0f} ha.",
    )


def _land_use_balance(plan: MasterPlan) -> Metric:
    """How far the delivered land uses drift from the land budget.

    Mean absolute deviation as a share of the site, so a scenario cannot score
    well by overshooting one use and undershooting another.
    """
    rows = plan.allocation.summary()
    site_ha = plan.site.area_ha
    if not rows or site_ha <= 0:
        return Metric("land_use_balance", 100.0, "% of site", 0.0,
                      "No allocation to compare against the budget.")
    deviation = sum(abs(r["achieved_ha"] - r["target_ha"]) for r in rows) / len(rows)
    raw = deviation / site_ha * 100
    worst = max(rows, key=lambda r: abs(r["variance_pct"]))
    return Metric(
        "land_use_balance", raw, "% of site mean deviation",
        _scale("land_use_balance", raw),
        f"Largest drift: {worst['use'].replace('_', ' ')} at "
        f"{worst['variance_pct']:+.1f}% of its target.",
    )
