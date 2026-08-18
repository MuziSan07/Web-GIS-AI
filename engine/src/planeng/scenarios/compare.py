"""Generate several settlement forms from one brief and compare them.

The client asked to see four scenarios scored side by side and then pick one.
This produces them from a single brief by varying only the settlement form, so
any difference in the scorecards is attributable to the form and not to a
different population, site, or ruleset.

The site is generated once and shared. Regenerating it per scenario would make
the comparison meaningless: two forms would be scored on different terrain.
"""

from __future__ import annotations

from dataclasses import dataclass, field

from ..brief.models import PlanningBrief, SettlementForm
from ..pipeline import MasterPlan, generate_master_plan
from ..site.model import Site
from ..site.synthetic import generate_site
from ..standards.loader import Standards, load_standards
from .forms import describe, weights_for
from .score import Scorecard, score

DEFAULT_FORMS = [
    SettlementForm.COMPACT,
    SettlementForm.LINEAR,
    SettlementForm.CLUSTERED,
    SettlementForm.INFRASTRUCTURE_MINIMUM,
]


@dataclass
class Scenario:
    form: SettlementForm
    plan: MasterPlan
    card: Scorecard
    rationale: str
    weight_changes: dict = field(default_factory=dict)
    duration_s: float = 0.0

    def summary(self) -> dict:
        return {
            **self.card.summary(),
            "rationale": self.rationale,
            "weight_changes": self.weight_changes,
            "duration_s": round(self.duration_s, 2),
        }


@dataclass
class Comparison:
    scenarios: list[Scenario] = field(default_factory=list)
    failures: dict[str, str] = field(default_factory=dict)

    @property
    def ranked(self) -> list[Scenario]:
        return sorted(self.scenarios, key=lambda s: -s.card.overall)

    @property
    def recommended(self) -> Scenario | None:
        return self.ranked[0] if self.scenarios else None

    def summary(self) -> dict:
        ranked = self.ranked
        best = ranked[0] if ranked else None

        # Say why the winner won, per metric, rather than only that it did.
        reasons: list[str] = []
        if best and len(ranked) > 1:
            runner_up = ranked[1]
            for m in best.card.metrics:
                other = runner_up.card.metric(m.name)
                if other is None or m.score - other.score <= 5:
                    continue
                # Show enough precision to see the difference. A reason that
                # reads "0.6 vs 0.6" is worse than no reason at all.
                digits = 1 if abs(m.raw - other.raw) >= 0.05 else 3
                reasons.append(
                    f"{m.name.replace('_', ' ')}: "
                    f"{m.raw:,.{digits}f} vs {other.raw:,.{digits}f} {m.unit}"
                )

        return {
            "scenarios": [s.summary() for s in ranked],
            "recommended": best.form.value if best else None,
            "recommended_because": reasons,
            "failures": self.failures,
        }


def compare(
    brief: PlanningBrief,
    forms: list[SettlementForm] | None = None,
    standards: Standards | None = None,
    site: Site | None = None,
    cell_size: float = 20.0,
) -> Comparison:
    """Run one brief through several settlement forms and score each.

    `cell_size` defaults coarser than a single plan's 10 m: four full pipelines
    at 10 m takes minutes, and the comparison is between forms, which a 20 m
    grid resolves perfectly well. Generate the winner at full resolution.
    """
    import time

    std = standards or load_standards(brief.jurisdiction)
    # One site for all scenarios, or the comparison measures terrain, not form.
    site = site or generate_site(
        name=brief.project_name,
        target_area_ha=brief.site_area_ha,
        cell_size=cell_size,
    )

    result = Comparison()

    for form in forms or DEFAULT_FORMS:
        scenario_brief = brief.model_copy(update={"settlement_form": form})
        t0 = time.perf_counter()
        try:
            plan = generate_master_plan(
                scenario_brief, site=site, standards=std,
                cell_size=cell_size, weights=weights_for(form),
            )
        except Exception as exc:
            # One form failing must not lose the other three.
            result.failures[form.value] = str(exc)[:200]
            continue

        info = describe(form)
        result.scenarios.append(Scenario(
            form=form,
            plan=plan,
            card=score(plan),
            rationale=info["rationale"],
            weight_changes=info["weight_changes"],
            duration_s=time.perf_counter() - t0,
        ))

    return result
