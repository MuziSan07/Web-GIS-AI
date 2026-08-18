"""Evaluation harness for the Brief Agent.

An agent that cannot be measured cannot be improved, and "it looked right in
the demo" is not a basis for choosing a model. This scores a model against
briefs whose correct extraction is known, field by field, so swapping models or
editing the prompt produces a number rather than an impression.

The cases deliberately include a bare prompt and a negation, which are where
small models fail: the interesting score is not the average but which fields a
given model gets wrong.
"""

from __future__ import annotations

from dataclasses import dataclass, field

from ..standards.loader import Standards, load_standards
from .brief_agent import understand


@dataclass
class Case:
    name: str
    prompt: str
    expect: dict            # field -> expected value
    note: str = ""


# Expectations are what a planner would read off the sentence, nothing more.
CASES: list[Case] = [
    Case(
        name="full_brief",
        prompt=(
            "Plan a sustainable 600 hectare settlement called Lehutshelo for "
            "25,000 people. Provide 6 primary schools and 2 secondary schools. "
            "Use high-density plots. Avoid wetlands and optimise for low "
            "infrastructure cost."
        ),
        expect={
            "project_name": "Lehutshelo",
            "site_area_ha": 600.0,
            "target_population": 25000,
            "density": "high",
            "facilities": {"primary_school": 6, "secondary_school": 2},
            "environmental_protection": "high",
            "has_objective": "minimise_infrastructure_cost",
        },
    ),
    Case(
        name="negation",
        prompt=(
            "Design a low-density 450 ha town for 12,000 residents with "
            "3 primary schools and no industrial area."
        ),
        expect={
            "site_area_ha": 450.0,
            "target_population": 12000,
            "density": "low",
            "facilities": {"primary_school": 3},
            "require_industrial": False,
        },
        note="Negation. Small models routinely miss 'no industrial area'.",
    ),
    Case(
        name="bare",
        prompt="Plan 600 ha for 18,000 people.",
        expect={
            "site_area_ha": 600.0,
            "target_population": 18000,
            "facilities": {},
            "environmental_protection": "standard",
        },
        note="Nothing stated. The model must not invent facilities or intent.",
    ),
    Case(
        name="plots_not_population",
        prompt="Plan 600 ha providing 5,000 residential plots and 2 clinics.",
        expect={
            "site_area_ha": 600.0,
            "target_residential_plots": 5000,
            "facilities": {"clinic": 2},
        },
    ),
    Case(
        name="linear_form",
        prompt=(
            "Plan a linear settlement along the main road on 300 ha for "
            "9,000 people, optimised for walkability."
        ),
        expect={
            "site_area_ha": 300.0,
            "target_population": 9000,
            "settlement_form": "linear",
            "has_objective": "maximise_accessibility",
        },
    ),
]


@dataclass
class CaseScore:
    case: str
    passed: list[str] = field(default_factory=list)
    failed: list[str] = field(default_factory=list)
    error: str = ""
    duration_s: float = 0.0

    @property
    def score(self) -> float:
        total = len(self.passed) + len(self.failed)
        return len(self.passed) / total if total else 0.0


def _check(brief, expect: dict) -> tuple[list[str], list[str]]:
    passed, failed = [], []

    def verdict(field_name: str, ok: bool, got) -> None:
        (passed if ok else failed).append(
            field_name if ok else f"{field_name} (got {got!r})"
        )

    for key, want in expect.items():
        if key == "facilities":
            got = brief.required_facilities
            verdict("facilities", got == want, got)
        elif key == "density":
            mix = brief.density_mix
            got = None if mix is None else max(mix.as_dict(), key=mix.as_dict().get)
            verdict("density", got == want, got)
        elif key == "has_objective":
            got = [o.value for o in brief.objectives]
            verdict(f"objective:{want}", want in got, got)
        elif key == "settlement_form":
            got = brief.settlement_form.value
            verdict("settlement_form", got == want, got)
        else:
            got = getattr(brief, key, None)
            verdict(key, got == want, got)

    return passed, failed


def run(
    model: str,
    std: Standards | None = None,
    cases: list[Case] | None = None,
) -> list[CaseScore]:
    """Score one model across the case set."""
    std = std or load_standards()
    scores: list[CaseScore] = []

    for case in cases or CASES:
        try:
            result = understand(case.prompt, std, model=model, allow_fallback=False)
        except Exception as exc:
            scores.append(CaseScore(case=case.name, error=str(exc)[:160]))
            continue
        passed, failed = _check(result.brief, case.expect)
        scores.append(CaseScore(
            case=case.name, passed=passed, failed=failed,
            duration_s=result.duration_s,
        ))

    return scores


def report(scores: list[CaseScore]) -> dict:
    fields_ok = sum(len(s.passed) for s in scores)
    fields_total = fields_ok + sum(len(s.failed) for s in scores)
    return {
        "cases": len(scores),
        "cases_perfect": sum(1 for s in scores if s.failed == [] and not s.error),
        "fields_correct": fields_ok,
        "fields_total": fields_total,
        "field_accuracy": round(fields_ok / fields_total * 100, 1) if fields_total else 0.0,
        "mean_duration_s": round(
            sum(s.duration_s for s in scores) / len(scores), 1) if scores else 0.0,
        "failures": {s.case: s.failed for s in scores if s.failed},
        "errors": {s.case: s.error for s in scores if s.error},
    }
