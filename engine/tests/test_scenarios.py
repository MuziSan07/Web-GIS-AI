"""Tests for scenario generation and scoring.

Comparisons run at a coarse cell size with two forms: this exercises every code
path, and four forms at fine resolution would put minutes into the suite for no
extra coverage.
"""

from __future__ import annotations

import pytest

from planeng.analysis.suitability import CRITERIA_WEIGHTS
from planeng.brief.models import DensityMix, PlanningBrief, SettlementForm
from planeng.scenarios import DEFAULT_FORMS, compare, describe, score, weights_for
from planeng.scenarios.score import SCALES, WEIGHTS, _scale
from planeng.site import generate_site
from planeng.standards import load_standards

TWO_FORMS = [SettlementForm.COMPACT, SettlementForm.INFRASTRUCTURE_MINIMUM]


@pytest.fixture(scope="module")
def std():
    return load_standards("botswana")


@pytest.fixture(scope="module")
def comparison(std):
    brief = PlanningBrief(
        project_name="Scenarios", site_area_ha=600, target_population=12000,
        density_mix=DensityMix(low=0.05, medium=0.35, high=0.60),
    )
    site = generate_site(name="Scenarios", target_area_ha=600, cell_size=40, seed=5)
    return compare(brief, forms=TWO_FORMS, standards=std, site=site, cell_size=40)


# -- forms are weightings, not separate algorithms --------------------------

def test_compact_is_the_baseline_weighting():
    assert weights_for(SettlementForm.COMPACT) == CRITERIA_WEIGHTS


def test_linear_prefers_the_road_over_the_centre():
    """A linear town has no single middle, so centrality must stop driving it."""
    w = weights_for(SettlementForm.LINEAR)["residential"]
    assert w["road_access"] > CRITERIA_WEIGHTS["residential"]["road_access"]
    assert w["centrality"] < CRITERIA_WEIGHTS["residential"]["centrality"]


def test_clustered_weakens_the_single_centre():
    w = weights_for(SettlementForm.CLUSTERED)["residential"]
    assert w["centrality"] < CRITERIA_WEIGHTS["residential"]["centrality"]
    assert w["amenity_proximity"] > CRITERIA_WEIGHTS["residential"]["amenity_proximity"]


def test_infrastructure_minimum_hugs_existing_services():
    w = weights_for(SettlementForm.INFRASTRUCTURE_MINIMUM)["residential"]
    assert w["road_access"] >= 0.5
    assert "amenity_proximity" not in w or w["amenity_proximity"] < 0.1


def test_zeroed_criteria_are_dropped_not_left_at_zero():
    """A zero-weight criterion still costs a raster multiply; drop it."""
    for form in DEFAULT_FORMS:
        for crit in weights_for(form).values():
            assert all(v > 0 for v in crit.values())


def test_every_form_keeps_a_usable_weighting():
    for form in DEFAULT_FORMS:
        for use, crit in weights_for(form).items():
            assert crit, f"{form.value}/{use} has no criteria left"
            assert sum(crit.values()) > 0


def test_describe_reports_what_changed_from_the_baseline():
    """A form must be inspectable, not a label on a number."""
    info = describe(SettlementForm.LINEAR)
    assert info["form"] == "linear"
    assert info["rationale"]
    changes = info["weight_changes"]["residential"]
    assert "road_access" in changes
    assert "->" in changes["road_access"]


def test_compact_reports_no_weight_changes():
    assert describe(SettlementForm.COMPACT)["weight_changes"] == {}


# -- scoring ----------------------------------------------------------------

def test_scale_is_clamped_and_directional():
    # Infrastructure efficiency improves as metres per plot falls.
    assert _scale("infrastructure_efficiency", 12.0) == 100.0
    assert _scale("infrastructure_efficiency", 30.0) == 0.0
    assert _scale("infrastructure_efficiency", 5.0) == 100.0     # clamped
    assert _scale("infrastructure_efficiency", 99.0) == 0.0      # clamped
    assert 40 < _scale("infrastructure_efficiency", 21.0) < 60


def test_every_metric_has_a_declared_scale_and_weight():
    """A metric whose scale lives inline is a judgement nobody can challenge."""
    assert set(SCALES) == set(WEIGHTS)


def test_scorecard_covers_all_four_client_criteria(comparison):
    expected = {"infrastructure_efficiency", "accessibility",
                "environmental_suitability", "land_use_balance"}
    for scenario in comparison.scenarios:
        assert {m.name for m in scenario.card.metrics} == expected


def test_every_metric_carries_a_raw_value_and_an_explanation(comparison):
    """The score ranks; the raw value is what a planner argues with."""
    for scenario in comparison.scenarios:
        for m in scenario.card.metrics:
            assert m.unit and m.explanation
            assert 0.0 <= m.score <= 100.0


def test_scores_are_measured_from_the_plan_not_the_brief(comparison):
    for scenario in comparison.scenarios:
        infra = scenario.card.metric("infrastructure_efficiency")
        expected = (scenario.plan.network.total_length_m
                    / scenario.plan.cadastre.total_plots)
        assert infra.raw == pytest.approx(expected, rel=1e-6)


def test_overall_is_the_weighted_mean(comparison):
    for scenario in comparison.scenarios:
        card = scenario.card
        expected = sum(m.score * WEIGHTS[m.name] for m in card.metrics) / sum(
            WEIGHTS[m.name] for m in card.metrics)
        assert card.overall == pytest.approx(expected)


def test_a_plan_with_no_plots_scores_zero_not_infinity(std):
    """Road length per plot divides by plot count; an empty layout must not
    produce inf or crash the ranking."""
    from planeng.scenarios.score import _infrastructure_efficiency

    class FakePlan:
        class cadastre: total_plots = 0
        class network: total_length_m = 5000.0

    m = _infrastructure_efficiency(FakePlan())
    assert m.score == 0.0
    assert m.raw == 0.0


# -- comparison -------------------------------------------------------------

def test_all_requested_forms_are_generated(comparison):
    assert len(comparison.scenarios) == len(TWO_FORMS)
    assert comparison.failures == {}


def test_forms_actually_produce_different_plans(comparison):
    """If two forms score identically, the weighting is not a real lever."""
    scores = [s.card.overall for s in comparison.scenarios]
    assert len(set(round(x, 3) for x in scores)) == len(scores)


def test_scenarios_share_one_site(comparison):
    """Different terrain per scenario would measure the site, not the form."""
    sites = {id(s.plan.site) for s in comparison.scenarios}
    assert len(sites) == 1


def test_ranking_is_by_overall_descending(comparison):
    overalls = [s.card.overall for s in comparison.ranked]
    assert overalls == sorted(overalls, reverse=True)
    assert comparison.recommended is comparison.ranked[0]


def test_recommendation_is_justified_per_metric(comparison):
    summary = comparison.summary()
    assert summary["recommended"] == comparison.ranked[0].form.value
    for reason in summary["recommended_because"]:
        assert " vs " in reason


def test_a_failing_form_does_not_lose_the_others(monkeypatch, std):
    """One form crashing must not discard the scenarios that succeeded."""
    import importlib

    # `from planeng.scenarios import compare` binds the re-exported *function*,
    # not the module it lives in, so patch the module explicitly.
    compare_module = importlib.import_module("planeng.scenarios.compare")
    real = compare_module.generate_master_plan
    calls = {"n": 0}

    def flaky(brief, **kwargs):
        calls["n"] += 1
        if brief.settlement_form is SettlementForm.LINEAR:
            raise RuntimeError("synthetic failure")
        return real(brief, **kwargs)

    monkeypatch.setattr(compare_module, "generate_master_plan", flaky)
    brief = PlanningBrief(project_name="X", site_area_ha=600,
                          target_population=12000)
    site = generate_site(name="X", target_area_ha=600, cell_size=40, seed=5)
    result = compare(brief, forms=[SettlementForm.COMPACT, SettlementForm.LINEAR],
                     standards=std, site=site, cell_size=40)

    assert len(result.scenarios) == 1
    assert "linear" in result.failures
    assert result.recommended is not None


def test_summary_is_json_serialisable(comparison):
    import json

    json.dumps(comparison.summary())
