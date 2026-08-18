"""Tests for the local-model agent layer.

Every test here runs without Ollama. The transport is stubbed, because a test
suite that needs a 2 GB model running is a test suite nobody runs — and because
what needs testing is the engine's handling of model output, not the model.
"""

from __future__ import annotations

import json

import pytest

from planeng.agents import brief_agent, ollama
from planeng.agents.brief_agent import (
    _reconcile_numbers,
    _to_brief,
    build_schema,
    facility_keys,
    understand,
)
from planeng.brief.models import Objective, SettlementForm
from planeng.standards import load_standards


@pytest.fixture(scope="module")
def std():
    return load_standards("botswana")


def _payload(**over) -> dict:
    base = {
        "project_name": "Lehutshelo",
        "site_area_ha": 600,
        "target_population": 25000,
        "density": "high",
        "settlement_form": "compact",
        "objectives": ["minimise_infrastructure_cost"],
        "required_facilities": {"primary_school": 6},
        "environmental_protection": "high",
        "assumptions": ["assumed compact form"],
        "clarifications_needed": [],
    }
    base.update(over)
    return base


def _stub(monkeypatch, *payloads):
    """Make the transport return canned payloads, one per call."""
    calls = {"n": 0, "messages": []}

    def fake_chat_json(messages, schema, **kwargs):
        calls["messages"].append(messages)
        i = min(calls["n"], len(payloads) - 1)
        calls["n"] += 1
        return payloads[i], ollama.Completion(
            text=json.dumps(payloads[i]), model="stub",
            duration_s=1.0, eval_tokens=100, ran_on_cpu=False,
        )

    monkeypatch.setattr(ollama, "available", lambda *a, **k: True)
    monkeypatch.setattr(ollama, "chat_json", fake_chat_json)
    return calls


# -- schema is generated from the ruleset -----------------------------------

def test_facility_keys_come_from_the_ruleset(std):
    keys = facility_keys(std)
    assert "primary_school" in keys
    assert "kgotla" in keys
    assert "water_treatment_plant" not in keys


def test_schema_restricts_facilities_to_ruleset_vocabulary(std):
    """Regression: unconstrained, the model invented `water_treatment_plants`
    and wrote `primary_schools` for `primary_school`."""
    schema = build_schema(std)
    facilities = schema["properties"]["required_facilities"]
    assert facilities["additionalProperties"] is False
    assert set(facilities["properties"]) == set(facility_keys(std))


def test_schema_requires_the_decision_fields(std):
    """Left optional, constrained decoding omits them and explicit planner
    intent lands in prose instead of the fields that drive the plan."""
    required = set(build_schema(std)["required"])
    assert {"density", "settlement_form", "objectives",
            "environmental_protection"} <= required


def test_schema_does_not_require_facilities(std):
    """Regression: requiring it made a bare prompt return all 16 facilities."""
    assert "required_facilities" not in build_schema(std)["required"]


def test_schema_grows_with_the_ruleset(std):
    """A planner adding a facility type to the YAML gets an agent that can be
    asked for it, without a code change."""
    std.data["facilities"]["community"]["youth_centre"] = {
        "threshold_population": 15000, "site_area_ha": 0.4,
        "catchment_m": 2000, "min_count": 0, "source": "PLACEHOLDER",
    }
    try:
        assert "youth_centre" in build_schema(std)["properties"]["required_facilities"]["properties"]
    finally:
        del std.data["facilities"]["community"]["youth_centre"]


# -- converting model output to a brief -------------------------------------

def test_maps_density_word_to_a_mix():
    brief = _to_brief(_payload(density="high"), "x")
    mix = brief.density_mix.as_dict()
    assert mix["high"] > mix["low"]


def test_unspecified_density_leaves_the_mix_to_the_ruleset():
    assert _to_brief(_payload(density="unspecified"), "x").density_mix is None


def test_duplicate_objectives_are_collapsed():
    """The model repeats an enum value when the same intent is stated twice;
    a repeated objective would double its weight."""
    brief = _to_brief(_payload(objectives=[
        "maximise_accessibility", "maximise_accessibility",
        "minimise_infrastructure_cost",
    ]), "x")
    assert brief.objectives == [
        Objective.MAXIMISE_ACCESSIBILITY, Objective.MINIMISE_INFRASTRUCTURE_COST,
    ]


def test_unknown_objective_is_dropped_not_crashed():
    brief = _to_brief(_payload(objectives=["make_it_nice"]), "x")
    assert brief.objectives == [Objective.BALANCE_LAND_USE]


def test_unknown_settlement_form_falls_back_to_compact():
    assert _to_brief(_payload(settlement_form="spiral"), "x").settlement_form \
        is SettlementForm.COMPACT


@pytest.mark.parametrize("name", ["null", "none", "N/A", "", "  ", "unspecified"])
def test_placeholder_project_names_are_replaced(name):
    """A constrained model asked for a required string emits the literal
    "null" rather than omitting the field."""
    assert _to_brief(_payload(project_name=name), "x").project_name \
        == "Untitled Settlement"


def test_zero_and_negative_facility_counts_are_dropped():
    brief = _to_brief(_payload(required_facilities={
        "primary_school": 3, "clinic": 0, "library": -1,
    }), "x")
    assert brief.required_facilities == {"primary_school": 3}


# -- the regex/LLM hybrid ---------------------------------------------------

def test_text_overrules_the_model_on_numbers():
    """Measured: the model misread areas and populations that a regex reads
    exactly. Digits beside a unit are a lexical problem, not a reasoning one."""
    raw, corrections, _ = _reconcile_numbers(
        _payload(site_area_ha=999, target_population=1),
        "Plan a 600 hectare settlement for 25,000 people.",
    )
    assert raw["site_area_ha"] == 600
    assert raw["target_population"] == 25000
    assert len(corrections) == 2


def test_plot_count_leaking_into_facilities_is_stripped():
    """Regression: on "5,000 residential plots" the model wrote 5000 into
    three facility counts."""
    raw, corrections, _ = _reconcile_numbers(
        _payload(required_facilities={
            "clinic": 2, "primary_school": 5000, "sports_field": 5000,
        }),
        "Plan 600 ha providing 5,000 residential plots and 2 clinics.",
    )
    assert raw["required_facilities"] == {"clinic": 2}
    assert any("plot count" in c for c in corrections)


def test_a_correct_model_reading_is_left_alone():
    raw, corrections, _ = _reconcile_numbers(
        _payload(site_area_ha=600, target_population=25000),
        "Plan a 600 hectare settlement for 25,000 people.",
    )
    assert corrections == []


def test_vague_prompt_is_reported_as_ungrounded():
    """The parser finds no numbers, so the model's are inventions — the caller
    must be told, not handed a confident plan built on nothing."""
    raw, corrections, grounded = _reconcile_numbers(
        _payload(), "Build a nice town somewhere.")
    assert grounded is False
    assert corrections == []


def test_ungrounded_brief_is_refused_with_the_models_questions(monkeypatch, std):
    """Regression: with the model enabled, "Build me something nice" returned a
    complete 600 ha master plan invented from nothing."""
    _stub(monkeypatch, _payload(
        clarifications_needed=["How large is the site?", "How many residents?"]))
    with pytest.raises(ValueError, match="invented figures"):
        understand("Build me something nice please.", std)


# -- orchestration ----------------------------------------------------------

def test_understand_returns_a_validated_brief(monkeypatch, std):
    _stub(monkeypatch, _payload())
    result = understand("Plan a 600 ha settlement for 25,000 people.", std)
    assert result.source == "llm"
    assert result.brief.project_name == "Lehutshelo"
    assert result.assumptions == ["assumed compact form"]


def test_ruleset_vocabulary_is_given_to_the_model(monkeypatch, std):
    calls = _stub(monkeypatch, _payload())
    understand("Plan a 600 ha settlement for 25,000 people.", std)
    user_message = calls["messages"][0][-1]["content"]
    assert "primary_school" in user_message
    assert "kgotla" in user_message


def test_invalid_brief_gets_one_repair_attempt(monkeypatch, std):
    """The model returns a brief the engine rejects; it sees the engine's own
    error and corrects it.

    The invalid field is deliberately non-numeric: numeric mistakes are caught
    earlier by the text reconciliation, so a bad area would never reach the
    repair path at all.
    """
    calls = _stub(
        monkeypatch,
        _payload(plan_horizon_years=0),     # must be > 0
        _payload(),
    )
    result = understand("Plan a 600 hectare settlement for 25,000 people.", std)
    assert result.repaired is True
    assert result.source == "llm"
    assert calls["n"] == 2
    assert any("repaired" in w for w in result.warnings)
    # The repair prompt must quote what the engine rejected.
    assert "rejected by the planning engine" in calls["messages"][1][-1]["content"]


def test_the_hybrid_rescues_a_brief_the_model_got_wrong(monkeypatch, std):
    """The model returned an impossible 10 ha / 25,000 brief, but the text says
    600 ha for 20,000. Reading the numbers off the text repairs it before the
    engine ever sees it — no retry, no fallback."""
    calls = _stub(monkeypatch, _payload(site_area_ha=10, target_population=25000))
    result = understand("Plan a 600 hectare town for 20,000 people.", std)
    assert result.source == "llm"
    assert result.repaired is False
    assert calls["n"] == 1
    assert result.brief.site_area_ha == 600
    assert result.brief.target_population == 20000


def test_an_impossible_brief_is_refused_rather_than_retried_forever(monkeypatch, std):
    """When the text itself is impossible, no amount of retrying helps. The
    agent must stop and surface the reason, not loop."""
    calls = _stub(monkeypatch, _payload(site_area_ha=10, target_population=25000))
    with pytest.raises(ValueError, match="persons/ha gross"):
        understand("Plan a 10 hectare town for 25,000 people.", std)
    assert calls["n"] == 2          # one attempt, one repair, then it stops


def test_no_model_falls_back_to_the_parser(monkeypatch, std):
    monkeypatch.setattr(ollama, "available", lambda *a, **k: False)
    result = understand("Plan a 600 hectare town for 20,000 people.", std)
    assert result.source == "fallback_parser"
    assert result.brief.target_population == 20000
    assert any("rule-based parser" in w for w in result.warnings)


def test_fallback_can_be_refused(monkeypatch, std):
    """A caller measuring the model must not be silently given a regex."""
    monkeypatch.setattr(ollama, "available", lambda *a, **k: False)
    with pytest.raises(ollama.OllamaError):
        understand("Plan 600 ha for 20,000 people.", std, allow_fallback=False)


def test_transport_failure_falls_back(monkeypatch, std):
    monkeypatch.setattr(ollama, "available", lambda *a, **k: True)

    def boom(*a, **k):
        raise ollama.OllamaError("model not found")

    monkeypatch.setattr(ollama, "chat_json", boom)
    result = understand("Plan a 600 hectare town for 20,000 people.", std)
    assert result.source == "fallback_parser"


def test_source_is_always_reported(monkeypatch, std):
    """A planner must be able to tell whether a model or a regex read them."""
    _stub(monkeypatch, _payload())
    assert understand("Plan 600 ha for 25,000 people.", std).summary()["source"] == "llm"


# -- transport --------------------------------------------------------------

def test_gpu_failure_markers_are_recognised():
    """The Vulkan crash that killed every call on this machine must be read as
    'retry on CPU', not as a bad request."""
    assert ollama._looks_like_gpu_failure(
        "llama-server process has terminated: exit status 0xe06d7363")
    assert ollama._looks_like_gpu_failure("device Vulkan0 does not support 16-bit storage")
    assert not ollama._looks_like_gpu_failure("model 'nope' not found")


def test_available_is_false_when_nothing_is_listening():
    assert ollama.available(host="http://localhost:1", timeout=0.2) is False


def test_completion_reports_throughput():
    c = ollama.Completion(text="x", model="m", duration_s=2.0,
                          eval_tokens=50, ran_on_cpu=True)
    assert c.tokens_per_second == 25.0


# -- HTTP surface (stubbed transport, no Ollama) ----------------------------

def test_api_refuses_an_ungrounded_prompt_rather_than_inventing(monkeypatch):
    """Regression: with the model enabled, "Build me something nice" returned a
    200 with a complete 600 ha master plan invented from nothing."""
    from fastapi.testclient import TestClient

    from planeng.api import app

    _stub(monkeypatch, _payload(
        clarifications_needed=["How large is the site?"]))
    res = TestClient(app).post("/api/plan/prompt", json={
        "prompt": "Build me something nice please.", "cell_size": 25.0,
    })
    assert res.status_code == 422
    assert "invented figures" in res.json()["detail"]


def test_api_reports_what_the_agent_assumed(monkeypatch):
    """A planner reviewing a plan must see what was inferred, not just read."""
    from fastapi.testclient import TestClient

    from planeng.api import app

    _stub(monkeypatch, _payload(site_area_ha=600, target_population=12000))
    res = TestClient(app).post("/api/plan/prompt", json={
        "prompt": "Plan a 600 hectare settlement for 12,000 people.",
        "cell_size": 25.0,
    })
    assert res.status_code == 200
    understanding = res.json()["understanding"]
    assert understanding["source"] == "llm"
    assert understanding["assumptions"] == ["assumed compact form"]


def test_agent_status_endpoint_states_the_mode(monkeypatch):
    from fastapi.testclient import TestClient

    from planeng.api import app

    monkeypatch.setattr(ollama, "available", lambda *a, **k: False)
    body = TestClient(app).get("/api/agent").json()
    assert body["reachable"] is False
    assert body["mode"] == "rule-based parser"
