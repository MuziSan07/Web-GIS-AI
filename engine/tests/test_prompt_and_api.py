"""Tests for prompt parsing and the HTTP API.

The prompt parser is the seam the LLM agent layer will replace. These tests
define the contract it must satisfy, so the model-backed implementation can be
measured against the same expectations.
"""

from __future__ import annotations

import pytest
from fastapi.testclient import TestClient

from planeng.api import app
from planeng.brief.models import Objective, SettlementForm
from planeng.nlp.prompt import parse_prompt


@pytest.fixture(scope="module")
def client():
    return TestClient(app)


# -- prompt parsing ---------------------------------------------------------

def test_reads_area_and_population():
    b = parse_prompt("Plan a 600 hectare settlement for 25,000 people.")
    assert b.site_area_ha == 600
    assert b.target_population == 25000


def test_reads_thousands_separators_and_ha_abbreviation():
    b = parse_prompt("Design 1,200 ha for 40,000 residents.")
    assert b.site_area_ha == 1200
    assert b.target_population == 40000


def test_square_kilometres_convert_to_hectares():
    b = parse_prompt("Plan 6 km2 for 25,000 people.")
    assert b.site_area_ha == 600


def test_plot_count_infers_population():
    b = parse_prompt("Plan 600 ha providing 5,000 residential plots.")
    assert b.target_residential_plots == 5000
    assert b.target_population == pytest.approx(17500, abs=1)


def test_explicit_facility_counts_are_extracted():
    b = parse_prompt(
        "Plan 600 ha for 25,000 people with 6 primary schools, "
        "2 secondary schools and 3 clinics.")
    assert b.required_facilities["primary_school"] == 6
    assert b.required_facilities["secondary_school"] == 2
    assert b.required_facilities["clinic"] == 3


def test_written_numbers_are_understood():
    b = parse_prompt("Plan 600 ha for 20,000 people with two secondary schools.")
    assert b.required_facilities["secondary_school"] == 2


def test_settlement_form_is_detected():
    assert parse_prompt("Plan a linear 600 ha town for 10,000 people "
                        "along the main road.").settlement_form is SettlementForm.LINEAR
    assert parse_prompt("Plan clustered neighbourhood units on 600 ha for "
                        "10,000 people.").settlement_form is SettlementForm.CLUSTERED


def test_objectives_are_detected():
    b = parse_prompt("Plan 600 ha for 20,000 people, optimise for low "
                     "infrastructure cost and walkability, protect wetlands.")
    assert Objective.MINIMISE_INFRASTRUCTURE_COST in b.objectives
    assert Objective.MAXIMISE_ACCESSIBILITY in b.objectives
    assert Objective.MAXIMISE_ENVIRONMENTAL_PROTECTION in b.objectives


def test_density_hint_is_applied():
    dense = parse_prompt("Plan 600 ha for 20,000 people with high-density plots.")
    sparse = parse_prompt("Plan 600 ha for 20,000 people with large plots.")
    assert dense.density_mix.high > sparse.density_mix.high


def test_industrial_can_be_excluded():
    b = parse_prompt("Plan 600 ha for 12,000 people with no industrial area.")
    assert b.require_industrial is False


def test_horizon_is_read():
    b = parse_prompt("Plan 600 ha for 20,000 people over a 30-year horizon.")
    assert b.plan_horizon_years == 30


def test_source_prompt_is_retained_for_audit():
    prompt = "Plan 600 ha for 20,000 people."
    assert parse_prompt(prompt).source_prompt == prompt


def test_incomplete_prompt_says_exactly_what_is_missing():
    with pytest.raises(ValueError, match="target population or plot count"):
        parse_prompt("Plan a 600 hectare settlement.")
    with pytest.raises(ValueError, match="site area"):
        parse_prompt("Plan a settlement for 25,000 people.")


# -- API --------------------------------------------------------------------

def test_health(client):
    assert client.get("/api/health").json()["status"] == "ok"


def test_standards_endpoint_exposes_validation_state(client):
    body = client.get("/api/standards").json()
    assert body["validated"] is False
    assert body["placeholder_paths"]
    assert "UNVALIDATED" in body["warning"]


def test_plan_from_prompt(client):
    res = client.post("/api/plan/prompt", json={
        "prompt": "Plan a 600 hectare settlement for 15,000 people "
                  "with high-density plots.",
        "cell_size": 25.0,
    })
    assert res.status_code == 200
    body = res.json()
    assert body["report"]["total_plots"] > 0
    assert body["geojson"]["type"] == "FeatureCollection"
    assert any(f["properties"]["layer"] == "zone" for f in body["geojson"]["features"])


def test_unreadable_prompt_returns_422_not_500(client):
    res = client.post("/api/plan/prompt",
                      json={"prompt": "Build me something nice please."})
    assert res.status_code == 422
    assert "Missing" in res.json()["detail"]


def test_plan_from_explicit_brief(client):
    res = client.post("/api/plan", json={
        "project_name": "Explicit Town", "site_area_ha": 600,
        "target_population": 12000, "density_low": 0.1,
        "density_medium": 0.4, "density_high": 0.5, "cell_size": 25.0,
    })
    assert res.status_code == 200
    assert res.json()["report"]["project_name"] == "Explicit Town"


def test_bad_density_mix_is_rejected(client):
    res = client.post("/api/plan", json={
        "project_name": "Bad", "site_area_ha": 600, "target_population": 12000,
        "density_low": 0.5, "density_medium": 0.5, "density_high": 0.5,
    })
    assert res.status_code == 422
    assert "sum to 1.0" in res.json()["detail"]


def test_impossible_density_is_rejected_with_explanation(client):
    res = client.post("/api/plan", json={
        "project_name": "Impossible", "site_area_ha": 10,
        "target_population": 25000, "cell_size": 25.0,
    })
    assert res.status_code == 422


def test_index_page_serves(client):
    res = client.get("/")
    assert res.status_code == 200
    assert "Master Plan Copilot" in res.text
