"""Tests for the standards rules engine.

These guard the properties a planner relies on: that a ruleset which does not
add up is rejected outright, and that unvalidated numbers can never present
themselves as statutory standards.
"""

from __future__ import annotations

import textwrap

import pytest

from planeng.standards import load_standards
from planeng.standards.loader import _validate


def test_bundled_ruleset_loads():
    std = load_standards("botswana")
    assert std.jurisdiction == "Botswana"
    assert std.get("demographics.persons_per_household") == 3.5


def test_dotted_access_and_value_collapse():
    std = load_standards("botswana")
    # A {value:, source:} rule collapses to its value...
    assert isinstance(std.get("residential.net_to_gross_factor"), float)
    # ...while a plain scalar is returned directly.
    assert std.get("roads.hierarchy.collector.reserve_width_m") == 20


def test_missing_rule_raises():
    std = load_standards("botswana")
    with pytest.raises(KeyError, match="no.such.rule|No such rule"):
        std.get("roads.hierarchy.monorail.reserve_width_m")


def test_unvalidated_ruleset_produces_a_warning_banner():
    std = load_standards("botswana")
    assert not std.is_validated
    warning = std.provenance_warning()
    assert warning is not None
    assert "UNVALIDATED" in warning
    assert std.placeholder_paths(), "placeholder rules should be enumerable"


def test_density_mix_must_sum_to_one():
    issues = _validate({"residential": {"default_density_mix": {
        "low": 0.2, "medium": 0.5, "high": 0.4, "source": "x"}}})
    assert any("sum to 1.0" in i.message for i in issues)


def test_land_budget_must_sum_to_100():
    issues = _validate({"land_budget": {"residential_pct": 50, "commercial_pct": 10}})
    assert any("sum to 100" in i.message for i in issues)


def test_carriageway_wider_than_reserve_is_an_error():
    issues = _validate({"roads": {"hierarchy": {"collector": {
        "reserve_width_m": 10, "carriageway_m": 12}}}})
    assert any(i.severity == "error" and "narrower" in i.message for i in issues)


def test_inverted_plot_range_is_an_error():
    issues = _validate({"residential": {"densities": {"medium": {
        "plot_area_min_m2": 900, "plot_area_max_m2": 400}}}})
    assert any(i.severity == "error" and "exceeds" in i.message for i in issues)


def test_broken_ruleset_refuses_to_load(tmp_path):
    bad = tmp_path / "broken.yaml"
    bad.write_text(textwrap.dedent("""
        meta: {jurisdiction: Test, ruleset_version: "0", status: PLACEHOLDER}
        residential:
          default_density_mix: {low: 0.9, medium: 0.9, high: 0.9}
    """), encoding="utf-8")
    with pytest.raises(ValueError, match="internally inconsistent"):
        load_standards(path=bad)


def test_bundled_land_budget_does_not_double_count_circulation():
    """Internal residential streets live in net_to_gross_factor.

    Settlement-level circulation must therefore cover only the arterial,
    distributor and collector network. A figure at the 18-22% level would be
    counting neighbourhood streets twice and inflating land demand.
    """
    std = load_standards("botswana")
    assert std.get("land_budget.circulation_pct") <= 12
