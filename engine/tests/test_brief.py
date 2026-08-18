"""Tests for the planning brief and programme derivation."""

from __future__ import annotations

import pytest
from pydantic import ValidationError

from planeng.brief import DensityMix, PlanningBrief, derive_programme
from planeng.standards import load_standards


@pytest.fixture(scope="module")
def std():
    return load_standards("botswana")


def brief(**kw) -> PlanningBrief:
    base = dict(project_name="Test Town", site_area_ha=600.0, target_population=25000)
    return PlanningBrief(**{**base, **kw})


# -- brief validation -------------------------------------------------------

def test_density_mix_must_sum_to_one():
    with pytest.raises(ValidationError, match="sum to 1.0"):
        DensityMix(low=0.5, medium=0.5, high=0.5)


def test_absurd_density_is_rejected_before_any_gis_work():
    with pytest.raises(ValidationError, match="persons/ha gross"):
        brief(site_area_ha=10, target_population=25000)


def test_blank_project_name_rejected():
    with pytest.raises(ValidationError):
        brief(project_name="   ")


def test_design_population_includes_expansion_allowance():
    b = brief(target_population=20000, expansion_allowance_pct=15)
    assert b.design_population == 23000


# -- dwelling schedule ------------------------------------------------------

def test_dwelling_counts_total_exactly(std):
    """Rounding must never lose or invent a plot."""
    for pop in (7777, 10001, 25000, 33333):
        prog = derive_programme(brief(target_population=pop), std)
        assert prog.total_dwellings == sum(d.dwellings for d in prog.dwellings)
        expected = -(-int(pop / 3.5 * 1.05) // 1)  # ceil of households * allowance
        assert abs(prog.total_dwellings - expected) <= 1


def test_explicit_plot_target_overrides_population(std):
    prog = derive_programme(brief(target_residential_plots=5000), std)
    assert prog.total_dwellings == 5000


def test_plot_target_short_of_population_warns(std):
    prog = derive_programme(brief(target_population=25000,
                                  target_residential_plots=2000), std)
    assert any("short of the" in w for w in prog.warnings)


def test_denser_mix_consumes_less_land(std):
    sparse = derive_programme(
        brief(density_mix=DensityMix(low=0.5, medium=0.3, high=0.2)), std)
    dense = derive_programme(
        brief(density_mix=DensityMix(low=0.0, medium=0.2, high=0.8)), std)
    assert dense.residential_gross_ha < sparse.residential_gross_ha


def test_gross_residential_area_exceeds_net(std):
    """Gross must include the internal street network."""
    prog = derive_programme(brief(), std)
    for d in prog.dwellings:
        assert d.gross_area_ha > d.net_area_ha


# -- facility schedule ------------------------------------------------------

def test_facility_counts_scale_with_population(std):
    small = derive_programme(brief(target_population=10000), std)
    large = derive_programme(brief(target_population=30000), std)
    assert (large.facility("primary_school").count
            > small.facility("primary_school").count)


def test_explicit_facility_requirement_is_honoured(std):
    prog = derive_programme(brief(required_facilities={"secondary_school": 4}), std)
    school = prog.facility("secondary_school")
    assert school.count == 4
    assert school.driver == "specified in brief"


def test_min_count_floor_applies_to_small_populations(std):
    """A settlement of 3,000 still needs one primary school, not zero."""
    prog = derive_programme(brief(site_area_ha=600, target_population=3000), std)
    assert prog.facility("primary_school").count >= 1


def test_area_driven_facility_uses_rate_not_threshold(std):
    prog = derive_programme(brief(target_population=20000), std)
    cemetery = prog.facility("cemetery")
    assert cemetery is not None
    assert "ha per 1,000" in cemetery.driver
    # 0.5 ha per 1,000 of design population (23,000) = 11.5 ha
    assert cemetery.total_area_ha == pytest.approx(11.5, abs=0.1)


def test_facilities_sized_on_design_not_target_population(std):
    """Infrastructure is sized for the horizon, not day one."""
    no_growth = derive_programme(brief(expansion_allowance_pct=0), std)
    with_growth = derive_programme(brief(expansion_allowance_pct=50), std)
    assert (with_growth.facility("clinic").count
            >= no_growth.facility("clinic").count)


# -- land budget and feasibility -------------------------------------------

def test_infeasible_brief_is_flagged_not_silently_accepted(std):
    """The engine must say so when the programme does not fit the site."""
    prog = derive_programme(
        brief(site_area_ha=600, target_population=25000,
              density_mix=DensityMix(low=0.5, medium=0.3, high=0.2)), std)
    assert prog.allocated_ha > 600
    assert any("DOES NOT FIT" in w for w in prog.warnings)


def test_feasible_brief_is_not_flagged(std):
    prog = derive_programme(brief(site_area_ha=600, target_population=9000), std)
    assert prog.allocated_ha <= 600
    assert not any("DOES NOT FIT" in w for w in prog.warnings)


def test_unvalidated_ruleset_warning_reaches_the_programme(std):
    """A placeholder ruleset must taint every downstream output."""
    prog = derive_programme(brief(), std)
    assert any("UNVALIDATED RULESET" in w for w in prog.warnings)


def test_industrial_land_can_be_omitted(std):
    with_ind = derive_programme(brief(require_industrial=True), std)
    without = derive_programme(brief(require_industrial=False), std)
    uses = {line.use for line in without.land_budget}
    assert "industrial" not in uses
    assert without.allocated_ha < with_ind.allocated_ha


# -- engineering demand -----------------------------------------------------

def test_water_demand_matches_standard(std):
    prog = derive_programme(brief(target_population=20000,
                                  expansion_allowance_pct=0), std)
    # 20,000 people x 150 l/day = 3,000 m3/day
    assert prog.infrastructure["water_average_day_m3"] == pytest.approx(3000, abs=1)


def test_sewage_is_a_fraction_of_water(std):
    prog = derive_programme(brief(), std)
    assert (prog.infrastructure["sewage_average_day_m3"]
            < prog.infrastructure["water_average_day_m3"])


def test_peak_exceeds_average_demand(std):
    prog = derive_programme(brief(), std)
    assert (prog.infrastructure["water_peak_day_m3"]
            > prog.infrastructure["water_average_day_m3"])
