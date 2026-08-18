"""Tests for the development control engine."""

from __future__ import annotations

import pytest
from shapely.geometry import box

from planeng.allocate import allocate_land
from planeng.analysis import analyse_constraints, compute_suitability
from planeng.brief import DensityMix, PlanningBrief, derive_programme
from planeng.cadastre import subdivide
from planeng.cadastre.parcels import Parcel
from planeng.control import apply_controls
from planeng.control.controls import _control_for, _envelope
from planeng.roads import build_blocks, generate_network
from planeng.site import generate_site
from planeng.standards import load_standards


@pytest.fixture(scope="module")
def std():
    return load_standards("botswana")


@pytest.fixture(scope="module")
def ctx(std):
    site = generate_site(target_area_ha=600, cell_size=15, seed=5)
    brief = PlanningBrief(
        project_name="Control", site_area_ha=600, target_population=12000,
        density_mix=DensityMix(low=0.10, medium=0.40, high=0.50),
    )
    prog = derive_programme(brief, std)
    cons = analyse_constraints(site, std)
    suit = compute_suitability(site, std, cons)
    alloc = allocate_land(site, prog, suit, cons)
    net = generate_network(site, std, cons, alloc, prog)
    blocks = build_blocks(site, std, alloc, net, prog)
    cad = subdivide(site, std, blocks, net, prog)
    rep = apply_controls(site, std, cad, net, alloc)
    return dict(site=site, std=std, net=net, cad=cad, alloc=alloc, rep=rep)


# -- compliance -------------------------------------------------------------

def test_every_plot_is_assessed(ctx):
    assert len(ctx["rep"].controls) == ctx["cad"].total_plots


def test_the_layout_complies_with_the_ruleset_it_came_from(ctx):
    """The generator must not breach its own standards.

    This caught a real defect: subdivision rounded the column count, producing
    columns narrower than the minimum frontage, and 1,854 of 3,394 plots were
    in breach. Flooring the column count fixed it.
    """
    rep = ctx["rep"]
    assert rep.summary()["compliance_pct"] == 100.0, rep.violations_by_rule()


def test_plot_area_below_the_minimum_is_a_violation(ctx, std):
    tiny = Parcel(geometry=box(0, 0, 10, 10), plot_number="X1", block_id="B1",
                  band="medium", use="residential", frontage_m=20.0)
    control = _control_for(tiny, std, None, [])
    assert not control.compliant
    assert any(v.startswith("plot_area") for v in control.violations)


def test_frontage_below_the_minimum_is_a_violation(ctx, std):
    narrow = Parcel(geometry=box(0, 0, 10, 60), plot_number="X2", block_id="B1",
                    band="medium", use="residential", frontage_m=10.0)
    control = _control_for(narrow, std, None, [])
    assert any(v.startswith("frontage") for v in control.violations)


def test_a_compliant_plot_reports_no_violations(std):
    good = Parcel(geometry=box(0, 0, 20, 30), plot_number="X3", block_id="B1",
                  band="medium", use="residential", frontage_m=20.0)
    assert _control_for(good, std, None, []).compliant


# -- envelope ---------------------------------------------------------------

def test_envelope_is_smaller_than_the_plot(ctx):
    for c in ctx["rep"].controls[:300]:
        if c.envelope is not None:
            assert c.envelope_area_m2 < c.plot_area_m2


def test_envelope_without_an_identifiable_frontage_uses_the_strictest_setback(std):
    """With no road to measure from, the front setback applies all round.

    Assuming the smaller side setback instead would overstate every envelope on
    a plot the engine cannot orient.
    """
    spec = std.get("residential.densities.medium")
    env = _envelope(box(0, 0, 20, 30), spec, None, [])
    assert env is not None
    strictest = max(spec["setback_front_m"], spec["setback_side_m"],
                    spec["setback_rear_m"])
    expected = (20 - 2 * strictest) * (30 - 2 * strictest)
    assert env.area == pytest.approx(expected, rel=0.02)


def test_envelope_with_a_frontage_applies_the_front_setback_only_to_the_front(std):
    """The whole point of edge-aware setbacks.

    A 20x30 m medium plot fronting a road along its southern edge keeps
    2-3 m at the sides and rear and gives up 4.5 m at the front, which leaves
    materially more envelope than applying 4.5 m all round.
    """
    from shapely.strtree import STRtree

    spec = std.get("residential.densities.medium")
    plot = box(0, 0, 20, 30)
    road = box(-5, -10, 25, 0)               # abuts the southern edge
    tree = STRtree([road])

    with_front = _envelope(plot, spec, tree, [road])
    without = _envelope(plot, spec, None, [])

    assert with_front is not None
    assert with_front.area > without.area
    # Sides and rear at the stricter of side/rear; front at the front setback.
    strictest = max(spec["setback_side_m"], spec["setback_rear_m"])
    expected = (20 - 2 * strictest) * (30 - strictest - spec["setback_front_m"])
    assert with_front.area == pytest.approx(expected, rel=0.05)


def test_envelope_is_conservative_not_generous(std):
    """Side and rear are merged to the stricter of the two.

    The merge must never report more envelope than per-edge setbacks would.
    """
    spec = std.get("residential.densities.medium")
    env = _envelope(box(0, 0, 20, 30), spec, None, [])
    generous = (20 - 2 * spec["setback_side_m"]) * (30 - 2 * spec["setback_side_m"])
    assert env.area <= generous + 0.01


def test_denser_bands_have_smaller_envelopes(ctx):
    means = {r["band"]: r["mean_envelope_m2"] for r in ctx["rep"].summary()["by_band"]}
    if "high" in means and "low" in means:
        assert means["high"] < means["low"]


def test_a_plot_too_small_for_its_setbacks_has_no_envelope(std):
    spec = std.get("residential.densities.low")     # 6 m front, 3 m side/rear
    assert _envelope(box(0, 0, 5, 5), spec, None, []) is None


def test_footprint_is_capped_by_whichever_rule_binds(ctx, std):
    """Coverage and setbacks are both limits; the tighter one governs."""
    for c in ctx["rep"].controls[:300]:
        spec = std.get(f"residential.densities.{c.band}")
        coverage_cap = c.plot_area_m2 * spec["coverage_max_pct"] / 100
        assert c.max_footprint_m2 <= coverage_cap + 0.5
        if c.envelope is not None:
            assert c.max_footprint_m2 <= c.envelope_area_m2 + 0.5


def test_floor_area_follows_the_far_standard(ctx, std):
    for c in ctx["rep"].controls[:200]:
        far = std.get(f"residential.densities.{c.band}.far_max")
        assert c.max_floor_area_m2 == pytest.approx(c.plot_area_m2 * far, rel=0.01)


def test_storeys_come_from_the_ruleset(ctx, std):
    for c in ctx["rep"].controls[:200]:
        assert c.max_storeys == std.get(
            f"residential.densities.{c.band}.storeys_max")


# -- land use compatibility -------------------------------------------------

def test_compatibility_conflicts_cite_a_rule(ctx):
    for c in ctx["rep"].conflicts:
        assert c["rule"].startswith("compatibility.")
        assert c["verdict"] in ("buffer_required", "incompatible")
        assert c["shared_boundary_m"] > 0


def test_conflicts_are_reported_as_warnings(ctx):
    rep = ctx["rep"]
    for c in rep.conflicts:
        assert any(c["a"] in w and c["b"] in w for w in rep.warnings)


# -- provenance -------------------------------------------------------------

def test_unvalidated_ruleset_taints_the_control_report(ctx):
    """Compliance against placeholder standards is not compliance."""
    assert any("UNVALIDATED RULESET" in w for w in ctx["rep"].warnings)


def test_summary_totals_are_consistent(ctx):
    s = ctx["rep"].summary()
    assert s["compliant"] + s["non_compliant"] == s["plots_assessed"]
    assert sum(b["plots"] for b in s["by_band"]) == s["plots_assessed"]
