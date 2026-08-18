"""Tests for internal access streets in over-deep blocks."""

from __future__ import annotations

import math

import numpy as np
import pytest
from shapely.geometry import box

from planeng.allocate import allocate_land
from planeng.analysis import analyse_constraints, compute_suitability
from planeng.brief import DensityMix, PlanningBrief, derive_programme
from planeng.roads import add_internal_streets, build_blocks, generate_network
from planeng.roads.blocks import Block, BlockLayout
from planeng.roads.internal import DEPTH_TOLERANCE, _rect_metrics
from planeng.roads.network import _target_plot_depth
from planeng.site import generate_site
from planeng.standards import load_standards


@pytest.fixture(scope="module")
def std():
    return load_standards("botswana")


@pytest.fixture(scope="module")
def ctx(std):
    site = generate_site(target_area_ha=600, cell_size=15, seed=5)
    brief = PlanningBrief(
        project_name="Internal", site_area_ha=600, target_population=12000,
        density_mix=DensityMix(low=0.10, medium=0.40, high=0.50),
    )
    prog = derive_programme(brief, std)
    cons = analyse_constraints(site, std)
    suit = compute_suitability(site, std, cons)
    alloc = allocate_land(site, prog, suit, cons)
    net = generate_network(site, std, cons, alloc, prog)
    blocks = build_blocks(site, std, alloc, net, prog)
    return dict(site=site, std=std, prog=prog, alloc=alloc, net=net, blocks=blocks)


# -- geometry helper --------------------------------------------------------

def test_rect_metrics_on_a_known_rectangle():
    short, long_len, ux, uy, cx, cy = _rect_metrics(box(0, 0, 200, 60))
    assert short == pytest.approx(60, abs=0.01)
    assert long_len == pytest.approx(200, abs=0.01)
    assert abs(ux) == pytest.approx(1.0, abs=0.01)   # long axis runs east-west
    assert uy == pytest.approx(0.0, abs=0.01)
    assert (cx, cy) == pytest.approx((100, 30), abs=0.01)


def test_rect_metrics_rejects_degenerate_geometry():
    assert _rect_metrics(box(0, 0, 10, 0)) is None


# -- splitting decisions ----------------------------------------------------

def _layout(*geoms) -> BlockLayout:
    return BlockLayout(blocks=[
        Block(geometry=g, use="residential", block_id=f"B{i:03d}")
        for i, g in enumerate(geoms)
    ])


def test_a_block_two_rows_deep_is_left_alone(ctx):
    """Splitting a block that already fits two rows costs a whole reserve."""
    std, prog, net = ctx["std"], ctx["prog"], ctx["net"]
    depth = _target_plot_depth(std, prog)
    width = std.get("roads.hierarchy.local_access.reserve_width_m")

    fine = box(0, 0, 300, 2 * depth)
    added = add_internal_streets(ctx["site"], std, net, _layout(fine), prog)
    assert added == []


def test_a_deep_block_gets_a_street(ctx):
    std, prog = ctx["std"], ctx["prog"]
    depth = _target_plot_depth(std, prog)
    width = std.get("roads.hierarchy.local_access.reserve_width_m")
    span = 2 * depth + width

    deep = box(0, 0, 300, span * 2.2)
    net = ctx["net"]
    before = len(net.segments)
    added = add_internal_streets(ctx["site"], std, net, _layout(deep), prog)
    assert added, "deep block was not given an internal street"
    assert len(net.segments) == before + len(added)
    for seg in added:
        assert seg.road_class == "local_access"
        assert seg.reserve_width_m == width


def test_deeper_blocks_get_more_streets(ctx):
    std, prog, net = ctx["std"], ctx["prog"], ctx["net"]
    depth = _target_plot_depth(std, prog)
    span = 2 * depth + std.get("roads.hierarchy.local_access.reserve_width_m")

    two = add_internal_streets(ctx["site"], std, net, _layout(box(0, 0, 300, span * 2.2)), prog)
    four = add_internal_streets(ctx["site"], std, net, _layout(box(0, 0, 300, span * 4.2)), prog)
    assert len(four) > len(two)


def test_streets_run_along_the_block_long_axis(ctx):
    """Streets across the short axis would meet the perimeter grid at 90 degrees
    and leave plots facing the wrong way."""
    std, prog, net = ctx["std"], ctx["prog"], ctx["net"]
    depth = _target_plot_depth(std, prog)
    span = 2 * depth + std.get("roads.hierarchy.local_access.reserve_width_m")

    # Long axis east-west.
    added = add_internal_streets(ctx["site"], std, net,
                                 _layout(box(0, 0, 400, span * 2.2)), prog)
    assert added
    for seg in added:
        (x0, y0), (x1, y1) = seg.geometry.coords[0], seg.geometry.coords[-1]
        assert abs(x1 - x0) > abs(y1 - y0), "street does not follow the long axis"


def test_streets_stay_inside_their_block(ctx):
    std, prog, net = ctx["std"], ctx["prog"], ctx["net"]
    depth = _target_plot_depth(std, prog)
    span = 2 * depth + std.get("roads.hierarchy.local_access.reserve_width_m")
    block = box(0, 0, 300, span * 2.4)

    for seg in add_internal_streets(ctx["site"], std, net, _layout(block), prog):
        assert seg.geometry.within(block.buffer(0.01))


def test_only_residential_blocks_are_split(ctx):
    std, prog, net = ctx["std"], ctx["prog"], ctx["net"]
    depth = _target_plot_depth(std, prog)
    span = 2 * depth + std.get("roads.hierarchy.local_access.reserve_width_m")

    layout = BlockLayout(blocks=[
        Block(geometry=box(0, 0, 300, span * 3), use="industrial", block_id="I1"),
    ])
    assert add_internal_streets(ctx["site"], std, net, layout, prog) == []


# -- effect on the real site ------------------------------------------------

def test_internal_streets_reduce_the_deep_block_count(ctx):
    """The measured effect is small because few blocks are actually too deep.

    Median block depth on this site is close to the two-row requirement, so the
    street grid is already sizing blocks correctly and only a minority need
    splitting. This test records the direction of the effect, not a large gain:
    the bulk of the remaining yield loss is irregular block edges, not depth.
    """
    site, std, prog, alloc = ctx["site"], ctx["std"], ctx["prog"], ctx["alloc"]
    net = generate_network(site, std, analyse_constraints(site, std), alloc, prog)
    blocks = build_blocks(site, std, alloc, net, prog)

    depth = _target_plot_depth(std, prog)
    span = 2 * depth + std.get("roads.hierarchy.local_access.reserve_width_m")

    def deep_count(layout):
        n = 0
        for b in layout.for_use("residential"):
            m = _rect_metrics(b.geometry)
            if m and m[0] > span * DEPTH_TOLERANCE:
                n += 1
        return n

    before = deep_count(blocks)
    added = add_internal_streets(site, std, net, blocks, prog)
    if not added:
        pytest.skip("no deep blocks on this site")
    net.rebuild_reserves(site)
    after = deep_count(build_blocks(site, std, alloc, net, prog))
    assert after < before


def test_most_blocks_already_fit_two_rows(ctx):
    """Documents why internal streets are a minor gain rather than a major one.

    An earlier estimate put this fix at roughly 30% more plots. It is not:
    the street grid derives its spacing from plot depth, so the median block is
    already the right depth and only a small minority need splitting.
    """
    std, prog, blocks = ctx["std"], ctx["prog"], ctx["blocks"]
    depth = _target_plot_depth(std, prog)
    span = 2 * depth + std.get("roads.hierarchy.local_access.reserve_width_m")

    shorts = []
    for b in blocks.for_use("residential"):
        m = _rect_metrics(b.geometry)
        if m:
            shorts.append(m[0])
    assert shorts
    too_deep = sum(1 for s in shorts if s > span * DEPTH_TOLERANCE)
    assert too_deep / len(shorts) < 0.25, (
        f"{too_deep}/{len(shorts)} blocks too deep - the grid spacing is wrong"
    )
    assert float(np.median(shorts)) < span * DEPTH_TOLERANCE
