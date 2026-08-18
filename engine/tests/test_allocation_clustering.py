"""Tests for precinct-based allocation and geometry robustness."""

from __future__ import annotations

import numpy as np
import pytest
from scipy import ndimage
from shapely.geometry import Polygon

from planeng.allocate import allocate_land
from planeng.allocate.zones import CLUSTERED, _min_patch_cells
from planeng.analysis import analyse_constraints, compute_suitability
from planeng.brief import DensityMix, PlanningBrief, derive_programme
from planeng.geom import clean, polygonise, safe_intersection
from planeng.site import generate_site
from planeng.standards import load_standards


@pytest.fixture(scope="module")
def std():
    return load_standards("botswana")


@pytest.fixture(scope="module")
def ctx(std):
    site = generate_site(target_area_ha=600, cell_size=15, seed=5)
    brief = PlanningBrief(
        project_name="Clustering", site_area_ha=600, target_population=12000,
        density_mix=DensityMix(low=0.10, medium=0.40, high=0.50),
    )
    prog = derive_programme(brief, std)
    cons = analyse_constraints(site, std)
    suit = compute_suitability(site, std, cons)
    alloc = allocate_land(site, prog, suit, cons)
    return dict(site=site, prog=prog, cons=cons, suit=suit, alloc=alloc)


# -- precinct growth --------------------------------------------------------

def test_institutional_land_is_grown_as_precincts(ctx):
    """A 5 ha secondary school cannot be sited on sub-hectare fragments.

    Before precinct growth, institutional land arrived as 38 patches averaging
    under a hectare and most facilities could not be placed.
    """
    site, alloc = ctx["site"], ctx["alloc"]
    struct = ndimage.generate_binary_structure(2, 2)
    lab, n = ndimage.label(alloc.mask_for("institutional"), structure=struct)
    sizes = sorted((site.cells_to_ha(int((lab == i).sum())) for i in range(1, n + 1)),
                   reverse=True)
    assert sizes, "no institutional land allocated"
    assert sizes[0] >= 5.0, f"largest institutional precinct is only {sizes[0]:.1f} ha"


def test_precincts_are_sized_from_the_facilities_that_must_fit(ctx, std):
    """Precinct size is a consequence of the schedule, not a magic number."""
    site, prog = ctx["site"], ctx["prog"]
    cells = _min_patch_cells(site, prog, "institutional")
    largest_facility = max(
        f.site_area_ha for f in prog.facilities
        if f.category in ("education", "health", "safety", "community",
                          "civic", "utilities")
    )
    assert site.cells_to_ha(cells) >= largest_facility


def test_precincts_are_distributed_not_merged(ctx):
    """Seeds are penalised near existing precincts.

    Without that penalty every seed lands beside the last one and the precincts
    coalesce into a single blob, which then cannot satisfy the separation
    standard between same-type facilities.
    """
    site, alloc = ctx["site"], ctx["alloc"]
    struct = ndimage.generate_binary_structure(2, 2)
    _, n = ndimage.label(alloc.mask_for("institutional"), structure=struct)
    assert n >= 2, "institutional land collapsed into a single precinct"


def test_only_institutional_is_clustered(ctx):
    """Clustering the other uses was measured and rejected.

    It shattered residential land and cost a third of the plot yield while
    placing no additional facility. The constant records that decision.
    """
    assert CLUSTERED == ("institutional",)


def test_clustering_does_not_shatter_residential(ctx):
    """Residential is the dominant use; fragmenting it destroys plot yield."""
    site, alloc = ctx["site"], ctx["alloc"]
    patches = polygonise(alloc.mask_for("residential"), site, min_area_m2=100)
    assert len(patches) < 200, f"residential fragmented into {len(patches)} patches"


def test_every_use_still_hits_its_quota(ctx):
    """Precinct growth must not leak area away from the land budget."""
    for row in ctx["alloc"].summary():
        assert abs(row["variance_pct"]) < 10, row


def test_allocation_still_respects_constraints(ctx):
    from planeng.allocate.zones import UNASSIGNED
    alloc, cons = ctx["alloc"], ctx["cons"]
    assigned = alloc.labels != UNASSIGNED
    assert not (assigned & ~cons.buildable).any()


# -- geometry robustness ----------------------------------------------------

def test_polygonise_repairs_pinched_regions():
    """Best-first growth produces regions that pinch to a single cell.

    Those polygonise into rings that touch themselves. They are invalid, they
    survive rotation and buffering unnoticed, and they surface much later as a
    GEOS side location conflict inside an unrelated module.
    """
    site = generate_site(target_area_ha=100, cell_size=10, seed=1)
    arr = np.zeros(site.shape, dtype=bool)
    # Two squares joined at one corner: the classic self-touching ring.
    arr[10:20, 10:20] = True
    arr[20:30, 20:30] = True
    for poly in polygonise(arr, site):
        assert poly.is_valid
        assert poly.geom_type == "Polygon"


def test_clean_repairs_an_invalid_polygon():
    bowtie = Polygon([(0, 0), (2, 2), (2, 0), (0, 2)])
    assert not bowtie.is_valid
    assert clean(bowtie).is_valid


def test_safe_intersection_matches_plain_intersection_when_valid():
    a = Polygon([(0, 0), (4, 0), (4, 4), (0, 4)])
    b = Polygon([(2, 2), (6, 2), (6, 6), (2, 6)])
    assert safe_intersection(a, b).area == pytest.approx(a.intersection(b).area)


def test_safe_intersection_survives_an_invalid_input():
    """The point of the helper: return a result rather than crash the pipeline."""
    bowtie = Polygon([(0, 0), (2, 2), (2, 0), (0, 2)])
    square = Polygon([(0, 0), (2, 0), (2, 2), (0, 2)])
    result = safe_intersection(bowtie, square)
    assert result is not None
    assert result.is_valid
