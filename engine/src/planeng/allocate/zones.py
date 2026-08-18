"""Land use allocation.

Assigns every buildable cell to a land use so that the land budget is met and
each use lands on the ground best suited to it.

The allocator is greedy on suitability with hard area quotas, followed by a
compactness pass. Greedy is the right first algorithm here: it is fast, it is
explainable to a planner cell by cell, and it produces a strong baseline that a
later mixed-integer formulation must beat before it earns its complexity. The
compactness pass matters because raw greedy output is speckled, and speckle is
not a plan.
"""

from __future__ import annotations

import heapq
import math
from dataclasses import dataclass, field

import numpy as np
from scipy import ndimage

from ..analysis.constraints import ConstraintResult
from ..analysis.suitability import SuitabilityResult
from ..brief.derive import Programme
from ..site.model import Site

UNASSIGNED = -1

# Uses allocated spatially at settlement scale. Circulation is not allocated
# here: settlement roads are generated later, and neighbourhood streets are
# already inside the residential gross-to-net factor.
ALLOCATABLE = [
    "residential",
    "commercial",
    "industrial",
    "institutional",
    "recreation_open_space",
]

# Uses grown as contiguous precincts rather than scattered cell by cell.
#
# Institutional only, on measurement. Institutional land must hold whole
# facilities - a 5 ha secondary school cannot sit on fragments - so growing it
# as precincts is what makes siting work. Clustering the other uses as well was
# tried and rejected: it shattered residential land into 292 patches instead of
# 125 and cost a third of the plot yield, while placing no additional facility.
#
#   greedy only          2,374 plots   43.4% net-to-gross   37 sited, 1 unplaced
#   institutional only   2,489 plots   43.8% net-to-gross   38 sited, 0 unplaced
#   all four uses        1,814 plots   29.0% net-to-gross   38 sited, 0 unplaced
#
CLUSTERED = ("institutional",)

# Facility categories that must fit on land of a given use, used to size how
# large a precinct of that use needs to be.
_USE_CATEGORIES = {
    "institutional": ("education", "health", "safety", "community", "civic",
                      "utilities"),
    "recreation_open_space": ("recreation",),
}


@dataclass
class AllocationResult:
    labels: np.ndarray                       # int index into `uses`, or UNASSIGNED
    uses: list[str]
    achieved_ha: dict[str, float] = field(default_factory=dict)
    target_ha: dict[str, float] = field(default_factory=dict)
    mean_suitability: dict[str, float] = field(default_factory=dict)

    def index_of(self, use: str) -> int:
        return self.uses.index(use)

    def mask_for(self, use: str) -> np.ndarray:
        return self.labels == self.index_of(use)

    def summary(self) -> list[dict]:
        rows = []
        for use in self.uses:
            target = self.target_ha.get(use, 0.0)
            achieved = self.achieved_ha.get(use, 0.0)
            rows.append({
                "use": use,
                "target_ha": round(target, 2),
                "achieved_ha": round(achieved, 2),
                "variance_ha": round(achieved - target, 2),
                "variance_pct": round((achieved - target) / target * 100, 1) if target else 0.0,
                "mean_suitability": round(self.mean_suitability.get(use, 0.0), 3),
            })
        return rows


def allocate_land(
    site: Site,
    programme: Programme,
    suitability: SuitabilityResult,
    constraints: ConstraintResult,
    compactness_passes: int = 10,
) -> AllocationResult:
    buildable = constraints.buildable
    n_buildable = int(buildable.sum())
    if n_buildable == 0:
        raise ValueError("No buildable land remains after constraints.")

    # -- targets ---------------------------------------------------------
    budget = {line.use: line.target_ha for line in programme.land_budget}
    targets_ha = {u: budget.get(u, 0.0) for u in ALLOCATABLE}
    demanded = sum(targets_ha.values())
    available_ha = site.cells_to_ha(n_buildable)

    # Constraints remove land, so scale the budget to what actually exists.
    # Preserving proportions keeps the land-use balance the planner asked for.
    if demanded > 0:
        scale = min(1.0, available_ha / demanded)
    else:
        scale = 1.0
    quotas = {u: site.ha_to_cells(ha * scale) for u, ha in targets_ha.items()}

    uses = [u for u in ALLOCATABLE if quotas.get(u, 0) > 0]
    labels = np.full(site.shape, UNASSIGNED, dtype=np.int16)

    # -- 1. clustered uses ------------------------------------------------
    # Non-residential uses are grown as contiguous regions before anything
    # else. Allocating them cell-by-cell alongside residential scatters them:
    # institutional land arrived as 38 fragments averaging under a hectare,
    # and a 5 ha secondary school cannot be sited on any of them. Real plans
    # consolidate these uses into precincts, and so does this.
    available = buildable.copy()
    for use in CLUSTERED:
        if use not in uses:
            continue
        used = _grow_regions(
            site, suitability.surfaces[use], available,
            quota_cells=quotas[use],
            min_patch_cells=_min_patch_cells(site, programme, use),
        )
        labels[used] = uses.index(use)
        available &= ~used

    # -- 2. everything else, greedily -------------------------------------
    # Rank every (cell, use) pair by suitability and take them in order. A cell
    # is claimed once; a use stops claiming when its quota is spent.
    remaining_uses = [u for u in uses if u not in CLUSTERED]
    if not remaining_uses:
        return _finalise(site, labels, uses, targets_ha, scale, suitability)

    idx = np.flatnonzero(available.ravel())
    if idx.size == 0:
        return _finalise(site, labels, uses, targets_ha, scale, suitability)
    scores = np.stack([suitability.surfaces[u].ravel()[idx] for u in remaining_uses])

    best_use = scores.argmax(axis=0)
    best_score = scores.max(axis=0)
    order = np.argsort(-best_score)          # strongest matches settle first

    remaining = {i: quotas[u] for i, u in enumerate(remaining_uses)}
    flat_labels = np.full(idx.size, UNASSIGNED, dtype=np.int16)

    for pos in order:
        u = int(best_use[pos])
        if remaining[u] > 0:
            flat_labels[pos] = u
            remaining[u] -= 1

    # Second pass: cells whose first choice was full go to their best available use.
    leftover = np.flatnonzero(flat_labels == UNASSIGNED)
    if leftover.size:
        for pos in leftover:
            ranked = np.argsort(-scores[:, pos])
            for u in ranked:
                if remaining[int(u)] > 0:
                    flat_labels[pos] = int(u)
                    remaining[int(u)] -= 1
                    break

    # The greedy pass works on its own reduced use list, so map those local
    # indices back onto the full one before writing them into the label grid.
    mapped = np.full(idx.size, UNASSIGNED, dtype=np.int16)
    for local, use in enumerate(remaining_uses):
        mapped[flat_labels == local] = uses.index(use)
    labels.ravel()[idx] = mapped

    # Compact only the greedily-assigned land. The grown precincts are already
    # contiguous, and letting the majority filter nibble at their edges would
    # undo the consolidation they exist to provide.
    labels = _compact(labels, available, passes=compactness_passes)

    return _finalise(site, labels, uses, targets_ha, scale, suitability)


def _finalise(site: Site, labels: np.ndarray, uses: list[str],
              targets_ha: dict[str, float], scale: float,
              suitability: SuitabilityResult) -> AllocationResult:
    res = AllocationResult(labels=labels, uses=uses)
    for i, use in enumerate(uses):
        m = labels == i
        res.achieved_ha[use] = site.cells_to_ha(int(m.sum()))
        res.target_ha[use] = targets_ha[use] * scale
        res.mean_suitability[use] = (
            float(suitability.surfaces[use][m].mean()) if m.any() else 0.0
        )
    return res


def _min_patch_cells(site: Site, programme: Programme, use: str) -> int:
    """How large a precinct of this use has to be to be useful.

    Sized from the largest facility that must fit on it, with headroom for the
    road reserves later cut out of the block. A precinct smaller than its
    largest intended tenant is land that can never serve its purpose.
    """
    categories = _USE_CATEGORIES.get(use)
    if categories:
        largest = max(
            (f.site_area_ha for f in programme.facilities
             if f.category in categories),
            default=0.0,
        )
        if largest > 0:
            return site.ha_to_cells(largest * 1.6)
    # Commercial and industrial have no facility schedule, but a coherent
    # quarter rather than a scatter of shops is still the goal.
    return site.ha_to_cells(3.0)


def _grow_regions(site: Site, surface: np.ndarray, available: np.ndarray,
                  quota_cells: int, min_patch_cells: int) -> np.ndarray:
    """Grow contiguous precincts from the best available seeds."""
    chosen = np.zeros(site.shape, dtype=bool)
    if quota_cells <= 0:
        return chosen

    rows, cols = site.shape
    free = available.copy()
    placed = 0

    # Successive seeds land next to the patch just placed, because that is
    # where suitability is still highest, and the precincts coalesce into one
    # blob. A settlement wants its schools and clinics distributed, so seeds
    # are penalised for proximity to precincts of the same use already sited.
    separation_m = 3.0 * math.sqrt(max(min_patch_cells, 1) * site.cell_area_m2)
    separation_cells = max(1.0, separation_m / site.cell_size)

    # Bounded so a pathological surface cannot spin: each iteration either
    # places a patch or burns a seed.
    for _ in range(1000):
        if placed >= quota_cells:
            break

        if chosen.any():
            d = ndimage.distance_transform_edt(~chosen)
            spread = np.clip(d / separation_cells, 0.0, 1.0)
        else:
            spread = 1.0

        masked = np.where(free, surface * (0.35 + 0.65 * spread), -np.inf)
        if not np.isfinite(masked).any():
            break
        seed = np.unravel_index(int(np.argmax(masked)), site.shape)
        if not free[seed]:
            break

        target = min(quota_cells - placed, max(min_patch_cells, 1))
        # Don't leave a stub too small to be a usable precinct behind.
        if 0 < quota_cells - placed - target < min_patch_cells * 0.5:
            target = quota_cells - placed

        region = _grow_one(surface, free, seed, target, rows, cols)
        if not region:
            free[seed] = False
            continue

        for r, c in region:
            chosen[r, c] = True
            free[r, c] = False
        placed += len(region)

    return chosen


def _grow_one(surface: np.ndarray, free: np.ndarray, seed, target: int,
              rows: int, cols: int) -> list[tuple[int, int]]:
    """Best-first expansion from a seed, so a precinct is sited well and whole."""
    region: list[tuple[int, int]] = []
    seen = {seed}
    heap = [(-float(surface[seed]), seed)]

    while heap and len(region) < target:
        _, (r, c) = heapq.heappop(heap)
        if not free[r, c]:
            continue
        region.append((r, c))
        for dr, dc in ((-1, 0), (1, 0), (0, -1), (0, 1)):
            nr, nc = r + dr, c + dc
            if 0 <= nr < rows and 0 <= nc < cols \
                    and (nr, nc) not in seen and free[nr, nc]:
                seen.add((nr, nc))
                heapq.heappush(heap, (-float(surface[nr, nc]), (nr, nc)))
    return region


# A cell is speckle when this few of its 3x3 neighbours share its use. At 3/9
# only fully isolated cells move, which leaves the ragged interlocking zones
# that make a generated plan look like camouflage rather than a layout. At 5/9
# any cell in a local minority is pulled into the use that surrounds it.
SPECKLE_THRESHOLD = 5 / 9


def _compact(labels: np.ndarray, buildable: np.ndarray, passes: int,
             threshold: float = SPECKLE_THRESHOLD) -> np.ndarray:
    """Majority filter that removes speckle while preserving each use's area.

    Swaps are made in pairs so the area assigned to every use is unchanged: a
    cell only moves to a neighbouring use if some cell of that use moves back.
    """
    if passes <= 0:
        return labels

    out = labels.copy()
    for _ in range(passes):
        proposals: dict[tuple[int, int], list[int]] = {}
        for u in np.unique(out[out != UNASSIGNED]):
            m = out == u
            # Count same-use neighbours in a 3x3 window.
            neigh = ndimage.uniform_filter(m.astype(float), size=3, mode="constant")
            # Isolated cells: fewer than 3 of 9 neighbours share their use.
            isolated = m & (neigh < threshold) & buildable
            for r, c in zip(*np.nonzero(isolated)):
                # Which use dominates this cell's surroundings?
                r0, r1 = max(0, r - 1), min(out.shape[0], r + 2)
                c0, c1 = max(0, c - 1), min(out.shape[1], c + 2)
                window = out[r0:r1, c0:c1]
                vals, counts = np.unique(window[window != UNASSIGNED], return_counts=True)
                if vals.size == 0:
                    continue
                dominant = int(vals[counts.argmax()])
                if dominant != u:
                    proposals.setdefault((int(u), dominant), []).append((r, c))

        # Execute only balanced exchanges, so no use gains or loses area.
        # Each unordered pair is handled exactly once, and a cell may move at
        # most once per pass: without both guards a cell can be reassigned twice
        # in one sweep and the area invariant silently drifts.
        swapped = False
        moved: set[tuple[int, int]] = set()
        for (a, b) in sorted({tuple(sorted(k)) for k in proposals}):
            cells_a = [c for c in proposals.get((a, b), []) if c not in moved]
            cells_b = [c for c in proposals.get((b, a), []) if c not in moved]
            n = min(len(cells_a), len(cells_b))
            if n == 0:
                continue
            for (r, c) in cells_a[:n]:
                out[r, c] = b
                moved.add((r, c))
            for (r, c) in cells_b[:n]:
                out[r, c] = a
                moved.add((r, c))
            swapped = True

        if not swapped:
            break

    return out
