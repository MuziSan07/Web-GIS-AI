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
    compactness_passes: int = 3,
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

    # -- greedy assignment ------------------------------------------------
    # Rank every (cell, use) pair by suitability and take them in order. A cell
    # is claimed once; a use stops claiming when its quota is spent.
    idx = np.flatnonzero(buildable.ravel())
    scores = np.stack([suitability.surfaces[u].ravel()[idx] for u in uses])  # (u, n)

    best_use = scores.argmax(axis=0)
    best_score = scores.max(axis=0)
    order = np.argsort(-best_score)          # strongest matches settle first

    remaining = {i: quotas[u] for i, u in enumerate(uses)}
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

    labels.ravel()[idx] = flat_labels

    # -- compactness ------------------------------------------------------
    labels = _compact(labels, buildable, passes=compactness_passes)

    # -- report -----------------------------------------------------------
    res = AllocationResult(labels=labels, uses=uses)
    for i, use in enumerate(uses):
        m = labels == i
        res.achieved_ha[use] = site.cells_to_ha(int(m.sum()))
        res.target_ha[use] = targets_ha[use] * scale
        res.mean_suitability[use] = (
            float(suitability.surfaces[use][m].mean()) if m.any() else 0.0
        )
    return res


def _compact(labels: np.ndarray, buildable: np.ndarray, passes: int) -> np.ndarray:
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
            isolated = m & (neigh < 3 / 9) & buildable
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
