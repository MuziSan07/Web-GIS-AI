"""Least-cost routing over the site grid.

Road alignments are found by shortest path on a cost surface rather than drawn
as straight lines, so the network responds to terrain and constraints the way a
road engineer would: it climbs gently, avoids wetlands, and crosses the valley
where crossing is cheapest.

The cost surface is the whole design argument, so it is built explicitly and is
inspectable rather than hidden inside the router.
"""

from __future__ import annotations

import numpy as np
from scipy.sparse import coo_matrix
from scipy.sparse.csgraph import dijkstra

from ..site.model import Site
from ..standards.loader import Standards

# Eight-connected movement. Diagonals cost sqrt(2) times a cardinal step.
_OFFSETS = [(-1, 0, 1.0), (1, 0, 1.0), (0, -1, 1.0), (0, 1, 1.0),
            (-1, -1, 2 ** 0.5), (-1, 1, 2 ** 0.5),
            (1, -1, 2 ** 0.5), (1, 1, 2 ** 0.5)]

BLOCKED = np.inf

# Constraints a road may cross with an engineered structure. A watercourse gets
# a culvert or bridge; a road can pass beneath a powerline servitude. These are
# costed, not forbidden - treating them as impassable fragments the site into
# islands and produces a plan that cannot actually be built out.
CROSSABLE = {"watercourse", "powerline"}

# Cost multiplier applied where a crossing structure would be required. High
# enough that the router crosses once, at the narrowest point, rather than
# meandering along the barrier.
CROSSING_COST = 70.0


def build_cost_surface(
    site: Site,
    std: Standards,
    constraints,
    road_class: str = "collector",
) -> np.ndarray:
    """Per-cell cost of putting road here, in nominal units per metre.

    Cost rises steeply as slope approaches the class gradient limit and becomes
    prohibitive beyond it, because a road that exceeds its design gradient is
    not a cheaper road, it is a non-compliant one.
    """
    slope = constraints.slope
    max_grad = std.get(f"roads.hierarchy.{road_class}.max_gradient_pct")

    ratio = np.clip(slope / max_grad, 0, None)
    cost = 1.0 + 4.0 * ratio ** 2
    cost[slope > max_grad] = 40.0            # steeply penalised, not forbidden

    # Crossable constraints are priced first, then hard constraints close over
    # the top. Order matters: a cell that is both a watercourse and a wetland is
    # a wetland, and must stay impassable regardless of iteration order.
    site_mask = site.mask()
    for key, arr in constraints.exclusions.items():
        if key in CROSSABLE:
            cost[arr & site_mask] = CROSSING_COST
    for key, arr in constraints.exclusions.items():
        if key not in CROSSABLE:
            cost[arr] = BLOCKED
    cost[~site_mask] = BLOCKED
    return cost


class GridRouter:
    """Dijkstra over the site grid, with the cost surface baked in."""

    def __init__(self, site: Site, cost: np.ndarray):
        self.site = site
        self.cost = cost
        self.shape = site.shape
        self._graph = self._build(cost, site.cell_size)

    @staticmethod
    def _build(cost: np.ndarray, cell_size: float):
        rows, cols = cost.shape
        n = rows * cols
        idx = np.arange(n).reshape(rows, cols)

        src_all, dst_all, w_all = [], [], []
        for dr, dc, step in _OFFSETS:
            r0, r1 = max(0, -dr), rows - max(0, dr)
            c0, c1 = max(0, -dc), cols - max(0, dc)
            src = idx[r0:r1, c0:c1]
            dst = idx[r0 + dr:r1 + dr, c0 + dc:c1 + dc]
            cs = cost[r0:r1, c0:c1]
            cd = cost[r0 + dr:r1 + dr, c0 + dc:c1 + dc]

            ok = np.isfinite(cs) & np.isfinite(cd)
            if not ok.any():
                continue
            # Trapezoidal cost across the step.
            w = (cs[ok] + cd[ok]) / 2.0 * step * cell_size
            src_all.append(src[ok])
            dst_all.append(dst[ok])
            w_all.append(w)

        if not src_all:
            raise ValueError("Cost surface leaves no traversable cells.")

        return coo_matrix(
            (np.concatenate(w_all),
             (np.concatenate(src_all), np.concatenate(dst_all))),
            shape=(n, n),
        ).tocsr()

    # -- interface --------------------------------------------------------
    def _flat(self, rc: tuple[int, int]) -> int:
        return rc[0] * self.shape[1] + rc[1]

    def _rc(self, flat: int) -> tuple[int, int]:
        return divmod(int(flat), self.shape[1])

    def solve_from(self, start: tuple[int, int]):
        """Single-source shortest paths. One call serves every destination."""
        dist, pred = dijkstra(
            self._graph, directed=False, indices=self._flat(start),
            return_predecessors=True,
        )
        return dist, pred

    def solve_from_cells(self, mask: np.ndarray):
        """Shortest paths from anywhere on a mask, in one pass.

        Used when the source is a whole existing network rather than a point:
        the cheapest link to "the road system" is wanted, not the cheapest link
        to one arbitrary manhole on it. `min_only` collapses every source into a
        single frontier, so this costs one Dijkstra rather than one per source.
        """
        starts = np.flatnonzero(mask.ravel())
        if starts.size == 0:
            raise ValueError("No source cells to route from.")
        dist, pred, _ = dijkstra(
            self._graph, directed=False, indices=starts,
            return_predecessors=True, min_only=True,
        )
        return dist, pred

    def trace(self, pred: np.ndarray, end: tuple[int, int]) -> list[tuple[int, int]]:
        """Walk the predecessor array back from a destination to the source."""
        cur = self._flat(end)
        cells: list[tuple[int, int]] = []
        guard = self.shape[0] * self.shape[1] + 5
        while cur >= 0 and guard > 0:
            cells.append(self._rc(cur))
            cur = int(pred[cur])
            guard -= 1
        if len(cells) < 2:
            return []
        return list(reversed(cells))

    def reachable(self, dist: np.ndarray, rc: tuple[int, int]) -> bool:
        return np.isfinite(dist[self._flat(rc)])


def nearest_valid_cell(
    target: tuple[int, int], valid: np.ndarray, max_radius: int = 40
) -> tuple[int, int] | None:
    """Snap a desired location onto the nearest usable cell.

    Zone centroids routinely fall on constrained land or just outside the site;
    snapping keeps routing robust instead of failing on an unlucky centroid.
    """
    r, c = target
    rows, cols = valid.shape
    if 0 <= r < rows and 0 <= c < cols and valid[r, c]:
        return (r, c)
    for rad in range(1, max_radius + 1):
        r0, r1 = max(0, r - rad), min(rows, r + rad + 1)
        c0, c1 = max(0, c - rad), min(cols, c + rad + 1)
        window = valid[r0:r1, c0:c1]
        if not window.any():
            continue
        rr, cc = np.nonzero(window)
        d = (rr + r0 - r) ** 2 + (cc + c0 - c) ** 2
        k = int(np.argmin(d))
        return (int(rr[k] + r0), int(cc[k] + c0))
    return None
