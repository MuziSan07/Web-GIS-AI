"""Facility siting.

Turns the facility schedule into located sites. A schedule that says "6 primary
schools" is a budget line; a plan needs to say where they go, and whether the
children who must walk to them actually live within walking distance.

Placement is greedy maximum-coverage against residential demand: each facility
goes where it serves the most people not already served by one of its own kind.
That is the standard formulation for this problem, it is explainable to a
planner site by site, and it produces the coverage statistic that matters -
the share of the population inside the catchment distance the ruleset requires.

Coverage is reported per facility type. A plan that provides the right number
of clinics but puts them all in one corner is a failure the count alone hides.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field

import numpy as np

from shapely import affinity
from shapely.geometry import Point, Polygon, box

from ..allocate.zones import AllocationResult
from ..brief.derive import Programme
from ..geom import clean, safe_intersection
from ..roads.blocks import Block, BlockLayout
from ..site.model import Site
from ..standards.loader import Standards

# Where each category of facility belongs. Falls back to any non-residential
# land when the preferred zone has no room left.
CATEGORY_ZONE = {
    "education": "institutional",
    "health": "institutional",
    "safety": "institutional",
    "community": "institutional",
    "civic": "institutional",
    "recreation": "recreation_open_space",
    "utilities": "institutional",
}

# Facilities that belong on the edge of the settlement rather than among it.
PERIPHERAL = {"cemetery"}

# Same-type facilities closer than this fraction of their catchment are
# effectively serving one place twice.
MIN_SEPARATION_FRACTION = 0.55


@dataclass
class SitedFacility:
    geometry: Polygon
    key: str
    category: str
    facility_id: str
    catchment_m: float

    @property
    def area_ha(self) -> float:
        return self.geometry.area / 10_000


@dataclass
class FacilityPlan:
    facilities: list[SitedFacility] = field(default_factory=list)
    coverage: dict[str, float] = field(default_factory=dict)
    unplaced: dict[str, int] = field(default_factory=dict)
    undersized: dict[str, int] = field(default_factory=dict)
    warnings: list[str] = field(default_factory=list)

    def of_type(self, key: str) -> list[SitedFacility]:
        return [f for f in self.facilities if f.key == key]

    def summary(self) -> dict:
        by_key: dict[str, dict] = {}
        for f in self.facilities:
            e = by_key.setdefault(f.key, {
                "category": f.category, "sited": 0, "area_ha": 0.0,
                "catchment_m": f.catchment_m,
            })
            e["sited"] += 1
            e["area_ha"] += f.area_ha
        return {
            "sited_total": len(self.facilities),
            "unplaced_total": sum(self.unplaced.values()),
            "by_type": [
                {
                    "key": k,
                    "category": v["category"],
                    "sited": v["sited"],
                    "unplaced": self.unplaced.get(k, 0),
                    "area_ha": round(v["area_ha"], 2),
                    "catchment_m": v["catchment_m"],
                    "undersized": self.undersized.get(k, 0),
                    # A facility with no catchment (a substation) has no
                    # coverage to report. Zero would read as a failure.
                    "coverage_pct": (round(self.coverage[k], 1)
                                     if k in self.coverage else None),
                }
                for k, v in sorted(by_key.items(),
                                   key=lambda kv: (kv[1]["category"], kv[0]))
            ],
        }


def site_facilities(
    site: Site,
    std: Standards,
    programme: Programme,
    blocks: BlockLayout,
    allocation: AllocationResult,
) -> FacilityPlan:
    plan = FacilityPlan()

    demand = _demand_points(blocks)
    if demand[2].size == 0:
        plan.warnings.append("No residential demand; facilities were not sited.")
        return plan

    # Remaining developable area per block, consumed as facilities are placed.
    available: dict[str, Polygon] = {
        b.block_id: b.geometry for b in blocks.blocks if b.use != "residential"
    }
    block_use = {b.block_id: b.use for b in blocks.blocks}
    centre = site.boundary.centroid

    # Large facilities first: a 5 ha secondary school has few candidate sites,
    # a 0.2 ha health post has many. Placing small ones first fragments the
    # large sites out of existence.
    requirements = sorted(programme.facilities,
                          key=lambda f: -f.site_area_ha)

    counter = 0
    for req in requirements:
        if req.site_area_ha <= 0 or req.count <= 0:
            continue
        preferred = CATEGORY_ZONE.get(req.category, "institutional")
        placed: list[SitedFacility] = []

        for _ in range(req.count):
            choice = _best_site(
                req, preferred, available, block_use, demand, placed,
                centre, site,
            )
            if choice is None:
                plan.unplaced[req.key] = plan.unplaced.get(req.key, 0) + 1
                continue

            block_id, geom = choice
            if geom.area < req.site_area_ha * 10_000 * 0.9:
                plan.undersized[req.key] = plan.undersized.get(req.key, 0) + 1
            counter += 1
            facility = SitedFacility(
                geometry=geom, key=req.key, category=req.category,
                facility_id=f"F{counter:04d}", catchment_m=req.catchment_m,
            )
            placed.append(facility)
            plan.facilities.append(facility)

            remainder = clean(available[block_id].difference(geom))
            if remainder.is_empty or remainder.area < 200:
                available.pop(block_id, None)
            else:
                available[block_id] = _largest_polygon(remainder)

    _compute_coverage(plan, demand)

    for key, n in plan.unplaced.items():
        plan.warnings.append(
            f"{n} {key.replace('_', ' ')} site(s) could not be placed: no "
            f"remaining zoned land of sufficient size. Increase institutional "
            f"or open space allocation, or reduce the provision standard."
        )

    for key, n in plan.undersized.items():
        plan.warnings.append(
            f"{n} {key.replace('_', ' ')} site(s) were carved smaller than the "
            f"standard because no block could hold the full area. Confirm the "
            f"reduced site is workable or enlarge the host block."
        )

    for key, pct in sorted(plan.coverage.items()):
        if pct < 70.0:
            plan.warnings.append(
                f"Only {pct:.0f}% of dwellings fall within the "
                f"{key.replace('_', ' ')} catchment. Provision meets the "
                f"numeric standard but the distribution does not serve the "
                f"whole settlement."
            )

    return plan


# ---------------------------------------------------------------------------

def _demand_points(blocks: BlockLayout):
    """Residential demand, weighted by the plots each block holds.

    Held as arrays rather than geometries: siting evaluates every candidate
    block against every demand point, so this inner loop dominates runtime.
    """
    xs, ys, ws = [], [], []
    for b in blocks.for_use("residential"):
        if b.indicative_plots <= 0:
            continue
        c = b.geometry.centroid
        xs.append(c.x)
        ys.append(c.y)
        ws.append(b.indicative_plots)
    return (np.asarray(xs), np.asarray(ys), np.asarray(ws, dtype=float))


def _best_site(req, preferred: str, available: dict[str, Polygon],
               block_use: dict[str, str], demand, placed, centre, site):
    """Greedy maximum coverage: serve the most people not already served."""
    need_m2 = req.site_area_ha * 10_000
    catchment = req.catchment_m or 1500.0
    min_sep = catchment * MIN_SEPARATION_FRACTION

    scored: list[tuple[float, str, Polygon]] = []

    for block_id, geom in available.items():
        if geom.area < need_m2 * 0.95:
            continue

        use = block_use.get(block_id)
        # A facility on the wrong zone is allowed, but only as a last resort.
        zone_penalty = 0.0 if use == preferred else 0.35

        pt = geom.centroid
        dx, dy, dw = demand
        served = float(dw[np.hypot(dx - pt.x, dy - pt.y) <= catchment].sum())
        total = float(dw.sum()) or 1.0
        score = served / total * (1.0 - zone_penalty)

        if req.key in PERIPHERAL:
            # Cemeteries belong at the edge; invert the centrality preference.
            score = pt.distance(centre) / max(site.area_ha, 1)
        elif served == 0:
            # Nothing to serve here, but the facility still has to go somewhere.
            score = -pt.distance(centre) / 1e6

        scored.append((score, block_id, geom))

    # Separation is tested on the carved site, not on the host block. The two
    # have different centroids, and checking the block instead lets facilities
    # end up far closer together than the standard allows.
    for _, block_id, geom in sorted(scored, key=lambda t: -t[0]):
        carved = _carve(geom, need_m2)
        c = carved.centroid
        if placed and min(c.distance(f.geometry.centroid) for f in placed) < min_sep:
            continue
        return block_id, carved

    return None


def _carve(geom: Polygon, need_m2: float) -> Polygon:
    """Take a site of the required area from a block.

    Cut as a rectangle aligned to the block's own axis so facility sites read as
    erven rather than as circles stamped on the plan.
    """
    # Blocks arrive here after repeated difference operations as earlier
    # facilities were carved out of them, and those can leave self-touching
    # rings. Rotating and intersecting an invalid ring raises a GEOS side
    # location conflict far from the cause, so validate on entry.
    geom = _largest_polygon(clean(geom))
    if geom.is_empty or geom.geom_type != "Polygon":
        return geom

    if geom.area <= need_m2 * 1.25:
        return geom

    rect = geom.minimum_rotated_rectangle
    coords = list(rect.exterior.coords)[:4]
    edge = max(
        zip(coords, coords[1:] + coords[:1]),
        key=lambda pq: math.hypot(pq[1][0] - pq[0][0], pq[1][1] - pq[0][1]),
    )
    angle = math.degrees(math.atan2(edge[1][1] - edge[0][1],
                                    edge[1][0] - edge[0][0]))

    origin = geom.centroid
    local = affinity.rotate(geom, -angle, origin=origin)
    minx, miny, maxx, maxy = local.bounds
    width = maxx - minx
    if width <= 0:
        return geom

    # Grow a strip from one end until it holds the required area.
    height = min(maxy - miny, need_m2 / width)
    for _ in range(24):
        strip = safe_intersection(box(minx, miny, maxx, miny + height), local)
        if strip.area >= need_m2 or height >= (maxy - miny):
            break
        height *= 1.18

    strip = safe_intersection(
        box(minx, miny, maxx, miny + min(height, maxy - miny)), local)
    if strip.is_empty:
        return geom
    return _largest_polygon(affinity.rotate(strip, angle, origin=origin))


def _largest_polygon(geom) -> Polygon:
    if geom.geom_type == "Polygon":
        return geom
    if geom.geom_type == "MultiPolygon":
        return max(geom.geoms, key=lambda g: g.area)
    return geom


def _compute_coverage(plan: FacilityPlan, demand) -> None:
    """Share of dwellings within the catchment of each facility type.

    This is the number that tells a planner whether provision actually works.
    A plan with the right number of clinics all in one corner passes the count
    and fails here, which is the point.
    """
    dx, dy, dw = demand
    total = float(dw.sum()) or 1.0

    by_key: dict[str, list[SitedFacility]] = {}
    for f in plan.facilities:
        by_key.setdefault(f.key, []).append(f)

    for key, facs in by_key.items():
        catchment = facs[0].catchment_m
        if not catchment:
            continue          # no catchment: coverage is not a meaningful metric
        best = np.full(dx.shape, np.inf)
        for f in facs:
            c = f.geometry.centroid
            np.minimum(best, np.hypot(dx - c.x, dy - c.y), out=best)
        plan.coverage[key] = float(dw[best <= catchment].sum()) / total * 100
