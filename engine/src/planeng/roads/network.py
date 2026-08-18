"""Road hierarchy generation.

Builds a four-tier network the way a road engineer would sequence it:

1. Connect to the external road, because a settlement with no external
   connection is not a settlement.
2. A primary spine through the commercial centre, following the cheapest
   terrain rather than a straight line.
3. District distributors from the spine to each major land use, honouring the
   junction spacing standard so the spine keeps its traffic function.
4. Local access streets forming blocks inside residential land.

Each tier is buffered to its reserve width from the ruleset. Reserves are what
subsequent modules subtract to obtain development blocks and, later, parcels.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
from scipy import ndimage
from shapely.geometry import LineString, MultiLineString, Point
from shapely.ops import unary_union

from ..allocate.zones import AllocationResult
from ..analysis.constraints import ConstraintResult
from ..geom import cells_to_line, polygonise
from ..site.model import Site
from ..standards.loader import Standards
from .routing import GridRouter, build_cost_surface, nearest_valid_cell

# Which land uses earn a district distributor of their own.
DISTRIBUTOR_USES = ("commercial", "industrial", "institutional")

# Ignore zone fragments below this when deciding what needs its own road.
MIN_PATCH_HA = 2.0
MAX_DISTRIBUTORS = 12


@dataclass
class RoadSegment:
    geometry: LineString
    road_class: str
    reserve_width_m: float

    @property
    def length_m(self) -> float:
        return self.geometry.length

    @property
    def reserve_area_m2(self) -> float:
        return self.length_m * self.reserve_width_m


@dataclass
class RoadNetwork:
    segments: list[RoadSegment] = field(default_factory=list)
    junctions: list[Point] = field(default_factory=list)
    reserve_polygon: object = None
    warnings: list[str] = field(default_factory=list)

    def by_class(self, road_class: str) -> list[RoadSegment]:
        return [s for s in self.segments if s.road_class == road_class]

    def length_by_class(self) -> dict[str, float]:
        out: dict[str, float] = {}
        for s in self.segments:
            out[s.road_class] = out.get(s.road_class, 0.0) + s.length_m
        return out

    @property
    def total_length_m(self) -> float:
        return sum(s.length_m for s in self.segments)

    @property
    def reserve_area_ha(self) -> float:
        return self.reserve_polygon.area / 10_000 if self.reserve_polygon else 0.0

    def summary(self, site: Site) -> dict:
        lengths = self.length_by_class()
        return {
            "total_length_km": round(self.total_length_m / 1000, 2),
            "reserve_area_ha": round(self.reserve_area_ha, 2),
            "reserve_pct_of_site": round(self.reserve_area_ha / site.area_ha * 100, 1),
            "junctions": len(self.junctions),
            "by_class": [
                {
                    "road_class": k,
                    "length_km": round(v / 1000, 2),
                    "segments": len(self.by_class(k)),
                }
                for k, v in sorted(lengths.items(), key=lambda kv: -kv[1])
            ],
        }


def generate_network(
    site: Site,
    std: Standards,
    constraints: ConstraintResult,
    allocation: AllocationResult,
    programme=None,
) -> RoadNetwork:
    net = RoadNetwork()
    slope = constraints.slope
    buildable = constraints.buildable

    reserve = {c: std.get(f"roads.hierarchy.{c}.reserve_width_m")
               for c in ("primary_arterial", "district_distributor",
                         "collector", "local_access")}

    # Occupied by the network so far, used to attach new branches.
    on_network = np.zeros(site.shape, dtype=bool)

    # -- 1. external connection ------------------------------------------
    access = _access_cell(site, buildable)
    if access is None:
        net.warnings.append(
            "No usable connection to the existing external road was found; "
            "the network was seeded from the most accessible site edge instead."
        )
        access = _fallback_access(site, buildable)
    if access is None:
        net.warnings.append("Site has no buildable land; no network generated.")
        return net

    # -- 2. primary spine -------------------------------------------------
    cost_primary = build_cost_surface(site, std, constraints, "primary_arterial")
    router = GridRouter(site, cost_primary)
    dist, pred = router.solve_from(access)

    cbd = _patch_centroid_cell(site, allocation, "commercial", buildable)
    spine_cells: list[tuple[int, int]] = []

    if cbd is not None and router.reachable(dist, cbd):
        spine_cells = router.trace(pred, cbd)
        # Continue past the centre to the far side so the spine crosses the site.
        far = _farthest_reachable(dist, buildable, router)
        if far is not None:
            d2, p2 = router.solve_from(cbd)
            tail = router.trace(p2, far)
            if len(tail) > 1:
                spine_cells = spine_cells + tail[1:]
    else:
        far = _farthest_reachable(dist, buildable, router)
        if far is not None:
            spine_cells = router.trace(pred, far)

    if len(spine_cells) < 2:
        net.warnings.append("Could not route a primary spine across the site.")
        return net

    _add(net, site, spine_cells, "primary_arterial", reserve["primary_arterial"],
         on_network)

    # -- 3. district distributors ----------------------------------------
    min_spacing = std.get("roads.hierarchy.primary_arterial.junction_spacing_min_m")
    cost_dist = build_cost_surface(site, std, constraints, "district_distributor")
    router_d = GridRouter(site, cost_dist)

    for use, cell in _distributor_targets(site, allocation, buildable):
        d, p = router_d.solve_from(cell)
        attach = _best_attachment(d, on_network, net.junctions, site, min_spacing)
        if attach is None:
            continue
        cells = router_d.trace(p, attach)
        if len(cells) < 2:
            continue
        _add(net, site, cells, "district_distributor",
             reserve["district_distributor"], on_network)
        net.junctions.append(Point(*(site.transform * (attach[1] + 0.5, attach[0] + 0.5))))

    # -- 4. collectors into residential neighbourhoods --------------------
    collector_spacing = std.get("roads.hierarchy.district_distributor.junction_spacing_min_m")
    for cell in _residential_targets(site, allocation, buildable):
        d, p = router_d.solve_from(cell)
        attach = _best_attachment(d, on_network, net.junctions, site, collector_spacing)
        if attach is None:
            continue
        cells = router_d.trace(p, attach)
        if len(cells) < 2:
            continue
        _add(net, site, cells, "collector", reserve["collector"], on_network)

    # -- 5. local access grid --------------------------------------------
    _add_local_grid(net, site, std, allocation, constraints,
                    reserve["local_access"], programme)

    # -- 6. connectivity check -------------------------------------------
    _check_connectivity(net, site, buildable, np.isfinite(cost_dist), access)

    # -- reserves ---------------------------------------------------------
    net.reserve_polygon = unary_union(
        [s.geometry.buffer(s.reserve_width_m / 2, cap_style=2) for s in net.segments]
    ).intersection(site.boundary)

    return net


# ---------------------------------------------------------------------------
# helpers
# ---------------------------------------------------------------------------

def _add(net: RoadNetwork, site: Site, cells, road_class: str,
         width: float, on_network: np.ndarray) -> None:
    line = cells_to_line(cells, site)
    if line is None or line.length < site.cell_size:
        return
    net.segments.append(RoadSegment(line, road_class, width))
    for r, c in cells:
        on_network[r, c] = True


def _access_cell(site: Site, buildable: np.ndarray) -> tuple[int, int] | None:
    """Where the existing external road meets the site."""
    road = site.layers.get("existing_road")
    if road is None:
        return None
    touch = road.intersection(site.boundary.buffer(site.cell_size * 2))
    if touch.is_empty:
        return None
    pt = touch.centroid if touch.geom_type != "Point" else touch
    inv = ~site.transform
    c, r = inv * (pt.x, pt.y)
    return nearest_valid_cell((int(r), int(c)), buildable)


def _fallback_access(site: Site, buildable: np.ndarray) -> tuple[int, int] | None:
    idx = np.flatnonzero(buildable.ravel())
    if idx.size == 0:
        return None
    r, c = divmod(int(idx[0]), site.shape[1])
    return (r, c)


def _farthest_reachable(dist: np.ndarray, buildable: np.ndarray,
                        router: GridRouter) -> tuple[int, int] | None:
    """The reachable buildable cell costliest to get to: the far side of the site."""
    d = dist.reshape(buildable.shape).copy()
    d[~buildable] = -np.inf
    d[~np.isfinite(d)] = -np.inf
    if not np.isfinite(d).any() or d.max() <= 0:
        return None
    return tuple(int(v) for v in np.unravel_index(int(np.argmax(d)), d.shape))


def _patches(site: Site, allocation: AllocationResult, use: str,
             min_ha: float = MIN_PATCH_HA):
    """Connected components of a land use, largest first."""
    if use not in allocation.uses:
        return []
    m = allocation.mask_for(use)
    lab, n = ndimage.label(m)
    out = []
    for i in range(1, n + 1):
        size = int((lab == i).sum())
        if site.cells_to_ha(size) >= min_ha:
            out.append((size, i, lab))
    out.sort(key=lambda t: -t[0])
    return out


def _patch_centroid_cell(site: Site, allocation: AllocationResult, use: str,
                         buildable: np.ndarray) -> tuple[int, int] | None:
    patches = _patches(site, allocation, use)
    if not patches:
        return None
    _, i, lab = patches[0]
    rr, cc = np.nonzero(lab == i)
    return nearest_valid_cell((int(rr.mean()), int(cc.mean())), buildable)


def _distributor_targets(site: Site, allocation: AllocationResult,
                         buildable: np.ndarray):
    """One target per significant non-residential zone patch."""
    targets = []
    for use in DISTRIBUTOR_USES:
        for _, i, lab in _patches(site, allocation, use)[:4]:
            rr, cc = np.nonzero(lab == i)
            cell = nearest_valid_cell((int(rr.mean()), int(cc.mean())), buildable)
            if cell is not None:
                targets.append((use, cell))
    return targets[:MAX_DISTRIBUTORS]


def _residential_targets(site: Site, allocation: AllocationResult,
                         buildable: np.ndarray):
    out = []
    for _, i, lab in _patches(site, allocation, "residential", min_ha=8.0)[:8]:
        rr, cc = np.nonzero(lab == i)
        cell = nearest_valid_cell((int(rr.mean()), int(cc.mean())), buildable)
        if cell is not None:
            out.append(cell)
    return out


def _best_attachment(dist: np.ndarray, on_network: np.ndarray,
                     junctions: list[Point], site: Site,
                     min_spacing_m: float) -> tuple[int, int] | None:
    """Cheapest point to join the existing network that respects junction spacing.

    Junction spacing is a real standard, not a nicety: distributors joining an
    arterial every 50 m destroy the arterial's traffic function.
    """
    if not on_network.any():
        return None
    d = dist.reshape(on_network.shape).copy()
    d[~on_network] = np.inf
    d[~np.isfinite(d)] = np.inf
    if not np.isfinite(d).any():
        return None

    order = np.argsort(d, axis=None)
    for flat in order[:4000]:
        if not np.isfinite(d.ravel()[flat]):
            break
        r, c = np.unravel_index(int(flat), d.shape)
        x, y = site.transform * (c + 0.5, r + 0.5)
        p = Point(x, y)
        if all(p.distance(j) >= min_spacing_m for j in junctions):
            return (int(r), int(c))
    return None


def _add_local_grid(net: RoadNetwork, site: Site, std: Standards,
                    allocation: AllocationResult, constraints: ConstraintResult,
                    width: float, programme=None) -> None:
    """Lay a street grid inside residential land to form blocks.

    Street spacing is driven by the plot depth the brief actually calls for, not
    by a single block-depth constant. The constant assumes low-density plots;
    applying it to a high-density brief produces blocks far deeper than two rows
    of plots, so the interiors cannot be reached and the layout under-delivers
    against the dwelling schedule.
    """
    if "residential" not in allocation.uses:
        return

    block_len = std.get("roads.block.length_typical_m")
    block_depth = _target_plot_depth(std, programme)
    # Two plot depths back to back, plus the street between them.
    cross_spacing = 2 * block_depth + width

    bearing = _spine_bearing(net)
    cos_t, sin_t = np.cos(bearing), np.sin(bearing)

    existing = unary_union(
        [s.geometry.buffer(s.reserve_width_m / 2, cap_style=2) for s in net.segments]
    ) if net.segments else None

    for poly in polygonise(allocation.mask_for("residential"), site,
                           min_area_m2=block_len * cross_spacing):
        area = poly
        if existing is not None:
            area = area.difference(existing)
        if area.is_empty:
            continue

        minx, miny, maxx, maxy = poly.bounds
        cx, cy = (minx + maxx) / 2, (miny + maxy) / 2
        reach = float(np.hypot(maxx - minx, maxy - miny)) / 2 + block_len

        for spacing, along in ((cross_spacing, True), (block_len, False)):
            n = int(reach / spacing) + 1
            for k in range(-n, n + 1):
                off = k * spacing
                if along:
                    # Lines parallel to the spine bearing.
                    ax, ay = cx - reach * cos_t - off * sin_t, cy - reach * sin_t + off * cos_t
                    bx, by = cx + reach * cos_t - off * sin_t, cy + reach * sin_t + off * cos_t
                else:
                    ax, ay = cx - reach * -sin_t - off * cos_t, cy - reach * cos_t - off * sin_t
                    bx, by = cx + reach * -sin_t - off * cos_t, cy + reach * cos_t - off * sin_t

                seg = LineString([(ax, ay), (bx, by)]).intersection(area)
                if seg.is_empty:
                    continue
                # A street earns its place if it serves a handful of plots.
                # Filtering on the full block length instead drops streets in
                # narrow and irregular parts of the site, and those areas then
                # survive as blocks too deep to subdivide from their frontages.
                min_useful = max(25.0, cross_spacing * 0.5)
                parts = list(seg.geoms) if isinstance(seg, MultiLineString) else [seg]
                for part in parts:
                    if part.geom_type == "LineString" and part.length >= min_useful:
                        net.segments.append(RoadSegment(part, "local_access", width))


def _check_connectivity(net: RoadNetwork, site: Site, buildable: np.ndarray,
                        traversable: np.ndarray, access: tuple[int, int]) -> None:
    """Report developable land the road network cannot reach.

    Measured over the traversable surface, not the buildable one: a road may
    cross a watercourse or a servitude with a structure, so land beyond one is
    reachable even though it cannot be built on. Only land behind a genuinely
    impassable constraint - a wetland, a protected area - is stranded, and that
    is a re-designation decision a planner must take consciously rather than
    discover on site.
    """
    # Eight-connectivity, to match the router's movement model.
    struct = ndimage.generate_binary_structure(2, 2)
    lab, n = ndimage.label(traversable, structure=struct)
    if n <= 1:
        return

    main = lab[access]
    stranded = buildable & (lab != main)
    stranded_cells = int(stranded.sum())
    stranded_ha = site.cells_to_ha(stranded_cells)
    if stranded_ha < 1.0:
        return

    islands = sum(
        1 for i in range(1, n + 1)
        if i != main and site.cells_to_ha(int((buildable & (lab == i)).sum())) >= 1.0
    )
    net.warnings.append(
        f"{stranded_ha:,.1f} ha of developable land sits in {islands} area(s) "
        f"cut off from the road network by constraint buffers. Crossings or "
        f"culverts are required to develop it, or it should be re-designated."
    )


def _target_plot_depth(std: Standards, programme) -> float:
    """Area-weighted plot depth for this brief, in metres.

    Depth follows from the standard itself: a plot of the typical area with the
    minimum frontage is exactly this deep.
    """
    def depth_of(band: str) -> float:
        spec = std.get(f"residential.densities.{band}")
        return spec["plot_area_typical_m2"] / spec["frontage_min_m"]

    if programme is not None and getattr(programme, "dwellings", None):
        total = sum(d.dwellings for d in programme.dwellings)
        if total > 0:
            return sum(d.dwellings * depth_of(d.band)
                       for d in programme.dwellings) / total

    mix = {k: v for k, v in std.get("residential.default_density_mix").items()
           if isinstance(v, (int, float))}
    return sum(depth_of(band) * share for band, share in mix.items())


def _spine_bearing(net: RoadNetwork) -> float:
    """Orientation of the primary spine, so the local grid aligns with it."""
    primary = net.by_class("primary_arterial")
    if not primary:
        return 0.0
    line = max(primary, key=lambda s: s.length_m).geometry
    (x0, y0), (x1, y1) = line.coords[0], line.coords[-1]
    return float(np.arctan2(y1 - y0, x1 - x0))
