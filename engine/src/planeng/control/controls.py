"""Development control.

Turns a cadastral layout into a plan a planning officer can assess. For every
plot it derives the building envelope the ruleset permits - what is left after
front, side and rear setbacks - and the coverage, floor area and height limits
that apply. It then checks the layout against the standards it was generated
from and reports what fails.

That last part matters more than it sounds. A generator that also marks its own
homework is worth little, so the checks here are written against the ruleset
directly rather than against the assumptions the generator made. When the two
disagree, that is a finding, not an error to be silenced.

Setbacks are applied by edge, not as a uniform inward buffer. A plot's front
setback is typically twice its side setback, so buffering uniformly by either
one misstates the envelope badly - and the envelope is the number a developer
actually builds to.
"""

from __future__ import annotations

from dataclasses import dataclass, field

from shapely.geometry import MultiLineString, Polygon
from shapely.ops import unary_union
from shapely.strtree import STRtree

from ..cadastre.parcels import CadastralLayout, Parcel
from ..geom import clean
from ..site.model import Site
from ..standards.loader import Standards

# How close a plot boundary must come to a road reserve to count as frontage.
FRONTAGE_SNAP_M = 2.5


@dataclass
class PlotControl:
    plot_number: str
    band: str
    plot_area_m2: float
    frontage_m: float
    envelope: Polygon | None
    envelope_area_m2: float
    max_footprint_m2: float
    max_floor_area_m2: float
    max_storeys: int
    violations: list[str] = field(default_factory=list)

    @property
    def compliant(self) -> bool:
        return not self.violations

    @property
    def buildable_pct(self) -> float:
        """Share of the plot inside the envelope, before coverage is applied."""
        if self.plot_area_m2 <= 0:
            return 0.0
        return self.envelope_area_m2 / self.plot_area_m2 * 100


@dataclass
class ControlReport:
    controls: list[PlotControl] = field(default_factory=list)
    conflicts: list[dict] = field(default_factory=list)
    warnings: list[str] = field(default_factory=list)

    @property
    def compliant_count(self) -> int:
        return sum(1 for c in self.controls if c.compliant)

    def violations_by_rule(self) -> dict[str, int]:
        out: dict[str, int] = {}
        for c in self.controls:
            for v in c.violations:
                rule = v.split(":", 1)[0]
                out[rule] = out.get(rule, 0) + 1
        return out

    def summary(self) -> dict:
        n = len(self.controls)
        by_band: dict[str, dict] = {}
        for c in self.controls:
            e = by_band.setdefault(c.band, {
                "plots": 0, "envelope_m2": 0.0, "footprint_m2": 0.0,
                "floor_area_m2": 0.0, "non_compliant": 0,
            })
            e["plots"] += 1
            e["envelope_m2"] += c.envelope_area_m2
            e["footprint_m2"] += c.max_footprint_m2
            e["floor_area_m2"] += c.max_floor_area_m2
            if not c.compliant:
                e["non_compliant"] += 1

        return {
            "plots_assessed": n,
            "compliant": self.compliant_count,
            "non_compliant": n - self.compliant_count,
            "compliance_pct": round(self.compliant_count / n * 100, 1) if n else 0.0,
            "violations_by_rule": self.violations_by_rule(),
            "land_use_conflicts": len(self.conflicts),
            "total_permitted_floor_area_m2": round(
                sum(c.max_floor_area_m2 for c in self.controls)),
            "by_band": [
                {
                    "band": b,
                    "plots": v["plots"],
                    "non_compliant": v["non_compliant"],
                    "mean_envelope_m2": round(v["envelope_m2"] / v["plots"], 1)
                    if v["plots"] else 0,
                    "max_footprint_ha": round(v["footprint_m2"] / 10_000, 2),
                    "permitted_floor_area_m2": round(v["floor_area_m2"]),
                }
                for b, v in sorted(by_band.items(), key=lambda kv: -kv[1]["plots"])
            ],
        }


def apply_controls(
    site: Site,
    std: Standards,
    cadastre: CadastralLayout,
    network,
    allocation=None,
) -> ControlReport:
    report = ControlReport()
    reserves = network.reserve_polygon if network is not None else None

    if (banner := std.provenance_warning()) is not None:
        report.warnings.append(banner)

    # Buffer the reserve footprint once. Doing it per plot means re-buffering a
    # site-wide multipolygon several thousand times, which took three minutes
    # for what is a fraction of a second's work.
    # The reserve footprint spans the whole site. Intersecting each plot
    # boundary against all of it is what made this stage slow; the parts are
    # indexed so a plot only meets the handful of reserve pieces near it.
    zone_parts: list = []
    if reserves is not None and not reserves.is_empty:
        buffered = reserves.buffer(FRONTAGE_SNAP_M)
        zone_parts = (list(buffered.geoms)
                      if buffered.geom_type == "MultiPolygon" else [buffered])
    tree = STRtree(zone_parts) if zone_parts else None

    for parcel in cadastre.parcels:
        report.controls.append(_control_for(parcel, std, tree, zone_parts))

    if allocation is not None:
        report.conflicts = _compatibility_conflicts(site, std, allocation)
        for c in report.conflicts:
            report.warnings.append(
                f"{c['a']} abuts {c['b']} over {c['shared_boundary_m']:,.0f} m "
                f"of shared boundary ({c['verdict']})."
            )

    non_compliant = len(report.controls) - report.compliant_count
    if non_compliant:
        report.warnings.append(
            f"{non_compliant:,} of {len(report.controls):,} plots breach at least "
            f"one development control. The layout must be corrected or the "
            f"standard varied before the plan is submitted."
        )

    return report


# ---------------------------------------------------------------------------

def _control_for(parcel: Parcel, std: Standards, tree, zone_parts) -> PlotControl:
    spec = std.get(f"residential.densities.{parcel.band}")
    area = parcel.area_m2
    violations: list[str] = []

    # -- plot dimension compliance ---------------------------------------
    if area < spec["plot_area_min_m2"]:
        violations.append(
            f"plot_area: {area:,.0f} m2 below the {spec['plot_area_min_m2']:,} m2 "
            f"minimum for {parcel.band} density")
    if area > spec["plot_area_max_m2"]:
        violations.append(
            f"plot_area: {area:,.0f} m2 above the {spec['plot_area_max_m2']:,} m2 "
            f"maximum for {parcel.band} density")
    if parcel.frontage_m < spec["frontage_min_m"]:
        violations.append(
            f"frontage: {parcel.frontage_m:,.1f} m below the "
            f"{spec['frontage_min_m']} m minimum")

    # -- building envelope ------------------------------------------------
    envelope = _envelope(parcel.geometry, spec, tree, zone_parts)
    env_area = envelope.area if envelope is not None and not envelope.is_empty else 0.0

    max_footprint = area * spec["coverage_max_pct"] / 100
    if env_area < max_footprint:
        # The setbacks bite before the coverage rule does, so the envelope is
        # the binding constraint. Report the number that actually applies.
        max_footprint = env_area
        if env_area <= 0:
            violations.append(
                "envelope: setbacks leave no buildable area on this plot")

    return PlotControl(
        plot_number=parcel.plot_number,
        band=parcel.band,
        plot_area_m2=round(area, 1),
        frontage_m=parcel.frontage_m,
        envelope=envelope,
        envelope_area_m2=round(env_area, 1),
        max_footprint_m2=round(max_footprint, 1),
        max_floor_area_m2=round(area * spec["far_max"], 1),
        max_storeys=spec["storeys_max"],
        violations=violations,
    )


def _envelope(plot: Polygon, spec: dict, tree, zone_parts) -> Polygon | None:
    """The area of a plot a building may occupy, after setbacks.

    The front setback is applied separately from the rest. It is typically
    twice the side setback, so folding it into a single inward buffer misstates
    the envelope by tens of square metres per plot and compounds into a badly
    wrong floor-area capacity for the settlement.

    Side and rear are merged to the stricter of the two rather than classified
    per edge. Distinguishing them costs three extra overlay operations per plot
    - eight seconds across a settlement - to move a boundary by half a metre.
    Merging is conservative: it never reports more envelope than the ruleset
    allows. Per-edge classification is worth doing when plot-level output is
    needed for a specific site, not for a settlement-wide capacity figure.
    """
    plot = clean(plot)
    if plot.is_empty or plot.geom_type != "Polygon":
        return None

    front = spec["setback_front_m"]
    side = spec["setback_side_m"]
    rear = spec["setback_rear_m"]

    boundary = plot.exterior
    frontage = None
    if tree is not None:
        hits = tree.query(boundary)
        if len(hits):
            nearby = unary_union([zone_parts[i] for i in hits])
            near = boundary.intersection(nearby)
            if not near.is_empty and near.geom_type in ("LineString", "MultiLineString"):
                frontage = near

    if frontage is None or frontage.length < 1e-6:
        # No identifiable frontage: fall back to the most restrictive setback
        # rather than quietly assuming the smallest.
        env = plot.buffer(-max(front, side, rear))
        return _largest(env)

    env = plot.buffer(-max(side, rear))
    if env.is_empty:
        return None
    # buffer() has already pulled the frontage back by the side setback, so cut
    # the remaining difference rather than the whole front setback again.
    extra = front - max(side, rear)
    if extra > 0:
        env = env.difference(frontage.buffer(front))

    return _largest(clean(env))


def _largest(geom) -> Polygon | None:
    if geom is None or geom.is_empty:
        return None
    if geom.geom_type == "Polygon":
        return geom
    parts = [g for g in getattr(geom, "geoms", []) if g.geom_type == "Polygon"]
    return max(parts, key=lambda g: g.area) if parts else None


def _compatibility_conflicts(site: Site, std: Standards, allocation) -> list[dict]:
    """Find incompatible land uses sharing a boundary.

    The compatibility matrix is a planning instrument: residential next to
    heavy industry is a defect regardless of how well each zone scores on its
    own suitability surface, and the allocator optimises each use separately so
    it cannot see this.
    """
    matrix = std.get("compatibility", {})
    if not matrix:
        return []

    from ..geom import polygonise

    shapes = {
        use: polygonise(allocation.mask_for(use), site, min_area_m2=2_000)
        for use in allocation.uses
    }

    out: list[dict] = []
    seen: set[tuple[str, str]] = set()

    for use_a, rules in matrix.items():
        if use_a not in shapes or not isinstance(rules, dict):
            continue
        for use_b, verdict in rules.items():
            if use_b not in shapes or verdict == "compatible":
                continue
            key = tuple(sorted((use_a, use_b)))
            if key in seen:
                continue
            seen.add(key)

            shared = 0.0
            for pa in shapes[use_a]:
                grown = pa.buffer(site.cell_size)
                for pb in shapes[use_b]:
                    if grown.intersects(pb):
                        shared += grown.intersection(pb.exterior).length

            if shared > site.cell_size * 4:
                out.append({
                    "a": use_a, "b": use_b, "verdict": verdict,
                    "shared_boundary_m": round(shared, 1),
                    "rule": f"compatibility.{use_a}.{use_b} = {verdict}",
                })

    return out
