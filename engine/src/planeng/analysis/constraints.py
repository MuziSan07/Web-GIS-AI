"""Constraint analysis: what land cannot be developed, and why.

Every excluded cell records the rule that excluded it. A planner asking "why is
this corner blank?" gets an auditable answer citing a ruleset path, which is the
difference between a plan that can be defended at a hearing and one that cannot.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

from ..site.model import Site
from ..standards.loader import Standards
from .terrain import slope_percent


@dataclass
class ConstraintResult:
    buildable: np.ndarray                    # bool, True where development is allowed
    exclusions: dict[str, np.ndarray] = field(default_factory=dict)
    reasons: dict[str, str] = field(default_factory=dict)
    slope: np.ndarray | None = None

    def area_ha(self, site: Site, key: str) -> float:
        return site.cells_to_ha(int((self.exclusions[key] & site.mask()).sum()))

    def summary(self, site: Site) -> list[dict]:
        """Per-constraint areas, for the report and the UI legend."""
        out = []
        for key, arr in self.exclusions.items():
            out.append({
                "constraint": key,
                "area_ha": round(self.area_ha(site, key), 2),
                "rule": self.reasons.get(key, ""),
            })
        return sorted(out, key=lambda d: -d["area_ha"])


def analyse_constraints(site: Site, std: Standards) -> ConstraintResult:
    """Apply every environmental and engineering exclusion from the ruleset."""
    mask = site.mask()
    res = ConstraintResult(buildable=mask.copy())

    def exclude(key: str, arr: np.ndarray, rule: str) -> None:
        arr = arr & mask
        res.exclusions[key] = arr
        res.reasons[key] = rule
        res.buildable &= ~arr

    # -- buffered vector constraints -------------------------------------
    buffers = {
        "wetland": ("environment.buffers.wetland_m", "wetland"),
        "watercourse": ("environment.buffers.watercourse_m", "watercourse"),
        "protected_area": ("environment.buffers.protected_area_m", "protected_area"),
        "powerline": ("environment.buffers.powerline_servitude_m", "powerline"),
    }
    for layer_name, (rule_path, _) in buffers.items():
        geom = site.layers.get(layer_name)
        if geom is None:
            continue
        distance = std.get(rule_path)
        # Servitudes are measured either side of the centreline, so a linear
        # feature is buffered by half the total reserve width.
        if layer_name == "powerline":
            distance = distance / 2
        buffered = geom.buffer(distance)
        exclude(layer_name, site.rasterize(buffered),
                f"{rule_path} = {std.get(rule_path)} m")

    # -- slope ------------------------------------------------------------
    slope = slope_percent(site)
    res.slope = slope
    limit = std.get("environment.slope.undevelopable_above_pct")
    exclude("steep_slope", slope > limit,
            f"environment.slope.undevelopable_above_pct = {limit} %")

    return res


def buildable_area_ha(site: Site, res: ConstraintResult) -> float:
    return site.cells_to_ha(int(res.buildable.sum()))
