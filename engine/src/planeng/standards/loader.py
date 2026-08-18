"""Planning standards rules engine.

Loads a jurisdiction ruleset from YAML and exposes it to the engine. Three
properties matter here and are enforced rather than assumed:

1. **Provenance.** Every rule carries a `source`. Rules still marked PLACEHOLDER
   are collected and surfaced in every report the engine produces, so an
   unvalidated number can never quietly present itself as a statutory standard.

2. **Internal consistency.** Density mixes and land budget shares are checked on
   load. A ruleset that does not add up fails immediately, not three modules later.

3. **Data, not code.** A planner amends the YAML. No developer is required to
   change a setback or a provision ratio.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import yaml

PLACEHOLDER = "PLACEHOLDER"
_BUNDLED = Path(__file__).parent


@dataclass
class RulesetIssue:
    """A consistency problem or provenance gap found in a ruleset."""

    severity: str  # "error" | "warning"
    path: str
    message: str

    def __str__(self) -> str:
        return f"[{self.severity.upper()}] {self.path}: {self.message}"


@dataclass
class Standards:
    """A loaded, validated planning ruleset."""

    data: dict[str, Any]
    path: Path
    issues: list[RulesetIssue] = field(default_factory=list)

    # -- access ------------------------------------------------------------
    def get(self, dotted: str, default: Any = None) -> Any:
        """Fetch a rule by dotted path, e.g. ``roads.hierarchy.collector.reserve_width_m``."""
        node: Any = self.data
        for part in dotted.split("."):
            if not isinstance(node, dict) or part not in node:
                if default is not None:
                    return default
                raise KeyError(f"No such rule: '{dotted}' (failed at '{part}')")
            node = node[part]
        # Rules written as {value:, source:} collapse to their value.
        if isinstance(node, dict) and "value" in node and "source" in node:
            return node["value"]
        return node

    def require(self, dotted: str) -> Any:
        return self.get(dotted)

    # -- provenance --------------------------------------------------------
    @property
    def is_validated(self) -> bool:
        return self.data.get("meta", {}).get("status") == "validated"

    @property
    def jurisdiction(self) -> str:
        return self.data.get("meta", {}).get("jurisdiction", "unknown")

    def placeholder_paths(self) -> list[str]:
        """Every rule path whose source is still PLACEHOLDER."""
        found: list[str] = []
        _walk_sources(self.data, "", found)
        return sorted(found)

    def provenance_warning(self) -> str | None:
        """The banner that must appear on every output produced from this ruleset."""
        n = len(self.placeholder_paths())
        if self.is_validated and n == 0:
            return None
        return (
            f"UNVALIDATED RULESET: {n} standard(s) in the {self.jurisdiction} "
            f"ruleset are placeholders awaiting confirmation by a Registered "
            f"Town Planner. Output is indicative and must not be presented as "
            f"evidence of statutory compliance."
        )

    @property
    def errors(self) -> list[RulesetIssue]:
        return [i for i in self.issues if i.severity == "error"]


def _walk_sources(node: Any, path: str, out: list[str]) -> None:
    if not isinstance(node, dict):
        return
    if node.get("source") == PLACEHOLDER:
        out.append(path or "<root>")
    for key, child in node.items():
        if key == "source":
            continue
        _walk_sources(child, f"{path}.{key}" if path else key, out)


# ---------------------------------------------------------------------------
# Validation
# ---------------------------------------------------------------------------

def _validate(data: dict[str, Any]) -> list[RulesetIssue]:
    issues: list[RulesetIssue] = []

    mix = data.get("residential", {}).get("default_density_mix", {})
    shares = {k: v for k, v in mix.items() if isinstance(v, (int, float))}
    if shares:
        total = sum(shares.values())
        if abs(total - 1.0) > 1e-6:
            issues.append(RulesetIssue(
                "error", "residential.default_density_mix",
                f"shares must sum to 1.0, got {total:.4f}",
            ))

    budget = data.get("land_budget", {})
    pct = {k: v for k, v in budget.items() if k.endswith("_pct")}
    if pct:
        total = sum(pct.values())
        if abs(total - 100.0) > 1e-6:
            issues.append(RulesetIssue(
                "error", "land_budget",
                f"shares must sum to 100, got {total}",
            ))

    # A density band whose minimum exceeds its maximum will silently produce
    # zero plots downstream, so catch it at load time.
    for band, spec in data.get("residential", {}).get("densities", {}).items():
        if not isinstance(spec, dict):
            continue
        lo, hi = spec.get("plot_area_min_m2"), spec.get("plot_area_max_m2")
        typ = spec.get("plot_area_typical_m2")
        if lo is not None and hi is not None and lo > hi:
            issues.append(RulesetIssue(
                "error", f"residential.densities.{band}",
                f"plot_area_min_m2 ({lo}) exceeds plot_area_max_m2 ({hi})",
            ))
        if typ is not None and lo is not None and hi is not None and not (lo <= typ <= hi):
            issues.append(RulesetIssue(
                "warning", f"residential.densities.{band}",
                f"plot_area_typical_m2 ({typ}) outside [{lo}, {hi}]",
            ))

    for name, spec in data.get("roads", {}).get("hierarchy", {}).items():
        if not isinstance(spec, dict):
            continue
        reserve, carriage = spec.get("reserve_width_m"), spec.get("carriageway_m")
        if reserve is not None and carriage is not None and carriage >= reserve:
            issues.append(RulesetIssue(
                "error", f"roads.hierarchy.{name}",
                f"carriageway ({carriage} m) must be narrower than the "
                f"reserve ({reserve} m) to leave room for verges and services",
            ))

    return issues


def load_standards(jurisdiction: str = "botswana", path: Path | None = None) -> Standards:
    """Load a ruleset, validate it, and refuse to return a structurally broken one."""
    src = path or (_BUNDLED / f"{jurisdiction.lower()}.yaml")
    if not src.exists():
        raise FileNotFoundError(f"No ruleset for '{jurisdiction}' at {src}")

    with open(src, "r", encoding="utf-8") as fh:
        data = yaml.safe_load(fh)

    issues = _validate(data)
    std = Standards(data=data, path=src, issues=issues)

    if std.errors:
        detail = "\n  ".join(str(e) for e in std.errors)
        raise ValueError(f"Ruleset '{src.name}' is internally inconsistent:\n  {detail}")

    return std
