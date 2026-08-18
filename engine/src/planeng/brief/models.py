"""The structured planning brief.

This is the contract between the AI layer and the planning engine. A natural
language prompt is parsed into a PlanningBrief and nothing else; from this point
on the pipeline is fully deterministic.

Keeping this boundary sharp is what makes the system testable. The same brief
always produces the same plan, and a brief can be written by hand when a planner
wants precise control rather than a prompt.
"""

from __future__ import annotations

from enum import Enum

from pydantic import BaseModel, Field, field_validator, model_validator


class DensityBand(str, Enum):
    LOW = "low"
    MEDIUM = "medium"
    HIGH = "high"


class SettlementForm(str, Enum):
    """Structural strategies the optimiser can be asked to pursue."""

    COMPACT = "compact"
    LINEAR = "linear"
    CLUSTERED = "clustered"
    INFRASTRUCTURE_MINIMUM = "infrastructure_minimum"


class Objective(str, Enum):
    MINIMISE_INFRASTRUCTURE_COST = "minimise_infrastructure_cost"
    MAXIMISE_ACCESSIBILITY = "maximise_accessibility"
    MAXIMISE_ENVIRONMENTAL_PROTECTION = "maximise_environmental_protection"
    MAXIMISE_DEVELOPABLE_YIELD = "maximise_developable_yield"
    BALANCE_LAND_USE = "balance_land_use"


class DensityMix(BaseModel):
    low: float = Field(ge=0, le=1)
    medium: float = Field(ge=0, le=1)
    high: float = Field(ge=0, le=1)

    @model_validator(mode="after")
    def _sums_to_one(self) -> "DensityMix":
        total = self.low + self.medium + self.high
        if abs(total - 1.0) > 1e-6:
            raise ValueError(f"density mix must sum to 1.0, got {total:.4f}")
        return self

    def as_dict(self) -> dict[str, float]:
        return {"low": self.low, "medium": self.medium, "high": self.high}


class PlanningBrief(BaseModel):
    """A complete, machine-checkable statement of what is to be planned."""

    # -- identity ----------------------------------------------------------
    project_name: str
    jurisdiction: str = "botswana"

    # -- the ask -----------------------------------------------------------
    site_area_ha: float = Field(gt=0, description="Gross site area in hectares")
    target_population: int = Field(gt=0)

    # Optional explicit plot target. When absent it is derived from population.
    target_residential_plots: int | None = Field(default=None, gt=0)

    density_mix: DensityMix | None = None
    settlement_form: SettlementForm = SettlementForm.COMPACT
    objectives: list[Objective] = Field(
        default_factory=lambda: [Objective.BALANCE_LAND_USE]
    )

    # -- horizon -----------------------------------------------------------
    plan_horizon_years: int = Field(default=20, gt=0, le=100)
    expansion_allowance_pct: float = Field(default=15.0, ge=0, le=100)

    # -- explicit requirements the planner wants honoured verbatim ---------
    # e.g. {"secondary_school": 2, "primary_school": 6}
    required_facilities: dict[str, int] = Field(default_factory=dict)

    # Land uses that must be provided even if not population-driven.
    require_industrial: bool = True
    require_cbd: bool = True

    # -- constraints -------------------------------------------------------
    environmental_protection: str = Field(
        default="high", pattern="^(low|standard|high)$"
    )
    avoid_slopes_above_pct: float | None = Field(default=None, gt=0, le=100)

    # -- provenance --------------------------------------------------------
    source_prompt: str | None = Field(
        default=None,
        description="The natural-language prompt this brief was derived from, "
                    "retained for audit.",
    )

    @field_validator("project_name")
    @classmethod
    def _name_not_blank(cls, v: str) -> str:
        if not v.strip():
            raise ValueError("project_name must not be blank")
        return v.strip()

    @model_validator(mode="after")
    def _density_is_plausible(self) -> "PlanningBrief":
        """Catch briefs that are arithmetically impossible before any GIS work runs."""
        gross_density = self.target_population / self.site_area_ha
        if gross_density > 400:
            raise ValueError(
                f"Brief requests {gross_density:.0f} persons/ha gross "
                f"({self.target_population:,} people on {self.site_area_ha:,.0f} ha). "
                f"This exceeds high-rise densities and is not deliverable as a "
                f"plotted settlement. Reduce population or increase site area."
            )
        return self

    @property
    def gross_density_pph(self) -> float:
        return self.target_population / self.site_area_ha

    @property
    def design_population(self) -> int:
        """Population the infrastructure is sized for, including expansion."""
        return int(round(self.target_population * (1 + self.expansion_allowance_pct / 100)))
