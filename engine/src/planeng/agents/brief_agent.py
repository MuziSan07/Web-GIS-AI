"""The Brief Agent: natural language to a validated PlanningBrief.

This is the whole of the AI layer's authority. The agent reads a planner's
sentence and fills in a brief; it never draws, sites, or measures anything.
Everything after `PlanningBrief` is the deterministic engine, so a hallucinating
model can produce a wrong *brief* - which a planner can see and correct on one
screen - but it cannot produce wrong geometry.

Three things make a small local model usable here:

* **The schema is generated from the ruleset**, not hand-written. The model is
  offered an enum of the facility keys that actually exist, so it cannot invent
  `water_treatment_plants` or spell `primary_school` as `primary_schools`.
* **Output is validated against the real PlanningBrief.** A brief that breaks a
  constraint is rejected and the model gets one repair attempt with the
  validation error quoted back to it.
* **Assumptions are a required output field.** Anything the model inferred
  rather than read is surfaced for review instead of buried in a number.
"""

from __future__ import annotations

from dataclasses import dataclass, field

from pydantic import ValidationError

from ..brief.models import DensityMix, Objective, PlanningBrief, SettlementForm
from ..nlp.prompt import extract_numbers, parse_prompt
from ..standards.loader import Standards, load_standards
from . import ollama

SYSTEM_PROMPT = """\
You extract structured town planning briefs from a planner's instruction.

Rules:
- Extract only what the planner actually stated or clearly implied.
- Never invent facilities, populations, or areas that were not asked for.
- Use only facility keys from the provided list. If a facility was not
  requested by name and number, leave it out - the engine derives the rest
  from population standards.
- Negation is binding. "no industrial area" means require_industrial = false.
- A plot or stand count ("5,000 plots", "5,000 stands") sets
  target_residential_plots, not target_population.
- Put anything you inferred rather than read into "assumptions".
- Put anything genuinely ambiguous into "clarifications_needed".

Reply with JSON only."""


@dataclass
class BriefResult:
    """A brief plus the reasoning trail a planner needs to check it."""

    brief: PlanningBrief
    source: str                       # "llm" | "fallback_parser"
    model: str = ""
    assumptions: list[str] = field(default_factory=list)
    clarifications_needed: list[str] = field(default_factory=list)
    warnings: list[str] = field(default_factory=list)
    duration_s: float = 0.0
    cpu_forced: bool = False
    repaired: bool = False

    def summary(self) -> dict:
        return {
            "source": self.source,
            "model": self.model,
            "assumptions": self.assumptions,
            "clarifications_needed": self.clarifications_needed,
            "warnings": self.warnings,
            "duration_s": round(self.duration_s, 2),
            "cpu_forced": self.cpu_forced,
            "repaired": self.repaired,
        }


def facility_keys(std: Standards) -> list[str]:
    """Every facility the ruleset knows about, as the model's allowed vocabulary."""
    keys: list[str] = []
    for entries in std.get("facilities").values():
        if isinstance(entries, dict):
            keys.extend(k for k, v in entries.items() if isinstance(v, dict))
    return sorted(keys)


def build_schema(std: Standards) -> dict:
    """JSON schema for the model, with the ruleset's own vocabulary baked in.

    Generating this from the standards rather than hardcoding it means a
    planner who adds a facility type to the YAML immediately gets an agent that
    can be asked for it - and one that still cannot ask for anything else.
    """
    keys = facility_keys(std)
    return {
        "type": "object",
        "properties": {
            "project_name": {"type": "string"},
            "site_area_ha": {"type": "number"},
            "target_population": {"type": "integer"},
            "target_residential_plots": {
                "type": ["integer", "null"],
                "description": (
                    "Set when the planner gives a count of plots, stands or "
                    "erven. Null when they gave only a population."
                ),
            },
            "density": {
                "type": "string",
                "enum": ["low", "medium", "high", "unspecified"],
                "description": (
                    "Plot-size intent. 'high' if the planner said high-density, "
                    "dense, or small plots; 'low' for low-density, spacious, or "
                    "large plots; 'unspecified' only if they said nothing."
                ),
            },
            "settlement_form": {
                "type": "string",
                "enum": [f.value for f in SettlementForm],
                "description": (
                    "'linear' along a road, 'clustered' for neighbourhood units, "
                    "'infrastructure_minimum' for lowest-cost, else 'compact'."
                ),
            },
            "objectives": {
                "type": "array",
                "items": {"type": "string", "enum": [o.value for o in Objective]},
                "description": (
                    "ONLY objectives the planner explicitly asked to optimise. "
                    "Return an empty array if they did not say. Cost or "
                    "affordability means minimise_infrastructure_cost; "
                    "walkability or access means maximise_accessibility; "
                    "wetlands, environment or conservation means "
                    "maximise_environmental_protection. Do not add objectives "
                    "that merely sound desirable."
                ),
            },
            "required_facilities": {
                "type": "object",
                "description": (
                    "Only facilities the planner named with a count. "
                    "Keys must come from the allowed list."
                ),
                "properties": {k: {"type": "integer"} for k in keys},
                "additionalProperties": False,
            },
            "environmental_protection": {
                "type": "string",
                "enum": ["low", "standard", "high"],
                "description": (
                    "'high' if the planner asked to avoid, protect or conserve "
                    "sensitive land such as wetlands; otherwise 'standard'."
                ),
            },
            "plan_horizon_years": {"type": "integer"},
            "require_industrial": {
                "type": "boolean",
                "description": (
                    "false ONLY if the planner said no industrial area, or to "
                    "exclude or omit industry. true otherwise."
                ),
            },
            "assumptions": {"type": "array", "items": {"type": "string"}},
            "clarifications_needed": {"type": "array", "items": {"type": "string"}},
        },
        "required": [
            "project_name",
            "site_area_ha",
            "target_population",
            # Required so the model must commit to a value. Left optional, the
            # grammar happily omits them and an explicit "high-density, low
            # infrastructure cost, avoid wetlands" instruction silently lands in
            # the assumptions prose instead of the fields that drive the plan.
            "density",
            "settlement_form",
            "objectives",
            "environmental_protection",
            # NOT required_facilities. Requiring it forces the grammar to emit
            # an object, and a bare prompt then comes back with every facility
            # key set to 1. Absent is the correct answer when the planner named
            # none - the engine derives counts from population standards.
            "assumptions",
            "clarifications_needed",
        ],
    }


# Plot-size intent maps to a mix; the ruleset owns the actual plot areas.
_DENSITY_MIX = {
    "low": DensityMix(low=0.45, medium=0.40, high=0.15),
    "medium": DensityMix(low=0.20, medium=0.50, high=0.30),
    "high": DensityMix(low=0.05, medium=0.35, high=0.60),
}


def understand(
    prompt: str,
    std: Standards | None = None,
    *,
    model: str = ollama.DEFAULT_MODEL,
    allow_fallback: bool = True,
) -> BriefResult:
    """Turn a planner's sentence into a validated brief.

    Falls back to the deterministic parser when no local model is reachable, so
    the product works on a laptop with Ollama stopped. The fallback is recorded
    on the result rather than hidden - a planner should know whether a model or
    a regex read their instruction.
    """
    std = std or load_standards()

    if not ollama.available():
        if not allow_fallback:
            raise ollama.OllamaError(
                "No local model reachable and fallback disabled. "
                "Start Ollama with `ollama serve`."
            )
        return _fallback(prompt, "No local model reachable; used the rule-based parser.")

    schema = build_schema(std)
    messages = [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": _user_message(prompt, std)},
    ]

    try:
        raw, completion = ollama.chat_json(messages, schema, model=model)
    except ollama.OllamaError as exc:
        if not allow_fallback:
            raise
        return _fallback(prompt, f"Local model failed ({exc}); used the rule-based parser.")

    warnings: list[str] = []
    raw, corrections, grounded = _reconcile_numbers(raw, prompt)
    warnings.extend(corrections)

    if not grounded:
        # The planner gave no area and no population, so every number in this
        # brief was invented by the model. Generating a confident master plan
        # from that would be the worst failure this system could have: it looks
        # authoritative and is founded on nothing. Ask instead.
        questions = [str(q) for q in raw.get("clarifications_needed", [])][:4]
        asks = "; ".join(questions) if questions else (
            "site area in hectares, and target population or plot count"
        )
        raise ValueError(
            "The brief does not state a site area or a population, so the plan "
            f"would be built on invented figures. Please specify: {asks}. "
            "Example: 'Plan a 600 hectare settlement for 25,000 people.'"
        )
    repaired = False

    try:
        brief = _to_brief(raw, prompt)
    except (ValidationError, ValueError) as exc:
        # One repair attempt: the model sees exactly what the engine rejected.
        # A second failure falls back rather than looping - a small model that
        # cannot satisfy the schema twice will not satisfy it on the third try.
        repaired = True
        messages.append({"role": "assistant", "content": completion.text})
        messages.append({
            "role": "user",
            "content": (
                f"That brief was rejected by the planning engine:\n{exc}\n\n"
                f"Return corrected JSON. Change only what the error names."
            ),
        })
        try:
            raw, completion = ollama.chat_json(messages, schema, model=model)
            brief = _to_brief(raw, prompt)
            warnings.append("The model's first brief was invalid and was repaired on retry.")
        except (ollama.OllamaError, ValidationError, ValueError) as exc2:
            if not allow_fallback:
                raise
            return _fallback(
                prompt,
                f"Local model produced an invalid brief twice ({exc2}); "
                f"used the rule-based parser.",
            )

    return BriefResult(
        brief=brief,
        source="llm",
        model=completion.model,
        assumptions=[str(a) for a in raw.get("assumptions", [])][:10],
        clarifications_needed=[str(c) for c in raw.get("clarifications_needed", [])][:10],
        warnings=warnings,
        duration_s=completion.duration_s,
        cpu_forced=completion.ran_on_cpu,
        repaired=repaired,
    )


# ---------------------------------------------------------------------------

# Fields a regular expression reads exactly and a 3B model does not. Measured:
# on "600 ha providing 5,000 residential plots", the model wrote 5000 into three
# facility counts. Digits next to a unit are a lexical problem, not a reasoning
# one - so the deterministic parser owns the numbers and the model owns intent
# (density, form, objectives, protection, which facilities were actually named).
_NUMERIC_FIELDS = ("site_area_ha", "target_population", "target_residential_plots")


def _reconcile_numbers(raw: dict, prompt: str) -> tuple[dict, list[str], bool]:
    """Let the deterministic parser overrule the model on pure numbers.

    The third return value says whether the brief's core numbers are actually
    *grounded* in the planner's text. A model asked to fill a required
    site_area_ha will always produce one, so an ungrounded number is an
    invention, not an extraction.
    """
    truth = extract_numbers(prompt)
    # Grounded means the planner actually stated a size, not that the brief is
    # achievable. An impossible-but-explicit brief must fail on its own terms
    # further down, with an error that matches what the planner wrote.
    grounded = truth["site_area_ha"] is not None and (
        truth["target_population"] is not None
        or truth["target_residential_plots"] is not None
    )
    if not grounded:
        return raw, [], False

    corrections: list[str] = []

    for key in _NUMERIC_FIELDS:
        want = truth[key]
        if want is None:
            continue
        got = raw.get(key)
        if got is None or abs(float(got) - float(want)) > 0.51:
            raw[key] = want
            if got is not None:
                corrections.append(
                    f"{key}: model read {got}, text says {want}. Text wins."
                )

    # A plot count the model mistook for a facility count is the specific
    # failure this guards against, so drop any facility whose count equals it.
    plots = truth.get("target_residential_plots")
    if plots and isinstance(raw.get("required_facilities"), dict):
        bogus = [k for k, v in raw["required_facilities"].items() if v == plots]
        for k in bogus:
            raw["required_facilities"].pop(k)
        if bogus:
            corrections.append(
                f"Dropped {', '.join(bogus)}: the model copied the plot count "
                f"({plots}) into facility counts."
            )

    return raw, corrections, True


def _user_message(prompt: str, std: Standards) -> str:
    keys = ", ".join(facility_keys(std))
    return (
        f"Allowed facility keys: {keys}\n\n"
        f"Planner's instruction:\n{prompt}"
    )


def _to_brief(raw: dict, prompt: str) -> PlanningBrief:
    """Convert the model's JSON into a real brief, validating on the way."""
    density = raw.get("density", "unspecified")
    mix = _DENSITY_MIX.get(density)

    facilities = {
        k: int(v)
        for k, v in (raw.get("required_facilities") or {}).items()
        if isinstance(v, (int, float)) and v > 0
    }

    valid = {m.value for m in Objective}
    objectives: list[Objective] = []
    for o in raw.get("objectives") or []:
        if o in valid and Objective(o) not in objectives:
            objectives.append(Objective(o))

    form = raw.get("settlement_form", SettlementForm.COMPACT.value)
    if form not in {f.value for f in SettlementForm}:
        form = SettlementForm.COMPACT.value

    # A constrained-decoding model asked for a required string will emit the
    # literal "null"/"none" rather than omit the field.
    name = (raw.get("project_name") or "").strip()
    if name.lower() in ("", "null", "none", "n/a", "unspecified", "untitled"):
        name = "Untitled Settlement"

    return PlanningBrief(
        project_name=name,
        site_area_ha=float(raw["site_area_ha"]),
        target_population=int(raw["target_population"]),
        target_residential_plots=(
            int(raw["target_residential_plots"])
            if raw.get("target_residential_plots") else None
        ),
        density_mix=mix,
        settlement_form=SettlementForm(form),
        objectives=objectives or [Objective.BALANCE_LAND_USE],
        required_facilities=facilities,
        environmental_protection=raw.get("environmental_protection", "standard"),
        # `or 20` here would swallow a zero: a model emitting 0 would be
        # silently corrected to a valid 20-year horizon instead of surfacing as
        # the mistake it is. Default only when the field is genuinely absent.
        plan_horizon_years=(
            20 if raw.get("plan_horizon_years") is None
            else int(raw["plan_horizon_years"])
        ),
        require_industrial=bool(raw.get("require_industrial", True)),
        source_prompt=prompt,
    )


def _fallback(prompt: str, reason: str) -> BriefResult:
    return BriefResult(
        brief=parse_prompt(prompt),
        source="fallback_parser",
        warnings=[reason],
    )
