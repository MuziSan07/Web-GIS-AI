"""Natural language brief parsing.

This is the seam where the AI agent layer plugs in. Today it is a deterministic
extractor: it reads numbers and keywords out of a planner's prompt and fills a
PlanningBrief. That keeps the demo working offline and, more importantly, gives
the LLM implementation something to be measured against.

When the agent layer lands it replaces `parse_prompt` only. Everything
downstream still receives a PlanningBrief and stays deterministic, so swapping
in a model cannot change how geometry is produced - only how the brief is read.
"""

from __future__ import annotations

import re

from ..brief.models import DensityMix, Objective, PlanningBrief, SettlementForm

_NUM = r"([\d]{1,3}(?:[,\s]\d{3})*(?:\.\d+)?|\d+(?:\.\d+)?)"


def _num(text: str) -> float:
    return float(text.replace(",", "").replace(" ", ""))


def _find(pattern: str, text: str) -> float | None:
    m = re.search(pattern, text, re.I)
    return _num(m.group(1)) if m else None


# Facility words a planner is likely to name explicitly, mapped to ruleset keys.
FACILITY_WORDS = {
    r"primary schools?": "primary_school",
    r"secondary schools?": "secondary_school",
    r"pre[- ]?schools?": "pre_school",
    r"clinics?": "clinic",
    r"health posts?": "health_post",
    r"hospitals?": "primary_hospital",
    r"police stations?": "police_station",
    r"fire stations?": "fire_station",
    r"community halls?": "community_hall",
    r"kgotlas?": "kgotla",
    r"libraries|library": "library",
    r"markets?": "market",
    r"sports? fields?": "sports_field",
}

_WORD_NUMBERS = {
    "one": 1, "two": 2, "three": 3, "four": 4, "five": 5, "six": 6,
    "seven": 7, "eight": 8, "nine": 9, "ten": 10, "twelve": 12,
}


def extract_numbers(prompt: str) -> dict[str, float | None]:
    """Read the quantities out of a prompt, without judging whether they work.

    Separate from `parse_prompt` because extraction and validation are
    different questions. "10 hectares for 25,000 people" states both numbers
    perfectly clearly; it is merely impossible. A caller asking "did the
    planner give me an area?" must not be told "no" just because the answer
    they gave cannot be built.
    """
    area = _find(rf"{_NUM}\s*(?:ha\b|hectares?)", prompt)
    if area is None:
        km2 = _find(rf"{_NUM}\s*(?:km2|km²|square kilometres?)", prompt)
        area = km2 * 100 if km2 is not None else None

    pop = _find(rf"(?:for|of|house|accommodate|population of)\s+{_NUM}\s*"
                r"(?:people|persons|residents|inhabitants|population)", prompt)
    if pop is None:
        pop = _find(rf"{_NUM}\s*(?:people|persons|residents|inhabitants)", prompt)

    plots = _find(rf"{_NUM}\s*(?:residential\s+)?(?:plots?|stands?|erven)", prompt)

    return {"site_area_ha": area, "target_population": pop,
            "target_residential_plots": plots}


def parse_prompt(prompt: str, default_name: str = "Untitled Settlement") -> PlanningBrief:
    """Extract a structured brief from a planner's natural-language instruction."""
    text = prompt.strip()

    # -- area -------------------------------------------------------------
    area = _find(rf"{_NUM}\s*(?:ha\b|hectares?)", text)
    if area is None:
        area = _find(rf"{_NUM}\s*(?:km2|km²|square kilometres?)", text)
        if area is not None:
            area *= 100

    # -- population -------------------------------------------------------
    pop = _find(rf"(?:for|of|house|accommodate|population of)\s+{_NUM}\s*"
                r"(?:people|persons|residents|inhabitants|population)", text)
    if pop is None:
        pop = _find(rf"{_NUM}\s*(?:people|persons|residents|inhabitants)", text)

    # -- plots ------------------------------------------------------------
    plots = _find(rf"{_NUM}\s*(?:residential\s+)?(?:plots?|stands?|erven)", text)

    if area is None or (pop is None and plots is None):
        missing = []
        if area is None:
            missing.append("site area in hectares")
        if pop is None and plots is None:
            missing.append("target population or plot count")
        raise ValueError(
            "Could not read a complete brief from the prompt. Missing: "
            + ", ".join(missing)
            + ". Example: 'Plan a 600 hectare settlement for 25,000 people.'"
        )

    # If only plots were given, infer a population so facilities can be sized.
    if pop is None:
        pop = plots * 3.5

    # -- settlement form --------------------------------------------------
    form = SettlementForm.COMPACT
    if re.search(r"\blinear\b|along the (main )?road|ribbon", text, re.I):
        form = SettlementForm.LINEAR
    elif re.search(r"\bcluster(ed|s)?\b|neighbourhood units?", text, re.I):
        form = SettlementForm.CLUSTERED
    elif re.search(r"minimum infrastructure|lowest cost|cheapest", text, re.I):
        form = SettlementForm.INFRASTRUCTURE_MINIMUM

    # -- objectives -------------------------------------------------------
    objectives: list[Objective] = []
    if re.search(r"infrastructure cost|low[- ]cost|minimi[sz]e cost|affordab", text, re.I):
        objectives.append(Objective.MINIMISE_INFRASTRUCTURE_COST)
    if re.search(r"walk|accessib|15[- ]minute|proximity", text, re.I):
        objectives.append(Objective.MAXIMISE_ACCESSIBILITY)
    if re.search(r"environment|wetland|sensitive|conserv|sustainab|green", text, re.I):
        objectives.append(Objective.MAXIMISE_ENVIRONMENTAL_PROTECTION)
    if not objectives:
        objectives = [Objective.BALANCE_LAND_USE]

    # -- density ----------------------------------------------------------
    mix = None
    if re.search(r"high[- ]density|dense|compact plots", text, re.I):
        mix = DensityMix(low=0.05, medium=0.35, high=0.60)
    elif re.search(r"low[- ]density|spacious|large plots", text, re.I):
        mix = DensityMix(low=0.45, medium=0.40, high=0.15)

    # -- explicit facility counts ----------------------------------------
    required: dict[str, int] = {}
    for pattern, key in FACILITY_WORDS.items():
        m = re.search(rf"(\d+|{'|'.join(_WORD_NUMBERS)})\s+{pattern}", text, re.I)
        if m:
            raw = m.group(1).lower()
            required[key] = _WORD_NUMBERS.get(raw, int(raw) if raw.isdigit() else 1)

    # -- protection level -------------------------------------------------
    protection = "standard"
    if re.search(r"high environmental|strictly avoid|protect.*sensitive|conservation",
                 text, re.I):
        protection = "high"
    elif re.search(r"minimal environmental|low protection", text, re.I):
        protection = "low"

    # -- horizon ----------------------------------------------------------
    horizon = _find(rf"{_NUM}\s*[- ]?year", text)

    # -- name -------------------------------------------------------------
    name_match = re.search(r"(?:called|named)\s+([A-Z][\w\s'-]{2,40})", text)
    name = name_match.group(1).strip() if name_match else default_name

    return PlanningBrief(
        project_name=name,
        site_area_ha=area,
        target_population=int(round(pop)),
        target_residential_plots=int(plots) if plots else None,
        density_mix=mix,
        settlement_form=form,
        objectives=objectives,
        required_facilities=required,
        environmental_protection=protection,
        plan_horizon_years=int(horizon) if horizon else 20,
        require_industrial=not re.search(r"no industr", text, re.I),
        source_prompt=prompt,
    )
