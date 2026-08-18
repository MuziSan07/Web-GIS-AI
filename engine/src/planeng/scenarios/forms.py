"""Settlement forms as suitability weightings.

A form is not a different algorithm — it is a different answer to "what makes
land good for housing here?". A compact town rewards being near the centre; a
linear one rewards being near the road. Expressing form as weights keeps every
scenario explainable in the same vocabulary a planner already reads off the
suitability report, and keeps one code path under all four.

The weights below are planning judgement, not statutory standards, which is why
they live in code beside the default weights rather than in the ruleset YAML.
"""

from __future__ import annotations

from copy import deepcopy

from ..analysis.suitability import CRITERIA_WEIGHTS, LandUse
from ..brief.models import SettlementForm

# What each form is arguing for, in the planner's own terms. Shown in the UI
# beside the score so a form is never just a label on a number.
FORM_RATIONALE = {
    SettlementForm.COMPACT: (
        "Growth pulled toward one centre. Shortest average trip, densest "
        "service catchments, least land consumed."
    ),
    SettlementForm.LINEAR: (
        "Development strung along the existing road. Cheapest to service "
        "initially and easy to extend, at the cost of longer cross-town trips."
    ),
    SettlementForm.CLUSTERED: (
        "Several distinct neighbourhoods with their own centres. Better local "
        "identity and walkable sub-centres; more duplicated infrastructure."
    ),
    SettlementForm.INFRASTRUCTURE_MINIMUM: (
        "Everything packed against existing services and the site edge to "
        "minimise new road, water and power runs. Cheapest to build, least "
        "generous in amenity."
    ),
}


def weights_for(form: SettlementForm) -> dict[LandUse, dict[str, float]]:
    """Suitability weights expressing a settlement form.

    Starts from the defaults and shifts the criteria that actually define the
    form, so a scenario differs from the baseline in a way that can be read off
    a diff rather than guessed at.
    """
    w = deepcopy(CRITERIA_WEIGHTS)

    if form is SettlementForm.COMPACT:
        # The default weighting already argues for a single strong centre.
        return w

    if form is SettlementForm.LINEAR:
        # Proximity to the road corridor dominates; centrality stops mattering
        # because a linear town has no single middle.
        w["residential"].update({"road_access": 0.45, "centrality": 0.05,
                                 "gentle_slope": 0.25})
        w["commercial"].update({"road_access": 0.60, "centrality": 0.10})
        w["institutional"].update({"road_access": 0.45, "centrality": 0.15})

    elif form is SettlementForm.CLUSTERED:
        # Weaken the pull of the single centre and let terrain and amenity
        # decide where the neighbourhoods land, which produces several nuclei
        # rather than one.
        w["residential"].update({"centrality": 0.05, "amenity_proximity": 0.30,
                                 "gentle_slope": 0.35, "road_access": 0.10})
        w["institutional"].update({"centrality": 0.20, "road_access": 0.35})

    elif form is SettlementForm.INFRASTRUCTURE_MINIMUM:
        # Everything hugs the existing road and the site edge, which is what
        # actually shortens new service runs. Amenity is what gets sacrificed.
        w["residential"].update({"road_access": 0.50, "edge_location": 0.15,
                                 "gentle_slope": 0.25, "centrality": 0.05,
                                 "amenity_proximity": 0.05,
                                 "elevated_ground": 0.0,
                                 "industrial_separation": 0.0})
        w["commercial"].update({"road_access": 0.60, "centrality": 0.20,
                                "gentle_slope": 0.20, "elevated_ground": 0.0})
        w["institutional"].update({"road_access": 0.50, "centrality": 0.20,
                                   "gentle_slope": 0.30,
                                   "industrial_separation": 0.0})

    return {use: {k: v for k, v in crit.items() if v > 0}
            for use, crit in w.items()}


def describe(form: SettlementForm) -> dict:
    """Form, rationale, and how its weights differ from the baseline."""
    w = weights_for(form)
    changed: dict[str, dict[str, str]] = {}
    for use, crit in w.items():
        base = CRITERIA_WEIGHTS[use]
        diffs = {
            k: f"{base.get(k, 0.0):.2f} -> {v:.2f}"
            for k, v in crit.items() if abs(v - base.get(k, 0.0)) > 1e-9
        }
        # A criterion dropped to zero is a change worth showing too.
        diffs.update({
            k: f"{v:.2f} -> 0.00"
            for k, v in base.items() if k not in crit and v > 0
        })
        if diffs:
            changed[use] = diffs

    return {
        "form": form.value,
        "rationale": FORM_RATIONALE[form],
        "weight_changes": changed,
    }
