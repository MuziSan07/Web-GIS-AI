"""HTTP API and web application host.

Deliberately thin. Every endpoint is a direct call into the engine, so anything
the browser can do, an agent can do through the same library calls.
"""

from __future__ import annotations

from pathlib import Path

from fastapi import FastAPI, HTTPException
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel, Field

from .agents import brief_agent, ollama
from .brief.models import DensityMix, PlanningBrief
from .nlp.prompt import parse_prompt
from .pipeline import generate_master_plan
from .scenarios import compare as compare_scenarios
from .standards.loader import load_standards

WEB_DIR = Path(__file__).parent / "web"

app = FastAPI(title="Spatial Planning Engine", version="0.1.0")


class PromptRequest(BaseModel):
    prompt: str = Field(min_length=8)
    cell_size: float = Field(default=10.0, ge=5.0, le=50.0)
    use_llm: bool = Field(
        default=True,
        description="Use the local model to read the brief. Falls back to the "
                    "deterministic parser when no model is reachable.",
    )


class BriefRequest(BaseModel):
    project_name: str = "Untitled Settlement"
    site_area_ha: float = Field(gt=0)
    target_population: int = Field(gt=0)
    density_low: float = Field(default=0.20, ge=0, le=1)
    density_medium: float = Field(default=0.50, ge=0, le=1)
    density_high: float = Field(default=0.30, ge=0, le=1)
    expansion_allowance_pct: float = Field(default=15.0, ge=0, le=100)
    require_industrial: bool = True
    cell_size: float = Field(default=10.0, ge=5.0, le=50.0)


def _run(brief: PlanningBrief, cell_size: float) -> dict:
    plan = generate_master_plan(brief, cell_size=cell_size)
    return {
        "report": plan.report(),
        "geojson": plan.to_geojson(),
        "brief": brief.model_dump(mode="json"),
    }


@app.post("/api/plan/prompt")
def plan_from_prompt(req: PromptRequest) -> dict:
    """Generate a master plan from a natural-language brief."""
    understanding = None
    try:
        if req.use_llm:
            result = brief_agent.understand(req.prompt)
            brief, understanding = result.brief, result.summary()
        else:
            brief = parse_prompt(req.prompt)
    except ValueError as exc:
        # A brief the engine cannot read is a user-correctable problem, not a
        # server fault. Say exactly what was missing.
        raise HTTPException(status_code=422, detail=str(exc)) from exc

    try:
        payload = _run(brief, req.cell_size)
    except ValueError as exc:
        raise HTTPException(status_code=422, detail=str(exc)) from exc

    # The agent's assumptions ride alongside the plan: a planner reviewing the
    # output needs to see what was inferred rather than read.
    payload["understanding"] = understanding
    return payload


@app.post("/api/plan")
def plan_from_brief(req: BriefRequest) -> dict:
    """Generate a master plan from explicit parameters."""
    total = req.density_low + req.density_medium + req.density_high
    if abs(total - 1.0) > 1e-6:
        raise HTTPException(
            status_code=422,
            detail=f"Density mix must sum to 1.0, got {total:.3f}",
        )
    try:
        brief = PlanningBrief(
            project_name=req.project_name,
            site_area_ha=req.site_area_ha,
            target_population=req.target_population,
            density_mix=DensityMix(
                low=req.density_low, medium=req.density_medium, high=req.density_high
            ),
            expansion_allowance_pct=req.expansion_allowance_pct,
            require_industrial=req.require_industrial,
        )
        return _run(brief, req.cell_size)
    except ValueError as exc:
        raise HTTPException(status_code=422, detail=str(exc)) from exc


@app.get("/api/standards")
def standards(jurisdiction: str = "botswana") -> dict:
    std = load_standards(jurisdiction)
    return {
        "jurisdiction": std.jurisdiction,
        "version": std.data["meta"]["ruleset_version"],
        "status": std.data["meta"]["status"],
        "validated": std.is_validated,
        "validated_by": std.data["meta"]["validated_by"],
        "placeholder_paths": std.placeholder_paths(),
        "warning": std.provenance_warning(),
        "ruleset": std.data,
    }


class ScenarioRequest(BaseModel):
    prompt: str | None = Field(default=None, min_length=8)
    site_area_ha: float | None = Field(default=None, gt=0)
    target_population: int | None = Field(default=None, gt=0)
    cell_size: float = Field(
        default=20.0, ge=10.0, le=50.0,
        description="Coarser than a single plan by default: four full pipelines "
                    "at 10 m take minutes, and form comparison resolves fine "
                    "at 20 m. Generate the chosen form at full resolution.",
    )
    use_llm: bool = True


@app.post("/api/scenarios")
def scenarios(req: ScenarioRequest) -> dict:
    """Generate several settlement forms from one brief and score each.

    Only the form varies: the site, ruleset and population are shared, so any
    difference between scorecards is attributable to the form.
    """
    try:
        if req.prompt:
            brief = (brief_agent.understand(req.prompt).brief if req.use_llm
                     else parse_prompt(req.prompt))
        elif req.site_area_ha and req.target_population:
            brief = PlanningBrief(
                project_name="Scenario Comparison",
                site_area_ha=req.site_area_ha,
                target_population=req.target_population,
            )
        else:
            raise HTTPException(
                status_code=422,
                detail="Provide either a prompt, or both site_area_ha and "
                       "target_population.",
            )
    except ValueError as exc:
        raise HTTPException(status_code=422, detail=str(exc)) from exc

    result = compare_scenarios(brief, cell_size=req.cell_size)
    if not result.scenarios:
        raise HTTPException(
            status_code=422,
            detail=f"No scenario could be generated. {result.failures}",
        )
    return result.summary()


@app.get("/api/agent")
def agent_status() -> dict:
    """What the AI layer is running on, so the UI can say so plainly."""
    reachable = ollama.available()
    return {
        "reachable": reachable,
        "host": ollama.DEFAULT_HOST,
        "default_model": ollama.DEFAULT_MODEL,
        "installed_models": ollama.installed_models() if reachable else [],
        "mode": "local model" if reachable else "rule-based parser",
    }


@app.get("/api/health")
def health() -> dict:
    return {"status": "ok", "version": app.version}


@app.get("/")
def index() -> FileResponse:
    return FileResponse(WEB_DIR / "index.html")


if WEB_DIR.exists():
    app.mount("/static", StaticFiles(directory=WEB_DIR), name="static")
