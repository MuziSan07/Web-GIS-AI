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

from .brief.models import DensityMix, PlanningBrief
from .nlp.prompt import parse_prompt
from .pipeline import generate_master_plan
from .standards.loader import load_standards

WEB_DIR = Path(__file__).parent / "web"

app = FastAPI(title="Spatial Planning Engine", version="0.1.0")


class PromptRequest(BaseModel):
    prompt: str = Field(min_length=8)
    cell_size: float = Field(default=10.0, ge=5.0, le=50.0)


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
    try:
        brief = parse_prompt(req.prompt)
    except ValueError as exc:
        # A brief the engine cannot read is a user-correctable problem, not a
        # server fault. Say exactly what was missing.
        raise HTTPException(status_code=422, detail=str(exc)) from exc

    try:
        return _run(brief, req.cell_size)
    except ValueError as exc:
        raise HTTPException(status_code=422, detail=str(exc)) from exc


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


@app.get("/api/health")
def health() -> dict:
    return {"status": "ok", "version": app.version}


@app.get("/")
def index() -> FileResponse:
    return FileResponse(WEB_DIR / "index.html")


if WEB_DIR.exists():
    app.mount("/static", StaticFiles(directory=WEB_DIR), name="static")
