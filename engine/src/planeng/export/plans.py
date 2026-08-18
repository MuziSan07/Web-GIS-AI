"""The plan set: printable, to a stated scale.

This is the deliverable a planning authority receives. It is not a screenshot of
the map, and the difference is entirely in the parts that are easy to leave out.

A drawing is only a drawing if it is to a **standard scale**. Fitting the site
to whatever space is left on the sheet produces something at 1:1837, which
cannot be scaled off with a rule and is therefore not a planning document. Here
the smallest standard scale that fits the frame is chosen, the drawing is laid
out at exactly that scale, and the scale is printed and drawn as a bar - so it
survives being photocopied at 94%, which is what happens to plans.

Every sheet also carries the north point, the coordinate reference system, and
the status of the ruleset the plan was generated from. That last one is not
decoration: this ruleset is unvalidated placeholder data, and a plan set that
looks statutory while resting on unverified standards is worse than no plan set
at all.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path

from reportlab.lib.colors import Color, HexColor
from reportlab.lib.pagesizes import A0, A1, A2, A3, landscape
from reportlab.lib.units import mm
from reportlab.pdfgen import canvas as pdfcanvas

from .layers import PlanLayers, build_layers

# Scales a planner can read off with a rule. Anything between them is not a
# scale, it is a fit.
STANDARD_SCALES = [200, 500, 1000, 1250, 2000, 2500, 5000, 10000, 20000, 50000]

# Sheet sizes a drawing office actually plots on, smallest first. A town layout
# is normally issued at A1 or A0; A3 is here because a schedule sheet does not
# need a plan chest and a reviewer would rather not unfold one.
SHEETS = [("A3", landscape(A3)), ("A2", landscape((420 * mm, 594 * mm))),
          ("A1", landscape(A1)), ("A0", landscape(A0))]

# The coarsest scale worth issuing a layout plan at. Past this, plot boundaries
# stop being legible and the drawing cannot be checked, so the answer is a
# bigger sheet rather than a smaller drawing.
USABLE_SCALE = 5000

MARGIN = 12 * mm
FRAME_PAD = 6 * mm

# Text and title block scale with the sheet. Seven point type is legible held in
# the hand on A3 and invisible pinned to a wall on A0, and a plan is read at the
# distance its sheet implies.
SHEET_SCALE = {"A3": 1.0, "A2": 1.35, "A1": 1.8, "A0": 2.4}
_sheet_factor = 1.0


def _f(value: float) -> float:
    """Scale a page dimension for the sheet currently being drawn."""
    return value * _sheet_factor


TITLE_BLOCK_BASE_W = 62 * mm

# Fill, stroke, line width. None means no fill.
STYLE = {
    "zones": (HexColor("#dce8f2"), HexColor("#9db8cc"), 0.2),
    "constraints": (HexColor("#f2dede"), HexColor("#c08a8a"), 0.3),
    "blocks": (None, HexColor("#999999"), 0.3),
    "parcels": (HexColor("#f7f7f2"), HexColor("#7a7a6a"), 0.15),
    "facilities": (HexColor("#e6d5f0"), HexColor("#8c5aa8"), 0.4),
    "service_sites": (HexColor("#dfd0ea"), HexColor("#6a3d8f"), 0.5),
    "road_reserve": (HexColor("#eeeeee"), HexColor("#bbbbbb"), 0.2),
    "road_centreline": (None, HexColor("#c0392b"), 0.5),
    "site_boundary": (None, HexColor("#000000"), 1.1),
    "water_mains": (None, HexColor("#1f78b4"), 0.4),
    "sewers": (None, HexColor("#8b5a2b"), 0.4),
    "electrical": (None, HexColor("#c9a227"), 0.4),
    "drainage_channels": (None, HexColor("#3a8a3a"), 0.8),
}

# Lateral offset in metres for each service, so the four do not overplot. This
# is not a drafting trick: services are laid at these offsets across the reserve
# precisely so they can be dug up independently, and drawing them stacked on the
# centreline hides both the water main under the electrical cable and the fact
# that the corridor has to be wide enough to hold them apart.
SERVICE_OFFSET_M = {
    "water_mains": -3.0,
    "sewers": 0.0,
    "electrical": 3.0,
}

ZONE_FILL = {
    "residential": HexColor("#f6e2b3"),
    "commercial": HexColor("#e8b7b7"),
    "industrial": HexColor("#d9c2e0"),
    "institutional": HexColor("#b9d3e8"),
    "recreation_open_space": HexColor("#c6e0b4"),
}


@dataclass
class PlanSetResult:
    path: Path
    sheets: list[str] = field(default_factory=list)
    scale: int = 0
    sheet: str = ""
    warnings: list[str] = field(default_factory=list)

    def summary(self) -> dict:
        return {
            "format": "PDF",
            "path": str(self.path),
            "sheets": self.sheets,
            "sheet_count": len(self.sheets),
            "scale": f"1:{self.scale:,}",
            "sheet_size": self.sheet,
            "warnings": self.warnings,
        }


def write_plan_set(plan, path: str | Path) -> PlanSetResult:
    """Write the full plan set to one PDF."""
    layers = build_layers(plan)
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)

    report = plan.report()
    std = plan.standards
    bounds = _bounds(layers)
    sheet_name, page, scale = _pick_sheet(bounds)

    global _sheet_factor
    _sheet_factor = SHEET_SCALE.get(sheet_name, 1.0)

    c = pdfcanvas.Canvas(str(path), pagesize=page)
    c.setTitle(f"{plan.brief.project_name} - Development Plan")
    result = PlanSetResult(path=path, scale=scale, sheet=sheet_name)

    sheets = [
        ("1", "Land Use Plan",
         ["zones", "constraints", "road_reserve", "road_centreline",
          "site_boundary"]),
        ("2", "Layout and Cadastral Plan",
         ["blocks", "parcels", "facilities", "road_reserve", "site_boundary"]),
        ("3", "Engineering Services Plan",
         ["road_reserve", "water_mains", "sewers", "electrical",
          "service_sites", "drainage_channels", "site_boundary"]),
    ]

    for number, title, names in sheets:
        _draw_sheet(c, page, layers, names, bounds, scale, number, title,
                    plan, std)
        result.sheets.append(f"{number} - {title}")
        c.showPage()

    _schedule_sheet(c, page, plan, report, "4", "Development Schedules", std)
    result.sheets.append("4 - Development Schedules")
    c.showPage()

    c.save()

    if scale > USABLE_SCALE:
        result.warnings.append(
            f"Issued at 1:{scale:,} on {sheet_name}, which is coarser than the "
            f"1:{USABLE_SCALE:,} a layout plan is normally checked at. Plot "
            f"boundaries will not be legible; issue the cadastral sheet as "
            f"tiles if the authority needs to check them.")

    if not std.is_validated:
        result.warnings.append(
            "Every sheet is stamped as generated from an unvalidated ruleset. "
            "Remove that stamp only once a registered planner has signed off "
            "the standards file.")
    return result


# ---------------------------------------------------------------------------
# layout
# ---------------------------------------------------------------------------

def _frame(page) -> tuple[float, float, float, float]:
    """Drawing frame: the page less margins and the title block."""
    width, height = page
    x0 = MARGIN
    y0 = MARGIN
    x1 = width - MARGIN - _f(TITLE_BLOCK_BASE_W)
    y1 = height - MARGIN
    return x0, y0, x1, y1


def _bounds(layers: PlanLayers):
    boundary = layers.layers.get("site_boundary")
    gdf = boundary if boundary is not None and len(boundary) else next(
        iter(layers.non_empty().values()))
    return tuple(gdf.total_bounds)


def _pick_scale(bounds, frame) -> int:
    """Smallest standard scale that still fits the site in the frame.

    Smallest, because a bigger drawing is a more useful one: at 1:2000 a planner
    can check a plot boundary, and at 1:10000 the same plot is a hairline.
    """
    minx, miny, maxx, maxy = bounds
    fx0, fy0, fx1, fy1 = frame
    avail_w = (fx1 - fx0) - 2 * FRAME_PAD
    avail_h = (fy1 - fy0) - 2 * FRAME_PAD

    site_w = max(maxx - minx, 1.0)
    site_h = max(maxy - miny, 1.0)

    for scale in STANDARD_SCALES:
        # metres on the ground -> metres on paper -> points
        if (site_w / scale) * 1000 * mm <= avail_w and \
           (site_h / scale) * 1000 * mm <= avail_h:
            return scale
    return STANDARD_SCALES[-1]


def _pick_sheet(bounds):
    """The smallest sheet that carries this site at a legible scale.

    Sheet size and scale are one decision, not two. Fixing the sheet first and
    accepting whatever scale results is how a 600 ha layout ends up issued at
    1:20,000, where a plot is a fifth of a millimetre across and the drawing
    cannot be checked by the person who has to approve it. So the smallest sheet
    that reaches a usable scale wins, and if none does, the largest sheet is
    used and the caller is told the drawing is coarse.
    """
    best = None
    for name, page in SHEETS:
        scale = _pick_scale(bounds, _frame(page))
        if best is None:
            best = (name, page, scale)
        if scale <= USABLE_SCALE:
            return name, page, scale
    name, page = SHEETS[-1]
    return name, page, _pick_scale(bounds, _frame(page))


def _projector(bounds, frame, scale: int):
    """Ground coordinates to page points, at exactly the chosen scale."""
    minx, miny, maxx, maxy = bounds
    fx0, fy0, fx1, fy1 = frame
    factor = (1000 * mm) / scale             # points per ground metre

    drawn_w = (maxx - minx) * factor
    drawn_h = (maxy - miny) * factor
    ox = fx0 + ((fx1 - fx0) - drawn_w) / 2
    oy = fy0 + ((fy1 - fy0) - drawn_h) / 2

    def project(x, y):
        return ox + (x - minx) * factor, oy + (y - miny) * factor

    return project, factor


# ---------------------------------------------------------------------------
# sheets
# ---------------------------------------------------------------------------

def _draw_sheet(c, page, layers, names, bounds, scale, number, title,
                plan, std):
    project, factor = _projector(bounds, _frame(page), scale)

    c.saveState()
    fx0, fy0, fx1, fy1 = _frame(page)
    c.setStrokeColor(HexColor("#333333"))
    c.setLineWidth(0.8)
    c.rect(fx0, fy0, fx1 - fx0, fy1 - fy0)
    c.clipPath(_clip(c, fx0, fy0, fx1, fy1), stroke=0)

    for name in names:
        gdf = layers.layers.get(name)
        if gdf is None or not len(gdf):
            continue
        fill, stroke, width = STYLE.get(name, (None, HexColor("#666666"), 0.3))
        offset = SERVICE_OFFSET_M.get(name, 0.0)
        for i, geom in enumerate(gdf.geometry):
            use_fill = fill
            if name == "zones":
                use_fill = ZONE_FILL.get(gdf["land_use"].iloc[i], fill)
            if offset:
                geom = _offset(geom, offset)
            _draw_geom(c, geom, project, use_fill, stroke, width)

    c.restoreState()

    _title_block(c, page, number, title, plan, std, scale)
    _north_point(c, page)
    _scale_bar(c, page, scale, factor)
    _legend(c, page, names)


def _offset(geom, distance: float):
    """Shift a line sideways by its corridor offset, tolerating failures.

    Offsetting a very short or self-touching run can fail or return an empty
    geometry, and a service run that vanishes from the drawing is worse than one
    drawn on the centreline.
    """
    try:
        shifted = geom.offset_curve(distance)
    except Exception:
        return geom
    return geom if shifted.is_empty else shifted


def _clip(c, x0, y0, x1, y1):
    path = c.beginPath()
    path.rect(x0, y0, x1 - x0, y1 - y0)
    return path


def _draw_geom(c, geom, project, fill: Color | None, stroke: Color,
               width: float) -> None:
    if geom is None or geom.is_empty:
        return
    kind = geom.geom_type

    if kind.startswith("Multi") or kind == "GeometryCollection":
        for part in geom.geoms:
            _draw_geom(c, part, project, fill, stroke, width)
        return

    c.setLineWidth(width)
    c.setStrokeColor(stroke)

    if kind == "Polygon":
        path = c.beginPath()
        _ring(path, geom.exterior.coords, project)
        for interior in geom.interiors:
            _ring(path, interior.coords, project)
        if fill is not None:
            c.setFillColor(fill)
        c.drawPath(path, stroke=1, fill=1 if fill is not None else 0)
    elif kind == "LineString":
        path = c.beginPath()
        pts = [project(x, y) for x, y in geom.coords]
        path.moveTo(*pts[0])
        for pt in pts[1:]:
            path.lineTo(*pt)
        c.drawPath(path, stroke=1, fill=0)
    elif kind == "Point":
        x, y = project(geom.x, geom.y)
        c.setFillColor(stroke)
        c.circle(x, y, 1.2, stroke=0, fill=1)


def _ring(path, coords, project) -> None:
    pts = [project(x, y) for x, y in coords]
    if not pts:
        return
    path.moveTo(*pts[0])
    for pt in pts[1:]:
        path.lineTo(*pt)
    path.close()


def _title_block(c, page, number, title, plan, std, scale) -> None:
    width, height = page
    x = width - MARGIN - _f(TITLE_BLOCK_BASE_W)
    y = MARGIN
    w, h = _f(TITLE_BLOCK_BASE_W), height - 2 * MARGIN

    c.setStrokeColor(HexColor("#333333"))
    c.setLineWidth(0.8)
    c.rect(x, y, w, h)

    tx = x + _f(4 * mm)
    ty = y + h - _f(8 * mm)

    def line(text, size=7, gap=4.2 * mm, bold=False, colour="#000000"):
        nonlocal ty
        size = size * _sheet_factor
        gap = gap * _sheet_factor
        c.setFont("Helvetica-Bold" if bold else "Helvetica", size)
        c.setFillColor(HexColor(colour))
        for chunk in _wrap(text, int((w - 8 * mm) / (size * 0.5))):
            c.drawString(tx, ty, chunk)
            ty -= gap
        ty -= 1.2 * mm

    line(plan.brief.project_name, size=10, bold=True)
    line(title, size=8, bold=True)
    line(f"Drawing {number} of 4")
    ty -= 3 * mm
    line(f"Scale 1:{scale:,}" if scale else "Not to scale")
    line(f"CRS {plan.site.crs}")
    line(f"Site area {plan.site.area_ha:,.1f} ha")
    line(f"Design population {plan.brief.design_population:,}")
    line(f"Plots {plan.cadastre.total_plots:,}")
    ty -= 3 * mm
    line("Ruleset", bold=True)
    line(f"{std.data['meta']['jurisdiction']} "
         f"v{std.data['meta']['ruleset_version']}")

    if not std.is_validated:
        ty -= 2 * mm
        line("NOT FOR STATUTORY USE", size=8, bold=True, colour="#B22222")
        line("Generated from an unvalidated placeholder ruleset. Standards "
             "have not been confirmed against Botswana planning legislation "
             "by a registered planner.", size=6, gap=3.2 * mm,
             colour="#B22222")

    c.setFont("Helvetica", 6 * _sheet_factor)
    c.setFillColor(HexColor("#666666"))
    c.drawString(tx, y + _f(5 * mm), "Generated plan - not a survey diagram")


def _north_point(c, page) -> None:
    fx0, fy0, fx1, fy1 = _frame(page)
    x, y = fx1 - _f(14 * mm), fy1 - _f(20 * mm)
    c.setStrokeColor(HexColor("#000000"))
    c.setFillColor(HexColor("#000000"))
    c.setLineWidth(0.8)
    path = c.beginPath()
    path.moveTo(x, y + _f(9 * mm))
    path.lineTo(x - _f(3 * mm), y - _f(4 * mm))
    path.lineTo(x, y - _f(1.5 * mm))
    path.lineTo(x + _f(3 * mm), y - _f(4 * mm))
    path.close()
    c.drawPath(path, stroke=1, fill=1)
    c.setFont("Helvetica-Bold", 8 * _sheet_factor)
    c.drawCentredString(x, y + _f(11 * mm), "N")


def _scale_bar(c, page, scale: int, factor: float) -> None:
    """A drawn bar as well as a printed ratio.

    The ratio is only true at the size the sheet was made. A bar stays true
    through every photocopy and reduction the drawing will go through.
    """
    fx0, fy0, _, _ = _frame(page)
    x, y = fx0 + _f(8 * mm), fy0 + _f(9 * mm)

    # A round number of metres that draws to a sensible bar length.
    for ground in (50, 100, 200, 250, 500, 1000, 2000):
        if ground * factor >= _f(30 * mm):
            break
    length = ground * factor
    segments = 4
    seg = length / segments

    c.setLineWidth(0.6)
    c.setStrokeColor(HexColor("#000000"))
    for i in range(segments):
        c.setFillColor(HexColor("#000000") if i % 2 == 0
                       else HexColor("#ffffff"))
        c.rect(x + i * seg, y, seg, _f(2.2 * mm), stroke=1, fill=1)

    c.setFont("Helvetica", 6 * _sheet_factor)
    c.setFillColor(HexColor("#000000"))
    c.drawString(x, y - _f(4 * mm), "0")
    c.drawRightString(x + length, y - _f(4 * mm), f"{ground} m")
    c.drawString(x, y + _f(3.4 * mm), f"Scale 1:{scale:,}")


def _legend(c, page, names) -> None:
    fx0, fy0, _, fy1 = _frame(page)
    x, y = fx0 + _f(8 * mm), fy1 - _f(10 * mm)
    c.setFont("Helvetica", 6.5 * _sheet_factor)

    # The zones layer is drawn in five colours, so one swatch labelled "zones"
    # explains nothing. Expand it into the land uses it actually shows.
    entries: list[tuple[str, object, object]] = []
    for name in names:
        if name == "zones":
            for use, colour in ZONE_FILL.items():
                entries.append((use.replace("_", " "), colour,
                                STYLE["zones"][1]))
            continue
        if name in STYLE:
            fill, stroke, _w = STYLE[name]
            entries.append((name.replace("_", " "), fill, stroke))

    for label, fill, stroke in entries:
        c.setStrokeColor(stroke)
        c.setFillColor(fill if fill is not None else HexColor("#ffffff"))
        c.rect(x, y, _f(4 * mm), _f(2.6 * mm), stroke=1, fill=1)
        c.setFillColor(HexColor("#000000"))
        c.drawString(x + _f(6 * mm), y + _f(0.4 * mm), label)
        y -= _f(5 * mm)


def _schedule_sheet(c, page, plan, report, number, title, std) -> None:
    from .schedules import land_budget, services_schedule

    fx0, fy0, fx1, fy1 = _frame(page)
    y = fy1 - _f(12 * mm)

    def heading(text):
        nonlocal y
        c.setFont("Helvetica-Bold", 9 * _sheet_factor)
        c.setFillColor(HexColor("#000000"))
        c.drawString(fx0 + _f(6 * mm), y, text)
        y -= _f(6 * mm)

    def table(rows, columns, widths):
        nonlocal y
        c.setFont("Helvetica-Bold", 6.5 * _sheet_factor)
        x = fx0 + _f(6 * mm)
        for col, w in zip(columns, widths):
            c.drawString(x, y, col.replace("_", " "))
            x += w
        y -= _f(3.6 * mm)
        c.setStrokeColor(HexColor("#999999"))
        c.setLineWidth(0.3)
        c.line(fx0 + _f(6 * mm), y + _f(1 * mm), fx1 - _f(6 * mm), y + _f(1 * mm))
        y -= _f(1.5 * mm)

        c.setFont("Helvetica", 6.5 * _sheet_factor)
        for row in rows:
            if y < fy0 + _f(14 * mm):
                break
            x = fx0 + _f(6 * mm)
            for col, w in zip(columns, widths):
                c.drawString(x, y, str(row.get(col, ""))[:26])
                x += w
            y -= _f(3.6 * mm)
        y -= _f(5 * mm)

    heading("Land budget")
    table(land_budget(plan),
          ["land_use", "target_ha", "achieved_ha", "variance_ha",
           "variance_pct"],
          [_f(46 * mm)] + [_f(26 * mm)] * 4)

    heading("Engineering services quantities")
    table(services_schedule(plan), ["service", "size_mm", "length_m", "unit"],
          [_f(46 * mm)] + [_f(26 * mm)] * 3)

    heading("Facility provision")
    rows = [
        {"facility": r["key"].replace("_", " "), "sited": r["sited"],
         "unplaced": r["unplaced"], "area_ha": r["area_ha"],
         "coverage_pct": r["coverage_pct"] if r["coverage_pct"] is not None
         else "n/a"}
        for r in plan.facility_plan.summary()["by_type"]
    ]
    table(rows, ["facility", "sited", "unplaced", "area_ha", "coverage_pct"],
          [_f(46 * mm)] + [_f(26 * mm)] * 4)

    _title_block(c, page, number, title, plan, std, 0)


def _wrap(text: str, width: int) -> list[str]:
    words, lines, line = text.split(), [], ""
    for word in words:
        if len(line) + len(word) + 1 > width and line:
            lines.append(line)
            line = word
        else:
            line = f"{line} {word}".strip()
    if line:
        lines.append(line)
    return lines or [""]
