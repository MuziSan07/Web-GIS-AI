"""Master Plan Copilot, as a hosted demo.

A second front end over the same engine, for a client who needs a link rather
than an install. Nothing is reimplemented here: this calls `generate_master_plan`,
`compare` and `export_all` exactly as the web application does, so the two
cannot disagree about what a plan is.

Three things are different because a shared, hosted, single container demo is
not a workstation.

**Work is cached, not repeated.** A hosted demo gets refreshed, re-run and
shared, and a browser refresh in Streamlit re-runs the whole script from the
top. Without caching, every stray interaction rebuilds a master plan, and the
container has one CPU to give.

**The model runs in the cloud.** There is no GPU here and nothing to pull, so
the brief is read by a hosted model. If no key is configured the engine falls
back to its rule based reader and the page says so rather than failing.

**Detail is capped.** The finest grid is honest work on a workstation and an out
of memory kill on a small container, so the demo offers the two coarser grids
and says why.
"""

from __future__ import annotations

import json
import math
import pathlib
import tempfile
import time

import pydeck as pdk
import streamlit as st

from planeng.agents import provider, understand
from planeng.brief.models import PlanningBrief
from planeng.export.bundle import export_all
from planeng.pipeline import generate_master_plan
from planeng.scenarios import compare
from planeng.standards import load_standards

st.set_page_config(page_title="Master Plan Copilot", page_icon=None,
                   layout="wide", initial_sidebar_state="expanded")

EXAMPLES = [
    "Plan a sustainable 600 hectare settlement for 25,000 people. Use high "
    "density plots. Provide 6 primary schools and 2 secondary schools, "
    "clinics, police and fire stations. Strictly avoid environmentally "
    "sensitive areas and keep infrastructure cost low.",
    "Design a low density 600 hectare town for 12,000 residents with 3 primary "
    "schools and no industrial area.",
    "Create a compact 450 hectare settlement for 18,000 people that is easy to "
    "walk around and cheap to service.",
    "Plan 600 hectares providing 5,000 residential plots, a market, 2 secondary "
    "schools and strong environmental protection.",
]

# Colours match the drawings the engine exports, so the screen and the plan set
# are the same document.
FILL = {
    "residential": [246, 226, 179], "commercial": [232, 183, 183],
    "industrial": [217, 194, 224], "institutional": [185, 211, 232],
    "recreation_open_space": [198, 224, 180],
}
SERVICE_COLOUR = {"water": [31, 120, 180], "sewer": [181, 118, 47],
                  "electricity": [230, 171, 2]}


# ---------------------------------------------------------------------------
# engine calls, cached
# ---------------------------------------------------------------------------

@st.cache_data(show_spinner=False, ttl=3600, max_entries=6)
def read_brief(prompt: str) -> dict:
    """Read the brief with the model, returning plain data.

    Plain data because Streamlit caches by pickling, and a cache that has to
    serialise engine objects fails on the first one that holds a geometry.
    """
    result = understand(prompt)
    return {
        "brief": result.brief.model_dump(mode="json"),
        "source": result.source,
        "model": result.model,
        "assumptions": result.assumptions,
        "clarifications": result.clarifications_needed,
        "warnings": result.warnings,
        "duration_s": result.duration_s,
    }


@st.cache_resource(show_spinner=False, ttl=3600, max_entries=3)
def build_plan(brief_json: str, cell_size: float):
    """Generate a master plan. Cached as a resource, not data.

    A plan holds numpy arrays and shapely geometry. `cache_resource` hands back
    the same object rather than a pickled copy, which is both faster and the
    only thing that works for geometry.
    """
    brief = PlanningBrief.model_validate_json(brief_json)
    return generate_master_plan(brief, cell_size=cell_size)


@st.cache_data(show_spinner=False, ttl=3600, max_entries=3)
def build_scenarios(brief_json: str, cell_size: float) -> dict:
    brief = PlanningBrief.model_validate_json(brief_json)
    return compare(brief, cell_size=cell_size).summary()


@st.cache_data(show_spinner=False, ttl=3600, max_entries=2)
def build_files(brief_json: str, cell_size: float) -> tuple[bytes, dict]:
    """Write every export format and return the archive bytes.

    Written to a temporary directory and read back, because the container has
    no durable disk and the browser needs the bytes anyway.
    """
    brief = PlanningBrief.model_validate_json(brief_json)
    plan = generate_master_plan(brief, cell_size=cell_size)
    with tempfile.TemporaryDirectory() as tmp:
        target = pathlib.Path(tmp) / "delivery"
        result = export_all(plan, target, ifc_plot_cap=300)
        archive = result.archive.read_bytes() if result.archive else b""
    return archive, result.summary()


# ---------------------------------------------------------------------------
# map
# ---------------------------------------------------------------------------

def deck(geojson: dict, show_services: bool) -> pdk.Deck:
    """The plan as map layers, drawn back to front.

    Colours are written onto each feature rather than passed as a function.
    A pydeck layer is serialised to JSON and evaluated in the browser, so a
    Python callable does not travel: it serialises to an empty object, deck.gl
    receives no colour, and the map renders nothing at all with no error to
    explain why. String accessors like `properties.fill_color` do travel.
    """
    features = geojson["features"]

    def tagged(layer: str, fill=None, line=None, **extra) -> dict:
        out = []
        for f in features:
            props = f["properties"]
            if props.get("layer") != layer:
                continue
            if any(props.get(k) != v for k, v in extra.items()):
                continue
            copy = {**f, "properties": dict(props)}
            if fill is not None:
                copy["properties"]["fill_color"] = (
                    fill(props) if callable(fill) else fill)
            if line is not None:
                copy["properties"]["line_color"] = (
                    line(props) if callable(line) else line)
            out.append(copy)
        return {"type": "FeatureCollection", "features": out}

    def zone_fill(props) -> list:
        return FILL.get(props.get("use"), [140, 140, 140])

    def poly(data, width=0.0, pickable=False, opacity=0.85):
        return pdk.Layer(
            "GeoJsonLayer", data, stroked=width > 0, filled=True,
            get_fill_color="properties.fill_color",
            get_line_color="properties.line_color",
            line_width_min_pixels=width, opacity=opacity, pickable=pickable)

    def line(data, width=1.0, pickable=False):
        return pdk.Layer(
            "GeoJsonLayer", data, stroked=True, filled=False,
            get_line_color="properties.line_color",
            line_width_min_pixels=width, pickable=pickable)

    layers = [
        poly(tagged("zone", fill=zone_fill), opacity=0.75),
        poly(tagged("constraint", fill=[209, 73, 91]), opacity=0.45),
        poly(tagged("road_reserve", fill=[70, 78, 94]), opacity=0.9),
        poly(tagged("block", fill=zone_fill, line=[20, 24, 32]),
             width=0.4, pickable=True, opacity=1.0),
        poly(tagged("facility", fill=[140, 90, 168], line=[10, 12, 18]),
             width=0.6, pickable=True),
        poly(tagged("service_site", fill=[152, 78, 163], line=[230, 234, 242]),
             width=1.0, pickable=True),
        line(tagged("road", line=[192, 57, 43]), width=1.3, pickable=True),
        line(tagged("drainage_channel", line=[77, 175, 74]), width=1.6),
    ]

    if show_services:
        for service, colour in SERVICE_COLOUR.items():
            layers.append(line(
                tagged("service_run", line=colour, service=service),
                width=0.9, pickable=True))

    boundary = tagged("boundary", line=[230, 234, 242])
    layers.append(line(boundary, width=2.0))

    centre, zoom = _view(boundary)
    return pdk.Deck(
        layers=layers,
        initial_view_state=pdk.ViewState(longitude=centre[0], latitude=centre[1],
                                         zoom=zoom, pitch=0),
        map_style=None,
        tooltip={"html": "<b>{use}{key}{road_class}{service}</b><br/>"
                         "{block_id}{plot_number}{label}"},
    )


def _view(boundary: dict, width_px: float = 1140.0,
          height_px: float = 520.0) -> tuple[tuple[float, float], float]:
    """Centre and zoom that make the site fill the frame.

    Fitted against the real pixel size of the chart, and against both axes.
    pydeck ships a `compute_view` helper, but it fits to an assumed viewport and
    on this site returns a zoom about one and a half levels too low, which draws
    a three kilometre town as a smudge in the middle of empty basemap.

    Latitude needs the Mercator correction: a degree of longitude at twenty five
    degrees south is about ten percent narrower on screen than a degree of
    latitude, and ignoring that under zooms a tall site.
    """
    xs: list[float] = []
    ys: list[float] = []

    def walk(node) -> None:
        if isinstance(node, (int, float)):
            return
        if node and isinstance(node[0], (int, float)):
            xs.append(node[0])
            ys.append(node[1])
            return
        for child in node:
            walk(child)

    for feature in boundary["features"]:
        walk(feature["geometry"]["coordinates"])

    if not xs:
        return (25.95, -24.93), 11.5

    centre = ((min(xs) + max(xs)) / 2, (min(ys) + max(ys)) / 2)
    span_lon = max(max(xs) - min(xs), 1e-9)
    span_lat = max(max(ys) - min(ys), 1e-9)
    # A degree of latitude covers more screen than a degree of longitude, by
    # one over the cosine of the latitude.
    span_lat_screen = span_lat / max(math.cos(math.radians(centre[1])), 1e-6)

    # 360 degrees spans 256 pixels at zoom 0, and doubles each level.
    zoom_x = math.log2(width_px / 256.0 * 360.0 / span_lon)
    zoom_y = math.log2(height_px / 256.0 * 360.0 / span_lat_screen)
    zoom = min(zoom_x, zoom_y) - 0.35          # a little air around the edge
    return centre, max(3.0, min(zoom, 17.0))


# ---------------------------------------------------------------------------
# page
# ---------------------------------------------------------------------------

def main() -> None:
    st.title("Master Plan Copilot")
    st.caption("Write a planning brief in plain English and get a complete "
               "town master plan: land use, roads, plots, facilities and "
               "engineering services.")

    status = provider.status()
    standards = load_standards("botswana")

    with st.sidebar:
        st.subheader("Planning brief")
        if "prompt" not in st.session_state:
            st.session_state.prompt = EXAMPLES[0]

        prompt = st.text_area("What do you want built?",
                              key="prompt", height=190,
                              label_visibility="collapsed")

        cell_size = st.select_slider(
            "Level of detail",
            options=[25.0, 20.0, 15.0],
            value=20.0,
            format_func=lambda v: {25.0: "Fast", 20.0: "Normal",
                                   15.0: "Fine"}[v],
            help="A finer grid resolves smaller plots and takes longer. The "
                 "finest grid the engine supports is left out of this demo "
                 "because the shared server it runs on does not have the "
                 "memory for it.",
        )

        run = st.button("Generate master plan", type="primary",
                        use_container_width=True)

        st.divider()
        st.caption("Try one of these")
        for i, example in enumerate(EXAMPLES[1:], start=1):
            if st.button(example[:58] + "...", key=f"eg{i}",
                         use_container_width=True):
                st.session_state.prompt = example
                st.rerun()

        st.divider()
        if status["reachable"]:
            st.success(f"Brief read by {status['default_model']} "
                       f"({status['where']})")
        else:
            st.warning("No language model reachable. The brief will be read by "
                       "the rule based reader instead.")

        if standards.is_validated:
            st.success("Planning rules confirmed")
        else:
            st.warning(f"Planning rules not confirmed yet. "
                       f"{len(standards.placeholder_paths())} values still "
                       f"need a registered planner. Output is indicative and "
                       f"is not evidence of statutory compliance.")

    if run or st.session_state.get("has_plan"):
        st.session_state.has_plan = True
        show(prompt, cell_size)
    else:
        st.info("Write a brief on the left, then press Generate master plan. "
                "The first plan takes about half a minute.")


def show(prompt: str, cell_size: float) -> None:
    with st.spinner("Reading the brief"):
        read = read_brief(prompt)
    brief_json = json.dumps(read["brief"])

    with st.spinner("Building the plan. This takes about half a minute."):
        started = time.perf_counter()
        plan = build_plan(brief_json, cell_size)
        elapsed = time.perf_counter() - started

    report = plan.report()

    cols = st.columns(5)
    cols[0].metric("Site", f"{report['land']['gross_site_ha']:,.0f} ha")
    cols[1].metric("People", f"{report['brief']['design_population']:,}")
    cols[2].metric("Plots", f"{report['cadastre']['total_plots']:,}")
    cols[3].metric("Roads", f"{report['roads']['total_length_km']:,.1f} km")
    cols[4].metric(
        "Services",
        f"{report['engineering']['totals']['reticulation_km']:,.0f} km")

    # The map sits above the tabs, not inside one, and that is not only a
    # layout preference. A deck.gl canvas measures its container when it is
    # created, and a tab panel that is not the open one has no size, so a map
    # placed in a tab renders as a forty pixel high empty box with no error.
    # It is also the right place for it: the drawing is the thing being
    # discussed, and it should stay on screen while the reader moves between
    # the report, the options and the files.
    services = st.checkbox("Show water, sewer and power", value=False)
    try:
        st.pydeck_chart(deck(plan.to_geojson(), services), height=520)
    except Exception as exc:
        st.error(f"The map could not be drawn: {exc}")
        st.caption("The plan itself is unaffected. Every number below and "
                   "every exported file is still correct.")

    st.caption(f"Built in {elapsed:,.1f} seconds. "
               f"Read by {read['model'] or 'the rule based reader'}.")

    tab_report, tab_options, tab_files = st.tabs(
        ["Report", "Town shape options", "Files"])

    with tab_report:
        if read["assumptions"]:
            st.write("**What the model took from your brief**")
            for line in read["assumptions"]:
                st.write("- " + line)
        if read["clarifications"]:
            st.write("**What it would ask you**")
            for line in read["clarifications"]:
                st.write("- " + line)

        for warning in report["warnings"][:12]:
            st.warning(warning)

        left, right = st.columns(2)
        with left:
            st.subheader("Land use")
            st.dataframe([
                {"Use": r["use"].replace("_", " "),
                 "Target ha": r["target_ha"], "Actual ha": r["achieved_ha"]}
                for r in report["allocation"]], hide_index=True,
                use_container_width=True)

            st.subheader("Facilities")
            st.dataframe([
                {"Facility": r["key"].replace("_", " "), "Built": r["sited"],
                 "Homes served": ("n/a" if r["coverage_pct"] is None
                                  else f"{r['coverage_pct']:.0f}%")}
                for r in report["facility_siting"]["by_type"]],
                hide_index=True, use_container_width=True)

        with right:
            e = report["engineering"]
            st.subheader("Water, sewer and power")
            st.write(f"Water mains: {e['water']['total_length_km']:,.1f} km, "
                     f"largest {e['water']['metrics']['largest_main_mm']:.0f} mm")
            st.write(f"Sewers: {e['sewer']['total_length_km']:,.1f} km, "
                     f"{e['sewer']['metrics']['manholes']:,} manholes")
            st.write(f"Sewer pump stations: {e['sewer']['pump_stations']}")
            st.write(f"Deepest trench: "
                     f"{e['sewer']['metrics']['max_depth_m']:.2f} m of "
                     f"{e['sewer']['metrics']['depth_limit_m']:.1f} m allowed")
            st.write(f"Power lines: "
                     f"{e['electricity']['total_length_km']:,.1f} km, "
                     f"{e['electricity']['metrics']['distribution_transformers']}"
                     f" transformers")
            st.write(f"Street lights: {e['lighting']['total_poles']:,} poles")

            st.subheader("Stormwater")
            s = e["stormwater"]
            st.write(f"Natural channels: {s['channels']}, "
                     f"{s['channel_length_km']:,.1f} km")
            st.write(f"Road crossings needing culverts: {s['culverts']}")
            st.write(f"Runoff rises from "
                     f"{s['metrics']['predevelopment_coefficient']:.2f} to "
                     f"{s['metrics']['composite_runoff_coefficient']:.2f}")

    with tab_options:
        st.write("The engine builds four different towns on the same land with "
                 "the same population, so any difference in the scores comes "
                 "from the shape alone.")
        if st.button("Compare four town shapes"):
            with st.spinner("Building four plans. This takes a few minutes."):
                result = build_scenarios(brief_json, 25.0)
            st.success(
                f"Best option: {result['recommended'].replace('_', ' ')}")
            st.dataframe([
                {"Shape": s["form"].replace("_", " "), "Score": s["overall"],
                 "Plots": s["plots"], "Roads km": s["road_km"]}
                for s in result["scenarios"]], hide_index=True,
                use_container_width=True)
            for reason in result["recommended_because"]:
                st.caption(reason)

    with tab_files:
        st.write("The plan can be written out in the format each consultant "
                 "works in: GeoPackage for QGIS and ArcGIS, DXF for AutoCAD, "
                 "IFC for Revit, schedules for the surveyor and a printed plan "
                 "set.")
        if st.button("Prepare files"):
            with st.spinner("Writing GIS, CAD, BIM, schedules and the plan set"):
                archive, summary = build_files(brief_json, 25.0)
            st.dataframe([
                {"File": pathlib.Path(
                    o.get("path") or o.get("directory", "")).name,
                 "Format": o.get("format") or "Schedules"}
                for o in summary["outputs"]], hide_index=True,
                use_container_width=True)
            st.download_button(
                "Download everything as a zip", data=archive,
                file_name="master-plan-delivery.zip", mime="application/zip",
                type="primary")
            for note in summary["warnings"][:5]:
                st.caption(note)


if __name__ == "__main__":
    main()
