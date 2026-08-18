"""Tests for the export formats.

What is worth asserting here is not that a file appeared - it is that the file
says the same thing the plan does. An export that loses the coordinate reference
system, truncates a land use, drops a layer or renumbers a plot produces a
perfectly valid file that is wrong, and every one of those has happened in this
module.

One plan is generated for the module and shared. Writing files is the slow part,
so formats are written once into a temporary directory and then inspected.
"""

from __future__ import annotations

import csv
import json
import zipfile

import geopandas as gpd
import pytest

from planeng.brief.models import PlanningBrief
from planeng.export.bundle import export_all
from planeng.export.cad import write_dxf
from planeng.export.layers import build_layers
from planeng.export.plans import STANDARD_SCALES, USABLE_SCALE, write_plan_set
from planeng.export.schedules import beacon_schedule, plot_schedule
from planeng.export.vector import write_geopackage, write_shapefile
from planeng.pipeline import generate_master_plan


@pytest.fixture(scope="module")
def plan():
    return generate_master_plan(
        PlanningBrief(project_name="Export Test", site_area_ha=600,
                      target_population=25000),
        cell_size=20.0,
    )


@pytest.fixture(scope="module")
def layers(plan):
    return build_layers(plan)


@pytest.fixture(scope="module")
def out(tmp_path_factory):
    return tmp_path_factory.mktemp("export")


# -- the layer model --------------------------------------------------------

def test_every_layer_holds_one_geometry_class(layers):
    """A shapefile refuses a mixed layer outright, and a mixed layer is almost
    always an overlay that returned a collection."""
    for name, gdf in layers.non_empty().items():
        assert len(set(gdf.geom_type)) == 1, f"{name} mixes geometry types"


def test_layers_are_in_the_site_projection(layers, plan):
    """Areas, lengths and CAD coordinates are all meaningless in degrees."""
    for gdf in layers.non_empty().values():
        assert gdf.crs.to_string() == plan.site.crs
        assert not gdf.crs.is_geographic


def test_land_use_values_are_not_truncated(layers):
    """The ten character limit is on shapefile field *names*, not values.

    Clipping values too turned recreation_open_space into recreation and every
    consumer matching on the string silently stopped matching.
    """
    uses = set(layers.layers["zones"]["land_use"])
    assert "recreation_open_space" in uses
    assert "residential" in uses


def test_field_names_survive_a_shapefile(layers):
    for name, gdf in layers.non_empty().items():
        for column in gdf.columns:
            if column == "geometry":
                continue
            assert len(column) <= 10, f"{name}.{column} will be truncated"


def test_plot_count_matches_the_plan(layers, plan):
    assert len(layers.layers["parcels"]) == plan.cadastre.total_plots


# -- GeoPackage -------------------------------------------------------------

def test_geopackage_round_trips_every_layer(layers, out):
    path = out / "plan.gpkg"
    result = write_geopackage(layers, path)
    assert path.exists()
    assert result.layers == len(layers.non_empty())

    back = gpd.read_file(path, layer="parcels")
    assert len(back) == len(layers.layers["parcels"])
    assert back.crs.to_string() == layers.crs


def test_geopackage_keeps_plot_numbers_intact(layers, out):
    write_geopackage(layers, out / "plan.gpkg")
    back = gpd.read_file(out / "plan.gpkg", layer="parcels")
    assert set(back["plot_no"]) == set(layers.layers["parcels"]["plot_no"])


# -- shapefile --------------------------------------------------------------

def test_shapefile_writes_a_projection_file(layers, out):
    """A shapefile without a .prj is a set of unlabelled numbers."""
    result = write_shapefile(layers, out / "shp")
    assert result.layers == len(layers.non_empty())
    assert (out / "shp" / "parcels.prj").exists()


def test_shapefile_limitations_are_reported_not_hidden(layers, out):
    result = write_shapefile(layers, out / "shp")
    assert any("null" in w for w in result.warnings)


# -- DXF --------------------------------------------------------------------

def test_dxf_is_written_in_site_coordinates(layers, out):
    """A DXF in degrees opens as a drawing two units across."""
    import ezdxf

    write_dxf(layers, out / "plan.dxf")
    doc = ezdxf.readfile(out / "plan.dxf")
    msp = doc.modelspace()

    minx, miny, maxx, maxy = layers.layers["site_boundary"].total_bounds
    points = []
    for entity in msp.query("LWPOLYLINE")[:50]:
        points.extend(entity.get_points("xy"))
    assert points
    assert any(minx <= x <= maxx and miny <= y <= maxy for x, y in points)


def test_dxf_layers_match_the_data_layers(layers, out):
    import ezdxf

    write_dxf(layers, out / "plan.dxf")
    doc = ezdxf.readfile(out / "plan.dxf")
    names = {layer.dxf.name for layer in doc.layers}
    assert set(layers.non_empty()) <= names


def test_dxf_carries_a_plot_label_per_plot(layers, out):
    import ezdxf

    result = write_dxf(layers, out / "plan.dxf")
    assert result.labels == len(layers.layers["parcels"])
    doc = ezdxf.readfile(out / "plan.dxf")
    assert len(doc.modelspace().query("TEXT[layer==\"parcel_numbers\"]")) \
        == result.labels


def test_dxf_states_units_as_metres(out, layers):
    import ezdxf

    write_dxf(layers, out / "plan.dxf")
    doc = ezdxf.readfile(out / "plan.dxf")
    assert doc.header["$INSUNITS"] == 6


# -- plan set ---------------------------------------------------------------

def test_plan_set_uses_a_standard_scale(plan, out):
    """A drawing at 1:1837 cannot be scaled off with a rule."""
    result = write_plan_set(plan, out / "plan_set.pdf")
    assert result.scale in STANDARD_SCALES


def test_plan_set_picks_a_sheet_that_keeps_the_drawing_legible(plan, out):
    result = write_plan_set(plan, out / "plan_set.pdf")
    if result.scale > USABLE_SCALE:
        assert result.sheet == "A0"
        assert any("coarser" in w for w in result.warnings)
    else:
        assert result.sheet in {"A3", "A2", "A1", "A0"}


def test_plan_set_has_all_four_sheets(plan, out):
    result = write_plan_set(plan, out / "plan_set.pdf")
    assert len(result.sheets) == 4
    assert (out / "plan_set.pdf").stat().st_size > 10_000


def test_an_unvalidated_ruleset_is_stamped_on_the_drawings(plan, out):
    result = write_plan_set(plan, out / "plan_set.pdf")
    if not plan.standards.is_validated:
        assert any("unvalidated" in w for w in result.warnings)


# -- schedules --------------------------------------------------------------

def test_plot_schedule_covers_every_plot(plan):
    rows = plot_schedule(plan)
    assert len(rows) == plan.cadastre.total_plots
    assert {r["plot_number"] for r in rows} == {
        p.plot_number for p in plan.cadastre.parcels}


def test_beacon_schedule_gives_every_plot_its_corners(layers):
    rows = beacon_schedule(layers)
    plots = {r["plot_number"] for r in rows}
    assert plots == set(layers.layers["parcels"]["plot_no"])
    assert all(r["crs"] == layers.crs for r in rows)


def test_beacon_numbering_is_reproducible(layers):
    """A ring can start anywhere and wind either way. Two exports that number
    the same plot differently cannot be checked against each other."""
    first = beacon_schedule(layers)
    second = beacon_schedule(layers)
    assert first == second


def test_beacons_start_at_the_north_west_corner(layers):
    rows = beacon_schedule(layers)
    by_plot: dict[str, list[dict]] = {}
    for row in rows:
        by_plot.setdefault(row["plot_number"], []).append(row)

    for corners in list(by_plot.values())[:200]:
        northmost = max(c["northing_m"] for c in corners)
        assert corners[0]["northing_m"] == northmost


def test_schedules_declare_that_beacons_are_not_surveyed(plan, out):
    from planeng.export.schedules import write_schedules

    result = write_schedules(plan, out / "schedules")
    assert "beacon_coordinates.csv" in result.files
    assert any("surveyor" in w for w in result.warnings)

    with (out / "schedules" / "beacon_coordinates.csv").open(
            encoding="utf-8") as fh:
        rows = list(csv.DictReader(fh))
    assert len(rows) > plan.cadastre.total_plots * 3


# -- IFC --------------------------------------------------------------------

def test_ifc_carries_plots_and_their_controls(plan, out):
    import ifcopenshell

    from planeng.export.bim import write_ifc

    result = write_ifc(plan, out / "plan.ifc", max_plots=60)
    model = ifcopenshell.open(str(out / "plan.ifc"))

    assert len(model.by_type("IfcSpatialZone")) == result.zones
    assert result.zones == 60
    assert model.by_type("IfcProject")
    assert model.by_type("IfcSite")

    psets = {p.Name for p in model.by_type("IfcPropertySet")}
    assert "Pset_DevelopmentControl" in psets
    assert "Pset_PlanningPlot" in psets


def test_ifc_geometry_sits_near_the_local_origin(plan, out):
    """IFC viewers lose centimetres on a seven million metre northing."""
    import ifcopenshell

    from planeng.export.bim import write_ifc

    write_ifc(plan, out / "plan.ifc", max_plots=60)
    model = ifcopenshell.open(str(out / "plan.ifc"))
    for point in model.by_type("IfcCartesianPoint")[:400]:
        for value in point.Coordinates:
            assert abs(value) < 100_000


def test_ifc_records_the_true_origin_so_it_can_be_put_back(plan, out):
    import ifcopenshell

    from planeng.export.bim import write_ifc

    result = write_ifc(plan, out / "plan.ifc", max_plots=20)
    model = ifcopenshell.open(str(out / "plan.ifc"))
    site = model.by_type("IfcSite")[0]
    assert plan.site.crs in site.Description
    assert f"{result.origin[0]:.3f}" in site.Description


# -- the bundle -------------------------------------------------------------

def test_bundle_writes_every_default_format(plan, out):
    result = export_all(plan, out / "delivery", ifc_plot_cap=40)
    names = {o.get("format") for o in result.outputs}
    assert {"GPKG", "DXF", "PDF", "IFC"} <= names
    assert (out / "delivery" / "README.txt").exists()
    assert (out / "delivery" / "layer_manifest.json").exists()


def test_bundle_readme_states_the_crs_and_the_ruleset_status(plan, out):
    export_all(plan, out / "delivery", ifc_plot_cap=40, archive=False)
    readme = (out / "delivery" / "README.txt").read_text(encoding="utf-8")
    assert plan.site.crs in readme
    if not plan.standards.is_validated:
        assert "NOT FOR STATUTORY USE" in readme
    assert "surveyor" in readme


def test_bundle_manifest_lists_fields_per_layer(plan, out, layers):
    export_all(plan, out / "delivery", ifc_plot_cap=40, archive=False)
    manifest = json.loads(
        (out / "delivery" / "layer_manifest.json").read_text(encoding="utf-8"))
    named = {row["layer"] for row in manifest}
    assert named == set(layers.non_empty())
    parcels = next(r for r in manifest if r["layer"] == "parcels")
    assert "plot_no" in parcels["fields"]


def test_bundle_archives_everything_it_wrote(plan, out):
    result = export_all(plan, out / "delivery", ifc_plot_cap=40)
    assert result.archive is not None and result.archive.exists()
    with zipfile.ZipFile(result.archive) as zf:
        names = zf.namelist()
    assert any(n.endswith("README.txt") for n in names)
    assert any(n.endswith(".gpkg") for n in names)
    assert any(n.endswith(".pdf") for n in names)


def test_shapefile_and_filegdb_are_off_by_default(plan, out):
    """They duplicate the GeoPackage and are lossy; a delivery is better for
    not carrying three copies of the same data."""
    result = export_all(plan, out / "default", ifc_plot_cap=20, archive=False)
    formats = {o.get("format") for o in result.outputs}
    assert "ESRI Shapefile" not in formats
    assert "OpenFileGDB" not in formats
