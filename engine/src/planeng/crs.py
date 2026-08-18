"""Coordinate reference system handling.

All planning computation happens in a projected CRS in metres. Areas, distances,
setbacks and road reserves are meaningless in degrees, so the engine refuses to
do geometric work in a geographic CRS.
"""

from __future__ import annotations

from pyproj import CRS

# Botswana spans UTM zones 34S and 35S. Zone 34S covers 18-24E, zone 35S 24-30E.
# Gaborone (~25.9E) falls in 35S, which is the sensible default for the country.
UTM_34S = "EPSG:32734"
UTM_35S = "EPSG:32735"
WGS84 = "EPSG:4326"

DEFAULT_CRS = UTM_35S


def utm_crs_for_longitude(longitude: float, southern: bool = True) -> str:
    """Return the UTM CRS covering a longitude.

    Used when ingesting client data supplied in geographic coordinates so the
    engine picks a projection appropriate to the site rather than a hardcoded one.
    """
    zone = int((longitude + 180) // 6) + 1
    epsg = (32700 if southern else 32600) + zone
    return f"EPSG:{epsg}"


def require_projected(crs_like) -> CRS:
    """Validate that a CRS is projected and in metres.

    Raises rather than silently producing degree-based areas, which is the most
    common and most damaging error in automated GIS pipelines.
    """
    crs = CRS.from_user_input(crs_like)
    if crs.is_geographic:
        raise ValueError(
            f"Planning computation requires a projected CRS in metres, got "
            f"geographic CRS '{crs.name}'. Reproject the site first "
            f"(see planeng.crs.utm_crs_for_longitude)."
        )
    unit = crs.axis_info[0].unit_name if crs.axis_info else "unknown"
    if unit not in ("metre", "meter"):
        raise ValueError(
            f"Planning computation requires a CRS with metre units, "
            f"got '{unit}' for '{crs.name}'."
        )
    return crs
