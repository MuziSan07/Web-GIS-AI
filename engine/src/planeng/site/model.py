"""The site: everything the engine knows about the land before planning starts.

A Site couples a boundary polygon with a raster grid and any number of named
constraint layers. All rasters share one transform and one CRS, so any two
layers can be combined cell-for-cell without resampling surprises.

Real client data and synthetic test data both arrive as a Site, which is what
lets the whole pipeline run today and switch to survey data unchanged.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
from affine import Affine
from rasterio import features
from shapely.geometry import MultiPolygon, Polygon, mapping

from ..crs import require_projected


@dataclass
class Site:
    name: str
    boundary: Polygon | MultiPolygon
    crs: str
    transform: Affine
    shape: tuple[int, int]           # (rows, cols)
    dem: np.ndarray                  # metres above datum
    layers: dict[str, object] = field(default_factory=dict)   # vector constraints
    rasters: dict[str, np.ndarray] = field(default_factory=dict)

    def __post_init__(self) -> None:
        require_projected(self.crs)
        if self.dem.shape != self.shape:
            raise ValueError(
                f"DEM shape {self.dem.shape} does not match site grid {self.shape}"
            )

    # -- grid geometry -----------------------------------------------------
    @property
    def cell_size(self) -> float:
        """Cell edge length in metres. The grid is always square."""
        return abs(self.transform.a)

    @property
    def cell_area_m2(self) -> float:
        return self.cell_size ** 2

    @property
    def cell_area_ha(self) -> float:
        return self.cell_area_m2 / 10_000

    @property
    def area_ha(self) -> float:
        """True boundary area, not the raster bounding box."""
        return self.boundary.area / 10_000

    def mask(self) -> np.ndarray:
        """Boolean array, True inside the site boundary."""
        if "_site" not in self.rasters:
            self.rasters["_site"] = self.rasterize(self.boundary)
        return self.rasters["_site"]

    def rasterize(self, geom) -> np.ndarray:
        """Burn a geometry onto the site grid."""
        if geom is None or geom.is_empty:
            return np.zeros(self.shape, dtype=bool)
        shapes = [(mapping(geom), 1)]
        burned = features.rasterize(
            shapes, out_shape=self.shape, transform=self.transform,
            fill=0, dtype="uint8", all_touched=False,
        )
        return burned.astype(bool)

    def cells_to_ha(self, n_cells: int | float) -> float:
        return float(n_cells) * self.cell_area_ha

    def ha_to_cells(self, ha: float) -> int:
        return int(round(ha / self.cell_area_ha))

    def summary(self) -> dict:
        return {
            "name": self.name,
            "crs": self.crs,
            "area_ha": round(self.area_ha, 1),
            "grid": f"{self.shape[0]} x {self.shape[1]} @ {self.cell_size:.0f} m",
            "cells_in_site": int(self.mask().sum()),
            "elevation_min_m": round(float(np.nanmin(self.dem[self.mask()])), 1),
            "elevation_max_m": round(float(np.nanmax(self.dem[self.mask()])), 1),
            "layers": sorted(self.layers.keys()),
        }
