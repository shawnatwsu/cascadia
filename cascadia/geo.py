"""CONUS (contiguous US) geometry — for masking data and drawing clean borders.

Research-grade maps should show only the area we actually model: the contiguous
US, on a white background, with no neighboring countries or ocean tint bleeding
into the colormaps. This module loads the US state polygons (Natural Earth, via
cartopy), drops Alaska/Hawaii/territories, and exposes:

  * `conus_states()` -> list of US state geometries (for outlines)
  * `conus_union()`  -> one (multi)polygon of CONUS (for masking)
  * `in_conus(lons, lats)` -> boolean mask of points inside CONUS land
"""
from __future__ import annotations

from functools import lru_cache

import numpy as np

_EXCLUDE = {"Alaska", "Hawaii", "Puerto Rico", "United States Virgin Islands",
            "Guam", "American Samoa", "Commonwealth of the Northern Mariana Islands"}


@lru_cache(maxsize=1)
def conus_states() -> tuple:
    import cartopy.io.shapereader as shpreader
    path = shpreader.natural_earth(resolution="50m", category="cultural",
                                   name="admin_1_states_provinces_lakes")
    geoms = []
    for rec in shpreader.Reader(path).records():
        a = rec.attributes
        admin = a.get("admin") or a.get("adm0_a3")
        name = a.get("name") or a.get("name_en") or ""
        if admin in ("United States of America", "USA") and name not in _EXCLUDE:
            geoms.append(rec.geometry)
    return tuple(geoms)


@lru_cache(maxsize=1)
def conus_union():
    from shapely.ops import unary_union
    return unary_union(list(conus_states()))


def in_conus(lons, lats) -> np.ndarray:
    """Boolean mask: which (lon, lat) points fall on contiguous-US land."""
    import shapely
    geom = conus_union()
    lons = np.asarray(lons, float); lats = np.asarray(lats, float)
    try:
        return np.asarray(shapely.contains_xy(geom, lons, lats), dtype=bool)
    except Exception:
        from shapely.geometry import Point
        from shapely.prepared import prep
        pg = prep(geom)
        return np.array([pg.contains(Point(x, y)) for x, y in zip(lons, lats)])


def mask_conus(cells):
    """Filter a cells DataFrame (lat/lon) to contiguous-US land."""
    if cells.empty:
        return cells
    keep = in_conus(cells["lon"].to_numpy(), cells["lat"].to_numpy())
    return cells[keep].reset_index(drop=True)
