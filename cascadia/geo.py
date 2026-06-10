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

# 5th National Climate Assessment (NCA5) CONUS regions -> member states.
NCA5_REGIONS: dict[str, tuple] = {
    "northwest": ("Washington", "Oregon", "Idaho"),
    "southwest": ("California", "Nevada", "Arizona", "New Mexico", "Utah", "Colorado"),
    "northern_great_plains": ("Montana", "Wyoming", "North Dakota", "South Dakota",
                              "Nebraska"),
    "southern_great_plains": ("Kansas", "Oklahoma", "Texas"),
    "midwest": ("Minnesota", "Iowa", "Missouri", "Wisconsin", "Illinois", "Indiana",
                "Michigan", "Ohio"),
    "southeast": ("Virginia", "Kentucky", "Tennessee", "North Carolina",
                  "South Carolina", "Georgia", "Florida", "Alabama", "Mississippi",
                  "Louisiana", "Arkansas"),
    "northeast": ("West Virginia", "Maryland", "Delaware", "New Jersey",
                  "Pennsylvania", "New York", "Connecticut", "Rhode Island",
                  "Massachusetts", "Vermont", "New Hampshire", "Maine",
                  "District of Columbia"),
}
NCA5_NAMES = {
    "northwest": "Northwest", "southwest": "Southwest",
    "northern_great_plains": "Northern Great Plains",
    "southern_great_plains": "Southern Great Plains", "midwest": "Midwest",
    "southeast": "Southeast", "northeast": "Northeast",
}


@lru_cache(maxsize=1)
def state_geoms() -> dict:
    """{state name -> geometry} for the contiguous US (+ DC)."""
    import cartopy.io.shapereader as shpreader
    # Non-lakes version so Great-Lakes-shore land (Chicago, Detroit, Cleveland)
    # is correctly inside its state rather than falling in a lake gap.
    path = shpreader.natural_earth(resolution="50m", category="cultural",
                                   name="admin_1_states_provinces")
    out = {}
    for rec in shpreader.Reader(path).records():
        a = rec.attributes
        admin = a.get("admin") or a.get("adm0_a3")
        name = a.get("name") or a.get("name_en") or ""
        if admin in ("United States of America", "USA") and name not in _EXCLUDE:
            out[name] = rec.geometry
    return out


@lru_cache(maxsize=1)
def conus_states() -> tuple:
    return tuple(state_geoms().values())


def region_states(region_key: str) -> tuple:
    """State geometries for an NCA5 region (or all CONUS)."""
    sg = state_geoms()
    if region_key in NCA5_REGIONS:
        return tuple(sg[s] for s in NCA5_REGIONS[region_key] if s in sg)
    return conus_states()


@lru_cache(maxsize=16)
def region_geometry(region_key: str):
    from shapely.ops import unary_union
    return unary_union(list(region_states(region_key)))


def region_bbox(region_key: str, pad: float = 0.3) -> tuple:
    """Bounding box (min_lon,min_lat,max_lon,max_lat) of a region, padded."""
    minx, miny, maxx, maxy = region_geometry(region_key).bounds
    return (minx - pad, miny - pad, maxx + pad, maxy + pad)


def mask_region(cells, region_key: str):
    """Filter cells to a specific NCA5 region's states (clean state borders)."""
    if cells.empty:
        return cells
    import shapely
    geom = region_geometry(region_key)
    lons = cells["lon"].to_numpy(); lats = cells["lat"].to_numpy()
    try:
        keep = np.asarray(shapely.contains_xy(geom, lons, lats), dtype=bool)
    except Exception:
        from shapely.geometry import Point
        from shapely.prepared import prep
        pg = prep(geom)
        keep = np.array([pg.contains(Point(x, y)) for x, y in zip(lons, lats)])
    return cells[keep].reset_index(drop=True)


def _norm(s: str) -> str:
    return s.strip().lower().replace("_", " ").replace("-", " ")


@lru_cache(maxsize=1)
def _state_lookup() -> dict:
    """{normalized key -> proper state name} for the 48 CONUS states + DC.
    Accepts names ('texas', 'new_york') and USPS codes ('tx', 'ny')."""
    codes = {"alabama": "al", "arizona": "az", "arkansas": "ar", "california": "ca",
             "colorado": "co", "connecticut": "ct", "delaware": "de",
             "district of columbia": "dc", "florida": "fl", "georgia": "ga",
             "idaho": "id", "illinois": "il", "indiana": "in", "iowa": "ia",
             "kansas": "ks", "kentucky": "ky", "louisiana": "la", "maine": "me",
             "maryland": "md", "massachusetts": "ma", "michigan": "mi",
             "minnesota": "mn", "mississippi": "ms", "missouri": "mo", "montana": "mt",
             "nebraska": "ne", "nevada": "nv", "new hampshire": "nh", "new jersey": "nj",
             "new mexico": "nm", "new york": "ny", "north carolina": "nc",
             "north dakota": "nd", "ohio": "oh", "oklahoma": "ok", "oregon": "or",
             "pennsylvania": "pa", "rhode island": "ri", "south carolina": "sc",
             "south dakota": "sd", "tennessee": "tn", "texas": "tx", "utah": "ut",
             "vermont": "vt", "virginia": "va", "washington": "wa",
             "west virginia": "wv", "wisconsin": "wi", "wyoming": "wy"}
    out = {}
    for name in state_geoms():           # only CONUS states (AK/HI excluded)
        key = _norm(name)
        out[key] = name
        if key in codes:
            out[codes[key]] = name
    return out


def resolve_state(key: str) -> str | None:
    """Proper CONUS state name for a user key ('texas', 'tx', 'New Mexico'), else None."""
    return _state_lookup().get(_norm(key))


def state_geometry(name: str):
    return state_geoms()[name]


def state_bbox(name: str, pad: float = 0.3) -> tuple:
    minx, miny, maxx, maxy = state_geoms()[name].bounds
    return (minx - pad, miny - pad, maxx + pad, maxy + pad)


def mask_state(cells, name: str):
    """Filter cells to a single state's polygon (clean borders)."""
    if cells.empty:
        return cells
    import shapely
    geom = state_geoms()[name]
    lons = cells["lon"].to_numpy(); lats = cells["lat"].to_numpy()
    try:
        keep = np.asarray(shapely.contains_xy(geom, lons, lats), dtype=bool)
    except Exception:
        from shapely.geometry import Point
        from shapely.prepared import prep
        pg = prep(geom)
        keep = np.array([pg.contains(Point(x, y)) for x, y in zip(lons, lats)])
    return cells[keep].reset_index(drop=True)


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
