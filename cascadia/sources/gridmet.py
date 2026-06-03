"""GRIDMET (gridMET / METDATA) — 4 km daily CONUS meteorology via OPeNDAP.

Finer than ERA5 (~4 km vs ~25 km) and it directly carries fire- and heat-
relevant variables: VPD, burning index, energy release component, dead-fuel
moisture, wind, solar radiation, min/max temperature & RH, specific humidity and
precipitation. OPeNDAP does server-side spatial/temporal subsetting, so regional
extraction is fast (a region-month is seconds) — unlike the time-chunked ERA5
Zarr. Free, no API key. CONUS-only, daily, 1979-present (~5-day latency).

Each variable is a separate aggregated file; we subset the region+period from
each, take its primary data variable, and merge into one daily cube (cached).
"""
from __future__ import annotations

from pathlib import Path

import pandas as pd
import xarray as xr

THREDDS = ("http://thredds.northwestknowledge.net:8080/thredds/dodsC/"
           "agg_met_{code}_1979_CurrentYear_CONUS.nc")

# our short name -> GRIDMET file code
VARS: dict[str, str] = {
    "precip_mm": "pr",
    "tmax_c": "tmmx",          # kelvin -> C below
    "tmin_c": "tmmn",
    "rh_max": "rmax",
    "rh_min": "rmin",
    "wind_ms": "vs",
    "srad_wm2": "srad",
    "vpd_kpa": "vpd",
    "burning_index": "bi",
    "erc": "erc",
    "fm100": "fm100",
    "fm1000": "fm1000",
    "sph": "sph",              # specific humidity (for wet-bulb)
}
KELVIN_VARS = {"tmax_c", "tmin_c"}


def region_daily(bbox: tuple[float, float, float, float], start: str, end: str,
                 cache_dir: Path, variables: list[str] | None = None,
                 verbose: bool = True) -> xr.Dataset:
    """Daily GRIDMET cube for a bbox (min_lon,min_lat,max_lon,max_lat) + period.

    Cached to a local NetCDF so the (already fast) OPeNDAP pull happens once.
    """
    minlon, minlat, maxlon, maxlat = bbox
    variables = variables or list(VARS)
    tag = f"gridmet_{minlon}_{minlat}_{maxlon}_{maxlat}_{start}_{end}".replace(".", "p")
    cache = Path(cache_dir) / f"{tag}.nc"
    if cache.exists():
        if verbose:
            print(f"GRIDMET cube (cached): {cache.name}")
        return xr.open_dataset(cache)

    data = {}
    for short in variables:
        code = VARS[short]
        url = THREDDS.format(code=code)
        try:
            ds = xr.open_dataset(url, engine="pydap")
            primary = [v for v in ds.data_vars if v != "crs"][0]
            da = ds[primary].sel(lat=slice(maxlat, minlat),
                                 lon=slice(minlon, maxlon),
                                 day=slice(start, end))
            if short in KELVIN_VARS:
                da = da - 273.15
            data[short] = da.load()
            if verbose:
                print(f"  GRIDMET {short:<13} ({code}) {dict(da.sizes)}")
        except Exception as exc:
            if verbose:
                print(f"  GRIDMET {short} FAILED: {repr(exc)[:120]}")

    cube = xr.Dataset(data).rename({"day": "time"})
    try:
        cube.to_netcdf(cache)
    except Exception:
        pass
    return cube


def point_daily(cube: xr.Dataset, lat: float, lon: float) -> pd.DataFrame:
    """Extract a point's daily series from a region cube (instant, local)."""
    p = cube.sel(lat=lat, lon=lon, method="nearest")
    return p.to_dataframe().reset_index().set_index("time")
