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
    "etr": "etr",              # reference ET (for an antecedent-wetness proxy)
}
KELVIN_VARS = {"tmax_c", "tmin_c"}


def region_daily(bbox: tuple[float, float, float, float], start: str, end: str,
                 cache_dir: Path, variables: list[str] | None = None,
                 stride: int = 1, verbose: bool = True) -> xr.Dataset:
    """Daily GRIDMET cube for a bbox (min_lon,min_lat,max_lon,max_lat) + period.

    `stride` coarsens the native 4 km grid (e.g. stride=6 -> ~24 km) so big
    regions like CONUS don't pull ~1 GB when the render grid is coarse anyway.
    Cached to a local NetCDF so the OPeNDAP pull happens once.
    """
    minlon, minlat, maxlon, maxlat = bbox
    variables = variables or list(VARS)
    tag = (f"gridmet_{minlon}_{minlat}_{maxlon}_{maxlat}_{start}_{end}_s{stride}"
           .replace(".", "p"))
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
            if stride > 1:
                da = da.isel(lat=slice(None, None, stride),
                             lon=slice(None, None, stride))
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


def gridmet_window(config) -> tuple[str, str]:
    """Target window for GRIDMET. Historical: the as-of window. Live: the most
    recent ~12 days (GRIDMET has ~5-day latency, so 'now' isn't available)."""
    from datetime import timedelta
    if config.is_historical:
        s, e = config.window()
        return s.date().isoformat(), e.date().isoformat()
    import datetime as _dt
    today = _dt.datetime.utcnow().date()
    return (today - timedelta(days=12)).isoformat(), (today - timedelta(days=2)).isoformat()


def heat_index_c(t_c, rh):
    """NWS heat index (deg C) from temperature (deg C) and RH (%).

    Rothfusz regression (computed in deg F, returned in deg C). Below ~27 deg C
    the heat index is ~the air temperature."""
    import numpy as np
    t = np.asarray(t_c, float) * 9 / 5 + 32
    r = np.clip(np.asarray(rh, float), 0, 100)
    hi = (-42.379 + 2.04901523 * t + 10.14333127 * r - 0.22475541 * t * r
          - 6.83783e-3 * t * t - 5.481717e-2 * r * r + 1.22874e-3 * t * t * r
          + 8.5282e-4 * t * r * r - 1.99e-6 * t * t * r * r)
    hi = np.where(t < 80, t, hi)
    return (hi - 32) * 5 / 9


def wet_bulb_c(t_c, rh):
    """Wet-bulb temperature (deg C) via Stull (2011), from T (deg C) and RH (%)."""
    import numpy as np
    t = np.asarray(t_c, float); r = np.clip(np.asarray(rh, float), 1, 100)
    return (t * np.arctan(0.151977 * np.sqrt(r + 8.313659))
            + np.arctan(t + r) - np.arctan(r - 1.676331)
            + 0.00391838 * r ** 1.5 * np.arctan(0.023101 * r) - 4.686035)


def derive_cell_features(cube: xr.Dataset, cells: pd.DataFrame) -> pd.DataFrame:
    """Aggregate the GRIDMET cube over its time window and sample at each grid
    cell, returning fire- and heat-danger features keyed by cell_id."""
    import numpy as np
    # Window aggregations: peak danger / driest fuels.
    agg = {
        "gm_burning_index": ("burning_index", "max"),
        "gm_erc": ("erc", "max"),
        "gm_fm100": ("fm100", "min"),
        "gm_fm1000": ("fm1000", "min"),
        "gm_vpd": ("vpd_kpa", "max"),
        "gm_tmax": ("tmax_c", "max"),
        "gm_rhmin": ("rh_min", "min"),
        "gm_wind": ("wind_ms", "max"),
    }
    tlat = xr.DataArray(cells["lat"].to_numpy(), dims="cell")
    tlon = xr.DataArray(cells["lon"].to_numpy(), dims="cell")
    out = {"cell_id": cells["cell_id"].to_numpy()}
    for name, (var, how) in agg.items():
        if var not in cube:
            continue
        field = getattr(cube[var], how)("time")
        out[name] = field.sel(lat=tlat, lon=tlon, method="nearest").to_numpy()
    df = pd.DataFrame(out)
    # Heat metrics from peak temperature + concurrent (afternoon ~= min) RH.
    if "gm_tmax" in df and "gm_rhmin" in df:
        df["heat_index_c"] = heat_index_c(df["gm_tmax"], df["gm_rhmin"])
        df["wet_bulb_c"] = wet_bulb_c(df["gm_tmax"], df["gm_rhmin"])
    return df
