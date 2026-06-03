"""ARCO-ERA5 weather backend for bulk historical training (no API key, no rate
limits).

Open-Meteo's archive API rate-limits hard, which caps training scale. ARCO-ERA5
(Google's Analysis-Ready Cloud-Optimized ERA5, public Zarr) has no key and no
limits. Its catch: it is *time-chunked* (one chunk per hour, global), so a
single-point long series is pathologically slow — but reading a whole REGION for
a time range is fast. So we pull a region once, aggregate hourly -> daily, cache
the compact daily cube locally, and then extract any point instantly. One-time
regional cost, unlimited fast local access afterwards.

ERA5 conventions handled here: longitude 0..360, latitude descending, precip in
metres, temperatures in kelvin, winds as u/v components in m/s.
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import xarray as xr

ERA5_ZARR = ("gs://weatherbench2/datasets/era5/"
             "1959-2023_01_10-full_37-1h-0p25deg-chunk-1.zarr")

# ERA5 var name -> our short name
VARS = {
    "total_precipitation": "precip",
    "2m_temperature": "t2m",
    "2m_dewpoint_temperature": "d2m",
    "10m_u_component_of_wind": "u10",
    "10m_v_component_of_wind": "v10",
    "volumetric_soil_water_layer_2": "swvl2",
}


def _rh_from_dewpoint(t_c: xr.DataArray, td_c: xr.DataArray) -> xr.DataArray:
    """Relative humidity (%) from temperature and dewpoint (both deg C)."""
    a, b = 17.625, 243.04
    return 100.0 * np.exp(a * td_c / (b + td_c)) / np.exp(a * t_c / (b + t_c))


def region_daily(bbox: tuple[float, float, float, float], start: str, end: str,
                 cache_dir: Path, verbose: bool = True) -> xr.Dataset:
    """Daily ERA5 cube for a bbox (min_lon,min_lat,max_lon,max_lat) and period.

    Aggregates: precip daily sum (mm), t2m daily max & mean (deg C), rh daily min
    (%), wind daily max (m/s), soil moisture daily mean (m3/m3). Cached to a
    local Zarr so the (slow) cloud read happens only once per region/period.
    """
    minlon, minlat, maxlon, maxlat = bbox
    tag = f"era5_{minlon}_{minlat}_{maxlon}_{maxlat}_{start}_{end}".replace(".", "p")
    cache = Path(cache_dir) / f"{tag}.zarr"
    if cache.exists():
        if verbose:
            print(f"ERA5 daily cube (cached): {cache.name}")
        return xr.open_zarr(cache)

    if verbose:
        print(f"ERA5: pulling region {bbox} {start}..{end} from ARCO (one-time)…")
    ds = xr.open_zarr(ERA5_ZARR, storage_options={"token": "anon"},
                      chunks={"time": 24 * 14})
    ds = ds[list(VARS)].rename(VARS)
    # ERA5 longitude is 0..360; convert our western-hemisphere bbox.
    lon0, lon1 = minlon % 360, maxlon % 360
    sub = ds.sel(latitude=slice(maxlat, minlat), longitude=slice(lon0, lon1),
                 time=slice(start, end))

    # Process month-by-month to bound memory; dask parallelises the chunk reads.
    months = pd.date_range(start, end, freq="MS")
    daily_parts = []
    for i, m0 in enumerate(months):
        m1 = (m0 + pd.offsets.MonthEnd(1))
        chunk = sub.sel(time=slice(m0.strftime("%Y-%m-%d"), m1.strftime("%Y-%m-%d")))
        if chunk.sizes.get("time", 0) == 0:
            continue
        tC = chunk["t2m"] - 273.15
        tdC = chunk["d2m"] - 273.15
        rh = _rh_from_dewpoint(tC, tdC)
        wind = np.sqrt(chunk["u10"] ** 2 + chunk["v10"] ** 2)
        part = xr.Dataset({
            "precip_mm": (chunk["precip"] * 1000.0).resample(time="1D").sum(),
            "t2m_max": tC.resample(time="1D").max(),
            "t2m_mean": tC.resample(time="1D").mean(),
            "rh_min": rh.resample(time="1D").min(),
            "wind_max": wind.resample(time="1D").max(),
            "soil": chunk["swvl2"].resample(time="1D").mean(),
        }).load()
        daily_parts.append(part)
        if verbose:
            print(f"  {m0.strftime('%Y-%m')}  ({i+1}/{len(months)})")

    daily = xr.concat(daily_parts, dim="time")
    daily.to_zarr(cache, mode="w")
    if verbose:
        print(f"ERA5 daily cube cached -> {cache.name}")
    return daily


def point_daily(cube: xr.Dataset, lat: float, lon: float) -> pd.DataFrame:
    """Extract a point's daily series from a cached region cube (instant)."""
    p = cube.sel(latitude=lat, longitude=lon % 360, method="nearest")
    df = p.to_dataframe().reset_index()
    keep = ["time", "precip_mm", "t2m_max", "t2m_mean", "rh_min", "wind_max", "soil"]
    return df[[c for c in keep if c in df.columns]].set_index("time")
