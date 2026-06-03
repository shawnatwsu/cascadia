"""Assemble a labelled wildfire-occurrence dataset from open data.

Labels come from NASA FIRMS active-fire detections (needs a free FIRMS_MAP_KEY);
features come from the Open-Meteo ERA5 archive — the SAME quantities the live
engine computes per cell (forward-window precip, peak deep soil moisture, peak
temperature), so there is no train/serve mismatch.

Method (mirrors the flood dataset):
  * sample a coarse grid of points across the region,
  * pull all FIRMS detections for the region over the fire seasons (chunked),
  * for each (point, sampled date) build features over the horizon window and a
    label = 1 if any detection falls within ~cell radius and the window.
"""
from __future__ import annotations

import io
from datetime import date, datetime, timedelta

import numpy as np
import pandas as pd
import requests

from ..config import Config
from ..sources.base import fetch_json

ARCHIVE_URL = "https://archive-api.open-meteo.com/v1/archive"
FIRMS_AREA = "https://firms.modaps.eosdis.nasa.gov/api/area/csv"

FIRE_FEATURES = ["precip_total_mm", "soil_moist_peak", "temp_max"]


def firms_key(config: Config) -> str | None:
    return config.env(config.sources.get("firms", {}).get("map_key_env", "FIRMS_MAP_KEY"))


# Historical labels need a Standard-Processing (archive) product; the live NRT
# product only covers ~the last 2 months. VIIRS_SNPP_SP goes back to 2012.
ARCHIVE_FIRE_SOURCE = "VIIRS_SNPP_SP"


def fetch_fire_detections(config: Config, start: str, end: str,
                          season: tuple[int, int] = (4, 10),
                          source: str = ARCHIVE_FIRE_SOURCE,
                          verbose: bool = True) -> pd.DataFrame:
    """All FIRMS detections in the region between start and end (chunked <=10d).

    Only fetches chunks whose start falls in the fire season, to conserve FIRMS
    transactions (the off-season has effectively no detections anyway).
    """
    key = firms_key(config)
    if not key:
        return pd.DataFrame(columns=["lat", "lon", "date"])
    r = config.region
    bbox = f"{r.min_lon},{r.min_lat},{r.max_lon},{r.max_lat}"
    # Cache the assembled detections so re-runs (e.g. to re-label) are instant
    # and don't re-spend FIRMS transactions.
    cache_file = (config.cache_dir /
                  f"fires_{source}_{bbox}_{start}_{end}.pkl".replace("/", "_"))
    if cache_file.exists():
        out = pd.read_pickle(cache_file)
        if verbose:
            print(f"FIRMS detections {start}..{end}: {len(out)} (cached)")
        return out
    d0, d1 = date.fromisoformat(start), date.fromisoformat(end)
    chunk = 5  # FIRMS area API caps day_range at 5
    frames, cur, n_req = [], d0, 0
    while cur <= d1:
        if not (season[0] <= cur.month <= season[1]):
            cur += timedelta(days=chunk)
            continue
        n_req += 1
        url = f"{FIRMS_AREA}/{key}/{source}/{bbox}/{chunk}/{cur.isoformat()}"
        try:
            txt = requests.get(url, timeout=90).text
            df = pd.read_csv(io.StringIO(txt))
            if not df.empty and "latitude" in df:
                frames.append(pd.DataFrame({
                    "lat": df["latitude"].astype(float),
                    "lon": df["longitude"].astype(float),
                    "date": pd.to_datetime(df["acq_date"], errors="coerce"),
                }))
        except Exception as exc:
            if verbose:
                print(f"  FIRMS chunk {cur} failed: {exc}")
        cur += timedelta(days=chunk)
    out = pd.concat(frames, ignore_index=True) if frames else pd.DataFrame(
        columns=["lat", "lon", "date"])
    try:
        out.to_pickle(cache_file)
    except Exception:
        pass  # caching is best-effort
    if verbose:
        print(f"FIRMS detections {start}..{end}: {len(out)} ({n_req} season requests)")
    return out


def _sample_points(config: Config, step: float = 0.4) -> list[tuple[float, float]]:
    r = config.region
    lats = np.arange(r.min_lat, r.max_lat + 1e-9, step)
    lons = np.arange(r.min_lon, r.max_lon + 1e-9, step)
    return [(round(float(la), 3), round(float(lo), 3)) for la in lats for lo in lons]


def _weather_daily(lat: float, lon: float, start: str, end: str,
                   config: Config) -> pd.DataFrame:
    params = {
        "latitude": round(lat, 3), "longitude": round(lon, 3),
        "start_date": start, "end_date": end,
        "hourly": "precipitation,temperature_2m,soil_moisture_28_to_100cm",
        "timezone": "UTC",
    }
    data = fetch_json(ARCHIVE_URL, params=params, cache_dir=config.cache_dir,
                      cache_ttl_s=30 * 86400)
    h = data.get("hourly", {})
    t = h.get("time", [])
    if not t:
        return pd.DataFrame()
    df = pd.DataFrame({
        "time": pd.to_datetime(t),
        "precip": h.get("precipitation", [np.nan] * len(t)),
        "temp": h.get("temperature_2m", [np.nan] * len(t)),
        "soil": h.get("soil_moisture_28_to_100cm", [np.nan] * len(t)),
    }).set_index("time")
    return pd.DataFrame({
        "precip_day": df["precip"].resample("1D").sum(),
        "temp_day": df["temp"].resample("1D").max(),
        "soil_day": df["soil"].resample("1D").mean(),
    })


# A fire-prone training box (WA + OR, incl. the dry interior where fires occur).
# Features are met-based so the trained model generalizes back to any region.
FIRE_TRAIN_BBOX = (-124.8, 42.0, -116.5, 49.0)


def build_fire_dataset(
    config: Config,
    years: tuple[int, int] = (2020, 2023),
    season: tuple[int, int] = (4, 10),     # Apr..Oct fire season
    horizon_days: int = 7,
    sample_every: int = 7,
    radius_km: float = 15.0,
    point_step: float = 1.0,
    region_bbox: tuple[float, float, float, float] | None = FIRE_TRAIN_BBOX,
    verbose: bool = True,
) -> pd.DataFrame:
    if not firms_key(config):
        raise RuntimeError(
            "No FIRMS_MAP_KEY found. Get a free key at "
            "https://firms.modaps.eosdis.nasa.gov/api/map_key/ and set it: "
            "$env:FIRMS_MAP_KEY='yourkey'")
    if region_bbox is not None:
        config = config.with_region(region_bbox, name="Fire training region")

    start, end = f"{years[0]}-01-01", f"{years[1]}-12-31"
    fires = fetch_fire_detections(config, start, end, season=season, verbose=verbose)
    flat = fires["lat"].to_numpy() if not fires.empty else np.array([])
    flon = fires["lon"].to_numpy() if not fires.empty else np.array([])
    fdate = fires["date"].to_numpy() if not fires.empty else np.array([])

    pts = _sample_points(config, step=point_step)
    if verbose:
        print(f"Sampling {len(pts)} points x fire seasons {years[0]}-{years[1]}…")

    import time
    examples = []
    for i, (lat, lon) in enumerate(pts):
        wx = _weather_daily(lat, lon, start, end, config)
        time.sleep(2.0)  # gentle throttle for the Open-Meteo archive rate limit
        if verbose and i % 10 == 0:
            print(f"  ...weather {i+1}/{len(pts)} points")
        if wx.empty:
            continue
        coslat = np.cos(np.radians(lat))
        # Precompute detections near this point (km).
        if flat.size:
            dy = (flat - lat) * 111.0
            dx = (flon - lon) * 111.0 * coslat
            near = (dx * dx + dy * dy) <= radius_km * radius_km
            near_dates = pd.to_datetime(fdate[near]) if near.any() else pd.DatetimeIndex([])
        else:
            near_dates = pd.DatetimeIndex([])

        idx = wx.index
        n = len(idx)
        for t in range(0, n - horizon_days, sample_every):
            d = idx[t]
            if not (season[0] <= d.month <= season[1]):
                continue
            win = slice(t, t + horizon_days)
            precip_total = float(wx["precip_day"].iloc[win].sum())
            soil_peak = float(wx["soil_day"].iloc[win].max())
            temp_max = float(wx["temp_day"].iloc[win].max())
            if np.isnan(precip_total) or np.isnan(temp_max):
                continue
            w_start, w_end = idx[t], idx[min(t + horizon_days, n - 1)]
            label = int(((near_dates >= w_start) & (near_dates <= w_end)).any())
            examples.append({
                "lat": lat, "lon": lon, "date": d.date().isoformat(),
                "precip_total_mm": precip_total,
                "soil_moist_peak": soil_peak,
                "temp_max": temp_max,
                "fire": label,
            })

    data = pd.DataFrame(examples)
    if verbose and not data.empty:
        print(f"Fire dataset: {len(data)} examples, {data['fire'].mean():.1%} positive, "
              f"{data[['lat','lon']].drop_duplicates().shape[0]} points.")
    return data
