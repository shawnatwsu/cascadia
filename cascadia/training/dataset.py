"""Assemble a labelled flood-prediction dataset from open data.

For each USGS streamflow gage in the region we pull, for a multi-year period:
  * NWIS daily values  — discharge (00060) + gage height (00065)
  * Open-Meteo archive — hourly precipitation + deep soil moisture (ERA5)

We then slide an `as_of` date across the record and, at each step, build the
SAME three features the live engine computes per cell, plus a binary label:

  features:  precip_total_mm  (sum of precip over the forecast window),
             soil_moist_peak  (max deep soil moisture over the window),
             flow_anomaly     (antecedent streamflow anomaly in [0,1])
  label:     1 if gage height exceeds its 95th-percentile stage anytime in the
             window (a flood), else 0.

Using forward-window precip as a feature mirrors inference, where the engine
consumes a precip *forecast* over the same horizon — so there is no train/serve
feature mismatch.
"""
from __future__ import annotations

import io
from datetime import date

import numpy as np
import pandas as pd
import requests

from ..config import Config
from ..sources.base import fetch_json

SITE_URL = "https://waterservices.usgs.gov/nwis/site/"
DV_URL = "https://waterservices.usgs.gov/nwis/dv/"
ARCHIVE_URL = "https://archive-api.open-meteo.com/v1/archive"

FEATURES = ["precip_total_mm", "soil_moist_peak", "flow_anomaly"]


def discover_gages(config: Config, max_sites: int = 20) -> pd.DataFrame:
    """List active gages in the region that report daily gage height (00065)."""
    r = config.region
    params = {
        "format": "rdb",
        "bBox": f"{r.min_lon:.6f},{r.min_lat:.6f},{r.max_lon:.6f},{r.max_lat:.6f}",
        "parameterCd": "00065",
        "siteType": "ST",          # stream
        "hasDataTypeCd": "dv",     # has daily values
        "siteStatus": "active",
    }
    resp = requests.get(SITE_URL, params=params, timeout=60,
                        headers={"User-Agent": "Cascadia/0.1"})
    resp.raise_for_status()
    rows = [ln for ln in resp.text.splitlines() if ln and not ln.startswith("#")]
    if len(rows) < 3:
        return pd.DataFrame(columns=["site_no", "lat", "lon", "name"])
    header = rows[0].split("\t")
    data = [ln.split("\t") for ln in rows[2:]]  # row[1] is the dtype spec line
    df = pd.DataFrame(data, columns=header)
    out = pd.DataFrame({
        "site_no": df["site_no"],
        "lat": pd.to_numeric(df["dec_lat_va"], errors="coerce"),
        "lon": pd.to_numeric(df["dec_long_va"], errors="coerce"),
        "name": df["station_nm"],
    }).dropna(subset=["lat", "lon"]).reset_index(drop=True)
    return out.head(max_sites)


def _fetch_dv(site_no: str, start: str, end: str, config: Config) -> pd.DataFrame:
    """Daily discharge + gage height for one site, indexed by date."""
    params = {
        "format": "json", "sites": site_no,
        "startDT": start, "endDT": end,
        "parameterCd": "00060,00065", "statCd": "00003",
    }
    data = fetch_json(DV_URL, params=params, cache_dir=config.cache_dir,
                      cache_ttl_s=30 * 86400)
    frames = {}
    for ts in data.get("value", {}).get("timeSeries", []):
        code = ts["variable"]["variableCode"][0]["value"]
        recs = ts["values"][0]["value"]
        if not recs:
            continue
        s = pd.Series(
            {pd.to_datetime(v["dateTime"]).date(): float(v["value"])
             for v in recs if v["value"] not in (None, "")},
            name={"00060": "discharge", "00065": "gage_height"}.get(code, code),
        )
        frames[s.name] = s
    if not frames:
        return pd.DataFrame()
    df = pd.DataFrame(frames)
    df.index = pd.to_datetime(df.index)
    return df.sort_index()


def _fetch_weather_daily(lat: float, lon: float, start: str, end: str,
                         config: Config) -> pd.DataFrame:
    """Daily precip total + mean deep soil moisture from the ERA5 archive."""
    params = {
        "latitude": round(lat, 3), "longitude": round(lon, 3),
        "start_date": start, "end_date": end,
        # ERA5-Land deep-soil layer (the archive equivalent of the forecast
        # model's 27_to_81cm layer used by the live engine).
        "hourly": "precipitation,soil_moisture_28_to_100cm",
        "timezone": "UTC",
    }
    data = fetch_json(ARCHIVE_URL, params=params, cache_dir=config.cache_dir,
                      cache_ttl_s=30 * 86400)
    h = data.get("hourly", {})
    times = h.get("time", [])
    if not times:
        return pd.DataFrame()
    df = pd.DataFrame({
        "time": pd.to_datetime(times),
        "precip": h.get("precipitation", [np.nan] * len(times)),
        "soil": h.get("soil_moisture_28_to_100cm", [np.nan] * len(times)),
    }).set_index("time")
    daily = pd.DataFrame({
        "precip_day": df["precip"].resample("1D").sum(),
        "soil_day": df["soil"].resample("1D").mean(),
    })
    return daily


def _flow_anomaly(discharge: pd.Series, t_idx: int) -> float:
    """Antecedent streamflow anomaly in [0,1] from the trailing ~30 days.

    Mirrors the spirit of features.indicators._streamflow_anomaly: percentile of
    the latest value in its trailing window + normalized recent rise.
    """
    lo = max(0, t_idx - 30)
    window = discharge.iloc[lo:t_idx + 1].dropna().to_numpy()
    if len(window) < 8 or np.allclose(window, window[0]):
        return 0.0
    latest = window[-1]
    pct = float((window < latest).mean())
    scale = np.median(np.abs(window)) + 1e-6
    rise = (window[-1] - window[max(0, len(window) - 7)]) / scale
    return float(np.clip(0.6 * pct + 0.4 * np.tanh(max(rise, 0.0)), 0.0, 1.0))


def build_dataset(
    config: Config,
    start: str = "2019-01-01",
    end: str = "2023-12-31",
    max_sites: int = 20,
    horizon_days: int = 7,
    sample_every: int = 3,
    flood_pctl: float = 0.95,
    verbose: bool = True,
) -> pd.DataFrame:
    """Build the pooled (gage x date) feature/label table."""
    gages = discover_gages(config, max_sites=max_sites)
    if verbose:
        print(f"Discovered {len(gages)} gages with daily gage-height data.")

    examples = []
    for _, g in gages.iterrows():
        site = g["site_no"]
        try:
            dv = _fetch_dv(site, start, end, config)
            wx = _fetch_weather_daily(g["lat"], g["lon"], start, end, config)
        except Exception as exc:
            if verbose:
                print(f"  skip {site}: {exc}")
            continue
        if dv.empty or "gage_height" not in dv or wx.empty:
            continue

        df = dv.join(wx, how="inner").sort_index()
        if len(df) < 60:
            continue
        gh = df["gage_height"]
        thr = float(gh.quantile(flood_pctl))
        disc = df.get("discharge", gh)  # fall back to stage if no discharge

        n = len(df)
        for t in range(30, n - horizon_days, sample_every):
            win = slice(t + 1, t + 1 + horizon_days)
            precip_total = float(df["precip_day"].iloc[win].sum())
            soil_peak = float(df["soil_day"].iloc[win].max())
            flow_anom = _flow_anomaly(disc, t)
            label = int(gh.iloc[win].max() >= thr)
            if np.isnan(precip_total) or np.isnan(soil_peak):
                continue
            examples.append({
                "site_no": site,
                "date": df.index[t].date().isoformat(),
                "precip_total_mm": precip_total,
                "soil_moist_peak": soil_peak,
                "flow_anomaly": flow_anom,
                "flood": label,
            })
        if verbose:
            print(f"  {site}: {len(df)} days, flood-stage thr={thr:.2f} ft")

    data = pd.DataFrame(examples)
    if verbose and not data.empty:
        print(f"Dataset: {len(data)} examples, "
              f"{data['flood'].mean():.1%} positive, "
              f"{data['site_no'].nunique()} gages.")
    return data
