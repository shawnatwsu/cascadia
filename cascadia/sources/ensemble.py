"""Forecast-ensemble uncertainty for the weather-driven hazards.

A single deterministic forecast hides how uncertain the next week is. The
Open-Meteo GFS *ensemble* (31 members) samples that uncertainty directly: each
member is a plausible weather realisation. Running the hazard predictors across
the members yields an honest spread (e.g. flood probability 0.10 [0.04-0.22])
instead of a false-precision point estimate. No API key.

Covers the meteorologically-driven hazards (flood, landslide trigger, heat).
Earthquake/wildfire/smoke have other (non-weather-ensemble) uncertainty sources.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
import requests

ENSEMBLE_URL = "https://ensemble-api.open-meteo.com/v1/ensemble"


def met_ensemble(lat: float, lon: float, horizon_days: int = 7) -> pd.DataFrame:
    """Per-member summaries of the next `horizon_days`: precip total, peak temp,
    min RH. One row per ensemble member."""
    params = {"latitude": round(lat, 4), "longitude": round(lon, 4),
              "hourly": "precipitation,temperature_2m,relative_humidity_2m",
              "forecast_days": horizon_days, "models": "gfs_seamless",
              "timezone": "UTC"}
    h = requests.get(ENSEMBLE_URL, params=params, timeout=40).json().get("hourly", {})
    rows = []
    pre = [k for k in h if k.startswith("precipitation")]
    for pk in pre:
        suffix = pk.replace("precipitation", "")
        precip = np.array(h.get(pk, []), float)
        temp = np.array(h.get("temperature_2m" + suffix, []), float)
        rh = np.array(h.get("relative_humidity_2m" + suffix, []), float)
        if precip.size == 0:
            continue
        rows.append({"precip_total_mm": float(np.nansum(precip)),
                     "temp_max": float(np.nanmax(temp)) if temp.size else np.nan,
                     "rh_min": float(np.nanmin(rh)) if rh.size else np.nan})
    return pd.DataFrame(rows)


def hazard_uncertainty(cell_features: pd.Series, lat: float, lon: float) -> dict:
    """Return {hazard: (median, p10, p90)} for the weather-driven hazards, from
    the forecast ensemble. Falls back to an empty dict on any failure."""
    from ..models.predictors import _p_heat, _p_landslide, _p_flood
    from ..models.trained import load_trained
    from .gridmet import heat_index_c
    try:
        ens = met_ensemble(lat, lon)
    except Exception:
        return {}
    if ens.empty:
        return {}

    flood_rows, ls_rows, heat_rows = [], [], []
    for _, m in ens.iterrows():
        r = cell_features.copy()
        r["precip_total_mm"] = m["precip_total_mm"]
        flood_rows.append(r.copy()); ls_rows.append(r.copy())
        rh = cell_features.get("rh_min", m["rh_min"])
        rheat = cell_features.copy()
        rheat["heat_index_c"] = float(heat_index_c(m["temp_max"], rh))
        heat_rows.append(rheat)

    fmodel = load_trained("flood")
    Xf = pd.DataFrame(flood_rows)
    flood = fmodel.predict(Xf) if fmodel is not None else _p_flood(Xf)
    landslide = _p_landslide(pd.DataFrame(ls_rows))
    heat = _p_heat(pd.DataFrame(heat_rows))

    out = {}
    for name, vals in (("flood", flood), ("landslide", landslide), ("heat", heat)):
        v = np.asarray(vals, float)
        out[name] = (float(np.median(v)), float(np.percentile(v, 10)),
                     float(np.percentile(v, 90)))
    return out
