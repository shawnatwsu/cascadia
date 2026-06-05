"""Observed surface PM2.5 (EPA AQS daily files) — ground truth for smoke.

To VALIDATE the fire->smoke cascade we need observed air quality. EPA's Air
Quality System publishes daily PM2.5 (parameter 88101) for every US monitor as
static yearly files — no API key. We use the 24-hour averages with site
coordinates, so we can compare our smoke prediction to what was actually
measured at the surface.
"""
from __future__ import annotations

import io
import zipfile
from pathlib import Path

import pandas as pd
import requests

AQS_URL = "https://aqs.epa.gov/aqsweb/airdata/daily_88101_{year}.zip"


def aqs_pm25_daily(year: int, cache_dir: Path) -> pd.DataFrame:
    """Daily 24-hour mean PM2.5 per monitor: columns [lat, lon, date, pm25, state]."""
    cache = Path(cache_dir) / f"aqs_pm25_{year}.pkl"
    if cache.exists():
        return pd.read_pickle(cache)

    blob = requests.get(AQS_URL.format(year=year), timeout=240).content
    z = zipfile.ZipFile(io.BytesIO(blob))
    csv = [n for n in z.namelist() if n.endswith(".csv")][0]
    df = pd.read_csv(z.open(csv), usecols=["Latitude", "Longitude", "Date Local",
                                           "Arithmetic Mean", "State Name",
                                           "Sample Duration"])
    df = df[df["Sample Duration"] == "24 HOUR"]
    out = pd.DataFrame({
        "lat": df["Latitude"].astype(float),
        "lon": df["Longitude"].astype(float),
        "date": pd.to_datetime(df["Date Local"], errors="coerce"),
        "pm25": df["Arithmetic Mean"].astype(float),
        "state": df["State Name"],
    }).dropna(subset=["date", "pm25"])
    out = out[out["pm25"] >= 0].reset_index(drop=True)
    try:
        out.to_pickle(cache)
    except Exception:
        pass
    return out


def aqs_window(year: int, start: str, end: str,
              bbox: tuple[float, float, float, float], cache_dir: Path) -> pd.DataFrame:
    """PM2.5 monitor-days within a date window and bounding box."""
    df = aqs_pm25_daily(year, cache_dir)
    minlon, minlat, maxlon, maxlat = bbox
    m = ((df["date"] >= start) & (df["date"] <= end)
         & df["lat"].between(minlat, maxlat) & df["lon"].between(minlon, maxlon))
    return df[m].reset_index(drop=True)
