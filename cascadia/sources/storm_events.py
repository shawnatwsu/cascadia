"""NWS Storm Events Database (NOAA NCEI) — documented hazard events.

The authoritative US record of who/where/when hazards actually happened, with
begin coordinates and dates. We use it to assemble a SYSTEMATIC sample of real
events (and, by shifting dates, matched non-events) so the parcel hindcast can
report a defensible hit rate / false-alarm rate instead of a few anecdotes.
No API key (static yearly gzip CSVs).
"""
from __future__ import annotations

import io
import re
from pathlib import Path

import pandas as pd
import requests

STORM_DIR = "https://www.ncei.noaa.gov/pub/data/swdi/stormevents/csvfiles/"


def _year_file(year: int) -> str:
    idx = requests.get(STORM_DIR, timeout=60).text
    pat = rf"StormEvents_details-ftp_v1\.0_d{year}_c\d+\.csv\.gz"
    matches = sorted(re.findall(pat, idx))
    if not matches:
        raise RuntimeError(f"no Storm Events file for {year}")
    return matches[-1]


def events(year: int, event_types: tuple, cache_dir: Path) -> pd.DataFrame:
    """Events of the given types in `year` with begin coordinates + date."""
    tag = "_".join(t.replace(" ", "") for t in event_types)
    cache = Path(cache_dir) / f"storm_{year}_{tag}.pkl"
    if cache.exists():
        return pd.read_pickle(cache)
    fname = _year_file(year)
    blob = requests.get(STORM_DIR + fname, timeout=240).content
    df = pd.read_csv(io.BytesIO(blob), compression="gzip", low_memory=False)
    df = df[df["EVENT_TYPE"].isin(event_types)].dropna(subset=["BEGIN_LAT", "BEGIN_LON"])
    out = pd.DataFrame({
        "lat": df["BEGIN_LAT"].astype(float),
        "lon": df["BEGIN_LON"].astype(float),
        "date": pd.to_datetime(df["BEGIN_YEARMONTH"].astype(str)
                               + df["BEGIN_DAY"].astype(int).astype(str).str.zfill(2),
                               format="%Y%m%d", errors="coerce"),
        "type": df["EVENT_TYPE"], "state": df["STATE"],
    }).dropna(subset=["date"]).reset_index(drop=True)
    # keep CONUS-ish
    out = out[out["lat"].between(24, 50) & out["lon"].between(-125, -66)]
    try:
        out.to_pickle(cache)
    except Exception:
        pass
    return out.reset_index(drop=True)


def sample_events(years, event_types, cache_dir: Path, n: int = 100,
                  seed: int = 0) -> pd.DataFrame:
    """A reproducible sample of events pooled across years."""
    frames = []
    for y in years:
        try:
            frames.append(events(y, event_types, cache_dir))
        except Exception:
            pass
    if not frames:
        return pd.DataFrame()
    pool = pd.concat(frames, ignore_index=True)
    if n and len(pool) > n:
        pool = pool.sample(n=n, random_state=seed).reset_index(drop=True)
    return pool
