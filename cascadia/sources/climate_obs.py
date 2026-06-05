"""Observed historical climate (NOAA NCEI nClimDiv statewide), for validation.

Ground truth for the ENSO seasonal outlook: monthly statewide precipitation and
temperature back to 1895, aggregated to the NCA5 regions. With this we can ask
the question a reviewer demands — does ENSO actually predict each region's
seasonal climate, and with what *measured* skill? No API key (static NCEI files).
"""
from __future__ import annotations

import re
from pathlib import Path

import numpy as np
import pandas as pd
import requests

CLIMDIV_DIR = "https://www.ncei.noaa.gov/pub/data/cirs/climdiv/"

# nClimDiv statewide state codes (001-048, alphabetical contiguous US).
STATE_CODES = {
    1: "Alabama", 2: "Arizona", 3: "Arkansas", 4: "California", 5: "Colorado",
    6: "Connecticut", 7: "Delaware", 8: "Florida", 9: "Georgia", 10: "Idaho",
    11: "Illinois", 12: "Indiana", 13: "Iowa", 14: "Kansas", 15: "Kentucky",
    16: "Louisiana", 17: "Maine", 18: "Maryland", 19: "Massachusetts",
    20: "Michigan", 21: "Minnesota", 22: "Mississippi", 23: "Missouri",
    24: "Montana", 25: "Nebraska", 26: "Nevada", 27: "New Hampshire",
    28: "New Jersey", 29: "New Mexico", 30: "New York", 31: "North Carolina",
    32: "North Dakota", 33: "Ohio", 34: "Oklahoma", 35: "Oregon",
    36: "Pennsylvania", 37: "Rhode Island", 38: "South Carolina",
    39: "South Dakota", 40: "Tennessee", 41: "Texas", 42: "Utah", 43: "Vermont",
    44: "Virginia", 45: "Washington", 46: "West Virginia", 47: "Wisconsin",
    48: "Wyoming",
}
ELEMENT_FILE = {"precip": "pcpnst", "temp": "tmpcst"}


def _latest_file(element: str, cache_dir: Path) -> str:
    code = ELEMENT_FILE[element]
    idx = requests.get(CLIMDIV_DIR, timeout=60).text
    return sorted(set(re.findall(rf"climdiv-{code}-v[0-9.]+-\d+", idx)))[-1]


def state_monthly(element: str, cache_dir: Path) -> pd.DataFrame:
    """Long-form statewide monthly series: columns [state, date, value]."""
    fname = _latest_file(element, cache_dir)
    cache = Path(cache_dir) / fname
    txt = (cache.read_text() if cache.exists()
           else requests.get(CLIMDIV_DIR + fname, timeout=120).text)
    if not cache.exists():
        cache.write_text(txt)

    rows = []
    for ln in txt.splitlines():
        if len(ln) < 10:
            continue
        try:
            scode = int(ln[0:3])
            year = int(ln[6:10])
        except ValueError:
            continue
        state = STATE_CODES.get(scode)
        if state is None:
            continue
        vals = ln[10:].split()
        for m, v in enumerate(vals[:12], start=1):
            fv = float(v)
            # climdiv missing flags: -9.99 (precip) / -99.90 (temp). Precip is
            # never negative, so drop any negative precip too.
            if (element == "precip" and fv < 0) or fv <= -99:
                continue
            rows.append((state, pd.Timestamp(year=year, month=m, day=1), fv))
    return pd.DataFrame(rows, columns=["state", "date", "value"])


def regional_monthly(element: str, cache_dir: Path) -> pd.DataFrame:
    """Monthly series per NCA5 region (mean of member states). Index = date."""
    from ..geo import NCA5_REGIONS
    sm = state_monthly(element, cache_dir)
    out = {}
    for region, states in NCA5_REGIONS.items():
        sub = sm[sm["state"].isin(states)]
        if sub.empty:
            continue
        out[region] = sub.groupby("date")["value"].mean()
    df = pd.DataFrame(out)
    df.index = pd.to_datetime(df.index)
    return df.sort_index()


def seasonal_anomaly(monthly: pd.DataFrame, months: tuple[int, ...],
                     baseline=(1991, 2020)) -> pd.DataFrame:
    """Seasonal mean anomaly per region for the given calendar months.

    Returns one value per year (labelled by the season's first year) = the
    season mean minus the per-region baseline-climatology season mean.
    """
    m = monthly[monthly.index.month.isin(months)].copy()
    # Group by season, then label by the season's CENTRE year so it aligns with
    # ENSO's ONI convention (e.g. DJF 1998 = Dec1997-Feb1998 -> labelled 1998).
    first = months[0]
    m["syear"] = np.where(m.index.month >= first, m.index.year, m.index.year - 1)
    # require all months present for a complete season
    counts = m.groupby("syear").size()
    seas = m.groupby("syear").mean(numeric_only=True)
    seas = seas[counts >= len(months)]   # complete seasons only
    base = seas.loc[(seas.index >= baseline[0]) & (seas.index <= baseline[1])].mean()
    anom = seas - base
    center_offset = 1 if months[len(months) // 2] < first else 0
    anom.index = anom.index + center_offset
    anom.index.name = "year"
    return anom
