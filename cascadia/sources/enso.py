"""ENSO state — the dominant driver of US seasonal hazard patterns.

El Nino / La Nina shifts the odds of a wet/dry, warm/cool season across the
country (e.g. La Nina -> drier, warmer, fire-prone Southwest & Southeast; wetter
Pacific Northwest). We ingest NOAA CPC's Oceanic Nino Index (ONI) — the official
ENSO index, 3-month running SST anomaly in the Nino-3.4 region — back to 1950,
for both the current state and as the feature history an ML model learns from.

No API key. Source: https://www.cpc.ncep.noaa.gov/data/indices/oni.ascii.txt
"""
from __future__ import annotations

import io
from dataclasses import dataclass

import pandas as pd

from ..config import Config
from .base import fetch_json  # noqa (kept for parity; we use requests below)

ONI_URL = "https://www.cpc.ncep.noaa.gov/data/indices/oni.ascii.txt"

# 3-month "season" code -> its centre month number.
_SEAS_MONTH = {"DJF": 1, "JFM": 2, "FMA": 3, "MAM": 4, "AMJ": 5, "MJJ": 6,
               "JJA": 7, "JAS": 8, "ASO": 9, "SON": 10, "OND": 11, "NDJ": 12}


@dataclass
class ENSOState:
    oni: float            # current ONI value
    phase: str            # "El Nino" | "La Nina" | "Neutral"
    season: str           # e.g. "MAM"
    year: int
    trend: float          # change over the last ~3 seasons

    def label(self) -> str:
        strength = (abs(self.oni) >= 1.5 and "very strong " or
                    abs(self.oni) >= 1.0 and "strong " or
                    abs(self.oni) >= 0.5 and "moderate " or "weak ")
        return f"{strength}{self.phase} (ONI {self.oni:+.2f})"


def _classify(oni: float) -> str:
    if oni >= 0.5:
        return "El Nino"
    if oni <= -0.5:
        return "La Nina"
    return "Neutral"


def fetch_oni(config: Config | None = None) -> pd.DataFrame:
    """ONI time series with a proper datetime index and phase column."""
    import requests
    cache = None
    if config is not None:
        cache = config.cache_dir / "oni.ascii.txt"
    if cache is not None and cache.exists():
        txt = cache.read_text(encoding="utf-8")
    else:
        txt = requests.get(ONI_URL, timeout=60).text
        if cache is not None:
            cache.write_text(txt, encoding="utf-8")
    df = pd.read_csv(io.StringIO(txt), sep=r"\s+")
    df["month"] = df["SEAS"].map(_SEAS_MONTH)
    df["date"] = pd.to_datetime(dict(year=df["YR"], month=df["month"], day=1))
    df = df.rename(columns={"ANOM": "oni"}).sort_values("date").reset_index(drop=True)
    df["phase"] = df["oni"].map(_classify)
    return df[["date", "YR", "SEAS", "oni", "phase"]]


def current_state(config: Config | None = None) -> ENSOState:
    df = fetch_oni(config)
    last = df.iloc[-1]
    trend = float(last["oni"] - df.iloc[-4]["oni"]) if len(df) >= 4 else 0.0
    return ENSOState(oni=float(last["oni"]), phase=str(last["phase"]),
                     season=str(last["SEAS"]), year=int(last["YR"]), trend=trend)
