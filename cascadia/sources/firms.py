"""NASA FIRMS active-fire detections (VIIRS/MODIS).

Needs a free MAP_KEY (https://firms.modaps.eosdis.nasa.gov/api/) supplied via
the FIRMS_MAP_KEY environment variable. Without it this adapter returns an
empty frame, so the pipeline still runs end-to-end (wildfire then relies on the
dryness + fire-weather proxy alone). With a key, observed thermal detections
become a direct evidence boost for the wildfire node.

The area API returns CSV: .../api/area/csv/{KEY}/{source}/{bbox}/{day_range}[/{date}]
where bbox is west,south,east,north.
"""
from __future__ import annotations

import io

import pandas as pd
import requests

from ..config import Config

AREA_URL = "https://firms.modaps.eosdis.nasa.gov/api/area/csv"


class FIRMS:
    kind = "active_fire"

    COLUMNS = ["kind", "lat", "lon", "time", "value"]

    def __init__(self, config: Config):
        self.config = config
        self.opts = config.sources.get("firms", {})

    def _key(self) -> str | None:
        return self.config.env(self.opts.get("map_key_env", "FIRMS_MAP_KEY"))

    def fetch(self) -> pd.DataFrame:
        key = self._key()
        if not key:
            return pd.DataFrame(columns=self.COLUMNS)

        r = self.config.region
        bbox = f"{r.min_lon},{r.min_lat},{r.max_lon},{r.max_lat}"
        source = self.opts.get("source", "VIIRS_SNPP_NRT")
        day_range = int(self.opts.get("day_range", 3))
        url = f"{AREA_URL}/{key}/{source}/{bbox}/{day_range}"
        if self.config.is_historical:
            start, _ = self.config.window()
            url += f"/{start.date().isoformat()}"

        try:
            resp = requests.get(url, timeout=60)
            resp.raise_for_status()
            df = pd.read_csv(io.StringIO(resp.text))
        except Exception:
            return pd.DataFrame(columns=self.COLUMNS)

        if df.empty or "latitude" not in df:
            return pd.DataFrame(columns=self.COLUMNS)

        out = pd.DataFrame(
            {
                "kind": self.kind,
                "lat": df["latitude"].astype(float),
                "lon": df["longitude"].astype(float),
                "time": pd.to_datetime(
                    df.get("acq_date", pd.Series(["" ] * len(df))), errors="coerce", utc=True
                ),
                # Fire radiative power if present, else detection confidence proxy.
                "value": df["frp"].astype(float) if "frp" in df else 1.0,
            }
        )
        return out
