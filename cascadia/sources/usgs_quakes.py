"""USGS earthquake feed (no API key).

Feed docs: https://earthquake.usgs.gov/earthquakes/feed/v1.0/geojson.php
Returns recent quakes; we clip to the region bbox and a minimum magnitude.
"""
from __future__ import annotations

from datetime import datetime, timedelta, timezone

import pandas as pd

from ..config import Config
from .base import fetch_json

FEED_URL = "https://earthquake.usgs.gov/earthquakes/feed/v1.0/summary/{feed}.geojson"
# FDSN catalog query — supports arbitrary time windows + bbox (no API key).
QUERY_URL = "https://earthquake.usgs.gov/fdsnws/event/1/query"


class USGSQuakes:
    kind = "earthquake"

    def __init__(self, config: Config):
        self.config = config
        self.opts = config.sources.get("usgs_quakes", {})

    def fetch(self) -> pd.DataFrame:
        min_mag = float(self.opts.get("min_magnitude", 2.5))
        if self.config.is_historical:
            # Query the catalog for the antecedent week through the horizon end,
            # clipped to the region bbox.
            start, end = self.config.window()
            r = self.config.region
            params = {
                "format": "geojson",
                "starttime": (start - timedelta(days=7)).date().isoformat(),
                "endtime": end.date().isoformat(),
                "minmagnitude": min_mag,
                "minlatitude": r.min_lat, "maxlatitude": r.max_lat,
                "minlongitude": r.min_lon, "maxlongitude": r.max_lon,
            }
            data = fetch_json(QUERY_URL, params=params, cache_dir=self.config.cache_dir)
        else:
            feed = self.opts.get("feed", "all_week")
            url = FEED_URL.format(feed=feed)
            data = fetch_json(url, cache_dir=self.config.cache_dir)

        rows = []
        for feat in data.get("features", []):
            geom = feat.get("geometry") or {}
            coords = geom.get("coordinates") or [None, None, None]
            lon, lat, depth = (coords + [None, None, None])[:3]
            props = feat.get("properties", {})
            mag = props.get("mag")
            if lon is None or lat is None or mag is None:
                continue
            if mag < min_mag:
                continue
            if not self.config.region.contains(lon, lat):
                continue
            t_ms = props.get("time")
            t = (
                datetime.fromtimestamp(t_ms / 1000, tz=timezone.utc)
                if t_ms
                else None
            )
            rows.append(
                {
                    "kind": self.kind,
                    "lat": lat,
                    "lon": lon,
                    "time": t,
                    "value": float(mag),         # magnitude
                    "depth_km": depth,
                    "place": props.get("place"),
                }
            )
        return pd.DataFrame(rows)
