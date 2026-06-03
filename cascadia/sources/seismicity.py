"""Long-term earthquake catalog for the smoothed-seismicity hazard prior.

Earthquakes are NOT short-term predictable. The scientifically defensible prior
is a *smoothed historical seismicity* model: decades of catalog epicenters,
spatially smoothed, give the long-run annual rate of events, which a Poisson
model turns into P(event in the horizon). This mirrors one of the standard
components of the USGS National Seismic Hazard Model.

This source pulls the multi-decade declustered-enough catalog (M >= threshold,
above catalog-completeness) once and caches it for a long time.
"""
from __future__ import annotations

import pandas as pd

from ..config import Config
from .base import fetch_json

QUERY_URL = "https://earthquake.usgs.gov/fdsnws/event/1/query"


class Seismicity:
    kind = "seismicity_catalog"

    def __init__(self, config: Config):
        self.config = config
        self.opts = config.sources.get("seismicity", {})

    def fetch(self) -> pd.DataFrame:
        r = self.config.region
        start = self.opts.get("catalog_start", "1970-01-01")
        min_mag = float(self.opts.get("catalog_min_magnitude", 4.0))
        params = {
            "format": "geojson",
            "starttime": start,
            "minmagnitude": min_mag,
            "minlatitude": r.min_lat, "maxlatitude": r.max_lat,
            "minlongitude": r.min_lon, "maxlongitude": r.max_lon,
            "orderby": "time",
        }
        # Cache for 30 days — the long-term catalog barely moves.
        data = fetch_json(QUERY_URL, params=params, cache_dir=self.config.cache_dir,
                          cache_ttl_s=30 * 86400, timeout=60)
        rows = []
        for feat in data.get("features", []):
            c = (feat.get("geometry") or {}).get("coordinates") or [None, None]
            lon, lat = (c + [None, None])[:2]
            mag = (feat.get("properties") or {}).get("mag")
            if lon is None or lat is None or mag is None:
                continue
            rows.append({"lat": lat, "lon": lon, "mag": float(mag)})
        df = pd.DataFrame(rows)
        df.attrs["catalog_start"] = start
        df.attrs["min_mag"] = min_mag
        return df
