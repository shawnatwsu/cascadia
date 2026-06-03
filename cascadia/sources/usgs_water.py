"""USGS NWIS instantaneous values — streamflow & gage height (no API key).

Docs: https://waterservices.usgs.gov/docs/instantaneous-values/
Streamflow anomalies are a leading indicator for riverine flooding and a
useful state variable for the saturation -> flood cascade edge.
"""
from __future__ import annotations

import pandas as pd

from ..config import Config
from .base import fetch_json

IV_URL = "https://waterservices.usgs.gov/nwis/iv/"

PARAM_NAME = {"00060": "discharge_cfs", "00065": "gage_height_ft"}


class USGSWater:
    kind = "streamflow"

    def __init__(self, config: Config):
        self.config = config
        self.opts = config.sources.get("usgs_water", {})

    def fetch(self) -> pd.DataFrame:
        r = self.config.region
        params = {
            "format": "json",
            "bBox": f"{r.min_lon:.6f},{r.min_lat:.6f},{r.max_lon:.6f},{r.max_lat:.6f}",
            "parameterCd": ",".join(self.opts.get("parameters", ["00060", "00065"])),
            "siteStatus": "active",
        }
        if self.config.is_historical:
            # Antecedent week up to the as-of date for the anomaly baseline.
            from datetime import timedelta
            start, _ = self.config.window()
            params["startDT"] = (start - timedelta(days=7)).date().isoformat()
            params["endDT"] = start.date().isoformat()
        else:
            params["period"] = self.opts.get("period", "P7D")
        data = fetch_json(IV_URL, params=params, cache_dir=self.config.cache_dir)

        rows = []
        series = data.get("value", {}).get("timeSeries", [])
        for ts in series:
            src = ts.get("sourceInfo", {})
            loc = src.get("geoLocation", {}).get("geogLocation", {})
            lat, lon = loc.get("latitude"), loc.get("longitude")
            site = src.get("siteName")
            var_code = (
                ts.get("variable", {})
                .get("variableCode", [{}])[0]
                .get("value")
            )
            for block in ts.get("values", []):
                for v in block.get("value", []):
                    try:
                        val = float(v.get("value"))
                    except (TypeError, ValueError):
                        continue
                    rows.append(
                        {
                            "kind": self.kind,
                            "lat": lat,
                            "lon": lon,
                            "site": site,
                            "param": PARAM_NAME.get(var_code, var_code),
                            "time": pd.to_datetime(v.get("dateTime"), utc=True),
                            "value": val,
                        }
                    )
        return pd.DataFrame(rows)
