"""National Weather Service active alerts (no API key; needs a User-Agent).

API docs: https://www.weather.gov/documentation/services-web-api
We pull active alerts for the configured state and tag each with a coarse
hazard family so the cascade graph can react to official warnings.
"""
from __future__ import annotations

import pandas as pd

from ..config import Config
from .base import fetch_json

ALERTS_URL = "https://api.weather.gov/alerts/active"

# Map NWS event strings -> coarse hazard families used by the cascade graph.
EVENT_FAMILY = {
    "flood": "flood",
    "flash flood": "flood",
    "coastal flood": "flood",
    "high wind": "wind",
    "wind": "wind",
    "winter storm": "winter",
    "heat": "heat",
    "excessive heat": "heat",
    "red flag": "fire_weather",
    "fire weather": "fire_weather",
    "storm": "storm",
}


def _family(event: str) -> str:
    e = (event or "").lower()
    for key, fam in EVENT_FAMILY.items():
        if key in e:
            return fam
    return "other"


class NWSAlerts:
    kind = "nws_alert"

    def __init__(self, config: Config):
        self.config = config
        self.opts = config.sources.get("nws_alerts", {})

    COLUMNS = ["kind", "family", "event", "severity", "certainty", "urgency",
               "headline", "onset", "expires", "areaDesc"]

    def fetch(self) -> pd.DataFrame:
        # The NWS API only serves *active* alerts; there is no historical query.
        # In historical mode we return an empty frame (predictors degrade to the
        # condition-only baseline, which is the honest behaviour for a backtest).
        if self.config.is_historical:
            return pd.DataFrame(columns=self.COLUMNS)

        params = {"area": self.config.region.state, "status": "actual"}
        headers = {"User-Agent": self.opts.get("user_agent", "Cascadia-hazard-engine/0.1")}
        data = fetch_json(
            ALERTS_URL, params=params, headers=headers, cache_dir=self.config.cache_dir
        )

        rows = []
        for feat in data.get("features", []):
            props = feat.get("properties", {})
            event = props.get("event", "")
            rows.append(
                {
                    "kind": self.kind,
                    "family": _family(event),
                    "event": event,
                    "severity": props.get("severity"),
                    "certainty": props.get("certainty"),
                    "urgency": props.get("urgency"),
                    "headline": props.get("headline"),
                    "onset": props.get("onset"),
                    "expires": props.get("expires"),
                    "areaDesc": props.get("areaDesc"),
                }
            )
        return pd.DataFrame(rows)
