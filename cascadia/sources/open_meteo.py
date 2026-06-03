"""Open-Meteo forecast (no API key).

Docs: https://open-meteo.com/en/docs
We sample the forecast on a coarse grid across the region and return hourly
precipitation + soil-moisture, the key leading indicators for the
precip -> saturation -> landslide/flood cascade.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from ..config import Config
from .base import fetch_json

API_URL = "https://api.open-meteo.com/v1/forecast"
ARCHIVE_URL = "https://archive-api.open-meteo.com/v1/archive"

# The forecast model and the ERA5 archive expose soil moisture on DIFFERENT
# layer boundaries. Map forecast layer names to their nearest archive (ERA5-Land)
# equivalents so historical runs get real soil-moisture data, not nulls.
FORECAST_TO_ARCHIVE = {
    "soil_moisture_0_to_1cm": "soil_moisture_0_to_7cm",
    "soil_moisture_1_to_3cm": "soil_moisture_0_to_7cm",
    "soil_moisture_3_to_9cm": "soil_moisture_7_to_28cm",
    "soil_moisture_9_to_27cm": "soil_moisture_7_to_28cm",
    "soil_moisture_27_to_81cm": "soil_moisture_28_to_100cm",
}


class OpenMeteo:
    kind = "weather_forecast"

    def __init__(self, config: Config):
        self.config = config
        self.opts = config.sources.get("open_meteo", {})

    def _sample_points(self, step: float | None = None) -> list[tuple[float, float]]:
        step = step or float(self.opts.get("sample_step", 0.5))
        r = self.config.region
        lats = np.arange(r.min_lat, r.max_lat + 1e-9, step)
        lons = np.arange(r.min_lon, r.max_lon + 1e-9, step)
        return [(round(float(la), 3), round(float(lo), 3)) for la in lats for lo in lons]

    def fetch(self) -> pd.DataFrame:
        hourly = self.opts.get(
            "hourly",
            ["precipitation", "soil_moisture_0_to_1cm", "soil_moisture_27_to_81cm"],
        )
        fdays = int(self.opts.get("forecast_days", 7))
        pts = self._sample_points()

        historical = self.config.is_historical
        if historical:
            start, end = self.config.window()
            url = ARCHIVE_URL
            # Translate forecast soil layers to their archive equivalents.
            request_hourly = [FORECAST_TO_ARCHIVE.get(v, v) for v in hourly]
        else:
            url = API_URL
            request_hourly = hourly

        frames = []
        for lat, lon in pts:
            params = {
                "latitude": lat,
                "longitude": lon,
                "hourly": ",".join(request_hourly),
                "timezone": "UTC",
            }
            if historical:
                # ERA5 reanalysis actuals over the target window.
                params["start_date"] = start.date().isoformat()
                params["end_date"] = end.date().isoformat()
            else:
                params["forecast_days"] = fdays
            data = fetch_json(url, params=params, cache_dir=self.config.cache_dir)
            h = data.get("hourly", {})
            times = h.get("time", [])
            if not times:
                continue
            df = pd.DataFrame({"time": pd.to_datetime(times, utc=True)})
            # Read the keys actually returned (archive names in historical mode);
            # downstream only keys off the "soil_moisture"/"precipitation" prefix.
            for var in request_hourly:
                df[var] = h.get(var, [np.nan] * len(times))
            df["lat"] = lat
            df["lon"] = lon
            df["kind"] = self.kind
            frames.append(df)

        return pd.concat(frames, ignore_index=True) if frames else pd.DataFrame()
