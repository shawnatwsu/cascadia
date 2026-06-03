"""USGS U.S. Landslide Inventory — historical landslide locations.

Used to build a data-driven landslide *susceptibility* prior (where landslides
have historically occurred), exactly analogous to the smoothed-seismicity prior
for earthquakes. The inventory is a compilation (not a complete temporal
catalog), so we use it for relative spatial susceptibility, not absolute rates.

Public ArcGIS FeatureServer (USGS Landslide Hazards Program). We pull feature
centroids in the region, paginated, and cache for a long time.
"""
from __future__ import annotations

import pandas as pd

from ..config import Config
from .base import fetch_json

SERVICE = ("https://services.arcgis.com/1GgsAFzlko7YxeAI/arcgis/rest/services/"
           "US_Landslide_Inventory/FeatureServer/0/query")


class LandslideInventory:
    kind = "landslide_inventory"

    def __init__(self, config: Config):
        self.config = config
        self.opts = config.sources.get("landslide_inventory", {})

    def fetch(self) -> pd.DataFrame:
        r = self.config.region
        page = 2000
        rows: list[dict] = []
        offset = 0
        while True:
            params = {
                "where": "1=1",
                "geometry": f"{r.min_lon},{r.min_lat},{r.max_lon},{r.max_lat}",
                "geometryType": "esriGeometryEnvelope",
                "inSR": "4326", "outSR": "4326",
                "spatialRel": "esriSpatialRelIntersects",
                "returnGeometry": "false", "returnCentroid": "true",
                "outFields": "Date,Confidence", "resultRecordCount": page,
                "resultOffset": offset, "f": "json",
            }
            data = fetch_json(SERVICE, params=params, cache_dir=self.config.cache_dir,
                              cache_ttl_s=30 * 86400, timeout=90)
            feats = data.get("features", [])
            if not feats:
                break
            for ft in feats:
                c = ft.get("centroid") or {}
                if c.get("x") is None or c.get("y") is None:
                    continue
                rows.append({"lon": c["x"], "lat": c["y"],
                             "date": (ft.get("attributes") or {}).get("Date")})
            if len(feats) < page:
                break
            offset += page
            if offset > 200000:   # safety cap
                break
        return pd.DataFrame(rows)
