"""Address-level risk — the ParcelRisk integration layer.

Geocode a US address (free U.S. Census geocoder, no key) to a point, then query
the cascade hazard engine at that location and return per-hazard probabilities,
the compound risk, and the cascade story. This is the bridge that lets ParcelRisk
(or any address-based product) sit on top of the Cascadia hazard engine.
"""
from __future__ import annotations

from dataclasses import dataclass, field

import pandas as pd

from .config import Config
from .pipeline import run_pipeline
from .sources.base import fetch_json

CENSUS_GEOCODER = ("https://geocoding.geo.census.gov/geocoder/locations/"
                   "onelineaddress")


@dataclass
class GeocodeResult:
    matched: bool
    lat: float | None = None
    lon: float | None = None
    matched_address: str | None = None
    state: str | None = None


def geocode(address: str, config: Config) -> GeocodeResult:
    """Geocode a US street address via the Census geocoder (no API key)."""
    params = {"address": address, "benchmark": "Public_AR_Current", "format": "json"}
    data = fetch_json(CENSUS_GEOCODER, params=params, cache_dir=config.cache_dir,
                      cache_ttl_s=30 * 86400)
    matches = (data.get("result", {}) or {}).get("addressMatches", [])
    if not matches:
        return GeocodeResult(matched=False)
    m = matches[0]
    coords = m.get("coordinates", {})
    comp = m.get("addressComponents", {})
    return GeocodeResult(
        matched=True,
        lat=float(coords["y"]), lon=float(coords["x"]),
        matched_address=m.get("matchedAddress"),
        state=comp.get("state"),
    )


@dataclass
class ParcelRisk:
    lat: float
    lon: float
    address: str | None
    compound_risk: float
    expected_hazards: float
    dominant_chain: str
    co_occurring: str
    hazards: dict[str, float] = field(default_factory=dict)

    def to_dict(self) -> dict:
        return {
            "address": self.address, "lat": self.lat, "lon": self.lon,
            "compound_risk": round(self.compound_risk, 4),
            "expected_hazards": round(self.expected_hazards, 4),
            "dominant_chain": self.dominant_chain,
            "co_occurring": self.co_occurring,
            "hazards": {k: round(v, 4) for k, v in self.hazards.items()},
        }


def assess_point(lat: float, lon: float, config: Config | None = None,
                 buffer_deg: float = 0.25, state: str | None = None,
                 address: str | None = None) -> ParcelRisk:
    """Run the cascade engine on a small box around a point and read the cell."""
    config = config or Config.load()
    bbox = (lon - buffer_deg, lat - buffer_deg, lon + buffer_deg, lat + buffer_deg)
    cfg = config.with_region(bbox, name=f"parcel @ {lat:.4f},{lon:.4f}",
                             state=state or config.region.state)
    res = run_pipeline(cfg, verbose=False)
    risk = res.risk
    d2 = (risk["lat"] - lat) ** 2 + (risk["lon"] - lon) ** 2
    row = risk.loc[d2.idxmin()]
    hazards = {h: float(row[f"p_{h}"]) for h in
               ["earthquake", "landslide", "flood", "wildfire", "heat"]
               if f"p_{h}" in row}
    return ParcelRisk(
        lat=lat, lon=lon, address=address,
        compound_risk=float(row["compound_risk"]),
        expected_hazards=float(row["expected_hazards"]),
        dominant_chain=str(row.get("dominant_chain", "")),
        co_occurring=str(row.get("co_occurring", "")),
        hazards=hazards,
    )


def assess_address(address: str, config: Config | None = None) -> dict:
    """Full address -> risk: geocode then assess. Returns a JSON-ready dict."""
    config = config or Config.load()
    g = geocode(address, config)
    if not g.matched:
        return {"address": address, "matched": False,
                "error": "address could not be geocoded"}
    pr = assess_point(g.lat, g.lon, config=config, state=g.state,
                      address=g.matched_address)
    out = pr.to_dict()
    out["matched"] = True
    return out
