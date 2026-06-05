"""Address-level risk — the ParcelRisk integration layer.

Geocode a US address (free U.S. Census geocoder, no key) to a point, then query
the cascade hazard engine at that location and return per-hazard probabilities,
the compound risk, and the cascade story. This is the bridge that lets ParcelRisk
(or any address-based product) sit on top of the Cascadia hazard engine.
"""
from __future__ import annotations

from dataclasses import dataclass, field

from pathlib import Path

import numpy as np
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
    # {hazard: (median, p10, p90)} from the forecast ensemble (weather-driven hazards)
    uncertainty: dict = field(default_factory=dict)

    def to_dict(self) -> dict:
        return {
            "address": self.address, "lat": self.lat, "lon": self.lon,
            "compound_risk": round(self.compound_risk, 4),
            "expected_hazards": round(self.expected_hazards, 4),
            "dominant_chain": self.dominant_chain,
            "co_occurring": self.co_occurring,
            "hazards": {k: round(v, 4) for k, v in self.hazards.items()},
            "uncertainty_p10_p90": {k: [round(v[1], 4), round(v[2], 4)]
                                    for k, v in self.uncertainty.items()},
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
    idx = d2.idxmin()
    row = risk.loc[idx]
    hazards = {h: float(row[f"p_{h}"]) for h in
               ["earthquake", "landslide", "flood", "wildfire", "heat", "smoke"]
               if f"p_{h}" in row}

    # Refine landslide with the ACTUAL local slope at the address (a ~250 m DEM
    # stencil), instead of the ~5 km cell average — so a flat lot reads as stable.
    feats = res.features
    frow = feats[feats["cell_id"] == row["cell_id"]].iloc[0].copy()
    try:
        from .sources.elevation import point_slope
        from .models.predictors import _p_landslide
        slope = point_slope(lat, lon)
        if np.isfinite(slope):
            frow["slope_deg"] = slope
            hazards["landslide"] = float(_p_landslide(pd.DataFrame([frow]))[0])
    except Exception:
        pass

    # Forecast-ensemble uncertainty for the weather-driven hazards (31 members).
    # Use the ensemble MEDIAN as the point value so it is consistent with its
    # interval and reflects the 7-day forecast (not the nowcast).
    uncertainty = {}
    try:
        from .sources.ensemble import hazard_uncertainty
        uncertainty = hazard_uncertainty(frow, lat, lon)
        for h, (med, _, _) in uncertainty.items():
            if h in hazards:
                hazards[h] = med
    except Exception:
        pass

    # Recompute compound metrics from the (refined) per-hazard probabilities.
    ps = np.array(list(hazards.values()))
    compound = float(1 - np.prod(1 - ps))
    expected = float(ps.sum())
    return ParcelRisk(
        lat=lat, lon=lon, address=address,
        compound_risk=compound, expected_hazards=expected,
        dominant_chain=str(row.get("dominant_chain", "")),
        co_occurring=str(row.get("co_occurring", "")),
        hazards=hazards, uncertainty=uncertainty,
    )


def parcel_report(address: str, out_path: str | Path = "cascadia_parcel_map.png",
                  config: Config | None = None, verbose: bool = True):
    """Render a one-page parcel hazard report: a locator map of the area (with a
    marker on the address) beside a bar chart of that address's hazard
    probabilities."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import cartopy.crs as ccrs
    import numpy as np
    from .cartomap import _grid, _bin_edges, _fmt, HAZARD_CMAPS, HAZARD_TITLES
    from matplotlib.colors import BoundaryNorm, ListedColormap

    config = config or Config.load()
    g = geocode(address, config)
    if not g.matched:
        if verbose:
            print(f"Could not geocode: {address}")
        return None
    pr = assess_point(g.lat, g.lon, config=config, state=g.state,
                      address=g.matched_address)

    # Re-run on the parcel's neighbourhood to get a risk surface to map.
    buf = 0.4
    bbox = (g.lon - buf, g.lat - buf, g.lon + buf, g.lat + buf)
    cfg = config.with_region(bbox, name="parcel area", state=g.state)
    risk = run_pipeline(cfg, verbose=False).risk

    proj = ccrs.AlbersEqualArea(central_longitude=g.lon, central_latitude=g.lat)
    fig = plt.figure(figsize=(13, 5.6))
    ax = fig.add_subplot(1, 2, 1, projection=proj)
    lons, lats, Z = _grid(risk, "expected_hazards")
    vmax = max(float(np.nanpercentile(Z[np.isfinite(Z)], 98)), 0.5)
    edges = _bin_edges(vmax)
    base = plt.get_cmap("inferno_r")
    cmap = ListedColormap(base(np.linspace(0.15, 1.0, len(edges) - 1)))
    mesh = ax.pcolormesh(lons, lats, Z, cmap=cmap, norm=BoundaryNorm(edges, cmap.N),
                         transform=ccrs.PlateCarree(), shading="nearest")
    try:
        from .geo import conus_states
        ax.add_geometries(conus_states(), ccrs.PlateCarree(), facecolor="none",
                          edgecolor="0.4", linewidth=0.5)
    except Exception:
        pass
    ax.plot(g.lon, g.lat, marker="*", markersize=20, color="#00d2ff",
            markeredgecolor="black", transform=ccrs.PlateCarree(), zorder=5)
    ax.set_extent([bbox[0], bbox[2], bbox[1], bbox[3]], crs=ccrs.PlateCarree())
    ax.set_title("Expected number of hazards (next 7 days)", fontsize=11, weight="bold")
    cb = fig.colorbar(mesh, ax=ax, orientation="horizontal", location="bottom",
                      shrink=1.0, fraction=0.06, pad=0.05, ticks=edges)
    cb.set_ticklabels([_fmt(e, vmax) for e in edges]); cb.set_label("expected # of hazards")

    # Bar chart of this parcel's per-hazard scores, flagging calibrated vs index.
    from .models.predictors import CALIBRATED
    ax2 = fig.add_subplot(1, 2, 2)
    order = sorted(pr.hazards.items(), key=lambda kv: kv[1])
    names = [HAZARD_TITLES.get(f"p_{k}", k) + ("" if k in CALIBRATED else " *")
             for k, _ in order]
    vals = [v for _, v in order]
    colors = [plt.get_cmap(HAZARD_CMAPS.get(f"p_{k}", "YlOrRd"))(0.7) for k, _ in order]
    bars = ax2.barh(names, vals, color=colors, edgecolor="0.3")
    # 10-90% forecast-ensemble interval for the weather-driven hazards.
    xerr_lo, xerr_hi, has_err = [], [], False
    for k, v in order:
        if k in pr.uncertainty:
            _, p10, p90 = pr.uncertainty[k]
            xerr_lo.append(max(0.0, v - p10)); xerr_hi.append(max(0.0, p90 - v)); has_err = True
        else:
            xerr_lo.append(0.0); xerr_hi.append(0.0)
    if has_err:
        ax2.errorbar(vals, range(len(vals)), xerr=[xerr_lo, xerr_hi], fmt="none",
                     ecolor="0.25", elinewidth=1.3, capsize=3, zorder=5)
    for b, v in zip(bars, vals):
        ax2.text(v + 0.012, b.get_y() + b.get_height() / 2, f"{v:.2f}",
                 va="center", fontsize=9)
    top = max(0.3, max(vals) * 1.25, max(pr.uncertainty.get(k, (0, 0, 0))[2]
                                         for k, _ in order) * 1.1 if pr.uncertainty else 0.3)
    ax2.set_xlim(0, top)
    ax2.set_xlabel("calibrated probability  |  * = relative hazard index (0-1)")
    ax2.set_title("Hazard level at this address (next 7 days)", fontsize=11, weight="bold")
    ax2.grid(axis="x", alpha=0.3)

    fig.suptitle(f"Cascadia parcel hazard report\n{pr.address}", fontsize=13, weight="bold")
    fig.text(0.5, 0.05, f"Combined: {pr.compound_risk:.2f}  ·  expected # hazards: "
             f"{pr.expected_hazards:.2f}  ·  dominant cascade: {pr.dominant_chain or '—'}",
             ha="center", fontsize=9, color="0.3")
    fig.text(0.5, 0.005,
             "flood & earthquake are calibrated probabilities; * landslide / wildfire / "
             "heat / smoke are relative 0-1 indices (area-scale danger, not address-"
             "specific odds). Error bars = 10-90% range across 31 GFS ensemble members "
             "(forecast uncertainty). Landslide uses the address's local DEM slope. "
             "Research prototype — defer to official sources.",
             ha="center", fontsize=7.5, color="0.45", wrap=True)
    fig.tight_layout(rect=[0, 0.03, 1, 0.93])
    out_path = Path(out_path)
    fig.savefig(out_path, dpi=140, bbox_inches="tight")
    plt.close(fig)
    if verbose:
        print(f"Parcel report written: {out_path}")
    return out_path


def assess_address(address: str, config: Config | None = None) -> dict:
    """Full address -> risk: geocode then assess. Returns a JSON-ready dict."""
    config = config or Config.load()
    g = geocode(address, config)
    if not g.matched:
        return {"address": address, "matched": False,
                "error": "address could not be geocoded"}
    pr = assess_point(g.lat, g.lon, config=config, state=g.state,
                      address=g.matched_address)
    from .models.predictors import HAZARD_KIND
    out = pr.to_dict()
    out["matched"] = True
    out["hazard_kind"] = {h: HAZARD_KIND.get(h, "index") for h in pr.hazards}
    out["note"] = (
        "flood & earthquake are calibrated probabilities; landslide/wildfire/heat/"
        "smoke are relative 0-1 hazard indices (area-scale danger, ~5km, not "
        "address-specific odds). Landslide is refined by the address's local DEM "
        "slope. Research prototype — not for operational decisions.")
    return out
