"""Terrain slope from a free DEM (Open-Meteo elevation API, Copernicus ~90 m).

Landslides are fundamentally a *slope* phenomenon, but the landslide-inventory
density prior only knows *where slides have happened* regionally — it can't tell a
flat valley floor from the hillside next to it. Adding slope fixes that: a flat
downtown lot in a landslide-prone region is correctly downweighted.

No API key. Elevation barely changes, so results are cached aggressively.
"""
from __future__ import annotations

import hashlib
import json
from pathlib import Path

import numpy as np
import requests

ELEV_URL = "https://api.open-meteo.com/v1/elevation"


def _fetch_elevations(lats, lons, timeout: int = 60, retries: int = 4) -> np.ndarray:
    """Elevation (m) for points, in chunks of 100 (the API limit), with retries on
    failed chunks (the batch DEM API rate-limits, leaving NaN holes otherwise)."""
    import time
    lats, lons = np.asarray(lats, float), np.asarray(lons, float)
    out = np.full(len(lats), np.nan)
    for i in range(0, len(lats), 100):
        sl = slice(i, i + 100)
        params = {"latitude": ",".join(f"{v:.5f}" for v in lats[sl]),
                  "longitude": ",".join(f"{v:.5f}" for v in lons[sl])}
        for attempt in range(retries):
            try:
                r = requests.get(ELEV_URL, params=params, timeout=timeout)
                r.raise_for_status()
                out[sl] = r.json()["elevation"]
                break
            except Exception:
                if attempt < retries - 1:
                    time.sleep(1.5 * (attempt + 1))
    return out


def point_slope(lat: float, lon: float, d: float = 0.0025) -> float:
    """Local slope (degrees) at a point, from a ~250 m DEM stencil (center+NSEW)."""
    pts_lat = [lat, lat + d, lat - d, lat, lat]
    pts_lon = [lon, lon, lon, lon + d, lon - d]
    e = _fetch_elevations(pts_lat, pts_lon)
    if np.isnan(e).any():
        return float("nan")
    coslat = np.cos(np.radians(lat))
    dz_dy = (e[1] - e[2]) / (2 * d * 111000.0)
    dz_dx = (e[3] - e[4]) / (2 * d * 111000.0 * coslat)
    return float(np.degrees(np.arctan(np.hypot(dz_dx, dz_dy))))


def cell_slopes(cells, cache_dir: Path, res_deg: float) -> np.ndarray:
    """Slope (degrees) per grid cell, from cell-centre elevations + neighbour
    finite differences. Cached per region (elevation is static)."""
    lats = cells["lat"].to_numpy(); lons = cells["lon"].to_numpy()
    key = hashlib.sha1(
        f"{lats.min():.3f}_{lats.max():.3f}_{lons.min():.3f}_{lons.max():.3f}"
        f"_{res_deg}_{len(lats)}".encode()).hexdigest()[:16]
    cache = Path(cache_dir) / f"slope_{key}.json"
    if cache.exists():
        try:
            return np.array(json.loads(cache.read_text()))
        except Exception:
            pass
    elev = _fetch_elevations(lats, lons)
    look = {(round(la, 4), round(lo, 4)): e
            for la, lo, e in zip(lats, lons, elev)}

    def g(la, lo):
        return look.get((round(la, 4), round(lo, 4)), np.nan)

    slopes = np.zeros(len(lats))
    for i, (la, lo) in enumerate(zip(lats, lons)):
        e0 = elev[i]
        if np.isnan(e0):
            continue
        en, es = g(la + res_deg, lo), g(la - res_deg, lo)
        ee, ew = g(la, lo + res_deg), g(la, lo - res_deg)
        en = e0 if np.isnan(en) else en; es = e0 if np.isnan(es) else es
        ee = e0 if np.isnan(ee) else ee; ew = e0 if np.isnan(ew) else ew
        coslat = np.cos(np.radians(la))
        dz_dy = (en - es) / (2 * res_deg * 111000.0)
        dz_dx = (ee - ew) / (2 * res_deg * 111000.0 * coslat)
        slopes[i] = np.degrees(np.arctan(np.hypot(dz_dx, dz_dy)))
    try:
        cache.write_text(json.dumps(slopes.tolist()))
    except Exception:
        pass
    return slopes


def slope_factor(slope_deg) -> np.ndarray:
    """Landslide slope susceptibility multiplier in [0,1]: ~0 below ~3 deg
    (flat = stable), rising to 1 by ~20 deg. Linear ramp 3->20 degrees."""
    return np.clip((np.asarray(slope_deg, float) - 3.0) / 17.0, 0.0, 1.0)


def cell_relief(cells, cache_dir: Path, res_deg: float) -> np.ndarray:
    """Local topographic relief (metres) per grid cell = elevation range over the
    cell + its 8 neighbours. A scale-appropriate terrain gate for COARSE grids,
    where a slope *angle* would be near-zero everywhere: flat terrain (Florida, the
    Great Plains) has ~tens of m of relief, real mountains have hundreds+. Cached
    per region (elevation is static)."""
    lats = cells["lat"].to_numpy(); lons = cells["lon"].to_numpy()
    key = hashlib.sha1(
        f"relief_{lats.min():.3f}_{lats.max():.3f}_{lons.min():.3f}_{lons.max():.3f}"
        f"_{res_deg}_{len(lats)}".encode()).hexdigest()[:16]
    cache = Path(cache_dir) / f"relief_{key}.json"
    if cache.exists():
        try:
            return np.array(json.loads(cache.read_text()))
        except Exception:
            pass
    elev = _fetch_elevations(lats, lons)
    # Spatial neighbourhood via KDTree on projected km — robust to grid alignment
    # / float rounding (the exact-coordinate lookup silently missed neighbours).
    from scipy.spatial import cKDTree
    coslat = np.cos(np.radians(np.nanmean(lats)))
    xy = np.column_stack([lons * 111.0 * coslat, lats * 111.0])
    tree = cKDTree(xy)
    radius_km = res_deg * 111.0 * 1.8   # ~the 8-neighbour ring
    relief = np.zeros(len(lats))
    for i in range(len(lats)):
        nb = tree.query_ball_point(xy[i], radius_km)
        vals = elev[nb]
        vals = vals[np.isfinite(vals)]
        relief[i] = float(np.ptp(vals)) if len(vals) >= 2 else 0.0
    try:
        cache.write_text(json.dumps(relief.tolist()))
    except Exception:
        pass
    return relief


def relief_factor(relief_m, r0: float = 40.0, r1: float = 300.0) -> np.ndarray:
    """Terrain gate in [0,1] from local relief (m): ~0 on flat ground (relief <
    ~40 m, e.g. Florida / the Plains), ramping to 1 by ~300 m of local relief."""
    return np.clip((np.asarray(relief_m, float) - r0) / (r1 - r0), 0.0, 1.0)
