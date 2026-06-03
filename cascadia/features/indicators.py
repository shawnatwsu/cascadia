"""Cross-sector leading indicators, aggregated to grid cells.

Turns the normalized observation frames from each source into a single
per-cell feature table. This is where signals from different sectors
(seismic, hydro-met, hydrology) are fused onto a common spatial frame.

One genuinely ML step lives here: an IsolationForest flags anomalous
streamflow behaviour (a leading indicator of flooding) without needing
labelled history — it learns "normal" from the gage's own recent record.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.ensemble import IsolationForest

from .grid import Grid


def _streamflow_anomaly(water: pd.DataFrame) -> pd.DataFrame:
    """Per-site streamflow anomaly score in [0,1] from recent discharge.

    Combines a robust percentile of the latest reading against the site's own
    recent window with an IsolationForest novelty score on (level, rate-of-rise).
    Returns one row per site with lat/lon and `flow_anomaly`.
    """
    if water.empty:
        return pd.DataFrame(columns=["lat", "lon", "flow_anomaly"])

    disc = water[water["param"] == "discharge_cfs"].copy()
    if disc.empty:
        return pd.DataFrame(columns=["lat", "lon", "flow_anomaly"])

    out = []
    for site, g in disc.groupby("site"):
        g = g.sort_values("time")
        vals = g["value"].to_numpy(dtype=float)
        if len(vals) < 8 or np.allclose(vals, vals[0]):
            continue
        # Percentile of the latest reading within the site's own window.
        latest = vals[-1]
        pct = (vals < latest).mean()
        # Rate of rise over the last few readings, normalized by site scale.
        scale = np.median(np.abs(vals)) + 1e-6
        rise = (vals[-1] - vals[-min(6, len(vals))]) / scale
        # IsolationForest on (value, local slope) — unsupervised novelty.
        slope = np.gradient(vals)
        X = np.column_stack([vals, slope])
        try:
            iso = IsolationForest(contamination=0.1, random_state=0).fit(X)
            nov = -iso.score_samples(X[-1:].reshape(1, -1))[0]  # higher = odder
            nov = float(np.clip((nov - 0.4) / 0.4, 0, 1))
        except Exception:
            nov = 0.0
        anomaly = float(np.clip(0.5 * pct + 0.3 * np.tanh(max(rise, 0)) + 0.2 * nov, 0, 1))
        out.append(
            {
                "lat": g["lat"].iloc[0],
                "lon": g["lon"].iloc[0],
                "flow_anomaly": anomaly,
            }
        )
    return pd.DataFrame(out)


def _forecast_point_features(weather: pd.DataFrame, horizon_days: int) -> pd.DataFrame:
    """Aggregate the forecast to its sample POINTS over the horizon.

    Returns one row per (lat, lon) forecast point with cumulative precip and
    peak deep-soil-moisture. These coarse points are later nearest-joined to
    grid cells, giving a smooth spatial field instead of a sparse exact match.
    """
    if weather.empty:
        return pd.DataFrame(columns=["lat", "lon", "precip_total_mm", "soil_moist_peak"])

    w = weather.copy()
    horizon_cut = w["time"].min() + pd.Timedelta(days=horizon_days)
    w = w[w["time"] <= horizon_cut]

    sm_cols = [c for c in w.columns if c.startswith("soil_moisture")]
    agg_spec = {
        "precip_total_mm": ("precipitation", "sum"),
        "soil_moist_peak": (sm_cols[-1], "max") if sm_cols else ("precipitation", "max"),
    }
    if "temperature_2m" in w.columns:
        agg_spec["temp_max"] = ("temperature_2m", "max")
    if "relative_humidity_2m" in w.columns:
        agg_spec["rh_min"] = ("relative_humidity_2m", "min")
    if "wind_speed_10m" in w.columns:
        agg_spec["wind_max"] = ("wind_speed_10m", "max")
    # Compute HDW PER HOUR then take the window max — the correct way (combining
    # separately-aggregated extremes would overestimate, since the hottest,
    # driest, and windiest moments don't co-occur).
    if {"temperature_2m", "relative_humidity_2m", "wind_speed_10m"} <= set(w.columns):
        w = w.assign(hdw_h=hot_dry_windy(
            w["temperature_2m"].to_numpy(), w["relative_humidity_2m"].to_numpy(),
            w["wind_speed_10m"].to_numpy()))
        agg_spec["hdw"] = ("hdw_h", "max")
    agg = w.groupby(["lat", "lon"]).agg(**agg_spec)
    return agg.reset_index()


def hot_dry_windy(temp_c: np.ndarray, rh_pct: np.ndarray,
                  wind: np.ndarray) -> np.ndarray:
    """Hot-Dry-Windy fire-weather index = VPD(hPa) x wind(m/s).

    A modern, operationally-used measure of atmospheric fire-spread potential
    (Srock et al. 2018). VPD is the vapour-pressure deficit; higher HDW = more
    dangerous fire weather. Open-Meteo wind is km/h -> convert to m/s.
    """
    es = 0.6108 * np.exp(17.27 * temp_c / (temp_c + 237.3))   # kPa
    vpd_kpa = np.clip(es * (1.0 - np.clip(rh_pct, 0, 100) / 100.0), 0, None)
    wind_ms = np.clip(wind, 0, None) / 3.6
    return vpd_kpa * 10.0 * wind_ms   # hPa * m/s


def _nearest_join(cells: pd.DataFrame, pts: pd.DataFrame, col: str) -> pd.Series:
    """Assign each cell the value of the nearest point observation (simple, fast)."""
    if pts.empty or col not in pts:
        return pd.Series(0.0, index=cells.index)
    pts = pts.dropna(subset=["lat", "lon", col])
    if pts.empty:
        return pd.Series(0.0, index=cells.index)
    plat = pts["lat"].to_numpy(); plon = pts["lon"].to_numpy(); pval = pts[col].to_numpy()
    vals = []
    for la, lo in zip(cells["lat"], cells["lon"]):
        d2 = (plat - la) ** 2 + (plon - lo) ** 2
        vals.append(float(pval[int(np.argmin(d2))]))
    return pd.Series(vals, index=cells.index)


def _smoothed_density(cell_lat, cell_lon, pt_lat, pt_lon, bandwidth_km):
    """Gaussian-kernel density at each cell from point locations, using a
    cKDTree fixed-radius query so it scales to CONUS (100k+ points, 100k+ cells).

    Returns the summed kernel weight per cell (not yet normalized)."""
    import numpy as np
    from scipy.spatial import cKDTree
    if len(pt_lat) == 0:
        return np.zeros(len(cell_lat))
    lat0 = float(np.mean(cell_lat))
    kx = 111.0 * np.cos(np.radians(lat0))
    pts = np.column_stack([np.asarray(pt_lon) * kx, np.asarray(pt_lat) * 111.0])
    cel = np.column_stack([np.asarray(cell_lon) * kx, np.asarray(cell_lat) * 111.0])
    tree = cKDTree(pts)
    r = 3.5 * bandwidth_km  # Gaussian negligible beyond ~3.5 sigma
    h2 = 2.0 * bandwidth_km * bandwidth_km
    dens = np.zeros(len(cel))
    neighbors = tree.query_ball_point(cel, r, workers=-1)
    for i, idx in enumerate(neighbors):
        if idx:
            d2 = ((pts[idx] - cel[i]) ** 2).sum(axis=1)
            dens[i] = np.exp(-d2 / h2).sum()
    return dens


def _seismic_base_prob(cells: pd.DataFrame, catalog: pd.DataFrame,
                       res_deg: float, horizon_days: int,
                       bandwidth_km: float = 30.0) -> pd.Series:
    """Smoothed-seismicity Poisson prior: P(>=1 catalog-magnitude event in the
    horizon) per cell, from decades of historical epicenters.

    Each historical event is spread over space by a 2-D Gaussian kernel
    (normalized so one event contributes a total of one expected event). Summed
    and divided by the catalog length, this yields a long-run annual rate per
    cell; a Poisson model converts rate -> probability over the horizon.
    """
    import numpy as np
    if catalog is None or catalog.empty:
        return pd.Series(1e-4, index=cells.index)

    years = max(1.0, catalog.attrs.get("years", 50.0))
    h = bandwidth_km
    coslat = np.cos(np.radians(float(cells["lat"].mean())))
    cell_area = (res_deg * 111.0) * (res_deg * 111.0 * coslat)
    dens = _smoothed_density(cells["lat"].to_numpy(), cells["lon"].to_numpy(),
                             catalog["lat"].to_numpy(), catalog["lon"].to_numpy(), h)
    # density (kernel sum) -> expected events per cell -> annual rate -> Poisson P
    annual_rate = dens * (cell_area / (2 * np.pi * h * h)) / years
    p = 1.0 - np.exp(-annual_rate * horizon_days / 365.25)
    return pd.Series(np.clip(p, 1e-4, 0.999), index=cells.index)


def _landslide_susceptibility(cells: pd.DataFrame, inventory: pd.DataFrame,
                              bandwidth_km: float = 8.0) -> pd.Series:
    """Relative landslide susceptibility in [0,1] from smoothed historical
    landslide density (USGS inventory). Like the seismicity prior but, since the
    inventory is a compilation rather than a complete temporal catalog, used as a
    *relative* spatial susceptibility, not an absolute rate. A small floor keeps
    rainfall triggering meaningful everywhere."""
    import numpy as np
    if inventory is None or inventory.empty:
        return pd.Series(0.3, index=cells.index)
    d = _smoothed_density(cells["lat"].to_numpy(), cells["lon"].to_numpy(),
                          inventory["lat"].to_numpy(), inventory["lon"].to_numpy(),
                          bandwidth_km)
    # Normalize by a high percentile (robust to a few dense clusters), floor 0.15.
    scale = np.percentile(d, 95) or 1.0
    return pd.Series(np.clip(0.15 + 0.85 * d / scale, 0.0, 1.0), index=cells.index)


def build_indicators(
    grid: Grid,
    quakes: pd.DataFrame,
    weather: pd.DataFrame,
    water: pd.DataFrame,
    alerts: pd.DataFrame,
    fires: pd.DataFrame | None = None,
    catalog: pd.DataFrame | None = None,
    inventory: pd.DataFrame | None = None,
    gridmet_cube=None,
    horizon_days: int = 7,
) -> pd.DataFrame:
    """Fuse all sources into one per-cell indicator table."""
    cells = grid.cells_frame(land_only=True)

    # --- Seismic: max recent magnitude per cell (for aftershock elevation)
    q = grid.assign(quakes)
    if not q.empty and "cell_id" in q and q["cell_id"].notna().any():
        qmax = q.dropna(subset=["cell_id"]).groupby("cell_id")["value"].max()
        cells["quake_mag"] = cells["cell_id"].map(qmax).fillna(0.0)
    else:
        cells["quake_mag"] = 0.0

    # --- Seismic hazard prior: smoothed historical seismicity -> Poisson prob.
    if catalog is not None and not catalog.empty:
        from datetime import datetime, timezone
        start = pd.to_datetime(catalog.attrs.get("catalog_start", "1970-01-01"))
        catalog.attrs["years"] = max(
            1.0, (datetime.now(timezone.utc).replace(tzinfo=None) - start).days / 365.25)
    cells["eq_base_prob"] = _seismic_base_prob(
        cells, catalog, grid.res, horizon_days)

    # --- Landslide susceptibility prior (smoothed USGS inventory density).
    cells["ls_susceptibility"] = _landslide_susceptibility(cells, inventory)

    # --- GRIDMET 4km fire/heat variables (burning index, ERC, fuel moisture,
    #     VPD, heat index, wet-bulb), sampled to cells. Optional.
    if gridmet_cube is not None:
        try:
            from ..sources.gridmet import derive_cell_features
            gm = derive_cell_features(gridmet_cube, cells)
            cells = cells.merge(gm, on="cell_id", how="left")
        except Exception:
            pass

    # --- Hydro-met forecast: nearest-join the coarse forecast points to every
    #     cell, yielding a smooth field (no sparse exact-match / mean-fill artifacts).
    fc = _forecast_point_features(weather, horizon_days)
    cells["precip_total_mm"] = _nearest_join(cells, fc, "precip_total_mm")
    cells["soil_moist_peak"] = _nearest_join(cells, fc, "soil_moist_peak")
    cells["temp_max"] = _nearest_join(cells, fc, "temp_max")
    cells["rh_min"] = _nearest_join(cells, fc, "rh_min")
    cells["wind_max"] = _nearest_join(cells, fc, "wind_max")
    # Physically-based fire-weather danger (operational HDW index), computed
    # hourly upstream and carried through as the window max.
    cells["hdw"] = _nearest_join(cells, fc, "hdw")

    # --- Hydrology: nearest-gage streamflow anomaly (ML-assisted)
    flow = _streamflow_anomaly(water)
    cells["flow_anomaly"] = _nearest_join(cells, flow, "flow_anomaly")

    # --- Active fire: count of FIRMS thermal detections per cell (observed
    #     evidence that lifts the wildfire node above the dryness-only proxy).
    if fires is not None and not fires.empty:
        fc_fire = grid.assign(fires).dropna(subset=["cell_id"])
        counts = fc_fire.groupby("cell_id").size()
        cells["active_fire"] = cells["cell_id"].map(counts).fillna(0.0)
    else:
        cells["active_fire"] = 0.0

    # --- Official warnings: coarse regional flags from active NWS alerts
    fams = set(alerts["family"].unique()) if not alerts.empty and "family" in alerts else set()
    cells["alert_flood"] = 1.0 if "flood" in fams else 0.0
    cells["alert_fire_weather"] = 1.0 if "fire_weather" in fams else 0.0
    cells["alert_heat"] = 1.0 if "heat" in fams else 0.0

    return cells
