"""GRIDMET-driven high-resolution hazard *conditions* map over a large region.

Unlike the main forecast pipeline (small region, Open-Meteo 7-day forecast), this
renders a 4 km **nowcast** of current hazard conditions across a big region —
PNW or CONUS — using GRIDMET (no rate limits, server-side OPeNDAP subsetting) for
all meteorological hazards, plus the seismic/landslide priors and USGS
streamflow.

Honest scope: GRIDMET is observed (to ~5 days ago), so this answers "where is
hazard elevated right now" rather than "what's coming next week". Soil moisture
isn't in GRIDMET, so landslide uses an antecedent precip-minus-ET wetness proxy
(which is the standard rainfall-trigger basis anyway), and the heuristic flood
predictor (precip + streamflow) is used rather than the soil-trained model.
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

from .config import Config, Region
from .features.grid import Grid
from .features.indicators import (_landslide_susceptibility, _nearest_join,
                                  _seismic_base_prob, _streamflow_anomaly,
                                  hot_dry_windy)
from .models import base_probabilities, default_cascade

REGIONS = {
    "pnw": (-124.8, 41.9, -116.5, 49.1),
    "conus": (-125.0, 24.5, -66.9, 49.5),
    "california": (-124.5, 32.5, -114.1, 42.1),
}
DEFAULT_RES = {"pnw": 0.1, "california": 0.1, "conus": 0.25}
# GRIDMET stride (native 4km): coarsen for big regions to bound the download.
GRIDMET_STRIDE = {"pnw": 1, "california": 1, "conus": 6}


def conditions_map(region_key: str = "pnw", resolution_deg: float | None = None,
                   out_path: str | Path = "cascadia_conditions_map.png",
                   verbose: bool = True):
    """Build and render the GRIDMET conditions nowcast for a named region."""
    from .sources.gridmet import (derive_cell_features, gridmet_window,
                                   region_daily)
    from .sources.seismicity import Seismicity
    from .sources.landslide_inventory import LandslideInventory
    from .sources import USGSWater, FIRMS
    from .cartomap import static_risk_map

    bbox = REGIONS[region_key]
    res = resolution_deg or DEFAULT_RES.get(region_key, 0.1)
    base = Config.load()
    region = Region(name=f"{region_key.upper()} conditions", bbox=bbox,
                    state=base.region.state, grid_resolution_deg=res)
    cfg = Config(region=region, horizon_days=base.horizon_days,
                 sources=base.sources, cache_dir=base.cache_dir, raw=base.raw)

    grid = Grid.from_region(region)
    cells = grid.cells_frame(land_only=True)
    log = (lambda *a: print(*a)) if verbose else (lambda *a: None)
    log(f"{region.name}: {len(cells)} cells @ {res}° ({bbox})")

    # --- GRIDMET 4km met + fire/heat -------------------------------------
    gstart, gend = gridmet_window(cfg)
    cube = region_daily(bbox, gstart, gend, cfg.cache_dir,
                        stride=GRIDMET_STRIDE.get(region_key, 1), verbose=False)
    log(f"GRIDMET {dict(cube.sizes)} ({gstart}..{gend})")
    gm = derive_cell_features(cube, cells)
    cells = cells.merge(gm, on="cell_id", how="left")

    # precip total + antecedent wetness proxy (precip - 0.6*ET), scaled into the
    # volumetric-soil-moisture range the saturation logic expects.
    tlat = cube["precip_mm"].sel  # noqa (clarity)
    import xarray as xr
    la = xr.DataArray(cells["lat"].to_numpy(), dims="cell")
    lo = xr.DataArray(cells["lon"].to_numpy(), dims="cell")
    precip_sum = cube["precip_mm"].sum("time").sel(lat=la, lon=lo, method="nearest").to_numpy()
    cells["precip_total_mm"] = precip_sum
    if "etr" in cube:
        et_sum = cube["etr"].sum("time").sel(lat=la, lon=lo, method="nearest").to_numpy()
    else:
        et_sum = np.zeros_like(precip_sum)
    wb = precip_sum - 0.6 * et_sum
    cells["soil_moist_peak"] = np.clip(0.15 + 0.004 * wb, 0.05, 0.45)

    # temp/RH/wind + HDW from GRIDMET window extremes (fallbacks; fire prefers BI)
    cells["temp_max"] = cells.get("gm_tmax", 20.0)
    cells["rh_min"] = cells.get("gm_rhmin", 40.0)
    cells["wind_max"] = cells.get("gm_wind", 3.0)
    cells["hdw"] = hot_dry_windy(cells["temp_max"].to_numpy(),
                                 cells["rh_min"].to_numpy(),
                                 cells["wind_max"].to_numpy() * 3.6)  # m/s->km/h

    # --- streamflow anomaly (USGS, region bbox) --------------------------
    try:
        water = USGSWater(cfg).fetch()
        flow = _streamflow_anomaly(water)
        cells["flow_anomaly"] = _nearest_join(cells, flow, "flow_anomaly")
    except Exception:
        cells["flow_anomaly"] = 0.1
    log(f"streamflow anomaly joined")

    # --- static priors ---------------------------------------------------
    cat = Seismicity(cfg).fetch()
    cells["eq_base_prob"] = _seismic_base_prob(cells, cat, res, cfg.horizon_days)
    cells["quake_mag"] = 0.0   # no per-cell aftershock layer in the broad nowcast
    inv = LandslideInventory(cfg).fetch()
    cells["ls_susceptibility"] = _landslide_susceptibility(cells, inv)
    log(f"priors: {len(cat)} quakes, {len(inv)} landslides")

    # --- observed fire + (no regional alerts in nowcast mode) ------------
    try:
        fires = FIRMS(cfg).fetch()
        if fires is not None and not fires.empty:
            fc = grid.assign(fires).dropna(subset=["cell_id"])
            cnt = fc.groupby("cell_id").size()
            cells["active_fire"] = cells["cell_id"].map(cnt).fillna(0.0)
        else:
            cells["active_fire"] = 0.0
    except Exception:
        cells["active_fire"] = 0.0
    cells["alert_flood"] = cells["alert_fire_weather"] = cells["alert_heat"] = 0.0

    # --- predict (heuristics; trained flood needs soil moisture) ---------
    bp = base_probabilities(cells, use_trained=False)
    risk = default_cascade().run(bp, cells)
    log(f"risk computed: compound mean {risk['compound_risk'].mean():.3f} "
        f"max {risk['compound_risk'].max():.3f}")

    out = static_risk_map(risk, region.name, out_path, panels=True, as_of="nowcast")
    log(f"Map written: {out}")
    return risk, out
