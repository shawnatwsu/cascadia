"""Sub-seasonal (weeks 2-6) hazard outlook — land-memory persistence baseline.

Honest framing: skillful sub-seasonal prediction is hard. Most of the genuine
2-6 week predictability for *fire, drought and heat* comes from slow land-surface
memory — deep/large dead-fuel moisture, drought state, and accumulated
precipitation deficit persist for weeks. This module builds an outlook from
exactly those slow GRIDMET variables (1000-hr fuel moisture, ERC, 90-day
precip-minus-ET deficit, recent warmth), and is explicitly a **persistence /
land-memory baseline** — the floor that operational dynamical S2S products
(NOAA CPC week-3-4, NMME) must beat. Those are the documented next upgrade.

Flood/landslide/earthquake are NOT sub-seasonally predictable from this and are
intentionally omitted from the outlook.
"""
from __future__ import annotations

from datetime import datetime, timedelta
from pathlib import Path

import numpy as np
import xarray as xr

from .config import Config, Region
from .conditions import REGIONS, DEFAULT_RES, GRIDMET_STRIDE
from .features.grid import Grid


def _sig(x):
    return 1.0 / (1.0 + np.exp(-x))


def subseasonal_outlook(region_key: str = "pnw", out_path: str | Path =
                        "cascadia_subseasonal_map.png", verbose: bool = True):
    """Build + render the weeks 2-6 fire/drought/heat outlook for a region."""
    from .sources.gridmet import region_daily
    from .cartomap import static_risk_map

    bbox = REGIONS[region_key]
    res = DEFAULT_RES.get(region_key, 0.1)
    base = Config.load()
    region = Region(name=f"{region_key.upper()} sub-seasonal", bbox=bbox,
                    state=base.region.state, grid_resolution_deg=res)
    cfg = Config(region=region, horizon_days=base.horizon_days,
                 sources=base.sources, cache_dir=base.cache_dir, raw=base.raw)
    grid = Grid.from_region(region)
    cells = grid.cells_frame(land_only=True)
    log = (lambda *a: print(*a)) if verbose else (lambda *a: None)
    log(f"{region.name}: {len(cells)} cells")

    # 90-day window of slow-memory variables (ending at GRIDMET latency edge).
    end = (datetime.utcnow().date() - timedelta(days=2))
    start = end - timedelta(days=90)
    cube = region_daily(bbox, start.isoformat(), end.isoformat(), cfg.cache_dir,
                        variables=["fm1000", "erc", "precip_mm", "etr",
                                   "tmax_c", "vpd_kpa"],
                        stride=GRIDMET_STRIDE.get(region_key, 1), verbose=False)
    log(f"GRIDMET 90d {dict(cube.sizes)} ({start}..{end})")

    la = xr.DataArray(cells["lat"].to_numpy(), dims="cell")
    lo = xr.DataArray(cells["lon"].to_numpy(), dims="cell")

    def samp(field):
        return field.sel(lat=la, lon=lo, method="nearest").to_numpy()

    # Slow-memory state (recent ~14-day mean = the persistent condition).
    recent = cube.isel(time=slice(-14, None))
    fm1000 = samp(recent["fm1000"].mean("time"))          # % (low = dry fuels)
    erc = samp(recent["erc"].mean("time"))                # higher = drier/danger
    vpd = samp(recent["vpd_kpa"].mean("time"))
    tmax = samp(recent["tmax_c"].mean("time"))
    # 90-day water-balance deficit (negative = drought).
    deficit = samp((cube["precip_mm"].sum("time") - 0.6 * cube["etr"].sum("time")))

    # --- Outlook layers (persistence of the slow state into weeks 2-6) -----
    # Fire potential: dry large fuels + high ERC + high VPD.
    fuel_dry = _sig((12.0 - fm1000) / 3.0)
    erc_d = _sig((erc - 55.0) / 18.0)
    vpd_d = _sig((vpd - 1.5) / 0.8)
    cells["fire_outlook"] = np.clip(0.45 * fuel_dry + 0.35 * erc_d + 0.20 * vpd_d, 0, 1)
    # Drought / dryness: precip-ET deficit + dry deep fuels.
    cells["drought_outlook"] = np.clip(_sig(-deficit / 80.0) * 0.7 + 0.3 * fuel_dry, 0, 1)
    # Heat tendency: persistence of recent warmth (absolute, VPD-aided).
    cells["heat_outlook"] = np.clip(_sig((tmax - 30.0) / 4.0), 0, 1)

    log("outlook means: fire %.3f drought %.3f heat %.3f" % (
        cells["fire_outlook"].mean(), cells["drought_outlook"].mean(),
        cells["heat_outlook"].mean()))

    out = static_risk_map(
        cells, region.name, out_path,
        cols=["fire_outlook", "drought_outlook", "heat_outlook"],
        suptitle=("Cascadia — sub-seasonal hazard outlook (weeks 2-6)\n"
                  f"{region.name}  ·  land-memory persistence baseline"))
    log(f"Map written: {out}")
    return cells, out
