"""ENSO-conditioned seasonal hazard outlook (1-3 months), per NCA5 region.

El Nino / La Nina shifts the odds of a wet/dry, warm/cool season differently in
each part of the country. This module reads the current ENSO state (ONI) and
applies documented regional teleconnection responses to produce a seasonal
*tendency* for fire/drought, wet/flood, and warm-season hazards.

This is a composite/teleconnection baseline (the published ENSO impacts scaled
by the current ONI). `training/train_enso.py` learns these regional responses
directly from 1950-present history and reports their skill — replacing the
hand-coded coefficients with data-driven ones.
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

from .config import Config, Region
from .features.grid import Grid

# Documented ENSO teleconnection response (per region) to ONI (positive = El Nino).
# (precip_coef, temp_coef): sign of the seasonal precip / temperature anomaly per
# unit ONI. La Nina (negative ONI) flips the signs automatically.
ENSO_RESPONSE = {
    "northwest":            (-0.50, 0.40),   # El Nino: drier, warmer PNW
    "southwest":            (0.50, -0.10),   # El Nino: wetter SW/CA
    "southern_great_plains": (0.60, -0.30),  # El Nino: wetter, cooler
    "northern_great_plains": (0.00, 0.40),   # El Nino: warmer north
    "midwest":              (-0.10, 0.40),   # El Nino: warmer, slightly drier
    "southeast":            (0.60, -0.30),   # El Nino: wetter, cooler SE
    "northeast":            (0.10, 0.20),    # El Nino: slightly warmer/wetter
}


def _sig(x):
    return 1.0 / (1.0 + np.exp(-x))


def seasonal_outlook(out_path: str | Path = "cascadia_seasonal_map.png",
                     oni_override: float | None = None, verbose: bool = True):
    """Render the ENSO-driven seasonal hazard outlook over CONUS by NCA region."""
    from . import geo
    from .sources.enso import current_state
    from .cartomap import static_risk_map

    cfg = Config.load()
    enso = current_state(cfg)
    oni = oni_override if oni_override is not None else enso.oni
    log = (lambda *a: print(*a)) if verbose else (lambda *a: None)
    log(f"ENSO: {enso.label()} | season {enso.season} {enso.year} | using ONI {oni:+.2f}")

    region = Region(name="CONUS seasonal", bbox=(-125.0, 24.5, -66.9, 49.5),
                    state="US", grid_resolution_deg=0.25)
    cells = geo.mask_conus(Grid.from_region(region).cells_frame(land_only=True))
    for col in ("seasonal_fire", "seasonal_flood", "seasonal_heat"):
        cells[col] = np.nan

    for rkey, (p_coef, t_coef) in ENSO_RESPONSE.items():
        mask = geo.mask_region(cells[["lat", "lon"]].assign(cell_id=cells["cell_id"]), rkey)
        if mask.empty:
            continue
        ids = set(mask["cell_id"])
        sel = cells["cell_id"].isin(ids)
        precip_anom = p_coef * oni        # + = wetter season
        temp_anom = t_coef * oni          # + = warmer season
        # Hazard tendencies (0.5 = climatological normal).
        cells.loc[sel, "seasonal_fire"] = np.clip(_sig(1.6 * (temp_anom - precip_anom)), 0, 1)
        cells.loc[sel, "seasonal_flood"] = np.clip(_sig(1.8 * precip_anom), 0, 1)
        cells.loc[sel, "seasonal_heat"] = np.clip(_sig(1.8 * temp_anom), 0, 1)

    log("seasonal tendency means: fire %.2f flood %.2f heat %.2f" % (
        cells["seasonal_fire"].mean(), cells["seasonal_flood"].mean(),
        cells["seasonal_heat"].mean()))

    out = static_risk_map(
        cells, "CONUS", out_path,
        cols=["seasonal_fire", "seasonal_flood", "seasonal_heat"],
        boundaries=geo.conus_states(), value_label="seasonal tendency (0.5=normal)",
        suptitle=(f"Cascadia — ENSO seasonal hazard outlook (next 1-3 months)\n"
                  f"{enso.label()} · {enso.season} {enso.year}"),
        description=(
            "IN PLAIN TERMS: El Nino and La Nina tilt the odds of a wet/dry, "
            "warm/cool season differently in each region. Given the current ocean "
            "state, this shows where fire/drought, flooding, and heat are MORE (>0.5) "
            "or LESS (<0.5) likely than a normal season over the next 1-3 months.   |   "
            "Method: NOAA ONI (El Nino index) x documented regional teleconnection "
            "responses (NCA5 regions). A composite baseline; train_enso.py learns "
            "these responses from 1950-present and scores their skill. Weak ENSO = "
            "weak signal (near 0.5 everywhere)."))
    log(f"Seasonal outlook written: {out}")
    return cells, out
