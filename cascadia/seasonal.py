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
                     oni_override: float | None = None, lead: int = 0,
                     verbose: bool = True):
    """Render the ENSO-driven seasonal hazard outlook over CONUS by NCA region.

    lead=0 uses the current ONI; lead=1/2/3 uses the trained ENSO forecast model's
    predicted ONI that many months ahead (a genuinely forward-looking outlook).
    """
    from . import geo
    from .sources.enso import current_state
    from .cartomap import static_risk_map

    cfg = Config.load()
    enso = current_state(cfg)
    log = (lambda *a: print(*a)) if verbose else (lambda *a: None)
    lead_note = ""
    if oni_override is not None:
        oni = oni_override
    elif lead > 0:
        from .training.train_enso import forecast
        fc = forecast(cfg)
        oni = fc.get(lead, enso.oni) if fc else enso.oni
        lead_note = f"  ·  {lead}-month ENSO forecast"
        log(f"ENSO {lead}-month forecast ONI {oni:+.2f} (current {enso.oni:+.2f})")
    else:
        oni = enso.oni
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
        # Hazard ANOMALY (0 = climatological normal; +/- = more/less likely than
        # normal). Kept on the ONI-anomaly scale so a weak ENSO reads as weak.
        cells.loc[sel, "seasonal_fire"] = np.clip(0.5 * (temp_anom - precip_anom), -0.5, 0.5)
        cells.loc[sel, "seasonal_flood"] = np.clip(0.6 * precip_anom, -0.5, 0.5)
        cells.loc[sel, "seasonal_heat"] = np.clip(0.6 * temp_anom, -0.5, 0.5)

    log("seasonal tendency means: fire %.2f flood %.2f heat %.2f" % (
        cells["seasonal_fire"].mean(), cells["seasonal_flood"].mean(),
        cells["seasonal_heat"].mean()))

    out = static_risk_map(
        cells, "CONUS", out_path,
        cols=["seasonal_fire", "seasonal_flood", "seasonal_heat"],
        boundaries=geo.conus_states(), diverging=True,
        value_label="anomaly vs normal (0 = normal)",
        provenance=(f"NOAA CPC Oceanic Nino Index (ONI {oni:+.2f}) x NCA5 regional "
                    "teleconnection composites · Albers equal-area"),
        suptitle=(f"Cascadia — ENSO seasonal hazard outlook (next 1-3 months)\n"
                  f"{enso.label()} · {enso.season} {enso.year}{lead_note}"),
        description=(
            "IN PLAIN TERMS: El Nino and La Nina tilt the odds of a wet/dry, "
            "warm/cool season differently in each region. RED = this hazard is MORE "
            "likely than a normal season; BLUE = LESS likely; WHITE = near normal. "
            "Right now ENSO is weak, so most regions are near normal (pale).   |   "
            "Method: NOAA ONI (El Nino index) x documented regional teleconnection "
            "responses (NCA5 regions) — a composite baseline. NOT YET VALIDATED "
            "against observed outcomes; train_enso.py learns/scores these responses "
            "from 1950-present (skill validation in progress)."))
    log(f"Seasonal outlook written: {out}")
    return cells, out
