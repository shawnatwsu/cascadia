"""Per-hazard *base* probabilities from per-cell indicators.

These are the "leaf" predictors the cascade graph sits on top of. Each maps a
cell's fused indicators to P(hazard initiates here within the horizon),
*before* any triggering from other hazards.

The MVP uses transparent, physically-motivated calibrated sigmoids so the
system runs immediately with no labelled training data. Each predictor
implements the same `predict(features) -> prob` interface, so a trained
scikit-learn model (logistic / gradient boosting) can be swapped in per hazard
once historical event labels are assembled — that is the "ML" half of the
hybrid, and the seam is intentionally clean.
"""
from __future__ import annotations

from typing import Callable

import numpy as np
import pandas as pd

HAZARDS = ["earthquake", "landslide", "flood", "wildfire", "heat", "smoke"]

# How honestly each leaf's number can be read. "probability" = a calibrated or
# physically-derived probability; "index" = a relative 0-1 hazard score (NOT a
# calibrated probability of occurrence). Surfaced in reports so users don't
# over-read the index hazards.
HAZARD_KIND = {
    "flood": "probability",        # trained model, isotonic-calibrated (validated)
    "earthquake": "probability",   # smoothed-seismicity Poisson prior
    "landslide": "index",          # inventory susceptibility x slope x rain trigger
    "wildfire": "index",           # GRIDMET fire-weather danger
    "heat": "index",               # heat-index / wet-bulb
    "smoke": "index",              # downwind plume transport
}
CALIBRATED = {h for h, k in HAZARD_KIND.items() if k == "probability"}


def _sigmoid(x: np.ndarray | float) -> np.ndarray | float:
    return 1.0 / (1.0 + np.exp(-x))


# Each predictor: DataFrame of cell indicators -> array of probabilities.
Predictor = Callable[[pd.DataFrame], np.ndarray]


def _p_earthquake(f: pd.DataFrame) -> np.ndarray:
    """Earthquakes are not short-term predictable. We combine two defensible,
    data-driven pieces via noisy-OR:

    1. a *smoothed-seismicity Poisson prior* (eq_base_prob, computed from decades
       of USGS catalog epicenters) — the long-run background hazard, and
    2. an *aftershock* elevation following a recent nearby mainshock, using an
       Utsu-style productivity scaling with magnitude (10^(M - Mref)).
    """
    base = f["eq_base_prob"].to_numpy() if "eq_base_prob" in f else np.full(len(f), 1e-3)
    mag = f["quake_mag"].to_numpy()
    # Expected count of M>=4 aftershocks in the window scales ~10^(M-5.5);
    # convert to a probability and bound it.
    lam = np.where(mag > 0, np.power(10.0, mag - 5.5), 0.0)
    p_after = np.clip(1.0 - np.exp(-lam), 0.0, 0.6)
    return 1.0 - (1.0 - base) * (1.0 - p_after)


def _saturation(soil: np.ndarray) -> np.ndarray:
    # Volumetric soil moisture (m3/m3) -> saturation index in [0,1].
    # ~0.45 is near field-saturation for most soils; thresholds are absolute,
    # NOT relative to the wettest cell, so a wet region reads as wet.
    return np.clip(soil / 0.45, 0.0, 1.0)


def _p_landslide(f: pd.DataFrame) -> np.ndarray:
    """Landslide = susceptibility (where) x rainfall trigger (when).

    Susceptibility is the smoothed USGS-inventory density prior; the trigger is
    the rainfall intensity + antecedent soil saturation. Both are required: a dry
    week on a susceptible slope, or heavy rain on flat stable ground, stays low.
    """
    precip = f["precip_total_mm"].to_numpy()
    sat = _saturation(f["soil_moist_peak"].to_numpy())
    trigger = _sigmoid(0.05 * (precip - 40.0) + 2.5 * (sat - 0.6))
    suscept = f["ls_susceptibility"].to_numpy() if "ls_susceptibility" in f else 0.5
    # Slope gate: landslides need a slope. A flat lot in a slide-prone region
    # (high inventory density) is still stable. Falls back to 1.0 if no DEM.
    if "slope_deg" in f:
        from ..sources.elevation import slope_factor
        suscept = suscept * slope_factor(f["slope_deg"].to_numpy())
    return np.clip(suscept * trigger, 1e-4, 0.999)


def _p_flood(f: pd.DataFrame) -> np.ndarray:
    # Riverine flood: forecast precip + observed streamflow anomaly + any
    # active official flood warning lifts the whole region's baseline.
    precip = f["precip_total_mm"].to_numpy()
    flow = f["flow_anomaly"].to_numpy()
    alert = f["alert_flood"].to_numpy()
    z = 0.04 * (precip - 50.0) + 2.5 * (flow - 0.5) + 1.2 * alert
    return _sigmoid(z)


def _p_wildfire(f: pd.DataFrame) -> np.ndarray:
    # Fire ignition/spread potential: dryness (inverse soil moisture) + low
    # precip + fire-weather (red flag) warnings. Wet cells -> near zero.
    precip = f["precip_total_mm"].to_numpy()
    fw = f["alert_fire_weather"].to_numpy()
    dryness = 1.0 - _saturation(f["soil_moist_peak"].to_numpy())
    fire_obs = np.clip(f["active_fire"].to_numpy(), 0, None) if "active_fire" in f else 0.0

    if "gm_burning_index" in f:
        # Preferred: GRIDMET operational fire-danger (NFDRS). Burning Index and
        # Energy Release Component rise with danger; 100-hr fuel moisture falls.
        bi = f["gm_burning_index"].to_numpy()
        erc = f["gm_erc"].to_numpy() if "gm_erc" in f else bi
        fm = f["gm_fm100"].to_numpy() if "gm_fm100" in f else np.full(len(f), 12.0)
        danger = _sigmoid((bi - 45.0) / 18.0)
        erc_d = _sigmoid((erc - 55.0) / 18.0)
        fuel_dry = _sigmoid((11.0 - fm) / 3.0)
        fw_idx = np.clip(0.45 * danger + 0.3 * erc_d + 0.25 * fuel_dry, 0, 1)
    elif "hdw" in f:
        fw_idx = _sigmoid((f["hdw"].to_numpy() - 120.0) / 40.0)
    else:
        fw_idx = _sigmoid(3.0 * (dryness - 0.7))

    # Danger is NOT occurrence: ignition is required, so map to a conservative,
    # capped probability that fire affects the cell over the horizon.
    p_base = np.clip(0.5 * fw_idx + 0.15 * fw, 0.0, 0.6)
    return 1.0 - (1.0 - p_base) * (1.0 - np.clip(np.tanh(fire_obs), 0, 1))


def _p_heat(f: pd.DataFrame) -> np.ndarray:
    """Dangerous-heat probability from the heat index and wet-bulb temperature.

    NWS thresholds: heat index ~32 C extreme caution, ~39 C danger, ~51 C
    extreme danger. Wet-bulb > ~28 C is dangerous, > ~31 C approaches the
    physiological limit. We take the noisy-OR of the two danger signals.
    """
    if "heat_index_c" in f:
        hi = f["heat_index_c"].to_numpy()
        p = _sigmoid((hi - 38.0) / 4.0)
        if "wet_bulb_c" in f:
            wb = f["wet_bulb_c"].to_numpy()
            p_wb = _sigmoid((wb - 28.0) / 2.0)
            p = 1.0 - (1.0 - p) * (1.0 - p_wb)
        return np.asarray(p)
    if "temp_max" in f:
        return _sigmoid((f["temp_max"].to_numpy() - 36.0) / 3.0)
    return np.full(len(f), 0.01)


def _p_smoke(f: pd.DataFrame) -> np.ndarray:
    """Wildfire-smoke air-quality hazard from downwind smoke-exposure potential.

    `smoke_potential` is a plume-transport proxy (fire radiative power carried
    downwind toward the cell). Mapped to a conservative probability of unhealthy
    smoke/air quality. Relative without observed PM2.5 calibration; observed
    AirNow/PurpleAir AQI can refine it later.
    """
    if "smoke_potential" in f:
        sp = np.clip(f["smoke_potential"].to_numpy(), 0, None)
        return np.clip(np.tanh(sp / 300.0), 1e-4, 0.95)
    return np.full(len(f), 1e-4)


PREDICTORS: dict[str, Predictor] = {
    "earthquake": _p_earthquake,
    "landslide": _p_landslide,
    "flood": _p_flood,
    "wildfire": _p_wildfire,
    "heat": _p_heat,
    "smoke": _p_smoke,
}


def base_probabilities(features: pd.DataFrame, use_trained: bool = True) -> pd.DataFrame:
    """Return a (cell x hazard) table of base initiation probabilities.

    For each hazard, a trained model in models_store/ (if present and
    use_trained) overrides the calibrated-sigmoid heuristic. Set use_trained
    False when a required trained feature is unavailable (e.g. the GRIDMET
    conditions map has no volumetric soil moisture for the trained flood model).
    """
    from .trained import load_trained

    out = pd.DataFrame({"cell_id": features["cell_id"].values})
    for hz, fn in PREDICTORS.items():
        trained = load_trained(hz) if use_trained else None
        probs = trained.predict(features) if trained is not None else fn(features)
        out[hz] = np.clip(probs, 1e-4, 0.999)
    return out
