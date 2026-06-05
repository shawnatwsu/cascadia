"""Unit tests for per-hazard predictor sanity (offline, deterministic)."""
import numpy as np
import pandas as pd

from cascadia.models.predictors import _p_earthquake, _p_heat, _p_landslide
from cascadia.sources.elevation import slope_factor


def test_slope_factor_endpoints():
    assert float(slope_factor(np.array([0.0]))[0]) == 0.0
    assert float(slope_factor(np.array([30.0]))[0]) == 1.0
    assert 0.3 < float(slope_factor(np.array([10.0]))[0]) < 0.5


def test_landslide_flat_lot_is_low():
    f = pd.DataFrame([{"precip_total_mm": 100.0, "soil_moist_peak": 0.45,
                       "ls_susceptibility": 0.9, "slope_deg": 1.0}])
    # flat ground is stable even with high susceptibility + heavy rain
    assert float(_p_landslide(f)[0]) < 0.05


def test_landslide_steep_higher_than_flat():
    base = dict(precip_total_mm=100.0, soil_moist_peak=0.45, ls_susceptibility=0.9)
    flat = float(_p_landslide(pd.DataFrame([{**base, "slope_deg": 1.0}]))[0])
    steep = float(_p_landslide(pd.DataFrame([{**base, "slope_deg": 25.0}]))[0])
    assert steep > flat


def test_earthquake_prob_in_range():
    f = pd.DataFrame([{"eq_base_prob": 0.0002, "quake_mag": 0.0}])
    p = float(_p_earthquake(f)[0])
    assert 0.0 <= p < 0.01


def test_heat_monotonic_in_index():
    cool = float(_p_heat(pd.DataFrame([{"heat_index_c": 20.0}]))[0])
    hot = float(_p_heat(pd.DataFrame([{"heat_index_c": 45.0}]))[0])
    assert hot > cool


def test_probabilities_bounded():
    f = pd.DataFrame([{"precip_total_mm": 0.0, "soil_moist_peak": 0.1,
                       "ls_susceptibility": 0.5, "slope_deg": 20.0}])
    assert 0.0 <= float(_p_landslide(f)[0]) <= 1.0
