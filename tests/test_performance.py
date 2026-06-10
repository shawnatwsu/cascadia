"""Smoke tests for the scaled, independent-event flood performance harness.

These avoid the network: they exercise the metric/threshold/bootstrap math and
the public API surface, so CI catches import or logic regressions without
hitting NWS / ERA5 / NWIS.
"""
import numpy as np
import pandas as pd
import pytest

from cascadia import validation_scaled as vs
from cascadia import validation_fire as vf
from cascadia.sources import storm_events as se


def test_public_api_exists():
    for fn in ("flood_prob_at", "flow_anomaly_at", "scaled_flood_hindcast",
               "lead_time_curve"):
        assert callable(getattr(vs, fn))
    for fn in ("fire_danger_at", "fire_event_hindcast"):
        assert callable(getattr(vf, fn))
    for fn in ("events", "sample_events"):
        assert callable(getattr(se, fn))


def test_window_feats_and_flow_asof():
    idx = pd.date_range("2021-11-01", periods=20, freq="D")
    daily = pd.DataFrame({"precip_day": np.arange(20.0),
                          "soil_day": np.linspace(0.2, 0.4, 20)}, index=idx)
    issue = pd.Timestamp("2021-11-05")
    feats = vs._window_feats(daily, issue, horizon=7)
    assert feats is not None
    # forward window is strictly AFTER issue -> excludes the issue day's precip
    expected = daily.loc[(daily.index > issue) &
                         (daily.index <= issue + pd.Timedelta(days=7))]
    assert feats[0] == float(expected["precip_day"].sum())
    # empty / no-gage discharge -> neutral fallback, no crash
    assert vs._flow_asof(None, issue) == vs.NEUTRAL_FLOW


def test_control_date_modes():
    rng = np.random.default_rng(0)
    lo, hi = pd.Timestamp("2000-01-01"), pd.Timestamp("2025-01-01")
    ev = pd.Timestamp("2020-08-15")
    # same_season: within ~2 weeks of the same day-of-year, ~1 year away
    for _ in range(20):
        c = vs.control_date(ev, "same_season", rng, lo, hi)
        assert lo <= c <= hi
        doy_gap = min(abs(c.dayofyear - ev.dayofyear),
                      365 - abs(c.dayofyear - ev.dayofyear))
        assert doy_gap <= 20, f"same-season control drifted seasons: {c}"
        assert abs((c - ev).days) >= 300, "same-season control too close to event"
    # shifted: 60–300 days away, can cross seasons
    for _ in range(20):
        c = vs.control_date(ev, "shifted", rng, lo, hi)
        assert 60 <= abs((c - ev).days) <= 300


def test_fire_danger_formula_monotonic(monkeypatch):
    """Drier/hotter GRIDMET inputs must yield higher danger, capped at 0.6."""
    pytest.importorskip("xarray")  # cascadia.sources.gridmet imports xarray
    from cascadia.config import Config
    cfg = Config.load()

    def fake_series(dangerous):
        bi = 90.0 if dangerous else 5.0
        erc = 90.0 if dangerous else 5.0
        fm = 3.0 if dangerous else 25.0
        return pd.DataFrame({"burning_index": [bi], "erc": [erc], "fm100": [fm]})

    import cascadia.sources.gridmet as gm
    monkeypatch.setattr(gm, "point_series", lambda *a, **k: fake_series(True))
    hi = vf.fire_danger_at(40.0, -120.0, "2020-08-15", cfg)
    monkeypatch.setattr(gm, "point_series", lambda *a, **k: fake_series(False))
    lo = vf.fire_danger_at(40.0, -120.0, "2020-08-15", cfg)
    assert 0.0 <= lo < hi <= 0.6


def test_render_and_metrics_on_synthetic(tmp_path):
    """A separable synthetic set should yield AUC well above chance and a
    well-formed result dict + figure."""
    pytest.importorskip("matplotlib")  # _render draws a figure
    from sklearn.metrics import roc_auc_score, roc_curve
    rng = np.random.default_rng(0)
    pos = np.clip(rng.normal(0.6, 0.15, 80), 0, 1)
    neg = np.clip(rng.normal(0.2, 0.15, 80), 0, 1)
    df = pd.DataFrame({
        "prob": np.concatenate([pos, neg]),
        "label": [1] * 80 + [0] * 80,
        "type": ["Flood"] * 80 + ["Flash Flood"] * 80,
    })
    y, p = df["label"].to_numpy(), df["prob"].to_numpy()
    auc = roc_auc_score(y, p)
    assert auc > 0.7
    fpr, tpr, _ = roc_curve(y, p)
    res = {"n_events": 80, "n_nonevents": 80, "roc_auc": float(auc),
           "auc_ci": (auc - 0.05, auc + 0.05), "threshold": 0.4,
           "hit_rate": 0.7, "false_alarm_rate": 0.2,
           "by_type": {"Flood": {"n": 80, "roc_auc": float(auc)}}}
    out = tmp_path / "perf.png"
    vs._render(df, fpr, tpr, res, out)
    assert out.exists() and out.stat().st_size > 0


def test_flood_prob_handles_empty_weather(monkeypatch):
    """No weather -> NaN, not a crash."""
    monkeypatch.setattr(vs, "_GAGE_CACHE", {})
    from cascadia.training import dataset
    monkeypatch.setattr(dataset, "_fetch_weather_daily",
                        lambda *a, **k: pd.DataFrame())
    from cascadia.config import Config
    val = vs.flood_prob_at(46.6, -122.9, "2021-11-15", Config.load(), use_flow=False)
    assert np.isnan(val)
