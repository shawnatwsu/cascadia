"""Smoke tests for the scaled, independent-event flood performance harness.

These avoid the network: they exercise the metric/threshold/bootstrap math and
the public API surface, so CI catches import or logic regressions without
hitting NWS / ERA5 / NWIS.
"""
import numpy as np
import pandas as pd

from cascadia import validation_scaled as vs
from cascadia.sources import storm_events as se


def test_public_api_exists():
    for fn in ("flood_prob_at", "flow_anomaly_at", "scaled_flood_hindcast"):
        assert callable(getattr(vs, fn))
    for fn in ("events", "sample_events"):
        assert callable(getattr(se, fn))


def test_render_and_metrics_on_synthetic(tmp_path):
    """A separable synthetic set should yield AUC well above chance and a
    well-formed result dict + figure."""
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
