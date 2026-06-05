"""Forecast verification metrics — the language of forecast skill.

Probabilistic forecasts are only credible with measured skill. This module
provides the standard scores a reviewer expects:
  * Brier score + Brier skill score (binary probabilistic),
  * Ranked Probability Score (RPS) + skill score (RPSS) for ordered categories
    (e.g. below/near/above-normal terciles), vs the climatological 1/3,1/3,1/3,
  * reliability-curve points (forecast prob vs observed frequency),
  * anomaly correlation.
"""
from __future__ import annotations

import numpy as np


def brier_score(p: np.ndarray, y: np.ndarray) -> float:
    p, y = np.asarray(p, float), np.asarray(y, float)
    return float(np.mean((p - y) ** 2))


def brier_skill_score(p: np.ndarray, y: np.ndarray) -> float:
    """BSS vs the base-rate climatology (1 = perfect, 0 = no better than base)."""
    y = np.asarray(y, float)
    base = np.full_like(y, y.mean())
    bs, bs_ref = brier_score(p, y), brier_score(base, y)
    return float(1 - bs / bs_ref) if bs_ref > 0 else 0.0


def rps(prob: np.ndarray, obs_cat: np.ndarray, n_cat: int = 3) -> float:
    """Ranked Probability Score for ordered categories (lower is better).

    prob: (n, n_cat) forecast probabilities; obs_cat: (n,) observed category idx.
    """
    prob = np.asarray(prob, float)
    onehot = np.eye(n_cat)[np.asarray(obs_cat, int)]
    cum_p = np.cumsum(prob, axis=1)
    cum_o = np.cumsum(onehot, axis=1)
    return float(np.mean(np.sum((cum_p - cum_o) ** 2, axis=1)))


def rpss(prob: np.ndarray, obs_cat: np.ndarray, n_cat: int = 3) -> float:
    """RPSS vs equal-odds climatology (1 = perfect, 0 = no better, <0 = worse)."""
    clim = np.full((len(obs_cat), n_cat), 1.0 / n_cat)
    r, r_clim = rps(prob, obs_cat, n_cat), rps(clim, obs_cat, n_cat)
    return float(1 - r / r_clim) if r_clim > 0 else 0.0


def reliability_curve(p: np.ndarray, y: np.ndarray, bins: int = 10):
    """Return (mean forecast prob, observed frequency, count) per probability bin."""
    p, y = np.asarray(p, float), np.asarray(y, float)
    edges = np.linspace(0, 1, bins + 1)
    idx = np.clip(np.digitize(p, edges) - 1, 0, bins - 1)
    fp, of, cnt = [], [], []
    for b in range(bins):
        m = idx == b
        if m.any():
            fp.append(p[m].mean()); of.append(y[m].mean()); cnt.append(int(m.sum()))
    return np.array(fp), np.array(of), np.array(cnt)


def anomaly_correlation(pred: np.ndarray, obs: np.ndarray) -> float:
    pred, obs = np.asarray(pred, float), np.asarray(obs, float)
    if len(pred) < 3 or pred.std() == 0 or obs.std() == 0:
        return float("nan")
    return float(np.corrcoef(pred, obs)[0, 1])


def terciles(x: np.ndarray) -> np.ndarray:
    """Classify values into 0=below, 1=near, 2=above (by 33/67 percentiles)."""
    x = np.asarray(x, float)
    lo, hi = np.nanpercentile(x, [100 / 3, 200 / 3])
    return np.where(x <= lo, 0, np.where(x >= hi, 2, 1))
