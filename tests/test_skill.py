"""Unit tests for the forecast-verification metrics (offline, deterministic)."""
import numpy as np

from cascadia.skill import (anomaly_correlation, brier_score, brier_skill_score,
                            reliability_curve, rps, rpss, terciles)


def test_brier_perfect():
    assert brier_score([1.0, 0.0, 1.0], [1, 0, 1]) == 0.0


def test_brier_half():
    assert abs(brier_score([0.5, 0.5], [1, 0]) - 0.25) < 1e-9


def test_bss_perfect():
    y = [1, 0, 1, 0, 1, 0]
    assert brier_skill_score(y, y) == 1.0


def test_bss_baserate_is_zero():
    y = np.array([1, 0, 1, 0])
    assert abs(brier_skill_score(np.full(4, y.mean()), y)) < 1e-9


def test_rpss_perfect_is_one():
    prob = np.array([[1, 0, 0], [0, 1, 0], [0, 0, 1]], float)
    assert abs(rpss(prob, np.array([0, 1, 2])) - 1.0) < 1e-9


def test_rpss_climatology_is_zero():
    prob = np.full((6, 3), 1 / 3)
    assert abs(rpss(prob, np.array([0, 1, 2, 0, 1, 2]))) < 1e-9


def test_rps_nonnegative():
    prob = np.array([[0.2, 0.3, 0.5], [0.6, 0.3, 0.1]])
    assert rps(prob, np.array([2, 0])) >= 0.0


def test_terciles_split():
    t = terciles(np.arange(9))
    assert t[0] == 0 and t[-1] == 2 and set(t) == {0, 1, 2}


def test_anomaly_correlation_perfect():
    a = np.array([1.0, 2.0, 3.0, 4.0])
    assert abs(anomaly_correlation(a, a) - 1.0) < 1e-9


def test_reliability_curve_monotone_for_perfect():
    p = np.array([0.05, 0.15, 0.95, 0.85])
    y = np.array([0, 0, 1, 1])
    fp, of, cnt = reliability_curve(p, y, bins=10)
    assert (np.diff(of) >= 0).all() and cnt.sum() == 4
