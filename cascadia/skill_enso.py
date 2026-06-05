"""Validate the ENSO -> regional-climate teleconnection against observations.

The seasonal outlook claims El Nino/La Nina shift each region's odds. This module
TESTS that against 75+ years of observed regional climate (NCEI), the way a
reviewer would:

  * correlation between seasonal ONI and observed regional precip/temperature
    anomalies, per NCA5 region and season — the teleconnection strength;
  * out-of-sample tercile skill (RPSS): leave-one-year-out ENSO-phase composite
    forecasts of below/near/above-normal, scored vs equal-odds climatology.

Output: a skill matrix figure (region x season) plus a printed summary, so the
seasonal outlook's claims carry honest, quantified skill — or honestly don't.
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

from .config import Config
from .skill import anomaly_correlation, rpss, terciles

SEASONS = {"DJF": (12, 1, 2), "MAM": (3, 4, 5), "JJA": (6, 7, 8), "SON": (9, 10, 11)}
REGION_ORDER = ["northwest", "southwest", "northern_great_plains",
                "southern_great_plains", "midwest", "southeast", "northeast"]
REGION_LABEL = {"northwest": "NW", "southwest": "SW",
                "northern_great_plains": "N.Plains", "southern_great_plains": "S.Plains",
                "midwest": "Midwest", "southeast": "SE", "northeast": "NE"}


def _phase(oni: np.ndarray) -> np.ndarray:
    return np.where(oni >= 0.5, 1, np.where(oni <= -0.5, -1, 0))


def _loo_composite_rpss(oni: np.ndarray, cats: np.ndarray) -> float:
    """Leave-one-out ENSO-phase composite tercile forecast skill (RPSS)."""
    ph = _phase(oni)
    n = len(oni)
    probs = np.full((n, 3), 1 / 3)
    for i in range(n):
        same = (ph == ph[i]) & (np.arange(n) != i)
        if same.sum() >= 5:
            freq = np.bincount(cats[same], minlength=3).astype(float)
            probs[i] = (freq + 0.5) / (freq.sum() + 1.5)   # Laplace-smoothed
    return rpss(probs, cats)


def validate_enso(out_path: str | Path = "cascadia_enso_skill.png",
                  verbose: bool = True):
    from .sources.enso import fetch_oni
    from .sources.climate_obs import regional_monthly, seasonal_anomaly

    cfg = Config.load()
    oni_df = fetch_oni(cfg)
    pcp = regional_monthly("precip", cfg.cache_dir)
    tmp = regional_monthly("temp", cfg.cache_dir)
    log = (lambda *a: print(*a)) if verbose else (lambda *a: None)

    corr = {"precip": {}, "temp": {}}
    skill = {"precip": {}, "temp": {}}
    for season, months in SEASONS.items():
        oni_s = oni_df[oni_df["SEAS"] == season].set_index("YR")["oni"]
        for var, monthly in (("precip", pcp), ("temp", tmp)):
            anom = seasonal_anomaly(monthly, months)
            for region in REGION_ORDER:
                if region not in anom.columns:
                    continue
                d = pd.concat([anom[region].rename("a"), oni_s.rename("o")],
                              axis=1).dropna()
                if len(d) < 25:
                    continue
                corr[var][(region, season)] = anomaly_correlation(d["o"], d["a"])
                skill[var][(region, season)] = _loo_composite_rpss(
                    d["o"].to_numpy(), terciles(d["a"].to_numpy()))

    _render(corr, skill, out_path)
    # printed summary: strongest teleconnections
    flat = sorted(((v, var, r, s) for var in corr for (r, s), v in corr[var].items()),
                  key=lambda t: -abs(t[0]))
    log("Strongest ENSO->regional correlations (|r|):")
    for v, var, r, s in flat[:6]:
        sk = skill[var].get((r, s), float("nan"))
        log(f"  {var:6} {REGION_LABEL[r]:8} {s}: r={v:+.2f}  tercile RPSS={sk:+.2f}")
    pos = [s for var in skill for s in skill[var].values() if s > 0]
    log(f"Out-of-sample tercile skill (RPSS>0) in {len(pos)}/"
        f"{sum(len(skill[v]) for v in skill)} region-seasons.")
    return corr, skill, out_path


def _render(corr, skill, out_path):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    seasons = list(SEASONS)
    fig, axes = plt.subplots(2, 2, figsize=(12, 8.5), constrained_layout=True)
    panels = [("precip", corr, "Precip: ONI correlation", "RdBu", -0.8, 0.8, axes[0, 0]),
              ("temp", corr, "Temperature: ONI correlation", "RdBu_r", -0.8, 0.8, axes[0, 1]),
              ("precip", skill, "Precip: tercile skill (RPSS)", "PuOr", -0.3, 0.3, axes[1, 0]),
              ("temp", skill, "Temperature: tercile skill (RPSS)", "PuOr", -0.3, 0.3, axes[1, 1])]
    for var, src, title, cmap, vmin, vmax, ax in panels:
        M = np.full((len(REGION_ORDER), len(seasons)), np.nan)
        for ri, r in enumerate(REGION_ORDER):
            for si, s in enumerate(seasons):
                if (r, s) in src[var]:
                    M[ri, si] = src[var][(r, s)]
        im = ax.imshow(M, cmap=cmap, vmin=vmin, vmax=vmax, aspect="auto")
        ax.set_xticks(range(len(seasons)), seasons)
        ax.set_yticks(range(len(REGION_ORDER)), [REGION_LABEL[r] for r in REGION_ORDER])
        ax.set_title(title, fontsize=11, weight="bold")
        for ri in range(len(REGION_ORDER)):
            for si in range(len(seasons)):
                if np.isfinite(M[ri, si]):
                    ax.text(si, ri, f"{M[ri, si]:+.2f}", ha="center", va="center",
                            fontsize=8, color="0.1")
        fig.colorbar(im, ax=ax, fraction=0.046, pad=0.03)
    fig.suptitle("Cascadia — ENSO -> regional climate teleconnection skill\n"
                 "validated against NCEI observed climate, 1950-present (NCA5 regions)",
                 fontsize=13, weight="bold")
    fig.text(0.5, -0.02, "Correlation (top): how strongly seasonal ENSO (ONI) tracks each "
             "region's observed precip/temperature anomaly. Tercile skill RPSS (bottom): "
             "leave-one-year-out ENSO-phase composite forecast of below/near/above-normal "
             "vs equal-odds climatology (>0 = skill). Blank = season excluded for short record.",
             ha="center", va="top", fontsize=8.5, color="0.3", wrap=True)
    Path(out_path)
    fig.savefig(out_path, dpi=140, bbox_inches="tight")
    plt.close(fig)
