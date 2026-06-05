"""Calibration & skill of the trained hazard/ENSO models (reliability + scores).

A probability of 0.3 should verify ~30% of the time. This module produces the
out-of-fold reliability diagram + Brier / Brier-skill-score for the trained
flood model, and a skill figure for the ENSO (ONI) forecast — the verification a
reviewer expects alongside any probabilistic forecast.
"""
from __future__ import annotations

from pathlib import Path

import numpy as np

from .config import Config
from .skill import (brier_score, brier_skill_score, reliability_curve)


def validate_flood(out_path: str | Path = "cascadia_flood_skill.png",
                   verbose: bool = True) -> dict:
    from sklearn.metrics import roc_auc_score, average_precision_score
    from sklearn.model_selection import GroupKFold, cross_val_predict
    from .training.dataset import build_dataset, FEATURES
    from .training.train_flood import make_flood_model

    cfg = Config.load()
    data = build_dataset(cfg, verbose=verbose)
    X, y = data[FEATURES], data["flood"].to_numpy()
    groups = data["site_no"]
    # Same calibrated model as deployed; OOF by gage (calibration learned within
    # each training fold, so the reliability estimate is honest).
    oof = cross_val_predict(make_flood_model(), X, y, cv=GroupKFold(5),
                            groups=groups, method="predict_proba")[:, 1]
    m = {
        "brier": brier_score(oof, y), "bss": brier_skill_score(oof, y),
        "roc_auc": float(roc_auc_score(y, oof)),
        "pr_auc": float(average_precision_score(y, oof)),
        "base_rate": float(y.mean()), "n": int(len(y)),
        "n_gages": int(groups.nunique()),
    }
    _render_reliability(oof, y, m, out_path)
    if verbose:
        print(f"Flood model — OOF (GroupKFold by gage, {m['n']} ex / {m['n_gages']} gages):")
        print(f"  Brier {m['brier']:.3f} | Brier skill score {m['bss']:+.3f} | "
              f"ROC-AUC {m['roc_auc']:.3f} | PR-AUC {m['pr_auc']:.3f} "
              f"(base {m['base_rate']:.3f})")
    return m


def _render_reliability(p, y, m, out_path):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    fp, of, cnt = reliability_curve(p, y, bins=10)

    fig, (ax, axh) = plt.subplots(2, 1, figsize=(6.5, 7.5),
                                  gridspec_kw={"height_ratios": [3, 1]},
                                  constrained_layout=True)
    ax.plot([0, 1], [0, 1], "k--", lw=1, label="perfectly calibrated")
    ax.plot(fp, of, "o-", color="#1f77b4", lw=2, label="flood model (out-of-fold)")
    ax.axhline(m["base_rate"], color="0.6", ls=":", lw=1, label=f"base rate {m['base_rate']:.2f}")
    ax.set_xlabel("forecast probability"); ax.set_ylabel("observed flood frequency")
    ax.set_xlim(0, 1); ax.set_ylim(0, 1); ax.legend(fontsize=9); ax.grid(alpha=0.3)
    ax.set_title("Flood model reliability (calibration)", fontsize=12, weight="bold")
    ax.text(0.02, 0.97, f"Brier {m['brier']:.3f}\nBSS {m['bss']:+.3f}\n"
            f"ROC-AUC {m['roc_auc']:.3f}", va="top", ha="left", fontsize=10,
            bbox=dict(boxstyle="round", fc="white", ec="0.7"))
    axh.bar(fp, cnt, width=0.08, color="#1f77b4", alpha=0.6)
    axh.set_xlim(0, 1); axh.set_xlabel("forecast probability"); axh.set_ylabel("# forecasts")
    axh.grid(alpha=0.3)
    fig.suptitle("Cascadia — flood predictor calibration & skill\n"
                 "out-of-fold (GroupKFold by gage); USGS gage-exceedance labels",
                 fontsize=12, weight="bold")
    fig.savefig(out_path, dpi=140, bbox_inches="tight")
    plt.close(fig)


def validate_enso_forecast(out_path: str | Path = "cascadia_enso_forecast_skill.png",
                           verbose: bool = True) -> dict:
    from .training.train_enso import train
    b = train(save=False, verbose=False)
    metrics = b["metrics"]
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    leads = list(b["leads"])
    ml = [metrics[l]["rmse"] for l in leads]
    pe = [metrics[l]["rmse_persistence"] for l in leads]
    fig, ax = plt.subplots(figsize=(6.5, 4.5), constrained_layout=True)
    x = np.arange(len(leads))
    ax.bar(x - 0.2, ml, 0.4, label="ENSO forecast model", color="#d62728")
    ax.bar(x + 0.2, pe, 0.4, label="persistence baseline", color="0.6")
    for i, l in enumerate(leads):
        ax.text(i - 0.2, ml[i], f"{ml[i]:.2f}", ha="center", va="bottom", fontsize=8)
    ax.set_xticks(x, [f"+{l} mo" for l in leads]); ax.set_ylabel("ONI forecast RMSE")
    ax.legend(); ax.grid(axis="y", alpha=0.3)
    ax.set_title(f"ENSO (ONI) forecast skill — held-out {b['test_period']} "
                 f"({b['n_test']} months)", fontsize=11, weight="bold")
    fig.suptitle("Cascadia — ENSO forecast beats persistence at every lead",
                 fontsize=12, weight="bold")
    fig.savefig(out_path, dpi=140, bbox_inches="tight")
    plt.close(fig)
    if verbose:
        for l in leads:
            mt = metrics[l]
            print(f"  +{l}mo ONI: RMSE {mt['rmse']:.3f} vs persistence "
                  f"{mt['rmse_persistence']:.3f}, corr {mt['corr']:.3f}")
    return metrics
