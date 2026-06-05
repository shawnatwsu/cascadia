"""Train + cross-validate the flood leaf predictor and save it.

Usage:
    python -m cascadia.training.train_flood            # default 2019-2023, 25 gages
    python -m cascadia.training.train_flood --years 2020 2023 --sites 30

Honest evaluation: probabilities are scored out-of-fold with GroupKFold by gage,
so a gage never appears in both train and test — no spatial leakage. We report
ROC-AUC, PR-AUC (vs. the base rate) and the Brier score, and compare against the
existing heuristic on the same out-of-fold examples.
"""
from __future__ import annotations

import argparse
from datetime import datetime, timezone
from pathlib import Path

import joblib
import numpy as np
from sklearn.calibration import CalibratedClassifierCV
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.metrics import (average_precision_score, brier_score_loss,
                             roc_auc_score)
from sklearn.model_selection import GroupKFold, cross_val_predict


def make_flood_model() -> CalibratedClassifierCV:
    """Gradient-boosting flood model wrapped in isotonic probability calibration.

    The balanced base model discriminates well but is over-confident; isotonic
    calibration (fit on an internal CV split) makes a predicted 0.6 mean ~0.6.
    """
    base = HistGradientBoostingClassifier(
        max_depth=3, learning_rate=0.08, max_iter=300,
        l2_regularization=1.0, class_weight="balanced", random_state=0)
    return CalibratedClassifierCV(base, method="isotonic", cv=3)

from ..config import Config
from .dataset import FEATURES, build_dataset

MODEL_DIR = Path(__file__).resolve().parent.parent / "models_store"
MODEL_PATH = MODEL_DIR / "flood_model.joblib"


def _heuristic_flood(X) -> np.ndarray:
    """The current calibrated-sigmoid flood predictor, for a fair baseline."""
    precip = X["precip_total_mm"].to_numpy()
    flow = X["flow_anomaly"].to_numpy()
    z = 0.04 * (precip - 50.0) + 2.5 * (flow - 0.5)
    return 1.0 / (1.0 + np.exp(-z))


def train(years: tuple[int, int] = (2019, 2023), sites: int = 25,
          sample_every: int = 3, save: bool = True, verbose: bool = True) -> dict:
    config = Config.load()
    start, end = f"{years[0]}-01-01", f"{years[1]}-12-31"
    data = build_dataset(config, start=start, end=end, max_sites=sites,
                         sample_every=sample_every, verbose=verbose)
    if data.empty or data["flood"].sum() < 10:
        raise RuntimeError("Not enough flood examples to train; widen years/sites.")

    X, y, groups = data[FEATURES], data["flood"].to_numpy(), data["site_no"]
    n_groups = groups.nunique()
    n_splits = min(5, n_groups)

    clf = make_flood_model()
    gkf = GroupKFold(n_splits=n_splits)
    oof = cross_val_predict(clf, X, y, cv=gkf, groups=groups,
                            method="predict_proba")[:, 1]

    base_rate = float(y.mean())
    metrics = {
        "roc_auc": float(roc_auc_score(y, oof)),
        "pr_auc": float(average_precision_score(y, oof)),
        "pr_auc_baseline": base_rate,
        "brier": float(brier_score_loss(y, oof)),
        "heuristic_roc_auc": float(roc_auc_score(y, _heuristic_flood(X))),
        "n_examples": int(len(y)),
        "n_gages": int(n_groups),
        "positive_rate": base_rate,
        "cv_folds": n_splits,
    }

    # Fit the final model on all data for deployment.
    clf.fit(X, y)
    bundle = {
        "model": clf, "features": FEATURES, "hazard": "flood",
        "metrics": metrics, "trained_utc": datetime.now(timezone.utc).isoformat(),
        "train_window": [start, end],
    }
    if save:
        MODEL_DIR.mkdir(parents=True, exist_ok=True)
        joblib.dump(bundle, MODEL_PATH)

    if verbose:
        print("\n=== Flood predictor — out-of-fold (GroupKFold by gage) ===")
        print(f"  examples           : {metrics['n_examples']} "
              f"({metrics['positive_rate']:.1%} positive) across {metrics['n_gages']} gages")
        print(f"  ROC-AUC            : {metrics['roc_auc']:.3f}  "
              f"(heuristic baseline {metrics['heuristic_roc_auc']:.3f})")
        print(f"  PR-AUC             : {metrics['pr_auc']:.3f}  "
              f"(no-skill baseline {metrics['pr_auc_baseline']:.3f})")
        print(f"  Brier score        : {metrics['brier']:.3f}  (lower is better)")
        if save:
            print(f"\n  saved -> {MODEL_PATH}")
    return bundle


def show_info() -> None:
    """Print the saved model card without retraining."""
    if not MODEL_PATH.exists():
        print(f"No trained model at {MODEL_PATH}. Run: python -m cascadia.training.train_flood")
        return
    bundle = joblib.load(MODEL_PATH)
    m = bundle["metrics"]
    print("=== Cascadia flood predictor — model card ===")
    print(f"  file        : {MODEL_PATH}")
    print(f"  trained     : {bundle['trained_utc']}")
    print(f"  train window: {bundle['train_window'][0]} .. {bundle['train_window'][1]}")
    print(f"  features    : {bundle['features']}")
    print(f"  examples    : {m['n_examples']} ({m['positive_rate']:.1%} positive), "
          f"{m['n_gages']} gages, {m['cv_folds']}-fold GroupKFold by gage")
    print(f"  ROC-AUC     : {m['roc_auc']:.3f}  (heuristic {m['heuristic_roc_auc']:.3f})")
    print(f"  PR-AUC      : {m['pr_auc']:.3f}  (no-skill {m['pr_auc_baseline']:.3f})")
    print(f"  Brier       : {m['brier']:.3f}")


def main() -> None:
    ap = argparse.ArgumentParser(description="Train the Cascadia flood predictor")
    ap.add_argument("--info", action="store_true", help="show saved model card, no training")
    ap.add_argument("--years", nargs=2, type=int, default=[2019, 2023])
    ap.add_argument("--sites", type=int, default=25)
    ap.add_argument("--sample-every", type=int, default=3)
    args = ap.parse_args()
    if args.info:
        show_info()
        return
    train(years=tuple(args.years), sites=args.sites, sample_every=args.sample_every)


if __name__ == "__main__":
    main()
