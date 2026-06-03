"""Train + cross-validate the wildfire leaf predictor and save it.

Needs a free FIRMS_MAP_KEY (labels). Usage:
    $env:FIRMS_MAP_KEY="yourkey"
    python -m cascadia.training.train_fire
    python -m cascadia.training.train_fire --info

Out-of-fold scoring uses GroupKFold by sample point (no point in both train and
test). Reports ROC-AUC, PR-AUC (vs. base rate) and Brier.
"""
from __future__ import annotations

import argparse
from datetime import datetime, timezone
from pathlib import Path

import joblib
import numpy as np
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.metrics import (average_precision_score, brier_score_loss,
                             roc_auc_score)
from sklearn.model_selection import GroupKFold, cross_val_predict

from ..config import Config
from .dataset_fire import FIRE_FEATURES, build_fire_dataset

MODEL_DIR = Path(__file__).resolve().parent.parent / "models_store"
MODEL_PATH = MODEL_DIR / "wildfire_model.joblib"


def train(years: tuple[int, int] = (2020, 2023), sample_every: int = 7,
          point_step: float = 1.0, save: bool = True, verbose: bool = True) -> dict:
    config = Config.load()
    data = build_fire_dataset(config, years=years, sample_every=sample_every,
                              point_step=point_step, verbose=verbose)
    if data.empty or data["fire"].sum() < 10:
        raise RuntimeError(
            f"Not enough fire examples to train (positives={int(data['fire'].sum()) if not data.empty else 0}). "
            "Widen years/region or check the FIRMS key.")

    X, y = data[FIRE_FEATURES], data["fire"].to_numpy()
    groups = data["lat"].astype(str) + "," + data["lon"].astype(str)
    n_splits = min(5, groups.nunique())

    clf = HistGradientBoostingClassifier(
        max_depth=3, learning_rate=0.08, max_iter=300,
        l2_regularization=1.0, class_weight="balanced", random_state=0)
    gkf = GroupKFold(n_splits=n_splits)
    oof = cross_val_predict(clf, X, y, cv=gkf, groups=groups,
                            method="predict_proba")[:, 1]

    base_rate = float(y.mean())
    metrics = {
        "roc_auc": float(roc_auc_score(y, oof)),
        "pr_auc": float(average_precision_score(y, oof)),
        "pr_auc_baseline": base_rate,
        "brier": float(brier_score_loss(y, oof)),
        "n_examples": int(len(y)), "n_points": int(groups.nunique()),
        "positive_rate": base_rate, "cv_folds": n_splits,
    }

    clf.fit(X, y)
    bundle = {
        "model": clf, "features": FIRE_FEATURES, "hazard": "wildfire",
        "metrics": metrics, "trained_utc": datetime.now(timezone.utc).isoformat(),
        "train_window": [f"{years[0]}-01-01", f"{years[1]}-12-31"],
    }
    if save:
        MODEL_DIR.mkdir(parents=True, exist_ok=True)
        joblib.dump(bundle, MODEL_PATH)

    if verbose:
        print("\n=== Wildfire predictor — out-of-fold (GroupKFold by point) ===")
        print(f"  examples : {metrics['n_examples']} ({base_rate:.1%} positive), "
              f"{metrics['n_points']} points")
        print(f"  ROC-AUC  : {metrics['roc_auc']:.3f}")
        print(f"  PR-AUC   : {metrics['pr_auc']:.3f}  (no-skill {base_rate:.3f})")
        print(f"  Brier    : {metrics['brier']:.3f}")
        if save:
            print(f"\n  saved -> {MODEL_PATH}")
    return bundle


def show_info() -> None:
    if not MODEL_PATH.exists():
        print(f"No wildfire model at {MODEL_PATH}. Run: python -m cascadia.training.train_fire")
        return
    b = joblib.load(MODEL_PATH); m = b["metrics"]
    print("=== Cascadia wildfire predictor — model card ===")
    print(f"  trained : {b['trained_utc']}  window {b['train_window'][0]}..{b['train_window'][1]}")
    print(f"  features: {b['features']}")
    print(f"  examples: {m['n_examples']} ({m['positive_rate']:.1%} positive), {m['n_points']} points")
    print(f"  ROC-AUC : {m['roc_auc']:.3f} | PR-AUC {m['pr_auc']:.3f} (no-skill {m['pr_auc_baseline']:.3f}) | Brier {m['brier']:.3f}")


def main() -> None:
    ap = argparse.ArgumentParser(description="Train the Cascadia wildfire predictor")
    ap.add_argument("--info", action="store_true")
    ap.add_argument("--years", nargs=2, type=int, default=[2019, 2023])
    ap.add_argument("--sample-every", type=int, default=5)
    args = ap.parse_args()
    if args.info:
        show_info()
        return
    train(years=tuple(args.years), sample_every=args.sample_every)


if __name__ == "__main__":
    main()
