"""Load trained per-hazard leaf models and expose them via the predictor API.

If a trained model exists in `models_store/`, it transparently replaces that
hazard's calibrated-sigmoid heuristic inside `base_probabilities`. If not, the
heuristic is used — so the engine always runs, trained or not.
"""
from __future__ import annotations

from functools import lru_cache
from pathlib import Path

import numpy as np
import pandas as pd

MODEL_DIR = Path(__file__).resolve().parent.parent / "models_store"


class TrainedPredictor:
    """Wraps a saved sklearn model behind the predictor interface."""

    def __init__(self, bundle: dict):
        self.model = bundle["model"]
        self.features = bundle["features"]
        self.hazard = bundle["hazard"]
        self.metrics = bundle.get("metrics", {})
        self.trained_utc = bundle.get("trained_utc")

    def predict(self, features: pd.DataFrame) -> np.ndarray:
        X = features[self.features]
        return self.model.predict_proba(X)[:, 1]

    def __repr__(self) -> str:
        auc = self.metrics.get("roc_auc")
        return (f"<TrainedPredictor {self.hazard} "
                f"ROC-AUC={auc:.3f} trained={self.trained_utc}>"
                if auc else f"<TrainedPredictor {self.hazard}>")


@lru_cache(maxsize=None)
def load_trained(hazard: str) -> TrainedPredictor | None:
    """Return the trained predictor for a hazard, or None if not trained yet."""
    path = MODEL_DIR / f"{hazard}_model.joblib"
    if not path.exists():
        return None
    try:
        import joblib
        return TrainedPredictor(joblib.load(path))
    except Exception:
        return None
