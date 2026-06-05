"""Learn to FORECAST ENSO (ONI) 1-3 months ahead from its recent trajectory.

ENSO is the dominant driver of US seasonal hazard patterns, so forecasting it a
few months out extends the lead time of the seasonal hazard outlook. This is a
genuine ML model trained on 1950-present ONI: features are the recent ONI
trajectory plus seasonality (ENSO has a spring predictability barrier); targets
are ONI at +1/+2/+3 months. We score it honestly against the persistence
baseline (assume ONI stays put) on a held-out recent period.

Usage:  python -m cascadia.training.train_enso
        python -m cascadia.training.train_enso --info     # show forecast + skill
"""
from __future__ import annotations

import argparse
from datetime import datetime, timezone
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.metrics import mean_squared_error

from ..config import Config
from ..sources.enso import fetch_oni

MODEL_DIR = Path(__file__).resolve().parent.parent / "models_store"
MODEL_PATH = MODEL_DIR / "enso_model.joblib"
LEADS = (1, 2, 3)
N_LAGS = 6


def _features(df: pd.DataFrame) -> pd.DataFrame:
    f = pd.DataFrame(index=df.index)
    for k in range(N_LAGS):
        f[f"oni_lag{k}"] = df["oni"].shift(k)
    f["trend"] = df["oni"] - df["oni"].shift(3)
    m = df["date"].dt.month
    f["month_sin"] = np.sin(2 * np.pi * m / 12)
    f["month_cos"] = np.cos(2 * np.pi * m / 12)
    return f


def train(split_year: int = 2012, save: bool = True, verbose: bool = True) -> dict:
    df = fetch_oni(Config.load())
    X = _features(df)
    feat_cols = list(X.columns)
    rows = []
    for lead in LEADS:
        rows.append(df["oni"].shift(-lead).rename(f"y{lead}"))
    Y = pd.concat(rows, axis=1)
    data = pd.concat([df[["date", "oni"]], X, Y], axis=1).dropna().reset_index(drop=True)

    is_train = data["date"].dt.year <= split_year
    Xtr, Xte = data.loc[is_train, feat_cols], data.loc[~is_train, feat_cols]
    models, metrics = {}, {}
    for lead in LEADS:
        yt = data.loc[is_train, f"y{lead}"]
        yv = data.loc[~is_train, f"y{lead}"]
        m = GradientBoostingRegressor(n_estimators=300, max_depth=3,
                                      learning_rate=0.05, subsample=0.9,
                                      random_state=0).fit(Xtr, yt)
        pred = m.predict(Xte)
        persist = data.loc[~is_train, "oni"]   # baseline: ONI stays put
        metrics[lead] = {
            "rmse": float(np.sqrt(mean_squared_error(yv, pred))),
            "rmse_persistence": float(np.sqrt(mean_squared_error(yv, persist))),
            "corr": float(np.corrcoef(yv, pred)[0, 1]),
            "corr_persistence": float(np.corrcoef(yv, persist)[0, 1]),
        }
        models[lead] = m

    bundle = {"models": models, "features": feat_cols, "leads": LEADS,
              "metrics": metrics, "n_lags": N_LAGS,
              "trained_utc": datetime.now(timezone.utc).isoformat(),
              "test_period": f">{split_year}", "n_test": int((~is_train).sum())}
    if save:
        MODEL_DIR.mkdir(parents=True, exist_ok=True)
        joblib.dump(bundle, MODEL_PATH)

    if verbose:
        print(f"=== ENSO (ONI) forecast skill — test {bundle['test_period']} "
              f"({bundle['n_test']} months) ===")
        print(f"{'lead':>5} {'ML RMSE':>9} {'persist':>9}  {'ML corr':>8} {'persist':>8}")
        for lead in LEADS:
            mt = metrics[lead]
            print(f"{lead:>4}mo {mt['rmse']:>9.3f} {mt['rmse_persistence']:>9.3f}  "
                  f"{mt['corr']:>8.3f} {mt['corr_persistence']:>8.3f}")
        if save:
            print(f"\nsaved -> {MODEL_PATH}")
    return bundle


def forecast(config: Config | None = None) -> dict:
    """Predict ONI at +1/+2/+3 months from the latest observed trajectory."""
    if not MODEL_PATH.exists():
        return {}
    bundle = joblib.load(MODEL_PATH)
    df = fetch_oni(config or Config.load())
    x = _features(df).iloc[[-1]][bundle["features"]]
    return {lead: float(bundle["models"][lead].predict(x)[0]) for lead in bundle["leads"]}


def show_info() -> None:
    if not MODEL_PATH.exists():
        print("No ENSO model. Run: python -m cascadia.training.train_enso")
        return
    b = joblib.load(MODEL_PATH)
    print(f"ENSO forecast model — trained {b['trained_utc']}, test {b['test_period']}")
    for lead in b["leads"]:
        mt = b["metrics"][lead]
        skill = 1 - mt["rmse"] / mt["rmse_persistence"]
        print(f"  +{lead}mo: RMSE {mt['rmse']:.3f} vs persistence {mt['rmse_persistence']:.3f} "
              f"({skill:+.0%} skill), corr {mt['corr']:.3f}")
    fc = forecast()
    if fc:
        from ..sources.enso import _classify
        nxt = ", ".join(f"+{k}mo {v:+.2f} ({_classify(v)})" for k, v in fc.items())
        print(f"  Forecast ONI: {nxt}")


def main() -> None:
    ap = argparse.ArgumentParser(description="Train the ENSO (ONI) forecast model")
    ap.add_argument("--info", action="store_true")
    ap.add_argument("--split-year", type=int, default=2012)
    args = ap.parse_args()
    if args.info:
        show_info()
    else:
        train(split_year=args.split_year)


if __name__ == "__main__":
    main()
