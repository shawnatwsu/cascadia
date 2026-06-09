"""Independent-event test for the WILDFIRE leaf, mirroring the flood version.

Positives = real fire location-days from NASA FIRMS satellite detections (NWS
Storm Events records wildfires by forecast zone with NO point coordinates, so it
can't be used here). Negatives = the same locations on shifted, calm dates. The
score is the product's own wildfire danger: GRIDMET NFDRS Burning Index + Energy
Release Component + 100-hr fuel moisture, mapped through the EXACT formula the
live engine uses (predictors._p_wildfire's GRIDMET branch). Reports ROC-AUC + a
hit/false-alarm operating point — the same defensible framing as the flood claim.

FIRMS (satellite thermal) is independent of GRIDMET (reanalysis weather), so this
is a genuine verification of the danger index against observed fire — the
standard way fire-danger indices are validated. Note danger is DIAGNOSTIC of
fire-prone conditions, not a multi-day-ahead forecast; and danger is seasonal, so
a random calm control can land in another high-danger month (a hard, honest test).
"""
from __future__ import annotations

import time
from pathlib import Path

import numpy as np
import pandas as pd

from .config import Config

GM_VARS = ["burning_index", "erc", "fm100"]

# A fire-prone western-US box (CA + Great Basin + PNW interior) where large
# wildfires and FIRMS detections concentrate.
WEST_BBOX = (-124.5, 33.0, -114.0, 49.0)


def sample_fire_events(config: Config, years=(2018, 2019, 2020, 2021),
                       n: int = 80, seed: int = 0, verbose: bool = True) -> pd.DataFrame:
    """Distinct real fire location-days sampled from FIRMS peak-season detections."""
    from .training.dataset_fire import fetch_fire_detections
    west = config.with_region(WEST_BBOX, name="Western US fire box")
    frames = []
    for y in years:
        f = fetch_fire_detections(west, f"{y}-08-01", f"{y}-08-31",
                                  season=(8, 8), verbose=verbose)
        if not f.empty:
            frames.append(f)
    if not frames:
        return pd.DataFrame(columns=["lat", "lon", "date"])
    pool = pd.concat(frames, ignore_index=True).dropna(subset=["date"])
    pool["date"] = pd.to_datetime(pool["date"]).dt.normalize()
    # De-duplicate so we sample distinct fire location-days, not many pixels of
    # the same fire (round to ~1 km / same day).
    pool["rlat"] = pool["lat"].round(2)
    pool["rlon"] = pool["lon"].round(2)
    pool = pool.drop_duplicates(subset=["rlat", "rlon", "date"])
    if n and len(pool) > n:
        pool = pool.sample(n=n, random_state=seed)
    out = pool[["lat", "lon", "date"]].reset_index(drop=True)
    out["type"] = "Wildfire"
    return out


def _sigmoid(x):
    return 1.0 / (1.0 + np.exp(-x))


def fire_danger_at(lat: float, lon: float, date, config: Config) -> float:
    """Operational wildfire danger at a point for the 7-day window ending at
    `date` — the engine's GRIDMET NFDRS mapping (peak BI/ERC, driest 100-hr fuel)."""
    from .sources.gridmet import point_series
    d = pd.Timestamp(date)
    start = (d - pd.Timedelta(days=6)).date().isoformat()
    end = d.date().isoformat()
    gm = point_series(lat, lon, start, end, config.cache_dir, variables=GM_VARS)
    if gm.empty or "burning_index" not in gm or gm["burning_index"].isna().all():
        return np.nan
    bi = float(gm["burning_index"].max())
    erc = float(gm["erc"].max()) if "erc" in gm else bi
    fm = float(gm["fm100"].min()) if "fm100" in gm else 12.0
    danger = _sigmoid((bi - 45.0) / 18.0)
    erc_d = _sigmoid((erc - 55.0) / 18.0)
    fuel_dry = _sigmoid((11.0 - fm) / 3.0)
    fw_idx = np.clip(0.45 * danger + 0.3 * erc_d + 0.25 * fuel_dry, 0, 1)
    return float(np.clip(0.5 * fw_idx, 0.0, 0.6))


def fire_event_hindcast(years=(2018, 2019, 2020, 2021), n: int = 80,
                        out_path: str | Path = "cascadia_fire_performance.png",
                        throttle_s: float = 0.2, control_mode: str = "shifted",
                        verbose: bool = True) -> dict:
    from sklearn.metrics import roc_auc_score, roc_curve
    from .validation_scaled import control_date
    cfg = Config.load()
    log = (lambda *a: print(*a)) if verbose else (lambda *a: None)
    ev = sample_fire_events(cfg, years=years, n=n, seed=0, verbose=verbose)
    if ev.empty:
        raise RuntimeError("no FIRMS fire events sampled (needs FIRMS_MAP_KEY)")
    log(f"Sampled {len(ev)} real FIRMS fire location-days ({years[0]}-{years[-1]}). "
        f"Scoring each + a matched control (control={control_mode})…")
    rng = np.random.default_rng(0)
    lo_date = pd.Timestamp("1979-06-01")
    hi_date = pd.Timestamp.utcnow().tz_localize(None) - pd.Timedelta(days=10)
    rows = []
    for i, e in ev.iterrows():
        p_event = fire_danger_at(e["lat"], e["lon"], e["date"], cfg)
        cdate = control_date(e["date"], control_mode, rng, lo_date, hi_date)
        p_ctrl = fire_danger_at(e["lat"], e["lon"], cdate, cfg)
        rows.append({"prob": p_event, "label": 1})
        rows.append({"prob": p_ctrl, "label": 0})
        if throttle_s:
            time.sleep(throttle_s)
        if verbose and (i + 1) % 20 == 0:
            log(f"  …{i + 1}/{len(ev)} events scored")
    df = pd.DataFrame(rows).dropna().reset_index(drop=True)
    y = df["label"].to_numpy(); p = df["prob"].to_numpy()
    auc = float(roc_auc_score(y, p))
    rng_b = np.random.default_rng(1)
    boots = [roc_auc_score(y[idx], p[idx]) for idx in
             (rng_b.integers(0, len(y), len(y)) for _ in range(1000))
             if len(np.unique(y[idx])) == 2]
    auc_lo, auc_hi = (float(np.percentile(boots, 2.5)),
                      float(np.percentile(boots, 97.5))) if boots else (auc, auc)
    fpr_c, tpr_c, thr = roc_curve(y, p)
    k = int(np.argmax(tpr_c - fpr_c))
    thr_opt = float(thr[k])
    pred = p >= thr_opt
    hit = float((pred & (y == 1)).sum() / max(1, (y == 1).sum()))
    fa = float((pred & (y == 0)).sum() / max(1, (y == 0).sum()))
    res = {"n_events": int((y == 1).sum()), "n_nonevents": int((y == 0).sum()),
           "roc_auc": auc, "auc_ci": (auc_lo, auc_hi), "threshold": thr_opt,
           "hit_rate": hit, "false_alarm_rate": fa, "control_mode": control_mode}
    log(f"\n=== WILDFIRE EVENT HINDCAST (FIRMS labels; control={control_mode}) ===")
    log(f"  events={res['n_events']}  non-events={res['n_nonevents']}")
    log(f"  ROC-AUC = {auc:.3f}  (95% CI [{auc_lo:.3f}, {auc_hi:.3f}])")
    log(f"  at thr={thr_opt:.3f}: HIT RATE {hit:.0%}, FALSE-ALARM RATE {fa:.0%}")
    log(f"  -> 'flags {hit:.0%} of real wildfires at a {fa:.0%} false-alarm rate'")
    _render_fire(df, fpr_c, tpr_c, res, out_path)
    return res


def _render_fire(df, fpr_c, tpr_c, res, out_path):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    fig, axes = plt.subplots(1, 2, figsize=(13, 5.2), constrained_layout=True)
    ax = axes[0]
    ci = res.get("auc_ci", (res["roc_auc"], res["roc_auc"]))
    ax.plot(fpr_c, tpr_c, color="#d95f02", lw=2.5,
            label=f"fire danger (AUC={res['roc_auc']:.3f}, 95% CI [{ci[0]:.2f},{ci[1]:.2f}])")
    ax.plot([0, 1], [0, 1], "k--", lw=1, label="no skill")
    ax.scatter([res["false_alarm_rate"]], [res["hit_rate"]], color="#1b9e77", zorder=5,
               s=60, label=f"operating point ({res['hit_rate']:.0%} hit, "
                           f"{res['false_alarm_rate']:.0%} false alarm)")
    ax.set_xlabel("false-alarm rate"); ax.set_ylabel("hit rate (detection)")
    ax.set_xlim(0, 1); ax.set_ylim(0, 1); ax.grid(alpha=0.3)
    ax.legend(loc="lower right", fontsize=9)
    ax.set_title("ROC — wildfire detection vs false alarms", fontsize=11, weight="bold")
    ax2 = axes[1]
    bins = np.linspace(0, 0.6, 21)
    ax2.hist(df[df["label"] == 1]["prob"], bins=bins, alpha=0.6, color="#d95f02",
             label="real wildfires")
    ax2.hist(df[df["label"] == 0]["prob"], bins=bins, alpha=0.6, color="#f6c79b",
             label="non-events (calm)")
    ax2.axvline(res["threshold"], color="k", ls="--", lw=1, label="operating threshold")
    ax2.set_xlabel("wildfire danger (engine's GRIDMET NFDRS mapping)")
    ax2.set_ylabel("count"); ax2.grid(alpha=0.3); ax2.legend(fontsize=9)
    ax2.set_title("Score separation", fontsize=11, weight="bold")
    fig.suptitle("Cascadia — wildfire leaf vs INDEPENDENT FIRMS satellite fires "
                 f"({res['n_events']} real fire days + {res['n_nonevents']} non-events)\n"
                 f"flags {res['hit_rate']:.0%} of real wildfires at a "
                 f"{res['false_alarm_rate']:.0%} false-alarm rate  (ROC-AUC {res['roc_auc']:.3f})",
                 fontsize=12.5, weight="bold")
    fig.text(0.5, -0.03, "Positives = NASA FIRMS satellite fire detections (independent of "
             "GRIDMET reanalysis weather); negatives = same locations on shifted calm dates. "
             "Score = the engine's own GRIDMET NFDRS mapping (peak Burning Index + Energy "
             "Release Component + driest 100-hr fuel). Danger is diagnostic, not a multi-day "
             "forecast; it is seasonal, so a calm control may fall in another high-danger month.",
             ha="center", va="top", fontsize=8.5, color="0.35", wrap=True)
    fig.savefig(out_path, dpi=140, bbox_inches="tight")
    plt.close(fig)
