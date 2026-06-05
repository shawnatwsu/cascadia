"""Validate the fire -> smoke CASCADE against observed air quality (multi-event).

The core novel claim of Cascadia is that modeling hazard *interactions* helps.
The most directly testable edge is fire -> smoke. For several major smoke
episodes we compare two ways to predict observed PM2.5 from active fires:

  * PROXIMITY (independent / no cascade): smoke ~ how much fire is nearby.
  * TRANSPORT (the cascade): fire carried DOWNWIND toward the monitor (wind).

We pool the monitor-days across events and test whether transport correlates
better with measured PM2.5 than proximity — with a BOOTSTRAP confidence interval
on the skill gain so it is not a one-event fluke. Ground truth: EPA AQS PM2.5.
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

from .config import Config

# Major smoke episodes (geographically + temporally diverse). bbox covers both
# the fires and the downwind monitors (incl. cross-border Canadian smoke in 2023).
EVENTS = [
    {"name": "CA Camp Fire (Nov 2018)", "year": 2018,
     "start": "2018-11-08", "end": "2018-11-20", "bbox": (-124.0, 35.0, -118.0, 42.0)},
    {"name": "West Coast (Sep 2020)", "year": 2020,
     "start": "2020-09-07", "end": "2020-09-20", "bbox": (-125.0, 37.0, -116.0, 49.0)},
    {"name": "Western US (Jul-Aug 2021)", "year": 2021,
     "start": "2021-07-15", "end": "2021-08-20", "bbox": (-125.0, 37.0, -116.0, 49.0)},
    {"name": "Canadian smoke, East (Jun 2023)", "year": 2023,
     "start": "2023-06-04", "end": "2023-06-12", "bbox": (-82.0, 38.0, -67.0, 52.0)},
]


def _spearman(a, b) -> float:
    from scipy.stats import spearmanr
    return float(spearmanr(a, b).correlation)


def _event_pairs(ev: dict, cfg: Config, radius_km=800.0, scale_km=250.0,
                 verbose=True) -> pd.DataFrame:
    """Per monitor-day: proximity & transport indices + observed PM2.5."""
    from scipy.spatial import cKDTree
    from .sources.airquality import aqs_window
    from .sources.gridmet import region_daily
    from .training.dataset_fire import fetch_fire_detections
    log = (lambda *a: print(*a)) if verbose else (lambda *a: None)

    obs = aqs_window(ev["year"], ev["start"], ev["end"], ev["bbox"], cfg.cache_dir)
    fires = fetch_fire_detections(cfg.with_region(ev["bbox"]), ev["start"], ev["end"],
                                  season=(1, 12), verbose=False)
    if obs.empty or fires.empty:
        log(f"  {ev['name']}: no data (monitors={len(obs)}, fires={len(fires)})")
        return pd.DataFrame()
    cube = region_daily(ev["bbox"], ev["start"], ev["end"], cfg.cache_dir,
                        variables=["wind_dir"], stride=2, verbose=False)

    lat0 = float(obs["lat"].mean()); kx = 111.0 * np.cos(np.radians(lat0))
    fxy = np.column_stack([fires["lon"].to_numpy() * kx, fires["lat"].to_numpy() * 111.0])
    fdate = pd.to_datetime(fires["date"]).to_numpy()
    tree = cKDTree(fxy); wd = cube["wind_dir"]

    rows = []
    for _, r in obs.iterrows():
        sx, sy = r["lon"] * kx, r["lat"] * 111.0
        idx = tree.query_ball_point([sx, sy], radius_km)
        if not idx:
            continue
        d = fxy[idx] - [sx, sy]
        dist = np.hypot(d[:, 0], d[:, 1]) + 1e-6
        recent = np.abs((fdate[idx] - np.datetime64(r["date"])) / np.timedelta64(1, "D")) <= 1.0
        if not recent.any():
            continue
        decay = np.exp(-dist / scale_km) * recent
        try:
            th = float(wd.sel(lat=r["lat"], lon=r["lon"], method="nearest")
                       .sel(time=np.datetime64(r["date"]), method="nearest"))
        except Exception:
            th = np.nan
        bearing = np.degrees(np.arctan2(d[:, 0], d[:, 1]))
        align = np.clip(np.cos(np.radians(bearing - th)), 0, 1) if np.isfinite(th) else 1.0
        rows.append((float(decay.sum()), float((decay * align).sum()),
                     float(r["pm25"]), ev["name"]))
    df = pd.DataFrame(rows, columns=["proximity", "transport", "pm25", "event"])
    log(f"  {ev['name']}: {len(df)} monitor-days, {len(fires)} fires")
    return df


def validate_fire_smoke_cascade(out_path: str | Path = "cascadia_cascade_skill.png",
                                events: list | None = None, n_boot: int = 3000,
                                verbose: bool = True):
    cfg = Config.load()
    log = (lambda *a: print(*a)) if verbose else (lambda *a: None)
    log("Pooling fire->smoke monitor-days across major smoke episodes…")
    parts = [_event_pairs(ev, cfg, verbose=verbose) for ev in (events or EVENTS)]
    df = pd.concat([p for p in parts if not p.empty], ignore_index=True)
    df = df[df["proximity"] > 0].reset_index(drop=True)

    r_prox, r_trans = _spearman(df["proximity"], df["pm25"]), _spearman(df["transport"], df["pm25"])
    # bootstrap the skill gain delta-r
    rng = np.random.default_rng(0)
    deltas = np.empty(n_boot)
    n = len(df)
    for b in range(n_boot):
        s = rng.integers(0, n, n)
        sub = df.iloc[s]
        deltas[b] = _spearman(sub["transport"], sub["pm25"]) - _spearman(sub["proximity"], sub["pm25"])
    lo, hi = np.percentile(deltas, [2.5, 97.5])
    p_gt0 = float((deltas > 0).mean())
    res = {"n": n, "n_events": df["event"].nunique(),
           "r_proximity": r_prox, "r_transport": r_trans,
           "skill_gain": r_trans - r_prox, "ci95": [float(lo), float(hi)],
           "frac_boot_positive": p_gt0}
    log(f"\n=== fire->smoke CASCADE — pooled over {res['n_events']} events, n={n} ===")
    log(f"  PROXIMITY (independent): Spearman r = {r_prox:+.3f}")
    log(f"  TRANSPORT (cascade)    : Spearman r = {r_trans:+.3f}")
    log(f"  skill gain delta-r     : {res['skill_gain']:+.3f}  "
        f"95% CI [{lo:+.3f}, {hi:+.3f}]  ({p_gt0:.1%} of bootstraps > 0)")
    log(f"  -> {'SIGNIFICANT: cascade adds skill' if lo > 0 else 'not significant'}")
    _render(df, res, deltas, out_path)
    return res


def _render(df, res, deltas, out_path):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    fig, axes = plt.subplots(1, 3, figsize=(16, 5), constrained_layout=True)
    events = sorted(df["event"].unique())
    colors = plt.get_cmap("tab10")(np.linspace(0, 1, len(events)))
    cmap = dict(zip(events, colors))
    for ax, col, title, r in (
            (axes[0], "proximity", "Independent: fire PROXIMITY", res["r_proximity"]),
            (axes[1], "transport", "Cascade: downwind TRANSPORT", res["r_transport"])):
        for ev in events:
            d = df[df["event"] == ev]
            ax.scatter(d[col], d["pm25"], s=9, alpha=0.4, color=cmap[ev], label=ev)
        ax.set_yscale("log"); ax.set_xlabel(f"{col} index")
        ax.set_ylabel("observed PM2.5 (ug/m3)")
        ax.set_title(f"{title}\nSpearman r = {r:+.3f}", fontsize=11, weight="bold")
        ax.grid(alpha=0.3)
    axes[0].legend(fontsize=7, loc="lower right")
    ax = axes[2]
    ax.hist(deltas, bins=40, color="#2c7fb8", alpha=0.8)
    ax.axvline(0, color="k", ls="--", lw=1)
    ax.axvline(res["skill_gain"], color="#d62728", lw=2, label="observed Δr")
    lo, hi = res["ci95"]
    ax.axvspan(lo, hi, color="#2c7fb8", alpha=0.15, label="95% CI")
    ax.set_xlabel("skill gain  Δr  (transport − proximity)")
    ax.set_ylabel("bootstrap count")
    ax.set_title(f"Skill gain: {res['skill_gain']:+.3f}\n95% CI [{lo:+.3f}, {hi:+.3f}]",
                 fontsize=11, weight="bold")
    ax.legend(fontsize=9)
    sig = "SIGNIFICANT — cascade adds skill" if lo > 0 else "not significant"
    fig.suptitle(f"Cascadia — fire→smoke cascade vs EPA observed PM2.5  "
                 f"({res['n_events']} smoke episodes, n={res['n']} monitor-days)\n"
                 f"{sig}: downwind transport beats proximity, Δr={res['skill_gain']:+.3f} "
                 f"(bootstrap {res['frac_boot_positive']:.0%} > 0)",
                 fontsize=12.5, weight="bold")
    fig.savefig(out_path, dpi=140, bbox_inches="tight")
    plt.close(fig)
