"""Validate the fire -> smoke CASCADE against observed air quality.

The core novel claim of Cascadia is that modeling hazard *interactions* helps.
The most directly testable edge is fire -> smoke. We compare two ways to predict
observed PM2.5 from active fires:

  * PROXIMITY (independent / no cascade): smoke ~ how much fire is nearby.
  * TRANSPORT (the cascade): smoke ~ fire carried DOWNWIND toward the monitor,
    using wind direction.

If the wind-aware transport model correlates better with measured PM2.5 than
proximity alone, the fire -> smoke cascade demonstrably *adds skill* — the
paper's central result, tested against EPA ground truth.

Default event: the September 2020 West-Coast wildfire smoke episode.
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

from .config import Config


def validate_fire_smoke_cascade(
    out_path: str | Path = "cascadia_cascade_skill.png",
    year: int = 2020, start: str = "2020-09-07", end: str = "2020-09-20",
    bbox: tuple[float, float, float, float] = (-125.0, 37.0, -116.0, 49.0),
    radius_km: float = 800.0, scale_km: float = 250.0, verbose: bool = True):
    from scipy.spatial import cKDTree
    from scipy.stats import spearmanr
    from .sources.airquality import aqs_window
    from .sources.gridmet import region_daily
    from .training.dataset_fire import fetch_fire_detections

    cfg = Config.load()
    log = (lambda *a: print(*a)) if verbose else (lambda *a: None)

    obs = aqs_window(year, start, end, bbox, cfg.cache_dir)
    log(f"observed PM2.5 monitor-days: {len(obs)} (max {obs['pm25'].max():.0f} ug/m3)")
    fcfg = cfg.with_region(bbox)
    fires = fetch_fire_detections(fcfg, start, end, season=(1, 12), verbose=False)
    log(f"FIRMS fire detections: {len(fires)}")
    cube = region_daily(bbox, start, end, cfg.cache_dir, variables=["wind_dir"],
                        stride=2, verbose=False)
    if obs.empty or fires.empty:
        raise RuntimeError("no observations or fires in window (need FIRMS_MAP_KEY)")

    # planar km coords
    lat0 = float(obs["lat"].mean()); kx = 111.0 * np.cos(np.radians(lat0))
    fxy = np.column_stack([fires["lon"].to_numpy() * kx, fires["lat"].to_numpy() * 111.0])
    fdate = pd.to_datetime(fires["date"]).to_numpy()
    tree = cKDTree(fxy)
    wd = cube["wind_dir"]

    prox, trans, pm = [], [], []
    for _, r in obs.iterrows():
        sx, sy = r["lon"] * kx, r["lat"] * 111.0
        idx = tree.query_ball_point([sx, sy], radius_km)
        if not idx:
            continue
        d = fxy[idx] - [sx, sy]
        dist = np.hypot(d[:, 0], d[:, 1]) + 1e-6
        # only fires active within +/-1 day of the observation
        dday = np.abs((fdate[idx] - np.datetime64(r["date"])) / np.timedelta64(1, "D"))
        w_time = dday <= 1.0
        if not w_time.any():
            continue
        decay = np.exp(-dist / scale_km) * w_time
        # wind direction at the monitor that day (deg the wind blows FROM)
        try:
            th = float(wd.sel(lat=r["lat"], lon=r["lon"], method="nearest")
                       .sel(time=np.datetime64(r["date"]), method="nearest"))
        except Exception:
            th = np.nan
        bearing = np.degrees(np.arctan2(d[:, 0], d[:, 1]))   # monitor -> fire
        align = np.clip(np.cos(np.radians(bearing - th)), 0, 1) if np.isfinite(th) else 1.0
        prox.append(float(decay.sum()))
        trans.append(float((decay * align).sum()))
        pm.append(float(r["pm25"]))

    df = pd.DataFrame({"proximity": prox, "transport": trans, "pm25": pm})
    df = df[(df["proximity"] > 0)].reset_index(drop=True)
    r_prox = spearmanr(df["proximity"], df["pm25"]).correlation
    r_trans = spearmanr(df["transport"], df["pm25"]).correlation
    result = {"n": len(df), "r_proximity": float(r_prox), "r_transport": float(r_trans),
              "skill_gain": float(r_trans - r_prox)}
    log(f"\n=== fire->smoke CASCADE validation (n={result['n']} monitor-days) ===")
    log(f"  PROXIMITY (independent)  : Spearman r = {r_prox:+.3f}")
    log(f"  TRANSPORT (cascade)      : Spearman r = {r_trans:+.3f}")
    log(f"  -> cascade skill gain    : {result['skill_gain']:+.3f}  "
        f"({'cascade ADDS skill' if result['skill_gain'] > 0 else 'no gain'})")
    _render(df, result, start, end, out_path)
    return result


def _render(df, res, start, end, out_path):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    fig, axes = plt.subplots(1, 2, figsize=(12, 5), constrained_layout=True)
    for ax, col, title, r in (
            (axes[0], "proximity", "Independent: fire PROXIMITY", res["r_proximity"]),
            (axes[1], "transport", "Cascade: downwind TRANSPORT", res["r_transport"])):
        ax.scatter(df[col], df["pm25"], s=8, alpha=0.3, color="#444")
        ax.set_yscale("log"); ax.set_xlabel(f"{col} index"); ax.set_ylabel("observed PM2.5 (ug/m3)")
        ax.set_title(f"{title}\nSpearman r = {r:+.3f}", fontsize=11, weight="bold")
        ax.grid(alpha=0.3)
    verdict = ("Cascade ADDS skill" if res["skill_gain"] > 0 else "No cascade gain")
    fig.suptitle(f"Cascadia — fire→smoke cascade validation vs EPA observed PM2.5 "
                 f"({start}…{end})\n{verdict}: transport beats proximity by "
                 f"Δr = {res['skill_gain']:+.3f}  (n={res['n']} monitor-days)",
                 fontsize=12, weight="bold")
    fig.text(0.5, -0.04, "Each point is a monitor-day. PROXIMITY = fire weighted by "
             "distance only (treats smoke as independent of wind). TRANSPORT = the "
             "same fires weighted by whether they lie UPWIND of the monitor (the "
             "fire→smoke cascade). Higher correlation with measured PM2.5 = more "
             "skill. Ground truth: EPA AQS 24h PM2.5.", ha="center", va="top",
             fontsize=8.5, color="0.3", wrap=True)
    fig.savefig(out_path, dpi=140, bbox_inches="tight")
    plt.close(fig)
