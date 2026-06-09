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
# A diverse, a-priori set of MAJOR documented US smoke episodes (2017–2023),
# chosen for notability + geographic/temporal spread — NOT screened by whether
# they support the cascade. Each bbox spans both the fires and the downwind
# monitors. More episodes = more clusters = a properly powered episode-level test.
EVENTS = [
    {"name": "PNW smoke summer (Sep 2017)", "year": 2017,
     "start": "2017-09-02", "end": "2017-09-12", "bbox": (-125.0, 42.0, -110.0, 49.5)},
    {"name": "N. California Carr/Mendocino (Aug 2018)", "year": 2018,
     "start": "2018-08-01", "end": "2018-08-14", "bbox": (-124.0, 36.0, -118.0, 42.0)},
    {"name": "CA Camp Fire (Nov 2018)", "year": 2018,
     "start": "2018-11-08", "end": "2018-11-20", "bbox": (-124.0, 35.0, -118.0, 42.0)},
    {"name": "Bay Area CZU/SCU (Aug 2020)", "year": 2020,
     "start": "2020-08-18", "end": "2020-08-30", "bbox": (-123.5, 36.0, -119.5, 39.5)},
    {"name": "West Coast (Sep 2020)", "year": 2020,
     "start": "2020-09-07", "end": "2020-09-20", "bbox": (-125.0, 37.0, -116.0, 49.0)},
    {"name": "Colorado Cameron Peak/E. Troublesome (Oct 2020)", "year": 2020,
     "start": "2020-10-14", "end": "2020-10-24", "bbox": (-109.0, 37.0, -102.0, 41.5)},
    {"name": "Western US (Jul-Aug 2021)", "year": 2021,
     "start": "2021-07-15", "end": "2021-08-20", "bbox": (-125.0, 37.0, -116.0, 49.0)},
    {"name": "Eastern US transport (late Jul 2021)", "year": 2021,
     "start": "2021-07-19", "end": "2021-07-28", "bbox": (-80.0, 38.0, -67.0, 47.0)},
    {"name": "Canadian smoke, East (Jun 2023)", "year": 2023,
     "start": "2023-06-04", "end": "2023-06-12", "bbox": (-82.0, 38.0, -67.0, 52.0)},
    {"name": "Canadian smoke, Midwest (late Jun 2023)", "year": 2023,
     "start": "2023-06-26", "end": "2023-07-03", "bbox": (-95.0, 38.0, -80.0, 48.5)},
    {"name": "Canadian smoke, PNW/Montana (Aug 2023)", "year": 2023,
     "start": "2023-08-15", "end": "2023-08-25", "bbox": (-120.0, 44.0, -104.0, 49.5)},
]


def _spearman(a, b) -> float:
    from scipy.stats import spearmanr
    a = np.asarray(a, float); b = np.asarray(b, float)
    if len(a) < 4 or np.all(a == a[0]) or np.all(b == b[0]):
        return np.nan
    return float(spearmanr(a, b).correlation)


def _delta(sub) -> float:
    """Skill gain (transport − proximity) Spearman-r on a subset."""
    return _spearman(sub["transport"], sub["pm25"]) - _spearman(sub["proximity"], sub["pm25"])


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
    n = len(df)
    rng = np.random.default_rng(0)

    # (a) PER-EPISODE skill gain — the honest evidence when clusters are few:
    # is transport > proximity CONSISTENTLY across independent episodes?
    ev_names = sorted(df["event"].unique())
    groups = {e: df[df["event"] == e] for e in ev_names}
    per_event = []
    for e in ev_names:
        g = groups[e]
        per_event.append({"event": e, "n": int(len(g)),
                          "r_proximity": _spearman(g["proximity"], g["pm25"]),
                          "r_transport": _spearman(g["transport"], g["pm25"]),
                          "delta": _delta(g)})
    deltas_ev = np.array([p["delta"] for p in per_event], float)
    n_pos = int(np.nansum(deltas_ev > 0)); n_ev_valid = int(np.isfinite(deltas_ev).sum())

    # (b) CLUSTER (block) bootstrap — resample whole EPISODES with replacement, so
    # the CI respects within-episode autocorrelation (the effective sample size is
    # ~#episodes, not #monitor-days). This is the defensible CI.
    boot_clu = []
    for _ in range(n_boot):
        chosen = rng.choice(ev_names, size=len(ev_names), replace=True)
        sub = pd.concat([groups[e] for e in chosen], ignore_index=True)
        d = _delta(sub)
        if np.isfinite(d):
            boot_clu.append(d)
    boot_clu = np.array(boot_clu)
    lo_c, hi_c = np.percentile(boot_clu, [2.5, 97.5])
    p_gt0_c = float((boot_clu > 0).mean())

    # (c) NAIVE monitor-day bootstrap — kept for transparency, but it IGNORES
    # autocorrelation and so overstates significance; reported as such.
    deltas = np.array([_delta(df.iloc[rng.integers(0, n, n)]) for _ in range(n_boot)])
    deltas = deltas[np.isfinite(deltas)]
    lo_n, hi_n = np.percentile(deltas, [2.5, 97.5])

    res = {"n": n, "n_events": df["event"].nunique(),
           "r_proximity": r_prox, "r_transport": r_trans,
           "skill_gain": r_trans - r_prox,
           "ci95_cluster": [float(lo_c), float(hi_c)], "frac_boot_positive": p_gt0_c,
           "ci95_naive": [float(lo_n), float(hi_n)],
           "per_event": per_event, "n_pos": n_pos, "n_ev_valid": n_ev_valid,
           "ci95": [float(lo_c), float(hi_c)]}  # headline = cluster CI
    log(f"\n=== fire->smoke CASCADE — pooled over {res['n_events']} episodes, n={n} monitor-days ===")
    log(f"  PROXIMITY (independent): Spearman r = {r_prox:+.3f}")
    log(f"  TRANSPORT (cascade)    : Spearman r = {r_trans:+.3f}")
    log(f"  pooled skill gain delta-r = {res['skill_gain']:+.3f}")
    log(f"  per-episode delta-r: transport beats proximity in {n_pos}/{n_ev_valid} episodes "
        f"[{', '.join(f'{d:+.3f}' for d in deltas_ev)}]")
    log(f"  CLUSTER bootstrap (by episode) 95% CI [{lo_c:+.3f}, {hi_c:+.3f}] "
        f"({p_gt0_c:.0%} > 0)  <- defensible")
    log(f"  naive monitor-day bootstrap 95% CI [{lo_n:+.3f}, {hi_n:+.3f}] "
        f"(autocorrelation-inflated; do not headline)")
    verdict = ("SIGNIFICANT (cluster CI excludes 0)" if lo_c > 0 else
               f"consistent but not cluster-significant: positive in {n_pos}/{n_ev_valid} "
               f"episodes, wide CI with only {n_ev_valid} clusters")
    log(f"  -> {verdict}")
    _render(df, res, boot_clu, deltas_ev, out_path)
    return res


def _render(df, res, boot_clu, deltas_ev, out_path):
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

    # Panel 3: cluster (by-episode) bootstrap + per-episode Δr dots (the honest view).
    ax = axes[2]
    lo, hi = res["ci95_cluster"]
    ax.hist(boot_clu, bins=40, color="#2c7fb8", alpha=0.75,
            label="episode-cluster bootstrap")
    ax.axvline(0, color="k", ls="--", lw=1)
    ax.axvline(res["skill_gain"], color="#d62728", lw=2, label="pooled Δr")
    ax.axvspan(lo, hi, color="#2c7fb8", alpha=0.15, label="cluster 95% CI")
    # per-episode Δr as a rug of points (sign consistency)
    pe = [p for p in res["per_event"] if np.isfinite(p["delta"])]
    yld = ax.get_ylim()[1]
    ax.scatter([p["delta"] for p in pe], [yld * 0.06] * len(pe),
               color="#d62728", edgecolor="k", s=55, zorder=5, marker="o",
               label=f"per-episode Δr ({res['n_pos']}/{res['n_ev_valid']} > 0)")
    ax.set_xlabel("skill gain  Δr  (transport − proximity)")
    ax.set_ylabel("bootstrap count")
    ax.set_title(f"Skill gain Δr = {res['skill_gain']:+.3f}\n"
                 f"cluster 95% CI [{lo:+.3f}, {hi:+.3f}]  ·  naive "
                 f"[{res['ci95_naive'][0]:+.3f}, {res['ci95_naive'][1]:+.3f}]",
                 fontsize=10.5, weight="bold")
    ax.legend(fontsize=8, loc="upper right")

    consistent = res["n_pos"] == res["n_ev_valid"]
    head = ("downwind transport beats proximity in "
            f"{res['n_pos']}/{res['n_ev_valid']} independent episodes"
            + (" — cluster CI excludes 0" if lo > 0 else
               f" (pooled Δr={res['skill_gain']:+.3f}; cluster CI wide with only "
               f"{res['n_ev_valid']} episodes)"))
    fig.suptitle(f"Cascadia — fire→smoke cascade vs EPA observed PM2.5  "
                 f"({res['n_events']} smoke episodes, n={res['n']} monitor-days)\n{head}",
                 fontsize=12, weight="bold")
    fig.text(0.5, -0.02, f"Honest stats: monitor-days within one episode are highly "
             f"autocorrelated, so the effective sample is the EPISODE ({res['n_ev_valid']} of "
             "them), not the monitor-day. We headline the per-episode sign consistency and an "
             "episode-cluster bootstrap (which respects that autocorrelation); the naive "
             "monitor-day CI is shown for transparency but is autocorrelation-inflated.",
             ha="center", va="top", fontsize=8.3, color="0.35", wrap=True)
    fig.savefig(out_path, dpi=140, bbox_inches="tight")
    plt.close(fig)
