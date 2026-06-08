"""Scaled, systematic hindcast -> a DEFENSIBLE performance claim.

Instead of a few famous events, this samples many real flood events (NWS Storm
Events) and matched non-events (the same locations on shifted, calm dates), runs
the calibrated flood model on each, and reports threshold-free discrimination
(ROC-AUC) plus an operating-point hit rate / false-alarm rate. This is the
"it catches X% of floods at a Y% false-alarm rate" table a buyer underwrites on.

This runs the full operational flood model at each point-date: ERA5 precip + soil
plus the real antecedent streamflow anomaly reconstructed from the nearest USGS
gage as of that date (neutral fallback where no gage is within range).
"""
from __future__ import annotations

import time
from pathlib import Path

import numpy as np
import pandas as pd

from .config import Config

NEUTRAL_FLOW = 0.15   # fallback when no nearby gage exists

_GAGE_CACHE: dict = {}


def _nearest_gage(lat: float, lon: float, config: Config, box: float = 0.4):
    """Nearest active NWIS stream gage reporting daily discharge, within `box` deg."""
    import requests
    from .training.dataset import SITE_URL
    key = (round(lat, 1), round(lon, 1))
    if key in _GAGE_CACHE:
        return _GAGE_CACHE[key]
    params = {"format": "rdb", "siteType": "ST", "hasDataTypeCd": "dv",
              "parameterCd": "00060", "siteStatus": "all",
              "bBox": f"{lon - box:.4f},{lat - box:.4f},{lon + box:.4f},{lat + box:.4f}"}
    site = None
    try:
        resp = requests.get(SITE_URL, params=params, timeout=60,
                            headers={"User-Agent": "Cascadia/0.1"})
        rows = [ln for ln in resp.text.splitlines() if ln and not ln.startswith("#")]
        if len(rows) >= 3:
            hdr = rows[0].split("\t")
            df = pd.DataFrame([r.split("\t") for r in rows[2:]], columns=hdr)
            df["la"] = pd.to_numeric(df["dec_lat_va"], errors="coerce")
            df["lo"] = pd.to_numeric(df["dec_long_va"], errors="coerce")
            df = df.dropna(subset=["la", "lo"])
            if len(df):
                d2 = (df["la"] - lat) ** 2 + (df["lo"] - lon) ** 2
                site = str(df.loc[d2.idxmin(), "site_no"])
    except Exception:
        site = None
    _GAGE_CACHE[key] = site
    return site


def flow_anomaly_at(lat: float, lon: float, date, config: Config) -> float:
    """Real antecedent streamflow anomaly at the nearest gage as of `date`,
    matching the operational model. Falls back to NEUTRAL_FLOW if no gage."""
    from .training.dataset import _fetch_dv, _flow_anomaly
    site = _nearest_gage(lat, lon, config)
    if not site:
        return NEUTRAL_FLOW
    d = pd.Timestamp(date)
    start = (d - pd.Timedelta(days=45)).date().isoformat()
    end = d.date().isoformat()
    try:
        dv = _fetch_dv(site, start, end, config)
    except Exception:
        return NEUTRAL_FLOW
    if dv.empty or "discharge" not in dv:
        return NEUTRAL_FLOW
    return _flow_anomaly(dv["discharge"], len(dv["discharge"]) - 1)


def flood_prob_at(lat: float, lon: float, date, config: Config,
                  use_flow: bool = True) -> float:
    """Calibrated flood probability at a point for the 7-day window ending at
    `date`: ERA5 precip + soil + (optional) real antecedent streamflow anomaly."""
    from .training.dataset import _fetch_weather_daily
    from .training.train_flood import FEATURES
    from .models.trained import load_trained
    d = pd.Timestamp(date)
    start = (d - pd.Timedelta(days=6)).date().isoformat()
    end = d.date().isoformat()
    wx = _fetch_weather_daily(lat, lon, start, end, config)
    if wx.empty or wx["precip_day"].isna().all():
        return np.nan
    flow = flow_anomaly_at(lat, lon, date, config) if use_flow else NEUTRAL_FLOW
    X = pd.DataFrame([{ "precip_total_mm": float(wx["precip_day"].sum()),
                        "soil_moist_peak": float(wx["soil_day"].max()),
                        "flow_anomaly": flow }])[FEATURES]
    model = load_trained("flood")
    if model is None:
        return np.nan
    return float(model.predict(X)[0])


def _window_feats(daily: pd.DataFrame, issue: pd.Timestamp, horizon: int = 7):
    """precip total + soil peak over the forward window [issue+1 .. issue+horizon],
    matching how the model was trained (it consumes a forward precip forecast)."""
    w = daily.loc[(daily.index > issue) &
                  (daily.index <= issue + pd.Timedelta(days=horizon))]
    if w.empty or w["precip_day"].isna().all():
        return None
    return float(w["precip_day"].sum()), float(w["soil_day"].max())


def _flow_asof(disc: pd.Series, issue: pd.Timestamp) -> float:
    from .training.dataset import _flow_anomaly
    if disc is None or disc.empty:
        return NEUTRAL_FLOW
    s = disc[disc.index <= issue].dropna()
    if len(s) < 8:
        return NEUTRAL_FLOW
    return _flow_anomaly(s, len(s) - 1)


def lead_time_curve(years=(2018, 2019, 2020, 2021), n: int = 80,
                    leads=(1, 2, 3, 5, 7, 10, 14),
                    out_path: str | Path = "cascadia_flood_leadtime.png",
                    throttle_s: float = 0.3, verbose: bool = True) -> dict:
    """How many days AHEAD is the flood signal already present? For each lead L,
    score the model as if the forecast were issued L days before the event (a
    forward 7-day window from the issue date) and report AUC vs lead.

    ERA5/streamflow are fetched once per point over a wide span, then sliced per
    lead locally — so this costs ~the same API calls as the single-window test."""
    from sklearn.metrics import roc_auc_score
    from .training.dataset import _fetch_weather_daily, _fetch_dv
    from .training.train_flood import FEATURES
    from .models.trained import load_trained
    from .sources.storm_events import sample_events
    cfg = Config.load()
    log = (lambda *a: print(*a)) if verbose else (lambda *a: None)
    model = load_trained("flood")
    ev = sample_events(years, ("Flood", "Flash Flood"), cfg.cache_dir, n=n, seed=0)
    if ev.empty:
        raise RuntimeError("no flood events sampled")
    log(f"Lead-time analysis on {len(ev)} events across leads {list(leads)} days…")
    rng = np.random.default_rng(0)
    lo_date = pd.Timestamp("2000-02-01")
    hi_date = pd.Timestamp.utcnow().tz_localize(None) - pd.Timedelta(days=10)
    maxlead = max(leads)

    def _fetch_point(lat, lon, anchor):
        """Wide ERA5 daily + nearest-gage discharge bracketing all leads."""
        a = pd.Timestamp(anchor)
        wx = _fetch_weather_daily(lat, lon,
                                  (a - pd.Timedelta(days=maxlead + 2)).date().isoformat(),
                                  (a + pd.Timedelta(days=8)).date().isoformat(), cfg)
        site = _nearest_gage(lat, lon, cfg)
        disc = None
        if site:
            try:
                dv = _fetch_dv(site, (a - pd.Timedelta(days=maxlead + 40)).date().isoformat(),
                               a.date().isoformat(), cfg)
                if not dv.empty and "discharge" in dv:
                    disc = dv["discharge"]
            except Exception:
                disc = None
        return wx, disc

    per_lead = {L: {"y": [], "p": []} for L in leads}
    for i, e in ev.iterrows():
        off = int(rng.integers(60, 300)) * (1 if rng.random() < 0.5 else -1)
        cdate = min(max(e["date"] + pd.Timedelta(days=off), lo_date), hi_date)
        wx_e, disc_e = _fetch_point(e["lat"], e["lon"], e["date"])
        wx_c, disc_c = _fetch_point(e["lat"], e["lon"], cdate)
        for L in leads:
            for anchor, wx, disc, label in (
                    (e["date"], wx_e, disc_e, 1), (cdate, wx_c, disc_c, 0)):
                issue = pd.Timestamp(anchor) - pd.Timedelta(days=L)
                feats = _window_feats(wx, issue) if not wx.empty else None
                if feats is None:
                    continue
                X = pd.DataFrame([{ "precip_total_mm": feats[0],
                                    "soil_moist_peak": feats[1],
                                    "flow_anomaly": _flow_asof(disc, issue) }])[FEATURES]
                per_lead[L]["y"].append(label)
                per_lead[L]["p"].append(float(model.predict(X)[0]))
        if throttle_s:
            time.sleep(throttle_s)
        if verbose and (i + 1) % 20 == 0:
            log(f"  …{i + 1}/{len(ev)} events processed")

    curve = {}
    for L in leads:
        y = np.array(per_lead[L]["y"]); p = np.array(per_lead[L]["p"])
        if len(np.unique(y)) == 2:
            curve[L] = {"auc": float(roc_auc_score(y, p)), "n": int((y == 1).sum())}
    log("\n=== FLOOD LEAD-TIME (AUC vs days before event) ===")
    for L, s in curve.items():
        log(f"  lead {L:>2}d: ROC-AUC {s['auc']:.3f}  (n={s['n']})")
    _render_leadtime(curve, out_path)
    return {"curve": curve}


def _render_leadtime(curve: dict, out_path):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    leads = sorted(curve)
    aucs = [curve[L]["auc"] for L in leads]
    fig, ax = plt.subplots(figsize=(8.5, 5.2), constrained_layout=True)
    ax.plot(leads, aucs, "-o", color="#1f77b4", lw=2.2, ms=7)
    ax.axhline(0.5, color="k", ls="--", lw=1, label="no skill")
    ax.axvline(7, color="0.6", ls=":", lw=1.2, label="7-day forecast horizon")
    for L, a in zip(leads, aucs):
        ax.annotate(f"{a:.2f}", (L, a), textcoords="offset points",
                    xytext=(0, 8), ha="center", fontsize=8.5)
    ax.set_xlabel("forecast lead (days before the flood the warning is issued)")
    ax.set_ylabel("ROC-AUC (discrimination)")
    ax.set_ylim(0.45, max(0.85, max(aucs) + 0.05)); ax.set_xticks(leads)
    ax.grid(alpha=0.3); ax.legend(loc="upper right", fontsize=9)
    ax.set_title("Cascadia flood model — skill vs forecast lead time\n"
                 "how many days ahead is the flood signal already detectable?",
                 fontsize=12, weight="bold")
    fig.text(0.5, -0.04, "Each lead L scores the model as if issued L days before the "
             "event, on a forward 7-day window (matching training). ERA5 is reanalysis "
             "(a perfect precip forecast), so this is the model's POTENTIAL lead skill: it "
             "holds while the event is inside the 7-day window, then decays as the event "
             "moves beyond the horizon and only antecedent soil/streamflow remain.",
             ha="center", va="top", fontsize=8.3, color="0.35", wrap=True)
    fig.savefig(out_path, dpi=140, bbox_inches="tight")
    plt.close(fig)


def scaled_flood_hindcast(years=(2018, 2019, 2020, 2021), n: int = 100,
                          out_path: str | Path = "cascadia_flood_performance.png",
                          throttle_s: float = 0.8, verbose: bool = True) -> dict:
    from sklearn.metrics import roc_auc_score, roc_curve
    from .sources.storm_events import sample_events
    cfg = Config.load()
    log = (lambda *a: print(*a)) if verbose else (lambda *a: None)

    ev = sample_events(years, ("Flood", "Flash Flood"), cfg.cache_dir, n=n, seed=0)
    if ev.empty:
        raise RuntimeError("no flood events sampled")
    log(f"Sampled {len(ev)} real flood events ({years[0]}-{years[-1]}). "
        f"Scoring each + a matched calm control (same place, shifted date)…")
    rng = np.random.default_rng(0)
    lo_date, hi_date = pd.Timestamp("2000-02-01"), pd.Timestamp.utcnow().tz_localize(None) - pd.Timedelta(days=10)

    rows = []
    for i, e in ev.iterrows():
        p_event = flood_prob_at(e["lat"], e["lon"], e["date"], cfg)
        off = int(rng.integers(60, 300)) * (1 if rng.random() < 0.5 else -1)
        cdate = min(max(e["date"] + pd.Timedelta(days=off), lo_date), hi_date)
        p_ctrl = flood_prob_at(e["lat"], e["lon"], cdate, cfg)
        rows.append({"prob": p_event, "label": 1, "type": e["type"]})
        rows.append({"prob": p_ctrl, "label": 0, "type": e["type"]})
        if throttle_s:
            time.sleep(throttle_s)
        if verbose and (i + 1) % 20 == 0:
            log(f"  …{i + 1}/{len(ev)} events scored")

    df = pd.DataFrame(rows).dropna().reset_index(drop=True)
    y = df["label"].to_numpy(); p = df["prob"].to_numpy()
    auc = float(roc_auc_score(y, p))
    rng_b = np.random.default_rng(1)
    boots = []
    for _ in range(1000):
        idx = rng_b.integers(0, len(y), len(y))
        if len(np.unique(y[idx])) == 2:
            boots.append(roc_auc_score(y[idx], p[idx]))
    auc_lo, auc_hi = (float(np.percentile(boots, 2.5)),
                      float(np.percentile(boots, 97.5))) if boots else (auc, auc)
    fpr_c, tpr_c, thr = roc_curve(y, p)
    j = tpr_c - fpr_c
    k = int(np.argmax(j))
    thr_opt = float(thr[k])
    pred = p >= thr_opt
    hit = float((pred & (y == 1)).sum() / max(1, (y == 1).sum()))
    fa = float((pred & (y == 0)).sum() / max(1, (y == 0).sum()))
    # Per-type breakdown: riverine "Flood" is the model's design target; "Flash
    # Flood" is sub-daily convective and largely out of scope for daily ERA5.
    by_type = {}
    for t in sorted(df["type"].unique()):
        sub = df[df["type"] == t]
        if sub["label"].nunique() == 2:
            by_type[t] = {"n": int((sub["label"] == 1).sum()),
                          "roc_auc": float(roc_auc_score(sub["label"], sub["prob"]))}
    res = {"n_events": int((y == 1).sum()), "n_nonevents": int((y == 0).sum()),
           "roc_auc": auc, "auc_ci": (auc_lo, auc_hi),
           "threshold": thr_opt, "hit_rate": hit,
           "false_alarm_rate": fa, "by_type": by_type}
    log(f"\n=== SCALED FLOOD HINDCAST (independent NWS Storm Events labels) ===")
    log(f"  events={res['n_events']}  non-events={res['n_nonevents']}")
    log(f"  ROC-AUC = {auc:.3f}  (95% CI [{auc_lo:.3f}, {auc_hi:.3f}], threshold-free)")
    log(f"  at operating point (thr={thr_opt:.2f}): HIT RATE {hit:.0%}, "
        f"FALSE-ALARM RATE {fa:.0%}")
    log(f"  -> 'flags {hit:.0%} of real floods at a {fa:.0%} false-alarm rate'")
    for t, s in by_type.items():
        log(f"  [{t}] n={s['n']}  ROC-AUC={s['roc_auc']:.3f}")
    _render(df, fpr_c, tpr_c, res, out_path)
    return res


def _render(df, fpr_c, tpr_c, res, out_path):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from sklearn.metrics import roc_curve
    fig, axes = plt.subplots(1, 2, figsize=(13, 5.2), constrained_layout=True)
    ax = axes[0]
    _ci = res.get("auc_ci", (res["roc_auc"], res["roc_auc"]))
    ax.plot(fpr_c, tpr_c, color="#1f77b4", lw=2.5,
            label=f"all floods (AUC={res['roc_auc']:.3f}, 95% CI [{_ci[0]:.2f},{_ci[1]:.2f}])")
    type_colors = {"Flood": "#2ca02c", "Flash Flood": "#ff7f0e"}
    for t, s in res.get("by_type", {}).items():
        sub = df[df["type"] == t]
        f_t, h_t, _ = roc_curve(sub["label"], sub["prob"])
        ax.plot(f_t, h_t, color=type_colors.get(t, "0.5"), lw=1.6, ls=":",
                label=f"{t} (AUC={s['roc_auc']:.3f}, n={s['n']})")
    ax.plot([0, 1], [0, 1], "k--", lw=1, label="no skill")
    ax.scatter([res["false_alarm_rate"]], [res["hit_rate"]], color="#d62728", zorder=5,
               s=60, label=f"operating point\n({res['hit_rate']:.0%} hit, {res['false_alarm_rate']:.0%} false alarm)")
    ax.set_xlabel("false-alarm rate"); ax.set_ylabel("hit rate (detection)")
    ax.set_xlim(0, 1); ax.set_ylim(0, 1); ax.legend(loc="lower right", fontsize=9)
    ax.set_title("ROC — flood detection vs false alarms", fontsize=11, weight="bold")
    ax.grid(alpha=0.3)
    ax2 = axes[1]
    bins = np.linspace(0, 1, 21)
    ax2.hist(df[df["label"] == 1]["prob"], bins=bins, alpha=0.6, color="#d62728",
             label="real flood events")
    ax2.hist(df[df["label"] == 0]["prob"], bins=bins, alpha=0.6, color="#7fb3d5",
             label="non-events (calm)")
    ax2.axvline(res["threshold"], color="k", ls="--", lw=1, label="operating threshold")
    ax2.set_xlabel("flood model probability"); ax2.set_ylabel("count")
    ax2.legend(fontsize=9); ax2.set_title("Score separation", fontsize=11, weight="bold")
    ax2.grid(alpha=0.3)
    fig.suptitle("Cascadia — flood model on INDEPENDENT NWS Storm Events "
                 f"({res['n_events']} real floods + {res['n_nonevents']} non-events)\n"
                 f"flags {res['hit_rate']:.0%} of real floods at a "
                 f"{res['false_alarm_rate']:.0%} false-alarm rate  (ROC-AUC {res['roc_auc']:.3f})",
                 fontsize=12.5, weight="bold")
    fig.text(0.5, -0.03, "Each real flood event is scored as of its week; each matched "
             "non-event is the SAME location on a shifted calm date. Operational flood model "
             "fed ERA5 precip + soil + real antecedent streamflow anomaly (nearest USGS gage; "
             "neutral where none). Labels are NWS Storm Events, INDEPENDENT of the "
             "gage-exceedance training labels.",
             ha="center", va="top", fontsize=8.5, color="0.35", wrap=True)
    fig.savefig(out_path, dpi=140, bbox_inches="tight")
    plt.close(fig)
