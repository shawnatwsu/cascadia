"""Parcel hindcast — would the product have flagged a REAL hazard event?

The honest product test: take addresses that actually experienced a documented
hazard, run the engine HISTORICALLY as of just before the event, and compare the
relevant hazard's level to a CALM control date at the same address. A useful
product should (a) be elevated at the event and (b) be low at the control — i.e.
it warns when it should and stays quiet when it shouldn't.

Only flood & wildfire are tested (the engine has historical data + skill for
them: ERA5 precip + the trained flood model; GRIDMET fire-danger). Earthquakes
are not short-term predictable; smoke needs FIRMS archive + a key.
"""
from __future__ import annotations

import pandas as pd

# Documented events: a real address, the event date, a calm control date, and
# the hazard that occurred. Addresses are geocoded via the US Census geocoder.
EVENTS = [
    {"name": "Chehalis River flood (Dec 2007)", "hazard": "flood",
     "address": "350 NW North St, Chehalis, WA",
     "event": "2007-12-03", "control": "2007-08-15"},
    {"name": "Nooksack flood, Everson (Nov 2021)", "hazard": "flood",
     "address": "111 W Main St, Everson, WA",
     "event": "2021-11-15", "control": "2021-08-15"},
    {"name": "Camp Fire, Paradise CA (Nov 2018)", "hazard": "wildfire",
     "address": "5599 Skyway, Paradise, CA",
     "event": "2018-11-08", "control": "2018-04-15"},
    {"name": "Almeda Fire, Talent OR (Sep 2020)", "hazard": "wildfire",
     "address": "206 E Main St, Talent, OR",
     "event": "2020-09-08", "control": "2020-04-15"},
    # A deliberately SAFE control address (flat inland city, no major hazard) to
    # check the model does not just flag everything.
    {"name": "Safe control: Topeka KS", "hazard": "flood",
     "address": "120 SE 6th Ave, Topeka, KS",
     "event": "2020-07-15", "control": "2020-02-15", "expect_low": True},
]


def run_hindcast(events: list | None = None, verbose: bool = True) -> pd.DataFrame:
    from .parcel import geocode, assess_point
    from .config import Config
    cfg = Config.load()
    log = (lambda *a: print(*a)) if verbose else (lambda *a: None)
    rows = []
    for ev in (events or EVENTS):
        g = geocode(ev["address"], cfg)
        if not g.matched:
            log(f"  [skip] could not geocode: {ev['address']}")
            continue
        hz = ev["hazard"]
        at_event = assess_point(g.lat, g.lon, state=g.state, as_of=ev["event"]).hazards.get(hz)
        at_control = assess_point(g.lat, g.lon, state=g.state, as_of=ev["control"]).hazards.get(hz)
        if at_event is None:
            continue
        expect_low = ev.get("expect_low", False)
        # PASS: for a real event, hazard rose AND was elevated; for a safe
        # control address, it stayed low at both dates.
        if expect_low:
            ok = (at_event < 0.15 and at_control < 0.15)
        else:
            ok = (at_event > at_control) and (at_event > 0.2)
        rows.append({"event": ev["name"], "hazard": hz,
                     "at_event": round(at_event, 3), "at_control": round(at_control, 3),
                     "lift": round(at_event - at_control, 3),
                     "verdict": "PASS" if ok else "review"})
        log(f"  {ev['name']:38} {hz:9} event={at_event:.2f} control={at_control:.2f} "
            f"lift={at_event - at_control:+.2f}  {'PASS' if ok else 'review'}")
    df = pd.DataFrame(rows)
    if verbose and not df.empty:
        npass = (df["verdict"] == "PASS").sum()
        log(f"\n{npass}/{len(df)} events behaved as a useful product should "
            f"(flagged real events, quiet at controls).")
    return df


def render_hindcast(df: pd.DataFrame, out_path: str | Path = "cascadia_hindcast.png"):
    """Bar chart: hazard level at the real event vs a calm control, per address."""
    from pathlib import Path
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import numpy as np

    df = df.iloc[::-1].reset_index(drop=True)   # top-to-bottom reads in order
    y = np.arange(len(df)); h = 0.38
    fig, ax = plt.subplots(figsize=(11, 0.9 * len(df) + 2), constrained_layout=True)
    ax.barh(y + h / 2, df["at_event"], h, color="#d62728", label="at the real event")
    ax.barh(y - h / 2, df["at_control"], h, color="#7fb3d5", label="calm control date")
    for i, r in df.iterrows():
        ax.text(max(r["at_event"], r["at_control"]) + 0.01, i,
                f"{'PASS' if r['verdict'] == 'PASS' else '?'}",
                va="center", fontsize=9, weight="bold",
                color="#1a7f37" if r["verdict"] == "PASS" else "#b00")
    ax.set_yticks(y, [f"{r['event']}\n({r['hazard']})" for _, r in df.iterrows()], fontsize=9)
    ax.set_xlabel("hazard level (flood = calibrated probability; wildfire = fire-danger index)")
    ax.set_xlim(0, max(0.8, df["at_event"].max() * 1.15))
    ax.legend(loc="lower right"); ax.grid(axis="x", alpha=0.3)
    npass = (df["verdict"] == "PASS").sum()
    ax.set_title(f"Cascadia parcel hindcast — would it have flagged real events?\n"
                 f"{npass}/{len(df)}: elevated at the actual event, quiet at a calm "
                 f"control date (same address)", fontsize=12, weight="bold")
    fig.text(0.5, -0.02, "Each address experienced the labeled hazard. The engine is run "
             "HISTORICALLY as of the event week vs a calm date. A useful product is HIGH at "
             "the event and LOW at the control (and low at a safe address). Research "
             "prototype; flood is calibrated, wildfire is a relative danger index.",
             ha="center", va="top", fontsize=8.5, color="0.35", wrap=True)
    out_path = Path(out_path)
    fig.savefig(out_path, dpi=140, bbox_inches="tight")
    plt.close(fig)
    return out_path
