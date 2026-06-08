"""Cascadia one-command launcher.

Friendly entry point so you can produce a result with a single command and have
it open automatically. Examples:

    python run.py              # live risk map -> opens in your browser
    python run.py train        # train the flood ML model, print the scorecard
    python run.py validate     # replay real disasters, print pass/fail
    python run.py serve        # launch the dashboard -> opens in your browser
    python run.py all          # train (if needed) + map, all at once

The Windows wrappers run.bat / run.ps1 also set up the environment for you, so a
double-click or `./run.ps1` is enough.
"""
from __future__ import annotations

import sys
import webbrowser
from pathlib import Path

ROOT = Path(__file__).resolve().parent
OUTPUT_DIR = ROOT / "outputs"


def _outfile(name: str) -> Path:
    """Absolute path inside the outputs/ folder (created on demand)."""
    OUTPUT_DIR.mkdir(exist_ok=True)
    return OUTPUT_DIR / name


def _open(path_or_url: str) -> None:
    try:
        webbrowser.open(path_or_url)
    except Exception:
        pass


def _announce(out) -> None:
    """Print the saved file's FULL path and open it."""
    p = Path(out).resolve()
    print(f"\n✓ Saved: {p}")
    print("  (opening it now…)")
    _open(p.as_uri())


def cmd_map(args: list[str]) -> None:
    from cascadia.config import Config
    from cascadia.pipeline import run_pipeline
    from cascadia.cartomap import static_risk_map

    print("Running Cascadia on live open feeds (first run fetches data)…\n")
    cfg = Config.load()
    res = run_pipeline(cfg)
    print("\n" + res.summary())
    print("\nTop risk cells:\n" + res.top_cells(8).to_string(index=False))

    as_of = cfg.as_of.date().isoformat() if cfg.as_of else "live"
    out = _outfile("live_forecast_map.png")
    static_risk_map(res.risk, cfg.region.name, out, panels=True, as_of=as_of,
                    value_label="hazard probability (next 7 days)",
                    provenance=("Open-Meteo 7-day forecast · USGS seismicity+streamflow "
                                "· USGS landslide inventory · NASA FIRMS · Albers equal-area"))
    _announce(out)


def cmd_train(args: list[str]) -> None:
    which = (args[0].lower() if args else "flood")
    targets = ["flood", "fire"] if which == "all" else [which]
    for t in targets:
        if t in ("flood",):
            from cascadia.training.train_flood import train
            train(verbose=True)
        elif t in ("fire", "wildfire"):
            from cascadia.training.train_fire import train
            try:
                train(verbose=True)
            except RuntimeError as e:
                print(f"\n⚠ Wildfire training skipped: {e}")
        else:
            print(f"Unknown hazard '{t}'. Try: flood, fire, all")
            return
    print("\n✓ Trained. Future runs use these models automatically "
          "(see '=TRAINED' in the run summary).")


def cmd_validate(args: list[str]) -> None:
    from cascadia.validate import backtest
    print("Replaying documented disasters (event window vs. calm control)…\n")
    df = backtest(verbose=False)
    print(df.to_string(index=False))
    passed = int(df["passed"].sum())
    print(f"\n✓ {passed}/{len(df)} events passed "
          f"(realized hazards flagged + clear risk lift).")


def cmd_serve(args: list[str]) -> None:
    import uvicorn
    port = 8077
    print(f"Starting dashboard at http://127.0.0.1:{port}/  (Ctrl+C to stop)…")
    _open(f"http://127.0.0.1:{port}/")
    uvicorn.run("cascadia.api:app", host="127.0.0.1", port=port, log_level="warning")


def cmd_all(args: list[str]) -> None:
    from cascadia.training.train_flood import MODEL_PATH
    if not MODEL_PATH.exists():
        print("No trained model yet — training first…\n")
        cmd_train(args)
        print("\n" + "=" * 60 + "\n")
    cmd_map(args)


def cmd_parcel(args: list[str]) -> None:
    if not args:
        print('Usage: python run.py parcel "123 Main St, City, ST"')
        return
    from cascadia.parcel import assess_address, parcel_report
    import json
    address = " ".join(args)
    print(f"Assessing: {address}\n(geocoding + running the cascade engine at that point…)\n")
    print(json.dumps(assess_address(address), indent=2))
    out = parcel_report(address, out_path=_outfile("parcel_report.png"))
    if out:
        _announce(out)


def cmd_conditions(args: list[str]) -> None:
    from cascadia.conditions import conditions_map, region_keys
    region = (args[0].lower() if args else "pnw")
    if region not in region_keys():
        print(f"Unknown region '{region}'. Choices: {', '.join(region_keys())}")
        return
    print(f"Building GRIDMET 4km hazard-conditions nowcast for {region.upper()}…\n")
    _, out = conditions_map(region, out_path=_outfile(f"conditions_{region}.png"))
    _announce(out)


def cmd_impact(args: list[str]) -> None:
    region = (args[0].lower() if args else "conus")
    from cascadia.impact import impact_map
    print(f"Building expected-IMPACT map for {region.upper()} "
          "(hazard probability x population)…\n")
    _, out = impact_map(region, out_path=_outfile(f"impact_{region}.png"))
    _announce(out)


def cmd_hindcast(args: list[str]) -> None:
    """Would the product have flagged real hazard events? (address hindcast)."""
    from cascadia.parcel_hindcast import run_hindcast, render_hindcast
    print("Hindcasting real hazard events (event week vs calm control, per address)…\n")
    df = run_hindcast(verbose=True)
    if not df.empty:
        out = _outfile("parcel_hindcast.png")
        render_hindcast(df, out)
        _announce(out)


def cmd_skill(args: list[str]) -> None:
    """Calibration & skill validation — the peer-review verification suite."""
    print("=== Cascadia skill & calibration validation ===\n")
    print("[1/3] ENSO -> regional climate teleconnection (vs NCEI observed)…")
    from cascadia.skill_enso import validate_enso
    validate_enso(out_path=_outfile("skill_enso_teleconnection.png"))
    print(f"  -> {_outfile('skill_enso_teleconnection.png')}\n")
    print("[2/3] Flood model calibration (out-of-fold reliability)…")
    from cascadia.skill_models import validate_flood, validate_enso_forecast
    validate_flood(out_path=_outfile("skill_flood_reliability.png"), verbose=True)
    print(f"  -> {_outfile('skill_flood_reliability.png')}\n")
    print("[3/4] ENSO (ONI) forecast skill…")
    validate_enso_forecast(out_path=_outfile("skill_enso_forecast.png"), verbose=True)
    print(f"  -> {_outfile('skill_enso_forecast.png')}\n")
    print("[4/4] Fire→smoke CASCADE vs observed PM2.5 (does the cascade add skill?)…")
    try:
        from cascadia.skill_cascade import validate_fire_smoke_cascade
        validate_fire_smoke_cascade(out_path=_outfile("skill_cascade.png"), verbose=True)
        print(f"  -> {_outfile('skill_cascade.png')}")
    except Exception as e:
        print(f"  (skipped: {e} — needs FIRMS_MAP_KEY)")
    print(f"\n✓ Skill figures saved in: {OUTPUT_DIR.resolve()}")
    _open(OUTPUT_DIR.resolve().as_uri())


def cmd_seasonal(args: list[str]) -> None:
    from cascadia.seasonal import seasonal_outlook
    lead = int(args[0]) if args and args[0].isdigit() else 0
    msg = (f"using the {lead}-month ENSO forecast" if lead else "using the current ENSO state")
    print(f"Building ENSO-driven seasonal hazard outlook ({msg})…\n")
    name = f"seasonal_lead{lead}.png" if lead else "seasonal_current.png"
    _, out = seasonal_outlook(out_path=_outfile(name), lead=lead)
    _announce(out)


def cmd_subseasonal(args: list[str]) -> None:
    from cascadia.subseasonal import subseasonal_outlook
    from cascadia.conditions import region_keys
    region = (args[0].lower() if args else "pnw")
    if region not in region_keys():
        print(f"Unknown region '{region}'. Choices: {', '.join(region_keys())}")
        return
    print(f"Building weeks 2-6 sub-seasonal outlook for {region.upper()}…\n")
    _, out = subseasonal_outlook(region, out_path=_outfile(f"subseasonal_{region}.png"))
    _announce(out)


COMMANDS = {
    "map": cmd_map, "": cmd_map,
    "train": cmd_train,
    "validate": cmd_validate,
    "serve": cmd_serve,
    "all": cmd_all,
    "parcel": cmd_parcel,
    "conditions": cmd_conditions,
    "subseasonal": cmd_subseasonal,
    "seasonal": cmd_seasonal,
    "impact": cmd_impact,
    "skill": cmd_skill,
    "hindcast": cmd_hindcast,
}


def main() -> None:
    # Make console output UTF-8 safe regardless of how Python was launched
    # (Windows consoles default to cp1252 and choke on ✓/· etc.).
    for stream in (sys.stdout, sys.stderr):
        try:
            stream.reconfigure(encoding="utf-8")
        except Exception:
            pass
    cmd = sys.argv[1].lower() if len(sys.argv) > 1 else ""
    if cmd in ("-h", "--help", "help"):
        print(__doc__)
        return
    handler = COMMANDS.get(cmd)
    if handler is None:
        print(f"Unknown command '{cmd}'. Try: {', '.join(k for k in COMMANDS if k)}")
        print("\n" + __doc__)
        return
    handler(sys.argv[2:])


if __name__ == "__main__":
    main()
