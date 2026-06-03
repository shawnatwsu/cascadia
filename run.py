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


def _open(path_or_url: str) -> None:
    try:
        webbrowser.open(path_or_url)
    except Exception:
        pass


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
    out = ROOT / "cascadia_risk_map.png"
    static_risk_map(res.risk, cfg.region.name, out, panels=True, as_of=as_of)
    print(f"\n✓ Map written: {out}")
    print("  Opening it…")
    _open(out.resolve().as_uri())


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
    from cascadia.parcel import assess_address
    import json
    address = " ".join(args)
    print(f"Assessing: {address}\n(geocoding + running the cascade engine at that point…)\n")
    print(json.dumps(assess_address(address), indent=2))


def cmd_conditions(args: list[str]) -> None:
    from cascadia.conditions import conditions_map, REGIONS
    region = (args[0].lower() if args else "pnw")
    if region not in REGIONS:
        print(f"Unknown region '{region}'. Choices: {', '.join(REGIONS)}")
        return
    print(f"Building GRIDMET 4km hazard-conditions nowcast for {region.upper()}…\n")
    _, out = conditions_map(region)
    print("  Opening map…")
    _open(Path(out).resolve().as_uri())


def cmd_impact(args: list[str]) -> None:
    region = (args[0].lower() if args else "conus")
    from cascadia.impact import impact_map
    print(f"Building expected-IMPACT map for {region.upper()} "
          "(hazard probability x population)…\n")
    _, out = impact_map(region)
    print(f"\n✓ Impact map: {out}")
    _open(Path(out).resolve().as_uri())


def cmd_subseasonal(args: list[str]) -> None:
    from cascadia.subseasonal import subseasonal_outlook
    from cascadia.conditions import REGIONS
    region = (args[0].lower() if args else "pnw")
    if region not in REGIONS:
        print(f"Unknown region '{region}'. Choices: {', '.join(REGIONS)}")
        return
    print(f"Building weeks 2-6 sub-seasonal outlook for {region.upper()}…\n")
    _, out = subseasonal_outlook(region)
    print("  Opening map…")
    _open(Path(out).resolve().as_uri())


COMMANDS = {
    "map": cmd_map, "": cmd_map,
    "train": cmd_train,
    "validate": cmd_validate,
    "serve": cmd_serve,
    "all": cmd_all,
    "parcel": cmd_parcel,
    "conditions": cmd_conditions,
    "subseasonal": cmd_subseasonal,
    "impact": cmd_impact,
}


def main() -> None:
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
