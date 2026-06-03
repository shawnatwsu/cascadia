"""Real-time monitoring + threshold alerting.

Runs the live pipeline, finds cells whose compound risk (or a specific hazard's
probability) crosses a threshold, and emits structured alerts. State is kept in
a small JSON file so an alert only fires when a cell *enters* alert status or
escalates — not every run. Wire `run_monitor` into the `/schedule` skill or any
cron to get recurring surveillance.
"""
from __future__ import annotations

import json
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from pathlib import Path

import pandas as pd

from .config import Config
from .pipeline import run_pipeline

STATE_FILE = "data/monitor_state.json"


@dataclass
class Alert:
    cell_id: int
    lat: float
    lon: float
    level: str                # "watch" | "warning"
    compound_risk: float
    dominant_chain: str
    co_occurring: str
    issued: str

    def line(self) -> str:
        return (f"[{self.level.upper():7}] risk={self.compound_risk:.2f} "
                f"@({self.lat:.2f},{self.lon:.2f})  "
                f"chain: {self.dominant_chain or '—'}  | co-occur: {self.co_occurring}")


def _band(risk: float, watch: float, warning: float) -> str | None:
    if risk >= warning:
        return "warning"
    if risk >= watch:
        return "watch"
    return None


def run_monitor(
    config: Config | None = None,
    watch: float = 0.6,
    warning: float = 0.8,
    state_path: str | Path = STATE_FILE,
    verbose: bool = True,
) -> list[Alert]:
    """Run the live pipeline and return *new or escalated* alerts."""
    config = config or Config.load()
    res = run_pipeline(config, verbose=verbose)
    now = datetime.now(timezone.utc).isoformat(timespec="seconds")

    state_path = Path(state_path)
    state_path.parent.mkdir(parents=True, exist_ok=True)
    prev = {}
    if state_path.exists():
        try:
            prev = json.loads(state_path.read_text(encoding="utf-8"))
        except json.JSONDecodeError:
            prev = {}

    rank = {None: 0, "watch": 1, "warning": 2}
    alerts: list[Alert] = []
    new_state: dict[str, str] = {}

    for _, row in res.risk.iterrows():
        level = _band(float(row["compound_risk"]), watch, warning)
        cid = str(int(row["cell_id"]))
        if level is None:
            continue
        new_state[cid] = level
        # Fire only when newly alerting or escalating in severity.
        if rank[level] > rank.get(prev.get(cid), 0):
            alerts.append(
                Alert(
                    cell_id=int(row["cell_id"]),
                    lat=float(row["lat"]), lon=float(row["lon"]),
                    level=level,
                    compound_risk=float(row["compound_risk"]),
                    dominant_chain=str(row.get("dominant_chain", "")),
                    co_occurring=str(row.get("co_occurring", "")),
                    issued=now,
                )
            )

    state_path.write_text(json.dumps(new_state), encoding="utf-8")

    if verbose:
        print(f"\n=== Monitor @ {now} ===")
        print(f"Cells alerting: {len(new_state)} "
              f"(warning={sum(v=='warning' for v in new_state.values())}, "
              f"watch={sum(v=='watch' for v in new_state.values())})")
        if alerts:
            print(f"NEW / escalated alerts: {len(alerts)}")
            for a in sorted(alerts, key=lambda x: -x.compound_risk)[:15]:
                print("  " + a.line())
        else:
            print("No new alerts since last run.")

    return alerts


if __name__ == "__main__":
    run_monitor()
