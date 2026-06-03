"""Event validation: replay a known historical disaster and test whether the
engine localizes the realized hazard chain.

This is a *conditions-consistency* validation: using the meteorological /
hydrological conditions that actually occurred over the event window (ERA5
archive + NWIS), does the cascade produce elevated risk — and the correct
dominant chain — at the locations that were actually impacted?

It is deliberately NOT a forecast-skill test (that needs archived forecasts,
which is future work). It answers a prior, necessary question: is the
conditions -> cascade -> risk mapping pointing at the right places and the
right hazards?
"""
from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
import pandas as pd

from .config import Config
from .pipeline import PipelineResult, run_pipeline


@dataclass
class Event:
    name: str
    as_of: str                       # ISO date the event window starts
    bbox: tuple[float, float, float, float]
    impact_points: list[tuple[float, float]]   # (lat, lon) actually impacted
    expected_hazards: list[str]      # hazards that were realized
    control_as_of: str               # a calm window in the same place (baseline)
    note: str = ""
    state: str = "WA"


# A small catalog of well-documented Cascadia compound events.
EVENTS: dict[str, Event] = {
    "chehalis_2007": Event(
        name="December 2007 Chehalis River flood (Lewis County, WA)",
        as_of="2007-12-02",
        control_as_of="2007-08-01",
        bbox=(-124.0, 46.2, -122.2, 47.1),
        impact_points=[(46.66, -122.97), (46.72, -122.95), (46.97, -123.81)],
        expected_hazards=["flood", "landslide"],
        note="Atmospheric river; record rainfall; catastrophic flooding + "
             "numerous landslides closed I-5 for days.",
    ),
    "nooksack_2021": Event(
        name="November 2021 Nooksack/Whatcom flood (Bellingham/Sumas, WA)",
        as_of="2021-11-13",
        control_as_of="2021-08-01",
        bbox=(-123.3, 48.3, -121.3, 49.05),
        impact_points=[(49.00, -122.26), (48.92, -122.34), (48.75, -122.48)],
        expected_hazards=["flood", "landslide"],
        note="Atmospheric river over saturated ground; Nooksack overtopped; "
             "major flooding in Sumas/Everson; landslides across NW WA.",
    ),
}


@dataclass
class ValidationResult:
    event: Event
    result: PipelineResult           # the event-window run (for mapping)
    impact_risk_event: float         # mean compound risk at impact cells, event window
    impact_risk_control: float       # ... during the calm control window
    risk_lift: float                 # event - control (the skill signal)
    hazard_risk_event: float         # P(realized hazards) at impact, event window
    hazard_risk_control: float       # ... control window
    hazard_lift: float               # the sharp, targeted skill signal
    chain_hit: bool                  # expected hazards present in impact chains
    impact_chains: list[str] = field(default_factory=list)
    impact_cooccur: list[str] = field(default_factory=list)

    @property
    def passed(self) -> bool:
        # Skillful if the realized hazards show up AND their probability clearly
        # rose vs the calm control window at the impacted locations.
        return self.chain_hit and self.hazard_lift > 0.2

    def report(self) -> str:
        ev = self.event
        verdict = "PASS" if self.passed else "review"
        lines = [
            f"=== Validation [{verdict}]: {ev.name} ===",
            f"Event window: {ev.as_of}   control window: {ev.control_as_of}",
            f"Note: {ev.note}",
            f"Cells: {len(self.result.risk)} | impact points: {len(ev.impact_points)}",
            "",
            f"P(realized hazards {ev.expected_hazards}) at impacted cells:",
            f"   event  ({ev.as_of}):   {self.hazard_risk_event:.3f}",
            f"   control({ev.control_as_of}): {self.hazard_risk_control:.3f}",
            f"   hazard lift (event - control): +{self.hazard_lift:.3f}   <-- skill signal",
            f"(compound risk, all hazards: event {self.impact_risk_event:.3f} / "
            f"control {self.impact_risk_control:.3f})",
            "",
            f"Realized hazards {ev.expected_hazards} present in impact chains: "
            f"{'YES' if self.chain_hit else 'no'}",
            f"Co-occurring hazards flagged at impact: "
            f"{', '.join(sorted(set(self.impact_cooccur))) or '(none)'}",
            "Dominant trigger chains at impact cells:",
        ]
        for c in self.impact_chains:
            lines.append(f"   - {c}")
        return "\n".join(lines)


def backtest(base_config: Config | None = None, verbose: bool = False) -> pd.DataFrame:
    """Run every catalogued event and return a one-row-per-event scorecard."""
    rows = []
    for key in EVENTS:
        vr = run_validation(key, base_config=base_config, verbose=verbose)
        rows.append(
            {
                "event": key,
                "as_of": vr.event.as_of,
                "expected": "+".join(vr.event.expected_hazards),
                "chain_hit": vr.chain_hit,
                "hazard_risk_event": round(vr.hazard_risk_event, 3),
                "hazard_risk_control": round(vr.hazard_risk_control, 3),
                "hazard_lift": round(vr.hazard_lift, 3),
                "passed": vr.passed,
            }
        )
    return pd.DataFrame(rows)


def _nearest_cell(risk: pd.DataFrame, lat: float, lon: float) -> pd.Series:
    d2 = (risk["lat"] - lat) ** 2 + (risk["lon"] - lon) ** 2
    return risk.loc[d2.idxmin()]


def _impact_risk(risk: pd.DataFrame, points: list[tuple[float, float]],
                 col: str = "compound_risk") -> float:
    rows = [_nearest_cell(risk, la, lo) for la, lo in points]
    return float(np.mean([float(r[col]) for r in rows]))


def _impact_hazard_risk(risk: pd.DataFrame, points: list[tuple[float, float]],
                        hazards: list[str]) -> float:
    """Mean over impact cells of P(any realized hazard) = noisy-OR of the
    realized hazards' probabilities — isolates the targeted signal from the
    compound score (which also carries unrelated seasonal hazards)."""
    rows = [_nearest_cell(risk, la, lo) for la, lo in points]
    vals = []
    for r in rows:
        surv = 1.0
        for h in hazards:
            surv *= (1.0 - float(r.get(f"p_{h}", 0.0)))
        vals.append(1.0 - surv)
    return float(np.mean(vals))


def run_validation(event_key: str, base_config: Config | None = None,
                   verbose: bool = True) -> ValidationResult:
    ev = EVENTS[event_key]
    base = (base_config or Config.load()).with_region(
        ev.bbox, name=ev.name, state=ev.state
    )

    # Event-window run (the disaster) and a control run (calm period, same place).
    res = run_pipeline(base.with_as_of(ev.as_of), verbose=verbose)
    control = run_pipeline(base.with_as_of(ev.control_as_of), verbose=False)

    impact_rows = [_nearest_cell(res.risk, la, lo) for la, lo in ev.impact_points]
    impact_chains = [str(r["dominant_chain"]) for r in impact_rows]
    impact_cooccur: list[str] = []
    for r in impact_rows:
        impact_cooccur += [h.strip() for h in str(r.get("co_occurring", "")).split("+")]

    risk_event = _impact_risk(res.risk, ev.impact_points)
    risk_control = _impact_risk(control.risk, ev.impact_points)
    haz_event = _impact_hazard_risk(res.risk, ev.impact_points, ev.expected_hazards)
    haz_control = _impact_hazard_risk(control.risk, ev.impact_points, ev.expected_hazards)

    chain_hit = any(
        any(h in chain for h in ev.expected_hazards) for chain in impact_chains
    )

    return ValidationResult(
        event=ev,
        result=res,
        impact_risk_event=risk_event,
        impact_risk_control=risk_control,
        risk_lift=risk_event - risk_control,
        hazard_risk_event=haz_event,
        hazard_risk_control=haz_control,
        hazard_lift=haz_event - haz_control,
        chain_hit=chain_hit,
        impact_chains=impact_chains,
        impact_cooccur=[c for c in impact_cooccur if c and "none" not in c],
    )
