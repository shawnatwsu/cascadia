"""FastAPI service exposing the Cascadia risk surface (Phase 2).

Run:  uvicorn cascadia.api:app --reload
Then: http://127.0.0.1:8000/        (dashboard)
      http://127.0.0.1:8000/docs    (OpenAPI)

The pipeline result is cached with a TTL so requests are cheap; live feeds are
only refetched when the cache expires.
"""
from __future__ import annotations

import time
from pathlib import Path

from fastapi import FastAPI, Query
from fastapi.responses import HTMLResponse, JSONResponse

from .config import Config
from .pipeline import PipelineResult, run_pipeline
from .validate import EVENTS, run_validation

app = FastAPI(title="Cascadia", version="0.1.0",
              description="Compound & cascading natural-hazard engine")

_CACHE: dict[str, tuple[float, PipelineResult]] = {}
_TTL_S = 1800  # 30 min


def _get_result(as_of: str | None = None) -> PipelineResult:
    key = as_of or "live"
    hit = _CACHE.get(key)
    if hit and (time.time() - hit[0]) < _TTL_S:
        return hit[1]
    cfg = Config.load()
    if as_of:
        cfg = cfg.with_as_of(as_of)
    res = run_pipeline(cfg, verbose=False)
    _CACHE[key] = (time.time(), res)
    return res


@app.get("/health")
def health() -> dict:
    return {"status": "ok", "service": "cascadia", "version": app.version}


@app.get("/risk")
def risk(as_of: str | None = Query(None, description="ISO date for historical mode")) -> JSONResponse:
    """Full per-cell risk surface as GeoJSON-ish records."""
    res = _get_result(as_of)
    cols = ["cell_id", "lat", "lon", "compound_risk", "expected_hazards",
            "dominant_chain", "co_occurring",
            "p_earthquake", "p_landslide", "p_flood", "p_wildfire"]
    cols = [c for c in cols if c in res.risk.columns]
    return JSONResponse({
        "region": res.config.region.name,
        "as_of": as_of or "live",
        "horizon_days": res.config.horizon_days,
        "cells": res.risk[cols].round(4).to_dict(orient="records"),
    })


@app.get("/risk/top")
def risk_top(n: int = 10, as_of: str | None = None) -> JSONResponse:
    res = _get_result(as_of)
    return JSONResponse(res.top_cells(n).round(4).to_dict(orient="records"))


@app.get("/events")
def events() -> dict:
    return {k: {"name": v.name, "as_of": v.as_of,
                "expected_hazards": v.expected_hazards} for k, v in EVENTS.items()}


@app.get("/validate/{event_key}")
def validate(event_key: str) -> JSONResponse:
    if event_key not in EVENTS:
        return JSONResponse({"error": f"unknown event '{event_key}'",
                             "known": list(EVENTS)}, status_code=404)
    vr = run_validation(event_key, verbose=False)
    return JSONResponse({
        "event": vr.event.name,
        "passed": vr.passed,
        "hazard_risk_event": round(vr.hazard_risk_event, 3),
        "hazard_risk_control": round(vr.hazard_risk_control, 3),
        "hazard_lift": round(vr.hazard_lift, 3),
        "chain_hit": vr.chain_hit,
        "impact_chains": vr.impact_chains,
    })


@app.get("/parcel")
def parcel(address: str | None = Query(None, description="US street address"),
           lat: float | None = None, lon: float | None = None) -> JSONResponse:
    """Address-level (ParcelRisk) query: geocode -> point cascade-risk assessment."""
    from .parcel import assess_address, assess_point
    if address:
        return JSONResponse(assess_address(address))
    if lat is not None and lon is not None:
        return JSONResponse({**assess_point(lat, lon).to_dict(), "matched": True})
    return JSONResponse({"error": "provide ?address= or ?lat=&lon="}, status_code=400)


@app.get("/", response_class=HTMLResponse)
def dashboard() -> str:
    html = Path(__file__).parent / "static" / "index.html"
    return html.read_text(encoding="utf-8")
