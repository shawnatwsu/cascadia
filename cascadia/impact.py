"""Impact = hazard probability x exposure.

Converts the hazard-probability surface into expected *impact* — the number of
people expected to be affected — by multiplying each cell's hazard probability
by its estimated population (open Census data). This is the metric emergency
managers and insurers actually act on.

`expected_affected` uses the compound risk (P at least one hazard); per-hazard
`impact_<hazard>` layers attribute the expected affected population to each
hazard. Population exposure can later be extended to building value / FEMA EAL.
"""
from __future__ import annotations

from pathlib import Path

import pandas as pd

from .config import Config
from .sources.exposure import assign_population, load_counties

HAZARDS_P = ["flood", "landslide", "wildfire", "heat", "earthquake"]


def add_impact(risk: pd.DataFrame, cache_dir: Path,
               res_deg: float | None = None) -> pd.DataFrame:
    """Add population + expected-affected columns to a risk surface."""
    counties = load_counties(cache_dir)
    out = risk.copy()
    out["population"] = assign_population(out, counties, res_deg)
    out["expected_affected"] = out["compound_risk"] * out["population"]
    for h in HAZARDS_P:
        if f"p_{h}" in out.columns:
            out[f"impact_{h}"] = out[f"p_{h}"] * out["population"]
    return out


def impact_map(region: str = "conus", out_path: str | Path = "cascadia_impact_map.png",
               verbose: bool = True):
    """Compute the conditions risk for a region, weight by population, and render
    the expected-impact maps."""
    from .conditions import conditions_map, DEFAULT_RES
    from .cartomap import static_risk_map

    risk, _ = conditions_map(region, verbose=verbose, render=False)
    region_name = risk.attrs.get("region_name", region.upper())
    res = risk.attrs.get("res", DEFAULT_RES.get(region, 0.1))
    risk = add_impact(risk, Config.load().cache_dir, res)

    total_pop = int(risk["population"].sum())
    total_aff = int(risk["expected_affected"].sum())
    if verbose:
        print(f"population in region: {total_pop:,} | expected affected "
              f"(sum compound x pop): {total_aff:,}")

    cols = ["expected_affected", "impact_flood", "impact_landslide",
            "impact_wildfire", "impact_heat", "population"]
    cols = [c for c in cols if c in risk.columns]
    out = static_risk_map(
        risk, region_name, out_path, cols=cols,
        value_label="expected people affected (per cell)",
        suptitle=(f"Cascadia — expected hazard IMPACT\n{region_name}  ·  "
                  "probability x population"),
        description=(
            "Expected people affected = hazard probability x estimated cell "
            "population (US Census county density). 'Expected people affected' "
            "uses the compound risk (any hazard); per-hazard panels attribute it "
            "to each hazard. Population panel shows the exposure surface. "
            "Per-panel binned scales — read each colorbar."))
    if verbose:
        print(f"Impact map written: {out}")
    return risk, out
