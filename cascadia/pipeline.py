"""End-to-end pipeline: ingest -> fuse -> predict -> cascade -> risk surface."""
from __future__ import annotations

from dataclasses import dataclass

import pandas as pd

from .config import Config
from .features import Grid, build_indicators
from .models import base_probabilities, default_cascade
from .sources import FIRMS, NWSAlerts, OpenMeteo, USGSQuakes, USGSWater
from .sources.seismicity import Seismicity
from .sources.landslide_inventory import LandslideInventory


@dataclass
class PipelineResult:
    config: Config
    raw: dict[str, pd.DataFrame]
    features: pd.DataFrame
    base_probs: pd.DataFrame
    risk: pd.DataFrame  # final per-cell cascade output

    def top_cells(self, n: int = 10) -> pd.DataFrame:
        cols = ["cell_id", "lat", "lon", "compound_risk", "expected_hazards",
                "dominant_chain"]
        return self.risk.sort_values("compound_risk", ascending=False).head(n)[cols]

    def predictor_status(self) -> str:
        """Report each hazard leaf's method (trained model, prior, or heuristic)."""
        from .models.predictors import HAZARDS
        from .models.trained import load_trained
        # Method for the non-trained leaves (data-driven where applicable).
        method = {"earthquake": "seismic-prior", "wildfire": "GRIDMET-fire",
                  "landslide": "suscept×trigger", "flood": "heuristic",
                  "heat": "heat-index"}
        parts = []
        for hz in HAZARDS:
            t = load_trained(hz)
            if t is not None:
                parts.append(f"{hz}=TRAINED(ROC-AUC {t.metrics.get('roc_auc', float('nan')):.3f})")
            else:
                parts.append(f"{hz}={method.get(hz, 'heuristic')}")
        return " | ".join(parts)

    def summary(self) -> str:
        r = self.risk
        lines = [
            f"Region: {self.config.region.name}",
            f"Predictors: {self.predictor_status()}",
            f"Cells: {len(r)} | horizon: {self.config.horizon_days}d",
            f"Observations: "
            + ", ".join(f"{k}={len(v)}" for k, v in self.raw.items()),
            f"Mean compound risk: {r['compound_risk'].mean():.3f} | "
            f"max: {r['compound_risk'].max():.3f}",
        ]
        chains = r[r["compound_risk"] > 0.2]["dominant_chain"].value_counts().head(5)
        if not chains.empty:
            lines.append("Top cascade chains (cells with risk > 0.2):")
            for chain, cnt in chains.items():
                lines.append(f"  [{cnt:>3} cells]  {chain}")
        return "\n".join(lines)


def run_pipeline(config: Config | None = None, verbose: bool = True) -> PipelineResult:
    config = config or Config.load()
    log = (lambda *a: print(*a)) if verbose else (lambda *a: None)

    log("• Ingesting open feeds…")
    raw = {}
    raw["quakes"] = USGSQuakes(config).fetch()
    raw["alerts"] = NWSAlerts(config).fetch()
    raw["water"] = USGSWater(config).fetch()
    raw["weather"] = OpenMeteo(config).fetch()
    raw["fires"] = FIRMS(config).fetch()
    raw["catalog"] = Seismicity(config).fetch()
    raw["inventory"] = LandslideInventory(config).fetch()
    for k, v in raw.items():
        log(f"   {k}: {len(v)} rows")

    # GRIDMET (4km CONUS) fire/heat variables — graceful if unavailable.
    gridmet_cube = None
    try:
        from .sources.gridmet import region_daily, gridmet_window
        r = config.region
        gstart, gend = gridmet_window(config)
        gridmet_cube = region_daily((r.min_lon, r.min_lat, r.max_lon, r.max_lat),
                                    gstart, gend, config.cache_dir, verbose=False)
        log(f"   gridmet: {dict(gridmet_cube.sizes)} ({gstart}..{gend})")
    except Exception as exc:
        log(f"   gridmet: unavailable ({str(exc)[:60]}) — using fallback fire/heat")

    log("• Building grid + fusing cross-sector indicators…")
    grid = Grid.from_region(config.region)
    features = build_indicators(
        grid,
        quakes=raw["quakes"],
        weather=raw["weather"],
        water=raw["water"],
        alerts=raw["alerts"],
        fires=raw["fires"],
        catalog=raw["catalog"],
        inventory=raw["inventory"],
        gridmet_cube=gridmet_cube,
        horizon_days=config.horizon_days,
    )
    log(f"   {len(features)} cells x {features.shape[1]} features")

    log("• Computing per-hazard base probabilities…")
    base = base_probabilities(features)

    log("• Propagating the cascade graph…")
    cascade = default_cascade()
    risk = cascade.run(base, features)

    log("• Done.")
    return PipelineResult(config=config, raw=raw, features=features,
                          base_probs=base, risk=risk)
