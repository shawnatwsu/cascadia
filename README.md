# Cascadia — a compound & cascading natural-hazard engine

> Most hazard systems answer *"what is the flood risk?"* — independently, one
> hazard at a time. **Cascadia** answers a different, harder question:
> *"given conditions right now, what chain of hazards is likely to unfold over
> the next week, and where?"* It fuses free, open data across sectors into a
> probabilistic **cascade graph** where hazards can trigger one another.

The name is a double meaning: the **Cascadia** bioregion (Pacific NW) *and* the
**cascading**-hazard idea at the core of the model.

## Why this is different

| Conventional systems | Cascadia |
|---|---|
| Single hazard (flood **or** fire **or** quake) | **Multi-hazard** on one frame |
| Hazards treated as independent | **Cascade graph**: quake→landslide→flood, fire→debris-flow, etc. |
| Static susceptibility lookups | **Condition-gated** triggers (a quake is far likelier to start a landslide on a *rain-saturated* slope) |
| One sector's data | **Cross-sector fusion**: seismic + hydro-met + hydrology + official warnings |
| "Risk = 0.4" | **Explainable chains**: the dominant trigger path per cell |

## How it works

```
open feeds ─▶ spatial grid ─▶ cross-sector ─▶ per-hazard ─▶ CASCADE GRAPH ─▶ compound
(no API key)   (0.1° cells)    indicators      base probs    (noisy-OR over     risk
                                (fusion)        (ML hybrid)    gated triggers)   surface
```

1. **Ingest** open feeds (all free; the four below need **no API key**):
   - USGS earthquake feed
   - NWS/NOAA active alerts (flood, fire-weather, heat…)
   - Open-Meteo forecast (precipitation + soil moisture)
   - USGS NWIS streamflow & gage height
   - *(optional)* NASA FIRMS active fire — needs a free `FIRMS_MAP_KEY`
2. **Fuse** every source onto a common grid of ~11 km cells (`features/`).
   An `IsolationForest` flags anomalous streamflow as a flood precursor — real
   ML on real data, no labels required.
3. **Base predictors** map each cell's indicators to a per-hazard initiation
   probability. The MVP uses transparent calibrated sigmoids; each predictor
   shares one interface so a trained scikit-learn model can drop in per hazard
   (the "ML" half of the hybrid).
4. **Cascade graph** (`models/cascade_graph.py`) — the contribution. Hazards are
   nodes; edges are physical trigger pathways with a base transfer probability
   and a **condition gate**. Probabilities propagate in topological order via
   noisy-OR, fusing self-initiation with upstream triggers, and the dominant
   trigger chain is traced per cell for explainability.

### The cascade DAG (default)

```
earthquake ──▶ landslide ──▶ flood
     │            ▲            ▲
     └────────────┼────────────┘  (dam/levee failure)
  wildfire ───────┘  (post-fire debris flow, realized when rain arrives)
```

## Quick start — one command

**Windows:** double-click **`run.bat`**, or in a terminal:

```powershell
.\run.ps1            # sets up everything the first time, then opens the risk map
```

That's it — it creates the virtual environment, installs dependencies (first
run only), runs the engine on live open feeds, and **opens the risk map in your
browser**. Other one-word commands:

```powershell
.\run.ps1 train       # train the flood ML model + print the scorecard
.\run.ps1 validate    # replay real disasters, print pass/fail
.\run.ps1 serve       # launch the interactive dashboard (opens browser)
.\run.ps1 conditions pnw     # 4km GRIDMET hazard nowcast over a big region
.\run.ps1 conditions conus   # ...or the whole US (national map)
.\run.ps1 subseasonal pnw    # weeks 2-6 fire/drought/heat outlook
.\run.ps1 parcel "1300 Franklin St, Vancouver, WA"   # address-level risk
```

One entry point (`run.py` / `run.ps1` / `run.bat`) drives every mode — the live
forecast map, the GRIDMET 4 km regional/national nowcast, the sub-seasonal
outlook, training, validation, the dashboard, and address queries — each
producing the same projected, per-hazard-colorbar cartographic output.

### Hazard methods (all data-driven / open data)

| Hazard | Method |
|---|---|
| Flood | Trained gradient-boosting model (ROC-AUC 0.949) |
| Earthquake | USGS smoothed-seismicity Poisson prior + aftershocks |
| Wildfire | Hot-Dry-Windy fire-danger index + live FIRMS detections |
| Landslide | USGS landslide-inventory susceptibility prior × rainfall trigger |

### Address-level risk (ParcelRisk integration)

```bash
python run.py parcel "1300 Franklin St, Vancouver, WA"   # or: GET /parcel?address=...
```
Geocodes the address (free U.S. Census geocoder) and returns per-hazard
probabilities, compound risk, and the cascade story for that point — the engine
as an address-level product backend.

<details>
<summary>Prefer plain Python / no wrapper?</summary>

```bash
python -m venv .venv && .venv\Scripts\activate
pip install -r requirements.txt
python run.py            # = the map command above
python run.py train      # train  •  validate  •  serve  •  all
```
</details>

Configure the region, horizon, and feeds in [`config.yaml`](config.yaml).
Set `FIRMS_MAP_KEY` (free) to enable NASA active-fire detections.

## The trained flood ML model

`./run.ps1 train` assembles labels automatically from open data (USGS gages
exceeding flood stage — no manual downloads, no API key), trains a gradient-
boosting model, and cross-validates it. Once trained, **every run uses it
automatically** (the summary line shows `flood=TRAINED(ROC-AUC 0.949)`).

Latest trained result (out-of-fold, 22 gages, 11.5k examples, 12.6% positive):

| metric | trained | baseline |
|---|---|---|
| ROC-AUC | **0.949** | 0.944 (heuristic) |
| PR-AUC | **0.731** | 0.126 (no-skill) |
| Brier | **0.085** | — |

The trained model and its metrics live in
[`cascadia/models_store/`](cascadia/models_store/). Delete the `.joblib` file to
fall back to the heuristic. Same recipe extends to the other hazards.

## Validation

Because the pipeline runs for any past date (Open-Meteo ERA5 archive + USGS FDSN
catalog), it replays documented disasters and checks that the *realized* hazards
spike at the impacted locations vs. a calm control window. Current results:

| Event | Realized | P(realized) control → event | Lift |
|---|---|---|---|
| Dec 2007 Chehalis flood (WA) | flood + landslide | 0.09 → **1.00** | **+0.91** |
| Nov 2021 Nooksack flood (WA) | flood + landslide | 0.34 → **1.00** | **+0.66** |

See [METHOD.md](METHOD.md) for the full method, math, and limitations.

## Roadmap (phased)

- [x] **Phase 1** — research pipeline on live open feeds + risk map
- [x] Land mask, FIRMS feed, historical "as-of" mode
- [x] Backtesting/validation harness (replay past events, score vs. realized)
- [x] Real-time monitoring + threshold alerting
- [x] **Phase 2** — FastAPI service + Leaflet map dashboard
- [x] **Trained flood predictor** with GroupKFold ROC/PR-AUC/Brier scoring
- [ ] Train the remaining leaves (landslide, wildfire) on USGS landslide
      inventory / NWS Storm Events / MTBS perimeters
- [ ] Sentinel-1 InSAR ground-deformation & soil-moisture rasters
- [ ] Exposure/vulnerability layer (OSM infrastructure, population) → expected loss

## Layout

```
cascadia/
  sources/      open-feed adapters (USGS quakes, NWS, Open-Meteo, NWIS, FIRMS)
  features/     grid (+ land mask) + cross-sector indicator fusion
  models/       per-hazard predictors (+ trained.py loader) + the cascade graph
  training/     dataset.py (auto-labelled) + train_flood.py (trains & scores)
  models_store/ saved trained models (*.joblib) + metrics
  pipeline.py   ingest → fuse → predict → cascade
  validate.py   historical event replay + backtest scoring
  monitor.py    live monitoring + threshold alerts
  api.py        FastAPI service  •  static/index.html  dashboard
  viz.py        folium risk map  •  __main__.py  CLI
```

*Phase 1 prototype. Predictions are experimental and not a substitute for
official warnings from USGS, NWS, or local emergency management.*
