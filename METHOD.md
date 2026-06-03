# Cascadia: a probabilistic cascade-graph method for compound natural-hazard prediction

*Method note — Phase 1 prototype, 2026-06*

## 1. Problem

Operational hazard systems are overwhelmingly **single-hazard and
single-sector**: one model for earthquakes (USGS), another for weather (NWS),
another for fire (FIRMS). They treat hazards as statistically independent and
ignore two facts that dominate real disaster losses:

1. **Hazards trigger other hazards.** An earthquake shakes a saturated slope
   into a landslide; the landslide dams a river; the dam fails and floods
   downstream. A wildfire strips a hillside; the next storm turns the burn scar
   into a debris flow. These *cascades* are where compounding loss comes from.
2. **The same hazard means different things under different conditions.** A M5
   quake on a dry slope is minor; on a rain-saturated slope it is a landslide
   generator. Triggering is **state-dependent**.

Cascadia is a method for predicting the *chain*, not the link — using only
open, mostly no-API-key data.

## 2. Method

### 2.1 Common spatial frame
The region is divided into a regular ~0.1° grid (~11 km cells); ocean cells are
removed with a bundled land mask. Every data source is normalized and fused onto
this frame (`features/indicators.py`).

### 2.2 Cross-sector indicators
Per cell, from open feeds:

| Indicator | Source | Sector |
|---|---|---|
| recent seismicity (max M) | USGS quakes / FDSN catalog | seismic |
| cumulative precip, peak soil moisture | Open-Meteo forecast / ERA5 archive | hydro-met |
| streamflow anomaly | USGS NWIS + `IsolationForest` | hydrology |
| active-fire detections | NASA FIRMS *(optional key)* | fire |
| official warnings | NWS active alerts | civil |

The streamflow anomaly is genuine unsupervised ML: an `IsolationForest` learns
each gage's "normal" (level, rate-of-rise) from its own recent record and scores
the latest reading — no labels required.

### 2.3 Base predictors (the per-hazard "leaves")
Each hazard `h` has a predictor mapping a cell's indicators to
`P_base(h)` = probability `h` *initiates* in the cell over the horizon, before
any triggering. Every predictor shares one interface; a trained model in
`models_store/{hazard}_model.joblib` transparently overrides that hazard's
calibrated-sigmoid heuristic — this is the **hybrid** seam.

**Trained flood leaf.** Labels are assembled automatically (`training/`): for
each USGS gage and each sampled date, a flood = gage height exceeding its
95th-percentile stage within the 7-day window; features are the *same three* the
live engine computes (forward-window precip, peak deep-soil-moisture, antecedent
streamflow anomaly), so there is no train/serve mismatch. A
`HistGradientBoostingClassifier` is scored **out-of-fold with GroupKFold by
gage** (no gage in both train and test). Result on 11.5k examples / 22 gages
(12.6% positive): **ROC-AUC 0.949** (heuristic 0.944), **PR-AUC 0.731**
(no-skill 0.126), **Brier 0.085**. The gain over the heuristic is modest on ROC
but the model is data-driven, calibrated, reproducible, and the recipe extends
to the other leaves.

### 2.4 The cascade graph (the contribution)
Hazards are nodes of a directed acyclic graph; edges are physical trigger
pathways. Each edge `p → c` carries:

- a **base transfer probability** `w` — "if `p` happens, how often does it set
  off `c`, all else equal", and
- a **condition gate** `g(cell) ∈ [0,1]` that scales `w` by the cell's live
  state (e.g. the earthquake→landslide gate rises with soil saturation).

Default DAG:

```
earthquake ──(w .45, gate↑saturation)──▶ landslide ──(w .30, gate↑streamflow)──▶ flood
earthquake ──(w .12, gate↑streamflow)─────────────────────────────────────────▶ flood
wildfire   ──(w .35, gate=precip)──────▶ landslide        (post-fire debris flow)
```

Probabilities propagate in topological order via **noisy-OR**, fusing a node's
own initiation with every upstream trigger:

```
P(c) = 1 − (1 − P_base(c)) · Π over parents p (1 − P(p)·w(p→c)·g(cell))
```

Two readouts per cell:
- **dominant trigger chain** — walk back from the highest-probability hazard,
  following an edge *only* when the parent's contribution exceeds the child's
  own base (so rain-initiated flooding is reported as `flood`, not falsely
  attributed to an unrelated quake);
- **co-occurring set** — all hazards with `P ≥ 0.5` (the "compound" view).

Compound risk per cell = noisy-OR over all hazards = `P(any hazard)`.

## 3. Validation

Because the pipeline can run for any past date (Open-Meteo ERA5 archive + USGS
FDSN catalog), we replay documented disasters and test whether the realized
hazards' probability rises at the impacted locations relative to a **calm
control window** in the same place. This is a *conditions-consistency* test, not
a forecast-skill test (which needs archived forecasts — see §4).

| Event | Realized | P(realized) control → event | Lift |
|---|---|---|---|
| Dec 2007 Chehalis River flood (Lewis Co., WA) | flood + landslide | 0.09 → **1.00** | **+0.91** |
| Nov 2021 Nooksack/Whatcom flood (WA) | flood + landslide | 0.34 → **1.00** | **+0.66** |

In both cases the dominant chain and co-occurring set correctly recover
`flood + landslide` — and crucially *not* wildfire — from the meteorological
conditions alone. (`cascadia/validate.py`, `backtest()`.)

## 4. Limitations & next steps

- **Only the flood leaf is trained so far.** Landslide, wildfire and earthquake
  remain calibrated heuristics. Next: train them on USGS landslide inventory,
  MTBS fire perimeters, and aftershock statistics, each with held-out skill.
- **Conditions-consistency ≠ forecast skill.** A true test needs archived
  *forecasts* as input and a proper hit/false-alarm scoring (ROC/Brier) over
  many event and non-event dates.
- **Edge weights and gates are expert priors.** They can be learned from
  co-occurrence statistics in multi-hazard catalogs.
- Compound risk conflates seasonally-different hazards; per-hazard surfaces are
  sharper (the validator already isolates the targeted hazards).
- No exposure/vulnerability layer yet (OSM infrastructure, population) — risk is
  hazard probability, not expected loss.

## 5. Why it is novel

The combination is, to our knowledge, not in any open system: a **condition-
gated probabilistic cascade graph** over **fused cross-sector open feeds**,
producing **explainable hazard chains** and runnable both live and historically
for backtesting — on a laptop, with no paid data.
