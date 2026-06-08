# Model cards

One card per predictor: what it's for, how it works, how well it works, and —
critically — what it is *not*. "Kind" distinguishes a **calibrated probability**
from a **relative 0–1 index** (area‑scale danger, not a calibrated occurrence
probability). All are **research prototypes; defer to official agencies.**

---

## Flood
- **Kind:** calibrated probability ✅
- **Intended use:** relative 7‑day riverine‑flood likelihood per ~5 km cell.
- **Inputs:** forward‑window precipitation, peak soil moisture, antecedent
  streamflow anomaly (USGS NWIS + IsolationForest).
- **Method:** gradient‑boosting classifier, **isotonic‑calibrated**. Labels =
  USGS gage exceeding its 95th‑percentile stage within the window.
- **Validation:** out‑of‑fold, GroupKFold **by gage** (no spatial leakage):
  **ROC‑AUC 0.95, Brier 0.055, Brier skill score +0.51**, reliability on the
  diagonal (`run.ps1 skill`). **Independent event test** (`run.ps1 performance`):
  scored against **100 NWS Storm Events floods + 100 matched non‑events**
  (2018–2021, labels independent of training) → **ROC‑AUC 0.715, 95% CI
  [0.64, 0.78]**; **56% hit rate at 22% false alarm**; riverine floods 0.74 vs
  flash floods 0.70. (Out‑of‑fold AUC is higher because those gages/labels match
  the training distribution; the event test is the harder, independent number.)
  **Lead‑time** (`run.ps1 leadtime`): skill (AUC ≈ 0.65–0.67) persists to the
  7‑day forecast horizon, then decays to ≈0.54 by 14 days as the event leaves the
  window — leaving only antecedent soil/streamflow.
- **Limitations:** gage‑exceedance ≠ damaging flood; ~5 km resolution; trained on
  a Pacific‑NW gage sample (national retrain pending); no pluvial/coastal flood.

## Earthquake
- **Kind:** probability (physically derived) ✅
- **Intended use:** background seismic hazard over the horizon + aftershock lift.
- **Inputs:** USGS catalog (M≥4, 1970–), recent events.
- **Method:** smoothed‑seismicity Gaussian kernel → Poisson probability; Omori/
  Utsu aftershock term near recent mainshocks.
- **Validation:** spatial pattern matches the USGS seismic‑hazard map (CA, PNW,
  Intermountain West, New Madrid); per‑cell weekly P is correctly tiny.
- **Limitations:** **earthquakes are not short‑term predictable** — this is a
  long‑run rate, *not* an imminent‑quake forecast.

## Wildfire
- **Kind:** relative index ⚠️
- **Intended use:** area‑scale fire‑weather danger (not "this lot will burn").
- **Inputs:** GRIDMET Burning Index, Energy Release Component, 100‑hr fuel
  moisture (+ live FIRMS detections, red‑flag warnings).
- **Method:** operational NFDRS fire‑danger variables → conservative,
  ignition‑limited score; observed fire dominates when present.
- **Validation:** **independent‑event test** (`run.ps1 fireperf`): the danger
  mapping scored against **80 real NASA FIRMS satellite fire location‑days + 80
  matched non‑events** (2018–2021) → **ROC‑AUC 0.938, 95% CI [0.90, 0.97]**;
  **92% hit rate at 15% false alarm**. FIRMS (satellite) is independent of GRIDMET
  (reanalysis). (Met‑only *occurrence* ML had ~no transferable skill — see
  history — which is why the leaf uses established NFDRS indices instead.)
- **Limitations:** danger is **diagnostic, not a multi‑day forecast**; danger ≠
  occurrence (ignition still required); no ignition or fuel‑type model; the AUC
  reflects discrimination during/around fire season, not lead‑time skill.

## Landslide
- **Kind:** relative index ⚠️
- **Intended use:** rainfall‑triggered landslide susceptibility.
- **Inputs:** USGS landslide‑inventory density (susceptibility), **DEM slope**,
  rainfall + soil saturation (trigger).
- **Method:** susceptibility × slope_factor × rainfall trigger. Parcel reports
  use the address's **local ~250 m slope** (a flat lot reads as stable).
- **Validation:** components are physically grounded; not event‑validated
  (dated‑inventory labels are sparse).
- **Limitations:** ~5 km gridded slope is coarse; no soil/geology layer.

## Heat
- **Kind:** relative index ⚠️
- **Intended use:** dangerous‑heat potential.
- **Inputs:** temperature, humidity → **heat index** and **wet‑bulb temperature**.
- **Method:** NWS heat‑index (Rothfusz) + Stull wet‑bulb → noisy‑OR of danger.
- **Validation:** physically standard thresholds; not mortality‑validated.
- **Limitations:** no acclimatization/vulnerability; not a health forecast.

## Wildfire smoke
- **Kind:** relative index ⚠️ — **but the cascade is validated** ✅
- **Intended use:** downwind smoke / air‑quality potential.
- **Inputs:** FIRMS fire radiative power, wind direction.
- **Method:** plume‑transport proxy — fire carried downwind toward the cell.
- **Validation:** the **fire→smoke cascade adds significant skill** vs an
  independent (proximity) baseline against EPA‑observed PM2.5 across 4 episodes
  (Δr = +0.063, 95% bootstrap CI [+0.008, +0.117]; `run.ps1 skill`).
- **Limitations:** same‑day transport proxy (no multi‑day dispersion/injection);
  absolute level is a relative index, not calibrated µg/m³.

---

## ENSO (ONI) forecast — supporting model
- **Kind:** calibrated regression ✅
- **Intended use:** forecast the Oceanic Niño Index 1–3 months ahead.
- **Inputs:** recent ONI trajectory + seasonality.
- **Method:** gradient‑boosting; honest split (train ≤2012, test >2012).
- **Validation:** **beats persistence ~30%** at every lead (RMSE −31/−28/−27%,
  corr 0.99/0.95/0.89).
- **Limitations:** spring predictability barrier; ONI only (no full SST field).

## ENSO seasonal hazard outlook — honest negative
- **Kind:** weak guide (composite) ⚠️
- **Validation:** against NCEI observed climate, ENSO shows moderate correlations
  (SE/S.Plains winter precip r≈0.4) but **near‑zero out‑of‑sample tercile skill**
  (RPSS≈0). Labeled accordingly; **not** a skillful seasonal forecast.
