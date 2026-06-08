# 🌎 Cascadia — a compound & cascading multi-hazard engine

[![tests](https://github.com/shawnatwsu/cascadia/actions/workflows/tests.yml/badge.svg)](https://github.com/shawnatwsu/cascadia/actions/workflows/tests.yml)
[![license: MIT](https://img.shields.io/badge/license-MIT-blue.svg)](LICENSE)

> Most hazard systems answer *"what is the flood risk?"* — one hazard at a time,
> in isolation. **Cascadia** answers a harder, more useful question:
> **"given conditions right now, which hazards are likely, where, who is exposed,
> and how might one hazard trigger another?"**
>
> It fuses **100% free, open data** into a probabilistic **cascade graph** across
> **six hazards**, scales from a single address to the whole United States, and —
> importantly — reports its **measured forecast skill** honestly.

The name is a double meaning: the **Cascadia** bioregion (Pacific Northwest)
*and* the **cascading**-hazard idea at the core of the model.

> ⚠️ **Research prototype — not for operational or life-safety decisions.** Always
> defer to official sources (NWS, USGS, FEMA, local emergency management).

---

## What it does

- **6 hazards** on one frame: **flood · earthquake · wildfire · landslide · heat · wildfire-smoke**
- **Cascade graph** — hazards can trigger one another (quake→landslide→flood, fire→smoke, heat→fire), with **condition-gated** edges (a quake is far likelier to start a landslide on a *rain-saturated* slope).
- **Any scale** — a single **address**, an **NCA5 region** (Northwest, Southeast, …), or the **whole CONUS** at 4 km.
- **Expected impact** — not just probability, but **expected people affected** (hazard × population).
- **Three forecast horizons** — a **7‑day** live forecast, a **weeks 2–6** sub‑seasonal outlook, and an **ENSO‑driven seasonal** outlook (with a real ENSO‑forecast ML model).
- **Measured skill, stated honestly** — not just reliability diagrams, but **independent‑event detection scores**: the flood model is tested on **100 real NWS floods** (ROC‑AUC **0.715**), the wildfire leaf on **80 satellite‑observed fires** (ROC‑AUC **0.938**), the fire→smoke **cascade** against EPA PM2.5 — each with bootstrap confidence intervals, matched non‑events, and labels *independent* of training. Where skill is weak (the seasonal outlook), we say so.

---

## Gallery

### Maps, nowcasts & forecasts

**National multi‑hazard nowcast** (`run.ps1 conditions conus`) — six hazards on one frame, 4 km, Albers equal‑area, each panel with its own scale:

![CONUS multi-hazard conditions](docs/conus_conditions.png)

**Expected human impact** (`run.ps1 impact conus`) — hazard probability × Census population = expected people affected:

![CONUS impact](docs/impact_conus.png)

**ENSO‑driven seasonal outlook** (`run.ps1 seasonal`) — fire/drought, wet/flood, and heat anomalies for the next 1–3 months, driven by the current ENSO state (and optionally an N‑month ENSO forecast). Diverging scales, honestly labeled — note the **weak Neutral ENSO** caption, not a hyped signal:

![ENSO seasonal outlook](docs/seasonal_outlook.png)

**Address‑level report** (`run.ps1 parcel "..."`) — a locator map + per‑hazard levels for any US address. Flood & earthquake are **calibrated probabilities**; the `*` hazards are **relative 0–1 indices** (area‑scale danger, not address‑specific odds). Landslide is refined by the address's **local DEM slope** (flat lot → stable), and the weather‑driven hazards carry **error bars** = 10–90% across **31 GFS ensemble members** (real forecast uncertainty):

![Parcel report](docs/parcel_report.png)

### Validation & testing — measured skill, honestly

> Every figure below is produced from open data by a one‑line command, with
> **matched non‑events**, **bootstrap confidence intervals**, and labels that are
> **independent of training** wherever possible. See the consolidated
> [validation table](#validation--forecast-skill) below.

**Flood model is calibrated** (`run.ps1 skill`) — out‑of‑fold (GroupKFold *by gage*, no spatial leakage), the reliability curve sits on the diagonal (a "0.6" verifies ~0.6); ROC‑AUC 0.95, Brier skill +0.51:

![Flood calibration](docs/flood_calibration.png)

**A defensible performance number — on 100 *independent* flood events** (`run.ps1 performance`) — beyond anecdotes, we score the **full operational flood model** (ERA5 precip + soil + real antecedent streamflow from the nearest USGS gage) on **100 real NWS Storm Events floods (2018–2021) + 100 matched non‑events** (same locations, shifted calm dates). Labels come from the NWS event database — *independent* of the gage‑exceedance labels the model was trained on. Result: **ROC‑AUC 0.715, 95% bootstrap CI [0.64, 0.78]** (excludes 0.5 — real, significant discrimination), flagging **56% of real floods at a 22% false‑alarm rate**. Riverine floods (the model's design target) score higher (AUC 0.74) than flash floods (0.70), as expected for a daily‑resolution model:

![Flood performance on independent events](docs/flood_performance.png)

**How far ahead is the warning?** (`run.ps1 leadtime`) — scoring the same events as if the forecast were issued *N days early* shows the flood signal stays detectable (**AUC ≈ 0.65–0.67**) out to the model's **7‑day horizon**, then decays toward chance at 10–14 days as the event moves beyond the forecast window and only antecedent soil/streamflow remain. Exactly the shape a 7‑day model should have:

![Flood lead-time skill](docs/flood_leadtime.png)

**Wildfire holds up too — on independent satellite fires** (`run.ps1 fireperf`) — the wildfire leaf (GRIDMET NFDRS Burning Index + Energy Release Component + 100‑hr fuel moisture) is scored against **80 real NASA FIRMS fire location‑days + 80 matched non‑events**. FIRMS (satellite thermal) is *independent* of GRIDMET (reanalysis weather), so this is a genuine verification: **ROC‑AUC 0.938, 95% CI [0.90, 0.97]**, flagging **92% of real wildfires at a 15% false‑alarm rate**. (Fire danger is *diagnostic* of fire‑prone conditions, not a multi‑day‑ahead forecast — and it's seasonal, so this is the discrimination, not a free lunch.)

![Wildfire performance on independent FIRMS fires](docs/fire_performance.png)

**The cascade *adds skill* — significantly** (`run.ps1 skill`) — validated against EPA‑measured PM2.5 across **4 smoke episodes** (2018–2023, West + cross‑border Canadian smoke; n = 1,131 monitor‑days). Modeling the fire→smoke **downwind transport** (the cascade) ~doubles the correlation vs fire **proximity** (treating smoke as independent): Δr = +0.063, **95% bootstrap CI [+0.008, +0.117]** (excludes zero). Direct evidence for the project's central claim:

![Fire→smoke cascade validation](docs/cascade_skill.png)

**Would it have flagged real events at real addresses?** (`run.ps1 hindcast`) — run the engine *historically* at addresses that actually experienced a hazard, vs a calm control date. It gave a **72% flood probability** at a Chehalis WA address the week of the 2007 flood (≈0 in calm August), flagged the Camp Fire and Almeda Fire areas, and stayed near zero at a safe inland address — warns when it should, quiet when it shouldn't:

![Parcel hindcast](docs/parcel_hindcast.png)

**Honest skill — ENSO is a *weak* US seasonal predictor** — moderate correlations (top), but near‑zero out‑of‑sample tercile skill (bottom). We show this rather than overstate it:

![ENSO teleconnection skill](docs/enso_skill.png)

---

## Quick start

**Requirements:** Python 3.10+ on Windows/macOS/Linux.

```powershell
git clone https://github.com/shawnatwsu/cascadia
cd cascadia
```

**Windows (easiest):** double‑click **`run.bat`**, or in a terminal:
```powershell
.\run.ps1                 # first run sets up the venv + installs deps, then runs
```

**Any OS (manual):**
```bash
python -m venv .venv
.venv/Scripts/activate         # Windows  (use: source .venv/bin/activate on mac/linux)
pip install -r requirements.txt
python run.py                  # = the live forecast map
```

**Optional — wildfire smoke & live fire detections** need a free NASA FIRMS key
(1‑minute signup at <https://firms.modaps.eosdis.nasa.gov/api/map_key/>):
```powershell
setx FIRMS_MAP_KEY "your_key_here"     # then reopen the terminal
```
Without it, smoke/observed‑fire simply read 0; everything else works.

> All generated maps are written to the **`outputs/`** folder, and the full path
> is printed each time.

---

## Commands

Run any of these as `.\run.ps1 <command>` (Windows) or `python run.py <command>`.

| Command | What you get |
|---|---|
| `run.ps1` *(no arg)* | **Live 7‑day forecast** for the local region (Open‑Meteo) |
| `run.ps1 conditions <region>` | **4 km hazard nowcast** for a region (GRIDMET) — all 6 hazards |
| `run.ps1 impact <region>` | **Expected people affected** (hazard × Census population) |
| `run.ps1 subseasonal <region>` | **Weeks 2–6** fire/drought/heat outlook (land‑memory) |
| `run.ps1 seasonal [1‑3]` | **ENSO seasonal outlook**; optional N‑month ENSO forecast lead |
| `run.ps1 parcel "<address>"` | **Address‑level** hazard report (JSON + a one‑page map) |
| `run.ps1 hindcast` | **Does it work?** Replays real hazard events at their addresses |
| `run.ps1 performance` | **How well?** Scores 100 independent NWS floods → ROC‑AUC, hit/false‑alarm |
| `run.ps1 leadtime` | **How far ahead?** Flood AUC vs forecast lead (1–14 days) |
| `run.ps1 fireperf` | **Wildfire skill** vs 80 independent FIRMS fires → ROC‑AUC (needs `FIRMS_MAP_KEY`) |
| `run.ps1 skill` | **Validation suite** → reliability/skill figures in `outputs/` |
| `run.ps1 validate` | Replay documented disasters (2007 & 2021 PNW floods) |
| `run.ps1 train [flood\|fire]` | (Re)train an ML predictor + print its scorecard |
| `run.ps1 serve` | Interactive **Leaflet dashboard** (opens browser) |

**Regions** (`<region>`): `conus`, plus the NCA5 regions
`northwest`, `southwest`, `northern_great_plains`, `southern_great_plains`,
`midwest`, `southeast`, `northeast` — and `pnw`, `california`.

```powershell
# examples
.\run.ps1 conditions conus
.\run.ps1 impact southeast
.\run.ps1 seasonal 3
.\run.ps1 parcel "1300 Franklin St, Vancouver, WA"
.\run.ps1 skill
```

> First run of a big region (`conus`, large NCA regions) takes a few minutes to
> fetch data; afterwards it's cached and fast.

---

## How it works

```
open feeds ─▶ spatial grid ─▶ cross-sector ─▶ per-hazard ─▶ CASCADE GRAPH ─▶ compound
(no API key)   (CONUS-clipped)  indicators      predictors    (noisy-OR over     + impact
                                 (fusion)        (ML + physics) gated triggers)   surface
```

1. **Ingest** open feeds onto a common grid (ocean/Mexico/Canada clipped out).
2. **Per‑hazard predictors** — a mix of trained ML and physically‑grounded models:
   - **flood** → trained gradient‑boosting model (isotonic‑calibrated)
   - **earthquake** → USGS smoothed‑seismicity Poisson prior + aftershocks
   - **wildfire** → GRIDMET fire‑danger (Burning Index / ERC / fuel moisture)
   - **landslide** → USGS landslide‑inventory susceptibility × rainfall trigger
   - **heat** → heat index + wet‑bulb temperature
   - **smoke** → downwind plume transport from FIRMS fires + wind
3. **Cascade graph** — probabilities propagate through trigger edges (noisy‑OR),
   producing a compound‑risk surface and the dominant **cascade chain** per cell.
4. **Exposure** — multiply by Census population → **expected people affected**.

---

## Validation & forecast skill

> **measured, not assumed.** Reproduce any row with the listed command (figures
> saved to `outputs/`). We report the *hard* numbers — out‑of‑distribution, with
> matched controls and bootstrap CIs — and flag where skill is weak.

### How we test

- **Independent labels.** Where possible, the *truth* comes from a different
  dataset than the model trained on — NWS Storm Events floods, NASA FIRMS
  satellite fires, EPA PM2.5 — so a good score can't be memorization.
- **Matched non‑events.** Every real event is paired with the *same location on a
  shifted calm date*, so the metric reflects discrimination (event vs non‑event),
  not just "did something bad happen in a risky place."
- **Out‑of‑fold / out‑of‑distribution.** The flood model's in‑sample skill is
  measured with **GroupKFold by gage** (no spatial leakage); its *headline* claim
  is the harder, fully independent NWS event test.
- **Confidence intervals.** Discrimination (ROC‑AUC) and the cascade effect carry
  **1,000‑sample bootstrap 95% CIs**; we only claim an effect when the CI excludes
  the no‑skill value.

### Results

| Question it answers | Command | Data & method | Sample | Result | Honest caveat |
|---|---|---|---|---|---|
| **Are the flood probabilities calibrated?** | `skill` | Out‑of‑fold (GroupKFold by gage), isotonic | 11.5k pt‑dates, 22 gages | ROC‑AUC **0.95**, Brier **0.055**, BSS **+0.51**, reliability on diagonal | In‑*distribution* (gages resemble training) |
| **Does it catch *real, independent* floods?** | `performance` | Full model (incl. real USGS streamflow) vs **NWS Storm Events** + matched non‑events | 100 floods + 100 non‑events | ROC‑AUC **0.715** (95% CI **0.64–0.78**); **56%** hit @ **22%** false alarm | Daily ~5 km grid under‑resolves flash floods (riverine 0.74 > flash 0.70) |
| **How many days ahead is the flood signal there?** | `leadtime` | Same events scored as if issued *L* days early (forward 7‑day window) | 80 events × 7 leads | AUC **≈0.66** to **7 days**, decays to **0.54** by 14 days | ERA5 = a *perfect* precip forecast → this is *potential* lead skill |
| **Does the wildfire leaf track real fire?** | `fireperf` | GRIDMET NFDRS danger vs **FIRMS** satellite fires + matched non‑events | 80 fire‑days + 80 non‑events | ROC‑AUC **0.938** (95% CI **0.90–0.97**); **92%** hit @ **15%** false alarm | Danger is *diagnostic*, not a multi‑day forecast; seasonal |
| **Does the cascade *add* skill?** *(the core claim)* | `skill` | Fire→smoke **transport** vs fire **proximity**, vs EPA PM2.5 | 1,131 monitor‑days, 4 episodes | Δr **+0.063** (95% CI **+0.008–+0.117**, excludes 0) | Correlational; 4 episodes |
| **Can it forecast ENSO itself?** | `skill` | Gradient‑boosting ONI forecast vs persistence | 1950– monthly | Beats persistence **~30%** (RMSE) at +1/+2/+3 mo | A single climate index |
| **Does ENSO usefully predict US seasons?** | `skill` | ENSO → regional climate vs NCEI observed | 1950– | r≈0.4 (SE/S. Plains winter precip), **RPSS≈0** | **Weak** US predictor — labeled as such, not hidden |
| **Would it have flagged famous disasters?** | `hindcast` | Historical replay at real addresses vs calm control | 5 events | **5/5** behave correctly (high at event, low at control) | Anecdotal sanity check, not a skill score |

The flood/wildfire numbers answer **different questions** and aren't directly
comparable: **0.715** is a true *7‑day‑ahead forecast* of riverine flooding, while
**0.938** is *same‑week danger discrimination* (the fire index is diagnostic, not
predictive). Both are honest; we label them that way throughout.

The engine also backtests against the **Dec 2007 Chehalis** and **Nov 2021
Nooksack** floods (`run.ps1 validate`).

---

## Data sources (all free / open)

| Source | Used for | Key? |
|---|---|---|
| USGS earthquakes + FDSN catalog | seismic hazard, aftershocks | no |
| USGS NWIS streamflow | flood | no |
| USGS Landslide Inventory | landslide susceptibility | no |
| NWS/NOAA alerts | official warnings | no |
| Open‑Meteo (forecast + ERA5) | live 7‑day forecast | no |
| GRIDMET (4 km, OPeNDAP) | fire/heat conditions, sub‑seasonal | no |
| NASA FIRMS | active fire, smoke, **wildfire validation** | **free key** |
| US Census (population, geocoder) | exposure, parcel lookup | no |
| NOAA CPC ONI + NCEI nClimDiv | ENSO + validation | no |
| NOAA NCEI Storm Events | **independent flood‑event labels** | no |
| EPA AQS PM2.5 | **cascade (smoke) validation** | no |

---

## Project layout

```
cascadia/
  sources/      open-feed adapters (USGS, NWS, Open-Meteo, GRIDMET, FIRMS, ENSO, Census…)
  features/     grid (+ CONUS clip) + cross-sector indicator fusion
  models/       per-hazard predictors + the cascade graph + trained-model loader
  training/     dataset builders + ML training (flood, fire, ENSO)
  geo.py        CONUS / NCA5 region geometry + masking
  conditions.py GRIDMET regional nowcast      impact.py  exposure × hazard
  subseasonal.py weeks 2-6 outlook            seasonal.py ENSO seasonal outlook
  skill*.py     calibration + cascade skill    cartomap.py  publication-style maps
  validation_scaled.py  independent flood-event ROC + lead-time (NWS Storm Events)
  validation_fire.py    independent wildfire ROC (FIRMS)
  parcel.py     address-level report           parcel_hindcast.py  real-event replay
  validate.py   historical-event backtests     api.py + static/    dashboard
run.py / run.ps1 / run.bat   one-command launcher
```

---

## Honest limitations

- **Research prototype.** Not validated for operational use; defer to official agencies.
- **Calibrated vs. index:** only **flood** (and the **earthquake** seismic prior) are calibrated probabilities. **Landslide / wildfire / heat / smoke are relative 0–1 hazard indices** — area‑scale danger, *not* calibrated odds of occurrence. Reports flag them with `*`.
- Maps are **~4–5 km cells**; the parcel report refines landslide with a local DEM slope, but other hazards remain area‑scale.
- The **seasonal outlook** has near‑zero out‑of‑sample probabilistic skill (ENSO is a weak US predictor) — it's a labeled weak guide.
- GRIDMET/Census are **US‑only**; live forecasting is point‑sampled (region size is bounded by API limits).

## Roadmap

- **Calibrate the wildfire danger→probability mapping** (it now *ranks* fire days
  well — AUC 0.938 — but the 0–0.6 score isn't a calibrated probability like flood's)
- Independent‑event tests for the remaining leaves (landslide, heat) as clean labels allow
- Full SST fields (OISST) + learned ENSO→regional teleconnections
- Sentinel‑1 InSAR / soil moisture; tsunami + volcano (lahar) cascades
- Gridded forecast (NDFD/GFS) for CONUS‑wide *forward* forecasts; hosted demo

---

## Reproducibility & provenance

- **Tests:** `pytest tests/` — offline unit tests for the cascade math, skill
  metrics, and predictor logic (run in CI on Python 3.10–3.12).
- **Pinned environment:** `requirements.txt` (loose) + `requirements.lock` (exact).
- **Data provenance:** every source, access method, and license is documented in
  [DATA_SOURCES.md](DATA_SOURCES.md). Validation data are static archives, so the
  `run.ps1 skill` results are reproducible.
- **Model cards:** per‑hazard intended use, method, skill, and limitations in
  [MODEL_CARDS.md](MODEL_CARDS.md) — including which outputs are *calibrated
  probabilities* vs *relative indices*.
- **Uncertainty:** weather‑driven hazards report a 10–90% interval from the
  31‑member GFS ensemble (forecast uncertainty), not just a point estimate.
- **Citation:** see [CITATION.cff](CITATION.cff).

## License

[MIT](LICENSE) — free to use, modify, and build on. Contributions welcome.

*Built with open data and a lot of honesty about what it can and can't do.*
