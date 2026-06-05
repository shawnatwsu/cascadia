# Data sources & provenance

Cascadia uses only **free, open** data. This document records each source, what
it provides, how it's accessed, its update cadence, and its terms — so every
output is traceable to its inputs (a requirement for agency / scientific use).

> Access is **live at run time** and **cached locally** under `data/cache/`.
> "Version/as-of" is therefore the data available on the run date; provenance
> lines on each map state the GRIDMET window used.

| Source | Provider | Variables / product | Access | Key? | Cadence | Terms |
|---|---|---|---|---|---|---|
| Earthquakes (real-time) | USGS | M≥ feed (all_week) | GeoJSON feed | no | minutes | public domain |
| Earthquake catalog | USGS | FDSN event query (M≥4, 1970–) | REST (GeoJSON) | no | continuous | public domain |
| Streamflow | USGS NWIS | discharge / gage height (00060/00065) | REST (JSON), daily values | no | sub-daily | public domain |
| Landslide inventory | USGS | US Landslide Inventory (points) | ArcGIS FeatureServer | no | periodic | public domain |
| Alerts | NOAA / NWS | active alerts (flood, fire-weather, heat…) | api.weather.gov | no | minutes | public domain |
| Live forecast | Open-Meteo | precip, temp, RH, wind, soil moisture (7-day) | REST | no | hourly | CC-BY 4.0 |
| Reanalysis (historical) | Open-Meteo / ERA5 | archive of the above | REST | no | daily | CC-BY 4.0 |
| Gridded met (4 km) | GRIDMET (U. Idaho) | precip, tmax/tmin, RH, wind, VPD, burning index, ERC, fuel moisture, solar | OPeNDAP (THREDDS) | no | daily (~5-day lag) | public, cite Abatzoglou (2013) |
| Active fire | NASA FIRMS | VIIRS/MODIS detections (NRT + archive SP) | REST CSV | **free key** | sub-daily | NASA open data |
| Terrain | Open-Meteo (Copernicus DEM) | elevation → slope | REST | no | static | open |
| Population | US Census | county population estimates (2023) | static CSV | no | annual | public domain |
| County geometry | US Census | gazetteer (centroids, land area) | static ZIP | no | annual | public domain |
| Geocoding | US Census | address → lat/lon | REST | no | continuous | public domain |
| ENSO index | NOAA CPC | Oceanic Niño Index (ONI), 1950– | static ASCII | no | monthly | public domain |
| Observed climate | NOAA NCEI | nClimDiv statewide precip/temp, 1895– | static files | no | monthly | public domain |
| Observed PM2.5 | EPA AQS | daily 24-h PM2.5 (88101) | static yearly ZIP | no | annual files | public domain |
| Region definitions | US Global Change Research Program | NCA5 regions (state groupings) | encoded in `geo.py` | no | static | public |
| Boundaries | Natural Earth | US states (via cartopy) | bundled/CDN | no | static | public domain |

## Reproducibility notes

- **Determinism:** model training uses fixed `random_state`; bootstrap uses a
  fixed seed. Given the same cached inputs, outputs are reproducible.
- **Environment:** `requirements.txt` (loose) + `requirements.lock` (exact,
  `pip freeze`). Unit tests run on a minimal subset (see `.github/workflows`).
- **Caching:** delete `data/cache/` to force a fresh pull. GRIDMET cubes are only
  cached when *all* requested variables succeed (no partial/corrupt caches).
- **Validation data** (EPA AQS, NCEI, ONI) are static archives, so the skill
  results in `run.py skill` are reproducible.
