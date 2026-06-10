"""Marketing explainer: 'ask for HEAT at any scale, nowcast or 7-day forecast'.

Renders one publication-quality composite — four nested scales (CONUS -> Southwest
region -> California -> a Palm Springs parcel), two rows: NOWCAST (GRIDMET,
cached) and 7-DAY FORECAST (Open-Meteo). Above each column is the command you'd
actually type. Smooth filled-contour rendering so both rows read consistently.
All real data.
"""
from __future__ import annotations

import glob
import os
import sys
import time
from pathlib import Path

import numpy as np
import requests
import xarray as xr
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.colors import BoundaryNorm, ListedColormap
import cartopy.crs as ccrs
import cartopy.feature as cfeature

ROOT = Path(__file__).resolve().parent.parent          # repo root (scripts/ -> ..)
sys.path.insert(0, str(ROOT))
from cascadia.geo import region_bbox, region_geometry, state_geoms, conus_union
from cascadia.sources.gridmet import heat_index_c

CACHE = ROOT / "data" / "cache"
OUT = ROOT / "outputs" / "heat_explainer.png"


def _latest_cube() -> str:
    """Newest cached CONUS GRIDMET cube (run `python run.py conditions conus` first)."""
    files = sorted(glob.glob(str(CACHE / "gridmet_-125p0_24p5_-66p9_49p5_*.nc")),
                   key=os.path.getmtime)
    if not files:
        raise SystemExit("No cached CONUS GRIDMET cube found.\n"
                         "Run this first:  python run.py conditions conus")
    return files[-1]


CUBE = _latest_cube()
PARCEL = ("Palm Springs, CA", 33.83, -116.545)
LEVELS = np.array([18, 22, 26, 30, 34, 38, 42, 46], float)  # feels-like degC
OMURL = "https://api.open-meteo.com/v1/forecast"


# ---------------------------------------------------------------- data
def nowcast_grid():
    ds = xr.open_dataset(CUBE)
    tmax = ds["tmax_c"].max("time").to_numpy()
    rhmin = ds["rh_min"].min("time").to_numpy()
    return ds["lat"].to_numpy(), ds["lon"].to_numpy(), heat_index_c(tmax, rhmin)


def _om_get(params, retries=6):
    for attempt in range(retries):
        try:
            r = requests.get(OMURL, params=params, timeout=60)
            if r.status_code == 429:
                time.sleep(10 * (attempt + 1)); continue
            return r.json()
        except Exception:
            time.sleep(5 * (attempt + 1))
    return None


def _fetch_field(lat1d, lon1d, tag, pace=10):
    """7-day max apparent temperature on a lat/lon grid (Open-Meteo). Cached."""
    cache = str(CACHE / f"heat_fc_{tag}.npz")
    if os.path.exists(cache):
        z = np.load(cache); return z["lat"], z["lon"], z["field"]
    LON, LAT = np.meshgrid(lon1d, lat1d)
    pts = list(zip(LAT.ravel(), LON.ravel()))
    vals = np.full(len(pts), np.nan)
    for i in range(0, len(pts), 90):
        ch = pts[i:i + 90]
        data = _om_get({"latitude": ",".join(f"{p[0]:.3f}" for p in ch),
                        "longitude": ",".join(f"{p[1]:.3f}" for p in ch),
                        "daily": "apparent_temperature_max", "forecast_days": 7,
                        "timezone": "UTC"})
        if data is not None:
            locs = data if isinstance(data, list) else [data]
            for j, loc in enumerate(locs):
                v = loc.get("daily", {}).get("apparent_temperature_max")
                if v:
                    vals[i + j] = np.nanmax(v)
        else:
            print(f"  {tag} chunk {i} failed")
        time.sleep(pace)
    field = vals.reshape(LAT.shape)
    try:
        np.savez(cache, lat=lat1d, lon=lon1d, field=field)
    except Exception:
        pass
    return lat1d, lon1d, field


def forecast_conus():
    return _fetch_field(np.arange(25.0, 49.01, 1.0), np.arange(-124.5, -67.0, 1.0), "conus1deg")


def forecast_box(b, step, tag, pad=0.3, pace=8):
    lat1d = np.arange(b[1] - pad, b[3] + pad + 1e-6, step)
    lon1d = np.arange(b[0] - pad, b[2] + pad + 1e-6, step)
    return _fetch_field(lat1d, lon1d, tag, pace=pace)


def forecast_point(lat, lon):
    data = _om_get({"latitude": lat, "longitude": lon,
                    "daily": "apparent_temperature_max", "forecast_days": 7,
                    "timezone": "UTC"})
    try:
        return float(np.nanmax(data["daily"]["apparent_temperature_max"]))
    except Exception:
        return np.nan


# ---------------------------------------------------------------- render
def _clip(ax, cf, geom):
    """Clip filled contours to a geometry — smooth fill, clean border, no holes
    (NaN-masking before contouring punches gaps; clipping after does not)."""
    if geom is None:
        return
    from cartopy.mpl.patch import geos_to_path
    from matplotlib.path import Path
    from matplotlib.patches import PathPatch
    patch = PathPatch(Path.make_compound_path(*geos_to_path(geom)),
                      transform=ccrs.PlateCarree(), fc="none", ec="none")
    ax.add_patch(patch)
    try:
        cf.set_clip_path(patch)
    except Exception:
        for art in getattr(cf, "collections", []):
            try:
                art.set_clip_path(patch)
            except Exception:
                pass


def panel(ax, lat1d, lon1d, Z, extent, geom, cmap, norm, pin=None):
    LON, LAT = np.meshgrid(lon1d, lat1d)
    Zm = np.ma.masked_invalid(Z)
    ax.set_extent(extent, crs=ccrs.PlateCarree())
    cf = ax.contourf(LON, LAT, Zm, levels=LEVELS, cmap=cmap, norm=norm,
                     extend="both", transform=ccrs.PlateCarree())
    try:
        cf.set_edgecolor("face")  # avoid contour seams (mpl >=3.8)
    except Exception:
        for art in getattr(cf, "collections", []):
            art.set_edgecolor("face")
    _clip(ax, cf, geom)
    ax.add_feature(cfeature.STATES.with_scale("50m"), lw=0.45, edgecolor="0.35")
    ax.add_feature(cfeature.COASTLINE.with_scale("50m"), lw=0.5, edgecolor="0.3")
    ax.add_feature(cfeature.BORDERS.with_scale("50m"), lw=0.5, edgecolor="0.3")
    ax.spines["geo"].set_edgecolor("0.6"); ax.spines["geo"].set_linewidth(0.8)
    if pin is not None:
        plat, plon, val = pin
        ax.plot(plon, plat, marker="o", ms=7, mfc="white", mec="black", mew=1.5,
                transform=ccrs.PlateCarree(), zorder=6)
        ax.annotate(f"{val:.0f}°C", xy=(plon, plat), xycoords=ax.transData,
                    xytext=(0, 14), textcoords="offset points", ha="center",
                    fontsize=13, weight="bold", zorder=7,
                    bbox=dict(boxstyle="round,pad=0.25", fc="white", ec="0.3"))
    return cf


def main():
    nlat, nlon, nhi = nowcast_grid(); print("nowcast ready")
    pval_now = float(nhi[np.argmin(np.abs(nlat - PARCEL[1]))][np.argmin(np.abs(nlon - PARCEL[2]))])
    ca = state_geoms()["California"]
    swb = region_bbox("southwest", 0.3)
    cab = ca.bounds
    print("forecast: CONUS grid…");   flat, flon, ffield = forecast_conus()
    print("forecast: California…");   cala, calo, cafield = forecast_box(cab, 0.35, "ca035")
    print("forecast: parcel patch…"); plat, plon, pfield = forecast_box(
        (PARCEL[2]-1.5, PARCEL[1]-1.2, PARCEL[2]+1.5, PARCEL[1]+1.2), 0.15, "parcel015", pad=0.0, pace=6)
    pval_fc = forecast_point(PARCEL[1], PARCEL[2])      # exact valley-floor point
    if not np.isfinite(pval_fc):
        pval_fc = float(np.nanmax(pfield))
    print(f"parcel  now {pval_now:.0f}C  ->  7-day {pval_fc:.0f}C")
    scales = [
        ("CONUS",            (-125, -66.9, 24.5, 49.5),                       conus_union(),
         'run.ps1 conditions conus'),
        ("Southwest region", (swb[0], swb[2], swb[1], swb[3]),                region_geometry("southwest"),
         'run.ps1 conditions southwest'),
        ("California",       (cab[0]-0.4, cab[2]+0.4, cab[1]-0.4, cab[3]+0.4), ca,
         'run.ps1 conditions california'),
        ("Palm Springs parcel", (PARCEL[2]-1.7, PARCEL[2]+1.7, PARCEL[1]-1.3, PARCEL[1]+1.3), None,
         'run.ps1 parcel "Palm Springs, CA"'),
    ]
    base = plt.get_cmap("YlOrRd")
    cmap = ListedColormap(base(np.linspace(0.06, 1.0, len(LEVELS) - 1)))
    cmap.set_under(base(0.0)); cmap.set_over(base(1.0))
    norm = BoundaryNorm(LEVELS, cmap.N)

    fig = plt.figure(figsize=(17.5, 9.2))
    gs = fig.add_gridspec(2, 4, hspace=0.18, wspace=0.07,
                          left=0.055, right=0.985, top=0.785, bottom=0.165)
    rows = [("NOWCAST", "today · GRIDMET", nlat, nlon, nhi),
            ("7-DAY FORECAST", "ahead · Open-Meteo", flat, flon, ffield)]
    # forecast row uses progressively finer grids as you zoom in
    fc_field = {0: (flat, flon, ffield), 1: (flat, flon, ffield),
                2: (cala, calo, cafield), 3: (plat, plon, pfield)}
    cf = None
    for ri, (rtitle, rsub, la, lo, Z) in enumerate(rows):
        for ci, (name, ext, geom, cmd) in enumerate(scales):
            ax = fig.add_subplot(gs[ri, ci], projection=ccrs.PlateCarree())
            la_i, lo_i, Z_i = (la, lo, Z) if ri == 0 else fc_field[ci]
            pin = (PARCEL[1], PARCEL[2], pval_now if ri == 0 else pval_fc) if ci == 3 else None
            cf = panel(ax, la_i, lo_i, Z_i, ext, geom, cmap, norm, pin=pin)
            if ri == 0:
                ax.set_title(name, fontsize=12, weight="bold", pad=22)
                ax.text(0.5, 1.045, cmd, transform=ax.transAxes, ha="center", va="bottom",
                        fontsize=9, family="monospace", color="white",
                        bbox=dict(boxstyle="round,pad=0.3", fc="#26323a", ec="none"))

    # row labels (left margin)
    for ri, yc in ((0, 0.625), (1, 0.345)):
        fig.text(0.018, yc, rows[ri][0], rotation=90, va="center", ha="center",
                 fontsize=14, weight="bold", color="#b1361e")
        fig.text(0.037, yc, rows[ri][1], rotation=90, va="center", ha="center",
                 fontsize=9, color="0.45")

    fig.suptitle("Cascadia — ask for HEAT at any scale, nowcast or 7-day forecast",
                 x=0.52, y=0.965, fontsize=20, weight="bold")
    fig.text(0.52, 0.905, "type the scale you want  →  get a map.  Whole country · a region · "
             "a state · a single address — all from free public data.",
             ha="center", fontsize=12, color="0.3")

    cax = fig.add_axes([0.33, 0.085, 0.34, 0.020])
    cb = fig.colorbar(cf, cax=cax, orientation="horizontal", ticks=LEVELS,
                      spacing="proportional", extend="both")
    cb.set_label("feels-like heat  (°C)        darker = more dangerous", fontsize=10.5)
    cb.ax.tick_params(labelsize=9)
    fig.text(0.52, 0.028, "github.com/shawnatwsu/cascadia   ·   research prototype, not for "
             "life-safety decisions", ha="center", fontsize=9, color="0.5")

    OUT.parent.mkdir(exist_ok=True)
    fig.savefig(str(OUT), dpi=160)
    print(f"Saved: {OUT}")


if __name__ == "__main__":
    main()
