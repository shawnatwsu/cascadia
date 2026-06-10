"""'Ask for HEAT at any scale' explainer graphic, for any contiguous-US state.

Four nested scales (CONUS -> the state's NCA5 region -> the state -> a parcel),
two rows: NOWCAST (GRIDMET, cached) and 7-DAY FORECAST (Open-Meteo), with the
command you'd type above each column. Smooth filled-contour rendering, clipped to
each geography. All real data.

    from cascadia.explainer import make_explainer
    make_explainer("texas")
"""
from __future__ import annotations

import glob
import os
import time
from pathlib import Path

import numpy as np

from .config import Config
from . import geo
from .sources.gridmet import heat_index_c

LEVELS = np.array([18, 22, 26, 30, 34, 38, 42, 46], float)   # feels-like degC
OMURL = "https://api.open-meteo.com/v1/forecast"

# A hot/representative parcel per state; states not listed fall back to the
# state's interior representative point.
PARCELS = {
    "California": ("Palm Springs, CA", 33.830, -116.545),
    "Texas": ("Laredo, TX", 27.506, -99.507),
    "Arizona": ("Phoenix, AZ", 33.448, -112.074),
    "Nevada": ("Las Vegas, NV", 36.170, -115.140),
    "Florida": ("Miami, FL", 25.775, -80.194),
    "Louisiana": ("New Orleans, LA", 29.951, -90.072),
    "Georgia": ("Atlanta, GA", 33.749, -84.388),
    "Illinois": ("Chicago, IL", 41.878, -87.630),
    "New York": ("New York, NY", 40.713, -74.006),
    "Washington": ("Seattle, WA", 47.606, -122.332),
    "Oregon": ("Portland, OR", 45.515, -122.679),
    "Colorado": ("Denver, CO", 39.739, -104.990),
    "Oklahoma": ("Oklahoma City, OK", 35.468, -97.516),
    "New Mexico": ("Albuquerque, NM", 35.084, -106.651),
    "Utah": ("Salt Lake City, UT", 40.761, -111.890),
}


def _latest_cube(cache_dir: Path) -> str:
    files = sorted(glob.glob(str(Path(cache_dir) / "gridmet_-125p0_24p5_-66p9_49p5_*.nc")),
                   key=os.path.getmtime)
    if not files:
        raise SystemExit("No cached CONUS GRIDMET cube found.\n"
                         "Run this first:  python run.py conditions conus")
    return files[-1]


def _region_for_state(name: str) -> str | None:
    for key, members in geo.NCA5_REGIONS.items():
        if name in members:
            return key
    return None


def _parcel_for(state: str):
    if state in PARCELS:
        return PARCELS[state]
    p = geo.state_geometry(state).representative_point()
    return (f"a {state} address", float(p.y), float(p.x))


# ----------------------------------------------------------------- data
def _nowcast_grid(cube_path):
    import xarray as xr
    ds = xr.open_dataset(cube_path)
    tmax = ds["tmax_c"].max("time").to_numpy()
    rhmin = ds["rh_min"].min("time").to_numpy()
    return ds["lat"].to_numpy(), ds["lon"].to_numpy(), heat_index_c(tmax, rhmin)


def _om_get(params, retries=6):
    import requests
    for attempt in range(retries):
        try:
            r = requests.get(OMURL, params=params, timeout=60)
            if r.status_code == 429:
                time.sleep(10 * (attempt + 1)); continue
            return r.json()
        except Exception:
            time.sleep(5 * (attempt + 1))
    return None


def _fetch_field(lat1d, lon1d, tag, cache_dir, pace=10, verbose=True):
    """7-day max apparent temperature on a lat/lon grid (Open-Meteo). Cached."""
    cache = str(Path(cache_dir) / f"heat_fc_{tag}.npz")
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
        elif verbose:
            print(f"  {tag} chunk {i} failed")
        time.sleep(pace)
    field = vals.reshape(LAT.shape)
    try:
        np.savez(cache, lat=lat1d, lon=lon1d, field=field)
    except Exception:
        pass
    return lat1d, lon1d, field


def _box(b, step, tag, cache_dir, pad=0.3, pace=8):
    lat1d = np.arange(b[1] - pad, b[3] + pad + 1e-6, step)
    lon1d = np.arange(b[0] - pad, b[2] + pad + 1e-6, step)
    return _fetch_field(lat1d, lon1d, tag, cache_dir, pace=pace)


def _point(lat, lon):
    data = _om_get({"latitude": lat, "longitude": lon,
                    "daily": "apparent_temperature_max", "forecast_days": 7,
                    "timezone": "UTC"})
    try:
        return float(np.nanmax(data["daily"]["apparent_temperature_max"]))
    except Exception:
        return np.nan


# ----------------------------------------------------------------- render
def _clip(ax, cf, geom):
    if geom is None:
        return
    import cartopy.crs as ccrs
    from cartopy.mpl.patch import geos_to_path
    from matplotlib.path import Path as MplPath
    from matplotlib.patches import PathPatch
    patch = PathPatch(MplPath.make_compound_path(*geos_to_path(geom)),
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


def _panel(ax, lat1d, lon1d, Z, extent, geom, cmap, norm, pin=None):
    import cartopy.crs as ccrs
    import cartopy.feature as cfeature
    LON, LAT = np.meshgrid(lon1d, lat1d)
    ax.set_extent(extent, crs=ccrs.PlateCarree())
    cf = ax.contourf(LON, LAT, np.ma.masked_invalid(Z), levels=LEVELS, cmap=cmap,
                     norm=norm, extend="both", transform=ccrs.PlateCarree())
    try:
        cf.set_edgecolor("face")
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


def make_explainer(state_key: str = "california", out_path: str | Path | None = None,
                   cache_dir: Path | None = None, verbose: bool = True) -> Path:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.colors import BoundaryNorm, ListedColormap
    import cartopy.crs as ccrs

    cfg = Config.load()
    cache_dir = cache_dir or cfg.cache_dir
    state = geo.resolve_state(state_key)
    if not state:
        raise SystemExit(f"'{state_key}' is not a contiguous-US state "
                         "(Alaska/Hawaii are out of scope).")
    rkey = _region_for_state(state)
    rname = geo.NCA5_NAMES.get(rkey, "region")
    pname, plat_c, plon_c = _parcel_for(state)
    tag = state.lower().replace(" ", "")
    log = (lambda *a: print(*a)) if verbose else (lambda *a: None)
    log(f"Heat explainer for {state} ({rname} region), parcel = {pname}")

    cube = _latest_cube(cache_dir)
    nlat, nlon, nhi = _nowcast_grid(cube)
    pval_now = float(nhi[np.argmin(np.abs(nlat - plat_c))][np.argmin(np.abs(nlon - plon_c))])
    log("forecast: CONUS grid…")
    flat, flon, ffield = _fetch_field(np.arange(25.0, 49.01, 1.0),
                                      np.arange(-124.5, -67.0, 1.0), "conus1deg", cache_dir)
    sb = geo.state_bbox(state, 0.0)
    log(f"forecast: {state}…")
    slat, slon, sfield = _box(sb, 0.35, f"state_{tag}_035", cache_dir)
    log("forecast: parcel…")
    pa, po, pfield = _box((plon_c - 1.5, plat_c - 1.2, plon_c + 1.5, plat_c + 1.2),
                          0.15, f"parcel_{tag}_015", cache_dir, pad=0.0, pace=6)
    pval_fc = _point(plat_c, plon_c)
    if not np.isfinite(pval_fc):
        pval_fc = float(np.nanmax(pfield))
    log(f"parcel  now {pval_now:.0f}C  ->  7-day {pval_fc:.0f}C")

    rb = geo.region_bbox(rkey, 0.3)
    stb = geo.state_bbox(state, 0.4)
    scales = [
        ("CONUS", (-125, -66.9, 24.5, 49.5), geo.conus_union(), "run.ps1 conditions conus"),
        (f"{rname} region", (rb[0], rb[2], rb[1], rb[3]), geo.region_geometry(rkey),
         f"run.ps1 conditions {rkey}"),
        (state, (stb[0], stb[2], stb[1], stb[3]), geo.state_geometry(state),
         f"run.ps1 conditions {tag}"),
        (f"{pname} parcel", (plon_c - 1.7, plon_c + 1.7, plat_c - 1.3, plat_c + 1.3),
         None, f'run.ps1 parcel "{pname}"'),
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
    fc_field = {0: (flat, flon, ffield), 1: (flat, flon, ffield),
                2: (slat, slon, sfield), 3: (pa, po, pfield)}
    cf = None
    for ri, row in enumerate(rows):
        la, lo, Z = row[2], row[3], row[4]
        for ci, (name, ext, gm, cmd) in enumerate(scales):
            ax = fig.add_subplot(gs[ri, ci], projection=ccrs.PlateCarree())
            la_i, lo_i, Z_i = (la, lo, Z) if ri == 0 else fc_field[ci]
            pin = (plat_c, plon_c, pval_now if ri == 0 else pval_fc) if ci == 3 else None
            cf = _panel(ax, la_i, lo_i, Z_i, ext, gm, cmap, norm, pin=pin)
            if ri == 0:
                ax.set_title(name, fontsize=12, weight="bold", pad=22)
                ax.text(0.5, 1.045, cmd, transform=ax.transAxes, ha="center", va="bottom",
                        fontsize=8.5, family="monospace", color="white",
                        bbox=dict(boxstyle="round,pad=0.3", fc="#26323a", ec="none"))
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

    out_path = Path(out_path) if out_path else (cfg.cache_dir.parent.parent / "outputs"
                                                / f"heat_explainer_{tag}.png")
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(str(out_path), dpi=160)
    plt.close(fig)
    log(f"Saved: {out_path}")
    return out_path
