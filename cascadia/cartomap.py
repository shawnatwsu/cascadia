"""Static, projected, publication-quality hazard maps (matplotlib + cartopy).

Replaces the flat folium wash for *analysis output*: discrete risk classes, a
proper colorbar/legend, state + coastline boundaries, and an optional
multi-panel layout showing the compound risk alongside each hazard. The folium
dashboard remains for interactive exploration.
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

# Discrete risk classes — fine at the low end (most weeks live there) so spatial
# structure is visible, coarser toward the dangerous tail.
RISK_BINS = [0.0, 0.02, 0.05, 0.10, 0.15, 0.20, 0.30, 0.50, 1.0]
RISK_LABELS = ["<0.02", "0.02–0.05", "0.05–0.10", "0.10–0.15",
               "0.15–0.20", "0.20–0.30", "0.30–0.50", ">0.50"]
# Perceptual green→yellow→orange→red→purple ramp (one colour per class).
RISK_COLORS = ["#1a9850", "#66bd63", "#a6d96a", "#fee08b",
               "#fdae61", "#f46d43", "#d73027", "#7b3294"]

HAZARD_TITLES = {
    "compound_risk": "Compound risk  (P any hazard)",
    "expected_hazards": "Expected number of hazards",
    "p_flood": "Flood", "p_landslide": "Landslide",
    "p_wildfire": "Wildfire", "p_earthquake": "Earthquake", "p_heat": "Heat",
    "p_smoke": "Wildfire smoke / air quality",
}

# Per-column colorbar units (otherwise the figure-level value_label is used).
COL_VALUE_LABEL = {
    "expected_hazards": "expected # of hazards", "population": "people",
    "compound_risk": "P(any hazard)",
}

# Per-hazard themed colormaps so each panel reads as "what it is" at a glance.
HAZARD_CMAPS = {
    "compound_risk": "inferno_r",
    "expected_hazards": "inferno_r",
    "p_flood": "Blues",
    "p_landslide": "YlOrBr",
    "p_wildfire": "YlOrRd",
    "p_earthquake": "Purples",
    "p_heat": "hot_r",
    "p_smoke": "Greys",
    # sub-seasonal (weeks 2-6) outlook layers
    "fire_outlook": "YlOrRd", "heat_outlook": "hot_r", "drought_outlook": "YlOrBr",
    # seasonal (1-3 month) ENSO outlook layers
    "seasonal_fire": "YlOrRd", "seasonal_flood": "Blues", "seasonal_heat": "hot_r",
}
HAZARD_TITLES.update({
    "fire_outlook": "Fire potential outlook (wk 2-6)",
    "heat_outlook": "Heat outlook (wk 2-6)",
    "drought_outlook": "Drought / dryness outlook (wk 2-6)",
    "seasonal_fire": "Fire / drought season outlook",
    "seasonal_flood": "Wet / flood season outlook",
    "seasonal_heat": "Warm-season (heat) outlook",
    # exposure & impact layers
    "population": "Population (per cell)",
    "expected_affected": "Expected people affected (any hazard)",
    "impact_flood": "Flood — expected people affected",
    "impact_landslide": "Landslide — expected people affected",
    "impact_wildfire": "Wildfire — expected people affected",
    "impact_heat": "Heat — expected people affected",
    "impact_earthquake": "Earthquake — expected people affected",
})
HAZARD_CMAPS.update({
    "population": "BuPu", "expected_affected": "magma_r",
    "impact_flood": "Blues", "impact_landslide": "YlOrBr",
    "impact_wildfire": "YlOrRd", "impact_heat": "hot_r",
    "impact_earthquake": "Purples",
})


def _grid(risk: pd.DataFrame, col: str):
    piv = risk.pivot_table(index="lat", columns="lon", values=col)
    return piv.columns.to_numpy(), piv.index.to_numpy(), piv.to_numpy()


def _add_boundaries(ax, lons, lats, boundaries=None):
    import cartopy.crs as ccrs
    from matplotlib.ticker import MaxNLocator
    # White background, US state outlines only (no ocean tint, no Mexico/Canada).
    ax.set_facecolor("white")
    try:
        from .geo import conus_states
        geoms = boundaries if boundaries is not None else conus_states()
        ax.add_geometries(geoms, ccrs.PlateCarree(), facecolor="none",
                          edgecolor="0.4", linewidth=0.4, zorder=3)
    except Exception:
        import cartopy.feature as cfeature
        ax.add_feature(cfeature.STATES.with_scale("50m"), edgecolor="0.4", linewidth=0.4)
    ax.set_extent([lons.min(), lons.max(), lats.min(), lats.max()],
                  crs=ccrs.PlateCarree())
    # Clean edge labels only — no criss-cross gridlines across the map.
    gl = ax.gridlines(draw_labels=True, linewidth=0)
    gl.xlines = gl.ylines = False
    gl.top_labels = gl.right_labels = False
    gl.xlocator = MaxNLocator(4)
    gl.ylocator = MaxNLocator(4)
    gl.xlabel_style = gl.ylabel_style = {"size": 8}


def _bin_edges(vmax: float) -> np.ndarray:
    """Nice, rounded discrete class edges from 0 to vmax (per-hazard)."""
    from matplotlib.ticker import MaxNLocator
    edges = MaxNLocator(nbins=6, min_n_ticks=4).tick_values(0.0, vmax)
    edges = edges[edges >= 0]
    if edges[0] > 0:
        edges = np.insert(edges, 0, 0.0)
    return edges


def _fmt(v: float, vmax: float) -> str:
    if vmax >= 1_000_000:
        return f"{v/1e6:.1f}M"
    if vmax >= 1000:
        return f"{v/1e3:.0f}k" if v >= 1000 else f"{v:.0f}"
    if vmax >= 10:
        return f"{v:.0f}"
    if vmax < 0.01:
        return f"{v:.4f}".rstrip("0").rstrip(".")
    if vmax < 0.1:
        return f"{v:.3f}".rstrip("0").rstrip(".")
    return f"{v:.2f}"


# Fixed diverging class edges for anomaly/tendency maps (centred on 0 = normal).
# Fixed (not auto-scaled) so a weak signal honestly looks weak, not stretched.
DIVERGING_EDGES = [-0.40, -0.25, -0.12, -0.04, 0.04, 0.12, 0.25, 0.40]


def _panel(fig, ax, lons, lats, Z, col, value_label="probability over horizon",
           boundaries=None, diverging=False):
    """Draw one hazard panel with a discrete, full-width binned colorbar.

    diverging=True uses a FIXED diverging scale centred on 0 (for anomaly /
    tendency layers) so weak signals are not exaggerated by auto-scaling."""
    import cartopy.crs as ccrs
    import matplotlib.pyplot as plt
    from matplotlib.colors import BoundaryNorm, ListedColormap

    finite = Z[np.isfinite(Z)]
    if diverging:
        edges = np.array(DIVERGING_EDGES)
        base = plt.get_cmap("RdBu_r")          # blue = below normal, red = above
        cmap = ListedColormap(base(np.linspace(0.0, 1.0, len(edges) - 1)))
        vmax = edges[-1]
    else:
        vmax = float(np.nanpercentile(finite, 98)) if finite.size else 1.0
        vmax = max(vmax, 1e-4)
        edges = _bin_edges(vmax)
        base = plt.get_cmap(HAZARD_CMAPS.get(col, "YlOrRd"))
        cmap = ListedColormap(base(np.linspace(0.15, 1.0, len(edges) - 1)))
    norm = BoundaryNorm(edges, cmap.N)

    mesh = ax.pcolormesh(lons, lats, Z, cmap=cmap, norm=norm,
                         transform=ccrs.PlateCarree(), shading="nearest")
    _add_boundaries(ax, lons, lats, boundaries=boundaries)
    ax.set_title(HAZARD_TITLES.get(col, col), fontsize=11, weight="bold")

    cb = fig.colorbar(mesh, ax=ax, orientation="horizontal", location="bottom",
                      shrink=1.0, fraction=0.07, pad=0.06, ticks=edges,
                      spacing="proportional")
    cb.set_ticklabels([_fmt(e, vmax) for e in edges])
    cb.set_label(COL_VALUE_LABEL.get(col, value_label), fontsize=8)
    cb.ax.tick_params(labelsize=7)


def static_risk_map(risk: pd.DataFrame, region_name: str,
                    out_path: str | Path = "cascadia_risk_map.png",
                    panels: bool = True, as_of: str = "live",
                    cols: list[str] | None = None, suptitle: str | None = None,
                    description: str | None = None,
                    value_label: str = "probability over horizon",
                    provenance: str | None = None, boundaries=None,
                    diverging: bool = False) -> Path:
    """Render the risk surface; each panel gets its own themed, binned colormap
    and a full-width colorbar so every hazard's spatial structure is legible.

    Pass `cols` to render an arbitrary set of layers (e.g. sub-seasonal outlooks).
    """
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import cartopy.crs as ccrs

    # Albers Equal-Area for CONUS — areas (and population/impact sums) are
    # visually honest, the standard for US thematic maps.
    proj = ccrs.AlbersEqualArea(central_longitude=-96, central_latitude=37.5,
                                standard_parallels=(29.5, 45.5))

    if cols is None:
        # Lead with "expected number of hazards" (P(any) saturates and is
        # uninformative when several hazards are elevated everywhere).
        lead = "expected_hazards" if "expected_hazards" in risk.columns else "compound_risk"
        cols = [lead]
        if panels:
            cols += [c for c in ["p_flood", "p_landslide", "p_wildfire",
                                 "p_earthquake", "p_heat", "p_smoke"] if c in risk.columns]

    ncol = 3 if len(cols) > 1 else 1
    nrow = int(np.ceil(len(cols) / ncol))
    fig, axes = plt.subplots(nrow, ncol, figsize=(5.4 * ncol, 4.6 * nrow),
                             subplot_kw={"projection": proj},
                             constrained_layout=True)
    axes = np.atleast_1d(axes).ravel()

    for ax, col in zip(axes, cols):
        lons, lats, Z = _grid(risk, col)
        _panel(fig, ax, lons, lats, Z, col, value_label=value_label,
               boundaries=boundaries, diverging=diverging)
    for ax in axes[len(cols):]:
        ax.axis("off")

    title = suptitle or f"Cascadia — compound & cascading multi-hazard risk\n{region_name}  ·  {as_of}"
    fig.suptitle(title, fontsize=15, weight="bold")
    # Descriptive sub-subtitle: explain exactly what the panels show.
    desc = description or (
        "Each panel: probability that the hazard occurs in each grid cell over the "
        "forecast horizon. Lead panel = expected number of hazards (sum of the "
        "per-hazard probabilities; P(any) saturates when several are elevated). "
        "Hazards are fused through a cascade graph (one hazard can trigger "
        "another). Colors are binned and scaled per panel, so classes differ "
        "between hazards — read each panel's own colorbar.")
    # One caption block BELOW the panels (description + provenance), so nothing
    # overlaps the colorbars or their labels.
    prov = provenance or "Sources: GRIDMET, USGS, NWS, Open-Meteo/ERA5, NASA FIRMS, US Census"
    fig.text(0.5, -0.02, desc, ha="center", va="top", fontsize=8.5,
             color="0.25", wrap=True)
    fig.text(0.5, -0.13, prov, ha="center", va="top", fontsize=7.5,
             color="0.45", style="italic", wrap=True)
    out_path = Path(out_path)
    fig.savefig(out_path, dpi=140, bbox_inches="tight")
    plt.close(fig)
    return out_path
