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
    "p_flood": "Flood", "p_landslide": "Landslide",
    "p_wildfire": "Wildfire", "p_earthquake": "Earthquake", "p_heat": "Heat",
}

# Per-hazard themed colormaps so each panel reads as "what it is" at a glance.
HAZARD_CMAPS = {
    "compound_risk": "inferno_r",
    "p_flood": "Blues",
    "p_landslide": "YlOrBr",
    "p_wildfire": "YlOrRd",
    "p_earthquake": "Purples",
    "p_heat": "hot_r",
    # sub-seasonal (weeks 2-6) outlook layers
    "fire_outlook": "YlOrRd", "heat_outlook": "hot_r", "drought_outlook": "YlOrBr",
}
HAZARD_TITLES.update({
    "fire_outlook": "Fire potential outlook (wk 2-6)",
    "heat_outlook": "Heat outlook (wk 2-6)",
    "drought_outlook": "Drought / dryness outlook (wk 2-6)",
})


def _grid(risk: pd.DataFrame, col: str):
    piv = risk.pivot_table(index="lat", columns="lon", values=col)
    return piv.columns.to_numpy(), piv.index.to_numpy(), piv.to_numpy()


def _add_boundaries(ax, lons, lats):
    import cartopy.crs as ccrs
    import cartopy.feature as cfeature
    from matplotlib.ticker import MaxNLocator
    try:
        ax.add_feature(cfeature.OCEAN.with_scale("50m"), facecolor="#dfe7ef", zorder=0)
        ax.add_feature(cfeature.STATES.with_scale("50m"), edgecolor="0.25", linewidth=0.5)
        ax.coastlines("50m", color="0.15", linewidth=0.5)
    except Exception:
        pass
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
    if vmax < 0.01:
        return f"{v:.4f}".rstrip("0").rstrip(".")
    if vmax < 0.1:
        return f"{v:.3f}".rstrip("0").rstrip(".")
    return f"{v:.2f}"


def _panel(fig, ax, lons, lats, Z, col):
    """Draw one hazard panel with a discrete, per-hazard binned colorbar that
    spans the full subplot width."""
    import cartopy.crs as ccrs
    import matplotlib.pyplot as plt
    from matplotlib.colors import BoundaryNorm, ListedColormap

    finite = Z[np.isfinite(Z)]
    vmax = float(np.nanpercentile(finite, 98)) if finite.size else 1.0
    vmax = max(vmax, 1e-4)
    edges = _bin_edges(vmax)
    base = plt.get_cmap(HAZARD_CMAPS.get(col, "YlOrRd"))
    cmap = ListedColormap(base(np.linspace(0.15, 1.0, len(edges) - 1)))
    norm = BoundaryNorm(edges, cmap.N)

    mesh = ax.pcolormesh(lons, lats, Z, cmap=cmap, norm=norm,
                         transform=ccrs.PlateCarree(), shading="nearest")
    _add_boundaries(ax, lons, lats)
    ax.set_title(HAZARD_TITLES.get(col, col), fontsize=11, weight="bold")

    cb = fig.colorbar(mesh, ax=ax, orientation="horizontal", location="bottom",
                      shrink=1.0, fraction=0.07, pad=0.06, ticks=edges,
                      spacing="proportional")
    cb.set_ticklabels([_fmt(e, vmax) for e in edges])
    cb.set_label("probability over horizon", fontsize=8)
    cb.ax.tick_params(labelsize=7)


def static_risk_map(risk: pd.DataFrame, region_name: str,
                    out_path: str | Path = "cascadia_risk_map.png",
                    panels: bool = True, as_of: str = "live",
                    cols: list[str] | None = None, suptitle: str | None = None,
                    description: str | None = None) -> Path:
    """Render the risk surface; each panel gets its own themed, binned colormap
    and a full-width colorbar so every hazard's spatial structure is legible.

    Pass `cols` to render an arbitrary set of layers (e.g. sub-seasonal outlooks).
    """
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import cartopy.crs as ccrs

    proj = ccrs.PlateCarree()  # straight, rectangular axes with clean lon/lat

    if cols is None:
        cols = ["compound_risk"]
        if panels:
            cols += [c for c in ["p_flood", "p_landslide", "p_wildfire",
                                 "p_earthquake", "p_heat"] if c in risk.columns]

    ncol = 3 if len(cols) > 1 else 1
    nrow = int(np.ceil(len(cols) / ncol))
    fig, axes = plt.subplots(nrow, ncol, figsize=(5.4 * ncol, 4.6 * nrow),
                             subplot_kw={"projection": proj},
                             constrained_layout=True)
    axes = np.atleast_1d(axes).ravel()

    for ax, col in zip(axes, cols):
        lons, lats, Z = _grid(risk, col)
        _panel(fig, ax, lons, lats, Z, col)
    for ax in axes[len(cols):]:
        ax.axis("off")

    title = suptitle or f"Cascadia — compound & cascading multi-hazard risk\n{region_name}  ·  {as_of}"
    fig.suptitle(title, fontsize=15, weight="bold")
    # Descriptive sub-subtitle: explain exactly what the panels show.
    desc = description or (
        "Each panel: probability that the hazard occurs in each grid cell over the "
        "forecast horizon. Compound = P(at least one hazard). Hazards are fused "
        "through a cascade graph (one hazard can trigger another). Colors are "
        "binned and scaled per panel, so classes differ between hazards — read "
        "each panel's own colorbar.")
    fig.text(0.5, -0.01, desc, ha="center", va="top", fontsize=8.5,
             color="0.25", wrap=True)
    out_path = Path(out_path)
    fig.savefig(out_path, dpi=140, bbox_inches="tight")
    plt.close(fig)
    return out_path
