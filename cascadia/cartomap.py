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


def _grid(risk: pd.DataFrame, col: str):
    piv = risk.pivot_table(index="lat", columns="lon", values=col)
    return piv.columns.to_numpy(), piv.index.to_numpy(), piv.to_numpy()


def _draw(ax, lons, lats, Z, cmap, norm):
    import cartopy.crs as ccrs
    import cartopy.feature as cfeature
    mesh = ax.pcolormesh(lons, lats, Z, cmap=cmap, norm=norm,
                         transform=ccrs.PlateCarree(), shading="nearest")
    try:
        ax.add_feature(cfeature.STATES.with_scale("50m"), edgecolor="0.3", linewidth=0.6)
        ax.coastlines("50m", color="0.2", linewidth=0.6)
        ax.add_feature(cfeature.OCEAN.with_scale("50m"), facecolor="#dfe7ef")
    except Exception:
        pass  # boundary download optional; the data still renders
    ax.set_extent([lons.min(), lons.max(), lats.min(), lats.max()],
                  crs=ccrs.PlateCarree())
    gl = ax.gridlines(draw_labels=True, linewidth=0.3, color="0.7", alpha=0.5)
    gl.top_labels = gl.right_labels = False
    return mesh


def static_risk_map(risk: pd.DataFrame, region_name: str,
                    out_path: str | Path = "cascadia_risk_map.png",
                    panels: bool = True, as_of: str = "live") -> Path:
    """Render the risk surface. If panels, show compound + each hazard."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import cartopy.crs as ccrs
    from matplotlib.colors import BoundaryNorm, ListedColormap

    cmap = ListedColormap(RISK_COLORS)
    norm = BoundaryNorm(RISK_BINS, cmap.N)
    proj = ccrs.LambertConformal(
        central_longitude=float(np.nanmean(risk["lon"])),
        central_latitude=float(np.nanmean(risk["lat"])))

    cols = ["compound_risk"]
    if panels:
        cols += [c for c in ["p_flood", "p_landslide", "p_wildfire", "p_earthquake", "p_heat"]
                 if c in risk.columns]

    if len(cols) == 1:
        fig, axes = plt.subplots(1, 1, figsize=(9, 8),
                                 subplot_kw={"projection": proj})
        axes = [axes]
    else:
        fig, axes = plt.subplots(2, 3, figsize=(16, 10),
                                 subplot_kw={"projection": proj})
        axes = axes.ravel()

    mesh = None
    for ax, col in zip(axes, cols):
        lons, lats, Z = _grid(risk, col)
        mesh = _draw(ax, lons, lats, Z, cmap, norm)
        ax.set_title(HAZARD_TITLES.get(col, col), fontsize=11, weight="bold")
    for ax in axes[len(cols):]:
        ax.axis("off")

    cbar = fig.colorbar(mesh, ax=axes, orientation="horizontal",
                        fraction=0.045, pad=0.06, aspect=40,
                        ticks=RISK_BINS)
    cbar.set_label("Hazard probability over the forecast horizon", fontsize=10)
    fig.suptitle(f"Cascadia — compound & cascading hazard risk\n{region_name}  ·  {as_of}",
                 fontsize=14, weight="bold")

    out_path = Path(out_path)
    fig.savefig(out_path, dpi=140, bbox_inches="tight")
    plt.close(fig)
    return out_path
