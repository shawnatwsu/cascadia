"""Render the compound-risk surface to an interactive folium map."""
from __future__ import annotations

from pathlib import Path

import folium
import numpy as np
import pandas as pd


def _color(risk: float) -> str:
    # green -> yellow -> red
    r = int(255 * min(1.0, risk * 2))
    g = int(255 * min(1.0, (1 - risk) * 2))
    return f"#{r:02x}{g:02x}30"


def risk_map(risk: pd.DataFrame, region_name: str, res_deg: float,
             out_path: str | Path = "cascadia_risk_map.html") -> Path:
    """Draw per-cell compound risk as a colored grid with cascade tooltips."""
    center = [risk["lat"].mean(), risk["lon"].mean()]
    m = folium.Map(location=center, zoom_start=7, tiles="CartoDB positron")
    half = res_deg / 2

    for _, row in risk.iterrows():
        risk_v = float(row["compound_risk"])
        if np.isnan(risk_v):
            continue
        lat, lon = row["lat"], row["lon"]
        bounds = [[lat - half, lon - half], [lat + half, lon + half]]
        tip = (
            f"<b>Compound risk: {risk_v:.2f}</b><br>"
            f"Expected hazards: {row['expected_hazards']:.2f}<br>"
            f"P(flood)={row.get('p_flood', 0):.2f} "
            f"P(landslide)={row.get('p_landslide', 0):.2f}<br>"
            f"P(wildfire)={row.get('p_wildfire', 0):.2f} "
            f"P(quake)={row.get('p_earthquake', 0):.2f}<br>"
            f"<i>Cascade: {row.get('dominant_chain', '')}</i>"
        )
        folium.Rectangle(
            bounds=bounds, color=None, weight=0, fill=True,
            fill_color=_color(risk_v), fill_opacity=0.45 + 0.4 * risk_v,
            tooltip=tip,
        ).add_to(m)

    title = (f'<div style="position:fixed;top:10px;left:50px;z-index:9999;'
             f'background:white;padding:6px 10px;border-radius:6px;'
             f'font-family:sans-serif;font-size:13px;box-shadow:0 1px 4px #0003">'
             f'<b>Cascadia</b> — compound &amp; cascading hazard risk<br>'
             f'<span style="font-size:11px">{region_name}</span></div>')
    m.get_root().html.add_child(folium.Element(title))

    out_path = Path(out_path)
    m.save(str(out_path))
    return out_path
