"""Exposure & impact — turn hazard *probability* into expected *impact*.

Joins open Census data (county population estimates + county gazetteer centroids
and land area) to estimate per-cell population, so the engine can report not just
"P(hazard)" but "expected people affected" = probability x exposed population.
This is the leap insurers and emergency managers need.

v1 uses county population density (nearest county) as the exposure surface;
building-value / Expected-Annual-Loss from FEMA's National Risk Index can layer
on top later for an insurance-grade impact estimate.
"""
from __future__ import annotations

import io
from pathlib import Path

import numpy as np
import pandas as pd
import requests

POP_URL = ("https://www2.census.gov/programs-surveys/popest/datasets/"
           "2020-2023/counties/totals/co-est2023-alldata.csv")
GAZ_URL = ("https://www2.census.gov/geo/docs/maps-data/data/gazetteer/"
           "2023_Gazetteer/2023_Gaz_counties_national.zip")


def _cached(url: str, cache_dir: Path, name: str, timeout: int = 240) -> str:
    p = Path(cache_dir) / name
    if p.exists():
        return p.read_text(encoding="utf-8", errors="replace")
    txt = requests.get(url, timeout=timeout).content.decode("latin-1")
    p.write_text(txt, encoding="utf-8")
    return txt


def _cached_zip_txt(url: str, cache_dir: Path, name: str, timeout: int = 240) -> str:
    """Fetch a .zip, extract its single text member, and cache the text."""
    import io
    import zipfile
    p = Path(cache_dir) / name
    if p.exists():
        return p.read_text(encoding="utf-8", errors="replace")
    blob = requests.get(url, timeout=timeout).content
    zf = zipfile.ZipFile(io.BytesIO(blob))
    txt = zf.read(zf.namelist()[0]).decode("latin-1")
    p.write_text(txt, encoding="utf-8")
    return txt


def load_counties(cache_dir: Path) -> pd.DataFrame:
    """Per-county exposure: FIPS, centroid lat/lon, population, area, density."""
    pop = pd.read_csv(io.StringIO(_cached(POP_URL, cache_dir, "census_pop.csv")),
                      low_memory=False)
    pop = pop[pop["COUNTY"] != 0].copy()
    pop["FIPS"] = (pop["STATE"].astype(str).str.zfill(2)
                   + pop["COUNTY"].astype(str).str.zfill(3))
    pop = pop[["FIPS", "POPESTIMATE2023"]].rename(columns={"POPESTIMATE2023": "population"})

    gaz = pd.read_csv(io.StringIO(_cached_zip_txt(GAZ_URL, cache_dir, "census_gaz.txt")),
                      sep="\t", engine="python", dtype={"GEOID": str})
    gaz.columns = [c.strip() for c in gaz.columns]
    gaz = gaz.rename(columns={"GEOID": "FIPS", "INTPTLAT": "lat",
                              "INTPTLONG": "lon", "ALAND": "land_m2"})
    gaz = gaz[["FIPS", "lat", "lon", "land_m2"]]
    for c in ("lat", "lon", "land_m2"):
        gaz[c] = pd.to_numeric(gaz[c], errors="coerce")

    df = pop.merge(gaz, on="FIPS", how="inner").dropna(subset=["lat", "lon", "land_m2"])
    df["area_km2"] = df["land_m2"] / 1e6
    df["pop_density"] = df["population"] / df["area_km2"].clip(lower=1e-3)
    return df.reset_index(drop=True)


def _infer_res(cells: pd.DataFrame) -> float:
    lats = np.sort(cells["lat"].unique())
    d = np.diff(lats)
    return float(np.median(d[d > 0])) if (d > 0).any() else 0.1


def assign_population(cells: pd.DataFrame, counties: pd.DataFrame,
                      res_deg: float | None = None) -> pd.Series:
    """Estimate population per cell = nearest-county density x cell land area."""
    from scipy.spatial import cKDTree
    res = res_deg or _infer_res(cells)
    lat0 = float(cells["lat"].mean())
    kx = 111.0 * np.cos(np.radians(lat0))
    tree = cKDTree(np.column_stack([counties["lon"] * kx, counties["lat"] * 111.0]))
    _, idx = tree.query(np.column_stack([cells["lon"] * kx, cells["lat"] * 111.0]))
    coslat = np.cos(np.radians(cells["lat"].to_numpy()))
    cell_area = (res * 111.0) * (res * 111.0 * coslat)
    # Distribute each county's TRUE population across its assigned cells in
    # proportion to cell area, so totals conserve (no density x area over-count).
    cpop = counties["population"].to_numpy()[idx]
    area_by_county = pd.Series(cell_area).groupby(idx).transform("sum").to_numpy()
    pop = np.where(area_by_county > 0, cpop * cell_area / area_by_county, 0.0)
    return pd.Series(np.clip(pop, 0, None), index=cells.index)
