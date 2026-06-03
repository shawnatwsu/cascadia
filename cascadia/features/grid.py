"""A simple regular lat/lon grid over the region.

Cells are the spatial unit for both the predictors and the cascade graph.
Kept deliberately lightweight (no GIS dependency) so it runs anywhere.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd

from ..config import Region


@dataclass
class Grid:
    region: Region
    res: float

    @classmethod
    def from_region(cls, region: Region) -> "Grid":
        return cls(region=region, res=region.grid_resolution_deg)

    def __post_init__(self) -> None:
        r = self.region
        # Cell centers
        self.lat_centers = np.arange(r.min_lat + self.res / 2, r.max_lat, self.res)
        self.lon_centers = np.arange(r.min_lon + self.res / 2, r.max_lon, self.res)
        self.n_lat = len(self.lat_centers)
        self.n_lon = len(self.lon_centers)

    @property
    def n_cells(self) -> int:
        return self.n_lat * self.n_lon

    def cell_id(self, lat: float, lon: float) -> int | None:
        r = self.region
        if not (r.min_lat <= lat <= r.max_lat and r.min_lon <= lon <= r.max_lon):
            return None
        i = min(int((lat - r.min_lat) / self.res), self.n_lat - 1)
        j = min(int((lon - r.min_lon) / self.res), self.n_lon - 1)
        return i * self.n_lon + j

    def cells_frame(self, land_only: bool = False) -> pd.DataFrame:
        """One row per cell with id and center coordinates.

        If `land_only`, ocean cells are dropped using a bundled land mask — they
        carry no meaningful terrestrial hazard signal and otherwise pollute the
        risk surface (and any training/validation set) with coastal artifacts.
        """
        rows = []
        for i, lat in enumerate(self.lat_centers):
            for j, lon in enumerate(self.lon_centers):
                rows.append(
                    {"cell_id": i * self.n_lon + j, "lat": round(float(lat), 4),
                     "lon": round(float(lon), 4)}
                )
        df = pd.DataFrame(rows)
        if land_only:
            try:
                from global_land_mask import globe
                mask = [bool(globe.is_land(la, lo)) for la, lo in zip(df["lat"], df["lon"])]
                df = df[pd.Series(mask, index=df.index)].reset_index(drop=True)
            except ImportError:
                pass  # land mask optional; fall back to all cells
        return df

    def assign(self, df: pd.DataFrame) -> pd.DataFrame:
        """Add a `cell_id` column to a frame that has lat/lon."""
        if df.empty or "lat" not in df or "lon" not in df:
            return df.assign(cell_id=pd.Series(dtype="float"))
        ids = [
            self.cell_id(la, lo) if pd.notna(la) and pd.notna(lo) else None
            for la, lo in zip(df["lat"], df["lon"])
        ]
        return df.assign(cell_id=ids)
