"""Configuration loading and the region grid definition."""
from __future__ import annotations

import os
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import yaml

PROJECT_ROOT = Path(__file__).resolve().parent.parent
DEFAULT_CONFIG_PATH = PROJECT_ROOT / "config.yaml"


@dataclass
class Region:
    name: str
    bbox: tuple[float, float, float, float]  # min_lon, min_lat, max_lon, max_lat
    state: str
    grid_resolution_deg: float

    @property
    def min_lon(self) -> float:
        return self.bbox[0]

    @property
    def min_lat(self) -> float:
        return self.bbox[1]

    @property
    def max_lon(self) -> float:
        return self.bbox[2]

    @property
    def max_lat(self) -> float:
        return self.bbox[3]

    def contains(self, lon: float, lat: float) -> bool:
        return (
            self.min_lon <= lon <= self.max_lon
            and self.min_lat <= lat <= self.max_lat
        )


@dataclass
class Config:
    region: Region
    horizon_days: int
    sources: dict[str, Any]
    cache_dir: Path
    raw: dict[str, Any] = field(default_factory=dict)
    # When set (a past date), the pipeline runs in *historical* mode: sources
    # pull archived data for the window [as_of, as_of + horizon] instead of live
    # feeds. This is what powers backtesting and event validation.
    as_of: datetime | None = None

    @property
    def is_historical(self) -> bool:
        return self.as_of is not None

    def window(self) -> tuple[datetime, datetime]:
        """The [start, end] datetime window the run targets (UTC)."""
        from datetime import timedelta
        start = self.as_of or datetime.now(timezone.utc)
        return start, start + timedelta(days=self.horizon_days)

    def with_region(self, bbox: tuple[float, float, float, float],
                    name: str | None = None, state: str | None = None) -> "Config":
        """Return a copy targeting a different bounding box (for event validation)."""
        new_region = Region(
            name=name or self.region.name,
            bbox=tuple(bbox),  # type: ignore[arg-type]
            state=state or self.region.state,
            grid_resolution_deg=self.region.grid_resolution_deg,
        )
        return Config(
            region=new_region, horizon_days=self.horizon_days,
            sources=self.sources, cache_dir=self.cache_dir, raw=self.raw,
            as_of=self.as_of,
        )

    def with_as_of(self, when: datetime | str | None) -> "Config":
        """Return a copy targeting a specific historical date."""
        if isinstance(when, str):
            when = datetime.fromisoformat(when)
        if when is not None and when.tzinfo is None:
            when = when.replace(tzinfo=timezone.utc)
        return Config(
            region=self.region, horizon_days=self.horizon_days,
            sources=self.sources, cache_dir=self.cache_dir, raw=self.raw,
            as_of=when,
        )

    @classmethod
    def load(cls, path: str | Path | None = None) -> "Config":
        path = Path(path) if path else DEFAULT_CONFIG_PATH
        with open(path, "r", encoding="utf-8") as fh:
            raw = yaml.safe_load(fh)

        r = raw["region"]
        region = Region(
            name=r["name"],
            bbox=tuple(r["bbox"]),  # type: ignore[arg-type]
            state=r["state"],
            grid_resolution_deg=float(r["grid_resolution_deg"]),
        )
        cache_dir = PROJECT_ROOT / raw.get("cache_dir", "data/cache")
        cache_dir.mkdir(parents=True, exist_ok=True)
        return cls(
            region=region,
            horizon_days=int(raw.get("horizon_days", 7)),
            sources=raw.get("sources", {}),
            cache_dir=cache_dir,
            raw=raw,
        )

    def env(self, key: str) -> str | None:
        return os.environ.get(key)
