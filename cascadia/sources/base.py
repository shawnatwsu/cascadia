"""Shared HTTP + caching helpers for source adapters."""
from __future__ import annotations

import hashlib
import json
import time
from pathlib import Path
from typing import Any

import requests

RETRY_STATUS = {429, 500, 502, 503, 504}

DEFAULT_HEADERS = {
    "User-Agent": "Cascadia-hazard-engine/0.1 (research)",
    "Accept": "application/json",
}


class SourceError(RuntimeError):
    """Raised when a feed cannot be retrieved or parsed."""


def _cache_key(url: str, params: dict[str, Any] | None) -> str:
    blob = url + json.dumps(params or {}, sort_keys=True)
    return hashlib.sha1(blob.encode()).hexdigest()[:16]


def fetch_json(
    url: str,
    params: dict[str, Any] | None = None,
    headers: dict[str, str] | None = None,
    cache_dir: Path | None = None,
    cache_ttl_s: int = 900,
    timeout: int = 30,
) -> Any:
    """GET JSON with a small on-disk cache (default 15 min TTL).

    Caching keeps repeated notebook runs fast and polite to public APIs.
    """
    hdrs = {**DEFAULT_HEADERS, **(headers or {})}
    cache_path: Path | None = None
    if cache_dir is not None:
        cache_path = cache_dir / f"{_cache_key(url, params)}.json"
        if cache_path.exists() and (time.time() - cache_path.stat().st_mtime) < cache_ttl_s:
            try:
                return json.loads(cache_path.read_text(encoding="utf-8"))
            except json.JSONDecodeError:
                pass  # fall through to refetch

    data = None
    last_exc: Exception | None = None
    for attempt in range(7):
        try:
            resp = requests.get(url, params=params, headers=hdrs, timeout=timeout)
            # Back off and retry on rate-limit / transient server errors. Rate
            # windows are often per-minute, so wait long enough for a reset.
            if resp.status_code in RETRY_STATUS:
                wait = float(resp.headers.get("Retry-After", 10 * 2 ** attempt))
                time.sleep(min(wait, 65))
                last_exc = SourceError(f"{resp.status_code} on {url}")
                continue
            resp.raise_for_status()
            data = resp.json()
            break
        except requests.RequestException as exc:
            last_exc = exc
            time.sleep(2 ** attempt)
    if data is None:
        # Serve a stale cache if we have one — resilience over freshness.
        if cache_path is not None and cache_path.exists():
            return json.loads(cache_path.read_text(encoding="utf-8"))
        raise SourceError(f"Failed to fetch {url}: {last_exc}") from last_exc

    if cache_path is not None:
        cache_path.write_text(json.dumps(data), encoding="utf-8")
    return data
