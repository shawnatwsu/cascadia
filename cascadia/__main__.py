"""CLI: python -m cascadia [--map] [--config path]"""
from __future__ import annotations

import argparse

from .config import Config
from .pipeline import run_pipeline


def main() -> None:
    ap = argparse.ArgumentParser(description="Cascadia compound-hazard engine")
    ap.add_argument("--config", default=None, help="path to config.yaml")
    ap.add_argument("--map", action="store_true", help="render an HTML risk map")
    ap.add_argument("--out", default="cascadia_risk_map.html", help="map output path")
    ap.add_argument("--top", type=int, default=10, help="N top-risk cells to print")
    args = ap.parse_args()

    config = Config.load(args.config)
    res = run_pipeline(config)

    print("\n========== SUMMARY ==========")
    print(res.summary())
    print(f"\n========== TOP {args.top} RISK CELLS ==========")
    print(res.top_cells(args.top).to_string(index=False))

    if args.map:
        from .viz import risk_map
        path = risk_map(res.risk, config.region.name,
                        config.region.grid_resolution_deg, args.out)
        print(f"\nMap written to: {path.resolve()}")


if __name__ == "__main__":
    main()
