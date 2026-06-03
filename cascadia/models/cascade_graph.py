"""The probabilistic cascade graph — Cascadia's core contribution.

Most operational systems estimate single-hazard risk independently. Here we
treat hazards as nodes in a directed acyclic graph whose edges encode physical
*triggering* pathways (e.g. earthquake -> landslide, landslide -> flood). Each
edge carries:

  * a base transfer probability `w` — "if the parent happens, how often does it
    set off the child, all else equal", and
  * a `gate(cell)` function in [0, 1] that modulates that transfer by the cell's
    *current conditions* — a quake is far likelier to trigger a landslide on a
    rain-saturated slope than a dry one. This condition-gating is what makes the
    cascade dynamic rather than a static lookup.

Probabilities propagate through the DAG in topological order using a noisy-OR
combination, so a node's final probability fuses its own base initiation with
every triggered contribution from upstream. We also trace, per cell, the single
most probable trigger path — the explainable "story" of the cascade.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Callable

import networkx as nx
import numpy as np
import pandas as pd

from .predictors import HAZARDS

# A gate maps a cell's indicator row -> multiplier in [0, 1].
Gate = Callable[[pd.Series], float]


def _saturation(cell: pd.Series) -> float:
    # Absolute volumetric-soil-moisture saturation index in [0,1].
    return float(np.clip(cell.get("soil_moist_peak", 0.0) / 0.45, 0.0, 1.0))


def _precip_idx(cell: pd.Series) -> float:
    # Forecast precip relative to a 50 mm "meaningful event" scale.
    return float(np.clip(cell.get("precip_total_mm", 0.0) / 50.0, 0.0, 1.0))


@dataclass
class CascadeGraph:
    g: nx.DiGraph
    # feature maxima for normalizing gates, populated per-run
    feat_max: dict[str, float] = field(default_factory=dict)

    def fit_scales(self, features: pd.DataFrame) -> None:
        for col in ["soil_moist_peak", "precip_total_mm", "flow_anomaly"]:
            if col in features:
                self.feat_max[col] = float(np.nanmax(features[col].to_numpy()) or 0.0)

    def topo_order(self) -> list[str]:
        return list(nx.topological_sort(self.g))

    def propagate_cell(self, base: dict[str, float], cell: pd.Series) -> dict:
        """Propagate base probabilities through the cascade for one cell.

        Returns final per-hazard probabilities, the induced (triggered-only)
        share, and the dominant trigger path.
        """
        prob: dict[str, float] = {}
        induced: dict[str, float] = {h: 0.0 for h in self.g.nodes}
        best_parent: dict[str, str | None] = {h: None for h in self.g.nodes}
        triggered: dict[str, bool] = {h: False for h in self.g.nodes}

        for node in self.topo_order():
            p0 = base.get(node, 0.0)
            # noisy-OR: start from "not initiated", knock down by each trigger
            survive = 1.0 - p0
            top_contrib = 0.0
            for parent in self.g.predecessors(node):
                edge = self.g.edges[parent, node]
                gate_val = edge["gate"](cell) if edge.get("gate") else 1.0
                gate_val = float(np.clip(gate_val, 0.0, 1.0))
                transfer = prob[parent] * edge["w"] * gate_val
                survive *= (1.0 - transfer)
                if transfer > top_contrib:
                    top_contrib = transfer
                    best_parent[node] = parent
            p_final = 1.0 - survive
            prob[node] = float(np.clip(p_final, 0.0, 1.0))
            induced[node] = float(np.clip(p_final - p0, 0.0, 1.0))
            # A node is "triggered" only if its single strongest upstream
            # contribution outweighs its own base initiation — otherwise it was
            # self-initiated (e.g. rain-driven flooding) and should not be drawn
            # as the *child* of an unrelated parent.
            triggered[node] = top_contrib > p0

        # Headline hazard = highest final probability. Walk parents back, but
        # only across edges that genuinely triggered the child.
        terminal = max(prob, key=lambda h: prob[h])
        chain = [terminal]
        cur = terminal
        while triggered.get(cur) and best_parent.get(cur):
            cur = best_parent[cur]
            chain.append(cur)
        chain = list(reversed(chain))

        # Hazards likely to co-occur this window (the "compound" view).
        co = sorted([h for h in self.g.nodes if prob[h] >= 0.5],
                    key=lambda h: prob[h], reverse=True)

        return {
            "prob": prob,
            "induced": induced,
            "dominant_chain": chain,
            "dominant_chain_str": " -> ".join(chain),
            "co_occurring": co,
            "co_occurring_str": " + ".join(co) if co else "(none > 0.5)",
        }

    def run(self, base_probs: pd.DataFrame, features: pd.DataFrame) -> pd.DataFrame:
        """Run the cascade for every cell. Returns an enriched per-cell frame."""
        self.fit_scales(features)
        feat_idx = features.set_index("cell_id")
        records = []
        for _, row in base_probs.iterrows():
            cid = row["cell_id"]
            base = {h: float(row[h]) for h in HAZARDS if h in row}
            cell = feat_idx.loc[cid] if cid in feat_idx.index else pd.Series(dtype=float)
            res = self.propagate_cell(base, cell)
            rec = {"cell_id": cid}
            for h in self.g.nodes:
                rec[f"p_{h}"] = res["prob"][h]
                rec[f"induced_{h}"] = res["induced"][h]
            # Compound risk: probability ANY hazard occurs (noisy-OR over hazards).
            rec["compound_risk"] = 1.0 - float(np.prod([1.0 - res["prob"][h] for h in self.g.nodes]))
            rec["expected_hazards"] = float(sum(res["prob"].values()))
            rec["dominant_chain"] = res["dominant_chain_str"]
            rec["co_occurring"] = res["co_occurring_str"]
            records.append(rec)
        out = pd.DataFrame(records)
        return out.merge(features[["cell_id", "lat", "lon"]], on="cell_id", how="left")


def default_cascade() -> CascadeGraph:
    """The Cascadia hazard cascade DAG with condition-aware gates."""
    g = nx.DiGraph()
    g.add_nodes_from(HAZARDS)

    # earthquake -> landslide: strong, and amplified on saturated slopes.
    g.add_edge("earthquake", "landslide", w=0.45,
               gate=lambda c: 0.4 + 0.6 * _saturation(c),
               rationale="seismic shaking destabilizes slopes; worse when saturated")
    # earthquake -> flood: dam/levee failure; modest, slightly raised by high flow.
    g.add_edge("earthquake", "flood", w=0.12,
               gate=lambda c: 0.5 + 0.5 * float(c.get("flow_anomaly", 0.0)),
               rationale="shaking can breach dams/levees, more dangerous at high flow")
    # wildfire -> landslide: post-fire debris flow, realized when rain arrives.
    g.add_edge("wildfire", "landslide", w=0.35,
               gate=lambda c: _precip_idx(c),
               rationale="burn scars produce debris flows once rain falls")
    # landslide -> flood: channel blockage / outburst; gated by water in system.
    g.add_edge("landslide", "flood", w=0.30,
               gate=lambda c: 0.3 + 0.7 * float(c.get("flow_anomaly", 0.0)),
               rationale="landslide dams impound then release water downstream")
    # heat -> wildfire: heatwaves dry fuels and prime ignition/spread; stronger
    # where soils are already dry.
    g.add_edge("heat", "wildfire", w=0.25,
               gate=lambda c: 0.3 + 0.7 * (1.0 - _saturation(c)),
               rationale="heatwaves desiccate fuels, raising fire potential")

    return CascadeGraph(g=g)
