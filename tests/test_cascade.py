"""Unit tests for the cascade graph propagation (offline, deterministic)."""
import networkx as nx
import pandas as pd

from cascadia.models.cascade_graph import default_cascade


def _cell(**kw):
    base = {"soil_moist_peak": 0.2, "flow_anomaly": 0.0, "precip_total_mm": 0.0}
    base.update(kw)
    return pd.Series(base)


def test_graph_is_acyclic():
    cg = default_cascade()
    assert nx.is_directed_acyclic_graph(cg.g)


def test_no_triggers_preserves_base():
    cg = default_cascade()
    base = {h: 0.0 for h in cg.g.nodes}
    base["flood"] = 0.1
    out = cg.propagate_cell(base, _cell())
    # nothing triggers flood (its parents have base 0), so it stays ~0.1
    assert abs(out["prob"]["flood"] - 0.1) < 1e-6


def test_trigger_raises_child():
    cg = default_cascade()
    base = {h: 0.0 for h in cg.g.nodes}
    base["earthquake"] = 0.8           # a big quake...
    out = cg.propagate_cell(base, _cell(soil_moist_peak=0.45, flow_anomaly=0.5))
    # ...should raise landslide via the (saturated-slope-gated) quake->landslide edge
    assert out["prob"]["landslide"] > 0.1


def test_compound_is_noisy_or():
    cg = default_cascade()
    base = {h: 0.0 for h in cg.g.nodes}
    base["flood"] = 0.5
    base["wildfire"] = 0.5
    risk = cg.run(pd.DataFrame([{**{"cell_id": 0}, **base}]),
                  pd.DataFrame([{"cell_id": 0, "lat": 45.0, "lon": -122.0,
                                 "soil_moist_peak": 0.2, "flow_anomaly": 0.0}]))
    # compound = P(at least one) >= the strongest single hazard
    assert risk["compound_risk"].iloc[0] >= 0.5
