"""State / region resolution (skipped in the minimal CI env that lacks cartopy)."""
import pytest


def test_state_resolution_excludes_ak_hi():
    pytest.importorskip("cartopy")  # state_geoms() loads Natural Earth via cartopy
    from cascadia import geo
    assert geo.resolve_state("texas") == "Texas"
    assert geo.resolve_state("TX") == "Texas"
    assert geo.resolve_state("new_york") == "New York"
    assert geo.resolve_state("New Mexico") == "New Mexico"
    # contiguous-US only
    assert geo.resolve_state("alaska") is None
    assert geo.resolve_state("hawaii") is None
    assert geo.resolve_state("not a state") is None


def test_conditions_resolves_state():
    pytest.importorskip("cartopy")
    from cascadia.conditions import _resolve_region, region_keys
    bbox, res, stride, label, mask_fn, boundaries = _resolve_region("texas")
    assert label == "Texas conditions"
    assert len(boundaries) == 1 and bbox[0] < bbox[2] and bbox[1] < bbox[3]
    assert "texas" in region_keys() and "alaska" not in region_keys()
