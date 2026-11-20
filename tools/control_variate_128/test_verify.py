"""Corrupted endpoint, source and timing evidence must fail independent checks."""
import copy
import pytest

from . import engine as e, verify as v
from .test_engine import initialized, saved_root, install_source


@pytest.mark.parametrize("damage", ["z", "y", "facts", "actor", "horizon", "terminal", "trace", "source", "projection"])
def test_root_audit_rejects_corrupted_receipt(tmp_path, monkeypatch, saved_root, damage):
    root, _ = install_source(tmp_path, monkeypatch, saved_root)
    ref = e.collect(root["spec"], "fixture", tmp_path)
    record = e.c.verified(ref)
    point = record["worlds"][0]["candidates"][0]["4"]
    if damage == "z": point["z"] = .123
    elif damage == "y": record["y"][0][0] = .123
    elif damage == "facts": point["public"]["players"][0]["tokens"][0] += 1
    elif damage == "actor": record["seat"] = 1-record["seat"]
    elif damage == "horizon": point["horizon"] = 5
    elif damage == "terminal": point["plies"] += 1
    elif damage == "trace": point["trace_sha256"] = "wrong"
    elif damage == "source": record["source"]["sha256"] = "wrong"
    else: record["statistics"]["reference"] = 2
    path = tmp_path/"corrupt.json"; e.old.save_new(path, record)
    with pytest.raises(ValueError): v.audit(root["spec"], e.c.artifact(path), "fixture", tmp_path)


@pytest.mark.parametrize("damage", ["cost", "order", "terminal", "world", "phase", "endpoint"])
def test_timing_audit_rejects_corruption(tmp_path, monkeypatch, saved_root, damage):
    root, _ = install_source(tmp_path, monkeypatch, saved_root)
    ref = e.timing(root["spec"], "fixture", tmp_path)
    record = e.c.verified(ref)
    if damage == "cost": record["timing"]["full_only"]["cpu"] += 1
    elif damage == "order": record["cells"][0]["samples"].reverse()
    elif damage == "terminal": record["cells"][0]["samples"][0]["result"]["outcome"]["turns"] += 1
    elif damage == "world": record["cells"][0]["world"] += 1
    elif damage == "phase": record["phase"] += 1
    else: record["cells"][0]["samples"][1]["result"]["endpoints"]["4"]["z"] = .123
    path = tmp_path/"corrupt_timing.json"; e.old.save_new(path, record)
    with pytest.raises(ValueError): v.audit_timing(root["spec"], e.c.artifact(path), "fixture", tmp_path)


def test_independent_value_math_does_not_call_collector(monkeypatch):
    state = e.old.SplendorEnv(seed=2289900777).initial_state()
    state.players[0].score = 5
    point = e.endpoint(state, 0, 4, 4, "horizon", None, "fixture")
    monkeypatch.setattr(e, "public_features", lambda *a: pytest.fail("Collector feature helper used"))
    assert v.verify_endpoint(point, 0, 4, e._COEFFICIENTS.tolist()) == pytest.approx(point["z"], abs=1e-12)
    bad = copy.deepcopy(point); bad["features"][0] += .1
    with pytest.raises(ValueError): v.verify_endpoint(bad, 0, 4, e._COEFFICIENTS.tolist())
