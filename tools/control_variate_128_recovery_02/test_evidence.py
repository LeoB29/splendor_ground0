"""Bounded checks for interrupted-prefix reuse, without trajectory replay."""
from copy import deepcopy

import pytest

from . import evidence as e


@pytest.fixture(scope="module")
def inherited_root():
    record = e.c.old.read_json(e.PRIOR/"train/roots/state_0000.json")
    saved = e.c.verified(record["source"])
    coefficients = e.c.verified(record["evaluator"])["coefficients"]
    return record, record["spec"], saved, record["source"], record["evaluator"], coefficients


def test_saved_root_structure_and_public_values_without_policy(inherited_root, monkeypatch):
    monkeypatch.setattr(e.verify, "replay_prefix", lambda *a, **k: pytest.fail("No replay during authentication"))
    assert e.check_record(*inherited_root) is True


@pytest.mark.parametrize("mutation", ["incomplete", "world", "missing_candidate", "horizon", "reference", "public_value", "projection", "saved_y"])
def test_corrupted_saved_root_rejected(inherited_root, mutation):
    args = list(inherited_root)
    root = args[0] = deepcopy(args[0])
    if mutation == "incomplete": root["complete"] = False
    elif mutation == "world": root["worlds"][0]["seed"] += 1
    elif mutation == "missing_candidate": root["worlds"][0]["candidates"].pop()
    elif mutation == "horizon": root["worlds"][0]["candidates"][0].pop("16")
    elif mutation == "reference": root["reference"] = (root["reference"]+1) % root["support"]
    elif mutation == "public_value": root["worlds"][0]["candidates"][0]["4"]["z"] += .1
    elif mutation == "projection": root["statistics"]["z"]["4"][0][0] += .1
    elif mutation == "saved_y": root["y"][0][0] = .25
    with pytest.raises((ValueError, AssertionError)):
        e.check_record(*args)


def test_active_time_excludes_pause_but_includes_interruption_allowance():
    launch = {"time": "2026-10-09T13:43:29.351727+00:00", "epoch_seconds": 1791553409.351727,
              "previous_elapsed_seconds": 163.5683434009552}
    progress = {"time": "2026-10-09T13:48:49.246616+00:00"}
    assert e.prior_active_seconds(launch, progress) == pytest.approx(543.4632322788239)


@pytest.mark.parametrize("launch_time,progress_time,previous", [
    ("2026-10-09T13:43:29.351727", "2026-10-09T13:48:49.246616+00:00", 0),
    ("2026-10-09T13:43:29.351727+00:00", "2026-10-09T13:42:49.246616+00:00", 0),
    ("2026-10-09T13:43:29.351727+00:00", "2026-10-09T13:48:49.246616+00:00", -1),
])
def test_invalid_prior_runtime_rejected(launch_time, progress_time, previous):
    with pytest.raises((ValueError, AssertionError)):
        e.prior_active_seconds({"time": launch_time, "epoch_seconds": 1791553409.351727,
                                "previous_elapsed_seconds": previous}, {"time": progress_time})


def test_authentication_rejects_scientific_advancement_before_reading_outcomes(tmp_path, monkeypatch):
    monkeypatch.setattr(e, "PRIOR", tmp_path)
    monkeypatch.setattr(e.r1, "OUTPUT", tmp_path)
    (tmp_path/"selection_lock.json").write_text("{}")
    monkeypatch.setattr(e.r1, "validate_freeze", lambda: pytest.fail("Advanced parent must stop immediately"))
    with pytest.raises((ValueError, AssertionError), match="advanced or failed"):
        e.authenticate()
