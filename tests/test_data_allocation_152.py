import copy
from concurrent.futures import ThreadPoolExecutor
import importlib.util
from pathlib import Path

import pytest
import torch

from splendor_ai.alphazero import run_data_allocation as p


def module(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    result = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(result)
    return result


@pytest.fixture(scope="module")
def audit():
    return module("audit152_test", "scripts/analysis/verify_data_allocation_152.py")


@pytest.fixture(scope="module")
def helpers():
    return module("helpers151_test", "tests/test_shared_cards_151.py")


def test_nested_first_eight_ignores_later_worlds(monkeypatch, tmp_path):
    monkeypatch.setattr(p.old, "OUTPUT", tmp_path)
    sample = {"index": 0, "actions": [{}, {}]}
    panel = {"samples": [sample]}
    records = [{"outcomes": [{"wdl": 0., "reason": "completed"},
                              {"wdl": (1. if w < 4 or w >= 8 else -1.), "reason": "completed"}]} for w in range(32)]
    path = tmp_path/"targets/train/state_0000.json"
    p.old.save_new(path, {"sample_sha256": p.old.digest(sample), "worlds": records})
    p.old.save_new(tmp_path/"train_targets_lock.json", {"files": {"0": {"sha256": p.old.file_hash(path)}}})
    q, y = p.first_eight_targets(panel)
    assert torch.equal(q, torch.zeros(1, 4)) and torch.equal(q, y)
    assert sum(w["outcomes"][1]["wdl"] for w in records)/32 == .75
    sample["actions"].append({})
    with pytest.raises(ValueError, match="Wrong original"):
        p.first_eight_targets(panel)


def test_concatenation_preserves_every_prefix_and_feature(helpers):
    small, extra = helpers.synthetic(3), helpers.synthetic(4, 153)
    broad = p.concatenate_data(small, extra)
    for key in broad:
        assert torch.equal(broad[key][:3], small[key])
        assert torch.equal(broad[key][3:], extra[key])
    del extra["q"]
    with pytest.raises(ValueError, match="schema"):
        p.concatenate_data(small, extra)


def test_unchanged_fit_and_independent_exposure(audit, helpers):
    train, val = helpers.synthetic(7), helpers.synthetic(3, 154)
    fitted = p.parent.fit(149001, train, val, updates=8)
    independent = audit.a151.independent_fit(149001, train, val, total=8)
    audit.compare(fitted, independent)
    expected = audit.independent_exposure(149001, 7, fitted["selected_update"], total=8)
    audit.compare(p.exposure(149001, 7, fitted["selected_update"], total=8), expected)
    assert expected["final"]["unique_states"] == expected["final"]["unique_games"] == 7
    assert expected["final"]["draws"] == 512


def test_full_grid_exposure_counts_all_states(audit):
    exposure = p.exposure(149002, 3072, 17)
    assert len(exposure["history"]) == 160 and exposure["final"]["draws"] == 262144
    assert exposure["selected"]["unique_states"] <= 17*64
    assert exposure["final"]["unique_states"] == 3072
    audit.compare(exposure, audit.independent_exposure(149002, 3072, 17))


def test_source_behavior_matches_independent_rng_and_keeps_greedy_target(monkeypatch, audit):
    real_env = p.old.SplendorEnv

    class RichEnv(real_env):
        def initial_state(self):
            state = super().initial_state()
            for player in state.players:
                player.bonuses = {c: 5 for c in player.bonuses}
                player.score = 14
            return state

    def scores(state, legal):
        return [10. if a.action_type.name.startswith("BUY") else -10. for a in legal]

    class FakeRaw:
        score_actions = staticmethod(scores)

    class FakeSource:
        def full_prediction(self, state, legal):
            return torch.tensor(scores(state, legal)), 0., {"observation": torch.zeros(295)}, None, None

    monkeypatch.setattr(p.old, "SplendorEnv", RichEnv)
    monkeypatch.setattr(p.old, "_RAW", FakeRaw())
    monkeypatch.setattr(p, "_SOURCE", FakeSource())
    original_raw = p.old._RAW
    record = p.source_record(0)
    assert record["reason"] == "completed"
    audit.audit_source_record(record)
    assert p.old._RAW is original_raw
    broken = copy.deepcopy(record)
    broken["trace"][0]["selected_row"] += 1
    with pytest.raises(AssertionError):
        audit.audit_source_record(broken)


def test_generator_only_prefetches_final_batch(monkeypatch, tmp_path):
    monkeypatch.setattr(p, "OUTPUT", tmp_path)
    monkeypatch.setattr(p, "ProcessPoolExecutor", ThreadPoolExecutor)
    monkeypatch.setattr(p, "initialize_worker", lambda: None)
    monkeypatch.setitem(p.CONFIG, "source_limit", 16)

    def task(index):
        p.old.save_new(tmp_path/"sources"/f"game_{index:04d}.json", {"index": index})
        return {"index": index, "reason": "completed", "sha256": "stub"}

    def select(records):
        assert [next(records)["index"] for _ in range(3)] == [0, 1, 2]
        return {"filtering": [0, 1, 2]}

    monkeypatch.setattr(p, "source_task", task)
    monkeypatch.setattr(p, "select_panel", select)
    p.generate_panel()
    lock = p.old.read_json(tmp_path/"sources_lock.json")
    assert len(lock["files"]) == 8 and lock["unused_prefetched"] == 5
    assert not (tmp_path/"sources/game_0008.json").exists()


def test_non_natural_prefetched_source_stops_before_selection(monkeypatch, tmp_path):
    monkeypatch.setattr(p, "OUTPUT", tmp_path)
    monkeypatch.setattr(p, "ProcessPoolExecutor", ThreadPoolExecutor)
    monkeypatch.setattr(p, "initialize_worker", lambda: None)
    monkeypatch.setitem(p.CONFIG, "source_limit", 8)
    monkeypatch.setattr(p, "source_task", lambda i: {"index": i, "reason": "max_turns" if i == 7 else "completed"})
    with pytest.raises(ValueError, match="Non-natural attempted"):
        p.generate_panel()


def screen_case():
    q = torch.tensor([[0., .25, -.5, 1.]]*4)
    rows = {arm: [{"seed": seed, "selected_update": 17,
                   "advantages": (q if arm == "C" else torch.zeros_like(q)).tolist(),
                   "choices": [1 if arm == "C" else 0]*4} for seed in (149001,149002,149003)] for arm in ("A", "B", "C")}
    return {"q": q, "y": q, "mask": torch.tensor([[False, True, True, False]]*4)}, rows


def test_independent_statistics_and_mechanism_contrasts(audit):
    data, rows = screen_case()
    result = p.statistics(data, rows)
    assert result["development_passed"] and result["C_better_than_B_on_both_means"]
    audit.compare(result, audit.independent_statistics(data, rows), tolerance=1e-12)
    rows["B"] = copy.deepcopy(rows["C"])
    result = p.statistics(data, rows)
    assert result["development_passed"] and not result["C_better_than_B_on_both_means"]
    assert result["contrasts"]["C_minus_B"] == {"mse_difference": 0., "wdl_difference": 0.}


@pytest.mark.parametrize("gate", ["mse_versus_A", "mse_versus_zero", "wdl_versus_035", "wdl_versus_A"])
def test_every_primary_gate_required(monkeypatch, gate):
    a = {"mse": .1, "wdl_gain_over_035": .01}
    c = {"mse": .05, "wdl_gain_over_035": .03}
    zero = .1
    if gate == "mse_versus_A": a["mse"] = .05
    elif gate == "mse_versus_zero": zero = .05
    elif gate == "wdl_versus_035": c["wdl_gain_over_035"] = .019
    else: a["wdl_gain_over_035"] = .03
    monkeypatch.setattr(p.parent, "metrics", lambda *args: {"zero_mse": zero, "arms": {"A": {"means": a}, "B": {"means": a}, "C": {"means": c}}})
    result = p.statistics(None, None)
    assert not result["development_passed"]
    assert [key for key, passed in result["development_gates"].items() if not passed] == [gate]


def test_namespace_budgets_and_no_test_split():
    sources = set(range(1520000000, 1520008192))
    new = {p.world_seed("new", s, w) for s in range(2304) for w in range(8)}
    probe = {p.world_seed("probe", s, w) for s in range(6) for w in range(4)}
    assert len(new) == 18432 and len(probe) == 24
    assert not sources & new and not sources & probe and not new & probe
    assert min(new) > 1519105003  # beyond151's largest namespace
    assert 768*32*4 == 3072*8*4 == 98304
    assert 2304*8*4 == p.CONFIG["new_continuations_max"]
    with pytest.raises(ValueError, match="performance-test"):
        p.world_seed("test", 0, 0)
    with pytest.raises(ValueError, match="budget"):
        p.collect({"split": "probe", "samples": [{"index": 0, "actions": [None]*25}]})
    with pytest.raises(ValueError, match="No performance"):
        p.collect({"split": "test", "samples": []})


def test_collector_replays_independently_and_rejects_tampering(monkeypatch, tmp_path, audit):
    states = module("positions152_test", "tests/test_afterstate_advantage_149.py").positions()
    state = states[2]
    action = next(a for a in p.old.SplendorEnv().legal_actions(state) if a.noble_id is not None)
    sample = {"index": 1, "snapshot": p.old.serialize_state_snapshot(state), "actions": [p.old.serialize_action(action)]}
    monkeypatch.setattr(p, "OUTPUT", tmp_path)
    info = p.collect_task("probe", sample)
    result = audit.audit_sample("probe", sample, info)
    assert result["continuations"] == 4 and result["reasons"] == {"completed": 4}
    info["moves"] += 1
    with pytest.raises(AssertionError):
        audit.audit_sample("probe", sample, info)


def configure_fake_run(monkeypatch, tmp_path, dev_pass, fail_panel=False):
    events = []
    monkeypatch.setattr(p, "OUTPUT", tmp_path)
    monkeypatch.setattr(p.parent, "OUTPUT", tmp_path/"prior")
    p.old.save_new(tmp_path/"prior/selection_lock.json", {"models": {str(s): {"mock_control": True} for s in (149001,149002,149003)}})
    monkeypatch.setattr(p, "check_identity", lambda: "bound")
    monkeypatch.setattr(p, "initialize_worker", lambda: None)
    monkeypatch.setattr(p, "generate_panel", lambda: {"split": "new", "samples": []})
    monkeypatch.setattr(p.old, "feature_data", lambda panel: {"x": torch.zeros(1,4)})
    monkeypatch.setattr(p.parent, "structured_data", lambda data: data)
    monkeypatch.setattr(p.old, "target_data", lambda *args, **kwargs: (torch.zeros(1,4),torch.zeros(1,4)))
    data = {k: torch.zeros(1,4) for k in ("x","q","y")}
    monkeypatch.setattr(p, "load_data", lambda *args: data)
    monkeypatch.setattr(p, "ProcessPoolExecutor", ThreadPoolExecutor)
    monkeypatch.setattr(p, "fit_task", lambda *args: {"mock_fit": True})
    monkeypatch.setattr(p, "predictions", lambda *args: {})
    monkeypatch.setattr(p, "statistics", lambda *args: {"development_passed": dev_pass})
    preflight = {"passed": True, "manifest": "bound"}
    for key, name in (("probe_audit_sha256", "probe_audit.json"), ("initial_data_lock_sha256", "initial_data_lock.json"), ("timing_sha256", "timing_probe.json")):
        p.old.save_new(tmp_path/name, {})
        preflight[key] = p.old.file_hash(tmp_path/name)
    p.old.save_new(tmp_path/"preflight.json", preflight)
    p.old.save_new(tmp_path/"sources_lock.json", {})
    for name in ("B", "validation"):
        p.old.tensor_save(tmp_path/f"{name}_data.pt", data)

    def collect(panel):
        assert "panel_audit" in events
        events.append("labels")
        p.old.save_new(tmp_path/"new_targets_lock.json", {})

    def audit(stage):
        events.append(stage+"_audit")
        if stage == "panel" and fail_panel:
            raise ValueError("panel audit rejected")
        result = {"passed": True, "manifest": "bound", "new_continuations": 0}
        if stage == "final":
            for field, name in (("result_sha256", "result.json"), ("selection_sha256", "selection_lock.json"), ("predictions_sha256", "validation_predictions.json")):
                result[field] = p.old.file_hash(tmp_path/name)
        p.old.save_new(tmp_path/f"{stage}_audit.json", result)
        return result

    monkeypatch.setattr(p, "collect", collect)
    monkeypatch.setattr(p, "audit_stage", audit)
    return events


def test_no_labels_if_source_panel_audit_fails(monkeypatch, tmp_path):
    events = configure_fake_run(monkeypatch, tmp_path, False, fail_panel=True)
    with pytest.raises(ValueError, match="panel audit rejected"):
        p.run()
    assert events == ["panel_audit"]
    assert not (tmp_path/"labels_launch.json").exists()


@pytest.mark.parametrize("passed", [False, True])
def test_complete_pipeline_always_audits_and_never_opens_performance_test(monkeypatch, tmp_path, passed):
    events = configure_fake_run(monkeypatch, tmp_path, passed)
    p.run()
    assert events == ["panel_audit", "labels", "final_audit"]
    status = p.old.read_json(tmp_path/"final_status.json")
    assert status["complete"] and status["development_passed"] == passed
    assert not status["fresh_performance_test"] and not status["arena"] and not status["automatic_successor"]
