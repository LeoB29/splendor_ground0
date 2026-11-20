"""Exact128-world player, bounded execution, provenance and failure contracts."""
import copy
from pathlib import Path

import pytest

from tools.learned_proposal_128 import experiment as p


@pytest.fixture(scope="module", autouse=True)
def models():
    p.initialize_models()


def private_state(rich=False):
    env = p.old.SplendorEnv(seed=2069800000)
    state = env.initial_state()
    for _ in range(2):
        move = next(v for v in env.legal_actions(state)
                    if v.action_type.name == "RESERVE_DECK" and v.tier == 2)
        state = env.step(state, move)
    if rich:
        for player in state.players:
            player.score = 14
            player.bonuses = {color: 5 for color in player.bonuses}
    return state


def test_balanced_independent_schedule_and_disjoint_namespaces():
    seen = set()
    for stage, count in (("probe", 12), ("main", 402), ("test", 6)):
        rows = p.specs(stage)
        assert rows == p.auditor().independent_specs(stage) and len(rows) == count
        seeds = {r["environment_seed"] for r in rows}
        assert len(seeds) == count // 2 and not seeds & seen
        seen.update(seeds)
        for fit in p.MODELS:
            assert sum(r["fit"] == fit for r in rows) == count // 3
        for left, right in zip(rows[::2], rows[1::2]):
            assert left["fit"] == right["fit"] and left["search_stream"] == right["search_stream"]
            assert left["environment_seed"] == right["environment_seed"]
            assert (left["candidate_seat"], right["candidate_seat"]) == (0, 1)
    assert p.CONFIG["bootstrap_seed"] == 2063000000
    assert p.CONFIG["audit_seed"] == 2064000000


@pytest.mark.parametrize("fit", [161001, 161002, 161003])
@pytest.mark.parametrize("seat", [0, 1])
def test_unchanged_learned_proposal_and_exact160_schedule(fit, seat):
    state = private_state()
    state.current_player = seat
    original_config = copy.deepcopy(p.baseline.CONFIG)
    before = p.old.digest(p.old.serialize_state_snapshot(state))
    original = p.previous.learned_proposal(state, "proposal128-fixture", fit)
    actual = p.learned_proposal(state, "proposal128-fixture", fit)
    assert actual["schedule"] == p.world128.root_schedule(state, "proposal128-fixture", 128)
    assert actual["schedule"]["world_seeds"][:32] == original["schedule"]["world_seeds"]
    assert len(actual["schedule"]["world_seeds"]) == 128
    for name in ("reference", "candidates", "fit", "group_keys", "legal_actions",
                 "learned_logits", "learned_group_scores"):
        assert actual[name] == original[name]
    expected = p.auditor().independent_proposal(state, "proposal128-fixture", fit)
    p.auditor().compare({k: v for k, v in actual.items() if k != "timing"}, expected)
    alternative = p.auditor().a157.hidden_alternative(state)
    changed = p.learned_proposal(alternative, "proposal128-fixture", fit)
    p.auditor().compare({k: v for k, v in actual.items() if k != "timing"},
                        {k: v for k, v in changed.items() if k != "timing"})
    assert p.old.digest(p.old.serialize_state_snapshot(state)) == before
    assert p.baseline.CONFIG == original_config


@pytest.mark.parametrize("fit", [161001, 161002, 161003])
def test_candidate_preserves_first32_trajectories_and_selects_all128(tmp_path, fit):
    state = private_state(True)
    before = p.old.digest(p.old.serialize_state_snapshot(state))
    _, previous = p.previous.choose_root(state, "128-prefix-fixture", tmp_path / "prior.json", "test", fit)
    action, actual = p.choose_root(state, "128-prefix-fixture", tmp_path / "new.json", "test", fit)
    assert actual["candidates"] == previous["candidates"]
    assert actual["worlds"][:32] == previous["worlds"]
    assert len(actual["worlds"]) == 128
    selection = p.world128.select(actual["worlds"], len(actual["candidates"]["actions"]), 128)
    assert actual["choice"] == selection["choice"] and actual["means"] == selection["means"]
    assert action == p.old.decode_action(actual["candidates"]["actions"][selection["choice"]])
    assert p.old.digest(p.old.serialize_state_snapshot(state)) == before


@pytest.mark.parametrize("seat", [0, 1])
def test_reference_is_exact_original160_including_second_random(tmp_path, seat):
    state = private_state(True)
    state.current_player = seat
    expected_action, expected = p.world128.choose_root(state, "exact160-fixture", tmp_path / "old.json", "test", 128)
    actual_action, actual = p.choose_root(state, "exact160-fixture", tmp_path / "new.json", "test")
    assert actual_action == expected_action
    for key in ("snapshot", "schedule", "privacy", "scores", "candidates", "worlds", "choice", "means"):
        assert actual[key] == expected[key]
    assert "proposal" not in actual and len(actual["worlds"]) == 128


def test_reference_has_no_161_inference(tmp_path, monkeypatch):
    monkeypatch.setattr(p.screen, "_MODELS", {})
    _, root = p.choose_root(private_state(True), "teacher-only", tmp_path / "root.json", "test")
    assert "proposal" not in root and len(root["worlds"]) == 128


@pytest.mark.parametrize("size", [1, 2, 3, 4])
def test_short_support_unchanged(size):
    keys = list("abcd")[:size]
    assert p.fourth_slot(keys, keys, [0.] * size) == keys


def test_learned_exact_ties_keep_canonical_order():
    assert p.fourth_slot(list("abcdef"), list("abce"), [0.] * 6) == list("abcd")
    with pytest.raises(ValueError):
        p.fourth_slot(list("abcdef"), list("aabc"), [0.] * 6)


@pytest.mark.parametrize("fit", [None, 161001])
def test_failed_root_preserved_without_retry(tmp_path, monkeypatch, fit):
    def invalid(*args):
        return {"reason": "max_turns", "winner": None, "turns": 3,
                "trace_sha256": "a" * 64, "final_state_sha256": "b" * 64}
    monkeypatch.setattr(p.old, "trajectory", invalid)
    path = tmp_path / "failed.json"
    with pytest.raises(ValueError):
        p.choose_root(private_state(True), "failure-fixture", path, "test", fit)
    root = p.old.read_json(path)
    assert root["failure"] and root["worlds"][0]["outcomes"][0]["wdl"] is None
    evidence = path.read_bytes()
    with pytest.raises(ValueError):
        p.choose_root(private_state(True), "failure-fixture", path, "test", fit)
    assert path.read_bytes() == evidence


def test_production_does_not_accept_synthetic_state(tmp_path):
    snapshot = p.old.serialize_state_snapshot(private_state(True))
    with pytest.raises(ValueError):
        p.game_path("test", 0, p.OUTPUT)
    with pytest.raises(ValueError):
        p.play_game("main", p.specs("main")[0], "test", tmp_path, snapshot)


def test_completed160_production_refusal_and_isolated_snapshot_guards(tmp_path):
    """Retain every legacy assertion under the now-completed production state."""
    from scripts.analysis import verify_world_budget_160 as a160
    teacher = p.world128
    snapshot = p.old.serialize_state_snapshot(private_state(True))
    spec = a160.independent_spec("main", 0)
    original = teacher.game_path("main", 0, teacher.OUTPUT)
    assert original.is_file()
    before = p.old.file_hash(original)
    with pytest.raises(FileExistsError):
        teacher.play_game("main", spec, "test", teacher.OUTPUT, snapshot)
    assert p.old.file_hash(original) == before
    isolated = tmp_path / "empty_original160_destination"
    with pytest.raises(ValueError):
        teacher.play_game("main", spec, "test", isolated, snapshot)
    assert not teacher.game_path("main", 0, isolated).exists()
    with pytest.raises(ValueError):
        teacher.game_path("test", 0, teacher.OUTPUT)
    with pytest.raises(ValueError):
        teacher.root_path("test", 0, 0, teacher.OUTPUT / "test_fixtures")
    with pytest.raises(AssertionError):
        a160.audit_game("main", 0, "a" * 64, "test", teacher.OUTPUT, snapshot)
    with pytest.raises(AssertionError):
        a160.audit_game("main", 0, "a" * 64, "test", isolated, snapshot)
    assert p.old.file_hash(original) == before


def test_independent_stratified_statistics():
    rows = [{"index": s["index"], "pair": s["pair"], "seat": s["candidate_seat"], "fit": s["fit"],
             "score": .5 if s["index"] % 11 == 0 else float(s["index"] % 7 < 4)}
            for s in p.specs("main")]
    p.auditor().compare(p.statistics(rows), p.auditor().independent_statistics(rows))
    rows[1]["pair"] = 999
    with pytest.raises(ValueError):
        p.statistics(rows)


def test_gate_interpretations():
    assert p.decision(.56, [.51, .61])["practical_progress"]
    assert p.decision(.53, [.51, .55])["classification"] == "positive_below_observed_target"
    assert p.decision(.49, [.44, .54])["classification"] == "below_material_target"
    assert p.decision(.52, [.47, .57])["classification"] == "inconclusive"
    assert not p.decision(.55, [.5, .6])["superiority_established"]


def test_outcome_blind_cost_separates_fixed_stress_from_game_scaling():
    value = p.cost_projection(12000, 4000, 2000, 100, 100000, 30 * 1024**3)
    expected = 1.25 / 8 * (33.5 * (12000 + 2000) + 162 * 20)
    assert value["forecast_seconds"] == expected and value["reasonable"]
    assert not p.cost_projection(1000000, 4000, 2000, 100, 100000, 30 * 1024**3)["reasonable"]
    assert not p.cost_projection(12000, 4000, 2000, 100, 100000, 5 * 1024**3)["reasonable"]
    for stress, count in ((0, 0), (0, 100), (2000, 0), (5000, 100)):
        with pytest.raises(ValueError):
            p.cost_projection(12000, 4000, stress, count, 100000, 30 * 1024**3)


def test_sleep_inhibition_releases_request_even_on_failure(monkeypatch):
    calls = []
    monkeypatch.setattr(p, "_set_sleep_state", lambda flags: calls.append(flags) or 0x80000000)
    with pytest.raises(RuntimeError):
        with p.inhibit_system_sleep() as receipt:
            assert receipt["system_required"] and not receipt["display_required"]
            raise RuntimeError("fixture")
    assert calls == [0x80000001, 0x80000000]


def test_sleep_inhibition_refusal_never_enters_run(monkeypatch):
    def refusal(flags):
        raise ValueError("refusal")
    monkeypatch.setattr(p, "_set_sleep_state", refusal)
    with pytest.raises(ValueError):
        with p.inhibit_system_sleep():
            pytest.fail("Entered experiment despite inability to inhibit sleep")


def test_invalid_freeze_cannot_launch(tmp_path, monkeypatch):
    monkeypatch.setattr(p, "OUTPUT", tmp_path)
    launches = []
    monkeypatch.setattr(p.windows, "detached_launch", lambda *args: launches.append(args))
    with pytest.raises(FileNotFoundError):
        p.launch()
    assert not launches


def test_retained_teacher_has_direct_completed_evidence():
    evidence = p.original_teacher_evidence()
    assert evidence["binding"] == "5e9e9202f44e7b526fff4be97cdcd3186805fbb5e8a184d21a6702de9d922df3"
    paths = {item["path"] for item in evidence["files"]}
    assert {f"outputs/world_budget_160/{name}" for name in
            ("final_status.json", "result.json", "freeze.json", "main_collect_lock.json", "main_audit_lock.json")}.issubset(paths)
    assert sum("/main/games/" in name for name in paths) == 400
    assert sum("/main/audits/" in name for name in paths) == 400


def coverage_fixture():
    selections = [{"root_id": "0000:000", "ordinary_world_indices": list(range(8)),
                   "replay_world_indices": list(range(128)), "stress": True},
                  {"root_id": "0000:001", "ordinary_world_indices": list(range(8)),
                   "replay_world_indices": list(range(8)), "stress": False}]
    inventory = [{"root_id": f"0000:{turn:03d}", "index": 0, "turn": turn,
                  "sha256": str(turn) * 64, "support": support}
                 for turn, support in ((0, 2), (1, 3))]
    roots = [{**metadata, **selection, "worlds": 128,
              "full_inner_replay": False, "full_fixture_replay": False,
              "continuations": recorded, "replayed_branches": replayed}
             for metadata, selection, recorded, replayed in
             zip(inventory, selections, (256, 384), (256, 24))]
    sample = {"inventory": inventory, "selections": selections,
              "coverage": {"games": 1, "roots": 2, "recorded_worlds": 256,
                           "recorded_continuations": 640, "replayed_worlds": 136,
                           "replayed_continuations": 280, "stress_roots": 1}}
    receipts = [{"sample_sha256": "s" * 64, "full_inner_replay": False, "roots": roots}]
    return receipts, sample


def test_finalization_reconciles_receipts_with_sealed_sample():
    receipts, sample = coverage_fixture()
    actual = p.checked_audit_coverage(receipts, sample, "s" * 64)
    assert actual == sample["coverage"]


@pytest.mark.parametrize("damage", ["sample_link", "missing_root", "count", "fixture", "sample_indices", "coverage"])
def test_finalization_rejects_audit_receipt_accounting_drift(damage):
    receipts, sample = coverage_fixture()
    if damage == "sample_link":
        receipts[0]["sample_sha256"] = "q" * 64
    elif damage == "missing_root":
        receipts[0]["roots"].pop()
    elif damage == "count":
        receipts[0]["roots"][1]["replayed_branches"] -= 1
    elif damage == "fixture":
        receipts[0]["roots"][0]["full_fixture_replay"] = True
    elif damage == "sample_indices":
        receipts[0]["roots"][1]["replay_world_indices"] = list(range(1, 9))
    else:
        sample["coverage"]["replayed_continuations"] += 1
    with pytest.raises(ValueError):
        p.checked_audit_coverage(receipts, sample, "s" * 64)
