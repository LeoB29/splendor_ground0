"""Fixed experiment, private information, full replay, failure and spawn contracts."""
import copy
from pathlib import Path
import sys
import time

import pytest

from scripts.experiments import learned_proposal_attribution as p
from scripts.experiments import verify_learned_proposal_attribution as a


@pytest.fixture(scope="module", autouse=True)
def models():
    p.initialize_models()


def private_state(rich=False):
    env = p.old.SplendorEnv(seed=2059800000)
    state = env.initial_state()
    for _ in range(2):
        move = next(v for v in env.legal_actions(state) if v.action_type.name == "RESERVE_DECK" and v.tier == 2)
        state = env.step(state, move)
    if rich:
        for player in state.players:
            player.score = 14
            player.bonuses = {color: 5 for color in player.bonuses}
    return state


def test_schedule_balanced_disjoint_and_independent():
    seen = set()
    for stage, count in (("probe", 12), ("main", 402), ("test", 6)):
        rows = p.specs(stage)
        assert rows == a.independent_specs(stage) and len(rows) == count
        seeds = {r["environment_seed"] for r in rows}
        assert len(seeds) == count // 2 and not seen & seeds
        seen |= seeds
        for fit in p.MODELS:
            assert sum(r["fit"] == fit for r in rows) == count // 3
        for left, right in zip(rows[::2], rows[1::2]):
            assert left["fit"] == right["fit"] and left["search_stream"] == right["search_stream"]
            assert left["environment_seed"] == right["environment_seed"]
            assert (left["candidate_seat"], right["candidate_seat"]) == (0, 1)


@pytest.mark.parametrize("fit", [161001, 161002, 161003])
@pytest.mark.parametrize("seat", [0, 1])
def test_exact_screen_intervention_private_information_and_both_seats(fit, seat):
    state = private_state()
    state.current_player = seat
    before = p.old.digest(p.old.serialize_state_snapshot(state))
    actual = p.learned_proposal(state, "proposal-fixture", fit)
    expected = a.independent_proposal(state, "proposal-fixture", fit)
    a.compare({k: v for k, v in actual.items() if k != "timing"}, expected)
    saved_screen = p.screen.proposals(state, "proposal-fixture")
    offered = actual["candidates"]
    assert offered["keys"] == saved_screen["arm_keys"][f"learned_{fit}"]
    assert offered["actions"] == [saved_screen["union_actions"][i] for i in saved_screen["arm_indices"][f"learned_{fit}"]]
    assert offered["actions"][:3] == actual["reference"]["actions"][:3]
    alternative = a.a157.hidden_alternative(state)
    assert p.old.digest(p.old.serialize_state_snapshot(alternative)) != before
    changed = p.learned_proposal(alternative, "proposal-fixture", fit)
    a.compare({k: v for k, v in actual.items() if k != "timing"}, {k: v for k, v in changed.items() if k != "timing"})
    assert p.old.digest(p.old.serialize_state_snapshot(state)) == before


@pytest.mark.parametrize("size", [1, 2, 3, 4])
def test_short_support_and_coincidences(size):
    keys = list("abcd")[:size]
    assert p.fourth_slot(keys, keys, [0.] * size) == keys


def test_canonical_order_ties_and_invalid_support():
    assert p.fourth_slot(list("abcdef"), list("abce"), [0.] * 6) == list("abcd")
    with pytest.raises(ValueError):
        p.fourth_slot(list("abcde"), list("abc"), [0.] * 5)
    with pytest.raises(ValueError):
        p.fourth_slot(list("abcde"), list("aabc"), [0.] * 5)


@pytest.mark.parametrize("fit", [161001, 161002, 161003])
def test_candidate_delegates_to_exact_successful_player(tmp_path, fit):
    state = private_state(True)
    before = p.old.digest(p.old.serialize_state_snapshot(state))
    original_config = copy.deepcopy(p.baseline.CONFIG)
    original_action, original = p.previous.choose_root(state, "exact-candidate-fixture", tmp_path / "original.json", "test", fit)
    actual_action, actual = p.choose_root(state, "exact-candidate-fixture", tmp_path / "candidate.json", "test", fit)
    assert actual_action == original_action
    for key in ("snapshot", "schedule", "privacy", "scores", "candidates", "worlds", "choice", "means"):
        assert actual[key] == original[key]
    assert p.old.digest(p.old.serialize_state_snapshot(state)) == before
    assert p.baseline.CONFIG == original_config


@pytest.mark.parametrize("seat", [0, 1])
def test_raw_rule_matches_frozen_screen_preserves_slots_and_hidden_information(seat):
    state = private_state()
    state.current_player = seat
    before = p.old.digest(p.old.serialize_state_snapshot(state))
    actual = p.raw_proposal(state, "raw-control-fixture")
    expected = a.independent_raw_proposal(state, "raw-control-fixture")
    a.compare({k: v for k, v in actual.items() if k != "timing"}, expected)
    screen = p.screen.proposals(state, "raw-control-fixture")
    assert actual["candidates"]["keys"] == screen["arm_keys"]["raw_control"]
    assert actual["candidates"]["actions"] == [screen["union_actions"][i] for i in screen["arm_indices"]["raw_control"]]
    assert actual["candidates"]["actions"][:3] == actual["reference"]["actions"][:3]
    alternative = a.a157.hidden_alternative(state)
    assert p.old.digest(p.old.serialize_state_snapshot(alternative)) != before
    altered = p.raw_proposal(alternative, "raw-control-fixture")
    a.compare({k: v for k, v in actual.items() if k != "timing"}, {k: v for k, v in altered.items() if k != "timing"})
    assert p.old.digest(p.old.serialize_state_snapshot(state)) == before


@pytest.mark.parametrize("size", [1, 2, 3, 4])
def test_raw_short_support_and_representative_order(size):
    keys = list("abcd")[:size]
    assert p.raw_fourth_slot(keys, keys, list(reversed(keys))) == keys


def test_raw_common_random_third_is_not_reordered_and_ranking_must_cover_groups():
    assert p.raw_fourth_slot(list("abcdef"), list("abef"), list("abcdef")) == list("abec")
    assert p.raw_fourth_slot(list("abcdef"), list("abcf"), list("abcdef")) == list("abcd")
    with pytest.raises(ValueError):
        p.raw_fourth_slot(list("abcdef"), list("abcd"), list("abcdee"))


def test_raw_score_ties_use_legal_order_then_first_occurrence(monkeypatch):
    state = private_state()
    monkeypatch.setattr(p.old._RAW, "score_actions", lambda state, actions: [0.] * len(actions))
    legal = p.old.SplendorEnv().legal_actions(state)
    expected = list(dict.fromkeys(repr(p.old.canonical_successor_action_key(state, action)) for action in legal))
    actual = p.raw_proposal(state, "raw-tie-fixture")
    assert actual["raw_rank"] == expected
    a.compare({k: v for k, v in actual.items() if k != "timing"}, a.independent_raw_proposal(state, "raw-tie-fixture"))


def test_raw_player_does_not_call_a_161_model(tmp_path, monkeypatch):
    monkeypatch.setattr(p.screen, "_MODELS", {})
    _, record = p.choose_root(private_state(True), "035-only-fixture", tmp_path / "raw.json", "test")
    assert record["proposal"]["rule"] == "original035"
    assert "learned_logits" not in record["proposal"]


@pytest.fixture(scope="module")
def tiny_games(tmp_path_factory):
    base = tmp_path_factory.mktemp("proposal_arena_games")
    snapshot = p.old.serialize_state_snapshot(private_state(True))
    p.old.save_new(base / "test_launch.json", {"manifest": "synthetic-binding"})
    games = [p.play_game("test", spec, "synthetic-binding", base, snapshot) for spec in p.specs("test")]
    return base, snapshot, games


def test_all_fits_and_seats_full_game_every_world_replay(tiny_games):
    base, snapshot, games = tiny_games
    for item in games:
        receipt = a.audit_game("test", item["index"], item["sha256"], "synthetic-binding", base, snapshot)
        assert receipt["game"]["fit"] == p.specs("test")[item["index"]]["fit"]
        assert {r["planner"] for r in receipt["roots"]} == {"candidate", "reference"}
        assert all(r["continuations"] == r["replayed_branches"] == 32*r["support"] for r in receipt["roots"])
        assert receipt["roots"][0]["privacy"] == {"own": 1, "opponent": 1}


@pytest.mark.parametrize("damage", ["world", "last_trace", "means", "choice", "winner", "proposal", "outer_state"])
def test_corruption_rejected_even_after_rehashing(tiny_games, tmp_path, damage):
    base, snapshot, games = tiny_games
    game = p.old.read_json(p.game_path("test", 0, base))
    p.old.save_new(tmp_path / "test_launch.json", {"manifest": "synthetic-binding"})
    for move in game["trace"]:
        value = p.old.read_json(p.root_path("test", 0, move["turn"], base))
        p.old.save_new(p.root_path("test", 0, move["turn"], tmp_path), value)
    path = p.root_path("test", 0, game["trace"][0]["turn"], tmp_path)
    root = p.old.read_json(path)
    if damage == "world":
        root["worlds"][-1]["state_sha256"] = "0" * 64
    elif damage == "last_trace":
        root["worlds"][-1]["outcomes"][-1]["trace_sha256"] = "0" * 64
    elif damage == "means":
        root["means"][0] += .125
    elif damage == "choice":
        root["choice"] = (root["choice"] + 1) % len(root["means"])
    elif damage == "winner":
        outcome = root["worlds"][-1]["outcomes"][-1]
        outcome["winner"] = None if outcome["winner"] is not None else 0
    elif damage == "proposal":
        root["proposal"]["learned_logits"][0] += .5
    else:
        game["trace"][0]["state_sha256"] = "0" * 64
    p.old.atomic_write(path, root)
    game["trace"][0]["root_sha256"] = p.old.file_hash(path)
    target = p.game_path("test", 0, tmp_path)
    p.old.save_new(target, game)
    with pytest.raises((AssertionError, ValueError)):
        a.audit_game("test", 0, p.old.file_hash(target), "synthetic-binding", tmp_path, snapshot)


def test_no_duplicate_or_production_fixture(tiny_games):
    base, snapshot, games = tiny_games
    path = p.game_path("test", 0, base)
    unchanged = path.read_bytes()
    with pytest.raises(FileExistsError):
        p.play_game("test", p.specs("test")[0], "synthetic-binding", base, snapshot)
    assert path.read_bytes() == unchanged
    with pytest.raises(ValueError):
        p.game_path("test", 0, p.OUTPUT)
    with pytest.raises(ValueError):
        p.play_game("main", p.specs("main")[0], "test", base, snapshot)


def test_control_proposal_corruption_rejected_after_rehashing(tiny_games, tmp_path):
    base, snapshot, games = tiny_games
    game = p.old.read_json(p.game_path("test", 1, base))
    assert game["trace"][0]["actor"] != game["spec"]["candidate_seat"]
    p.old.save_new(tmp_path / "test_launch.json", {"manifest": "synthetic-binding"})
    for move in game["trace"]:
        value = p.old.read_json(p.root_path("test", 1, move["turn"], base))
        p.old.save_new(p.root_path("test", 1, move["turn"], tmp_path), value)
    path = p.root_path("test", 1, game["trace"][0]["turn"], tmp_path)
    root = p.old.read_json(path)
    root["proposal"]["raw_rank"] = list(reversed(root["proposal"]["raw_rank"]))
    p.old.atomic_write(path, root)
    game["trace"][0]["root_sha256"] = p.old.file_hash(path)
    target = p.game_path("test", 1, tmp_path)
    p.old.save_new(target, game)
    with pytest.raises((ValueError, AssertionError)):
        a.audit_game("test", 1, p.old.file_hash(target), "synthetic-binding", tmp_path, snapshot)


@pytest.mark.parametrize("fit", [None, 161001])
def test_failed_root_preserved_and_retry_rejected(tmp_path, monkeypatch, fit):
    def bad(*args):
        return {"reason": "max_turns", "winner": None, "turns": 3, "trace_sha256": "a" * 64, "final_state_sha256": "b" * 64}
    monkeypatch.setattr(p.old, "trajectory", bad)
    path = tmp_path / "failed.json"
    with pytest.raises(ValueError):
        p.choose_root(private_state(True), "test", path, "test", fit)
    record = p.old.read_json(path)
    assert record["failure"] and record["worlds"][0]["outcomes"][0]["wdl"] is None
    original = path.read_bytes()
    with pytest.raises(ValueError):
        p.choose_root(private_state(True), "test", path, "test", fit)
    assert path.read_bytes() == original


def statistic_rows():
    return [{"index": r["index"], "pair": r["pair"], "seat": r["candidate_seat"], "fit": r["fit"],
             "score": (.5 if r["index"] % 11 == 0 else float(r["index"] % 7 < 4))} for r in p.specs("main")]


def test_independent_pair_stratified_statistics_and_decisions():
    rows = statistic_rows()
    a.compare(p.statistics(rows), a.independent_statistics(rows))
    assert p.decision(.56, [.51, .61])["practical_progress"]
    assert p.decision(.53, [.51, .55])["classification"] == "positive_below_observed_target"
    assert p.decision(.49, [.44, .54])["classification"] == "below_material_target"
    assert p.decision(.52, [.47, .57])["classification"] == "inconclusive"
    assert not p.decision(.55, [.5, .6])["superiority_established"]
    rows[1]["pair"] = 5
    with pytest.raises(ValueError):
        p.statistics(rows)


def test_timing_and_storage_gate_has_no_strength_inputs():
    lock = {"elapsed_seconds": 100}
    assert p.cost_projection(lock, lock, 1000, 20 * 1024**3)["reasonable"]
    assert not p.cost_projection({"elapsed_seconds": 10000}, lock, 1000, 20 * 1024**3)["reasonable"]
    assert not p.cost_projection(lock, lock, 1000, 5 * 1024**3)["reasonable"]


def test_invalid_freeze_cannot_launch(tmp_path, monkeypatch):
    monkeypatch.setattr(p, "OUTPUT", tmp_path)
    launched = []
    monkeypatch.setattr(p.windows, "detached_launch", lambda *args: launched.append(args))
    with pytest.raises(FileNotFoundError):
        p.launch()
    assert not launched


def test_two_actual_hidden_spawn_pools_collect_and_fully_replay(tmp_path):
    base = tmp_path / "synthetic"
    base.mkdir()
    fixture = {"kind": "learned-proposal-attribution-synthetic-v1", "base": str(base), "binding": "spawn-fixture-only",
               "snapshot": p.old.serialize_state_snapshot(private_state(True))}
    path = tmp_path / "fixture.json"
    p.old.save_new(path, fixture)
    launch = p.windows.detached_launch([sys.executable, "-u", "-m", "scripts.experiments.verify_learned_proposal_attribution", "--fixture", path],
                                       base / "out.txt", base / "err.txt", p.ROOT)
    result_path = base / "fixture_result.json"
    deadline = time.monotonic() + 120
    while not result_path.exists():
        if time.monotonic() > deadline:
            errors = [path.read_text() for path in base.rglob("*err*.txt")]
            pytest.fail(f"Synthetic hidden pools timed out PID {launch['pid']}: {errors}")
        time.sleep(.05)
    result = p.old.read_json(result_path)
    assert result["passed"] and result["games"] == 6
    assert result["continuations"] == result["replayed_branches"] and result["roots"] >= 12
    for kind in ("test_games", "test_audits"):
        healths = list((base / "workers" / kind).glob("*_health.json"))
        assert healths
        for path in healths:
            health = p.old.read_json(path)
            if sys.platform == "win32":
                assert Path(health["executable"]).name == "pythonw.exe"
                assert Path(health["parent_executable"]).name == "pythonw.exe"
                assert health["console_window"] == health["console_process_count"] == 0
