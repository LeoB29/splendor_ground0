"""160 fixed-world-budget, game-integrity and independent paired-audit tests."""
import copy
from pathlib import Path
import sys
import time

import pytest

from splendor_ai.alphazero import world_budget_160 as h
from splendor_ai.alphazero import run_world_budget as p
from splendor_ai.alphazero import windowless_workers_160 as windowless
from splendor_ai.engine.actions import ActionType
from scripts.analysis import verify_world_budget_160 as a


@pytest.fixture(scope="module", autouse=True)
def raw_model():
    h.initialize_worker()


def rich_private_state():
    env = h.old.SplendorEnv(seed=1609800000)
    state = env.initial_state()
    for _ in range(2):
        action = next(move for move in env.legal_actions(state)
                      if move.action_type == ActionType.RESERVE_DECK and move.tier == 2)
        state = env.step(state, action)
    for player in state.players:
        player.score = 14
        player.bonuses = {color: 5 for color in player.bonuses}
    return state


@pytest.fixture(scope="module")
def collected_roots(tmp_path_factory):
    folder = tmp_path_factory.mktemp("160_nested_roots")
    state = rich_private_state()
    before = h.old.digest(h.old.serialize_state_snapshot(state))
    frozen_config = copy.deepcopy(h.baseline.CONFIG)
    reference_action, reference = h.baseline.choose_root(state, "160-test-nested", folder/"155.json", "test")
    low_action, low = h.choose_root(state, "160-test-nested", folder/"32.json", "test", 32)
    high_action, high = h.choose_root(state, "160-test-nested", folder/"128.json", "test", 128)
    assert h.baseline.CONFIG == frozen_config
    assert h.old.digest(h.old.serialize_state_snapshot(state)) == before
    return state, (reference_action, reference), (low_action, low), (high_action, high)


def test_exact155_reference_and128_nested_world_prefix(collected_roots):
    state, (reference_action, reference), (low_action, low), (_, high) = collected_roots
    assert reference_action == low_action
    for key in ("snapshot", "scores", "candidates", "privacy", "schedule", "worlds", "means", "choice"):
        assert reference[key] == low[key]
    for key in ("snapshot", "scores", "candidates", "privacy"):
        assert high[key] == low[key]
    assert high["schedule"]["candidate_nonce"] == low["schedule"]["candidate_nonce"]
    assert high["schedule"]["world_seeds"][:32] == low["schedule"]["world_seeds"]
    assert high["worlds"][:32] == low["worlds"]
    assert sum(len(w["outcomes"]) for w in high["worlds"]) == 4*sum(len(w["outcomes"]) for w in low["worlds"])
    assert high["means"] == [sum(w["outcomes"][j]["wdl"] for w in high["worlds"])/128
                              for j in range(len(high["candidates"]["actions"]))]


@pytest.mark.parametrize("worlds", [32, 128])
def test_root_rng_remains_lawful_stateless_and_shared(worlds):
    state = rich_private_state()
    alternative = a.a158.hidden_alternative(state)
    assert h.old.digest(h.old.serialize_state_snapshot(alternative)) != h.old.digest(h.old.serialize_state_snapshot(state))
    schedule = h.root_schedule(state, "160-test-stream", worlds)
    assert schedule == a.independent_schedule(state, "160-test-stream", worlds)
    assert schedule == h.root_schedule(alternative, "160-test-stream", worlds)
    assert len(set(schedule["world_seeds"])) == worlds
    assert h.candidate_set(state, schedule["candidate_nonce"]) == h.candidate_set(alternative, schedule["candidate_nonce"])
    if worlds == 128:
        assert schedule["world_seeds"][99].endswith("|99")
        assert schedule["world_seeds"][100].endswith("|100")


@pytest.mark.parametrize("worlds", [32, 128])
@pytest.mark.parametrize("support", [1, 2, 3, 4])
def test_smaller_support_first_maximum_and_invalid_ending_rejection(worlds, support):
    rows = [{"outcomes": [{"reason": "completed", "wdl": 0.} for _ in range(support)]} for _ in range(worlds)]
    assert h.select(rows, support, worlds) == {"choice": 0, "means": [0.]*support}
    if support > 1:
        for row in rows:
            for result in row["outcomes"][1:]:
                result["wdl"] = 1.
        assert h.select(rows, support, worlds)["choice"] == 1
    rows[-1]["outcomes"][-1]["reason"] = "max_turns"
    with pytest.raises(ValueError):
        h.select(rows, support, worlds)


@pytest.mark.parametrize("worlds", [0, 31, 64, 129, True, 128.])
def test_unplanned_world_budgets_rejected(worlds):
    with pytest.raises(ValueError):
        h.root_schedule(rich_private_state(), "test", worlds)


@pytest.mark.parametrize("worlds", [32, 128])
def test_failed_root_keeps_evidence_and_cannot_retry(tmp_path, monkeypatch, worlds):
    def bad_trajectory(*args):
        return {"reason": "repetition_cutoff", "winner": None, "turns": 3,
                "trace_sha256": "a"*64, "final_state_sha256": "b"*64}
    monkeypatch.setattr(h.old, "trajectory", bad_trajectory)
    path = tmp_path/f"failure_{worlds}.json"
    with pytest.raises(ValueError):
        h.choose_root(rich_private_state(), "test", path, "test", worlds)
    record = h.old.read_json(path)
    assert record["failure"] and record["worlds"][0]["outcomes"][0]["wdl"] is None
    original = path.read_bytes()
    with pytest.raises(FileExistsError):
        h.choose_root(rich_private_state(), "test", path, "test", worlds)
    assert path.read_bytes() == original


def test_hash_selected_replay_worlds_are_fixed_and_outcome_blind():
    for stage, count in (("probe", 8), ("main", 400)):
        for index in (0, count-1):
            for worlds in (32, 128):
                expected = sorted(sorted(range(worlds), key=lambda wi: h.old.digest(
                    ["160-audit-world-v1", stage, index, 23, wi]))[:2])
                assert a.replay_indices(stage, index, 23, worlds) == expected
                assert len(set(expected)) == 2 and all(0 <= wi < worlds for wi in expected)


@pytest.fixture(scope="module")
def tiny_games(tmp_path_factory):
    folder = tmp_path_factory.mktemp("160_whole_games")
    snapshot = h.old.serialize_state_snapshot(rich_private_state())
    h.old.save_new(folder/"test_launch.json", {"manifest": "160-test-binding"})
    games = [h.play_game("test", a.independent_spec("test", i), "160-test-binding", folder, snapshot)
             for i in (0, 1)]
    return folder, snapshot, games


def test_whole_game_outer_replay_and_all_root_arithmetic(tiny_games):
    folder, snapshot, games = tiny_games
    outputs = [a.audit_game("test", item["index"], item["sha256"], "160-test-binding", folder, snapshot)
               for item in games]
    assert outputs[0]["initial_sha256"] == outputs[1]["initial_sha256"]
    for output in outputs:
        assert output["game"]["reason"] == "completed"
        assert {root["worlds"] for root in output["roots"]} == {32, 128}
        assert sum(root["replayed_branches"] for root in output["roots"]) == 2*sum(root["support"] for root in output["roots"])
    for i in (0, 1):
        path = h.game_path("test", i, folder)
        original = path.read_bytes()
        with pytest.raises(FileExistsError):
            h.play_game("test", a.independent_spec("test", i), "160-test-binding", folder, snapshot)
        assert path.read_bytes() == original


@pytest.mark.parametrize("damage", ["world", "means", "choice", "sampled_trace", "winner", "outer_state"])
def test_semantic_tamper_cannot_be_repaired_by_rehashing(tiny_games, tmp_path, damage):
    source, snapshot, games = tiny_games
    game = h.old.read_json(h.game_path("test", 0, source))
    h.old.save_new(tmp_path/"test_launch.json", {"manifest": "160-test-binding"})
    for move in game["trace"]:
        original = h.root_path("test", 0, move["turn"], source)
        payload = h.old.read_json(original)
        target = h.root_path("test", 0, move["turn"], tmp_path)
        h.old.save_new(target, payload)
    root_path = h.root_path("test", 0, game["trace"][0]["turn"], tmp_path)
    root = h.old.read_json(root_path)
    if damage == "world":
        root["worlds"][-1]["state_sha256"] = "0"*64
    elif damage == "means":
        root["means"][0] += .125
    elif damage == "choice":
        root["choice"] = (root["choice"]+1)%len(root["candidates"]["actions"])
    elif damage == "sampled_trace":
        wi = a.replay_indices("test", 0, game["trace"][0]["turn"], len(root["worlds"]))[0]
        root["worlds"][wi]["outcomes"][0]["trace_sha256"] = "0"*64
    elif damage == "winner":
        outcome = root["worlds"][-1]["outcomes"][0]
        outcome["winner"] = 1-outcome["winner"] if outcome["winner"] is not None else 0
    else:
        game["trace"][0]["state_sha256"] = "0"*64
    h.old.atomic_write(root_path, root)
    game["trace"][0]["root_sha256"] = h.old.file_hash(root_path)
    target = h.game_path("test", 0, tmp_path)
    h.old.save_new(target, game)
    with pytest.raises(AssertionError):
        a.audit_game("test", 0, h.old.file_hash(target), "160-test-binding", tmp_path, snapshot)


def test_test_snapshot_and_test_output_cannot_enter_production():
    snapshot = h.old.serialize_state_snapshot(rich_private_state())
    with pytest.raises(ValueError):
        h.game_path("test", 0, h.OUTPUT)
    with pytest.raises(ValueError):
        h.root_path("test", 0, 0, h.OUTPUT/"test_fixtures")
    with pytest.raises(ValueError):
        h.play_game("main", a.independent_spec("main", 0), "test", h.OUTPUT, snapshot)
    with pytest.raises(AssertionError):
        a.audit_game("main", 0, "a"*64, "test", h.OUTPUT, snapshot)


def test_exact_auditor_cli_and_real_windowless_spawn(tiny_games, tmp_path):
    folder, snapshot, games = tiny_games
    fixture = {"kind": "160-spawn-test-fixture-v1", "base_dir": str(folder.resolve()),
               "initial_snapshot": snapshot, "games": games, "binding": "160-test-binding"}
    fixture_path = tmp_path/"fixture.json"
    h.old.save_new(fixture_path, fixture)
    stdout, stderr = tmp_path/"stdout.txt", tmp_path/"stderr.txt"
    command = [sys.executable, "-u", str(Path(a.__file__).resolve()), "spawn-probe", "--fixture", str(fixture_path)]
    launch = windowless.detached_launch(command, stdout, stderr, Path(__file__).resolve().parents[1])
    path = folder/"spawn_result.json"
    deadline = time.monotonic()+100
    while not path.exists():
        if time.monotonic() > deadline:
            logs = [candidate.read_text(encoding="utf-8") for candidate in
                    (stderr, folder/"spawn_parent_stderr.txt") if candidate.exists()]
            pytest.fail(f"160 spawned audit timed out at PID{launch['pid']}: {logs}")
        time.sleep(.05)
    result = h.old.read_json(path)
    assert result["passed"] and result["games"] == 2 and result["roots"] >= 4
    assert result["replayed_branches"] >= 8
    for health in result["worker_health"]:
        if sys.platform == "win32":
            assert Path(health["executable"]).name.lower() == "pythonw.exe"
            assert Path(health["parent_executable"]).name.lower() == "pythonw.exe"
            assert health["console_window"] == health["console_process_count"] == 0


def game_rows(scores):
    return [{"index": i, "pair": i//2, "seat": i % 2, "score": score,
             "margin": 2*score-1, "turns": 60} for i, score in enumerate(scores)]


def test_independent_pair_bootstrap_retains_seat_pair_as_the_sampling_unit():
    # Every seed pair splits its games. Resampling the individual games would
    # incorrectly introduce uncertainty despite all observed pairmeans=.5.
    stats = a.independent_statistics(game_rows([1., 0.]*200))
    assert stats["score_share"] == .5 and stats["ci95"] == [.5, .5]
    assert stats["classification"] == "below_material_target"
    assert stats["seats"]["0"]["score_share"] == 1. and stats["seats"]["1"]["score_share"] == 0.
    rows = game_rows([1., 0.]*200)
    rows[1]["pair"] = 1
    with pytest.raises(AssertionError):
        a.independent_statistics(rows)
    with pytest.raises(AssertionError):
        a.independent_statistics(game_rows([.5]*399))


def test_sealed_schedule_matches_independent_fresh_complete_pairs():
    initial_seeds, root_streams = set(), set()
    for stage, count in (("probe", 8), ("main", 400)):
        schedule = p.specs(stage)
        assert schedule == [a.independent_spec(stage, i) for i in range(count)]
        stage_seeds, stage_streams = set(), set()
        for left, right in zip(schedule[::2], schedule[1::2]):
            assert left["candidate_seat"] == 0 and right["candidate_seat"] == 1
            for key in ("pair", "environment_seed", "search_stream"):
                assert left[key] == right[key]
            stage_seeds.add(left["environment_seed"])
            stage_streams.add(left["search_stream"])
        assert len(stage_seeds) == len(stage_streams) == count//2
        assert not stage_seeds & initial_seeds and not stage_streams & root_streams
        initial_seeds |= stage_seeds
        root_streams |= stage_streams
        for ancestor in (h.baseline, a.a158.p):
            earlier = ancestor.schedule(stage)
            assert not stage_seeds & {s["environment_seed"] for s in earlier}
            assert not stage_streams & {s["search_stream"] for s in earlier}


@pytest.mark.parametrize("scores", [
    [1., 0.]*200,
    [.5]*400,
    [1., 0.]*180+[1., 1.]*20,
    [1., 1.]*96+[0., 0.]*104,
    [1., 1.]*95+[0., 0.]*105,
    [.5, 0., 1., 1.]*100,
])
def test_primary_paired_statistics_match_independent_integer_reconstruction(scores):
    rows = game_rows(scores)
    actual = p.statistics(rows)
    a.compare(actual, a.independent_statistics(rows))
    if scores == [1., 0.]*180+[1., 1.]*20:
        assert actual["score_share"] == .55 and actual["classification"] == "positive"
        # Seat performance and score margins are descriptive, never extra gates.
        assert actual["seats"]["1"]["score_share"] == .1
        for row in rows:
            row["margin"] = -100
        assert p.statistics(rows)["classification"] == "positive"
    elif scores == [1., 1.]*96+[0., 0.]*104:
        assert actual["ci95"][1] == .55 and actual["classification"] == "inconclusive"
    elif scores == [1., 1.]*95+[0., 0.]*105:
        assert actual["ci95"][1] < .55 and actual["classification"] == "below_material_target"
