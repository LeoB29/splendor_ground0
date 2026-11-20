"""Runner integrity and paired arithmetic using synthetic records only."""
from contextlib import nullcontext
from copy import deepcopy
import hashlib
import json

import pytest

from tools.external_strength_screen_20261018 import runner, rules


def _write(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, sort_keys=True), encoding="utf-8")


def _clock(status="ok"):
    return {"request_id": "request-00000001", "clock": "perf_counter_ns", "status": status,
            "accepted": status == "ok", "deadline_seconds": 1.0, "elapsed_seconds": 1.1,
            "receipt_elapsed_seconds": .2 if status == "ok" else None,
            "received_before_deadline": status == "ok", "result": None, "detail": "Synthetic fixture"}


def test_clock_accepts_timely_receipt_even_when_validation_finishes_later():
    runner.check_clock(_clock(), "request-00000001")
    runner.check_clock(_clock("timeout"), "request-00000001")


@pytest.mark.parametrize("changes", [
    {"accepted": False}, {"accepted": 1}, {"received_before_deadline": False},
    {"deadline_seconds": 0}, {"deadline_seconds": 1.01}, {"deadline_seconds": True},
    {"elapsed_seconds": -.1}, {"receipt_elapsed_seconds": float("nan")},
    {"receipt_elapsed_seconds": 1.01, "received_before_deadline": False},
    {"elapsed_seconds": .1, "receipt_elapsed_seconds": .2},
    {"request_id": "request-00000002"}, {"clock": "monotonic_ns"},
])
def test_clock_rejects_inconsistent_flags_bounds_and_identity(changes):
    receipt = {**_clock(), **changes}
    with pytest.raises(ValueError):
        runner.check_clock(receipt, "request-00000001")


def test_schedule_has_30_distinct_worlds_and_exact_seat_reversals():
    schedule = runner.make_schedule()
    assert [row["game"] for row in schedule] == list(range(60))
    assert len({row["seed"] for row in schedule}) == 30
    assert sum(row["ppo_seat"] == 0 for row in schedule) == 30
    for pair in range(30):
        first, second = schedule[2 * pair:2 * pair + 2]
        assert first["pair"] == second["pair"] == pair
        assert first["seed"] == second["seed"]
        assert first["ppo_seat"] == pair % 2 and second["ppo_seat"] == 1 - first["ppo_seat"]
        expected = str(int(hashlib.sha256(f"{runner.NAMESPACE}:pair:{pair}".encode()).hexdigest(), 16))
        assert first["seed"] == expected


def _summary_games(outcomes):
    games = []
    for spec in runner.make_schedule():
        score = outcomes[spec["pair"]][spec["game"] % 2]
        initial = {"synthetic_world": spec["pair"]}
        winner = None if score == .5 else spec["ppo_seat"] if score == 1 else 1 - spec["ppo_seat"]
        games.append({**spec, "complete": True, "initial": initial,
                      "initial_sha256": runner.c.digest(initial), "winner": winner,
                      "ppo_score": score, "reason": "natural"})
    return games


def test_bootstrap_resamples_whole_pairs_not_individual_games():
    games = _summary_games([(1., 0.)] * 30)
    result = runner.summarize(games)
    assert result["ppo_mean_score"] == .5
    assert result["paired_bootstrap_95_percent_interval"] == [.5, .5]
    assert (result["ppo_wins"], result["draws"], result["ppo_losses"]) == (30, 0, 30)
    assert result["by_ppo_seat"] == {"0": .5, "1": .5}
    assert result["bootstrap_replicates"] == 50000


def test_paired_bootstrap_nonconstant_arithmetic():
    # Twenty winning pairs and ten losing pairs: bootstrap wins are Binomial(30, 2/3).
    # Its stable 2.5%/97.5% count quantiles are 15 and 25, rather than 60-game quantiles.
    games = _summary_games([(1., 1.)] * 20 + [(0., 0.)] * 10)
    result = runner.summarize(games)
    assert result["ppo_mean_score"] == pytest.approx(2 / 3)
    assert result["paired_bootstrap_95_percent_interval"] == [15 / 30, 25 / 30]
    assert (result["ppo_wins"], result["draws"], result["ppo_losses"]) == (40, 0, 20)


@pytest.mark.parametrize("corruption", ["score", "hash", "seed", "duplicate", "pair_world", "winner_bool", "incomplete"])
def test_summary_rejects_tampered_score_hash_and_pair_identity(corruption):
    games = _summary_games([(1., 0.)] * 30)
    if corruption == "score":
        games[0]["ppo_score"] = 0.
    elif corruption == "hash":
        games[0]["initial_sha256"] = "0" * 64
    elif corruption == "seed":
        games[0]["seed"] = "123"
    elif corruption == "duplicate":
        games[1] = deepcopy(games[0])
    elif corruption == "pair_world":
        games[0]["initial"] = {"synthetic_world": "another"}
        games[0]["initial_sha256"] = runner.c.digest(games[0]["initial"])
    elif corruption == "winner_bool":
        games[0]["winner"] = False
    else:
        games[0]["complete"] = False
    with pytest.raises(ValueError):
        runner.summarize(games)


@pytest.fixture
def single_forfeit(tmp_path, monkeypatch):
    """Only a seeded initial state and synthetic timeout; no action is played."""
    monkeypatch.setattr(runner.c, "ROOT", tmp_path)
    base = tmp_path / "synthetic-screen"
    spec = runner.make_schedule()[0]
    state = rules.CorrectedEnv(seed=int(spec["seed"])).initial_state()
    initial = rules.serialize_snapshot(state)
    clock = _clock("timeout")
    row = {"ply": 0, "actor": 0, "player": "ppo", "before_sha256": runner.c.digest(initial),
           "observer": rules.observation_payload(state), "clock": deepcopy(clock)}
    folder = base / "games" / "000"
    row_path = folder / "moves" / "0000.json"
    worker_path = base / "workers" / "game-000" / "request-00000001.json"
    _write(row_path, row)
    _write(worker_path, clock)
    game = {**spec, "complete": True, "ruleset_id": rules.RULESET_ID,
            "catalogue_sha256": rules.CATALOGUE_SHA256, "initial": initial,
            "initial_sha256": runner.c.digest(initial), "final": deepcopy(initial),
            "moves": [{"file": row_path.relative_to(tmp_path).as_posix(), "sha256": runner.c.file_hash(row_path)}],
            "reason": "player_forfeit_timeout", "winner": 1, "failing_player": "ppo", "ppo_score": 0.}
    _write(folder / "initial.json", {**spec, "initial": initial})
    _write(folder / "result.json", game)
    return {"base": base, "spec": spec, "game": game, "row": row, "row_path": row_path,
            "folder": folder, "worker_path": worker_path, "root": tmp_path}


def _save_forfeit(fixture, *, bind_row=True):
    _write(fixture["row_path"], fixture["row"])
    if bind_row:
        fixture["game"]["moves"][0]["sha256"] = runner.c.file_hash(fixture["row_path"])
    _write(fixture["folder"] / "result.json", fixture["game"])


def test_single_forfeit_replay_binds_worker_receipt_without_invented_transition(single_forfeit):
    f = single_forfeit
    game, count, latency = runner.audit_game(f["base"], f["spec"])
    assert game == f["game"] and count == 0
    assert latency == {"ppo": [], "raw035": []}
    assert game["initial"] == game["final"]


@pytest.mark.parametrize("corruption", ["worker_receipt", "move_hash", "winner", "failing_player", "actor",
    "player", "ply", "chain", "request_id", "resource_as_forfeit", "support_as_forfeit", "reason", "move_after_forfeit"])
def test_forfeit_audit_rejects_receipt_attribution_and_progression_corruption(single_forfeit, corruption):
    f = single_forfeit
    if corruption == "worker_receipt":
        changed = deepcopy(f["row"]["clock"])
        changed["detail"] = "Different preserved receipt"
        _write(f["worker_path"], changed)
    elif corruption == "move_hash":
        f["game"]["moves"][0]["sha256"] = "0" * 64
    elif corruption == "winner":
        f["game"]["winner"], f["game"]["ppo_score"] = 0, 1.
    elif corruption == "failing_player":
        f["game"]["failing_player"] = "raw035"
    elif corruption in ("actor", "ply"):
        f["row"][corruption] = 1
    elif corruption == "player":
        f["row"]["player"] = "raw035"
    elif corruption == "chain":
        f["row"]["before_sha256"] = "0" * 64
    elif corruption == "request_id":
        f["row"]["clock"]["request_id"] = "request-00000002"
        _write(f["worker_path"], f["row"]["clock"])
    elif corruption == "resource_as_forfeit":
        f["row"]["clock"]["status"] = "rss_limit"
        f["game"]["reason"] = "player_forfeit_rss_limit"
        _write(f["worker_path"], f["row"]["clock"])
    elif corruption == "support_as_forfeit":
        f["row"]["clock"]["detail"] = "Empty adapter support"
        _write(f["worker_path"], f["row"]["clock"])
    elif corruption == "reason":
        f["game"]["reason"] = "natural"
    else:
        extra = deepcopy(f["row"])
        extra["ply"] = 1
        path = f["folder"] / "moves" / "0001.json"
        _write(path, extra)
        f["game"]["moves"].append({"file": path.relative_to(f["root"]).as_posix(), "sha256": runner.c.file_hash(path)})
    _save_forfeit(f, bind_row=corruption != "move_hash")
    with pytest.raises(ValueError):
        runner.audit_game(f["base"], f["spec"])


def test_final_source_verification_overrides_completed_workflow(tmp_path, monkeypatch):
    base = tmp_path / "synthetic-finalization"
    _write(base / "freeze.json", {"inputs": [], "topology": {"selected_logical_processor": 0}})
    _write(base / "schedule.json", [])
    after_workflow = False

    def verify(_items):
        if after_workflow:
            raise ValueError("Synthetic final source mutation")

    def boundary(*_args, **_kwargs):
        nonlocal after_workflow
        after_workflow = True

    monkeypatch.setattr(runner.c, "verify", verify)
    monkeypatch.setattr(runner.c, "execution_lease", nullcontext)
    monkeypatch.setattr(runner.c, "boundary", boundary)
    monkeypatch.setattr(runner, "qualify", lambda *_: {"passed": True, "synthetic": True})
    monkeypatch.setattr(runner, "summarize", lambda games: {"synthetic_games": len(games)})
    monkeypatch.setattr(runner, "audit_all_games", lambda *_: None)
    monkeypatch.setattr(runner, "play_game", lambda *_: pytest.fail("No gameplay permitted in this test"))
    assert runner.run(base) == 1
    completion = runner.c.read(base / "completion.json")
    assert completion["status"] == "failed" and completion["frozen_inputs_unchanged"] is False
    assert "Synthetic final source mutation" in completion["failure"]
    assert (base / "final_source_failure.json").exists()
