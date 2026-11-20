"""158 contract tests use isolated fixtures, never either frozen outcome panel."""
from collections import Counter
import ast
import copy
import hashlib
from pathlib import Path
import subprocess
import sys

import numpy as np
import pytest
import torch

from splendor_ai.alphazero import learned_continuation_158 as h
from splendor_ai.alphazero import run_learned_continuation as p
from splendor_ai.engine.actions import Action, ActionType
from scripts.analysis import verify_learned_continuation_158 as a
from scripts.analysis import verify_fast_selector_transfer_157 as a157


@pytest.fixture(scope="module", autouse=True)
def models():
    h.initialize_worker()
    return h.MODELS_CACHE


def private_position(seat=0, rich=False):
    env = h.old.SplendorEnv(seed=1589800000)
    state = env.initial_state()
    for _ in range(2):
        action = next(x for x in env.legal_actions(state)
                      if x.action_type == ActionType.RESERVE_DECK and x.tier == 2)
        state = env.step(state, action)
    state.current_player = seat
    if rich:
        for player in state.players:
            player.bonuses = {color: 5 for color in player.bonuses}
            player.score = 14
    return state


def test_balanced_fresh_pair_schedule_and_outcome_blind_assignment():
    environments, streams = set(), set()
    for stage, games in (("probe", 12), ("main", 402)):
        rows = p.schedule(stage)
        assert rows == [a.independent_spec(stage, i) for i in range(games)]
        assert len(rows) == games
        rank = sorted(range(games//2), key=lambda pair: h.old.digest([
            "158-model-assignment-v1", stage, pair]))
        expected = {pair: h.MODELS[j % 3] for j, pair in enumerate(rank)}
        assert Counter(r["model_id"] for r in rows) == {model: games//3 for model in h.MODELS}
        for left, right in zip(rows[::2], rows[1::2]):
            assert left["pair"] == right["pair"] == left["index"]//2
            assert left["candidate_seat"] == 0 and right["candidate_seat"] == 1
            for key in ("environment_seed", "search_stream", "model_id"):
                assert left[key] == right[key]
            assert left["model_id"] == expected[left["pair"]]
            assert left["search_stream"] == h.old.digest(["158-root-stream-v1", stage, left["pair"]])
        stage_env = {row["environment_seed"] for row in rows}
        stage_stream = {row["search_stream"] for row in rows}
        assert len(stage_env) == len(stage_stream) == games//2
        assert not environments & stage_env and not streams & stage_stream
        for ancestor in (h.baseline, h.f.parent):
            assert not stage_env & {row["environment_seed"] for row in ancestor.schedule(stage)}
        environments |= stage_env
        streams |= stage_stream
    assert p.CONFIG["games"] == 402 and p.CONFIG["probe_games"] == 12
    assert p.CONFIG["worlds"] == 32 and p.CONFIG["workers"] == 8
    assert p.CONFIG["bootstrap_replicates"] == 50000
    with pytest.raises((AssertionError, ValueError)):
        p.schedule("adaptive_extension")


def test_rng_is_lawful_stateless_and_shared_across_root_alternatives():
    state = private_position()
    changed = a157.hidden_alternative(state)
    stream = "158-test-root-stream"
    schedule = h.baseline.root_schedule(state, stream)
    assert schedule == h.baseline.root_schedule(changed, stream)
    first = h.continuation_stream(stream, 3, 7, 1)
    assert first == h.old.digest(["158-continuation-v1", stream, 3, 7, 1])
    # Neither alternative index nor mutable RNG consumption is an input.
    proposals = [h.continuation_stream(stream, 3, step, actor)
                 for step, actor in ((7, 1), (8, 0), (9, 1))]
    reordered = [h.continuation_stream(stream, 3, step, actor)
                 for step, actor in ((9, 1), (7, 1), (8, 0))]
    assert proposals == [reordered[1], reordered[2], reordered[0]]
    variants = [h.continuation_stream(stream, 4, 7, 1),
                h.continuation_stream(stream, 3, 8, 1),
                h.continuation_stream(stream, 3, 7, 0)]
    assert len(set([first, *variants])) == 4
    # The next simulated actor cannot infer the root player's private card
    # through the proposal RNG. Its own lawful information determines the nonce.
    env = h.old.SplendorEnv()
    forced = next(move for move in env.legal_actions(state) if move.action_type == ActionType.TAKE_TOKENS)
    inner = env.step(state, forced)
    alternative = a157.hidden_alternative(inner)
    assert h.old.digest(h.old.serialize_state_snapshot(inner)) != h.old.digest(h.old.serialize_state_snapshot(alternative))
    inner_stream = h.continuation_stream(stream, 3, 1, inner.current_player)
    action, evidence = h.f.choose(inner, inner_stream, h.MODELS_CACHE[h.MODELS[0]])
    changed_action, changed_evidence = h.f.choose(alternative, inner_stream, h.MODELS_CACHE[h.MODELS[0]])
    assert action == changed_action and evidence == changed_evidence


@pytest.mark.parametrize("model_id", h.MODELS)
def test_exact_selected_controller_reloads_and_private_decisions(model_id, models):
    selected = h.selected_models()[model_id]
    assert h.old.file_hash(Path(selected["path"])) == selected["sha256"]
    model = models[model_id]
    loaded = h.f.StrongPolicyInference(Path(selected["path"])).model
    assert not model.training and not loaded.training
    assert all(torch.equal(v, loaded.state_dict()[k]) for k, v in model.state_dict().items())
    env = h.old.SplendorEnv()
    state = private_position()
    for step in range(4):
        assert a.raw_decision(state) == h.old._RAW.choose_action(env, state)
        stream = h.continuation_stream("158-test", 0, step, state.current_player)
        action, evidence = h.f.choose(state, stream, model)
        replayed, replay = a157.independent_decision(state, stream, loaded)
        assert action == replayed
        a157.compare(evidence, replay)
        assert evidence["privacy"] == h.baseline.privacy(state)
        state = env.step(state, action)


@pytest.mark.parametrize("support", [1, 2, 3, 4])
def test_controller_full_legal_attention_alias_max_and_baseline_first_tie(monkeypatch, support):
    # Legal alias A has its representative at0 and its highest score at2.
    legal = [Action(ActionType.TAKE_TOKENS, take_tokens=(color,))
             for color in ("white", "blue", "green", "red", "black")]
    keys = ["A", "B", "C", "D"]
    offered = [1, 0, 3, 4][:support]
    offered_keys = ["B", "A", "C", "D"][:support]
    candidates = {"actions": [h.old.serialize_action(legal[i]) for i in offered],
                  "keys": [repr(k) for k in offered_keys]}
    class Env:
        def legal_actions(self, state):
            return legal
    class Scorer:
        config = object()
        def __call__(self, observation, features, mask):
            assert features.shape[1] == len(legal) and bool(mask.all())
            return torch.tensor([[0., 1., 1., -1., -2.]]), torch.tensor([0.])
    monkeypatch.setattr(h.old, "SplendorEnv", Env)
    monkeypatch.setattr(h.f.parent, "root_schedule", lambda *args: {"candidate_nonce": "fixed"})
    monkeypatch.setattr(h.f.parent, "candidate_set", lambda *args: candidates)
    monkeypatch.setattr(h.f.parent, "privacy", lambda *args: {"own": 0, "opponent": 0})
    monkeypatch.setattr(h.f, "encode_position", lambda *args: (
        {"observation": torch.zeros(1), "features": torch.zeros((5, 1)),
         "groups": torch.tensor([0, 1, 0, 2, 3])}, keys, legal))
    action, evidence = h.f.choose(object(), "fixed", Scorer())
    assert action == legal[1] and evidence["choice"] == 0
    assert evidence["offered_scores"] == [1., 1., -1., -2.][:support]


def game_rows(scores):
    return [{"index": spec["index"], "pair": spec["pair"], "model_id": spec["model_id"],
             "seat": spec["candidate_seat"], "score": score, "margin": score*2-1,
             "turns": 60, "reason": "completed", "timing": {"wall": 1., "cpu": 1.}}
            for spec, score in zip(p.schedule("main"), scores)]


@pytest.mark.parametrize("scores", [[1., 0.]*201, [.5]*402, [1.]*260+[0.]*142])
def test_independent_stratified_complete_pair_statistics(scores):
    rows = game_rows(scores)
    stats = p.statistics(rows)
    a.compare(stats, a.independent_statistics(rows))
    if scores in ([1., 0.]*201, [.5]*402):
        assert stats["share"] == .5 and stats["ci95"] == [.5, .5]
    damaged = copy.deepcopy(rows)
    damaged[1]["model_id"] = next(model for model in h.MODELS if model != damaged[0]["model_id"])
    with pytest.raises(AssertionError):
        p.statistics(damaged)
    with pytest.raises(AssertionError):
        p.statistics(rows[:-1]+[rows[0]])


def test_aggregate_gates_use_exact_half_points_and_no_per_fit_veto(monkeypatch):
    monkeypatch.setattr(p.np, "quantile", lambda *args, **kwargs: np.array([.51, .9]))
    fail = p.statistics(game_rows([1.]*221+[0.]*181))
    passed = p.statistics(game_rows([1.]*221+[.5]+[0.]*180))
    assert fail["share"] < .55 <= passed["share"]
    assert not fail["statistical_gates"]["material_share"]
    assert passed["statistical_pass"]
    # A predeclared uniform mixture is judged jointly, without choosing a fit.
    rows = game_rows([1.]*402)
    for row in rows:
        if row["model_id"] == h.MODELS[0]:
            row["score"] = 0.
            row["margin"] = -1
    stats = p.statistics(rows)
    assert stats["models"][h.MODELS[0]]["share"] == 0.
    assert stats["statistical_pass"]
    monkeypatch.setattr(p.np, "quantile", lambda *args, **kwargs: np.array([.50, .9]))
    assert not p.statistics(rows)["statistical_gates"]["positive_lower_bound"]


def test_bootstrap_exact_half_boundary_with_different_stratum_means():
    rows = game_rows([.5]*402)
    scores = {h.MODELS[0]: (.5, 0.), h.MODELS[1]: (.5, .5), h.MODELS[2]: (1., .5)}
    for row in rows:
        row["score"] = scores[row["model_id"]][row["seat"]]
    stats = p.statistics(rows)
    a.compare(stats, a.independent_statistics(rows))
    assert stats["share"] == .5 and stats["ci95"] == [.5, .5]
    assert not stats["statistical_gates"]["positive_lower_bound"]


def test_mutable_progress_retries_only_windows_sharing_errors(tmp_path, monkeypatch):
    monkeypatch.setattr(h, "OUTPUT", tmp_path)
    calls, sleeps = [], []
    real_write = h.old.atomic_write
    def write(path, record):
        calls.append(Path(path))
        if len(calls) <= 2:
            error = PermissionError("test Windows share lock")
            error.winerror = 5 if len(calls) == 1 else 32
            raise error
        real_write(path, record)
    monkeypatch.setattr(h.old, "atomic_write", write)
    monkeypatch.setattr(h.time, "sleep", sleeps.append)
    path = tmp_path/"progress.json"
    h.progress_write(path, {"phase": "test"})
    assert calls == [path]*3 and sleeps == [.1, .1]
    assert h.old.read_json(path) == {"phase": "test"}
    before = len(calls)
    with pytest.raises((AssertionError, ValueError)):
        h.progress_write(tmp_path/"result.json", {"must_not_write": True})
    assert len(calls) == before and not (tmp_path/"result.json").exists()
    def denied(*args):
        error = PermissionError("non-Windows or different permission failure")
        error.winerror = 13
        raise error
    monkeypatch.setattr(h.old, "atomic_write", denied)
    with pytest.raises(PermissionError):
        h.progress_write(path, {})
    assert len(sleeps) == 2


def test_progress_retry_budget_is_finite(tmp_path, monkeypatch):
    monkeypatch.setattr(h, "OUTPUT", tmp_path)
    attempts, sleeps = [], []
    def locked(*args):
        attempts.append(1)
        error = PermissionError("persistent test lock")
        error.winerror = 32
        raise error
    monkeypatch.setattr(h.old, "atomic_write", locked)
    monkeypatch.setattr(h.time, "sleep", sleeps.append)
    with pytest.raises(PermissionError):
        h.progress_write(tmp_path/"audit_progress.json", {})
    assert len(attempts) == 51 and sleeps == [.1]*50


def test_clone_continuations_reproduce_every_unchanged155_root_outcome(tmp_path, monkeypatch):
    state = private_position(rich=True)
    clone = h.f.StrongPolicyInference(h.old.CHECKPOINT).model
    monkeypatch.setitem(h.MODELS_CACHE, h.MODELS[0], clone)
    before = h.old.digest(h.old.serialize_state_snapshot(state))
    learned_action, learned = h.choose_root(state, "158-test-clone", tmp_path/"clone.json", "test", h.MODELS[0])
    ref_action, reference = h.choose_root(state, "158-test-clone", tmp_path/"reference.json", "test", None)
    untouched_action, untouched = h.baseline.choose_root(state, "158-test-clone", tmp_path/"untouched155.json", "test")
    assert learned_action == ref_action == untouched_action
    for key in ("scores", "candidates", "schedule", "choice", "means"):
        assert learned[key] == reference[key] == untouched[key]
    assert reference["worlds"] == untouched["worlds"]
    decisions = 0
    for world, original in zip(learned["worlds"], reference["worlds"]):
        assert world["seed"] == original["seed"] and world["state_sha256"] == original["state_sha256"]
        for result, raw in zip(world["outcomes"], original["outcomes"]):
            assert {key: result[key] for key in raw} == raw
            decisions += result["learned_decisions"]
    assert decisions > 0 and h.old.digest(h.old.serialize_state_snapshot(state)) == before


def test_learned_continuation_both_actors_exact_decision_digest_and_order_independence(monkeypatch):
    state = private_position()
    env = h.old.SplendorEnv()
    legal = env.legal_actions(state)
    forced = h.old._RAW.choose_action(env, state, legal)
    stream = "158-test-complete-continuation"
    decisions = []
    real_choose = h.f.choose
    def recorded(inner, proposal_stream, scorer):
        action, evidence = real_choose(inner, proposal_stream, scorer)
        decisions.append({"actor": inner.current_player, "stream": proposal_stream, "decision": evidence})
        return action, evidence
    monkeypatch.setattr(h.f, "choose", recorded)
    result = h.trajectory(state, forced, stream, 3, h.MODELS[0])
    assert result["reason"] in h.old.VALID_NATURAL_TERMINATIONS
    assert {row["actor"] for row in decisions} == {0, 1}
    assert result["learned_decisions"] == len(decisions) == result["turns"]-1
    digest = hashlib.sha256()
    for step, row in enumerate(decisions, start=1):
        assert row["stream"] == h.continuation_stream(stream, 3, step, row["actor"])
        digest.update(h.old.digest({"step": step, **row}).encode())
    assert result["decision_sha256"] == digest.hexdigest()
    replay = a.replay_branch(state, forced, state.current_player, stream, 3, h.MODELS_CACHE[h.MODELS[0]])
    assert {key: value for key, value in replay.items() if key != "wdl"} == result
    # Running another forced alternative cannot consume the first one's RNG.
    h.trajectory(state, next(action for action in legal if action != forced), stream, 3, h.MODELS[0])
    assert h.trajectory(state, forced, stream, 3, h.MODELS[0]) == result


@pytest.mark.parametrize("ending,expected_turns", [("forced_pass_deadlock", 2),
    ("repetition_cutoff", 3), ("no_progress_cutoff", 60), ("max_turns", 0),
    ("illegal_or_missing_action", 0), ("stalled", 0), ("exception", 1)])
def test_inner_termination_preserves_invalid_trajectory(monkeypatch, ending, expected_turns):
    state = private_position()
    state.turn_index = 400 if ending == "max_turns" else 0
    action = Action(ActionType.PASS)
    class CycleEnv:
        def legal_actions(self, current):
            return [] if ending == "stalled" else [action]
        def step(self, current, selected):
            successor = current.copy_shallow()
            successor.turn_index += 1
            successor.current_player = 1-current.current_player
            return successor
    def choose(*args):
        if ending == "exception":
            raise RuntimeError("injected chooser exception")
        return action, {"fixture": True}
    monkeypatch.setattr(h.old, "SplendorEnv", CycleEnv)
    monkeypatch.setattr(h.f, "choose", choose)
    monkeypatch.setattr(h.old, "is_forced_pass_turn", lambda legal: ending == "forced_pass_deadlock")
    monkeypatch.setattr(h.old, "is_progress_transition", lambda *args: False)
    monkeypatch.setattr(h.old, "state_signature", lambda current: "same" if ending == "repetition_cutoff" else current.turn_index)
    result = h.trajectory(state, None if ending == "illegal_or_missing_action" else action,
                          "test", 0, h.MODELS[0])
    assert result["reason"] == ending and result["turns"] == expected_turns
    if ending == "forced_pass_deadlock":
        assert result["winner"] is None and "failure" not in result and "trace" not in result
    else:
        assert len(result["trace"]) == expected_turns
        assert h.old.digest(result["final_state"]) == result["final_state_sha256"]
    if ending == "exception":
        assert "injected chooser exception" in result["failure"]


def test_invalid_root_is_preserved_and_never_retried(tmp_path, monkeypatch):
    state = private_position(rich=True)
    def invalid(*args):
        return {"reason": "repetition_cutoff", "winner": None, "turns": 3,
                "trace_sha256": "retained", "final_state_sha256": "retained",
                "decision_sha256": "retained", "learned_decisions": 2,
                "trace": [{"retained": True}], "final_state": {"retained": True}}
    monkeypatch.setattr(h, "trajectory", invalid)
    path = tmp_path/"failed.json"
    with pytest.raises(ValueError, match="Invalid learned rollout"):
        h.choose_root(state, "test", path, "binding", h.MODELS[0])
    record = h.old.read_json(path)
    assert record["failure"] and record["worlds"][0]["outcomes"][0]["wdl"] is None
    assert record["worlds"][0]["outcomes"][0]["trace"] == [{"retained": True}]
    before = h.old.file_hash(path)
    with pytest.raises(FileExistsError):
        h.choose_root(state, "test", path, "binding", h.MODELS[0])
    assert h.old.file_hash(path) == before


def test_failed_outer_game_is_preserved_without_root_search(tmp_path, monkeypatch):
    monkeypatch.setattr(h, "OUTPUT", tmp_path)
    env = h.old.SplendorEnv(seed=1589800000)
    class CappedEnv:
        def __init__(self, seed=None):
            pass
        def initial_state(self):
            state = env.initial_state()
            state.turn_index = 400
            return state
    monkeypatch.setattr(h.old, "SplendorEnv", CappedEnv)
    monkeypatch.setattr(h, "choose_root", lambda *args: pytest.fail("capped state must not search"))
    with pytest.raises(ValueError, match="max_turns"):
        h.play_game("probe", p.schedule("probe")[0], "test")
    path = h.game_path("probe", 0)
    digest = h.old.file_hash(path)
    record = h.old.read_json(path)
    assert record["failure"] and record["reason"] == "max_turns" and not record["trace"]
    with pytest.raises(FileExistsError):
        h.play_game("probe", p.schedule("probe")[0], "test")
    assert h.old.file_hash(path) == digest


def test_complete_games_and_exact_spawned_auditor_then_semantic_tamper(tmp_path, monkeypatch):
    monkeypatch.setattr(h, "OUTPUT", tmp_path)
    state = private_position(rich=True)
    snapshot = h.old.serialize_state_snapshot(state)
    original_env = h.old.SplendorEnv
    class TinyEnv:
        def __init__(self, seed=None):
            self.env = original_env(seed=1589800000)
        def initial_state(self):
            return h.old.deserialize_state_snapshot(snapshot)
        def legal_actions(self, current):
            return self.env.legal_actions(current)
        def step(self, current, action):
            return self.env.step(current, action)
    h.old.save_new(tmp_path/"test_launch.json", {"manifest": "158-test-binding"})
    with monkeypatch.context() as patch:
        patch.setattr(h.old, "SplendorEnv", TinyEnv)
        games = [h.play_game("test", a.test_spec(index), "158-test-binding") for index in (0, 1)]
    fixture = {"kind": "158-spawn-test-fixture-v1", "games": games,
               "binding": "158-test-binding", "base_dir": str(tmp_path.resolve()),
               "initial_snapshot": snapshot}
    fixture_path = tmp_path/"spawn_fixture.json"
    h.old.save_new(fixture_path, fixture)
    # Invoke the exact production script as __main__, which then dispatches
    # audit_game itself in fresh workers. Imported-module-only checks missed
    # the Windows spawning failure in157.
    completed = subprocess.run([sys.executable, str(p.VERIFIER), "spawn-probe", "--fixture", str(fixture_path)],
                               capture_output=True, text=True, check=True, timeout=120)
    spawned = ast.literal_eval(completed.stdout.strip().splitlines()[-1])
    assert spawned["passed"] and spawned["games"] == 2
    assert spawned["roots"] >= 4 and spawned["continuations"] >= 4*32
    context = {"base_dir": str(tmp_path.resolve()), "initial_snapshot": snapshot, "spec": a.test_spec(0)}
    checked = a.audit_game("test", 0, games[0]["sha256"], "158-test-binding", context)
    assert checked["game"]["reason"] == "completed"
    assert {root["planner"] for root in checked["roots"]} == {"candidate", "reference"}
    for spec in (a.test_spec(0), a.test_spec(1)):
        path = h.game_path("test", spec["index"])
        before = h.old.file_hash(path)
        with pytest.raises(FileExistsError):
            h.play_game("test", spec, "158-test-binding")
        assert h.old.file_hash(path) == before
    # Rehashing corrupted evidence cannot make an invented continuation valid.
    game_path = h.game_path("test", 0)
    game = h.old.read_json(game_path)
    root_path = h.root_path("test", 0, game["trace"][0]["turn"])
    root = h.old.read_json(root_path)
    root["worlds"][0]["outcomes"][0]["decision_sha256"] = "semantic-tamper"
    h.old.atomic_write(root_path, root)
    game["trace"][0]["root_sha256"] = h.old.file_hash(root_path)
    h.old.atomic_write(game_path, game)
    with pytest.raises(AssertionError):
        a.audit_game("test", 0, h.old.file_hash(game_path), "158-test-binding", context)
    with pytest.raises(AssertionError):
        a.audit_game("main", 0, games[0]["sha256"], "158-test-binding", context)


def test_identity_tracks_explicit_inventory_and_rejects_changed_test_evidence(tmp_path, monkeypatch):
    monkeypatch.setattr(p, "OUTPUT", tmp_path)
    monkeypatch.setattr(p, "parent_inputs", lambda: {"frozen_parent": True})
    dependency = tmp_path/"required.py"
    dependency.write_text("required = 1\n", encoding="utf-8")
    monkeypatch.setattr(p, "INVENTORY", (dependency,))
    before = p.identity()
    (tmp_path/"unrelated_dirty_work.py").write_text("preserve = True\n", encoding="utf-8")
    assert p.identity() == before
    dependency.write_text("required = 2\n", encoding="utf-8")
    assert p.identity() != before
    h.old.save_new(tmp_path/"preparation"/"tests.json", {"full_repository": True,
                   "exit_code": 0, "identity": before})
    with pytest.raises(AssertionError):
        p.verify_test_evidence()


@pytest.mark.parametrize("suffix", ["", "test_fixtures"])
def test_test_stage_cannot_target_production_evidence(monkeypatch, suffix):
    monkeypatch.setattr(h, "OUTPUT", h._PRODUCTION_OUTPUT/suffix)
    with pytest.raises(ValueError, match="production evidence"):
        h.game_path("test", 0)
    with pytest.raises(ValueError, match="production evidence"):
        h.root_path("test", 0, 0)


@pytest.mark.parametrize("passed", [False, True])
def test_entire_main_audit_required_for_either_statistical_outcome(tmp_path, monkeypatch, passed):
    monkeypatch.setattr(p, "OUTPUT", tmp_path)
    monkeypatch.setattr(p, "validate_freeze", lambda: "binding")
    for name, payload in (("probe_audit.json", {"passed": True}), ("timing_probe.json", {}),
                          ("probe_launch.json", {"time": h.old.now()})):
        h.old.save_new(tmp_path/name, payload)
    h.old.save_new(tmp_path/"preflight.json", {"passed": True, "manifest": "binding",
                   "audit_sha256": h.old.file_hash(tmp_path/"probe_audit.json"),
                   "timing_sha256": h.old.file_hash(tmp_path/"timing_probe.json")})
    events = []
    monkeypatch.setattr(p, "launch_stage", lambda *args: events.append("launch"))
    monkeypatch.setattr(p, "collect", lambda *args: events.append("collect"))
    monkeypatch.setattr(p, "observations", lambda *args: {"games": []})
    monkeypatch.setattr(p, "statistics", lambda *args: {"statistical_pass": passed, "share": .6 if passed else .5,
                        "ci95": [.4, .7], "wdl": {"wins": 201, "draws": 0, "losses": 201}, "mean_margin": 0.})
    monkeypatch.setattr(p, "operational", lambda *args: {"continuations": 128})
    monkeypatch.setattr(p, "progress", lambda *args, **kwargs: None)
    def audit(stage):
        assert (tmp_path/"result.json").exists() and not (tmp_path/"final_status.json").exists()
        events.append("audit")
        report = {"passed": True, "games": 402, "roots": 1, "continuations": 128}
        h.old.save_new(tmp_path/"main_audit.json", report)
        return report
    monkeypatch.setattr(p, "audit_stage", audit)
    p.run_main("binding")
    assert events == ["launch", "collect", "audit"]
    final = h.old.read_json(tmp_path/"final_status.json")
    assert final["complete"] and final["advance_gate_passed"] == passed
    assert not final["training"] and not final["promotion"] and not final["automatic_successor"]
