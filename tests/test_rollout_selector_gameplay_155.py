import importlib.util
from pathlib import Path
import random
import sys

import numpy as np
import pytest

from splendor_ai.alphazero import run_rollout_selector_gameplay as p
from splendor_ai.engine.actions import Action, ActionType
from splendor_ai.engine.state import RESERVE_SOURCE_DECK


@pytest.fixture(scope="module")
def audit():
    spec = importlib.util.spec_from_file_location("audit155_test", "scripts/analysis/verify_rollout_selector_gameplay_155.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


@pytest.fixture(scope="module")
def raw():
    p.old.initialize_worker()
    return p.old._RAW


def private_position(seat=0, rich=False):
    env = p.old.SplendorEnv(seed=155)
    state = env.initial_state()
    for _ in range(2):
        action = next(a for a in env.legal_actions(state) if a.action_type == ActionType.RESERVE_DECK and a.tier == 2)
        state = env.step(state, action)
    state.current_player = seat
    if rich:
        for player in state.players:
            player.bonuses = {c: 5 for c in player.bonuses}
            player.score = 14
    return state


def outcome(value=0., reason="completed"):
    return {"reason": reason, "wdl": value}


@pytest.mark.parametrize("support", [1, 2, 3, 4])
def test_shared_world_selection_retains_small_support_and_exact_ties(support):
    worlds = [{"outcomes": [outcome() for _ in range(support)]} for _ in range(32)]
    assert p.select(worlds, support) == {"choice": 0, "means": [0.]*support}
    worlds[0]["outcomes"][-1] = outcome(1.)
    assert p.select(worlds, support)["choice"] == support-1


@pytest.mark.parametrize("reason", ["repetition_cutoff", "no_progress_cutoff", "max_turns", "stalled", "illegal_or_missing_action"])
def test_invalid_inner_ending_is_never_scored(reason):
    worlds = [{"outcomes": [outcome()]} for _ in range(32)]
    worlds[7]["outcomes"][0] = outcome(None, reason)
    with pytest.raises(ValueError):
        p.select(worlds, 1)


def test_schedule_and_main_budgets_are_frozen_and_disjoint(audit):
    seen_env, seen_streams = set(), set()
    for stage, count in (("probe", 8), ("main", 400)):
        rows = p.schedule(stage)
        assert rows == [audit.independent_spec(stage, i) for i in range(count)]
        envs = {r["environment_seed"] for r in rows}
        streams = {r["search_stream"] for r in rows}
        assert len(envs) == count//2 and len(streams) == count
        assert not envs & seen_env and not streams & seen_streams
        seen_env |= envs
        seen_streams |= streams
        for a, b in zip(rows[::2], rows[1::2]):
            assert a["environment_seed"] == b["environment_seed"] and (a["candidate_seat"], b["candidate_seat"]) == (0, 1)
    assert p.CONFIG["worlds"] == 32 and p.CONFIG["workers"] == 8
    assert p.CONFIG["main_continuation_ceiling"] == 400*200*4*32
    with pytest.raises(ValueError):
        p.schedule("tuned")


@pytest.mark.parametrize("seat", [0, 1])
def test_full_private_root_hidden_invariance_and_independent_replay(tmp_path, audit, raw, seat):
    state = private_position(seat, rich=True)
    before = p.old.serialize_state_snapshot(state)
    changed = audit.hidden_alternative(state)
    assert p.old.serialize_state_snapshot(changed) != before
    stream = p.schedule("probe")[seat]["search_stream"]
    assert p.root_schedule(state, stream) == audit.independent_schedule(state, stream) == p.root_schedule(changed, stream)
    # Full production 32-world budget, including own/opponent blind reserves.
    action, record = p.choose_root(state, stream, tmp_path/"root.json", "unit-test")
    second, alternate = p.choose_root(changed, stream, tmp_path/"alternate.json", "unit-test")
    assert action == second and record["privacy"] == {"own": 1, "opponent": 1}
    for key in ("scores", "candidates", "schedule", "worlds", "choice", "means"):
        assert record[key] == alternate[key]
    assert before == p.old.serialize_state_snapshot(state)
    for seed, row in zip(record["schedule"]["world_seeds"], record["worlds"]):
        world = audit.a154.independent_world(state, seed)
        assert row["state_sha256"] == p.old.digest(p.old.serialize_state_snapshot(world))
        assert world.players[seat].reserved_cards == state.players[seat].reserved_cards
        for payload, result in zip(record["candidates"]["actions"], row["outcomes"]):
            assert audit.a154.a149.replay_branch(world, p.old.decode_action(payload), seat) == result
    assert any(a.action_type == ActionType.BUY_RESERVED for a in p.old.SplendorEnv().legal_actions(state))
    assert p.choose_root(state, stream, tmp_path/"root.json", "unit-test")[0] == action
    with pytest.raises(AssertionError):
        p.choose_root(state, stream, tmp_path/"root.json", "changed-binding")


def test_root_failure_evidence_preserved_no_fallback_or_retry(tmp_path, monkeypatch, raw):
    state = private_position(rich=True)
    def invalid(*args):
        return {"reason": "repetition_cutoff", "winner": None, "turns": 8,
                "trace_sha256": "invalid", "final_state_sha256": "invalid"}
    monkeypatch.setattr(p.old, "trajectory", invalid)
    path = tmp_path/"failed.json"
    with pytest.raises(ValueError):
        p.choose_root(state, "independent", path, "test")
    digest = p.old.file_hash(path)
    saved = p.old.read_json(path)
    assert saved["worlds"][0]["outcomes"][0]["wdl"] is None and saved["failure"]
    with pytest.raises(AssertionError):
        p.choose_root(state, "independent", path, "test")
    assert p.old.file_hash(path) == digest


def test_payment_returns_nobles_equal_turn_terminal_and_dry_decks(audit, raw):
    env = p.old.SplendorEnv()
    state = private_position(rich=True)
    state.current_player = state.start_player = 0
    action = next(a for a in env.legal_actions(state) if a.action_type == ActionType.BUY_RESERVED and a.noble_id)
    successor = env.step(state, action)
    assert not successor.terminal and successor.pending_round_end and successor.current_player == 1
    next_action = next(a for a in env.legal_actions(successor) if a.action_type == ActionType.BUY_VISIBLE)
    assert env.step(successor, next_action).terminal
    state.start_player = 1
    assert env.step(state, action).terminal
    for candidate in env.legal_actions(state):
        replay = audit.a154.a149.replay_branch(state, candidate, state.current_player)
        expected = p.old.trajectory(state, candidate)
        expected["wdl"] = p.old.outcome_value(expected["winner"], state.current_player) if expected["reason"] in p.old.VALID_NATURAL_TERMINATIONS else None
        assert replay == expected
    # Payment and token-return variants remain legal candidates/groups.
    payment = private_position()
    player = payment.players[0]
    player.tokens = {"white": 2, "blue": 2, "green": 2, "red": 1, "black": 1, "gold": 2}
    for color, amount in player.tokens.items():
        payment.bank_tokens[color] = (5 if color == "gold" else 4)-amount-payment.players[1].tokens[color]
    player.bonuses = {c: 1 for c in player.bonuses}
    legal = env.legal_actions(payment)
    assert any(a.return_tokens for a in legal)
    assert any(a.spend_tokens and "gold" in a.spend_tokens for a in legal)
    nonce = p.root_schedule(payment, "fixture")["candidate_nonce"]
    assert p.candidate_set(payment, nonce) == audit.a154.independent_candidates(payment, "155-live-v1", nonce)
    # Catalog-complete exhausted tier: move cards to public purchased inventory.
    dry = private_position(rich=True)
    cards = dry.hidden_tier_decks[1]
    dry.players[1].purchased_cards.extend(cards)
    dry.hidden_tier_decks[1] = []
    dry.deck_counts[1] = 0
    for seed in ("155-test-dry-0", "155-test-dry-1"):
        world = p.old.determinize_for_observer(dry, dry.current_player, random.Random(seed))
        assert p.old.serialize_state_snapshot(world) == p.old.serialize_state_snapshot(audit.a154.independent_world(dry, seed))
        assert not any(a.action_type == ActionType.RESERVE_DECK and a.tier == 1 for a in env.legal_actions(world))


def rows_for(scores):
    return [{"index": i, "pair": i//2, "seat": i%2, "score": score, "margin": score*2-1,
             "turns": 64, "reason": "completed", "timing": {"wall": 1., "cpu": 1.}}
            for i, score in enumerate(scores)]


@pytest.mark.parametrize("scores", [[1.]*240+[0.]*160, [1., 0.]*200, [.5]*400])
def test_whole_pair_bootstrap_reproduces_independent_arithmetic(audit, scores):
    rows = rows_for(scores)
    result = p.statistics(rows)
    audit.compare(result, audit.independent_statistics(rows))
    if scores == [1., 0.]*200:
        assert result["ci95"] == [.5, .5]  # Every complete pair is exactly balanced.
    assert result["statistical_pass"] == (result["share"] >= .55 and result["ci95"][0] > .5)
    rows[1]["pair"] = 900
    with pytest.raises(AssertionError):
        p.statistics(rows)


def test_each_strength_gate_is_required(monkeypatch):
    monkeypatch.setattr(p.np, "quantile", lambda *args: np.array([.50, .8]))
    assert not p.statistics(rows_for([1.]*240+[0.]*160))["statistical_pass"]
    monkeypatch.setattr(p.np, "quantile", lambda *args: np.array([.51, .8]))
    assert not p.statistics(rows_for([1.]*216+[0.]*184))["statistical_pass"]


@pytest.mark.parametrize("seat", [0, 1])
def test_complete_game_collection_replay_statistics_inputs_and_tamper_detection(tmp_path, monkeypatch, raw, audit, seat):
    monkeypatch.setattr(p, "OUTPUT", tmp_path)
    original_env = p.old.SplendorEnv
    class TinyEnv:
        def __init__(self, seed=None):
            self.env = original_env(seed=155)
        def initial_state(self):
            state = self.env.initial_state()
            for player in state.players:
                player.bonuses = {c: 5 for c in player.bonuses}
                player.score = 14
            return state
        def legal_actions(self, state):
            return self.env.legal_actions(state)
        def step(self, state, action):
            return self.env.step(state, action)
    monkeypatch.setattr(p.old, "SplendorEnv", TinyEnv)
    p.old.save_new(tmp_path/"probe_launch.json", {"manifest": "test"})
    spec = p.schedule("probe")[seat]
    info = p.play_game("probe", spec, "test")
    audited = audit.audit_game("probe", seat, info["sha256"], "test")
    assert audited["game"]["index"] == seat and audited["roots"]
    assert p.play_game("probe", spec, "test") == info
    monkeypatch.setattr(p, "schedule", lambda stage: [spec])
    p.old.save_new(tmp_path/"probe_games_lock.json", {"files": {str(seat): info["sha256"]}})
    data = p.observations("probe")
    audit.compare(data, {"games": [audited["game"]], "roots": audited["roots"], "raw_timings": audited["raw_timings"]})
    audit.compare(p.operational(data), audit.independent_operational(data))
    game = p.old.read_json(p.game_path("probe", seat))
    game["trace"][0]["actor"] = 1-game["trace"][0]["actor"]
    p.old.atomic_write(p.game_path("probe", seat), game)  # Isolated tamper fixture only.
    with pytest.raises(AssertionError):
        audit.audit_game("probe", seat, p.old.file_hash(p.game_path("probe", seat)), "test")


def test_game_reuse_and_outer_invalid_endings_preserve_evidence(tmp_path, monkeypatch, raw):
    monkeypatch.setattr(p, "OUTPUT", tmp_path)
    original_env = p.old.SplendorEnv
    class CappedEnv:
        def __init__(self, seed=None):
            self.env = original_env(seed=155)
        def initial_state(self):
            state = self.env.initial_state()
            state.turn_index = 400
            return state
        def legal_actions(self, state):
            return self.env.legal_actions(state)
    monkeypatch.setattr(p.old, "SplendorEnv", CappedEnv)
    with pytest.raises(ValueError, match="max_turns"):
        p.play_game("probe", p.schedule("probe")[0], "test")
    path = p.game_path("probe", 0)
    sha = p.old.file_hash(path)
    assert p.old.read_json(path)["failure"]
    with pytest.raises(AssertionError):
        p.play_game("probe", p.schedule("probe")[0], "test")
    assert p.old.file_hash(path) == sha


@pytest.mark.parametrize("ending", ["repetition_cutoff", "no_progress_cutoff", "forced_pass_deadlock", "stalled", "illegal_or_missing_action"])
def test_outer_termination_contract(tmp_path, monkeypatch, ending):
    monkeypatch.setattr(p, "OUTPUT", tmp_path)
    initial = p.old.SplendorEnv(seed=155).initial_state()
    legal_action = Action(action_type=ActionType.PASS)
    class CycleEnv:
        def __init__(self, seed=None):
            pass
        def initial_state(self):
            return initial.copy_shallow()
        def legal_actions(self, state):
            return [] if ending == "stalled" else [legal_action]
        def step(self, state, action):
            successor = state.copy_shallow()
            successor.turn_index += 1
            successor.current_player = 1-state.current_player
            return successor
    class PassPolicy:
        def choose_action(self, *args):
            return legal_action
    monkeypatch.setattr(p.old, "SplendorEnv", CycleEnv)
    monkeypatch.setattr(p.old, "_RAW", PassPolicy())
    monkeypatch.setattr(p.old, "is_forced_pass_turn", lambda legal: ending == "forced_pass_deadlock")
    monkeypatch.setattr(p.old, "is_progress_transition", lambda *args: False)
    monkeypatch.setattr(p.old, "state_signature", lambda state: "same" if ending == "repetition_cutoff" else state.turn_index)
    def choice(state, stream, path, binding):
        p.old.save_new(path, {"fixture": True})
        return (None if ending == "illegal_or_missing_action" else legal_action), {}
    monkeypatch.setattr(p, "choose_root", choice)
    if ending == "forced_pass_deadlock":
        p.play_game("probe", p.schedule("probe")[0], "test")
    else:
        with pytest.raises(ValueError, match=ending):
            p.play_game("probe", p.schedule("probe")[0], "test")
    record = p.old.read_json(p.game_path("probe", 0))
    assert record["reason"] == ending
    if ending == "forced_pass_deadlock":
        assert record["turns"] == 2 and record["winner"] is None and "failure" not in record
    elif ending == "no_progress_cutoff":
        assert record["turns"] == 60
    elif ending == "repetition_cutoff":
        assert record["turns"] == 3


@pytest.mark.parametrize("passed", [True, False])
def test_main_always_finishes_audit_before_closure(tmp_path, monkeypatch, passed):
    monkeypatch.setattr(p, "OUTPUT", tmp_path)
    monkeypatch.setattr(p, "check_identity", lambda: "test")
    monkeypatch.setattr(p, "verify_test_evidence", lambda: "tests")
    for name, value in (("probe_audit.json", {"passed": True}), ("timing_probe.json", {}),
                        ("freeze.json", {"tests_sha256": "tests"}), ("probe_launch.json", {"time": p.old.now()})):
        p.old.save_new(tmp_path/name, value)
    p.old.save_new(tmp_path/"preflight.json", {"passed": True, "manifest": "test",
                   "audit_sha256": p.old.file_hash(tmp_path/"probe_audit.json"),
                   "timing_sha256": p.old.file_hash(tmp_path/"timing_probe.json")})
    events = []
    monkeypatch.setattr(p, "launch_stage", lambda *args: events.append("launch"))
    monkeypatch.setattr(p, "collect", lambda *args: events.append("collect"))
    monkeypatch.setattr(p, "observations", lambda *args: {"games": []})
    monkeypatch.setattr(p, "statistics", lambda *args: {"statistical_pass": passed, "share": .6 if passed else .5,
                        "ci95": [.4, .7], "wdl": {"wins": 200, "draws": 0, "losses": 200}})
    monkeypatch.setattr(p, "operational", lambda *args: {"continuations": 128})
    def audit_stage(stage):
        assert (tmp_path/"result.json").exists() and not (tmp_path/"final_status.json").exists()
        events.append("audit")
        result = {"passed": True, "games": 400, "roots": 1, "continuations": 128}
        p.old.save_new(tmp_path/"main_audit.json", result)
        return result
    monkeypatch.setattr(p, "audit_stage", audit_stage)
    p.run_main()
    assert events == ["launch", "collect", "audit"]
    final = p.old.read_json(tmp_path/"final_status.json")
    assert final["complete"] and final["advance_gate_passed"] == passed
    assert not final["training"] and not final["promotion"] and not final["automatic_successor"]


def test_recorded_failure_blocks_resume_before_any_simulation(tmp_path, monkeypatch):
    monkeypatch.setattr(p, "OUTPUT", tmp_path)
    monkeypatch.setattr(sys, "argv", ["runner", "resume"])
    p.old.save_new(tmp_path/"failure_preflight.json", {"failure": "preserved"})
    monkeypatch.setattr(p, "prepare", lambda **kwargs: pytest.fail("must not retry"))
    with pytest.raises(ValueError, match="requires review"):
        p.main()
