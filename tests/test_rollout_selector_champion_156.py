import copy
from dataclasses import asdict
import importlib.util
from pathlib import Path
import sys

import pytest

from splendor_ai.alphazero import run_rollout_selector_champion as p
from splendor_ai.alphazero import resolver_champion_156 as c
from splendor_ai.engine.actions import Action, ActionType
from splendor_ai.training.run_action_preference_repair import rollout_score_margin_loop_safe


@pytest.fixture(scope="module")
def audit():
    spec = importlib.util.spec_from_file_location("audit156_test", "scripts/analysis/verify_rollout_selector_champion_156.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


@pytest.fixture(scope="module")
def scorer():
    p.old.initialize_worker()
    return c.r084._LegalActionScorer(c.r084.CHECKPOINT, "cpu", "legal_action_attention_scorer")


def private_state(seat=0):
    env = p.old.SplendorEnv(seed=156)
    state = env.initial_state()
    for _ in range(2):
        action = next(a for a in env.legal_actions(state) if a.action_type == ActionType.RESERVE_DECK and a.tier == 2)
        state = env.step(state, action)
    state.current_player = seat
    # Synthetic legal boundary fixture; all card identities remain in inventory.
    for player in state.players:
        player.bonuses = {color: 3 for color in player.bonuses}
        player.score = 8
    return state


def test_frozen_contracts_and_fresh_rng_schedule(audit):
    assert p.choose_root is p.parent.choose_root and p.select is p.parent.select
    assert p.CONFIG["games"] == 400 and p.CONFIG["probe_games"] == p.CONFIG["workers"] == 8
    streams, environments, world_seeds, continuation_seeds = set(), set(), set(), set()
    for stage, n in (("probe", 8), ("main", 400)):
        cfg = c.config(stage)
        assert c.frozen_contract_matches(stage)
        assert cfg.determinizations*cfg.root_actions*cfg.opponent_actions*cfg.discovery_continuations == 192
        assert p.schedule(stage) == [audit.independent_spec(stage, i) for i in range(n)]
        es = {r["environment_seed"] for r in p.schedule(stage)}
        ss = {r["search_stream"] for r in p.schedule(stage)}
        assert not environments & es and not streams & ss
        assert not es & {r["environment_seed"] for r in p.parent.schedule(stage)}
        environments |= es
        streams |= ss
        for index in range(n):
            for decision in range(200):
                seat = 1-index%2
                root_index = c.r084.gameplay_belief_root_index(cfg, index, seat, decision)
                for det in range(3):
                    seed = 49000000+root_index*10003+det
                    assert seed not in world_seeds
                    world_seeds.add(seed)
                    for rep in range(4):
                        seed = c.r084.gameplay_continuation_seed(cfg, index, seat, decision, det, rep)
                        assert seed not in continuation_seeds and 0 <= seed < 2**63
                        continuation_seeds.add(seed)
    assert len(streams) == 408 and p.CONFIG["main_continuation_ceiling"] == 400*200*(128+192)


@pytest.mark.parametrize("seat", [0, 1])
def test_full_champion_search_matches_reference_replay_and_hidden_invariance(tmp_path, monkeypatch, audit, scorer, seat):
    monkeypatch.setattr(p, "OUTPUT", tmp_path)
    p.old.save_new(tmp_path/"probe_launch.json", {"manifest": "test"})
    state = private_state(seat)
    # Keep the acting player first in the round so root purchases do not end it.
    state.start_player = seat
    bot = c.RecordedChampion("probe", seat, seat, scorer=scorer)
    reference = c.r084.FrozenPublicBeliefResolverBot(c.config("probe"), seat, seat, scorer=scorer)
    before = reference.telemetry
    path = p.champion_root_path("probe", seat, state.turn_index)
    action, record = bot.choose_recorded(state, path, "test")
    assert record["kind"] == "search" and len(record["branches"]) == 192
    move = {"action": p.old.serialize_action(action), "champion_root_sha256": p.old.file_hash(path)}
    summary, after = audit.audit_champion("probe", seat, state, move, "test", scorer, before, reference)
    assert after == bot.telemetry and summary == c.summary(record, seat, state.turn_index)
    changed = audit.hidden_alternative(state)
    alternate = c.RecordedChampion("probe", seat, seat, scorer=scorer)
    action2, other = alternate.choose_recorded(changed, tmp_path/"alternate.json", "test")
    assert action2 == action
    for key in ("scores", "worlds", "branches", "action", "parent", "resolution", "after"):
        audit.compare(record[key], other[key])
    resumed = c.RecordedChampion("probe", seat, seat, scorer=scorer)
    assert resumed.choose_recorded(state, path, "test")[0] == action
    assert resumed.telemetry == bot.telemetry
    # Hash-valid content tampering must still fail reconstruction.
    tampered = p.old.read_json(path)
    tampered["branches"][0]["trace_sha256"] = "tampered"
    p.old.atomic_write(path, tampered)
    move["champion_root_sha256"] = p.old.file_hash(path)
    with pytest.raises(AssertionError):
        audit.audit_champion("probe", seat, state, move, "test", scorer, before)


def test_champion_terminal_root_fallback_and_exhausted_deck(tmp_path, monkeypatch, audit, scorer):
    monkeypatch.setattr(p, "OUTPUT", tmp_path)
    p.old.save_new(tmp_path/"probe_launch.json", {})
    state = private_state()
    state.start_player = 1
    state.players[0].score = 14
    state.players[0].bonuses = {color: 5 for color in state.players[0].bonuses}
    state.players[1].purchased_cards.extend(state.hidden_tier_decks[1])
    state.hidden_tier_decks[1] = []
    state.deck_counts[1] = 0
    bot = c.RecordedChampion("probe", 0, 0, scorer=scorer)
    reference = c.r084.FrozenPublicBeliefResolverBot(c.config("probe"), 0, 0, scorer=scorer)
    before = reference.telemetry
    path = p.champion_root_path("probe", 0, state.turn_index)
    action, row = bot.choose_recorded(state, path, "test")
    assert row["kind"] == "fallback" and row["reason"] == "root_action_terminal" and not row["branches"]
    move = {"action": p.old.serialize_action(action), "champion_root_sha256": p.old.file_hash(path)}
    audit.audit_champion("probe", 0, state, move, "test", scorer, before, reference)


def test_capture_hooks_restore_originals_after_failure(tmp_path, scorer, monkeypatch):
    worlds = c.r084.deterministic_belief_range
    def failed(*args, **kwargs):
        raise RuntimeError("injected test failure")
    monkeypatch.setattr(c.r084, "rollout_score_margin_loop_safe", failed)
    bot = c.RecordedChampion("probe", 0, 0, scorer=scorer)
    path = tmp_path/"failure.json"
    state = private_state()
    with pytest.raises(RuntimeError, match="injected"):
        bot.choose_recorded(state, path, "test")
    assert c.r084.deterministic_belief_range is worlds and c.r084.rollout_score_margin_loop_safe is failed
    sha = p.old.file_hash(path)
    assert p.old.read_json(path)["failure"]
    with pytest.raises(AssertionError):
        c.RecordedChampion("probe", 0, 0, scorer=scorer).choose_recorded(state, path, "test")
    assert p.old.file_hash(path) == sha


def test_independent084_loop_adjudication_and_terminal_response(audit, scorer, monkeypatch):
    state = p.old.SplendorEnv(seed=156).initial_state()
    action = Action(action_type=ActionType.PASS)
    class CycleEnv:
        def legal_actions(self, state):
            return [action]
        def step(self, state, action):
            nxt = state.copy_shallow()
            nxt.turn_index += 1
            nxt.current_player = 1-state.current_player
            return nxt
    class Scores:
        def score_actions(self, state, legal):
            from splendor_ai.training.run_policy_search_disagreement import ModelActionScore
            return (ModelActionScore(action=action, action_index=0, score=0.),)
    env, fake = CycleEnv(), Scores()
    original, row = c.traced_branch(rollout_score_margin_loop_safe, env, fake, state, action, 0, 156, .1, .5, 3, 400, 20., 64)
    monkeypatch.setattr(p.old, "SplendorEnv", CycleEnv)
    replay = audit.replay_champion_branch(state, action, 0, 156, fake)
    assert row == replay and original.loop_adjudicated and original.resolved and not original.hard_cap_reached
    assert original.fallback_triggers == 64


@pytest.mark.parametrize("seat", [0, 1])
def test_complete_mixed_game_audit_and_observation_summary(tmp_path, monkeypatch, audit, scorer, seat):
    monkeypatch.setattr(p, "OUTPUT", tmp_path)
    original_env = p.old.SplendorEnv
    class TinyEnv:
        def __init__(self, seed=None):
            self.env = original_env(seed=156)
        def initial_state(self):
            state = self.env.initial_state()
            for player in state.players:
                player.bonuses = {color: 5 for color in player.bonuses}
                player.score = 14
            return state
        def legal_actions(self, state):
            return self.env.legal_actions(state)
        def step(self, state, action):
            return self.env.step(state, action)
    monkeypatch.setattr(p.old, "SplendorEnv", TinyEnv)
    #084 constructs its environment internally; match the fixture there too.
    monkeypatch.setattr(c.r084, "SplendorEnv", TinyEnv)
    p.old.save_new(tmp_path/"probe_launch.json", {})
    spec = p.schedule("probe")[seat]
    info = p.play_game("probe", spec, "test")
    checked = audit.audit_game("probe", seat, info["sha256"], "test")
    assert p.play_game("probe", spec, "test") == info
    monkeypatch.setattr(p, "schedule", lambda stage: [spec])
    p.old.save_new(tmp_path/"probe_games_lock.json", {"files": {str(seat): info["sha256"]}})
    data = p.observations("probe")
    assert data == {"games": [checked["game"]], "roots": checked["roots"], "champion_roots": checked["champion_roots"]}
    audit.compare(p.operational(data), audit.independent_operational(data))


def game_rows(scores):
    return [{"index": i, "pair": i//2, "seat": i%2, "score": score,
             "margin": score*2-1, "turns": 64} for i, score in enumerate(scores)]


@pytest.mark.parametrize("scores", [[1.]*240+[0.]*160, [1., 0.]*200, [.5]*400])
def test_pair_bootstrap_and_all_gates_are_independently_reconstructed(scores, audit):
    rows = game_rows(scores)
    audit.compare(p.statistics(rows), audit.independent_statistics(rows))
    if scores == [1., 0.]*200:
        result = p.statistics(rows)
        assert result["ci95"] == [.5, .5] and not result["statistical_gates"]["both_seats"]


def test_margin_and_seat_safeguards_can_block_otherwise_passing_match(audit):
    rows = game_rows([1.]*240+[0.]*160)
    assert p.statistics(rows)["statistical_pass"]
    for r in rows:
        r["margin"] = -1
    result = p.statistics(rows)
    assert not result["statistical_pass"] and not result["statistical_gates"]["positive_margin"]
    rows = game_rows([1., 0.]*160+[1., 1.]*40)
    result = p.statistics(rows)
    assert result["statistical_gates"]["material_share"] and not result["statistical_gates"]["both_seats"]
    audit.compare(result, audit.independent_statistics(rows))


@pytest.mark.parametrize("passed", [False, True])
def test_full_audit_runs_regardless_of_strength_gate(tmp_path, monkeypatch, passed):
    monkeypatch.setattr(p, "OUTPUT", tmp_path)
    monkeypatch.setattr(p, "check_identity", lambda: "test")
    monkeypatch.setattr(p, "verify_test_evidence", lambda: "tests")
    for name, row in (("probe_audit.json", {"passed": True}), ("timing_probe.json", {}),
                      ("freeze.json", {"tests_sha256": "tests"}), ("probe_launch.json", {"time": p.old.now()})):
        p.old.save_new(tmp_path/name, row)
    p.old.save_new(tmp_path/"preflight.json", {"passed": True, "manifest": "test",
                   "audit_sha256": p.old.file_hash(tmp_path/"probe_audit.json"),
                   "timing_sha256": p.old.file_hash(tmp_path/"timing_probe.json")})
    events = []
    monkeypatch.setattr(p, "launch_stage", lambda *args: events.append("launch"))
    monkeypatch.setattr(p, "collect", lambda *args: events.append("collect"))
    monkeypatch.setattr(p, "observations", lambda *args: {"games": []})
    monkeypatch.setattr(p, "statistics", lambda *args: {"statistical_pass": passed, "share": .6,
                        "ci95": [.4, .7], "wdl": {"wins": 200, "draws": 0, "losses": 200}})
    monkeypatch.setattr(p, "operational", lambda *args: {"continuations": 128, "champion": {"continuations": 192}})
    def audited(stage):
        assert (tmp_path/"result.json").exists() and not (tmp_path/"final_status.json").exists()
        events.append("audit")
        row = {"passed": True, "games": 400, "roots": 1, "champion_roots": 1, "continuations": 320}
        p.old.save_new(tmp_path/"main_audit.json", row)
        return row
    monkeypatch.setattr(p, "audit_stage", audited)
    p.run_main()
    assert events == ["launch", "collect", "audit"]
    assert p.old.read_json(tmp_path/"final_status.json")["advance_gate_passed"] == passed


def test_failure_refuses_automatic_retry(tmp_path, monkeypatch):
    monkeypatch.setattr(p, "OUTPUT", tmp_path)
    p.old.save_new(tmp_path/"failure_preserved.json", {})
    monkeypatch.setattr(sys, "argv", ["156", "resume"])
    monkeypatch.setattr(p, "prepare", lambda **kwargs: pytest.fail("must not run"))
    with pytest.raises(ValueError, match="requires review"):
        p.main()
