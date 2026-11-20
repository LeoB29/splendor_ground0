"""Recovery tests use tiny external fixtures and never repeat production games."""
import json
import subprocess
import sys

import pytest

from scripts.analysis import recover_learned_continuation_158 as r
from splendor_ai.engine.actions import ActionType


def test_existing_certificate_tamper_and_extra_record_are_rejected(tmp_path):
    path = tmp_path/"test/replay/game_0000.json"
    record = {"manifest": "fixture", "game_sha256": "game-sha",
              "reconstruction": {"game": {"index": 0}, "audit_timing": {"wall": 1., "cpu": 1.}}}
    r.old.save_new(path, record)
    hashes = {"0": r.old.file_hash(path)}
    files = {"0": "game-sha"}
    assert set(r.certificates(tmp_path, "test", files, "fixture", hashes)) == {0}
    with pytest.raises(AssertionError):
        r.certificates(tmp_path, "test", {"0": "wrong-game"}, "fixture", hashes)
    record["reconstruction"]["game"]["index"] = 1
    r.old.atomic_write(path, record)
    with pytest.raises(AssertionError, match="Prior replay changed"):
        r.certificates(tmp_path, "test", files, "fixture", hashes)
    r.old.save_new(tmp_path/"test/replay/game_0001.json", record)
    with pytest.raises(AssertionError, match="Unexpected/missing"):
        r.certificates(tmp_path, "test", files, "fixture", hashes)


def test_no_completed_certificate_is_dispatched(tmp_path, monkeypatch):
    def forbidden_pool(*args, **kwargs):
        raise AssertionError("No replay may be repeated")
    monkeypatch.setattr(r, "ProcessPoolExecutor", forbidden_pool)
    rows = {0: {"already": "certified"}}
    assert r.replay_missing(tmp_path, "test", {"0": "sha"}, "binding", rows) is rows


def test_partial_panel_cannot_close_on_failed_strength_gate():
    with pytest.raises(AssertionError, match="Full402"):
        r.combine({i: {"statistical_pass": False} for i in range(401)})


def test_exact_recovery_cli_preserves_one_certificate_and_spawns_missing_replay(tmp_path, monkeypatch):
    h = r.p.h
    h.initialize_worker()
    monkeypatch.setattr(h, "OUTPUT", tmp_path)
    env = h.old.SplendorEnv(seed=1589800000)
    state = env.initial_state()
    for _ in range(2):
        action = next(x for x in env.legal_actions(state)
                      if x.action_type == ActionType.RESERVE_DECK and x.tier == 2)
        state = env.step(state, action)
    for player in state.players:
        player.bonuses = {color: 5 for color in player.bonuses}
        player.score = 14
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
    binding = "158-recovery-test-binding"
    h.old.save_new(tmp_path/"test_launch.json", {"manifest": binding})
    with monkeypatch.context() as patch:
        patch.setattr(h.old, "SplendorEnv", TinyEnv)
        games = [h.play_game("test", r.a.test_spec(i), binding) for i in (0, 1)]
    context = {"base_dir": str(tmp_path.resolve()), "initial_snapshot": snapshot, "spec": r.a.test_spec(0)}
    prior = r.a.audit_game("test", 0, games[0]["sha256"], binding, context)
    prior_path = tmp_path/"test/replay/game_0000.json"
    r.old.save_new(prior_path, {"manifest": binding, "game_sha256": games[0]["sha256"], "reconstruction": prior})
    prior_sha = r.old.file_hash(prior_path)
    fixture = {"kind": "158-spawn-test-fixture-v1", "games": games, "binding": binding,
               "base_dir": str(tmp_path.resolve()), "initial_snapshot": snapshot,
               "prior_replay_hashes": {"0": prior_sha}}
    fixture_path = tmp_path/"fixture.json"
    r.old.save_new(fixture_path, fixture)
    completed = subprocess.run([sys.executable, str(r.SCRIPT), "spawn-probe", "--fixture", str(fixture_path)],
                               capture_output=True, text=True, check=True, timeout=120)
    output = json.loads(completed.stdout.strip().splitlines()[-1])
    assert output["passed"] and output["reused"] == output["newly_replayed"] == 1
    assert output["roots"] >= 4
    assert r.old.file_hash(prior_path) == prior_sha
    recovered = r.old.read_json(tmp_path/"test/replay/game_0001.json")
    assert recovered["game_sha256"] == games[1]["sha256"]
    assert {root["planner"] for root in recovered["reconstruction"]["roots"]} == {"candidate", "reference"}
