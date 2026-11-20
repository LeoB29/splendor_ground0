"""Independent trace, feature, label and paired-statistic audit for 142."""
from __future__ import annotations

import argparse
from collections import Counter
from dataclasses import replace
from pathlib import Path
import time

import numpy as np
import torch

from splendor_ai.alphazero import run_strong_policy_pilot as pilot
from splendor_ai.alphazero.strong_policy import BudgetSearch, StrongPolicyInference, encode_position
from splendor_ai.bots.model_bot import LegalActionAttentionScorerBot
from splendor_ai.diagnostics import is_progress_transition, state_signature
from splendor_ai.encoding import canonical_successor_action_key
from splendor_ai.engine.env import SplendorEnv
from splendor_ai.termination import adjudicate_scores, is_forced_pass_turn, outcome_value
from splendor_ai.training.replay import serialize_action, serialize_state_snapshot


def audit_game(path, config, model_config, *, features=True):
    record = pilot.read_json(path)
    env = SplendorEnv(seed=record["seed"])
    state = env.initial_state()
    rows = None
    if "tensor_sha256" in record:
        tensor_path = path.with_suffix(".pt")
        assert pilot.file_hash(tensor_path) == record["tensor_sha256"]
        data = torch.load(tensor_path, weights_only=True)
        assert data["seed"] == record["seed"] and data["input_hash"] == record["input_hash"]
        rows = data["rows"]
        assert len(rows) == len(record["trace"])
    visits = Counter()
    forced = no_progress = 0
    for index, step in enumerate(record["trace"]):
        assert not state.terminal
        visits[state_signature(state)] += 1
        assert visits[state_signature(state)] < config["repetition_limit"]
        assert state.turn_index < config["max_turns"]
        assert step["turn"] == state.turn_index and step["actor"] == state.current_player
        assert step["state_hash"] == pilot.digest(serialize_state_snapshot(state))
        legal = env.legal_actions(state)
        matching = [action for action in legal if serialize_action(action) == step["action"]]
        assert len(matching) == 1
        action = matching[0]
        encoded, keys, _ = encode_position(state, model_config, legal)
        if "simulations" in step:
            assert step["simulations"] == sum(step["root_visits"])
            assert step["simulations"] >= 1
            assert len(step["root_keys"]) == len(set(step["root_keys"])) == len(keys)
            assert set(step["root_keys"]) == {repr(key) for key in keys}
            assert step["selected_key"] == repr(canonical_successor_action_key(state, action))
        if rows is not None:
            row = rows[index]
            assert row["actor"] == state.current_player and row["turn"] == state.turn_index
            if features:
                for key in ("observation", "features", "groups"):
                    assert torch.equal(row[key], encoded[key]), (path, index, key)
                assert row["group_count"] == encoded["group_count"]
            if "policy_target" in row:
                probability = dict(zip(step["root_keys"], (n / step["simulations"] for n in step["root_visits"]), strict=True))
                expected = torch.tensor([probability[repr(key)] for key in keys], dtype=torch.float32)
                assert torch.equal(row["policy_target"], expected)
                assert abs(float(expected.sum()) - 1) < 1e-6
        next_state = env.step(state, action)
        forced = forced + 1 if is_forced_pass_turn(legal) else 0
        no_progress = 0 if is_progress_transition(state, next_state, action) else no_progress + 1
        state = next_state
        if index < len(record["trace"]) - 1:
            assert forced < 2 and no_progress < config["no_progress_limit"]
    # JSON stringifies deck_counts' integer keys; canonical serialization is
    # the persisted-state equality relation, as it is for every step hash.
    assert pilot.digest(serialize_state_snapshot(state)) == pilot.digest(record["final_state"])
    reason = record["termination_reason"]
    if reason == "completed":
        assert state.terminal
        winner = state.winner
    elif reason == "forced_pass_deadlock":
        assert forced == 2
        winner = adjudicate_scores(state)
    else:
        assert not record["valid_natural"]
        if reason == "max_turns":
            assert state.turn_index >= config["max_turns"]
        elif reason == "repetition_cutoff":
            assert visits[state_signature(state)] + 1 >= config["repetition_limit"]
        elif reason == "no_progress_cutoff":
            assert no_progress >= config["no_progress_limit"]
        elif reason == "stalled":
            assert not env.legal_actions(state)
        else:
            raise AssertionError(reason)
        winner = None
    assert winner == record["winner"]
    assert record["positions"] == len(record["trace"])
    assert record["valid_natural"] == (reason in ("completed", "forced_pass_deadlock"))
    if rows is not None:
        for row in rows:
            assert row["value_target"] == (outcome_value(winner, row["actor"]) if record["valid_natural"] else None)
    if record["stage"] == "arena":
        assert record["candidate_points"] == (.5 if winner is None else float(winner == record["candidate_seat"]))
        for step in record["trace"]:
            assert step["model_index"] == int(step["actor"] == record["candidate_seat"])
    return record


def preflight(manifest):
    model = StrongPolicyInference(pilot.CHECKPOINT)
    original = LegalActionAttentionScorerBot(pilot.CHECKPOINT)
    env = SplendorEnv(seed=pilot.CONFIG["probe_seed"])
    state = env.initial_state()
    probes = []
    for turn in range(60):
        legal = env.legal_actions(state)
        logits, _, _, _, _ = model.full_prediction(state, legal)
        assert tuple(logits.tolist()) == original.score_actions(state, legal)
        if turn in (0, 20, 40):
            result = BudgetSearch(model, pilot.SEARCH).run(env, state, legal, self_play=False, seed=1429000099)
            timed = BudgetSearch(model, pilot.SEARCH).run(env, state, legal, self_play=False, seed=1429000099, seconds=.4)
            probes.append({"turn": turn, "fixed_seconds": result.elapsed_seconds,
                           "timed_seconds": timed.elapsed_seconds, "timed_simulations": timed.simulations,
                           "logits_exact": True})
        state = env.step(state, legal[int(logits.argmax())])
        if state.terminal:
            break
    old_seeds = {name: pilot.CONFIG[name] for name in ("bootstrap_seed", "selfplay_seed")}
    old_search = pilot.SEARCH
    root = pilot.OUTPUT / "probes"
    try:
        pilot.CONFIG["bootstrap_seed"] = pilot.CONFIG["probe_seed"] + 100
        pilot.CONFIG["selfplay_seed"] = pilot.CONFIG["probe_seed"] + 200
        pilot.SEARCH = replace(pilot.SEARCH, simulations=8)
        pilot._MODELS = [model]
        for stage in ("bootstrap", "selfplay"):
            pilot.play_game(stage, 0, "off_schedule_preflight", root)
            audit_game(root / stage / "game_0000.json", pilot.CONFIG, model.model.config)
    finally:
        pilot.CONFIG.update(old_seeds)
        pilot.SEARCH = old_search
    assert manifest == pilot.identity()
    report = {"manifest": pilot.digest(manifest), "status": "passed", "probes": probes,
              "whole_game_replay_and_feature_checks": 2, "protected_games_collected": 0}
    pilot.atomic_write(pilot.OUTPUT / "preflight.json", report)
    print(report, flush=True)


def verify(manifest):
    model = StrongPolicyInference(pilot.CHECKPOINT)
    counts = {}
    arena = []
    for stage, count in (("bootstrap", 256), ("selfplay", 128), ("arena", 400)):
        summary = pilot.read_json(pilot.OUTPUT / stage / "summary.json")
        assert summary["binding"]["manifest"] == pilot.digest(manifest)
        for index in range(count):
            path = pilot.OUTPUT / stage / f"game_{index:04d}.json"
            assert pilot.file_hash(path) == summary["files"][path.name]
            record = audit_game(path, pilot.CONFIG, model.model.config)
            assert record["input_hash"] == pilot.digest(summary["binding"])
            expected_seed = pilot.CONFIG[f"{stage}_seed"] + (index//2 if stage == "arena" else index)
            assert record["seed"] == expected_seed and record["index"] == index
            if stage == "selfplay":
                assert all(step["simulations"] == 128 for step in record["trace"])
            if stage == "arena":
                arena.append(record)
            if (index+1) % 32 == 0:
                print(f"verified {stage} {index+1}/{count}", flush=True)
        counts[stage] = count
    source = StrongPolicyInference(pilot.CHECKPOINT).model
    parent = StrongPolicyInference(pilot.OUTPUT / "parent.pt").model
    assert pilot.policy_hash(source) == pilot.policy_hash(parent)
    for stage, checkpoint in (("bootstrap", "parent.pt"), ("selfplay", "successor.pt")):
        report = pilot.read_json(pilot.OUTPUT / f"{stage}_training.json")
        assert report["output_sha256"] == pilot.file_hash(pilot.OUTPUT / checkpoint)
        assert report["data_summary_sha256"] == pilot.file_hash(pilot.OUTPUT / stage / "summary.json")
        assert not set(report["train_game_indices"]) & set(report["validation_game_indices"])
        assert sorted(report["train_game_indices"] + report["validation_game_indices"]) == list(range(counts[stage]))
    # Independently calculate the scalar statistic and exact seeded pair bootstrap.
    pair_values = np.array([(arena[2*i]["candidate_points"] + arena[2*i+1]["candidate_points"])/2 for i in range(200)])
    rng = np.random.default_rng(142999)
    samples = [float(pair_values[rng.integers(0, 200, 200)].mean()) for _ in range(50000)]
    ci = np.percentile(samples, [2.5, 97.5]).tolist()
    result = pilot.read_json(pilot.OUTPUT / "result.json")
    assert abs(result["candidate_score_share"] - float(pair_values.mean())) < 1e-12
    assert np.allclose(ci, result["paired_ci95"], atol=1e-12, rtol=0)
    assert manifest == pilot.identity()
    report = {"status": "passed", "manifest": pilot.digest(manifest), "games_verified": counts,
              "all_trajectories_legal": True, "all_training_features_and_targets_reconstructed": True,
              "parent_policy_exactly_035": True, "paired_ci95": ci,
              "result_sha256": pilot.file_hash(pilot.OUTPUT / "result.json")}
    pilot.atomic_write(pilot.OUTPUT / "verification.json", report)
    print(report, flush=True)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--preflight", action="store_true")
    args = parser.parse_args()
    torch.set_num_threads(1)
    manifest = pilot.read_json(pilot.OUTPUT / "manifest.json")
    assert manifest == pilot.identity()
    if args.preflight:
        preflight(manifest)
    else:
        verify(manifest)


if __name__ == "__main__":
    main()
