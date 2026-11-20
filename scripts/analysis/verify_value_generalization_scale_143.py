"""Independent all-game behavior, feature, model-selection and result audit."""
from __future__ import annotations

import argparse
from collections import Counter
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path
import random
import time

import numpy as np
import torch

from splendor_ai.alphazero import run_value_generalization_scale as experiment
from splendor_ai.alphazero.strong_policy import StrongPolicyInference
from splendor_ai.bots.model_bot import LegalActionAttentionScorerBot
from splendor_ai.diagnostics import is_progress_transition, state_signature
from splendor_ai.encoding import encode_public_observation_tensor, fit_feature_vector
from splendor_ai.engine.constants import TOKEN_COLORS
from splendor_ai.engine.env import SplendorEnv
from splendor_ai.termination import adjudicate_scores, is_forced_pass_turn, outcome_value
from splendor_ai.training.replay import serialize_action, serialize_state_snapshot, _serialize_state_for_diagnostics

_SCORER = None
_HEADS = None
_COEFFICIENTS = None


def initialize(test_models=False):
    global _SCORER, _HEADS, _COEFFICIENTS
    torch.set_num_threads(1)
    _SCORER = LegalActionAttentionScorerBot(experiment.CHECKPOINT)
    if test_models:
        _HEADS = [StrongPolicyInference(experiment.OUTPUT / f"{name}.pt").model.value_head for name in ("small", "large")]
        _COEFFICIENTS = torch.tensor(experiment.read_json(experiment.OUTPUT / "simple.json")["coefficients"], dtype=torch.float64)


def audit_one(index, root=None, protected=True):
    root = experiment.OUTPUT if root is None else Path(root)
    path = root / "games" / f"game_{index:04d}.json"
    record = experiment.read_json(path)
    assert experiment.file_hash(path.with_suffix(".pt")) == record["tensor_sha256"]
    data = torch.load(path.with_suffix(".pt"), weights_only=True)
    assert data["index"] == record["index"] == index
    assert data["seed"] == record["seed"] and data["binding"] == record["binding"]
    if protected:
        assert record["seed"] == 1_430_000_000 + index
        assert record["split"] == ("training" if index < 4096 else "validation" if index < 4608 else "test")
        summary_name = "development_summary.json" if index < 4608 else "test_summary.json"
        summary = experiment.read_json(root / summary_name)
        assert summary["files"][str(index)] == experiment.file_hash(path)
        assert record["binding"] == summary["manifest"]
    env = SplendorEnv(seed=record["seed"])
    state = env.initial_state()
    rng = random.Random(record["seed"] + 143003)
    visits = Counter()
    forced = no_progress = 0
    observations = []
    for row_index, step in enumerate(record["trace"]):
        assert not state.terminal and state.turn_index < 400
        visits[state_signature(state)] += 1
        assert visits[state_signature(state)] < 4
        assert step["state_hash"] == experiment.digest(serialize_state_snapshot(state))
        assert step["actor"] == int(data["actors"][row_index]) == state.current_player
        assert step["turn"] == int(data["turns"][row_index]) == state.turn_index
        legal = env.legal_actions(state)
        scores = _SCORER.score_actions(state, legal)
        best = max(range(len(scores)), key=lambda i: scores[i])
        expected = best
        if state.turn_index < 20 and rng.random() < .2:
            top = sorted(range(len(scores)), key=lambda i: (-scores[i], i))[:3]
            weights = torch.softmax(torch.tensor(scores)[top] / .5, 0).tolist()
            expected = rng.choices(top, weights=weights, k=1)[0]
        assert step["greedy_row"] == best and step["selected_row"] == expected
        assert serialize_action(legal[expected]) == step["action"]
        obs = torch.tensor(fit_feature_vector(encode_public_observation_tensor(state, state.current_player).vector,
                                             _SCORER._model.config.observation_size), dtype=torch.float32)
        assert torch.equal(obs, data["observations"][row_index])
        observations.append(obs)
        # Reconstruct the small evaluator's contract directly from public state.
        a, b = state.players[state.current_player], state.players[1-state.current_player]
        public = [a.score/15-b.score/15]
        public.extend((a.bonuses[c]-b.bonuses[c])/7 for c in TOKEN_COLORS)
        public.extend((a.tokens[c]-b.tokens[c])/7 for c in TOKEN_COLORS)
        public.extend(((a.tokens["gold"]-b.tokens["gold"])/5,
                       (len(a.purchased_cards)-len(b.purchased_cards))/20,
                       (len(a.reserved_cards)-len(b.reserved_cards))/3,
                       (len(a.nobles)-len(b.nobles))/3))
        assert torch.allclose(torch.tensor(public), data["simple"][row_index], atol=1e-7, rtol=0)
        next_state = env.step(state, legal[expected])
        forced = forced + 1 if is_forced_pass_turn(legal) else 0
        no_progress = 0 if is_progress_transition(state, next_state, legal[expected]) else no_progress + 1
        state = next_state
        if row_index < len(record["trace"]) - 1:
            assert forced < 2 and no_progress < 60
    # collect_game_replay exposes its diagnostic endpoint schema, while the
    # per-move hashes above use complete reconstructable state snapshots.
    assert experiment.digest(_serialize_state_for_diagnostics(state)) == experiment.digest(record["final_state"])
    reason = record["termination_reason"]
    if reason == "completed":
        assert state.terminal
        winner = state.winner
    elif reason == "forced_pass_deadlock":
        assert forced == 2
        winner = adjudicate_scores(state)
    else:
        winner = None
        if reason == "max_turns":
            assert state.turn_index >= 400
        elif reason == "repetition_cutoff":
            assert visits[state_signature(state)] + 1 >= 4
        elif reason == "no_progress_cutoff":
            assert no_progress >= 60
        elif reason == "stalled":
            assert not env.legal_actions(state)
        else:
            raise AssertionError(reason)
    assert winner == record["winner"]
    assert record["valid_natural"] == (reason in ("completed", "forced_pass_deadlock"))
    assert len(record["trace"]) == record["positions"] == len(data["latent"])
    with torch.no_grad():
        latent = _SCORER._model.state_encoder(torch.stack(observations))
    assert torch.equal(latent, data["latent"])
    if record["valid_natural"]:
        labels = torch.tensor([outcome_value(winner, int(actor)) for actor in data["actors"]])
        assert torch.equal(labels, data["targets"])
    else:
        assert data["targets"] is None
    mse = None
    if protected and index >= 4608 and record["valid_natural"]:
        predicted = torch.load(root / "test_predictions" / f"game_{index:04d}.pt", weights_only=True)
        assert predicted["selection_sha256"] == experiment.file_hash(root / "selection_lock.json")
        with torch.no_grad():
            actual = np.column_stack((np.zeros(len(data["targets"])), _HEADS[0](latent).squeeze(-1).numpy(),
                                     _HEADS[1](latent).squeeze(-1).numpy(),
                                     (data["simple"].double() @ _COEFFICIENTS).clamp(-1, 1).numpy()))
        assert np.array_equal(actual, predicted["predictions"].numpy())
        mse = ((actual - data["targets"].numpy()[:, None])**2).mean(0)
        assert np.allclose(mse, predicted["mse"].numpy(), rtol=0, atol=1e-12)
    return {"index": index, "positions": record["positions"], "valid": record["valid_natural"],
            "termination": reason, "mse": None if mse is None else mse.tolist()}


def audit_training(manifest):
    source = StrongPolicyInference(experiment.CHECKPOINT).model
    source_hash = experiment.policy_hash(source)
    for name, ids in (("small", experiment.small_indices()), ("large", list(range(4096)))):
        report = experiment.read_json(experiment.OUTPUT / f"{name}_training.json")
        assert report["manifest"] == experiment.digest(manifest) and report["status"] == "complete"
        assert set(report["training_ids"]) <= set(ids)
        assert len(report["training_ids"]) >= math_ceil_98(len(ids))
        assert set(report["validation_ids"]) <= set(range(4096, 4608))
        assert not set(report["training_ids"]) & set(report["validation_ids"])
        best, best_step, stale = report["history"][0]["validation_game_mse"], 0, 0
        for checkpoint_number, item in enumerate(report["history"][1:], 1):
            assert item["step"] == checkpoint_number * 128 <= 4096
            improves = item["validation_game_mse"] < best - .001
            assert item["selected"] == improves
            if improves:
                best, best_step, stale = item["validation_game_mse"], item["step"], 0
            else:
                stale += 1
            if checkpoint_number < len(report["history"]) - 1:
                assert stale < 4
        assert stale >= 4 or report["completed_updates"] == 4096
        assert best_step == report["selected_step"]
        assert abs(best - report["selected_validation_game_mse"]) < 1e-7
        model = StrongPolicyInference(experiment.OUTPUT / f"{name}.pt").model
        assert experiment.policy_hash(model) == source_hash == report["final_policy_sha256"]
        chosen = torch.load(experiment.OUTPUT / "selected_heads" / f"{name}_{best_step:04d}.pt", weights_only=True)["state_dict"]
        assert all(torch.equal(tensor, chosen[key]) for key, tensor in model.value_head.state_dict().items())
    baseline = experiment.read_json(experiment.OUTPUT / "simple.json")
    assert baseline["selected_index"] == min(range(5), key=lambda i: (baseline["candidates"][i]["validation_game_mse"], i))
    assert [r["ridge"] for r in baseline["candidates"]] == [.001, .01, .1, 1., 10.]
    training = experiment.load_data(list(range(4096)), manifest)
    validation = experiment.load_data(list(range(4096, 4608)), manifest)
    # Verify each ridge normal equation and held-out selection metric directly.
    x, y, w = training.simple.double(), training.y.double(), training.weights.double() / len(training.ids)
    gram, rhs = x.T @ (x * w[:, None]), x.T @ (y * w)
    for candidate in baseline["candidates"]:
        coefficients = torch.tensor(candidate["coefficients"], dtype=torch.float64)
        residual = (gram + candidate["ridge"] * torch.eye(15, dtype=torch.float64)) @ coefficients - rhs
        assert float(residual.abs().max()) < 1e-10
        predicted = (validation.simple.double() @ coefficients).clamp(-1, 1).float()
        assert abs(float(experiment.game_mse(predicted, validation)) - candidate["validation_game_mse"]) < 1e-7
    for name in ("small", "large"):
        head = StrongPolicyInference(experiment.OUTPUT / f"{name}.pt").model.value_head
        actual = experiment.head_predictions(head, validation)
        expected = experiment.read_json(experiment.OUTPUT / f"{name}_training.json")["selected_validation_game_mse"]
        assert abs(float(experiment.game_mse(actual, validation)) - expected) < 1e-7


def math_ceil_98(n):
    return (98 * n + 99) // 100


def preflight(manifest):
    initialize()
    experiment.initialize_worker()
    started = time.perf_counter()
    root = experiment.OUTPUT / "probes"
    records = []
    for index in range(2):
        experiment.collect_one(index, experiment.digest(manifest), root=root, seed_override=experiment.CONFIG["probe_seed"] + index)
        records.append(audit_one(index, root, protected=False))
    assert experiment.identity() == manifest
    elapsed = time.perf_counter() - started
    size = sum(p.stat().st_size for p in (root / "games").glob("*"))
    report = {"manifest": experiment.digest(manifest), "status": "passed", "probe_games": 2,
              "probe_positions": sum(r["positions"] for r in records), "probe_wall_seconds": elapsed,
              "complete_behavior_trace_observation_latent_baseline_label_audit": True,
              "projected_serial_collect_plus_audit_hours": elapsed * 2560 / 3600,
              "projected_corpus_bytes": size * 2560, "protected_games": 0}
    assert report["projected_serial_collect_plus_audit_hours"] < 24
    assert report["projected_corpus_bytes"] < 20_000_000_000
    experiment.atomic_write(experiment.OUTPUT / "preflight.json", report)
    print(report, flush=True)


def verify(manifest):
    experiment.check_selection_lock(manifest)
    audit_training(manifest)
    rows = []
    with ProcessPoolExecutor(max_workers=8, initializer=initialize, initargs=(True,)) as pool:
        jobs = [pool.submit(audit_one, index) for index in range(5120)]
        for completed, job in enumerate(as_completed(jobs), 1):
            rows.append(job.result())
            if completed % 128 == 0:
                print(f"verified {completed}/5120 games", flush=True)
    rows.sort(key=lambda row: row["index"])
    errors = np.array([row["mse"] for row in rows if row["index"] >= 4608 and row["mse"] is not None])
    result = experiment.read_json(experiment.OUTPUT / "result.json")
    assert np.allclose(errors, result["game_mse_rows"], atol=1e-12, rtol=0)
    rng = np.random.default_rng(143999)
    boot = np.empty((50000, 4))
    for i in range(50000):
        boot[i] = errors[rng.integers(0, len(errors), len(errors))].mean(0)
    means = errors.mean(0)
    for i, name in enumerate(("neutral", "small", "large", "simple")):
        assert abs(means[i] - result["mse"][name]) < 1e-12
    for name, a, b in (("large_minus_neutral", 2, 0), ("large_minus_small", 2, 1), ("large_minus_simple", 2, 3), ("simple_minus_neutral", 3, 0)):
        assert abs(means[a]-means[b] - result["contrasts"][name]["mean"]) < 1e-12
        expected = np.percentile(boot[:, a]-boot[:, b], [2.5, 97.5])
        assert np.allclose(expected, result["contrasts"][name]["ci95"], atol=1e-12, rtol=0)
    assert result["natural_test_gate"] == all(row["valid"] for row in rows[4608:])
    assert result["large_beats_neutral"] == (np.percentile(boot[:, 2]-boot[:, 0], 97.5) < 0)
    assert result["large_material_improvement"] == (result["large_beats_neutral"] and means[2] <= .95*means[0])
    assert result["scale_benefit_established"] == (np.percentile(boot[:, 2]-boot[:, 1], 97.5) < 0)
    assert experiment.identity() == manifest
    report = {"status": "passed", "manifest": experiment.digest(manifest), "games_verified": len(rows),
              "positions_verified": sum(r["positions"] for r in rows),
              "termination_counts": dict(Counter(r["termination"] for r in rows)),
              "original_035_behavior_reproduced": True, "all_features_latents_labels_reconstructed": True,
              "selected_checkpoint_policy_retention": True, "selection_rule_verified": True,
              "all_test_predictions_and_statistics_reconstructed": True,
              "result_sha256": experiment.file_hash(experiment.OUTPUT / "result.json")}
    experiment.atomic_write(experiment.OUTPUT / "verification.json", report)
    print(report, flush=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--preflight", action="store_true")
    args = parser.parse_args()
    torch.set_num_threads(1)
    manifest = experiment.read_json(experiment.OUTPUT / "manifest.json")
    assert experiment.identity() == manifest
    preflight(manifest) if args.preflight else verify(manifest)
