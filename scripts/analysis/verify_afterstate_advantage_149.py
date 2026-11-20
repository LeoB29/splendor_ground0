"""Separate-process complete replay/refit audit and independent statistics for 149."""
from collections import Counter
from concurrent.futures import ProcessPoolExecutor, as_completed
import hashlib
from pathlib import Path
import random
import time

import numpy as np
import torch

from splendor_ai.alphazero import run_afterstate_advantage as p
from splendor_ai.alphazero.afterstate_149 import deterministic_part, features
from splendor_ai.bots.hidden_information import determinize_for_observer
from splendor_ai.diagnostics import state_signature, is_progress_transition
from splendor_ai.engine.env import SplendorEnv
from splendor_ai.training.replay import serialize_state_snapshot, deserialize_state_snapshot, serialize_action
from splendor_ai.termination import adjudicate_scores, VALID_NATURAL_TERMINATIONS


def replay_branch(world, action, actor):
    env = SplendorEnv()
    pending = deterministic_part(world, action)
    draw = None if pending.chance_kind is None else world.hidden_tier_decks[pending.chance_tier][0].card_id
    composed = pending.resolve(draw)
    assert serialize_state_snapshot(composed) == serialize_state_snapshot(env.step(world, action))
    current, count, passes, idle = world, 0, 0, 0
    visits, hashes = Counter(), hashlib.sha256()
    winner, reason = None, "completed"
    while not current.terminal:
        visits[state_signature(current)] += 1
        if visits[state_signature(current)] >= 4:
            reason = "repetition_cutoff"
            break
        if current.turn_index >= 400:
            reason = "max_turns"
            break
        legal = env.legal_actions(current)
        if not legal:
            reason = "stalled"
            break
        if count == 0:
            chosen, nxt = action, composed
        else:
            scores = p._RAW.score_actions(current, legal)
            chosen = legal[max(range(len(legal)), key=lambda i: scores[i])]
            nxt = env.step(current, chosen)
        hashes.update(p.digest(serialize_action(chosen)).encode())
        passes = passes + 1 if len(legal) == 1 and legal[0].action_type.name == "PASS" else 0
        idle = 0 if is_progress_transition(current, nxt, chosen) else idle + 1
        current = nxt
        count += 1
        if passes == 2:
            reason, winner = "forced_pass_deadlock", adjudicate_scores(current)
            break
        if idle >= 60:
            reason = "no_progress_cutoff"
            break
    if reason == "completed":
        winner = current.winner
    valid = reason in VALID_NATURAL_TERMINATIONS
    return {"reason": reason, "winner": winner, "turns": count, "trace_sha256": hashes.hexdigest(),
            "final_state_sha256": p.digest(serialize_state_snapshot(current)),
            "wdl": (0. if winner is None else 1. if winner == actor else -1.) if valid else None}


def audit_sample(split, sample):
    source = deserialize_state_snapshot(sample["snapshot"])
    record = p.read_json(p.OUTPUT / "targets" / split / f"state_{sample['index']:04d}.json")
    assert len(record["worlds"]) == p.CONFIG["worlds"][split]
    assert record["sample_sha256"] == p.digest(sample)
    count, types, reasons, turns, paired_variance = 0, Counter(), Counter(), 0, []
    values = []
    for wi, saved in enumerate(record["worlds"]):
        seed = p.CONFIG["world_seed_start"][split] + sample["index"] * 1000 + wi
        assert saved["seed"] == seed
        world = determinize_for_observer(source, source.current_player, random.Random(seed))
        assert p.digest(serialize_state_snapshot(world)) == saved["world_sha256"]
        results = []
        for ai, payload in enumerate(sample["actions"]):
            action = p.decode_action(payload)
            if wi == 0:
                for arm in p.ARMS:
                    assert torch.equal(features(source, action, arm), features(world, action, arm))
            rebuilt = replay_branch(world, action, source.current_player)
            assert rebuilt == saved["outcomes"][ai], (split, sample["index"], wi, ai, rebuilt, saved["outcomes"][ai])
            assert rebuilt["reason"] in VALID_NATURAL_TERMINATIONS
            types[action.action_type.name] += 1
            reasons[rebuilt["reason"]] += 1
            turns += rebuilt["turns"]
            count += 1
            results.append(rebuilt["wdl"])
        values.append(results)
    values = np.asarray(values)
    if values.shape[1] > 1:
        paired_variance = ((values[:, 1:]-values[:, :1]).var(0, ddof=1) / len(values)).tolist()
    return {"continuations": count, "moves": turns, "root_action_types": dict(types),
            "termination_counts": dict(reasons), "advantage_mean_variance": paired_variance}


def audit_fit(arm, seed, training, validation, info):
    saved = torch.load(info["path"], weights_only=True)
    rebuilt = p.fit_model(arm, seed, training, validation)
    for key in ("arm", "seed", "width", "selected_update", "history", "parameters"):
        assert rebuilt[key] == saved[key], (arm, seed, key)
    for key in ("selected_state", "final_state"):
        assert saved[key].keys() == rebuilt[key].keys()
        for name in rebuilt[key]:
            assert torch.equal(saved[key][name], rebuilt[key][name]), (arm, seed, key, name)
    return arm, seed


def independent_stats(q, y, mask, predictions, saved):
    n = len(q)
    rows = []
    for i in range(n):
        contrast, after_base, control_base, mse_diff = [], [], [], []
        count = int(mask[i].sum())
        for k in range(3):
            c, a = predictions["control"][k], predictions["afterstate"][k]
            qc, qa, qb = float(q[i, c["choices"][i]]), float(q[i, a["choices"][i]]), float(q[i, 0])
            contrast.append(qa-qc)
            after_base.append(qa-qb)
            control_base.append(qc-qb)
            losses = []
            for model in (a, c):
                losses.append(sum((model["advantages"][i][j]-float(y[i, j]))**2 for j in range(4) if mask[i, j]) / max(count, 1))
            mse_diff.append(losses[0]-losses[1])
        rows.append([np.mean(x) for x in (contrast, after_base, control_base, mse_diff)])
    rows = np.asarray(rows)
    rng = np.random.default_rng(149999)
    replicates = []
    for _ in range(50):
        resampled = rng.integers(0, n, size=(1000, n))
        replicates.append(rows[resampled].mean(axis=1))
    interval = np.quantile(np.concatenate(replicates), [.025, .975], axis=0)
    names = ("afterstate_minus_control", "afterstate_minus_baseline", "control_minus_baseline", "mse_afterstate_minus_control")
    for j, name in enumerate(names):
        assert np.allclose(interval[:, j], saved["contrasts"][name]["ci95"], atol=1e-12, rtol=0)
        assert abs(rows[:, j].mean()-saved["contrasts"][name]["mean"]) < 1e-12
    gates = {names[j]: bool(rows[:, j].mean() >= .02 and interval[0, j] > 0 and (interval[1, j]-interval[0, j])/2 <= .02) for j in (0, 1)}
    gates["mse"] = bool(interval[1, 3] < 0)
    assert gates == saved["statistical_gates"]
    assert all(gates.values()) == saved["statistical_pass"]


def main():
    started = time.perf_counter()
    p.check_identity()
    p.initialize_worker()
    selection = p.read_json(p.OUTPUT / "selection_lock.json")
    choices = p.read_json(p.OUTPUT / "test_choices_lock.json")
    assert choices["selection_sha256"] == p.file_hash(p.OUTPUT / "selection_lock.json")
    assert choices["panel_sha256"] == p.file_hash(p.OUTPUT / "test_panel.json")
    assert choices["features_sha256"] == p.file_hash(p.OUTPUT / "test_features.pt")
    assert selection["development_lock"] == p.file_hash(p.OUTPUT / "development_lock.json")
    devlock = p.read_json(p.OUTPUT / "development_lock.json")
    panels, data = {}, {}
    source_ids = set()
    for split in ("train", "validation", "test"):
        panel_path = p.OUTPUT / f"{split}_panel.json"
        panels[split] = p.read_json(panel_path)
        rebuilt = p.build_panel(split, audit=True)
        assert p.digest(rebuilt) == p.digest(panels[split])
        for sample in rebuilt["samples"]:
            assert sample["source_seed"] not in source_ids
            source_ids.add(sample["source_seed"])
            if split == "test":
                src = p.OUTPUT / "sources" / "test" / f"game_{sample['source_index']:04d}.json"
                assert src.stat().st_mtime_ns > (p.OUTPUT / "selection_lock.json").stat().st_mtime_ns
        rebuilt_features = p.feature_data(rebuilt)
        data[split] = torch.load(p.OUTPUT / f"{split}_features.pt", weights_only=True)
        for key in rebuilt_features:
            assert torch.equal(rebuilt_features[key], data[split][key]), (split, key)
        q, data[split]["y"] = p.target_data(rebuilt)
        if split != "test":
            assert p.file_hash(panel_path) == devlock[split]["panel"]
            assert p.file_hash(p.OUTPUT / f"{split}_features.pt") == devlock[split]["features"]
            assert p.file_hash(p.OUTPUT / f"{split}_targets_lock.json") == selection["targets"][split]
        else:
            for path in (p.OUTPUT / "targets" / "test").glob("*.json"):
                assert path.stat().st_mtime_ns > (p.OUTPUT / "test_choices_lock.json").stat().st_mtime_ns
    assert p.test_predictions(data["test"], selection) == choices["predictions"]
    independent_stats(q, data["test"]["y"], data["test"]["mask"], choices["predictions"], p.read_json(p.OUTPUT / "result.json"))
    print({"audit": "all_sources_features_locks_predictions_statistics_passed"}, flush=True)
    count, moves, types, reasons, variance = 0, 0, Counter(), Counter(), []
    with ProcessPoolExecutor(max_workers=8, initializer=p.initialize_worker) as pool:
        jobs = [pool.submit(audit_sample, split, sample) for split, panel in panels.items() for sample in panel["samples"]]
        for i, job in enumerate(as_completed(jobs), 1):
            row = job.result()
            count += row["continuations"]
            moves += row["moves"]
            types.update(row["root_action_types"])
            reasons.update(row["termination_counts"])
            variance.extend(row["advantage_mean_variance"])
            if i % 8 == 0 or i == len(jobs):
                print({"audit": "continuations", "states": i, "total": len(jobs)}, flush=True)
    with ProcessPoolExecutor(max_workers=6) as pool:
        jobs = [pool.submit(audit_fit, arm, seed, data["train"], data["validation"], selection["models"][f"{arm}_{seed}"])
                for seed in p.CONFIG["training_seeds"] for arm in p.ARMS]
        for job in as_completed(jobs):
            print({"audit": "exact_full_fit_reproduced", "model": job.result()}, flush=True)
    p.check_identity()
    p.save_new(p.OUTPUT / "audit.json", {"passed": True, "seconds": time.perf_counter()-started,
                "source_states": len(source_ids), "continuations": count, "moves": moves,
                "root_action_types": dict(types), "termination_counts": dict(reasons), "complete_fits_reproduced": 6,
                "mean_action_advantage_mc_variance": float(np.mean(variance)),
                "result_sha256": p.file_hash(p.OUTPUT / "result.json")})


if __name__ == "__main__":
    main()
