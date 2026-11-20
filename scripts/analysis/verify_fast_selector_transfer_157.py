"""157 independent export, complete-fit and complete-arena verifier.

Shares engine, serialization, canonical keys, observer encoders, Torch model,
cross_entropy/AdamW and frozen154 candidate reconstruction. Split, batching,
padding, membership/loss construction, selection, game loop and statistics are
separately implemented here. No teacher continuations are rerun: all source
outcomes are hash-bound to the already completed155/156 independent audits.
"""
from collections import Counter, defaultdict
from pathlib import Path
from functools import lru_cache
import argparse
import importlib.util
import math
import random
import time

import numpy as np
import torch
from torch.nn import functional as F
from torch.nn.utils.rnn import pad_sequence

from splendor_ai.alphazero import run_fast_selector_transfer as p
from splendor_ai.encoding import encode_public_observation_tensor, encode_legal_action_features, fit_feature_vector
from splendor_ai.encoding.public_belief import canonical_public_belief_key
from splendor_ai.engine.state import RESERVE_SOURCE_VISIBLE

f, old, OUTPUT = p.f, p.old, p.OUTPUT
spec154 = importlib.util.spec_from_file_location("audit154_for157", "scripts/analysis/verify_rollout_selector_154.py")
a154 = importlib.util.module_from_spec(spec154)
spec154.loader.exec_module(a154)


def compare(a, b):
    if isinstance(a, dict):
        assert isinstance(b, dict) and a.keys() == b.keys()
        for k in a:
            compare(a[k], b[k])
    elif isinstance(a, (list, tuple)):
        assert len(a) == len(b)
        for x, y in zip(a, b):
            compare(x, y)
    elif isinstance(a, float):
        assert math.isfinite(a) and math.isfinite(b) and abs(a-b) <= 1e-12, (a, b)
    else:
        assert a == b, (a, b)


@lru_cache(maxsize=1)
def independent_split():
    result = {}
    for exp in (155, 156):
        candidates = range(200) if exp == 155 else range(1, 200)
        order = sorted(candidates, key=lambda pair: old.digest(["157-pair-split-v1", exp, pair]))
        for pair in range(200):
            result[f"{exp}:{pair}"] = "validation" if pair in order[:40] else "train"
    return result


def hidden_alternative(state):
    alternative = state.copy_shallow()
    for deck in alternative.hidden_tier_decks.values():
        deck.reverse()
    opponent = alternative.players[1-state.current_player]
    for index, card in enumerate(opponent.reserved_cards):
        deck = alternative.hidden_tier_decks[card.tier]
        if opponent.reserved_card_source(index) != RESERVE_SOURCE_VISIBLE and deck:
            opponent.reserved_cards[index], deck[0] = deck[0], card
    return alternative


def independent_inputs(state, config):
    legal = old.SplendorEnv().legal_actions(state)
    obs = torch.tensor(fit_feature_vector(encode_public_observation_tensor(state, state.current_player).vector,
                                         config.observation_size), dtype=torch.float32)
    features = torch.tensor([fit_feature_vector(a.vector, config.action_feature_size) for a in
                             encode_legal_action_features(state=state, legal_actions=legal, player_id=state.current_player)], dtype=torch.float32)
    keys = list(old.actions_by_canonical_successor_key(state, legal))
    groups = [keys.index(old.canonical_successor_action_key(state, a)) for a in legal]
    return obs, features, groups, keys, legal


def input_invariance(state, config):
    original = independent_inputs(state, config)
    changed_state = hidden_alternative(state)
    changed = independent_inputs(changed_state, config)
    assert torch.equal(original[0], changed[0]) and torch.equal(original[1], changed[1])
    assert original[2:] == changed[2:]
    assert canonical_public_belief_key(state, state.current_player) == canonical_public_belief_key(changed_state, state.current_player)
    return original


def source_game(spec, export_item):
    module = f.SOURCES[spec["experiment"]]
    path = Path(spec["path"])
    assert old.file_hash(path) == spec["sha256"]
    game = old.read_json(path)
    assert game["manifest"] == spec["parent_binding"] and "failure" not in game
    assert old.file_hash(Path(export_item["path"])) == export_item["sha256"]
    exported = torch.load(export_item["path"], weights_only=True, map_location="cpu")
    env = old.SplendorEnv(seed=game["spec"]["environment_seed"])
    state = env.initial_state()
    config = old._RAW._model.config
    by_turn = {s["turn"]: s for s in spec["roots"]}
    count = 0
    coverage = Counter()
    for move in game["trace"]:
        assert move["turn"] == state.turn_index and move["actor"] == state.current_player
        assert move["state_sha256"] == old.digest(old.serialize_state_snapshot(state))
        if state.current_player == game["spec"]["candidate_seat"]:
            source = module.root_path("main", spec["index"], state.turn_index)
            assert old.file_hash(source) == by_turn[state.turn_index]["sha256"] == move["root_sha256"]
            record = old.read_json(source)
            assert record["manifest"] == spec["parent_binding"] and "failure" not in record
            assert old.digest(record["snapshot"]) == move["state_sha256"]
            belief = old.digest(canonical_public_belief_key(state, state.current_player))
            nonce = old.digest(["155-candidate-v1", game["spec"]["search_stream"], belief])
            assert record["schedule"]["candidate_nonce"] == nonce
            candidates = a154.independent_candidates(state, "155-live-v1", nonce)
            assert candidates == record["candidates"]
            assert a154.independent_candidates(hidden_alternative(state), "155-live-v1", nonce) == candidates
            obs, features, groups, keys, legal = input_invariance(state, config)
            row = exported[count]
            assert set(row) == {"observation", "features", "groups", "group_count", "members", "id", "pair", "split", "teacher", "seat", "phase", "privacy"}
            assert torch.equal(row["observation"], obs) and torch.equal(row["features"], features)
            assert row["groups"].tolist() == groups and row["group_count"] == len(keys)
            members = [[i for i, a in enumerate(legal) if repr(old.canonical_successor_action_key(state, a)) == key]
                       for key in candidates["keys"]]
            assert [m.tolist() for m in row["members"]] == members and all(members)
            scores = old._RAW.score_actions(state, legal)
            compare(list(scores), record["scores"])
            with torch.no_grad():
                raw_logits, _ = old._RAW._model(obs[None], features[None], torch.ones((1, len(legal)), dtype=torch.bool))
            assert raw_logits[0].tolist() == list(scores)
            maxima = [max(scores[i] for i in member) for member in members]
            assert max(range(len(maxima)), key=lambda i: maxima[i]) == 0
            assert candidates["actions"][0] == old.serialize_action(legal[max(range(len(legal)), key=lambda i: scores[i])])
            assert len(record["worlds"]) == 32
            assert all(len(w["outcomes"]) == len(members) for w in record["worlds"])
            assert all(o["wdl"] in (-1., 0., 1.) and o["reason"] in old.VALID_NATURAL_TERMINATIONS
                       for w in record["worlds"] for o in w["outcomes"])
            means = [sum(w["outcomes"][i]["wdl"] for w in record["worlds"])/32 for i in range(len(members))]
            compare(means, record["means"])
            selected = max(range(len(means)), key=lambda i: means[i])
            assert selected == row["teacher"] == record["choice"]
            assert candidates["actions"][selected] == move["action"]
            assert row["id"] == f"{spec['experiment']}:{spec['index']}:{state.turn_index}"
            assert row["pair"] == spec["pair"] == f"{spec['experiment']}:{spec['index']//2}"
            assert row["split"] == independent_split()[spec["pair"]]
            assert row["seat"] == state.current_player and row["phase"] == (0 if state.turn_index < 20 else 1 if state.turn_index < 40 else 2)
            privacy = {"own": sum(p.reserved_card_source(i) != RESERVE_SOURCE_VISIBLE for i in range(len(p.reserved_cards)))
                       for p in [state.players[state.current_player]]}
            privacy["opponent"] = sum(state.players[1-state.current_player].reserved_card_source(i) != RESERVE_SOURCE_VISIBLE
                                     for i in range(len(state.players[1-state.current_player].reserved_cards)))
            assert row["privacy"] == privacy == record["privacy"]
            coverage.update({"rows": 1, row["split"]: 1, f"support_{len(members)}": 1,
                             f"seat_{row['seat']}": 1, f"phase_{row['phase']}": 1,
                             "own_private": int(privacy["own"] > 0), "opponent_private": int(privacy["opponent"] > 0)})
            count += 1
        action = old.decode_action(move["action"])
        assert action in env.legal_actions(state)
        state = env.step(state, action)
    assert count == len(exported) == len(spec["roots"]) == export_item["rows"]
    assert old.digest(old.serialize_state_snapshot(state)) == old.digest(game["final_state"])
    assert game["reason"] in old.VALID_NATURAL_TERMINATIONS
    return {"experiment": spec["experiment"], "index": spec["index"], "coverage": dict(coverage), "export_sha256": export_item["sha256"]}


def audit_export():
    binding = p.validate_freeze()
    specs = old.read_json(OUTPUT/"sources.json")
    assert specs == f.source_inventory()
    assert old.read_json(OUTPUT/"splits.json") == independent_split()
    lock = old.read_json(OUTPUT/"export_lock.json")
    assert lock["manifest"] == binding
    by_name = {Path(s["path"]).name: s for s in lock["files"]}
    reports = p.run_jobs("independent_export_replay", source_game,
                        [(s, by_name[f"{s['experiment']}_{s['index']:04d}.pt"]) for s in specs])
    coverage = Counter()
    for report in reports:
        coverage.update(report["coverage"])
    assert coverage["rows"] == 24342 and len(reports) == 800
    old.save_new(OUTPUT/"export_audit.json", {"manifest": binding, "passed": True, "coverage": dict(coverage),
                 "every_root_hidden_invariance": True, "every_root_initial035_equivalence": True,
                 "files": sorted(reports, key=lambda r: (r["experiment"], r["index"])),
                 "export_lock_sha256": old.file_hash(OUTPUT/"export_lock.json")})


def audit_batches(rows, seed, epoch):
    indices = [i for i, _ in enumerate(rows)]
    generator = random.Random("157-batches-v1|%s|%s" % (seed, epoch))
    generator.shuffle(indices)
    ordered = sorted(indices, key=lambda i: rows[i]["features"].size(0))
    parts = [ordered[start:start+64] for start in range(0, len(ordered), 64)]
    generator.shuffle(parts)
    return parts


def audit_batch_tensors(rows):
    observations = torch.stack([r["observation"] for r in rows])
    actions = pad_sequence([r["features"] for r in rows], batch_first=True, padding_value=0.)
    mask = torch.arange(actions.shape[1])[None] < torch.tensor([len(r["features"]) for r in rows])[:, None]
    return observations, actions, mask


def audit_objective(logits, rows, arm):
    losses, predictions = [], []
    for i, row in enumerate(rows):
        group_ids = [int(row["groups"][member[0]]) for member in row["members"]]
        values = torch.stack([torch.amax(logits[i, torch.where(row["groups"] == group)[0]]) for group in group_ids])
        target = row["teacher"] if arm == "treatment" else 0
        losses.append(F.cross_entropy(values.unsqueeze(0), torch.tensor([target])))
        predictions.append(int(values.detach().argmax()))
    return torch.stack(losses), predictions


def audit_validation(model, rows, arm):
    model.eval()
    detail = []
    with torch.no_grad():
        for batch in audit_batches(rows, 157000, -1):
            examples = [rows[i] for i in batch]
            logits, _ = model(*audit_batch_tensors(examples))
            losses, predictions = audit_objective(logits, examples, arm)
            for r, loss, prediction in zip(examples, losses.tolist(), predictions):
                assert math.isfinite(loss)
                detail.append({"id": r["id"], "pair": r["pair"], "loss": loss, "choice": prediction,
                               "teacher": r["teacher"], "seat": r["seat"], "phase": r["phase"],
                               "privacy": r["privacy"], "support": len(r["members"])})
    detail.sort(key=lambda r: r["id"])
    groups = sorted(set(r["pair"] for r in detail))
    means = []
    for group in groups:
        values = [r["loss"] for r in detail if r["pair"] == group]
        means.append(sum(values)/len(values))
    return {"pair_loss": sum(means)/len(means), "rows": detail}


def replay_fit(arm, seed):
    all_rows = f.load_data()  # Cache already independently rebuilt and hash-checked.
    training = [r for r in all_rows if independent_split()[r["pair"]] == "train"]
    validation = [r for r in all_rows if independent_split()[r["pair"]] == "validation"]
    folder = OUTPUT/"fits"/f"{arm}_{seed}"
    expected = old.read_json(folder/"result.json")
    model = f.StrongPolicyInference(old.CHECKPOINT).model
    initial_head = {n: t.clone() for n, t in model.state_dict().items() if n.startswith("value_head.")}
    for name, parameter in model.named_parameters():
        parameter.requires_grad_(not name.startswith("value_head."))
    parameters = [v for n, v in model.named_parameters() if not n.startswith("value_head.")]
    optimizer = torch.optim.AdamW(parameters, lr=.00001, weight_decay=.0001)
    steps = (len(training)+63)//64
    points = sorted({v for v in (0, 1, 2, 4, 8, 16, 32, 64) if v <= 4*steps} | {e*steps+math.ceil(q*steps/4) for e in range(4) for q in (1, 2, 3, 4)})
    assert [r["update"] for r in expected["curves"]] == points
    update, best, selected = 0, math.inf, None
    observed, sequence = [], []
    started = time.perf_counter()
    def inspect():
        nonlocal best, selected
        curve = expected["curves"][points.index(update)]
        path = folder/f"checkpoint_{update:04d}.pt"
        assert old.file_hash(path) == curve["checkpoint_sha256"]
        payload = torch.load(path, map_location="cpu", weights_only=True)
        assert payload["generation"] == update and payload["metadata"] == {"experiment": 157, "arm": arm, "seed": seed}
        assert payload["model_state_dict"].keys() == model.state_dict().keys()
        for name, tensor in model.state_dict().items():
            assert torch.isfinite(tensor).all() and torch.equal(tensor, payload["model_state_dict"][name]), (arm, seed, update, name)
            if name in initial_head:
                assert torch.equal(tensor, initial_head[name])
        report = audit_validation(model, validation, arm)
        val_path = folder/f"validation_{update:04d}.json"
        assert old.file_hash(val_path) == curve["validation_sha256"]
        assert report == old.read_json(val_path), (arm, seed, update, "validation mismatch")
        assert report["pair_loss"] == curve["pair_loss"]
        assert curve["summary"] == f.prediction_summary(report["rows"], arm)
        if report["pair_loss"] < best:
            best, selected = report["pair_loss"], update
        observed.append({"update": update, "pair_loss": report["pair_loss"], "all_tensors_identical": True})
        old.atomic_write(folder/"audit_progress.json", {"time": old.now(), "update": update, "total_updates": 4*steps,
                         "elapsed_seconds": time.perf_counter()-started})
    inspect()
    for epoch in range(4):
        for indices in audit_batches(training, seed, epoch):
            rows = [training[i] for i in indices]
            sequence.append([r["id"] for r in rows])
            optimizer.zero_grad(set_to_none=True)
            model.train()
            logits, _ = model(*audit_batch_tensors(rows))
            loss = audit_objective(logits, rows, arm)[0].mean()
            assert torch.isfinite(loss)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(parameters, 1., error_if_nonfinite=True)
            optimizer.step()
            assert all(torch.isfinite(v).all() for v in model.state_dict().values())
            update += 1
            if update in points:
                inspect()
    assert update == expected["updates"] == 4*steps and selected == expected["selected_update"]
    assert old.digest(sequence) == expected["batch_sequence_sha256"]
    assert len(training) == expected["train_rows"] and len(validation) == expected["validation_rows"]
    result = {"arm": arm, "seed": seed, "selected": selected, "curves": observed,
              "complete_fit_identical": True, "batch_sequence_sha256": old.digest(sequence), "elapsed_seconds": time.perf_counter()-started}
    old.save_new(folder/"audit.json", result)
    return {"arm": arm, "seed": seed, "sha256": old.file_hash(folder/"audit.json")}


def audit_fits():
    binding = p.validate_freeze()
    lock = old.read_json(OUTPUT/"fits_lock.json")
    assert lock["manifest"] == binding
    assert lock["export_lock_sha256"] == old.file_hash(OUTPUT/"export_lock.json")
    assert lock["export_audit_sha256"] == old.file_hash(OUTPUT/"export_audit.json")
    for info in lock["files"]:
        assert old.file_hash(Path(info["path"])) == info["sha256"]
    reports = p.run_jobs("independent_complete_fits", replay_fit, [(arm, seed) for arm in f.ARMS for seed in f.SEEDS], workers=3)
    for seed in f.SEEDS:
        a, b = [old.read_json(OUTPUT/"fits"/f"{arm}_{seed}"/"result.json") for arm in f.ARMS]
        assert a["batch_sequence_sha256"] == b["batch_sequence_sha256"]
    old.save_new(OUTPUT/"fits_audit.json", {"manifest": binding, "passed": True,
                 "six_complete_fits_bitwise_identical": True, "files": sorted(reports, key=lambda r: (r["arm"], r["seed"])),
                 "fits_lock_sha256": old.file_hash(OUTPUT/"fits_lock.json")})


def independent_spec(stage, index):
    pair, within = divmod(index, 12)
    assert 0 <= pair < (4 if stage == "probe" else 400)
    arm, within_arm = divmod(within, 6)
    seed, seat = divmod(within_arm, 2)
    return {"index": index, "pair": pair, "seat": seat, "arm": ("treatment", "control")[arm],
            "seed": (157001, 157002, 157003)[seed], "environment_seed": (1579000000 if stage == "probe" else 1570000000)+pair,
            "stream": old.digest(["157-proposer-stream-v1", stage, pair, seat])}


def independent_decision(state, stream, model):
    belief = old.digest(canonical_public_belief_key(state, state.current_player))
    nonce = old.digest(["155-candidate-v1", stream, belief])
    candidates = a154.independent_candidates(state, "155-live-v1", nonce)
    obs, features, groups, keys, legal = input_invariance(state, model.config)
    with torch.no_grad():
        logits, _ = model(obs[None], features[None], torch.ones((1, len(legal)), dtype=torch.bool))
    assert torch.isfinite(logits).all()
    assert a154.independent_candidates(hidden_alternative(state), "155-live-v1", nonce) == candidates
    values = logits[0].tolist()
    maxima = [max(values[i] for i, group in enumerate(groups) if repr(keys[group]) == key) for key in candidates["keys"]]
    selected = max(range(len(maxima)), key=lambda i: maxima[i])
    players = [state.players[state.current_player], state.players[1-state.current_player]]
    counts = [sum(player.reserved_card_source(i) != RESERVE_SOURCE_VISIBLE for i in range(len(player.reserved_cards))) for player in players]
    evidence = {"candidate_nonce": nonce, "candidates": candidates, "logits": values, "offered_scores": maxima,
                "choice": selected, "privacy": dict(zip(("own", "opponent"), counts))}
    return old.decode_action(candidates["actions"][selected]), evidence


def replay_game(stage, index, sha):
    spec = independent_spec(stage, index)
    path = p.game_path(stage, index)
    assert old.file_hash(path) == sha
    assert path.stat().st_mtime_ns >= (OUTPUT/f"{stage}_launch.json").stat().st_mtime_ns
    game = old.read_json(path)
    binding = old.digest(old.read_json(OUTPUT/"manifest.json"))
    assert game["manifest"] == binding and game["spec"] == spec and "failure" not in game
    env = old.SplendorEnv(seed=spec["environment_seed"])
    state = env.initial_state()
    model = p._MODELS[f"{spec['arm']}_{spec['seed']}"]
    seen, passes, idle = Counter(), 0, 0
    moves = []
    for offset, row in enumerate(game["trace"]):
        assert not state.terminal and state.turn_index < 400 and passes < 2
        seen[old.state_signature(state)] += 1
        assert seen[old.state_signature(state)] < 4
        assert row["turn"] == state.turn_index and row["actor"] == state.current_player
        assert row["state_sha256"] == old.digest(old.serialize_state_snapshot(state))
        legal = env.legal_actions(state)
        assert legal and row["legal_support"] == len(legal)
        assert row["phase"] == (0 if state.turn_index < 20 else 1 if state.turn_index < 40 else 2)
        move = {"arm": spec["arm"], "seed": spec["seed"], "index": index, "turn": state.turn_index,
                "phase": row["phase"], "timing": row["timing"]}
        if state.current_player == spec["seat"]:
            action, evidence = independent_decision(state, spec["stream"], model)
            compare(evidence, row["evidence"])
            move.update(role="controller", support=len(evidence["candidates"]["actions"]), choice=evidence["choice"], privacy=evidence["privacy"])
        else:
            action = old._RAW.choose_action(env, state, legal)
            assert "evidence" not in row
            move["role"] = "raw"
        assert action in legal and old.serialize_action(action) == row["action"]
        assert set(row["timing"]) == {"wall", "cpu"} and all(math.isfinite(v) and v >= 0 for v in row["timing"].values())
        nxt = env.step(state, action)
        passes = passes+1 if old.is_forced_pass_turn(legal) else 0
        idle = 0 if old.is_progress_transition(state, nxt, action) else idle+1
        assert passes >= 2 or idle < 60
        state = nxt
        moves.append(move)
    assert old.digest(old.serialize_state_snapshot(state)) == old.digest(game["final_state"])
    reason = "forced_pass_deadlock" if passes >= 2 else "completed"
    assert reason == game["reason"] and (state.terminal or passes >= 2)
    winner = old.adjudicate_scores(state) if passes >= 2 else state.winner
    assert winner == game["winner"]
    assert game["elapsed_seconds"] >= sum(row["timing"]["wall"] for row in game["trace"])
    seat = spec["seat"]
    observation = {**spec, "score": .5 if winner is None else float(winner == seat),
                   "margin": state.players[seat].score-state.players[1-seat].score,
                   "reason": reason, "turns": len(moves)}
    report = {"manifest": binding, "game_sha256": sha, "game": observation, "moves": moves}
    target = OUTPUT/stage/"replay"/f"game_{index:04d}.json"
    old.save_new(target, report)
    return {"index": index, "path": target.as_posix(), "sha256": old.file_hash(target)}


def independent_statistics(rows):
    assert len(rows) == 4800
    lookup = {(r["pair"], r["arm"], r["seed"], r["seat"]): r for r in rows}
    assert len(lookup) == 4800
    per_pair = []
    for pair in range(400):
        per_pair.append([sum((lookup[pair, arm, seed, 0]["score"]+lookup[pair, arm, seed, 1]["score"])/2
                             for seed in (157001, 157002, 157003))/3 for arm in ("treatment", "control")])
    matrix = np.asarray(per_pair)
    generator = np.random.default_rng(1573000000)
    resampled = np.empty((50000, 2))
    for start in range(0, 50000, 1000):
        indices = generator.integers(400, size=(1000, 400))
        resampled[start:start+1000] = np.sum(matrix[indices], axis=1)/400
    interval = np.percentile(resampled[:, 0], [2.5, 97.5]).tolist()
    difference_interval = np.percentile(resampled[:, 0]-resampled[:, 1], [2.5, 97.5]).tolist()
    means = [sum(r["score"] for r in rows if r["arm"] == arm)/2400 for arm in ("treatment", "control")]
    difference = (sum(r["score"] for r in rows if r["arm"] == "treatment")-sum(r["score"] for r in rows if r["arm"] == "control"))/2400
    treatment_rows = [r for r in rows if r["arm"] == "treatment"]
    seats = [sum(r["score"] for r in treatment_rows if r["seat"] == seat)/1200 for seat in (0, 1)]
    margin = sum(r["margin"] for r in treatment_rows)/2400
    fits = {}
    for arm in ("treatment", "control"):
        for seed in (157001, 157002, 157003):
            subset = [r for r in rows if r["arm"] == arm and r["seed"] == seed]
            fits[f"{arm}_{seed}"] = {"share": sum(r["score"] for r in subset)/800,
                 "margin": sum(r["margin"] for r in subset)/800,
                 "seat_shares": [sum(r["score"] for r in subset if r["seat"] == seat)/400 for seat in (0, 1)]}
    return {"treatment_share": means[0], "control_share": means[1], "treatment_ci": interval,
            "difference": difference, "difference_ci": difference_interval, "treatment_seats": seats,
            "treatment_margin": margin, "per_fit": fits,
            "practical_pass": means[0] >= .55 and interval[0] > .5 and margin > 0 and min(seats) >= .5,
            "attribution_pass": difference >= .05 and difference_interval[0] > 0}


def check_timings(rows, result):
    def verify_stats(group, reported):
        for kind in ("wall", "cpu"):
            values = sorted(r["timing"][kind] for r in group)
            n = len(values)
            location = (n-1)*.95
            low, high = math.floor(location), math.ceil(location)
            percentile = values[low]+(values[high]-values[low])*(location-low)
            expected = {"count": n, "mean": sum(values)/n, "p95": percentile, "max": values[-1], "sum": sum(values)}
            # Sums of host timings depend on summation order; compare relative 1e-12.
            for key, value in expected.items():
                assert math.isclose(value, reported[kind][key], rel_tol=1e-12, abs_tol=1e-12)
    for arm in f.ARMS:
        for seed in f.SEEDS:
            for role in ("controller", "raw"):
                group = [r for r in rows if (r["arm"], r["seed"], r["role"]) == (arm, seed, role)]
                report = result["per_fit"][f"{arm}_{seed}"][role]
                verify_stats(group, report["all"])
                for phase, entry in report["phase"].items():
                    verify_stats([r for r in group if r["phase"] == int(phase)], entry)
                for support, entry in report["support"].items():
                    verify_stats([r for r in group if r["support"] == int(support)], entry)
    means = {role: sum(result["per_fit"][f"treatment_{seed}"][role]["all"]["wall"]["mean"] for seed in f.SEEDS)/3 for role in ("raw", "controller")}
    assert means["controller"] == result["treatment_mean_wall"] and means["raw"] == result["matched_raw_mean_wall"]
    assert result["ratio"] == means["controller"]/means["raw"]
    assert result["passed"] == (means["controller"] < .1 and means["controller"] <= 10*means["raw"])


def audit_arena(stage):
    started = time.perf_counter()
    binding = p.validate_freeze()
    selected = old.read_json(OUTPUT/"selected_lock.json")
    assert selected["manifest"] == binding and len(selected["selected"]) == 6
    assert selected["fits_lock_sha256"] == old.file_hash(OUTPUT/"fits_lock.json")
    assert selected["fits_audit_sha256"] == old.file_hash(OUTPUT/"fits_audit.json")
    for key, item in selected["selected"].items():
        assert old.file_hash(Path(item["path"])) == item["sha256"]
        fit = old.read_json(OUTPUT/"fits"/key/"result.json")
        expected = min(fit["curves"], key=lambda r: (r["pair_loss"], r["update"]))
        assert item["update"] == expected["update"] and item["sha256"] == expected["checkpoint_sha256"]
    launch = old.read_json(OUTPUT/f"{stage}_launch.json")
    assert launch["manifest"] == binding and launch["selected_lock_sha256"] == old.file_hash(OUTPUT/"selected_lock.json")
    assert launch["schedules_sha256"] == old.file_hash(OUTPUT/"schedules.json")
    assert (OUTPUT/"selected_lock.json").stat().st_mtime_ns <= (OUTPUT/f"{stage}_launch.json").stat().st_mtime_ns
    lock = old.read_json(OUTPUT/f"{stage}_games_lock.json")
    assert lock["manifest"] == binding and lock["launch_sha256"] == old.file_hash(OUTPUT/f"{stage}_launch.json")
    n = 48 if stage == "probe" else 4800
    assert [r["index"] for r in lock["files"]] == list(range(n))
    assert p.schedule(stage) == [independent_spec(stage, i) for i in range(n)]
    files = p.run_jobs(f"independent_{stage}_replay", replay_game,
                       [(stage, r["index"], r["sha256"]) for r in lock["files"]], initializer=p.initialize_games)
    files.sort(key=lambda r: r["index"])
    games, moves = [], []
    for item in files:
        assert old.file_hash(Path(item["path"])) == item["sha256"]
        report = old.read_json(Path(item["path"]))
        games.append(report["game"])
        moves.extend(report["moves"])
    old.save_new(OUTPUT/f"{stage}_observations.json", {"games": games, "moves": moves})
    result = {"manifest": binding, "passed": True, "games": n, "files": files,
              "games_lock_sha256": old.file_hash(OUTPUT/f"{stage}_games_lock.json"),
              "observations_sha256": old.file_hash(OUTPUT/f"{stage}_observations.json"),
              "elapsed_seconds": time.perf_counter()-started}
    if stage == "main":
        stats = p.statistics(games)
        compare(stats, independent_statistics(games))
        timing = p.timings(moves)
        check_timings(moves, timing)
        result["recomputed_result_digest"] = old.digest({"statistics": stats, "timings": timing})
    else:
        result["projected_main_seconds"] = 100*(result["elapsed_seconds"]+lock["elapsed_seconds"])
        result["timings"] = p.timings(moves)
        check_timings(moves, result["timings"])
    assert p.validate_freeze() == binding
    assert launch["selected_lock_sha256"] == old.file_hash(OUTPUT/"selected_lock.json")
    old.save_new(OUTPUT/f"{stage}_audit.json", result)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("stage", choices=("export", "fits", "probe", "main"))
    stage = parser.parse_args().stage
    with old.process_lock(OUTPUT/"verifier.lock"):
        if stage == "export":
            audit_export()
        elif stage == "fits":
            audit_fits()
        else:
            audit_arena(stage)


if __name__ == "__main__":
    main()
