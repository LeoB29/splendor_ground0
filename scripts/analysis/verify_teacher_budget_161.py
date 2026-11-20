"""Independent 161 tensor/label reconstruction, complete CPU fit and game replay.

Shares frozen engine, encoders, canonical keys, Torch/AdamW and independent157
batch/loss/decision primitives. Does not call161 export, training, game loop or
statistics to derive expected answers. No160 inner continuations are rerun.
"""
from collections import Counter
from pathlib import Path
import math
import time

import numpy as np
import torch

from scripts.analysis.verify_learned_continuation_158 import a157 as a
from splendor_ai.alphazero import teacher_budget_161 as h
from splendor_ai.encoding.public_belief import canonical_public_belief_key

old = h.old


def independent_split():
    order = sorted(range(200), key=lambda p: old.digest(["161-source-pair-split-v1", 160, p]))
    return {f"160:{p}": "validation" if p in order[:40] else "train" for p in range(200)}


def independent_labels(root):
    outcomes = root["worlds"]
    n = len(root["candidates"]["actions"])
    assert len(outcomes) == 128 and 1 <= n <= 4
    totals = {32: [0] * n, 128: [0] * n}
    for wi, world in enumerate(outcomes):
        assert len(world["outcomes"]) == n
        for j, out in enumerate(world["outcomes"]):
            assert out["wdl"] in (-1., 0., 1.) and out["reason"] in old.VALID_NATURAL_TERMINATIONS
            totals[128][j] += int(out["wdl"])
            if wi < 32:
                totals[32][j] += int(out["wdl"])
    choices = {f"world{n}": sums.index(max(sums)) for n, sums in totals.items()}
    assert choices["world128"] == root["choice"]
    assert root["means"] == [v/128 for v in totals[128]]
    return choices


def inspect_row(row, state, root, spec, stream, config):
    nonce = old.digest(["155-candidate-v1", stream, old.digest(canonical_public_belief_key(state, state.current_player))])
    assert root["schedule"]["candidate_nonce"] == nonce
    candidates = a.a154.independent_candidates(state, "155-live-v1", nonce)
    assert candidates == root["candidates"]
    assert a.a154.independent_candidates(a.hidden_alternative(state), "155-live-v1", nonce) == candidates
    obs, features, groups, keys, legal = a.input_invariance(state, config)
    members = [[i for i, action in enumerate(legal) if repr(old.canonical_successor_action_key(state, action)) == key]
               for key in candidates["keys"]]
    assert all(members)
    assert torch.equal(row["observation"], obs) and torch.equal(row["features"], features)
    assert row["groups"].tolist() == groups and row["group_count"] == len(keys)
    assert [m.tolist() for m in row["members"]] == members
    expected_labels = independent_labels(root)
    private = {}
    for role, player in (("own", state.players[state.current_player]), ("opponent", state.players[1-state.current_player])):
        private[role] = sum(player.reserved_card_source(i) != a.RESERVE_SOURCE_VISIBLE for i in range(len(player.reserved_cards)))
    meta = {"id": f"160:{spec['index']}:{state.turn_index}", "pair": f"160:{spec['index']//2}",
            "split": independent_split()[spec["pair"]], "teacher": expected_labels["world128"],
            "labels": expected_labels, "seat": state.current_player,
            "phase": min(state.turn_index//20, 2), "privacy": private}
    assert private == root["privacy"]
    tensors = {"observation", "features", "groups", "group_count", "members"}
    assert set(row) == set(meta) | tensors
    assert {k: row[k] for k in meta} == meta
    scores = list(old._RAW.score_actions(state, legal))
    a.compare(scores, root["scores"])
    assert candidates["actions"][0] == old.serialize_action(legal[scores.index(max(scores))])
    return {"rows": 1, row["split"]: 1, "labels_changed": int(expected_labels["world32"] != expected_labels["world128"]),
            f"support_{len(members)}": 1, f"phase_{row['phase']}": 1,
            "own_private": int(private["own"] > 0), "opponent_private": int(private["opponent"] > 0)}


def source_game(spec, item, base=h.OUTPUT):
    assert old.file_hash(Path(spec["path"])) == spec["sha256"]
    assert old.file_hash(Path(item["path"])) == item["sha256"]
    exported = torch.load(item["path"], weights_only=True, map_location="cpu")
    game = old.read_json(Path(spec["path"]))
    assert game["manifest"] == spec["parent_binding"] and "failure" not in game
    env = old.SplendorEnv(seed=game["spec"]["environment_seed"])
    state = env.initial_state()
    by_turn = {r["turn"]: r for r in spec["roots"]}
    count, coverage = 0, Counter()
    for move in game["trace"]:
        assert not state.terminal
        assert move["turn"] == state.turn_index and move["actor"] == state.current_player
        assert move["state_sha256"] == old.digest(old.serialize_state_snapshot(state))
        if state.current_player == game["spec"]["candidate_seat"]:
            path = h.source.root_path("main", spec["index"], state.turn_index)
            assert old.file_hash(path) == by_turn[state.turn_index]["sha256"] == move["root_sha256"]
            root = old.read_json(path)
            assert root["manifest"] == spec["parent_binding"] and "failure" not in root
            assert old.digest(root["snapshot"]) == move["state_sha256"]
            coverage.update(inspect_row(exported[count], state, root, spec, game["spec"]["search_stream"], old._RAW._model.config))
            assert root["candidates"]["actions"][root["choice"]] == move["action"]
            count += 1
        action = old.decode_action(move["action"])
        assert action in env.legal_actions(state)
        state = env.step(state, action)
    assert count == len(exported) == len(spec["roots"]) == item["rows"]
    assert old.digest(old.serialize_state_snapshot(state)) == old.digest(game["final_state"])
    assert game["reason"] in old.VALID_NATURAL_TERMINATIONS
    path = base / "export_audit" / f"game_{spec['index']:04d}.json"
    old.save_new(path, {"source": spec, "export": item, "coverage": dict(coverage), "passed": True})
    return h.artifact(path, index=spec["index"])


def replay_fit(arm, seed, rows=None, base=h.OUTPUT):
    if rows is not None:
        assert base.resolve() != h.OUTPUT.resolve()
    else:
        rows = h.load_data(arm, base)  # Inputs already independently reconstructed.
        assert all(r["split"] == independent_split()[r["pair"]] for r in rows)
    training = [r for r in rows if r["split"] == "train"]
    validation = [r for r in rows if r["split"] == "validation"]
    folder = base / "fits" / f"{arm}_{seed}"
    expected = old.read_json(folder / "result.json")
    init = h.initializer(seed)
    assert expected["initializer"] == init
    model = h.f.StrongPolicyInference(init["path"]).model
    initial_head = {n: t.clone() for n, t in model.state_dict().items() if n.startswith("value_head.")}
    for name, parameter in model.named_parameters():
        parameter.requires_grad_(not name.startswith("value_head."))
    parameters = [v for n, v in model.named_parameters() if not n.startswith("value_head.")]
    optimizer = torch.optim.AdamW(parameters, lr=.00001, weight_decay=.0001)
    assert not optimizer.state
    steps = (len(training)+63)//64
    points = sorted({v for v in (0, 1, 2, 4, 8, 16, 32, 64) if v <= 4*steps} |
                    {e*steps+math.ceil(q*steps/4) for e in range(4) for q in (1, 2, 3, 4)})
    assert [r["update"] for r in expected["curves"]] == points
    update, best, selected = 0, math.inf, None
    sequence, observed = [], []
    started = time.perf_counter()

    def inspect():
        nonlocal best, selected
        curve = expected["curves"][points.index(update)]
        path = folder / f"checkpoint_{update:04d}.pt"
        assert old.file_hash(path) == curve["checkpoint_sha256"]
        payload = torch.load(path, map_location="cpu", weights_only=True)
        assert payload["generation"] == update
        assert payload["metadata"] == {"experiment": 161, "arm": arm, "seed": seed}
        assert payload["model_state_dict"].keys() == model.state_dict().keys()
        for name, tensor in model.state_dict().items():
            assert torch.isfinite(tensor).all() and torch.equal(tensor, payload["model_state_dict"][name]), (arm, seed, update, name)
            if name in initial_head:
                assert torch.equal(tensor, initial_head[name])
        val_path = folder / f"validation_{update:04d}.json"
        assert old.file_hash(val_path) == curve["validation_sha256"]
        report = a.audit_validation(model, validation, "treatment")
        assert report == old.read_json(val_path) and report["pair_loss"] == curve["pair_loss"]
        if report["pair_loss"] < best:
            best, selected = report["pair_loss"], update
        observed.append(update)
        print({"replay_fit": f"{arm}_{seed}", "update": update, "total_updates": 4*steps}, flush=True)

    inspect()
    for epoch in range(4):
        for indices in a.audit_batches(training, seed, epoch):
            batch = [training[i] for i in indices]
            sequence.append([r["id"] for r in batch])
            optimizer.zero_grad(set_to_none=True)
            model.train()
            logits, _ = model(*a.audit_batch_tensors(batch))
            loss = a.audit_objective(logits, batch, "treatment")[0].mean()
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
    result = {"arm": arm, "seed": seed, "selected": selected, "checkpoints": observed,
              "complete_fit_bitwise_identical": True, "result_sha256": old.file_hash(folder / "result.json"),
              "elapsed_seconds": time.perf_counter()-started}
    old.save_new(folder / "audit.json", result)
    return h.artifact(folder / "audit.json", arm=arm, seed=seed)


def independent_spec(stage, index):
    pair, within = divmod(index, 12)
    assert stage in ("probe", "main") and 0 <= pair < (4 if stage == "probe" else 400)
    comparison, within = divmod(within, 6)
    fit, seat = divmod(within, 2)
    return {"index": index, "pair": pair, "comparison": ("transfer", "progress")[comparison],
            "seed": (161001, 161002, 161003)[fit], "seat": seat,
            "environment_seed": (1619000000 if stage == "probe" else 1610000000)+pair,
            "stream": old.digest(["161-proposer-stream-v1", stage, pair])}


def replay_game(stage, spec, item, binding, base=h.OUTPUT):
    path = Path(item["path"])
    assert old.file_hash(path) == item["sha256"]
    game = old.read_json(path)
    assert game["manifest"] == binding and game["spec"] == spec and "failure" not in game
    if stage != "test":
        assert "test_initial_snapshot" not in game and independent_spec(stage, spec["index"]) == spec
    else:
        assert base.resolve() != h.OUTPUT.resolve()
    env = old.SplendorEnv(seed=spec["environment_seed"])
    state = old.deserialize_state_snapshot(game["test_initial_snapshot"]) if "test_initial_snapshot" in game else env.initial_state()
    seen, passes, idle = Counter(), 0, 0
    timings = []
    for row in game["trace"]:
        assert not state.terminal and state.turn_index < 400 and passes < 2
        seen[old.state_signature(state)] += 1
        assert seen[old.state_signature(state)] < 4
        assert row["turn"] == state.turn_index and row["actor"] == state.current_player
        assert row["state_sha256"] == old.digest(old.serialize_state_snapshot(state))
        legal = env.legal_actions(state)
        assert legal and row["legal_support"] == len(legal)
        assert row["phase"] == min(state.turn_index//20, 2)
        role = "world128" if state.current_player == spec["seat"] else ("world32" if spec["comparison"] == "transfer" else "anchor")
        key = f"{role}_{spec['seed']}"
        assert row["model"] == key
        action, evidence = a.independent_decision(state, spec["stream"], h.MODELS[key])
        a.compare(evidence, row["evidence"])
        assert action in legal and old.serialize_action(action) == row["action"]
        assert set(row["timing"]) == {"wall", "cpu"} and all(math.isfinite(v) and v >= 0 for v in row["timing"].values())
        timings.append({"model": key, "phase": row["phase"], **row["timing"]})
        nxt = env.step(state, action)
        passes = passes+1 if old.is_forced_pass_turn(legal) else 0
        idle = 0 if old.is_progress_transition(state, nxt, action) else idle+1
        assert passes >= 2 or idle < 60
        state = nxt
    assert old.digest(old.serialize_state_snapshot(state)) == old.digest(game["final_state"])
    reason = "forced_pass_deadlock" if passes >= 2 else "completed"
    assert reason == game["reason"] and (state.terminal or passes >= 2)
    winner = old.adjudicate_scores(state) if passes >= 2 else state.winner
    assert winner == game["winner"]
    assert game["elapsed_seconds"] >= sum(row["wall"] for row in timings)
    seat = spec["seat"]
    observation = {**spec, "score": .5 if winner is None else float(winner == seat),
                   "margin": state.players[seat].score-state.players[1-seat].score, "reason": reason, "turns": len(timings)}
    target = base / stage / "audit" / f"game_{spec['index']:04d}.json"
    old.save_new(target, {"manifest": binding, "game_sha256": item["sha256"], "game": observation, "timings": timings})
    return h.artifact(target, index=spec["index"])


def independent_statistics(rows):
    assert len(rows) == 4800
    lookup = {(r["pair"], r["comparison"], r["seed"], r["seat"]): r for r in rows}
    assert len(lookup) == 4800 and all(r["score"] in (0., .5, 1.) for r in rows)
    comparisons = ("transfer", "progress")
    sums = np.zeros((400, 2), dtype=np.int64)
    for pair in range(400):
        for j, comparison in enumerate(comparisons):
            for seed in (161001, 161002, 161003):
                for seat in (0, 1):
                    sums[pair, j] += int(2*lookup[pair, comparison, seed, seat]["score"])
    rng = np.random.default_rng(1613000000)
    resampled = np.concatenate([np.sum(sums[rng.integers(0, 400, (1000, 400))], axis=1) for _ in range(50)]) / 4800
    result = {}
    for j, comparison in enumerate(comparisons):
        group = [r for r in rows if r["comparison"] == comparison]
        halfpoints = sum(int(2*r["score"]) for r in group)
        interval = np.percentile(resampled[:, j], [2.5, 97.5]).tolist()
        result[comparison] = {"score": halfpoints/4800, "ci95": interval, "halfpoints": halfpoints,
                              "wins": len([r for r in group if r["score"] == 1]),
                              "draws": len([r for r in group if r["score"] == .5]),
                              "losses": len([r for r in group if r["score"] == 0]),
                              "margin": sum(r["margin"] for r in group)/2400,
                              "seats": [sum(r["score"] for r in group if r["seat"] == seat)/1200 for seat in (0, 1)],
                              "per_fit": {str(seed): sum(r["score"] for r in group if r["seed"] == seed)/800 for seed in (161001, 161002, 161003)},
                              "pass": interval[0] > .5 and (j == 0 or halfpoints >= 2640)}
    result["both_pass"] = result["transfer"]["pass"] and result["progress"]["pass"]
    return result
