"""Independent162 source/teacher reconstruction, fit replay and fresh-game audit.

Shares frozen engine, encoders, Torch/AdamW, independently written157 decision/
loss primitives and160 world reconstruction. Sampling, replacement batches,
source schedules, teacher checks, fit loop and statistics are independently
implemented. Two prospectively selected worlds/root get complete inner replay.
"""
from collections import Counter
from pathlib import Path
import math
import random
import time

import numpy as np
import torch

from scripts.analysis import verify_teacher_budget_161 as a161
from scripts.analysis import verify_world_budget_160 as a160
from splendor_ai.alphazero import student_refresh_162 as h
from splendor_ai.encoding.public_belief import canonical_public_belief_key

old, a = h.old, a161.a


def independent_split():
    ranked = sorted(range(128), key=lambda i: old.digest(["162-source-split-v1", i]))
    return {**a161.independent_split(), **{f"162:{i}": "validation" if i in ranked[:26] else "train" for i in range(128)}}


def independent_turns(game):
    rows = game["trace"]
    assert len(rows) >= 6 and len({r["turn"] for r in rows}) == len(rows)
    ordered = sorted(rows, key=lambda r: old.digest(["162-position-sample-v1", game["spec"]["index"], r["turn"]]))
    cells, chosen = set(), set()
    for row in ordered:
        cell = (row["actor"], row["phase"])
        assert row["actor"] in (0, 1) and row["phase"] in (0, 1, 2)
        if cell not in cells:
            cells.add(cell)
            chosen.add(row["turn"])
    for row in ordered:
        if len(chosen) == 6:
            break
        chosen.add(row["turn"])
    assert len(chosen) == 6
    return sorted(chosen)


def audit_samples(samples, items, binding):
    assert len(items) == 384 and len(samples) == 2304
    expected = []
    for index, item in enumerate(items):
        assert item["index"] == index and old.file_hash(Path(item["path"])) == item["sha256"]
        game = old.read_json(Path(item["path"]))
        spec = independent_spec("source", index)
        assert game["spec"] == spec and game["manifest"] == binding
        turns = independent_turns(game)
        for turn in turns:
            move = next(r for r in game["trace"] if r["turn"] == turn)
            expected.append({"index": len(expected), "source": item, "source_index": index,
                             "pair": f"162:{spec['pair']}", "seed": spec["seed"], "turn": turn,
                             "seat": move["actor"], "phase": move["phase"], "stream": spec["stream"],
                             "split": independent_split()[f"162:{spec['pair']}"]})
    assert samples == expected
    return True


def independent_probe_indices(specs):
    ordered = sorted(specs, key=lambda s: old.digest(["162-timing-probe-v1", s["index"]]))
    cells, chosen = set(), set()
    for spec in ordered:
        key = (spec["seed"], spec["seat"], spec["phase"])
        if key not in cells:
            cells.add(key)
            chosen.add(spec["index"])
    for spec in ordered:
        if len(chosen) == 18:
            break
        chosen.add(spec["index"])
    assert len(chosen) == 18
    return sorted(chosen)


def independent_plan(training, seed, epoch):
    slots = list(range(len(training)))
    generator = random.Random("162-batches-v1|%s|%s" % (seed, epoch))
    generator.shuffle(slots)
    slots.sort(key=lambda i: len(training[i]["features"]))
    chunks = [slots[k:k+64] for k in range(0, len(slots), 64)]
    generator.shuffle(chunks)
    old_indices = [i for i in range(len(training)) if training[i]["id"].split(":")[0] == "160"]
    assert old_indices and all(r["split"] == "train" for r in training)
    result = []
    for chunk in chunks:
        replay = []
        for i in chunk:
            row = training[i]
            if row["id"].split(":")[0] == "160":
                replay.append(i)
            else:
                assert row["id"].split(":")[0] == "162"
                rng = random.Random("162-old-replacement-v1|%s|%s|%s" % (seed, epoch, row["id"]))
                replay.append(old_indices[rng.randrange(len(old_indices))])
        result.append((chunk, replay))
    return result


def replay_indices(index):
    return sorted(sorted(range(128), key=lambda i: old.digest(["162-audit-world-v1", index, i]))[:2])


def audit_teacher(spec, item, binding, base=h.OUTPUT):
    assert old.file_hash(Path(item["path"])) == item["sha256"]
    receipt = old.read_json(Path(item["path"]))
    assert receipt["manifest"] == binding and receipt["spec"] == spec
    assert old.file_hash(Path(spec["source"]["path"])) == spec["source"]["sha256"]
    game = old.read_json(Path(spec["source"]["path"]))
    assert game["manifest"] == binding and game["spec"]["index"] == spec["source_index"]
    assert spec["turn"] in independent_turns(game)
    move = next(r for r in game["trace"] if r["turn"] == spec["turn"])
    state = old.deserialize_state_snapshot(move["snapshot"])
    assert old.digest(move["snapshot"]) == move["state_sha256"]
    assert spec["seat"] == state.current_player and spec["phase"] == min(state.turn_index//20, 2)
    assert spec["pair"] == f"162:{game['spec']['pair']}" and spec["split"] == independent_split()[spec["pair"]]
    assert spec["stream"] == game["spec"]["stream"] and spec["seed"] == game["spec"]["seed"]
    for field in ("root", "tensor"):
        assert old.file_hash(Path(receipt[field]["path"])) == receipt[field]["sha256"]
    root = old.read_json(Path(receipt["root"]["path"]))
    assert root["manifest"] == binding and "failure" not in root and "continuation_model" not in root
    assert old.digest(root["snapshot"]) == move["state_sha256"]
    schedule = a160.independent_schedule(state, spec["stream"], 128)
    assert root["schedule"] == schedule
    offered = a.a154.independent_candidates(state, "155-live-v1", schedule["candidate_nonce"])
    assert root["candidates"] == move["evidence"]["candidates"] == offered
    alternative = a.hidden_alternative(state)
    assert a160.independent_schedule(alternative, spec["stream"], 128) == schedule
    assert a.a154.independent_candidates(alternative, "155-live-v1", schedule["candidate_nonce"]) == offered
    assert len(root["worlds"]) == 128
    chosen_worlds = replay_indices(spec["index"])
    totals = [0]*len(offered["actions"])
    branches, moves = 0, 0
    for wi, (seed, world_row) in enumerate(zip(schedule["world_seeds"], root["worlds"])):
        assert world_row["seed"] == seed and len(world_row["outcomes"]) == len(totals)
        world = a.a154.independent_world(state, seed)
        world_sha = old.digest(old.serialize_state_snapshot(world))
        assert world_row["state_sha256"] == world_sha
        assert old.digest(old.serialize_state_snapshot(a.a154.independent_world(alternative, seed))) == world_sha
        assert world.players[state.current_player].reserved_cards == state.players[state.current_player].reserved_cards
        legal = old.SplendorEnv().legal_actions(world)
        for j, (payload, outcome) in enumerate(zip(offered["actions"], world_row["outcomes"])):
            action = old.decode_action(payload)
            assert action in legal
            a160._outcome(outcome, state.current_player, state.turn_index)
            if wi in chosen_worlds:
                assert outcome == a160.a158.replay_branch(world, action, state.current_player, spec["stream"], wi)
                branches += 1
            totals[j] += outcome["wdl"]
            moves += outcome["turns"]
    means = [v/128 for v in totals]
    assert root["means"] == means and root["choice"] == means.index(max(means))
    a160._timing(root["timing"])
    row = torch.load(receipt["tensor"]["path"], weights_only=True, map_location="cpu")
    inspect_row(row, state, root, spec, old._RAW._model.config)
    path = base / "teacher" / "audits" / f"root_{spec['index']:04d}.json"
    old.save_new(path, {"manifest": binding, "receipt": item, "passed": True,
                 "replay_world_indices": chosen_worlds, "replayed_branches": branches,
                 "continuations": 128*len(totals), "moves": moves,
                 "coverage": {"seat": state.current_player, "phase": spec["phase"], "support": len(totals),
                              "split": spec["split"], "privacy": row["privacy"]}})
    return h.artifact(path, index=spec["index"])


def inspect_row(row, state, root, spec, config):
    nonce = old.digest(["155-candidate-v1", spec["stream"], old.digest(canonical_public_belief_key(state, state.current_player))])
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
    expected_labels = a161.independent_labels(root)
    private = {}
    for role, player in (("own", state.players[state.current_player]), ("opponent", state.players[1-state.current_player])):
        private[role] = sum(player.reserved_card_source(i) != a.RESERVE_SOURCE_VISIBLE for i in range(len(player.reserved_cards)))
    meta = {"id": f"162:{spec['source_index']}:{state.turn_index}", "pair": spec["pair"],
            "split": independent_split()[spec["pair"]], "teacher": expected_labels["world128"],
            "seat": state.current_player,
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


def replay_fit(arm, seed, rows=None, base=h.OUTPUT):
    if rows is not None:
        assert base.resolve() != h.PRODUCTION_OUTPUT and h.PRODUCTION_OUTPUT not in base.resolve().parents
    else:
        rows = h.load_data(base)  # Inputs already independently reconstructed.
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
    sequence, slots, observed = [], [], []
    started = time.perf_counter()

    def inspect():
        nonlocal best, selected
        curve = expected["curves"][points.index(update)]
        path = folder / f"checkpoint_{update:04d}.pt"
        assert old.file_hash(path) == curve["checkpoint_sha256"]
        payload = torch.load(path, map_location="cpu", weights_only=True)
        assert payload["generation"] == update
        assert payload["metadata"] == {"experiment": 162, "arm": arm, "seed": seed}
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
        for shared, replay in independent_plan(training, seed, epoch):
            indices = shared if arm == "refresh" else replay
            slots.append([training[i]["id"] for i in shared])
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
    assert old.digest(slots) == expected["slot_sequence_sha256"]
    assert len(training) == expected["train_rows"] and len(validation) == expected["validation_rows"]
    result = {"arm": arm, "seed": seed, "selected": selected, "checkpoints": observed,
              "complete_fit_bitwise_identical": True, "result_sha256": old.file_hash(folder / "result.json"),
              "elapsed_seconds": time.perf_counter()-started}
    old.save_new(folder / "audit.json", result)
    return h.artifact(folder / "audit.json", arm=arm, seed=seed)


def independent_spec(stage, index):
    if stage == "source":
        assert 0 <= index < 384
        cluster, fit = divmod(index, 3)
        return {"index": index, "pair": cluster, "seed": 162001+fit, "seat": 0, "comparison": "selfplay",
                "environment_seed": 1621000000+cluster, "stream": old.digest(["162-source-stream-v1", cluster])}
    pair, within = divmod(index, 12)
    assert stage in ("probe", "main") and 0 <= pair < (4 if stage == "probe" else 400)
    comparison, within = divmod(within, 6)
    fit, seat = divmod(within, 2)
    return {"index": index, "pair": pair, "comparison": ("transfer", "progress")[comparison],
            "seed": (162001, 162002, 162003)[fit], "seat": seat,
            "environment_seed": (1629000000 if stage == "probe" else 1620000000)+pair,
            "stream": old.digest(["162-proposer-stream-v1", stage, pair])}


def replay_game(stage, spec, item, binding, base=h.OUTPUT):
    path = Path(item["path"])
    assert old.file_hash(path) == item["sha256"]
    game = old.read_json(path)
    assert game["manifest"] == binding and game["spec"] == spec and "failure" not in game
    if stage != "test":
        assert "test_initial_snapshot" not in game and independent_spec(stage, spec["index"]) == spec
    else:
        assert base.resolve() != h.PRODUCTION_OUTPUT and h.PRODUCTION_OUTPUT not in base.resolve().parents
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
        if spec["comparison"] == "selfplay":
            assert old.digest(row["snapshot"]) == row["state_sha256"]
        legal = env.legal_actions(state)
        assert legal and row["legal_support"] == len(legal)
        assert row["phase"] == min(state.turn_index//20, 2)
        role = "anchor" if spec["comparison"] == "selfplay" else ("refresh" if state.current_player == spec["seat"] else ("replay" if spec["comparison"] == "transfer" else "anchor"))
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
            for seed in (162001, 162002, 162003):
                for seat in (0, 1):
                    sums[pair, j] += int(2*lookup[pair, comparison, seed, seat]["score"])
    rng = np.random.default_rng(1623000000)
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
                              "per_fit": {str(seed): sum(r["score"] for r in group if r["seed"] == seed)/800 for seed in (162001, 162002, 162003)},
                              "pass": interval[0] > .5 and (j == 0 or halfpoints >= 2640)}
    result["both_pass"] = result["transfer"]["pass"] and result["progress"]["pass"]
    return result
