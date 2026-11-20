"""157 frozen data, candidate-restricted classification and cheap inference.

The scorer sees every concrete legal action. Offered group membership is used
only outside the network. No simulator transition constructs a model input.
"""
from __future__ import annotations

from collections import defaultdict
from pathlib import Path
import math
import random
import time

import torch
from torch.nn import functional as F

from splendor_ai.alphazero import run_rollout_selector_champion as parent
from splendor_ai.alphazero.strong_policy import StrongPolicyInference, encode_position, save_model

old = parent.old
OUTPUT = Path("outputs/fast_selector_transfer_157")
ARMS = ("treatment", "control")
SEEDS = (157001, 157002, 157003)
CONFIG = {
    "epochs": 4, "batch_size": 64, "lr": 1e-5, "weight_decay": 1e-4,
    "gradient_norm": 1., "fit_workers": 3, "workers": 8, "torch_threads": 1,
    "seeds": list(SEEDS), "main_pairs": 400, "probe_pairs": 4,
    "environment_start": {"main": 1570000000, "probe": 1579000000},
    "split_tag": "157-pair-split-v1", "batch_tag": "157-batches-v1",
    "stream_tag": "157-proposer-stream-v1", "bootstrap_seed": 1573000000,
    "bootstrap_replicates": 50000, "max_turns": 400,
    "repetition_limit": 4, "no_progress_limit": 60,
    "mean_wall_limit": .1, "relative_wall_limit": 10.,
    "promotion": False, "automatic_successor": False,
}
SOURCES = {155: parent.parent, 156: parent}


def split_membership():
    result = {}
    for experiment in SOURCES:
        available = [i for i in range(200) if (experiment, i) != (156, 0)]
        ranked = sorted(available, key=lambda i: old.digest([CONFIG["split_tag"], experiment, i]))
        validation = set(ranked[:40])
        result.update({f"{experiment}:{i}": "validation" if i in validation else "train" for i in range(200)})
    return result


def source_inventory():
    """Bind every source game and its already certified candidate roots."""
    result = []
    for experiment, module in SOURCES.items():
        binding = module.check_identity()
        audit = old.read_json(module.OUTPUT/"main_audit.json")
        lock_path = module.OUTPUT/"main_games_lock.json"
        assert audit["passed"] and audit["games_lock_sha256"] == old.file_hash(lock_path)
        lock = old.read_json(lock_path)
        assert set(lock["files"]) == {str(i) for i in range(400)}
        expected_schedule = module.schedule("main")
        for index in range(400):
            path = module.game_path("main", index)
            sha = old.file_hash(path)
            assert sha == lock["files"][str(index)]
            replay = module.OUTPUT/"main"/"replay"/f"game_{index:04d}.json"
            assert old.file_hash(replay) == audit["replay_files"][str(index)]
            assert old.read_json(replay)["game_sha256"] == sha
            game = old.read_json(path)
            assert game["manifest"] == binding and game["spec"] == expected_schedule[index]
            roots = [{"turn": move["turn"], "sha256": move["root_sha256"]}
                     for move in game["trace"] if move["actor"] == game["spec"]["candidate_seat"]]
            assert roots and all("root_sha256" in m for m in game["trace"] if m["actor"] == game["spec"]["candidate_seat"])
            result.append({"experiment": experiment, "index": index, "pair": f"{experiment}:{index//2}",
                           "path": path.as_posix(), "sha256": sha, "parent_binding": binding, "roots": roots})
    assert sum(len(g["roots"]) for g in result if g["experiment"] == 155) == 12185
    assert sum(len(g["roots"]) for g in result if g["experiment"] == 156) == 12157
    return result


def shard_path(spec):
    return OUTPUT/"export"/f"{spec['experiment']}_{spec['index']:04d}.pt"


def make_row(state, record, spec, split, config):
    row, keys, actions = encode_position(state, config)
    offered = [old.canonical_successor_action_key(state, old.decode_action(a)) for a in record["candidates"]["actions"]]
    assert len(set(offered)) == len(offered) and 1 <= len(offered) <= 4
    row["members"] = [torch.nonzero(row["groups"] == keys.index(k), as_tuple=True)[0] for k in offered]
    assert all(len(m) for m in row["members"])
    row.update(id=f"{spec['experiment']}:{spec['index']}:{state.turn_index}", pair=spec["pair"],
               split=split, teacher=int(record["choice"]), seat=state.current_player,
               phase=old.phase(state.turn_index), privacy=parent.privacy(state))
    assert 0 <= row["teacher"] < len(offered)
    return row


def export_game(spec, splits):
    model = StrongPolicyInference(old.CHECKPOINT).model
    module = SOURCES[spec["experiment"]]
    game = old.read_json(Path(spec["path"]))
    assert old.file_hash(Path(spec["path"])) == spec["sha256"]
    by_turn = {m["turn"]: m for m in game["trace"]}
    rows = []
    for item in spec["roots"]:
        path = module.root_path("main", spec["index"], item["turn"])
        assert old.file_hash(path) == item["sha256"]
        record = old.read_json(path)
        assert record["manifest"] == spec["parent_binding"] and "failure" not in record
        assert old.digest(record["snapshot"]) == by_turn[item["turn"]]["state_sha256"]
        assert record["candidates"]["actions"][record["choice"]] == by_turn[item["turn"]]["action"]
        state = old.deserialize_state_snapshot(record["snapshot"])
        rows.append(make_row(state, record, spec, splits[spec["pair"]], model.config))
    target = shard_path(spec)
    old.tensor_save(target, rows)
    return {"path": target.as_posix(), "sha256": old.file_hash(target), "rows": len(rows)}


def load_data():
    lock = old.read_json(OUTPUT/"export_lock.json")
    rows = []
    for item in lock["files"]:
        path = Path(item["path"])
        assert old.file_hash(path) == item["sha256"]
        data = torch.load(path, map_location="cpu", weights_only=True)
        assert len(data) == item["rows"]
        rows.extend(data)
    assert len(rows) == 24342 and len({r["id"] for r in rows}) == len(rows)
    return rows


def batches(rows, seed, epoch):
    order = list(range(len(rows)))
    rng = random.Random(f"157-batches-v1|{seed}|{epoch}")
    rng.shuffle(order)
    order.sort(key=lambda i: len(rows[i]["features"]))
    result = [order[i:i+CONFIG["batch_size"]] for i in range(0, len(order), CONFIG["batch_size"])]
    rng.shuffle(result)
    return result


def checkpoint_updates(n):
    steps = math.ceil(n/CONFIG["batch_size"])
    return sorted({v for v in (0, 1, 2, 4, 8, 16, 32, 64) if v <= 4*steps} | {
        epoch*steps+math.ceil(quarter*steps/4) for epoch in range(4) for quarter in range(1, 5)})


def collate(rows):
    count = max(len(r["features"]) for r in rows)
    features = torch.zeros((len(rows), count, rows[0]["features"].shape[1]), dtype=torch.float32)
    mask = torch.zeros((len(rows), count), dtype=torch.bool)
    for i, row in enumerate(rows):
        n = len(row["features"])
        features[i, :n] = row["features"]
        mask[i, :n] = True
    return torch.stack([r["observation"] for r in rows]), features, mask


def restricted_scores(logits, row):
    return torch.stack([torch.amax(logits[m]) for m in row["members"]])


def objective(logits, rows, arm):
    scores = [restricted_scores(logits[i], row) for i, row in enumerate(rows)]
    losses = torch.stack([F.cross_entropy(s[None], torch.tensor([r["teacher"] if arm == "treatment" else 0]))
                          for s, r in zip(scores, rows)])
    return losses, [int(s.detach().argmax()) for s in scores]


def validation(model, rows, arm):
    model.eval()
    records = []
    with torch.no_grad():
        for indices in batches(rows, 157000, -1):
            subset = [rows[i] for i in indices]
            logits, _ = model(*collate(subset))
            losses, choices = objective(logits, subset, arm)
            for row, loss, choice in zip(subset, losses.tolist(), choices):
                assert math.isfinite(loss)
                records.append({"id": row["id"], "pair": row["pair"], "loss": loss, "choice": choice,
                                "teacher": row["teacher"], "seat": row["seat"], "phase": row["phase"],
                                "privacy": row["privacy"], "support": len(row["members"])})
    records.sort(key=lambda r: r["id"])
    pairs = defaultdict(list)
    for r in records:
        pairs[r["pair"]].append(r["loss"])
    loss = sum(sum(pairs[k])/len(pairs[k]) for k in sorted(pairs))/len(pairs)
    return {"pair_loss": loss, "rows": records}


def prediction_summary(records, arm):
    def summarize(rows):
        return {"rows": len(rows), "mean_loss": sum(r["loss"] for r in rows)/len(rows),
                "teacher_agreement": sum(r["choice"] == r["teacher"] for r in rows)/len(rows),
                "own_label_agreement": sum(r["choice"] == (r["teacher"] if arm == "treatment" else 0) for r in rows)/len(rows),
                "movement_from035": sum(r["choice"] != 0 for r in rows)/len(rows)}
    result = {"all": summarize(records), "subgroups": {}}
    for field in ("seat", "phase", "support", "own_private", "opponent_private"):
        get = (lambda r: int(r["privacy"][field.removesuffix("_private")] > 0)) if field.endswith("_private") else (lambda r: r[field])
        result["subgroups"][field] = {str(value): summarize([r for r in records if get(r) == value])
                                     for value in sorted({get(r) for r in records})}
    return result


def trainable_model():
    model = StrongPolicyInference(old.CHECKPOINT).model
    for name, value in model.named_parameters():
        value.requires_grad_(not name.startswith("value_head."))
    return model


def fit_one(arm, seed):
    old.initialize_worker()
    all_rows = load_data()
    training = [r for r in all_rows if r["split"] == "train"]
    valid = [r for r in all_rows if r["split"] == "validation"]
    model = trainable_model()
    parameters = [p for p in model.parameters() if p.requires_grad]
    optimizer = torch.optim.AdamW(parameters, lr=CONFIG["lr"], weight_decay=CONFIG["weight_decay"])
    folder = OUTPUT/"fits"/f"{arm}_{seed}"
    assert not folder.exists()
    folder.mkdir(parents=True)
    points = checkpoint_updates(len(training))
    best, selected, curves, update = math.inf, None, [], 0
    wall = time.perf_counter()
    def observe():
        nonlocal best, selected
        report = validation(model, valid, arm)
        path = folder/f"checkpoint_{update:04d}.pt"
        save_model(path, model, update, {"experiment": 157, "arm": arm, "seed": seed})
        old.save_new(folder/f"validation_{update:04d}.json", report)
        curves.append({"update": update, "pair_loss": report["pair_loss"], "checkpoint_sha256": old.file_hash(path),
                       "validation_sha256": old.file_hash(folder/f"validation_{update:04d}.json"),
                       "summary": prediction_summary(report["rows"], arm)})
        if report["pair_loss"] < best:
            best, selected = report["pair_loss"], update
        old.atomic_write(folder/"progress.json", {"time": old.now(), "update": update, "total_updates": max(points),
                         "elapsed_seconds": time.perf_counter()-wall})
    observe()
    batch_ids = []
    for epoch in range(CONFIG["epochs"]):
        for indices in batches(training, seed, epoch):
            subset = [training[i] for i in indices]
            batch_ids.append([r["id"] for r in subset])
            model.train()
            optimizer.zero_grad(set_to_none=True)
            logits, _ = model(*collate(subset))
            losses, _ = objective(logits, subset, arm)
            loss = losses.mean()
            assert torch.isfinite(loss)
            loss.backward()
            norm = torch.nn.utils.clip_grad_norm_(parameters, CONFIG["gradient_norm"], error_if_nonfinite=True)
            assert torch.isfinite(norm)
            optimizer.step()
            assert all(torch.isfinite(v).all() for v in model.state_dict().values())
            update += 1
            if update in points:
                observe()
    assert update == max(points)
    result = {"arm": arm, "seed": seed, "selected_update": selected, "curves": curves,
              "updates": update, "train_rows": len(training), "validation_rows": len(valid),
              "batch_sequence_sha256": old.digest(batch_ids), "elapsed_seconds": time.perf_counter()-wall}
    old.save_new(folder/"result.json", result)
    return {"arm": arm, "seed": seed, "path": (folder/"result.json").as_posix(), "sha256": old.file_hash(folder/"result.json")}


def choose(state, stream, scorer):
    """One unchanged035 proposal evaluation, one learned complete-set evaluation."""
    nonce = parent.root_schedule(state, stream)["candidate_nonce"]
    candidates = parent.candidate_set(state, nonce)
    legal = old.SplendorEnv().legal_actions(state)
    # Match raw035's no_grad execution path (including attention fastpath).
    with torch.no_grad():
        row, keys, _ = encode_position(state, scorer.config, legal)
        logits, _ = scorer(row["observation"][None], row["features"][None], torch.ones((1, len(legal)), dtype=torch.bool))
    assert torch.isfinite(logits).all()
    key_to_group = {repr(k): i for i, k in enumerate(keys)}
    scores = [float(torch.amax(logits[0][row["groups"] == key_to_group[k]])) for k in candidates["keys"]]
    choice = max(range(len(scores)), key=lambda i: scores[i])
    evidence = {"candidate_nonce": nonce, "candidates": candidates, "logits": logits[0].tolist(),
                "offered_scores": scores, "choice": choice, "privacy": parent.privacy(state)}
    return old.decode_action(candidates["actions"][choice]), evidence
