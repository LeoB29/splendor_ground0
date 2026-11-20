"""162: bounded teacher feedback on frozen161 student self-play positions."""
from collections import Counter
from pathlib import Path
import math
import random
import time
import traceback

import numpy as np
import torch

from splendor_ai.alphazero import teacher_budget_161 as previous
from splendor_ai.alphazero import world_budget_160 as teacher

f, old = previous.f, previous.old
OUTPUT = Path("outputs/student_refresh_162")
PRODUCTION_OUTPUT = OUTPUT.resolve()
ARMS = ("refresh", "replay")
SEEDS = (162001, 162002, 162003)
COMPARISONS = ("transfer", "progress")
INITIALIZERS = {
    162001: (161001, 563, "8d83d69b9691569f187beb93f2636635ab419c835442d02f0eb6fc511866fa62"),
    162002: (161002, 488, "b06e37d3640adb846de562daaaae4d526bf586a848a1e1783f2bc4d2c3899c96"),
    162003: (161003, 563, "8d4a258f625003e0fad9241560db0f606c23d1d8fab3a9c1afd3bb0c2f11a66a"),
}
CONFIG = {"epochs": 4, "batch_size": 64, "lr": 1e-5, "weight_decay": 1e-4,
          "gradient_norm": 1., "fit_workers": 3, "workers": 8, "torch_threads": 1,
          "source_clusters": 128, "source_games": 384, "new_rows": 2304,
          "new_validation_clusters": 26, "old_train_rows": 9577, "old_validation_rows": 2408,
          "new_train_rows": 1836, "new_validation_rows": 468, "teacher_probe_roots": 18,
          "worlds": 128, "inner_replay_worlds": 2, "main_pairs": 400, "probe_pairs": 4,
          "bootstrap_seed": 1623000000, "bootstrap_resamples": 50000,
          "source_seed": 1621000000, "main_seed": 1620000000, "probe_seed": 1629000000,
          "source_stream": "162-source-stream-v1", "stream_tag": "162-proposer-stream-v1",
          "split_tag": "162-source-split-v1", "sample_tag": "162-position-sample-v1",
          "batch_tag": "162-batches-v1", "replacement_tag": "162-old-replacement-v1",
          "audit_tag": "162-audit-world-v1", "device": "cpu", "fresh_optimizer": True,
          "automatic_resume": False, "promotion": False, "automatic_successor": False,
          "scheduled_monitoring": False}


def initializer(seed):
    parent_seed, update, sha = INITIALIZERS[seed]
    path = previous.OUTPUT / "fits" / f"world128_{parent_seed}" / f"checkpoint_{update:04d}.pt"
    assert old.file_hash(path) == sha
    return {"path": path.as_posix(), "sha256": sha, "seed": parent_seed, "update": update}


def artifact(path, **values):
    return {"path": Path(path).as_posix(), "sha256": old.file_hash(Path(path)), **values}


def splits():
    order = sorted(range(128), key=lambda i: old.digest([CONFIG["split_tag"], i]))
    valid = set(order[:26])
    return {**previous.splits(), **{f"162:{i}": "validation" if i in valid else "train" for i in range(128)}}


def source_schedule():
    return [{"index": i*3+j, "pair": i, "seed": seed, "seat": 0, "comparison": "selfplay",
             "environment_seed": CONFIG["source_seed"]+i,
             "stream": old.digest([CONFIG["source_stream"], i])}
            for i in range(128) for j, seed in enumerate(SEEDS)]


def sampled_turns(game):
    """Six roots per game: one in each actor/phase cell, identity-only shortage fill."""
    spec = game["spec"]
    moves = game["trace"]
    assert len(moves) >= 6 and len({r["turn"] for r in moves}) == len(moves)
    def rank(row):
        return old.digest([CONFIG["sample_tag"], spec["index"], row["turn"]])
    chosen = []
    for seat in (0, 1):
        for phase in (0, 1, 2):
            cell = [r for r in moves if r["actor"] == seat and r["phase"] == phase]
            if cell:
                chosen.append(min(cell, key=rank)["turn"])
    remainder = sorted((r for r in moves if r["turn"] not in chosen), key=rank)
    chosen.extend(r["turn"] for r in remainder[:6-len(chosen)])
    assert len(set(chosen)) == 6
    return sorted(chosen)


def sample_inventory(items, binding):
    assert len(items) == CONFIG["source_games"]
    result = []
    for item, expected in zip(items, source_schedule()):
        assert item["index"] == expected["index"] and old.file_hash(Path(item["path"])) == item["sha256"]
        game = old.read_json(Path(item["path"]))
        assert game["manifest"] == binding and game["spec"] == expected and "failure" not in game
        for turn in sampled_turns(game):
            move = next(r for r in game["trace"] if r["turn"] == turn)
            result.append({"index": len(result), "source": item, "source_index": expected["index"],
                           "pair": f"162:{expected['pair']}", "seed": expected["seed"], "turn": turn,
                           "seat": move["actor"], "phase": move["phase"], "stream": expected["stream"],
                           "split": splits()[f"162:{expected['pair']}"]})
    assert len(result) == 2304
    return result


def probe_indices(specs):
    selected = []
    for seed in SEEDS:
        for seat in (0, 1):
            for phase in (0, 1, 2):
                cell = [s for s in specs if (s["seed"], s["seat"], s["phase"]) == (seed, seat, phase)]
                if cell:
                    selected.append(min(cell, key=lambda s: old.digest(["162-timing-probe-v1", s["index"]]))["index"])
    remainder = sorted((s for s in specs if s["index"] not in selected),
                       key=lambda s: old.digest(["162-timing-probe-v1", s["index"]]))
    selected.extend(s["index"] for s in remainder[:18-len(selected)])
    return sorted(selected)


def query_teacher(spec, binding, base=OUTPUT):
    assert old.file_hash(Path(spec["source"]["path"])) == spec["source"]["sha256"]
    game = old.read_json(Path(spec["source"]["path"]))
    assert game["manifest"] == binding and game["spec"]["index"] == spec["source_index"]
    move = next(r for r in game["trace"] if r["turn"] == spec["turn"])
    state = old.deserialize_state_snapshot(move["snapshot"])
    assert old.digest(move["snapshot"]) == move["state_sha256"]
    folder = base / "teacher"
    root_path = folder / "roots" / f"root_{spec['index']:04d}.json"
    _, record = teacher.choose_root(state, spec["stream"], root_path, binding, 128)
    assert record["candidates"] == move["evidence"]["candidates"]
    source = {"experiment": 162, "index": spec["source_index"], "pair": spec["pair"]}
    row = f.make_row(state, record, source, spec["split"], old._RAW._model.config)
    tensor_path = folder / "tensors" / f"root_{spec['index']:04d}.pt"
    assert not tensor_path.exists()
    old.tensor_save(tensor_path, row)
    path = folder / "receipts" / f"root_{spec['index']:04d}.json"
    old.save_new(path, {"manifest": binding, "spec": spec, "root": artifact(root_path), "tensor": artifact(tensor_path)})
    return artifact(path, index=spec["index"])


def load_data(base=OUTPUT):
    rows = previous.load_data("world128")
    assert Counter(r["split"] for r in rows) == {"train": 9577, "validation": 2408}
    lock = old.read_json(base / "teacher_lock.json")
    assert len(lock["files"]) == 2304
    for item in lock["files"]:
        assert old.file_hash(Path(item["path"])) == item["sha256"]
        receipt = old.read_json(Path(item["path"]))
        for field in ("root", "tensor"):
            assert old.file_hash(Path(receipt[field]["path"])) == receipt[field]["sha256"]
        row = torch.load(receipt["tensor"]["path"], weights_only=True, map_location="cpu")
        assert row["split"] == receipt["spec"]["split"] == splits()[row["pair"]]
        rows.append(row)
    assert len(rows) == 14289 and len({r["id"] for r in rows}) == len(rows)
    assert Counter(r["split"] for r in rows) == {"train": 11413, "validation": 2876}
    return rows


def batch_plan(training, seed, epoch):
    """Shared slots; replay substitutes an old row only at a fresh-row slot."""
    older = [i for i, r in enumerate(training) if r["id"].startswith("160:")]
    assert older and all(r["split"] == "train" for r in training)
    order = list(range(len(training)))
    rng = random.Random(f"{CONFIG['batch_tag']}|{seed}|{epoch}")
    rng.shuffle(order)
    order.sort(key=lambda i: len(training[i]["features"]))
    chunks = [order[i:i+64] for i in range(0, len(order), 64)]
    rng.shuffle(chunks)
    replacement = {}
    for i, row in enumerate(training):
        if not row["id"].startswith("160:"):
            assert row["id"].startswith("162:")
            replacement[i] = random.Random(f"{CONFIG['replacement_tag']}|{seed}|{epoch}|{row['id']}").choice(older)
    return [(chunk, [replacement.get(i, i) for i in chunk]) for chunk in chunks]


def train_fit(arm, seed, rows=None, base=OUTPUT):
    """Rows override only permits isolated fixtures; production always loads the locked corpus."""
    assert arm in ARMS and seed in SEEDS
    if rows is not None:
        assert base.resolve() != PRODUCTION_OUTPUT and PRODUCTION_OUTPUT not in base.resolve().parents
    else:
        rows = load_data(base)
    model = f.StrongPolicyInference(initializer(seed)["path"]).model
    initial_head = {n: t.clone() for n, t in model.state_dict().items() if n.startswith("value_head.")}
    for name, parameter in model.named_parameters():
        parameter.requires_grad_(not name.startswith("value_head."))
    parameters = [p for p in model.parameters() if p.requires_grad]
    optimizer = torch.optim.AdamW(parameters, lr=CONFIG["lr"], weight_decay=CONFIG["weight_decay"])
    assert not optimizer.state
    training = [r for r in rows if r["split"] == "train"]
    valid = [r for r in rows if r["split"] == "validation"]
    assert training and valid
    folder = base / "fits" / f"{arm}_{seed}"
    folder.mkdir(parents=True, exist_ok=False)
    points = f.checkpoint_updates(len(training))
    best, selected, curves, update = math.inf, None, [], 0
    wall = time.perf_counter()

    def observe():
        nonlocal best, selected
        for name, tensor in initial_head.items():
            assert torch.equal(tensor, model.state_dict()[name])
        report = f.validation(model, valid, "treatment")
        path = folder / f"checkpoint_{update:04d}.pt"
        f.save_model(path, model, update, {"experiment": 162, "arm": arm, "seed": seed})
        val_path = folder / f"validation_{update:04d}.json"
        old.save_new(val_path, report)
        curves.append({"update": update, "pair_loss": report["pair_loss"],
                       "checkpoint_sha256": old.file_hash(path), "validation_sha256": old.file_hash(val_path)})
        if report["pair_loss"] < best:
            best, selected = report["pair_loss"], update
        print({"fit": f"{arm}_{seed}", "update": update, "total_updates": max(points),
               "elapsed_seconds": time.perf_counter()-wall}, flush=True)

    observe()
    sequence, slots = [], []
    for epoch in range(4):
        for shared, replay in batch_plan(training, seed, epoch):
            indices = shared if arm == "refresh" else replay
            slots.append([training[i]["id"] for i in shared])
            batch = [training[i] for i in indices]
            sequence.append([r["id"] for r in batch])
            model.train()
            optimizer.zero_grad(set_to_none=True)
            logits, _ = model(*f.collate(batch))
            loss = f.objective(logits, batch, "treatment")[0].mean()
            assert torch.isfinite(loss)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(parameters, 1., error_if_nonfinite=True)
            optimizer.step()
            assert all(torch.isfinite(v).all() for v in model.state_dict().values())
            update += 1
            if update in points:
                observe()
    assert update == max(points)
    result = {"arm": arm, "seed": seed, "initializer": initializer(seed), "selected_update": selected,
              "curves": curves, "updates": update, "train_rows": len(training), "validation_rows": len(valid),
              "slot_sequence_sha256": old.digest(slots), "batch_sequence_sha256": old.digest(sequence), "elapsed_seconds": time.perf_counter()-wall}
    old.save_new(folder / "result.json", result)
    return artifact(folder / "result.json", arm=arm, seed=seed)


def schedule(stage):
    assert stage in ("probe", "main")
    result = []
    for pair in range(CONFIG[f"{stage}_pairs"]):
        for comparison in COMPARISONS:
            for seed in SEEDS:
                for seat in (0, 1):
                    result.append({"index": len(result), "pair": pair, "comparison": comparison,
                                   "seed": seed, "seat": seat, "environment_seed": CONFIG[f"{stage}_seed"]+pair,
                                   "stream": old.digest([CONFIG["stream_tag"], stage, pair])})
    return result


def model_keys(spec):
    return (f"refresh_{spec['seed']}",
            f"{'replay' if spec['comparison'] == 'transfer' else 'anchor'}_{spec['seed']}")


MODELS = {}


def load_source_models():
    MODELS.clear()
    for seed in SEEDS:
        MODELS[f"anchor_{seed}"] = f.StrongPolicyInference(initializer(seed)["path"]).model


def load_models(base=OUTPUT):
    selected = old.read_json(base / "selected_lock.json")
    MODELS.clear()
    for key, item in selected["models"].items():
        assert old.file_hash(Path(item["path"])) == item["sha256"]
        MODELS[key] = f.StrongPolicyInference(item["path"]).model


def game_path(stage, index, base=OUTPUT):
    assert stage in ("source", "probe", "main", "test")
    if stage == "test":
        assert base.resolve() != PRODUCTION_OUTPUT and PRODUCTION_OUTPUT not in base.resolve().parents
    return base / stage / "games" / f"game_{index:04d}.json"


def play_game(stage, spec, binding, base=OUTPUT, test_initial_snapshot=None):
    path = game_path(stage, spec["index"], base)
    assert not path.exists(), "No game replacement or automatic resume"
    assert test_initial_snapshot is None or stage == "test"
    env = old.SplendorEnv(seed=spec["environment_seed"])
    state = env.initial_state() if test_initial_snapshot is None else old.deserialize_state_snapshot(test_initial_snapshot)
    record = {"manifest": binding, "spec": spec, "time": old.now(), "trace": []}
    if test_initial_snapshot is not None:
        record["test_initial_snapshot"] = test_initial_snapshot
    keys = (f"anchor_{spec['seed']}",)*2 if stage == "source" or spec["comparison"] == "selfplay" else model_keys(spec)
    wall = time.perf_counter()
    seen, forced, idle = Counter(), 0, 0
    reason, winner = "completed", None
    try:
        while not state.terminal:
            seen[old.state_signature(state)] += 1
            assert seen[old.state_signature(state)] < 4, "repetition_cutoff"
            assert state.turn_index < 400, "max_turns"
            legal = env.legal_actions(state)
            assert legal, "stalled"
            key = keys[0 if state.current_player == spec["seat"] else 1]
            row = {"turn": state.turn_index, "actor": state.current_player,
                   "state_sha256": old.digest(old.serialize_state_snapshot(state)),
                   "legal_support": len(legal), "phase": old.phase(state.turn_index), "model": key}
            if stage == "source" or spec["comparison"] == "selfplay":
                row["snapshot"] = old.serialize_state_snapshot(state)
            sw, sc = time.perf_counter(), time.process_time()
            action, row["evidence"] = f.choose(state, spec["stream"], MODELS[key])
            row["timing"] = {"wall": time.perf_counter()-sw, "cpu": time.process_time()-sc}
            assert action is not None and action in legal, "illegal_or_missing_action"
            row["action"] = old.serialize_action(action)
            record["trace"].append(row)
            nxt = env.step(state, action)
            forced = forced+1 if old.is_forced_pass_turn(legal) else 0
            idle = 0 if old.is_progress_transition(state, nxt, action) else idle+1
            state = nxt
            if forced >= 2:
                reason, winner = "forced_pass_deadlock", old.adjudicate_scores(state)
                break
            assert idle < 60, "no_progress_cutoff"
        if reason == "completed":
            winner = state.winner
        record.update(reason=reason, winner=winner)
    except BaseException:
        record["failure"] = traceback.format_exc()
        raise
    finally:
        record.update(final_state=old.serialize_state_snapshot(state), elapsed_seconds=time.perf_counter()-wall)
        old.save_new(path, record)
    return artifact(path, index=spec["index"])


def statistics(rows):
    assert len(rows) == 4800
    lookup = {(r["pair"], r["comparison"], r["seed"], r["seat"]): r for r in rows}
    assert len(lookup) == 4800
    # Integer halfpoints preserve the exact 55% boundary.
    matrix = np.array([[sum(int(2*lookup[pair, comparison, seed, seat]["score"])
                            for seed in SEEDS for seat in (0, 1))
                        for comparison in COMPARISONS] for pair in range(400)], dtype=np.int64)
    assert all(r["score"] in (0., .5, 1.) for r in rows)
    rng = np.random.default_rng(CONFIG["bootstrap_seed"])
    sample = np.empty((50000, 2))
    for start in range(0, 50000, 1000):
        ids = rng.integers(400, size=(1000, 400))
        sample[start:start+1000] = matrix[ids].sum(axis=1) / 4800
    result = {}
    for j, comparison in enumerate(COMPARISONS):
        group = [r for r in rows if r["comparison"] == comparison]
        halfpoints = int(matrix[:, j].sum())
        ci = np.percentile(sample[:, j], [2.5, 97.5]).tolist()
        result[comparison] = {"score": halfpoints / 4800, "ci95": ci, "halfpoints": halfpoints,
                              "wins": sum(r["score"] == 1 for r in group),
                              "draws": sum(r["score"] == .5 for r in group),
                              "losses": sum(r["score"] == 0 for r in group),
                              "margin": sum(r["margin"] for r in group) / 2400,
                              "seats": [sum(r["score"] for r in group if r["seat"] == s) / 1200 for s in (0, 1)],
                              "per_fit": {str(seed): sum(r["score"] for r in group if r["seed"] == seed) / 800 for seed in SEEDS},
                              "pass": ci[0] > .5 and (comparison == "transfer" or halfpoints * 100 >= 55 * 4800)}
    result["both_pass"] = all(result[c]["pass"] for c in COMPARISONS)
    return result
