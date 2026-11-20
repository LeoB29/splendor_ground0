"""161: matched incremental learning from saved 128 / first-32 teacher labels."""
from collections import Counter
from pathlib import Path
import math
import time
import traceback

import numpy as np
import torch

from splendor_ai.alphazero import fast_selector_157 as f
from splendor_ai.alphazero import world_budget_160 as source

old = f.old
OUTPUT = Path("outputs/teacher_budget_161")
ARMS = ("world128", "world32")
SEEDS = (161001, 161002, 161003)
COMPARISONS = ("transfer", "progress")
INITIALIZERS = {
    161001: (157001, 1220, "e2508f56d22e9b9ae2f87004dd9962718e5d3b4f6bd0824088d51f1d27f82b21"),
    161002: (157002, 1220, "adebd57d57ca53406f5aab01f69a2544885b657b225e88b8c6068e0f6d8ac0f4"),
    161003: (157003, 1144, "960a258c75d5854da64d5bca98bbfceb8c333bbd95ec845f9ae65206539f6ecb"),
}
CONFIG = {"epochs": 4, "batch_size": 64, "lr": 1e-5, "weight_decay": 1e-4,
          "gradient_norm": 1., "fit_workers": 3, "workers": 8, "torch_threads": 1,
          "rows": 11985, "validation_pairs": 40, "main_pairs": 400, "probe_pairs": 4,
          "bootstrap_seed": 1613000000, "bootstrap_resamples": 50000,
          "split_tag": "161-source-pair-split-v1", "stream_tag": "161-proposer-stream-v1",
          "main_seed": 1610000000, "probe_seed": 1619000000,
          "automatic_resume": False, "promotion": False, "automatic_successor": False,
          "scheduled_monitoring": False, "device": "cpu", "fresh_optimizer": True}


def initializer(seed):
    old_seed, update, sha = INITIALIZERS[seed]
    path = f.OUTPUT / "fits" / f"treatment_{old_seed}" / f"checkpoint_{update:04d}.pt"
    assert old.file_hash(path) == sha
    return {"path": path.as_posix(), "sha256": sha, "seed": old_seed, "update": update}


def splits():
    order = sorted(range(200), key=lambda pair: old.digest([CONFIG["split_tag"], 160, pair]))
    return {f"160:{pair}": "validation" if pair in order[:40] else "train" for pair in range(200)}


def labels(record):
    support = len(record["candidates"]["actions"])
    worlds = record["worlds"]
    assert 1 <= support <= 4 and len(worlds) == 128
    assert all(len(w["outcomes"]) == support for w in worlds)
    assert all(o["wdl"] in (-1., 0., 1.) and o["reason"] in old.VALID_NATURAL_TERMINATIONS
               for w in worlds for o in w["outcomes"])
    sums = {arm: [sum(w["outcomes"][j]["wdl"] for w in worlds[:n]) for j in range(support)]
            for arm, n in (("world128", 128), ("world32", 32))}
    result = {arm: max(range(support), key=lambda j: values[j]) for arm, values in sums.items()}
    assert record["choice"] == result["world128"]
    assert record["means"] == [v / 128 for v in sums["world128"]]
    return result


def source_inventory():
    lock = old.read_json(source.OUTPUT / "main_collect_lock.json")
    assert set(lock["files"]) == {str(i) for i in range(400)}
    result = []
    for index in range(400):
        path = source.game_path("main", index)
        assert old.file_hash(path) == lock["files"][str(index)]
        game = old.read_json(path)
        assert game["manifest"] == lock["manifest"] and "failure" not in game
        assert game["reason"] in old.VALID_NATURAL_TERMINATIONS
        result.append({"experiment": 160, "index": index, "pair": f"160:{index//2}",
                       "parent_binding": lock["manifest"], "path": path.as_posix(),
                       "sha256": lock["files"][str(index)],
                       "roots": [{"turn": m["turn"], "sha256": m["root_sha256"]}
                                 for m in game["trace"] if m["actor"] == game["spec"]["candidate_seat"]]})
    assert sum(len(s["roots"]) for s in result) == CONFIG["rows"]
    return result


def artifact(path, **values):
    return {"path": Path(path).as_posix(), "sha256": old.file_hash(Path(path)), **values}


def export_game(spec, split, base=OUTPUT):
    assert old.file_hash(Path(spec["path"])) == spec["sha256"]
    game = old.read_json(Path(spec["path"]))
    moves = {m["turn"]: m for m in game["trace"]}
    rows = []
    for item in spec["roots"]:
        path = source.root_path("main", spec["index"], item["turn"])
        assert old.file_hash(path) == item["sha256"] == moves[item["turn"]]["root_sha256"]
        root = old.read_json(path)
        assert root["manifest"] == spec["parent_binding"] and "failure" not in root
        assert old.digest(root["snapshot"]) == moves[item["turn"]]["state_sha256"]
        assert root["candidates"]["actions"][root["choice"]] == moves[item["turn"]]["action"]
        state = old.deserialize_state_snapshot(root["snapshot"])
        row = f.make_row(state, root, spec, split, old._RAW._model.config)
        row["labels"] = labels(root)
        rows.append(row)
    target = base / "export" / f"game_{spec['index']:04d}.pt"
    assert not target.exists()
    old.tensor_save(target, rows)
    return artifact(target, index=spec["index"], rows=len(rows))


def load_data(arm, base=OUTPUT):
    assert arm in ARMS
    rows = []
    lock = old.read_json(base / "export_lock.json")
    for item in lock["files"]:
        assert old.file_hash(Path(item["path"])) == item["sha256"]
        data = torch.load(item["path"], weights_only=True, map_location="cpu")
        assert len(data) == item["rows"]
        rows.extend({**r, "teacher": r["labels"][arm]} for r in data)
    assert len(rows) == CONFIG["rows"] and len({r["id"] for r in rows}) == len(rows)
    return rows


def train_fit(arm, seed, rows=None, base=OUTPUT):
    """Rows override only permits isolated fixtures; production always loads the locked corpus."""
    assert arm in ARMS and seed in SEEDS
    if rows is not None:
        assert base.resolve() != OUTPUT.resolve()
    else:
        rows = load_data(arm, base)
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
        f.save_model(path, model, update, {"experiment": 161, "arm": arm, "seed": seed})
        val_path = folder / f"validation_{update:04d}.json"
        old.save_new(val_path, report)
        curves.append({"update": update, "pair_loss": report["pair_loss"],
                       "checkpoint_sha256": old.file_hash(path), "validation_sha256": old.file_hash(val_path)})
        if report["pair_loss"] < best:
            best, selected = report["pair_loss"], update
        print({"fit": f"{arm}_{seed}", "update": update, "total_updates": max(points),
               "elapsed_seconds": time.perf_counter()-wall}, flush=True)

    observe()
    sequence = []
    for epoch in range(4):
        for indices in f.batches(training, seed, epoch):
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
              "batch_sequence_sha256": old.digest(sequence), "elapsed_seconds": time.perf_counter()-wall}
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
    return (f"world128_{spec['seed']}",
            f"{'world32' if spec['comparison'] == 'transfer' else 'anchor'}_{spec['seed']}")


MODELS = {}


def load_models(base=OUTPUT):
    selected = old.read_json(base / "selected_lock.json")
    MODELS.clear()
    for key, item in selected["models"].items():
        assert old.file_hash(Path(item["path"])) == item["sha256"]
        MODELS[key] = f.StrongPolicyInference(item["path"]).model


def game_path(stage, index, base=OUTPUT):
    assert stage in ("probe", "main", "test")
    if stage == "test":
        assert base.resolve() != OUTPUT.resolve()
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
    keys = model_keys(spec)
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
