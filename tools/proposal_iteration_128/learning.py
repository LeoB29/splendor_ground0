"""One fixed feedback generation, with independently reconstructed inputs/fits.

No frozen module's output/config globals are modified. Existing 161 exports are
accepted only through their independent audit receipts. New learned128 inputs
are checked against replayed actual games and independent observer encoders;
source continuation verification remains the already completed sampled audit.
"""
from __future__ import annotations

from collections import Counter
from functools import lru_cache
import math
from pathlib import Path
import random
import time

import torch

from scripts.analysis import verify_fast_selector_transfer_157 as independent
from splendor_ai.alphazero import teacher_budget_161 as previous
from tools.learned_proposal_128 import experiment as source
from . import common

old, f = previous.old, previous.f
ROOT, OUTPUT = common.ROOT, common.OUTPUT
ARMS = ("treatment", "replay")
FITS = (161001, 161002, 161003)
TRAIN_SEEDS = {161001: 207001, 161002: 207002, 161003: 207003}
CONFIG = {"epochs": 4, "batch_size": 64, "lr": 1e-5, "weight_decay": 1e-4,
          "gradient_norm": 1., "torch_threads": 1, "fit_workers": 3,
          "old_rows": 11985, "old_train_rows": 9577, "old_validation_rows": 2408,
          "new_rows": 11951, "new_pairs_per_fit": 67, "new_validation_pairs_per_fit": 14,
          "split_tag": "proposal-iteration-128-source-pair-split-v1",
          "batch_tag": "proposal-iteration-128-batches-v1",
          "replacement_tag": "proposal-iteration-128-old-replacement-v1",
          "device": "cpu", "fresh_optimizer": True, "initial_checkpoint_eligible": True,
          "validation": "mean_of_whole_pair_mean_offered_group_cross_entropy",
          "selection": "earliest_strict_minimum", "unoffered_actions_are_negative": False}


def _item(path, **metadata):
    path = Path(path).resolve()
    name = path.relative_to(ROOT).as_posix() if path.is_relative_to(ROOT) else path.as_posix()
    return {"path": name, "sha256": old.file_hash(path), **metadata}


def _path(item):
    path = ROOT / item["path"]
    assert old.file_hash(path) == item["sha256"], f"Changed learning artifact: {path}"
    return path


def _json(item):
    return old.read_json(_path(item))


def initializer(fit):
    assert fit in FITS
    update, sha = source.MODELS[fit]
    path = ROOT / previous.OUTPUT / "fits" / f"world128_{fit}" / f"checkpoint_{update:04d}.pt"
    assert old.file_hash(path) == sha
    return _item(path, source_fit=fit, selected_update=update)


def source_dependencies():
    paths = [ROOT / previous.OUTPUT / name for name in
             ("export_lock.json", "export_audit_lock.json", "export_summary.json", "splits.json", "sources.json")]
    paths += [source.OUTPUT / name for name in
              ("final_audit.json", "main_complete.json", "main_games_lock.json", "main_audits_lock.json", "schedule.json")]
    return [_item(path) for path in paths] + [initializer(fit) for fit in FITS]


@lru_cache(maxsize=1)
def splits():
    result = previous.splits()
    schedule = source.specs("main")
    for fit in FITS:
        pairs = sorted({s["pair"] for s in schedule if s["fit"] == fit})
        assert len(pairs) == 67
        order = sorted(pairs, key=lambda p: old.digest([CONFIG["split_tag"], fit, p]))
        validation = set(order[:14])
        result.update({f"learned128:{p}": "validation" if p in validation else "train" for p in pairs})
    assert len(result) == 401
    return result


@lru_cache(maxsize=1)
def independent_splits():
    # Reconstruct source assignment independently, without its schedule helper.
    assignment = sorted(range(201), key=lambda p: old.digest(["learned-proposal-128-fit-v1", "main", p]))
    by_fit = {fit: assignment[i::3] for i, fit in enumerate(FITS)}
    old_order = sorted(range(200), key=lambda p: old.digest(["161-source-pair-split-v1", 160, p]))
    value = {f"160:{p}": "validation" if p in old_order[:40] else "train" for p in range(200)}
    for fit, pairs in by_fit.items():
        ranked = sorted(pairs, key=lambda p: old.digest(["proposal-iteration-128-source-pair-split-v1", fit, p]))
        for pair in pairs:
            value[f"learned128:{pair}"] = "validation" if ranked.index(pair) < 14 else "train"
    return value


def _old_exports():
    lock = old.read_json(ROOT / previous.OUTPUT / "export_lock.json")
    audit_lock = old.read_json(ROOT / previous.OUTPUT / "export_audit_lock.json")
    assert lock["manifest"] == audit_lock["manifest"]
    assert len(lock["files"]) == len(audit_lock["files"]) == 400
    assert [r["index"] for r in lock["files"]] == list(range(400))
    assert [r["index"] for r in audit_lock["files"]] == list(range(400))
    count = Counter()
    for item, audit_item in zip(lock["files"], audit_lock["files"]):
        receipt = _json(audit_item)
        assert receipt["passed"] and receipt["export"] == item
        rows = torch.load(_path(item), map_location="cpu", weights_only=True)
        assert len(rows) == item["rows"] == receipt["coverage"]["rows"]
        assert receipt["source"]["index"] == item["index"]
        for row in rows:
            assert row["split"] == independent_splits()[row["pair"]]
            assert row["id"].startswith(f"160:{item['index']}:")
            assert row["teacher"] == row["labels"]["world128"]
            count[row["split"]] += 1
    assert count == {"train": 9577, "validation": 2408}
    return lock["files"], audit_lock["files"]


def _source_games():
    audit = old.read_json(source.OUTPUT / "final_audit.json")
    assert audit["passed"] and not audit["full_inner_replay"] and audit["games"] == 402
    complete = _json(audit["main"])
    games = _json(complete["game_lock"])
    audits = _json(complete["audit_lock"])
    assert games["manifest"] == audits["manifest"] == complete["manifest"] == audit["manifest"]
    assert len(games["files"]) == len(audits["files"]) == 402
    schedule = source.specs("main")
    assert old.read_json(source.OUTPUT / "schedule.json")["main"] == schedule
    result = []
    for index, (item, receipt_item) in enumerate(zip(games["files"], audits["files"])):
        assert item["index"] == receipt_item["index"] == index
        game, receipt = _json(item), _json(receipt_item)
        assert game["spec"] == schedule[index] and game["manifest"] == audit["manifest"]
        assert receipt["manifest"] == audit["manifest"] and receipt["passed"]
        assert receipt["game_artifact"] == item and not receipt["full_inner_replay"]
        assert "failure" not in game and game["reason"] in old.VALID_NATURAL_TERMINATIONS
        assert len(game["trace"]) == len(receipt["roots"])
        result.append({"game": item, "audit": receipt_item, "spec": game["spec"], "parent_binding": audit["manifest"]})
    return result


def inspect_row(row, state, root, spec, config):
    """Independent tensor/membership/label reconstruction, including hidden swap."""
    obs, features, groups, keys, legal = independent.input_invariance(state, config)
    offered = root["candidates"]
    assert len(offered["keys"]) == len(set(offered["keys"])) == len(offered["actions"])
    assert [repr(old.canonical_successor_action_key(state, old.decode_action(a))) for a in offered["actions"]] == offered["keys"]
    members = [[i for i, action in enumerate(legal) if repr(old.canonical_successor_action_key(state, action)) == key]
               for key in offered["keys"]]
    assert all(members) and 1 <= len(members) <= 4
    assert torch.equal(row["observation"], obs) and torch.equal(row["features"], features)
    assert row["groups"].tolist() == groups and row["group_count"] == len(keys)
    assert [m.tolist() for m in row["members"]] == members
    assert len(root["worlds"]) == 128
    totals = [0.] * len(members)
    for world in root["worlds"]:
        assert len(world["outcomes"]) == len(members)
        for index, outcome in enumerate(world["outcomes"]):
            assert outcome["wdl"] in (-1., 0., 1.) and outcome["reason"] in old.VALID_NATURAL_TERMINATIONS
            totals[index] += outcome["wdl"]
    means = [v / 128 for v in totals]
    label = means.index(max(means))
    assert means == root["means"] and label == root["choice"]
    privacy = {role: sum(player.reserved_card_source(i) != independent.RESERVE_SOURCE_VISIBLE
                         for i in range(len(player.reserved_cards)))
               for role, player in (("own", state.players[state.current_player]), ("opponent", state.players[1-state.current_player]))}
    meta = {"id": f"learned128:{spec['index']}:{state.turn_index}", "pair": f"learned128:{spec['pair']}",
            "split": independent_splits()[f"learned128:{spec['pair']}"], "teacher": label,
            "seat": state.current_player, "phase": min(state.turn_index//20, 2), "privacy": privacy}
    assert privacy == root["privacy"]
    assert set(row) == set(meta) | {"observation", "features", "groups", "group_count", "members"}
    assert {key: row[key] for key in meta} == meta
    return {"rows": 1, row["split"]: 1, f"fit_{spec['fit']}": 1, f"support_{len(members)}": 1,
            f"phase_{row['phase']}": 1, f"seat_{row['seat']}": 1,
            "own_private": int(privacy["own"] > 0), "opponent_private": int(privacy["opponent"] > 0),
            "label_ties": int(sum(v == max(totals) for v in totals) > 1)}


def export_game(item, binding, base=OUTPUT):
    game, receipt = _json(item["game"]), _json(item["audit"])
    spec = item["spec"]
    assert game["spec"] == spec and game["manifest"] == item["parent_binding"]
    assert receipt["passed"] and receipt["game_artifact"] == item["game"]
    rows, roots = [], []
    for move, checked in zip(game["trace"], receipt["roots"]):
        assert move["turn"] == checked["turn"] and move["root_sha256"] == checked["sha256"]
        if move["actor"] != spec["candidate_seat"]:
            continue
        assert checked["planner"] == "candidate" and checked["fit"] == spec["fit"]
        path = source.root_path("main", spec["index"], move["turn"])
        assert old.file_hash(path) == move["root_sha256"]
        root = old.read_json(path)
        assert root["manifest"] == item["parent_binding"] and "failure" not in root
        assert old.digest(root["snapshot"]) == move["state_sha256"]
        assert root["candidates"]["actions"][root["choice"]] == move["action"]
        state = old.deserialize_state_snapshot(root["snapshot"])
        row = f.make_row(state, root, {"experiment": "learned128", "index": spec["index"],
                         "pair": f"learned128:{spec['pair']}"}, splits()[f"learned128:{spec['pair']}"], old._RAW._model.config)
        rows.append(row)
        roots.append({"turn": move["turn"], "sha256": move["root_sha256"]})
    target = Path(base) / "data" / "exports" / f"game_{spec['index']:04d}.pt"
    assert not target.exists()
    old.tensor_save(target, rows)
    exported = _item(target, index=spec["index"], rows=len(rows))
    # Audit the saved tensors against actual-game replay, not the extraction snapshot.
    saved = torch.load(_path(exported), map_location="cpu", weights_only=True)
    env = old.SplendorEnv(seed=spec["environment_seed"])
    state, index, coverage = env.initial_state(), 0, Counter()
    for move in game["trace"]:
        assert (move["turn"], move["actor"]) == (state.turn_index, state.current_player)
        assert old.digest(old.serialize_state_snapshot(state)) == move["state_sha256"]
        if state.current_player == spec["candidate_seat"]:
            path = source.root_path("main", spec["index"], move["turn"])
            assert old.file_hash(path) == roots[index]["sha256"]
            root = old.read_json(path)
            coverage.update(inspect_row(saved[index], state, root, spec, old._RAW._model.config))
            assert root["candidates"]["actions"][saved[index]["teacher"]] == move["action"]
            index += 1
        action = old.decode_action(move["action"])
        assert action in env.legal_actions(state)
        state = env.step(state, action)
    assert index == len(rows) == len(saved) and state.terminal
    assert old.digest(old.serialize_state_snapshot(state)) == old.digest(game["final_state"])
    audit_path = Path(base) / "data" / "audits" / f"game_{spec['index']:04d}.json"
    old.save_new(audit_path, {"manifest": binding, "passed": True, "source": item, "export": exported,
                            "source_roots": roots, "coverage": dict(coverage), "actual_game_replayed": True,
                            "every_row_hidden_input_invariance": True, "continuations_replayed": 0,
                            "source_continuation_verification": "inherited_completed_sampled_learned128_audit"})
    return exported, _item(audit_path, index=spec["index"])


def prepare_data(binding, base=OUTPUT):
    base = Path(base)
    assert not (base / "data_lock.json").exists()
    old.initialize_worker()
    assert splits() == independent_splits()
    dependencies = source_dependencies()
    exports, old_audits = _old_exports()
    sources = _source_games()
    new_exports, new_audits, coverage = [], [], Counter()
    for item in sources:
        exported, audit = export_game(item, binding, base)
        new_exports.append(exported)
        new_audits.append(audit)
        coverage.update(_json(audit)["coverage"])
        if (item["spec"]["index"]+1) % 20 == 0:
            print({"learning_export_games": item["spec"]["index"]+1, "total_games": 402}, flush=True)
    assert coverage["rows"] == 11951
    assert [coverage[f"fit_{fit}"] for fit in FITS] == [4001, 3998, 3952]
    assert [coverage[f"support_{n}"] for n in range(1, 5)] == [135, 170, 199, 11447]
    assert Counter(v for k, v in splits().items() if k.startswith("learned128:")) == {"train": 159, "validation": 42}
    audit_path = base / "data_audit.json"
    old.save_new(audit_path, {"manifest": binding, "passed": True, "source_dependencies": dependencies,
                            "old_export_audits": old_audits, "new_export_audits": new_audits,
                            "new_coverage": dict(coverage), "old_rows": 11985,
                            "retained_old_split": True, "new_pair_splits": splits(),
                            "new_independent_tensor_reconstruction": True, "continuations_replayed": 0})
    lock = {"manifest": binding, "config": CONFIG, "source_dependencies": dependencies,
            "audit": _item(audit_path), "old_exports": exports, "new_exports": new_exports,
            "splits": splits(), "rows": 23936,
            "train_rows": 9577+coverage["train"], "validation_rows": 2408+coverage["validation"]}
    old.save_new(base / "data_lock.json", lock)
    load_data(binding, base)
    return _item(base / "data_lock.json")


def load_data(binding, base=OUTPUT):
    lock = old.read_json(Path(base) / "data_lock.json")
    assert lock["manifest"] == binding and lock["config"] == CONFIG
    assert lock["splits"] == splits() == independent_splits()
    audit = _json(lock["audit"])
    assert audit["passed"] and audit["manifest"] == binding
    assert audit["source_dependencies"] == lock["source_dependencies"]
    for item in lock["source_dependencies"]:
        _path(item)
    rows = []
    for kind in ("old", "new"):
        assert len(lock[f"{kind}_exports"]) == len(audit[f"{kind}_export_audits"]) == (400 if kind == "old" else 402)
        for item, checked in zip(lock[f"{kind}_exports"], audit[f"{kind}_export_audits"]):
            receipt = _json(checked)
            assert receipt["passed"] and receipt["export"] == item
            loaded = torch.load(_path(item), map_location="cpu", weights_only=True)
            assert len(loaded) == item["rows"]
            if kind == "old":
                loaded = [{**row, "teacher": row["labels"]["world128"]} for row in loaded]
            rows.extend(loaded)
    assert len(rows) == lock["rows"] == 23936 and len({r["id"] for r in rows}) == len(rows)
    assert Counter(r["split"] for r in rows) == {"train": lock["train_rows"], "validation": lock["validation_rows"]}
    assert all(r["split"] == lock["splits"][r["pair"]] for r in rows)
    return rows


def batch_plan(training, seed, epoch):
    assert all(r["split"] == "train" for r in training)
    older = [i for i, row in enumerate(training) if row["id"].startswith("160:")]
    assert older and all(r["id"].startswith(("160:", "learned128:")) for r in training)
    order = list(range(len(training)))
    rng = random.Random(f"{CONFIG['batch_tag']}|{seed}|{epoch}")
    rng.shuffle(order)
    order.sort(key=lambda i: len(training[i]["features"]))
    chunks = [order[i:i+64] for i in range(0, len(order), 64)]
    rng.shuffle(chunks)
    replacement = {i: random.Random(f"{CONFIG['replacement_tag']}|{seed}|{epoch}|{row['id']}").choice(older)
                   for i, row in enumerate(training) if row["id"].startswith("learned128:")}
    return [(chunk, [replacement.get(i, i) for i in chunk]) for chunk in chunks]


def independent_plan(training, seed, epoch):
    assert all(row["split"] == "train" for row in training)
    order = [i for i, _ in enumerate(training)]
    generator = random.Random("proposal-iteration-128-batches-v1|%s|%s" % (seed, epoch))
    generator.shuffle(order)
    order = sorted(order, key=lambda i: training[i]["features"].size(0))
    chunks = [order[start:start+64] for start in range(0, len(order), 64)]
    generator.shuffle(chunks)
    older = [i for i in range(len(training)) if training[i]["id"].split(":")[0] == "160"]
    assert older
    result = []
    for chunk in chunks:
        replacement = []
        for index in chunk:
            if training[index]["id"].split(":")[0] == "160":
                replacement.append(index)
            else:
                assert training[index]["id"].split(":")[0] == "learned128"
                rng = random.Random("proposal-iteration-128-old-replacement-v1|%s|%s|%s" % (seed, epoch, training[index]["id"]))
                replacement.append(older[rng.randrange(len(older))])
        result.append((chunk, replacement))
    return result


def _rows(binding, base, rows):
    if rows is None:
        return load_data(binding, base)
    # A fixture must never write into the production evidence tree.
    assert not Path(base).resolve().is_relative_to(OUTPUT.resolve())
    return rows


def fit_job(fit, arm, binding, base=OUTPUT, rows=None):
    assert fit in FITS and arm in ARMS
    old.initialize_worker()
    data = _rows(binding, base, rows)
    model = f.StrongPolicyInference(str(_path(initializer(fit)))).model
    initial_head = {name: value.clone() for name, value in model.state_dict().items() if name.startswith("value_head.")}
    for name, parameter in model.named_parameters():
        parameter.requires_grad_(not name.startswith("value_head."))
    parameters = [p for p in model.parameters() if p.requires_grad]
    optimizer = torch.optim.AdamW(parameters, lr=1e-5, weight_decay=1e-4)
    assert not optimizer.state
    training, validation = ([r for r in data if r["split"] == kind] for kind in ("train", "validation"))
    assert training and validation
    folder = Path(base) / "fits" / f"{arm}_{fit}"
    folder.mkdir(parents=True, exist_ok=False)
    points = f.checkpoint_updates(len(training))
    best, selected, update, curves = math.inf, None, 0, []
    started = time.perf_counter()

    def observe():
        nonlocal best, selected
        assert all(torch.equal(value, model.state_dict()[name]) for name, value in initial_head.items())
        report = f.validation(model, validation, "treatment")
        checkpoint = folder / f"checkpoint_{update:04d}.pt"
        metadata = {"experiment": "proposal_iteration_128", "arm": arm, "fit": fit, "seed": TRAIN_SEEDS[fit], "manifest": binding}
        f.save_model(checkpoint, model, update, metadata)
        validation_path = folder / f"validation_{update:04d}.json"
        old.save_new(validation_path, report)
        curves.append({"update": update, "pair_loss": report["pair_loss"],
                       "checkpoint_sha256": old.file_hash(checkpoint), "validation_sha256": old.file_hash(validation_path)})
        if report["pair_loss"] < best:
            best, selected = report["pair_loss"], update
        old.atomic_write(folder / "progress.json", {"manifest": binding, "fit": fit, "arm": arm,
                         "update": update, "total_updates": max(points), "elapsed_seconds": time.perf_counter()-started})
        print({"fit": f"{arm}_{fit}", "update": update, "total_updates": max(points)}, flush=True)

    observe()
    sequence, slots = [], []
    for epoch in range(4):
        for shared, replay in batch_plan(training, TRAIN_SEEDS[fit], epoch):
            indices = shared if arm == "treatment" else replay
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
            assert all(torch.isfinite(t).all() for t in model.state_dict().values())
            update += 1
            if update in points:
                observe()
    assert update == max(points)
    result = {"manifest": binding, "arm": arm, "fit": fit, "seed": TRAIN_SEEDS[fit], "initializer": initializer(fit),
              "data_lock": _item(Path(base) / "data_lock.json") if rows is None else None,
              "selected_update": selected, "curves": curves, "updates": update,
              "train_rows": len(training), "validation_rows": len(validation), "row_exposures": 4*len(training),
              "slot_sequence_sha256": old.digest(slots), "batch_sequence_sha256": old.digest(sequence),
              "elapsed_seconds": time.perf_counter()-started}
    old.save_new(folder / "result.json", result)
    return _item(folder / "result.json", fit=fit, arm=arm)


def audit_fit_job(fit, arm, binding, base=OUTPUT, rows=None):
    assert fit in FITS and arm in ARMS
    old.initialize_worker()
    data = _rows(binding, base, rows)
    training, validation = ([r for r in data if r["split"] == kind] for kind in ("train", "validation"))
    folder = Path(base) / "fits" / f"{arm}_{fit}"
    expected = old.read_json(folder / "result.json")
    assert expected["manifest"] == binding and expected["initializer"] == initializer(fit)
    assert expected["arm"] == arm and expected["fit"] == fit and expected["seed"] == 207001+FITS.index(fit)
    if rows is None:
        assert expected["data_lock"] == _item(Path(base) / "data_lock.json")
    model = f.StrongPolicyInference(str(_path(initializer(fit)))).model
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
        assert payload["metadata"] == {"experiment": "proposal_iteration_128", "arm": arm, "fit": fit,
                                       "seed": 207001+FITS.index(fit), "manifest": binding}
        assert payload["model_state_dict"].keys() == model.state_dict().keys()
        for name, tensor in model.state_dict().items():
            assert torch.isfinite(tensor).all() and torch.equal(tensor, payload["model_state_dict"][name]), (arm, fit, update, name)
            if name in initial_head:
                assert torch.equal(tensor, initial_head[name])
        val_path = folder / f"validation_{update:04d}.json"
        assert old.file_hash(val_path) == curve["validation_sha256"]
        report = independent.audit_validation(model, validation, "treatment")
        assert report == old.read_json(val_path) and report["pair_loss"] == curve["pair_loss"]
        if report["pair_loss"] < best:
            best, selected = report["pair_loss"], update
        observed.append(update)
        print({"replay_fit": f"{arm}_{fit}", "update": update, "total_updates": 4*steps}, flush=True)

    inspect()
    for epoch in range(4):
        for shared, replay in independent_plan(training, 207001+FITS.index(fit), epoch):
            slots.append([training[i]["id"] for i in shared])
            batch = [training[i] for i in (shared if arm == "treatment" else replay)]
            sequence.append([r["id"] for r in batch])
            optimizer.zero_grad(set_to_none=True)
            model.train()
            logits, _ = model(*independent.audit_batch_tensors(batch))
            loss = independent.audit_objective(logits, batch, "treatment")[0].mean()
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
    assert expected["row_exposures"] == 4*len(training)
    selected_payload = torch.load(folder / f"checkpoint_{selected:04d}.pt", map_location="cpu", weights_only=True)
    initial_payload = torch.load(_path(initializer(fit)), map_location="cpu", weights_only=True)
    identical = all(torch.equal(value, initial_payload["model_state_dict"][name])
                    for name, value in selected_payload["model_state_dict"].items())
    result = {"manifest": binding, "arm": arm, "fit": fit, "selected_update": selected, "checkpoints": observed,
              "complete_fit_bitwise_identical": True, "value_head_frozen": True, "selected_equals_initializer": identical,
              "result": _item(folder / "result.json"), "elapsed_seconds": time.perf_counter()-started}
    old.save_new(folder / "audit.json", result)
    return _item(folder / "audit.json", fit=fit, arm=arm)


def seal_models(binding, base=OUTPUT):
    base = Path(base)
    data_lock = _item(base / "data_lock.json")
    load_data(binding, base)
    models, results, audits = {}, [], []
    unchanged = []
    for fit in FITS:
        matched = []
        for arm in ARMS:
            folder = base / "fits" / f"{arm}_{fit}"
            result_item, audit_item = _item(folder / "result.json"), _item(folder / "audit.json")
            result, audit = _json(result_item), _json(audit_item)
            assert result["manifest"] == audit["manifest"] == binding
            assert result["arm"] == audit["arm"] == arm and result["fit"] == audit["fit"] == fit
            assert audit["complete_fit_bitwise_identical"] and audit["value_head_frozen"]
            assert audit["result"] == result_item and result["data_lock"] == data_lock
            assert result["initializer"] == initializer(fit)
            selected = result["selected_update"]
            assert selected == audit["selected_update"]
            for curve in result["curves"]:
                assert old.file_hash(folder / f"checkpoint_{curve['update']:04d}.pt") == curve["checkpoint_sha256"]
                assert old.file_hash(folder / f"validation_{curve['update']:04d}.json") == curve["validation_sha256"]
            model = _item(folder / f"checkpoint_{selected:04d}.pt", source_fit=fit, selected_update=selected,
                          training_seed=TRAIN_SEEDS[fit], arm=arm, selected_equals_initializer=audit["selected_equals_initializer"],
                          unchanged_from_anchor=audit["selected_equals_initializer"])
            models[f"{arm}_{fit}"] = model
            results.append(result_item)
            audits.append(audit_item)
            matched.append(result)
            if arm == "treatment":
                unchanged.append(audit["selected_equals_initializer"])
        assert all(matched[0][key] == matched[1][key] for key in
                   ("initializer", "data_lock", "train_rows", "validation_rows", "updates", "row_exposures", "slot_sequence_sha256"))
        models[f"anchor_{fit}"] = initializer(fit)
    value = {"manifest": binding, "models": models, "data_lock": data_lock,
             "fit_results": results, "fit_audits": audits, "equal_dose_verified": True,
             "all_treatments_unchanged": all(unchanged), "training_config": CONFIG}
    old.save_new(base / "models_lock.json", value)
    return _item(base / "models_lock.json")
