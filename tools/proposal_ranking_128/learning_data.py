"""Read-only parent reuse plus newly sealed, independently checked ranking targets."""
from collections import Counter
from pathlib import Path

import torch

from tools.proposal_iteration_128 import learning as previous
from . import common as c

old, ROOT, OUTPUT, SOURCE = c.old, c.ROOT, c.OUTPUT, c.SOURCE
PARENT_BINDING = "1a3c53142516a5d8f08376d0a7227ae9199952a9e077e1309d13fdcdf6d40ace"


def delta(root):
    support = len(root["candidates"]["actions"])
    assert 1 <= support <= 4 and len(root["worlds"]) == 128
    assert all(len(w["outcomes"]) == support for w in root["worlds"])
    assert all(o["wdl"] in (-1., 0., 1.) and o["reason"] in old.VALID_NATURAL_TERMINATIONS
               for world in root["worlds"] for o in world["outcomes"])
    totals = [sum(w["outcomes"][i]["wdl"] for w in root["worlds"]) for i in range(support)]
    assert root["means"] == [x/128 for x in totals]
    assert root["choice"] == totals.index(max(totals))
    return (totals[2]-totals[3])/256 if support == 4 else 0.


def independent_delta(root):
    size = len(root["candidates"]["keys"])
    assert len(set(root["candidates"]["keys"])) == size == len(root["candidates"]["actions"])
    assert 1 <= size <= 4 and len(root["worlds"]) == 128
    sums = [0.] * size
    difference = 0.
    for world in root["worlds"]:
        assert len(world["outcomes"]) == size
        scores = []
        for i, outcome in enumerate(world["outcomes"]):
            assert type(outcome["wdl"]) in (int, float) and outcome["wdl"] in (-1, 0, 1)
            assert outcome["reason"] in old.VALID_NATURAL_TERMINATIONS
            scores.append((outcome["wdl"]+1)/2)
            sums[i] += outcome["wdl"]
        if size == 4:
            difference += scores[2]-scores[3]
    assert [v/128 for v in sums] == root["means"]
    assert max(range(size), key=lambda i: sums[i]) == root["choice"]
    return difference/128


def same_row(left, right):
    assert left.keys() == right.keys()
    for key, value in left.items():
        if isinstance(value, torch.Tensor):
            assert torch.equal(value, right[key]), key
        elif key == "members":
            assert len(value) == len(right[key]) and all(torch.equal(x, y) for x, y in zip(value, right[key]))
        else:
            assert value == right[key], key


def _context(kind, receipt):
    if kind == "old":
        source = receipt["source"]
        game_item = {"path": source["path"], "sha256": source["sha256"]}
        roots, binding = source["roots"], source["parent_binding"]
        root_dir = ROOT / "outputs/world_budget_160/main/roots" / f"game_{source['index']:04d}"
        prefix = "160"
    else:
        source = receipt["source"]
        game_item = source["game"]
        source_audit = c.verified(source["audit"])
        assert source_audit["passed"] and source_audit["game_artifact"] == game_item
        roots, binding = receipt["source_roots"], source["parent_binding"]
        root_dir = ROOT / "outputs/learned_proposal_128/main/roots" / f"game_{source['spec']['index']:04d}"
        prefix = "learned128"
    game = c.verified(game_item)
    assert game["manifest"] == binding and "failure" not in game
    assert game["reason"] in old.VALID_NATURAL_TERMINATIONS
    moves = {m["turn"]: m for m in game["trace"] if m["actor"] == game["spec"]["candidate_seat"]}
    assert set(moves) == {r["turn"] for r in roots}
    return game_item, game, moves, roots, binding, root_dir, prefix


def export_shard(kind, item, audit_item, binding, base=OUTPUT):
    receipt = c.verified(audit_item)
    assert receipt["passed"] and receipt["export"] == item
    game_item, game, moves, roots, parent_binding, folder, prefix = _context(kind, receipt)
    source_rows = torch.load(c.verify_hash(item), map_location="cpu", weights_only=True)
    assert len(source_rows) == item["rows"] == len(roots)
    target_rows, root_items = [], []
    for row, recorded in zip(source_rows, roots, strict=True):
        c.budget_boundary(base)
        path = folder / f"turn_{recorded['turn']:03d}.json"
        root_item = c.artifact(path)
        assert root_item["sha256"] == recorded["sha256"] == moves[recorded["turn"]]["root_sha256"]
        root = c.verified(root_item)
        assert root["manifest"] == parent_binding and "failure" not in root
        assert old.digest(root["snapshot"]) == moves[recorded["turn"]]["state_sha256"]
        assert root["candidates"]["actions"][root["choice"]] == moves[recorded["turn"]]["action"]
        assert row["id"] == f"{prefix}:{game['spec']['index']}:{recorded['turn']}"
        assert row["pair"] == f"{prefix}:{game['spec']['pair']}"
        assert row["split"] == previous.independent_splits()[row["pair"]]
        assert len(row["members"]) == len(root["candidates"]["keys"])
        if kind == "old":
            row = {**row, "teacher": row["labels"]["world128"]}
        assert row["teacher"] == root["choice"]
        target_rows.append({**row, "ranking_delta": delta(root)})
        root_items.append(root_item)
    destination = Path(base) / "data" / kind / f"game_{item['index']:04d}.pt"
    assert not destination.exists()
    old.tensor_save(destination, target_rows)
    exported = c.artifact(destination, rows=len(target_rows), kind=kind, index=item["index"])
    saved = torch.load(c.verify_hash(exported), map_location="cpu", weights_only=True)
    coverage = Counter()
    # Re-read saved targets and roots; reconstruct target arithmetic independently.
    for actual, original, root_item in zip(saved, source_rows, root_items, strict=True):
        c.budget_boundary(base)
        if kind == "old":
            original = {**original, "teacher": original["labels"]["world128"]}
        same_row({k: v for k, v in actual.items() if k != "ranking_delta"}, original)
        assert actual["split"] == previous.independent_splits()[actual["pair"]]
        expected = independent_delta(c.verified(root_item))
        assert actual["ranking_delta"] == expected
        coverage.update({"rows": 1, actual["split"]: 1, "short_support": int(len(actual["members"]) < 4),
                         "zero_delta": int(expected == 0.), "positive_delta": int(expected > 0.), "negative_delta": int(expected < 0.)})
    audit_path = Path(base) / "data" / "audits" / f"{kind}_{item['index']:04d}.json"
    old.save_new(audit_path, {"manifest": binding, "passed": True, "export": exported,
                            "parent_export": item, "parent_audit": audit_item, "source_game": game_item,
                            "source_roots": root_items, "coverage": dict(coverage), "independent_targets": True,
                            "source_tensor_verification": "Inherited fully reconstructed and unchanged parent exports",
                            "new_continuation_simulations": 0})
    return exported, c.artifact(audit_path)


def prepare_data(binding, base=OUTPUT):
    base = Path(base)
    assert not (base / "data_lock.json").exists()
    parent_item = c.artifact(SOURCE / "data_lock.json")
    parent_lock = c.verified(parent_item)
    assert parent_lock["manifest"] == PARENT_BINDING and parent_lock["config"] == previous.CONFIG
    parent_audit = c.verified(parent_lock["audit"])
    assert parent_audit["manifest"] == PARENT_BINDING and parent_audit["passed"]
    assert parent_lock["splits"] == previous.splits() == previous.independent_splits()
    for item in parent_lock["source_dependencies"]:
        c.verify_hash(item)
    exports, audits, coverage = [], [], Counter()
    for kind in ("old", "new"):
        parents, checked = parent_lock[f"{kind}_exports"], parent_audit[f"{kind}_export_audits"]
        assert len(parents) == len(checked) == (400 if kind == "old" else 402)
        for item, receipt in zip(parents, checked, strict=True):
            out, audit = export_shard(kind, item, receipt, binding, base)
            exports.append(out)
            audits.append(audit)
            coverage.update(c.verified(audit)["coverage"])
            if len(exports) % 40 == 0:
                print({"ranking_target_shards": len(exports), "total": 802}, flush=True)
    rows = [row for item in exports for row in torch.load(c.verify_hash(item), weights_only=True, map_location="cpu")]
    assert len(rows) == 23936 and len({r["id"] for r in rows}) == 23936
    assert Counter(r["split"] for r in rows) == {"train": 19047, "validation": 4889}
    training = [r for r in rows if r["split"] == "train"]
    numerator = sum(int(round(abs(r["ranking_delta"])*256)) for r in training)
    normalizer = sum(abs(r["ranking_delta"]) for r in training)/len(training)
    assert normalizer == numerator/(256*19047) and normalizer > 0
    old.save_new(base / "data_audit.json", {"manifest": binding, "passed": True, "parent_data_lock": parent_item,
                 "files": audits, "coverage": dict(coverage), "normalizer_numerator": numerator,
                 "normalizer_denominator": 256*19047, "normalizer_M": normalizer,
                 "preserved_row_order_and_splits": True, "independent_targets": True})
    old.save_new(base / "data_lock.json", {"manifest": binding, "parent_data_lock": parent_item,
                 "parent_data_audit": parent_lock["audit"], "exports": exports, "audit": c.artifact(base / "data_audit.json"),
                 "rows": len(rows), "train_rows": 19047, "validation_rows": 4889,
                 "splits": parent_lock["splits"], "normalizer_M": normalizer,
                 "ordered_row_ids_sha256": old.digest([r["id"] for r in rows])})
    return c.artifact(base / "data_lock.json")


def load_data(binding, base=OUTPUT):
    lock = old.read_json(Path(base) / "data_lock.json")
    assert lock["manifest"] == binding
    parent = c.verified(lock["parent_data_lock"])
    audit = c.verified(lock["audit"])
    assert parent["manifest"] == PARENT_BINDING and audit["manifest"] == binding and audit["passed"]
    assert lock["splits"] == previous.independent_splits() and len(lock["exports"]) == len(audit["files"]) == 802
    rows = []
    for item, checked in zip(lock["exports"], audit["files"], strict=True):
        receipt = c.verified(checked)
        assert receipt["manifest"] == binding and receipt["passed"] and receipt["export"] == item
        c.verify_hash(receipt["parent_export"])
        c.verify_hash(receipt["parent_audit"])
        values = torch.load(c.verify_hash(item), weights_only=True, map_location="cpu")
        assert len(values) == item["rows"]
        rows.extend(values)
    assert len(rows) == len({r["id"] for r in rows}) == 23936
    assert old.digest([r["id"] for r in rows]) == lock["ordered_row_ids_sha256"]
    assert all(r["split"] == lock["splits"][r["pair"]] for r in rows)
    assert Counter(r["split"] for r in rows) == {"train": 19047, "validation": 4889}
    normalizer = sum(abs(r["ranking_delta"]) for r in rows if r["split"] == "train")/19047
    assert normalizer == lock["normalizer_M"] == audit["normalizer_M"] and normalizer > 0
    return rows, normalizer
