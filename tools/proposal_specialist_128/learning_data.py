"""Immutable inherited corpus with independently checked exact-slot targets."""
from collections import Counter
from pathlib import Path

import torch

from tools.proposal_breadth_learning_128 import learning_data as inherited
from tools.proposal_breadth_learning_128 import learning as old_learning
from tools.proposal_ranking_128_recovery_01.reporting import StatusReporter
from . import common as c

ROOT, OUTPUT, PARENT = c.ROOT, c.OUTPUT, c.PARENT
old = c.old
CONFIG = {**inherited.CONFIG, "objective": "historical_CE_new_exact_first_three_outsider_CE",
          "empty_singleton_loss": "differentiable_zero_keep_rows_batches_steps"}
splits, independent_splits = inherited.splits, inherited.independent_splits
PARENT_RESULT_SHA = "7aba7182dfb585ddb183fa09e1a0ebcb13b61b3bf3fc6ea20639f1b0b2f75989"
PARENT_DATA_SHA = "da7f4df92cfc5cd247cdc32bf4d6fec026bb23235de22ec745b9031ec9a11fc1"


def closed_parent():
    """Verify closed provenance without invoking an expired ancestor runtime."""
    freeze = old.read_json(PARENT / "freeze.json")
    manifest = c.verified(freeze["manifest"])
    assert freeze["binding"] == c.PARENT_BINDING == old.digest(manifest)
    assert freeze["owner_authorized"]
    for path, sha in manifest["identity"]["dependencies"].items():
        c.verify_hash({"path": path, "sha256": sha})
    status = old.read_json(PARENT / "final_status.json")
    assert status["manifest"] == c.PARENT_BINDING and status["complete"] and status["all_prescribed_audits_pass"]
    assert status["result"]["sha256"] == PARENT_RESULT_SHA
    result, audit = c.verified(status["result"]), c.verified(status["audit"])
    assert result["manifest"] == audit["manifest"] == c.PARENT_BINDING and audit["passed"]
    assert audit["result"] == status["result"]
    assert result["learning"] == audit["learning"] == c.artifact(PARENT / "learning_complete.json")
    complete = old.read_json(PARENT / "learning_complete.json")
    assert complete["manifest"] == c.PARENT_BINDING and complete["data"]["sha256"] == PARENT_DATA_SHA
    lock = c.verified(complete["data"])
    data_audit = c.verified(lock["audit"])
    assert lock["manifest"] == data_audit["manifest"] == c.PARENT_BINDING and data_audit["passed"]
    assert lock["config"] == inherited.CONFIG and lock["splits"] == splits() == independent_splits()
    assert (lock["rows"], lock["train_rows"], lock["validation_rows"]) == (35886, 28508, 7378)
    assert data_audit["retained_tensors_targets_splits_unchanged"] and data_audit["independent_breadth_tensor_target_reconstruction"]
    models = c.verified(complete["models"])
    assert models["manifest"] == c.PARENT_BINDING and models["data_lock"] == complete["data"]
    assert models["training_config"] == old_learning.CONFIG and models["matched_dose_and_batches"]
    fits, fit_audits = c.verified(complete["fits"]), c.verified(complete["fit_audits"])
    assert models["fits"] == fits["files"] and models["audits"] == fit_audits["files"]
    return complete, lock, data_audit


def control_evidence(fit_id):
    complete, data, _ = closed_parent()
    fits, audits = c.verified(complete["fits"]), c.verified(complete["fit_audits"])
    assert fits["manifest"] == audits["manifest"] == c.PARENT_BINDING
    index = c.FITS.index(fit_id)
    item, checked = fits["files"][index], audits["files"][index]
    result, audit = c.verified(item), c.verified(checked)
    assert item["fit"] == checked["fit"] == result["fit"] == audit["fit"] == fit_id
    assert result["manifest"] == audit["manifest"] == c.PARENT_BINDING
    assert audit["result"] == c.artifact(c.verify_hash(item)) and audit["complete_fit_bitwise_identical"]
    assert audit["value_head_frozen"] and audit["equal_dose_and_batches"] and audit["common_broad_validation"]
    assert result["config"] == old_learning.CONFIG and result["data_lock"] == complete["data"]
    assert result["initializer"] == old_learning.previous.initializer(fit_id)
    assert result["seed"] == old_learning.TRAIN_SEEDS[fit_id]
    assert (result["updates"], result["row_exposures"], result["train_rows"], result["validation_rows"]) == (1784,114032,28508,7378)
    assert result["equal_dose_and_batches"] and result["batch_sequence_sha256"]["broad"] == result["batch_sequence_sha256"]["narrow"]
    assert [curve["update"] for curve in result["trajectories"]["broad"]] == audit["checkpoints"]
    for curve in result["trajectories"]["broad"]:
        c.verify_hash(curve["checkpoint"])
        assert c.verified(curve["validation"])["pair_loss"] == curve["pair_loss"]
    return {"result": item, "audit": checked, "original_selected_update": result["selected_updates"]["broad"]}, result


def source_evidence():
    complete, lock, audit = closed_parent()
    found = {}
    def add(item):
        c.verify_hash(item)
        value = {key:item[key] for key in ("path","sha256")}
        assert item["path"] not in found or found[item["path"]] == value
        found[item["path"]] = value
    for name in ("freeze.json", "manifest.json", "final_status.json", "final_audit.json", "result.json", "report.md", "learning_complete.json"):
        add(c.artifact(PARENT/name))
    for item in complete.values():
        if isinstance(item,dict) and "sha256" in item:
            add(item)
    add(lock["audit"])
    for item in [*lock["source_evidence"], *lock["exports"], *audit["breadth_audits"]]:
        add(item)
    for fit_id in c.FITS:
        evidence, result = control_evidence(fit_id)
        add(evidence["result"]); add(evidence["audit"]); add(result["initializer"])
        for curve in result["trajectories"]["broad"]:
            add(curve["checkpoint"]); add(curve["validation"])
    return [found[key] for key in sorted(found)]


def targets(root):
    inherited.targets(root)
    count = len(root["candidates"]["keys"])
    first = min(3,count)
    sums = [sum(world["outcomes"][i]["wdl"] for world in root["worlds"]) for i in range(first,count)]
    return {"outsider_start": first, "outsider_support": count-first,
            "outsider_teacher": first+sums.index(max(sums)) if sums else -1}


def independent_targets(root):
    inherited.independent_targets(root)
    count = len(root["candidates"]["actions"])
    indices = list(range(3,count))
    values = {index:sum(int(world["outcomes"][index]["wdl"]) for world in root["worlds"]) for index in indices}
    selected = max(indices,key=lambda index:values[index]) if indices else -1
    return {"outsider_start": min(count,3), "outsider_support": len(indices), "outsider_teacher": selected}


def augment_shard(item, checked, binding, base=OUTPUT):
    c.budget_boundary(base)
    receipt = c.verified(checked)
    assert receipt["manifest"] == c.PARENT_BINDING and receipt["passed"] and receipt["export"] == item
    assert receipt["actual_game_replayed"] and receipt["independent_targets"] and receipt["every_row_hidden_input_invariance"]
    values = torch.load(c.verify_hash(item),map_location="cpu",weights_only=True)
    assert len(values) == item["rows"] == len(receipt["source_roots"])
    rows, coverage = [], Counter()
    spec = receipt["source"]["spec"]
    for row, root_item in zip(values,receipt["source_roots"],strict=True):
        c.budget_boundary(base)
        root = c.verified(root_item)
        assert root["manifest"] == c.BREADTH_BINDING
        assert root["root_identity"] == ["main",spec["index"],root_item["turn"]]
        assert row["id"] == f"breadth128:{spec['index']}:{root_item['turn']}"
        assert row["pair"] == f"breadth128:{spec['pair']}" and row["split"] == independent_splits()[row["pair"]]
        assert len(row["members"]) == len(root["candidates"]["keys"])
        original = inherited.independent_targets(root)
        assert all(row[key] == value for key,value in original.items())
        new = targets(root)
        assert new == independent_targets(root)
        rows.append({**row,**new})
        coverage.update({"rows":1, row["split"]:1, f"outsiders_{new['outsider_support']}":1,
                         "teacher_in_first_three":int(row["teacher"] < 3)})
    path = Path(base)/"data/exports"/f"game_{item['index']:04d}.pt"
    assert not path.exists()
    old.tensor_save(path,rows)
    exported = c.artifact(path,index=item["index"],rows=len(rows))
    saved = torch.load(c.verify_hash(exported),map_location="cpu",weights_only=True)
    for before, after, rebuilt in zip(values,saved,rows,strict=True):
        assert set(after) == set(before) | {"outsider_start","outsider_support","outsider_teacher"}
        for key,value in before.items():
            if isinstance(value,torch.Tensor):
                assert torch.equal(value,after[key])
            elif key == "members":
                assert all(torch.equal(a,b) for a,b in zip(value,after[key],strict=True))
            else:
                assert value == after[key]
        assert all(after[key] == rebuilt[key] for key in ("outsider_start","outsider_support","outsider_teacher"))
    audit_path = Path(base)/"data/audits"/f"game_{item['index']:04d}.json"
    old.save_new(audit_path,{"manifest":binding,"passed":True,"export":exported,"parent_export":item,
        "parent_audit":checked,"source_roots":receipt["source_roots"],"coverage":dict(coverage),
        "inherited_tensors_splits_unchanged":True,"independent_outsider_targets":True,"continuations_replayed":0})
    return exported,c.artifact(audit_path,index=item["index"])


def prepare_data(binding, base=OUTPUT):
    c.budget_boundary(base)
    assert not (Path(base)/"data_lock.json").exists()
    complete,parent,parent_audit = closed_parent()
    evidence = source_evidence()
    rows = inherited.retained_rows(base)
    exports,audits,coverage = [],[],Counter()
    reporter = StatusReporter(Path(base)/"data",{"manifest":binding})
    for item, checked in zip(parent["exports"],parent_audit["breadth_audits"],strict=True):
        new,audit = augment_shard(item,checked,binding,base)
        exports.append(new); audits.append(audit)
        coverage.update(c.verified(audit)["coverage"])
        rows.extend(torch.load(c.verify_hash(new),map_location="cpu",weights_only=True))
        if len(exports) % 20 == 0:
            reporter.event("specialist_target_export",completed=len(exports),total=402)
    assert len(rows) == len({row["id"] for row in rows}) == 35886
    assert old.digest([row["id"] for row in rows]) == parent["ordered_row_ids_sha256"]
    assert Counter(row["split"] for row in rows) == {"train":28508,"validation":7378}
    assert coverage["rows"] == 11950
    audit_path = Path(base)/"data_audit.json"
    old.save_new(audit_path,{"manifest":binding,"passed":True,"parent_data_lock":complete["data"],
        "source_evidence":evidence,"breadth_audits":audits,"coverage":dict(coverage),
        "retained_rows":23936,"inherited_tensors_splits_unchanged":True,"independent_outsider_targets":True,
        "continuations_replayed":0,"historical_audits_inherited":True})
    old.save_new(Path(base)/"data_lock.json",{"manifest":binding,"config":CONFIG,"parent_data_lock":complete["data"],
        "source_evidence":evidence,"audit":c.artifact(audit_path),"exports":exports,"splits":splits(),
        "rows":35886,"train_rows":28508,"validation_rows":7378,"ordered_row_ids_sha256":parent["ordered_row_ids_sha256"]})
    return c.artifact(Path(base)/"data_lock.json")


def validate_data(binding, base=OUTPUT):
    c.budget_boundary(base)
    lock = old.read_json(Path(base)/"data_lock.json")
    assert lock["manifest"] == binding and lock["config"] == CONFIG
    assert lock["splits"] == splits() == independent_splits()
    assert lock["parent_data_lock"] == c.artifact(PARENT/"data_lock.json")
    assert lock["parent_data_lock"]["sha256"] == PARENT_DATA_SHA
    audit = c.verified(lock["audit"])
    assert audit["manifest"] == binding and audit["passed"] and audit["parent_data_lock"] == lock["parent_data_lock"]
    assert audit["inherited_tensors_splits_unchanged"] and audit["independent_outsider_targets"]
    assert lock["source_evidence"] == audit["source_evidence"]
    for item in lock["source_evidence"]:
        c.verify_hash(item)
    assert len(lock["exports"]) == len(audit["breadth_audits"]) == 402
    for index,(item,checked) in enumerate(zip(lock["exports"],audit["breadth_audits"],strict=True)):
        receipt = c.verified(checked)
        assert item["index"] == checked["index"] == index and receipt["export"] == item
        assert receipt["manifest"] == binding and receipt["passed"] and receipt["independent_outsider_targets"]
        c.verify_hash(item)
    assert (lock["rows"],lock["train_rows"],lock["validation_rows"]) == (35886,28508,7378)
    return lock


def load_data(binding,base=OUTPUT):
    lock = validate_data(binding,base)
    rows = inherited.retained_rows(base)
    for item in lock["exports"]:
        values = torch.load(c.verify_hash(item),map_location="cpu",weights_only=True)
        assert len(values) == item["rows"]
        rows.extend(values)
    assert len(rows) == len({row["id"] for row in rows}) == 35886
    assert old.digest([row["id"] for row in rows]) == lock["ordered_row_ids_sha256"]
    assert all(row["split"] == lock["splits"][row["pair"]] for row in rows)
    return rows
