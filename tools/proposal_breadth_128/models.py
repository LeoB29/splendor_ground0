"""Seal the three original161 anchors through the completed ranking provenance.

This is a read-only identity check of existing models and completion receipts.
It neither replays ancestor learning nor loads the old training corpus/models.
"""
from pathlib import Path

from . import common as c

ROOT, OUTPUT = c.ROOT, c.OUTPUT
SOURCE, SOURCE_BINDING = c.SOURCE, c.SOURCE_BINDING
RECOVERY_BINDING = "ec057b887261f27f1a57f625c5df4581aa0420b0c5a6067ae3fd82cd254b8493"
ANCHORS = {
    161001: (563, "8d83d69b9691569f187beb93f2636635ab419c835442d02f0eb6fc511866fa62"),
    161002: (488, "b06e37d3640adb846de562daaaae4d526bf586a848a1e1783f2bc4d2c3899c96"),
    161003: (563, "8d4a258f625003e0fad9241560db0f606c23d1d8fab3a9c1afd3bb0c2f11a66a"),
}


def anchor_items():
    result = {}
    for fit, (update, expected_sha) in ANCHORS.items():
        path = ROOT / "outputs/teacher_budget_161/fits" / f"world128_{fit}" / f"checkpoint_{update:04d}.pt"
        item = c.artifact(path, source_fit=fit, selected_update=update, unchanged_from_anchor=True)
        c.require(item["sha256"] == expected_sha, "Original161 anchor checkpoint changed")
        result[f"anchor_{fit}"] = item
    c.require(tuple(ANCHORS) == c.FITS, "Unexpected model fit population")
    return result


def continuation_item():
    item = c.artifact(ROOT / c.old.CHECKPOINT)
    c.require(item["sha256"] == c.old.CHECKPOINT_SHA, "Frozen035 continuation checkpoint changed")
    return item


def _closed_parent():
    items = {}

    def read(item):
        value = c.verified(item)
        items[item["path"]] = {key: item[key] for key in ("path", "sha256")}
        return value

    def named(name):
        return c.artifact(SOURCE / name)

    freeze = read(named("freeze.json"))
    manifest = read(freeze["manifest"])
    c.require(freeze["binding"] == SOURCE_BINDING == c.old.digest(manifest)
              and freeze["owner_authorized"], "Source frozen binding changed")
    status_item = named("final_status.json")
    status = read(status_item)
    c.require(status["manifest"] == SOURCE_BINDING and status["complete"]
              and status["all_prescribed_audits_pass"], "Source ranking study did not complete its audits")
    recovery_freeze = read(named("recovery_01/freeze.json"))
    recovery_manifest = read(recovery_freeze["manifest"])
    c.require(recovery_freeze["binding"] == RECOVERY_BINDING == c.old.digest(recovery_manifest)
              and recovery_manifest["original_binding"] == SOURCE_BINDING
              and recovery_manifest["owner_authorized"], "Source recovery frozen binding changed")
    recovery = read(named("recovery_01/closure.json"))
    c.require(recovery["manifest"] == SOURCE_BINDING and recovery["complete"]
              and recovery["recovery_binding"] == RECOVERY_BINDING
              and recovery["scientific_status"] == status_item, "Source recovery completion link")
    read(recovery["preserved_original_failure"])
    audit, result = read(status["audit"]), read(status["result"])
    c.require(audit["manifest"] == result["manifest"] == SOURCE_BINDING and audit["passed"]
              and result["all_prescribed_audits_pass"] and audit["result"] == status["result"],
              "Source completion/result linkage")
    learning_item = named("learning_complete.json")
    c.require(audit["learning"] == result["learning"] == learning_item, "Source learning completion link")
    learning = read(learning_item)
    registry_item = named("models_lock.json")
    c.require(learning["manifest"] == SOURCE_BINDING and learning["models"] == registry_item,
              "Source learning/model seal linkage")
    registry = read(registry_item)
    c.require(registry["manifest"] == SOURCE_BINDING, "Source model registry binding")
    # Bind the learning completion's direct receipts without rerunning their work.
    for key in ("data", "fits", "fit_audits"):
        receipt = read(learning[key])
        c.require(receipt["manifest"] == SOURCE_BINDING, "Source learning receipt binding")
    anchors = anchor_items()
    for key, item in anchors.items():
        c.require(registry["models"][key] == item, "Source anchor fit/update/hash mismatch")
        items[item["path"]] = {field: item[field] for field in ("path", "sha256")}
    continuation = continuation_item()
    items[continuation["path"]] = continuation
    return {"models": anchors, "continuation_model": continuation, "source_binding": SOURCE_BINDING,
            "source_status": status_item, "source_models": registry_item,
            "source_learning": learning_item,
            "source_evidence": [items[key] for key in sorted(items)]}


def source_evidence():
    return _closed_parent()["source_evidence"]


def prepare(binding, base=OUTPUT):
    c.budget_boundary(base)
    path = Path(base) / "models_lock.json"
    c.require(not path.exists(), "One-shot anchor registry; preserve existing seal")
    value = {"manifest": binding, **_closed_parent(), "training": False,
             "ancestor_learning_replayed": False, "scope": "Three unchanged original161 anchors only"}
    c.old.save_new(path, value)
    validate(binding, base)
    return c.artifact(path)


def validate(binding, base=OUTPUT):
    c.budget_boundary(base)
    value = c.old.read_json(Path(base) / "models_lock.json")
    expected = _closed_parent()
    c.require(value["manifest"] == binding and not value["training"]
              and not value["ancestor_learning_replayed"], "Breadth model seal binding/scope")
    c.require(value["scope"] == "Three unchanged original161 anchors only", "Wrong model scope")
    c.require(all(value[key] == item for key, item in expected.items()), "Changed breadth model/provenance seal")
    c.require(set(value["models"]) == {f"anchor_{fit}" for fit in c.FITS}, "Unexpected model roles")
    return value
