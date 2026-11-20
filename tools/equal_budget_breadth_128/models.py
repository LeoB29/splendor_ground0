"""Seal original161/035 models through the closed breadth study only.

This authenticates direct completion and model receipts without loading a
network, walking historical training corpora or repeating ancestor audits.
"""
from pathlib import Path

from . import common as c

ROOT, OUTPUT = c.ROOT, c.OUTPUT
SOURCE, SOURCE_BINDING = c.SOURCE, c.SOURCE_BINDING
SOURCE_RESULT_SHA = "2592d17b5ed5f1ca2a7f463ffa27b9e803db3600b0b3f62ea177aab26e2a2570"
SOURCE_AUDIT_SHA = "f9273a1612636b4c13be6e47768a8ba05a0159f3b176aeddea5aca076faabf55"
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
        minimal = {key: item[key] for key in ("path", "sha256")}
        c.require(item["path"] not in items or items[item["path"]] == minimal,
                  "Conflicting source artifact hashes")
        items[item["path"]] = minimal
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
              and status["all_prescribed_audits_pass"], "Source breadth study did not complete its audits")
    c.require(status["result"]["sha256"] == SOURCE_RESULT_SHA
              and status["audit"]["sha256"] == SOURCE_AUDIT_SHA,
              "Published source completion changed")
    audit, result = read(status["audit"]), read(status["result"])
    c.require(audit["manifest"] == result["manifest"] == SOURCE_BINDING and audit["passed"]
              and result["all_prescribed_audits_pass"] and audit["result"] == status["result"],
              "Source completion/result linkage")
    registry_item = named("models_lock.json")
    c.require(audit["models"] == result["models"] == registry_item,
              "Source completion/model seal linkage")
    registry = read(registry_item)
    c.require(registry["manifest"] == SOURCE_BINDING and registry["training"] is False
              and registry["ancestor_learning_replayed"] is False,
              "Source model registry binding/scope")
    anchors = anchor_items()
    c.require(set(registry["models"]) == set(anchors), "Source anchor model population changed")
    for key, item in anchors.items():
        c.require(registry["models"][key] == item, "Source anchor fit/update/hash mismatch")
        items[item["path"]] = {field: item[field] for field in ("path", "sha256")}
    continuation = continuation_item()
    c.require(registry["continuation_model"] == continuation,
              "Source continuation checkpoint identity mismatch")
    items[continuation["path"]] = continuation
    # This registry's own historical source_evidence is already closed. Do not
    # expand it, reopen old fits or replay simulation merely to reuse the models.
    return {"models": anchors, "continuation_model": continuation,
            "source_binding": SOURCE_BINDING, "source_status": status_item,
            "source_models": registry_item, "source_result": status["result"],
            "source_audit": status["audit"],
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
              and not value["ancestor_learning_replayed"], "Equal-budget model seal binding/scope")
    c.require(value["scope"] == "Three unchanged original161 anchors only", "Wrong model scope")
    c.require(all(value[key] == item for key, item in expected.items()), "Changed model/provenance seal")
    c.require(set(value["models"]) == {f"anchor_{fit}" for fit in c.FITS}, "Unexpected model roles")
    return value
