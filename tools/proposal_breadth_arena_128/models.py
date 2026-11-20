"""Anchor-only model seal through the completed, positive breadth screen.

Checks bounded completion receipts and four existing model binaries. It never
loads ancestor training examples or replays an old learning or search operation.
"""
from pathlib import Path

from tools.proposal_breadth_128 import models as parent
from . import common as c

ROOT, OUTPUT = c.ROOT, c.OUTPUT
SOURCE, SOURCE_BINDING = c.SOURCE, c.SOURCE_BINDING
ANCHORS = parent.ANCHORS
RESULT_SHA = "2592d17b5ed5f1ca2a7f463ffa27b9e803db3600b0b3f62ea177aab26e2a2570"
AUDIT_SHA = "f9273a1612636b4c13be6e47768a8ba05a0159f3b176aeddea5aca076faabf55"
REGISTRY_SHA = "ecf3df3d421985443eab86c59d5cea6d066c8e4dfe30f0423cdacdae17fac117"
SCOPE = "Three unchanged original161 anchors and frozen035 continuations only"
CLASSIFICATION = "positive_conditional_breadth_contribution"


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
    # This mature helper has no expired ancestor execution-budget checks.
    ancestor = parent._closed_parent()
    items = {item["path"]: item for item in ancestor["source_evidence"]}

    def named(name):
        return c.artifact(SOURCE / name)

    def retain(item):
        value = {key: item[key] for key in ("path", "sha256")}
        c.require(item["path"] not in items or items[item["path"]] == value,
                  "Conflicting source evidence identity")
        items[item["path"]] = value

    def read(item):
        value = c.verified(item)
        retain(item)
        return value

    freeze = read(named("freeze.json"))
    manifest = read(freeze["manifest"])
    c.require(freeze["binding"] == SOURCE_BINDING == c.old.digest(manifest)
              and freeze["owner_authorized"], "Source breadth frozen binding changed")
    status_item = named("final_status.json")
    status = read(status_item)
    c.require(status["manifest"] == SOURCE_BINDING and status["complete"]
              and status["all_prescribed_audits_pass"]
              and status["classification"] == CLASSIFICATION,
              "Source breadth study did not complete positively with its audits")
    c.require(status["result"] == {"path": c.artifact(SOURCE / "result.json")["path"], "sha256": RESULT_SHA}
              and status["audit"] == {"path": c.artifact(SOURCE / "final_audit.json")["path"], "sha256": AUDIT_SHA},
              "Changed completed breadth result/audit identity")
    result, audit = read(status["result"]), read(status["audit"])
    c.require(result["manifest"] == audit["manifest"] == SOURCE_BINDING
              and result["all_prescribed_audits_pass"] and audit["passed"]
              and result["classification"] == CLASSIFICATION
              and audit["result"] == status["result"], "Source breadth completion/result linkage")
    c.require(audit["independent_statistics"]["classification"] == CLASSIFICATION
              and audit["independent_statistics"]["gate_pass"], "Source breadth investment gate failed")
    c.require(status["report"] == named("report.md"), "Source breadth report linkage")
    c.verify_hash(status["report"])
    retain(status["report"])

    registry_item = named("models_lock.json")
    c.require(registry_item["sha256"] == REGISTRY_SHA
              and result["models"] == audit["models"] == registry_item, "Source breadth model seal link")
    registry = read(registry_item)
    c.require(registry["manifest"] == SOURCE_BINDING and not registry["training"]
              and not registry["ancestor_learning_replayed"]
              and registry["scope"] == "Three unchanged original161 anchors only"
              and all(registry[key] == value for key, value in ancestor.items()),
              "Source breadth/ranking model provenance mismatch")
    anchors, continuation = anchor_items(), continuation_item()
    c.require(registry["models"] == anchors and registry["continuation_model"] == continuation,
              "Source anchor fit/update/hash mismatch")
    for item in [*anchors.values(), continuation]:
        retain(item)

    stages = {}
    for stage, expected_states in (("probe", 18), ("main", 360)):
        stage_item = named(f"{stage}_complete.json")
        c.require(audit["stages"][stage] == stage_item, "Source breadth stage completion link")
        value = read(stage_item)
        c.require(value["manifest"] == SOURCE_BINDING and value["all_prescribed_audits_pass"]
                  and value["states"] == value["coverage"]["states"] == expected_states,
                  "Source breadth stage completion coverage")
        for kind in ("collection", "audits"):
            c.require(value[kind] == named(f"{stage}_{kind}_lock.json"), "Source breadth stage receipt link")
            lock = read(value[kind])
            c.require(lock["manifest"] == SOURCE_BINDING and len(lock["files"]) == expected_states
                      and {item["index"] for item in lock["files"]} == set(range(expected_states)),
                      "Source breadth stage receipt identity coverage")
        stages[stage] = stage_item
    return {"models": anchors, "continuation_model": continuation,
            "source_binding": SOURCE_BINDING, "source_status": status_item,
            "source_models": registry_item, "source_stages": stages,
            "source_evidence": [items[key] for key in sorted(items)]}


def source_evidence():
    return _closed_parent()["source_evidence"]


def prepare(binding, base=OUTPUT):
    c.budget_boundary(base)
    path = Path(base) / "models_lock.json"
    c.require(not path.exists(), "One-shot anchor registry; preserve existing seal")
    c.old.save_new(path, {"manifest": binding, **_closed_parent(), "training": False,
                          "ancestor_learning_replayed": False, "scope": SCOPE})
    validate(binding, base)
    return c.artifact(path)


def validate(binding, base=OUTPUT):
    c.budget_boundary(base)
    value = c.old.read_json(Path(base) / "models_lock.json")
    expected = _closed_parent()
    c.require(value["manifest"] == binding and not value["training"]
              and not value["ancestor_learning_replayed"] and value["scope"] == SCOPE,
              "Breadth arena model seal binding/scope")
    c.require(all(value[key] == item for key, item in expected.items()), "Changed breadth arena model/provenance seal")
    c.require(set(value["models"]) == {f"anchor_{fit}" for fit in c.FITS}, "Unexpected model roles")
    return value
