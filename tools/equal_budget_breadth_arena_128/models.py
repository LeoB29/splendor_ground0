"""Reuse original161/035 models through the completed fresh uniform screen.

Only direct completion, panel and model receipts are authenticated. No network
is loaded here and no ancestor training, trajectory or root audit is repeated.
"""
from collections import Counter
from pathlib import Path

from . import common as c

ROOT, OUTPUT = c.ROOT, c.OUTPUT
SOURCE, SOURCE_BINDING = c.SOURCE, c.SOURCE_BINDING
ANCHORS = {
    161001: (563, "8d83d69b9691569f187beb93f2636635ab419c835442d02f0eb6fc511866fa62"),
    161002: (488, "b06e37d3640adb846de562daaaae4d526bf586a848a1e1783f2bc4d2c3899c96"),
    161003: (563, "8d4a258f625003e0fad9241560db0f606c23d1d8fab3a9c1afd3bb0c2f11a66a"),
}
RESULT_SHA = "99e7a00941de58d5364ca66219393f1380b134bce24402790c81780fb395292e"
AUDIT_SHA = "32b5104c7f1ef2b6b5ba1f73ffdc967a4bad4241e22175727813164e63172d60"
REPORT_SHA = "633c46cba82d0dc7504d2abe77ab8746e90151e22e28121be4d252df661336eb"
REGISTRY_SHA = "1454158d2221b4b1a99295b51b14243b647981702937e1b3aeb4b73293f204da"
SCOPE = "Three unchanged original161 anchors and frozen035 continuations only"
CLASSIFICATION = "positive_conditional_equal_budget_breadth"


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

    def retain(item):
        value = {key: item[key] for key in ("path", "sha256")}
        c.require(item["path"] not in items or items[item["path"]] == value,
                  "Conflicting source evidence identity")
        items[item["path"]] = value

    def read(item):
        value = c.verified(item)
        retain(item)
        return value

    def named(name):
        return c.artifact(SOURCE / name)

    freeze = read(named("freeze.json"))
    manifest = read(freeze["manifest"])
    c.require(freeze["binding"] == SOURCE_BINDING == c.old.digest(manifest)
              and freeze["owner_authorized"] is True, "Source frozen binding changed")
    status_item = named("final_status.json")
    status = read(status_item)
    c.require(status["manifest"] == SOURCE_BINDING and status["complete"] is True
              and status["all_prescribed_audits_pass"] is True
              and status["classification"] == CLASSIFICATION,
              "Source uniform study did not complete positively with its audits")
    for key, filename, expected in (("result", "result.json", RESULT_SHA),
                                    ("audit", "final_audit.json", AUDIT_SHA),
                                    ("report", "report.md", REPORT_SHA)):
        c.require(status[key] == {"path": named(filename)["path"], "sha256": expected},
                  f"Changed completed uniform {key} identity")
    result, audit = read(status["result"]), read(status["audit"])
    c.verify_hash(status["report"])
    retain(status["report"])
    c.require(result["manifest"] == audit["manifest"] == SOURCE_BINDING
              and result["all_prescribed_audits_pass"] is True and audit["passed"] is True
              and result["classification"] == CLASSIFICATION
              and audit["result"] == status["result"], "Source uniform completion/result linkage")
    statistics, independent = result["statistics"], audit["independent_statistics"]
    for value in (statistics, independent):
        c.require(value["classification"] == CLASSIFICATION and value["gate_pass"] is True
                  and value["anchor_practical_gain"] is True and value["anchor_superiority"] is True,
                  "Source uniform investment gate failed")
    c.require(statistics["states"] == statistics["source_games"] == 360
              and statistics["states_per_cell"] == 20 and statistics["fits"] == list(c.FITS)
              and statistics["cells"] == [[fit, seat, phase] for fit in c.FITS
                                           for seat in (0, 1) for phase in (0, 1, 2)],
              "Source uniform statistical panel differs")
    for contrast, value in statistics["contrasts"].items():
        c.require(independent["contrasts"][contrast] == {
            "mean": value["mean_score_difference"], "ci95": value["ci95"]},
            "Source uniform independent statistics differ")

    registry_item = named("models_lock.json")
    c.require(registry_item["sha256"] == REGISTRY_SHA
              and result["models"] == audit["models"] == registry_item, "Source uniform model seal link")
    registry = read(registry_item)
    c.require(registry["manifest"] == SOURCE_BINDING and registry["training"] is False
              and registry["ancestor_learning_replayed"] is False
              and registry["scope"] == "Three unchanged original161 anchors only",
              "Source uniform model registry binding/scope")
    anchors, continuation = anchor_items(), continuation_item()
    c.require(registry["models"] == anchors and registry["continuation_model"] == continuation,
              "Source anchor fit/update/hash mismatch")
    for item in [*anchors.values(), continuation]:
        retain(item)

    selection_item = named("selection_seal.json")
    c.require(result["selection"] == selection_item, "Source uniform selection seal link")
    selection_seal = read(selection_item)
    selection = read(selection_seal["selection"])
    c.require(selection_seal["manifest"] == selection["manifest"] == SOURCE_BINDING,
              "Source uniform selected panel binding")
    expected_cells = {(fit, seat, phase) for fit in c.FITS for seat in (0, 1) for phase in (0, 1, 2)}
    stages = {}
    for stage, count, per_cell in (("probe", 18, 1), ("main", 360, 20)):
        specs = selection[stage]
        cells = Counter((row["fit"], row["seat"], row["phase"]) for row in specs)
        c.require(len(specs) == count and [row["index"] for row in specs] == list(range(count))
                  and len({row["game_index"] for row in specs}) == count
                  and set(cells) == expected_cells and set(cells.values()) == {per_cell},
                  "Source uniform selected panel coverage")
        stage_item = named(f"{stage}_complete.json")
        c.require(audit["stages"][stage] == stage_item, "Source uniform stage completion link")
        value = read(stage_item)
        c.require(value["manifest"] == SOURCE_BINDING and value["all_prescribed_audits_pass"] is True
                  and value["states"] == value["coverage"]["states"] == count,
                  "Source uniform stage completion coverage")
        if stage == "main":
            c.require(result["coverage"] == value["coverage"], "Source uniform coverage linkage")
        for kind in ("collection", "audits"):
            c.require(value[kind] == named(f"{stage}_{kind}_lock.json"), "Source uniform stage receipt link")
            lock = read(value[kind])
            c.require(lock["manifest"] == SOURCE_BINDING and len(lock["files"]) == count
                      and [item["index"] for item in lock["files"]] == list(range(count)),
                      "Source uniform stage receipt identity coverage")
        stages[stage] = stage_item
    # The source registry's historical chain is already closed. Retain its
    # seal; do not walk historical training examples, root files or replays.
    return {"models": anchors, "continuation_model": continuation,
            "source_binding": SOURCE_BINDING, "source_status": status_item,
            "source_models": registry_item, "source_result": status["result"],
            "source_audit": status["audit"], "source_selection": selection_item,
            "source_stages": stages, "source_evidence": [items[key] for key in sorted(items)]}


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
    c.require(value["manifest"] == binding and value["training"] is False
              and value["ancestor_learning_replayed"] is False and value["scope"] == SCOPE,
              "Equal-budget arena model seal binding/scope")
    c.require(all(value[key] == item for key, item in expected.items()), "Changed arena model/provenance seal")
    c.require(set(value["models"]) == {f"anchor_{fit}" for fit in c.FITS}, "Unexpected model roles")
    return value
