"""Reuse the closed full-corpus experiment, changing only the learning rate.

Original tensors, splits, old labels and panel bytes remain immutable. Their
exhaustive original conversion and lawful-input audits are inherited through the
fixed completed full-fit result. No 4.9-GB raw-root reconversion or old fitting is
needed; every artifact consumed by the new experiment is hash-authenticated.
"""
from pathlib import Path

import torch

from tools.full_teacher_fit_128 import source as previous
from tools.full_teacher_fit_128 import common as previous_common
from . import common as c

LOWRATE = c.ROOT / "outputs/full_teacher_fit_128"
LOWRATE_BINDING = "7358b31b4aef99574197d80da1cd67cd6504b8bbb2e2ffc3441931bae2fbe3a3"
LOWRATE_RESULT_SHA = "5b3c2b471ef980b9eccb0e69130889683adaa57f423bd295f070ff86638b3e6d"
LOWRATE_AUDIT_SHA = "8553bebf5162f02c2b99a3a38610f05bf583d5ad0bd7d21cab8d4c7c9df419e8"
LOWRATE_DATA_SHA = "90be3dcea0a1f9f329f06b21f83e960aa9568ee38cae0b1dc6d2b82b48d3c6c2"
LOWRATE_MODELS = {
    161001: "87365e34cf84f965ea98567a9052381c5713405c1916649b2492b6c42b5c409e",
    161002: "3799bb8fc34158e34dc4f1671602a332a828b23b90e50865d4cc605bb2940235",
    161003: "5394dc9c0b2bb654af03cafdaa58c4df016740ec2f02c58e07d8f6170b7702b3",
}
# These mature pure checks do not call predecessor runtime or budget entry points.
CORPUS_COUNTS, ROW_FIELDS = previous.CORPUS_COUNTS, previous.ROW_FIELDS
ORIGINAL_FIELDS, NEW_FIELDS = previous.ORIGINAL_FIELDS, previous.NEW_FIELDS
SPECIALIST, SPECIALIST_BINDING, SPECIALIST_MODELS = previous.SPECIALIST, previous.SPECIALIST_BINDING, previous.SPECIALIST_MODELS
RESULT_SHA, AUDIT_SHA = previous.RESULT_SHA, previous.AUDIT_SHA
inherited_models, inherited_source = previous.inherited_models, previous.inherited_source
LegalActionAttentionScorerConfig = previous.LegalActionAttentionScorerConfig
validate_schedule, validate_row, validate_corpus = previous.validate_schedule, previous.validate_row, previous.validate_corpus
row_digest, old_targets, reconstruct_row = previous.row_digest, previous.old_targets, previous.reconstruct_row
_check_panel_identity, _plain = previous._check_panel_identity, previous._plain

RECIPE_FIELDS = (
    "train_rows", "validation_rows", "updates", "batch_size", "weight_decay",
    "gradient_norm", "value_head_frozen", "fresh_optimizer", "checkpoints",
    "checkpoint_selection", "historical_replay", "objective", "train_seeds",
    "batch_tag", "device", "dtype", "torch_threads", "selection_uses_intermediate_checkpoints",
)
DATA_FIELDS = (
    "schedule", "files", "row_digests", "counts", "panel_file", "panel_digest",
    "rows", "train_rows", "validation_rows", "original_train_probe_ids",
    "assessment_in_payload", "historical_replay", "independent_lawful_inputs_audit_inherited",
    "source_trace_membership_checked", "new_old_target_arithmetic_checked",
    "new_simulations", "historical_fit_replays",
)


def validate_recipe(low_config, high_config=None):
    high = c.CONFIG if high_config is None else high_config
    c.require(low_config["lr"] == 1e-5 and high["lr"] == 1e-4,
              "Only the prespecified 10x learning-rate contrast is authorized")
    c.require(all(low_config[key] == high[key] for key in RECIPE_FIELDS),
              "Fitting recipe differs beyond learning rate")
    return True


def closed_parent():
    """Authenticate immutable reuse and exact low-rate controls, without replay."""
    refs = {}
    def add(ref, read=True):
        item = _plain(ref)
        c.require(item["path"] not in refs or refs[item["path"]] == item,
                  "Conflicting inherited artifact identities")
        if item["path"] in refs and not read:
            return c.ROOT/item["path"]
        refs[item["path"]] = item
        return c.verified(item) if read else c.verify_hash(item)

    freeze = add(c.artifact(LOWRATE/"freeze.json"))
    manifest = add(freeze["manifest"])
    c.require(freeze["owner_authorized"] is True
              and freeze["binding"] == LOWRATE_BINDING == c.old.digest(manifest), "Low-rate source freeze changed")
    c.require(manifest["identity"]["config"] == previous_common.CONFIG,
              "Low-rate source recipe changed")
    validate_recipe(manifest["identity"]["config"])
    for path, sha in manifest["identity"]["dependencies"].items():
        add({"path": path, "sha256": sha}, False)
    checks = add(manifest["checks"])
    c.require(checks["exit_code"] == 0 and checks["identity"] == manifest["identity"],
              "Low-rate source checks do not match freeze")
    for ref in checks["logs"]:
        add(ref, False)
    status_ref = c.artifact(LOWRATE/"final_status.json")
    status = add(status_ref)
    c.require(status["manifest"] == LOWRATE_BINDING and status["complete"] is True
              and status["all_prescribed_audits_pass"] is True
              and status["result"]["sha256"] == LOWRATE_RESULT_SHA
              and status["audit"]["sha256"] == LOWRATE_AUDIT_SHA, "Low-rate source is not exact completed experiment")
    result, audit = add(status["result"]), add(status["audit"])
    add(status["report"], False)
    c.require(result["manifest"] == audit["manifest"] == LOWRATE_BINDING
              and result["all_prescribed_audits_pass"] is True and audit["passed"] is True
              and audit["result"] == status["result"] and audit["training_reproduced"] is True
              and result["assessment_audit"] == audit["assessment"], "Low-rate completion linkage changed")
    c.require(result["data"]["sha256"] == LOWRATE_DATA_SHA, "Low-rate exact data lock changed")
    data = add(result["data"])
    c.require(data["manifest"] == LOWRATE_BINDING and data["parent"] == manifest["source_parent"]
              and (data["rows"], data["train_rows"], data["validation_rows"]) == (11950, 9461, 2489)
              and data["assessment_in_payload"] is False and data["historical_replay"] is False
              and data["independent_lawful_inputs_audit_inherited"] is True
              and data["source_trace_membership_checked"] is True
              and data["new_old_target_arithmetic_checked"] is True,
              "Original full-corpus data/audit identity changed")
    original = data["parent"]
    c.require(data["schedule"] == original["schedule"]
              and original["source_binding"] == c.SOURCE_BINDING
              and original["assessment_used_for_selection"] is False,
              "Original source/panel identity changed")
    validate_schedule(data["schedule"])
    # Assessment retains the original teacher/BROAD/NARROW context result.
    # Bind that directly accessed completion path as well as its sealed result.
    context_status_ref = c.artifact(c.SOURCE/"final_status.json")
    c.require(context_status_ref == original["completion"], "Original context status identity changed")
    context_status = add(context_status_ref)
    c.require(context_status["manifest"] == c.SOURCE_BINDING and context_status["complete"] is True
              and context_status["all_prescribed_audits_pass"] is True
              and context_status["result"] == original["result"]
              and context_status["audit"] == original["audit"]
              and context_status["result"]["sha256"] == RESULT_SHA
              and context_status["audit"]["sha256"] == AUDIT_SHA,
              "Original context completion changed")
    context, context_audit = add(context_status["result"]), add(context_status["audit"])
    c.require(context["manifest"] == context_audit["manifest"] == c.SOURCE_BINDING
              and context["all_prescribed_audits_pass"] is True and context_audit["passed"] is True
              and context_audit["result"] == context_status["result"], "Original context audit changed")
    add(context_status["report"], False)
    for ref in [*data["files"].values(), data["panel_file"]]:
        add(ref, False)
    # The closed full-fit result seals the complete older 18,958-reference
    # ancestry. Reauthenticate files needed here; do not reopen unused raw roots.
    for descriptor in original["evaluation_sources"]:
        add(descriptor["selection"], False)
        add(descriptor["root"], False)
        add(descriptor["audit"], False)
    models = dict(original["models"])
    for ref in models.values():
        add(ref, False)
    registry = add(result["models"])
    c.require(registry["manifest"] == LOWRATE_BINDING and registry["fixed_final"] is True
              and registry["all_fit_audits_pass"] is True
              and registry["audits"] == audit["fits"]
              and set(registry["models"]) == {f"focused_{fit}" for fit in c.FITS}
              and [ref["fit"] for ref in registry["fits"]] == list(c.FITS)
              == [ref["fit"] for ref in registry["audits"]], "Low-rate registry incomplete")
    fits = {}
    for fit, fit_ref, audit_ref in zip(c.FITS, registry["fits"], registry["audits"], strict=True):
        trained, checked = add(fit_ref), add(audit_ref)
        model = registry["models"][f"focused_{fit}"]
        validate_recipe(trained["config"])
        c.require(trained["manifest"] == checked["manifest"] == LOWRATE_BINDING
                  and trained["config"] == manifest["identity"]["config"]
                  and trained["fit"] == checked["fit"] == fit and checked["passed"] is True
                  and checked["result"] == _plain(fit_ref)
                  and checked["complete_fit_bitwise_identical"] is True
                  and checked["all_training_metrics_independently_reproduced"] is True
                  and checked["value_head_frozen"] is True
                  and trained["initializer"] == models[f"anchor_{fit}"]
                  and trained["selected_update"] == checked["selected_update"] == model["selected_update"] == 1784
                  and trained["final_always_selected"] is checked["final_always_selected"] is True
                  and trained["updates"] == 1784 and trained["train_rows"] == 9461
                  and trained["row_exposures"] == checked["row_exposures"] == 114044
                  and trained["batch_sequence_sha256"] == checked["batch_sequence_sha256"]
                  and trained["per_row_exposures"] == checked["per_row_exposures"]
                  and trained["validation_read"] is checked["validation_read"] is False
                  and trained["assessment_read"] is checked["assessment_read"] is False
                  and trained["selected_model"] == checked["selected_model"] == _plain(model)
                  and model["source_fit"] == fit and model["sha256"] == LOWRATE_MODELS[fit],
                  "Low-rate fit/model/reproduction identity changed")
        c.require([point["update"] for point in trained["checkpoints"]] == previous_common.CONFIG["checkpoints"],
                  "Low-rate diagnostic checkpoints changed")
        for point in trained["checkpoints"]:
            add(point["checkpoint"], False)
            add(point["training"], False)
        c.require(trained["checkpoints"][-1]["checkpoint"] == trained["selected_model"]
                  and trained["checkpoints"][-1]["training"] == trained["selected_training"]
                  == checked["selected_training"], "Low-rate final selection changed")
        models[f"lowrate_{fit}"] = model
        fits[str(fit)] = {"result": fit_ref, "audit": audit_ref}

    # Reading the old audit's completion metadata does not select a new model or
    # materialize any continuation matrix. Old assessment arrays are hash-only.
    assessment_audit = add(result["assessment_audit"])
    c.require(assessment_audit["manifest"] == LOWRATE_BINDING and assessment_audit["passed"] is True
              and assessment_audit["result"] == result["assessment"]
              and assessment_audit["models"] == result["models"]
              and assessment_audit["states"] == 336 and assessment_audit["unknown_outcomes"] == 0
              and assessment_audit["all_new_arithmetic_pass"] is True
              and assessment_audit["full_validation_old_target_metrics_independently_reproduced"] is True,
              "Low-rate assessment closure changed")
    evidence = {"assessment": result["assessment"], "predictions": assessment_audit["predictions"],
                "validation_old_targets": result["validation_old_targets"], "models": result["models"]}
    for ref in evidence.values():
        add(ref, False)
    return {"source_binding": c.SOURCE_BINDING, "lowrate_binding": LOWRATE_BINDING,
            "schedule": original["schedule"], "corpus": original["corpus"],
            "model_config": original["model_config"], "models": models,
            "evaluation_sources": original["evaluation_sources"],
            "reused_data": result["data"], "lowrate_fits": fits, "lowrate_evidence": evidence,
            "completion": status_ref, "result": status["result"], "audit": status["audit"],
            "evidence": [refs[path] for path in sorted(refs)],
            "inventory": {path: refs[path]["sha256"] for path in sorted(refs)},
            "new_simulations": 0, "historical_fit_replays": 0,
            "new_target_reconstructions": 0, "assessment_used_for_selection": False,
            "original_full_conversion_audit_inherited": True,
            "inherited_coverage": original["inherited_coverage"]}


def prepare(binding, base=c.OUTPUT, parent=None):
    base = Path(base)
    if parent is None:
        parent = closed_parent()
    else:
        freeze = c.old.read_json(base/"freeze.json")
        manifest = c.verified(freeze["manifest"])
        c.require(freeze["binding"] == binding == c.old.digest(manifest)
                  and manifest["source_parent"] == parent, "Supplied source differs from new freeze")
    original = c.verified(parent["reused_data"])
    value = {key: original[key] for key in DATA_FIELDS}
    value.update(manifest=binding, parent=parent, inherited_data=parent["reused_data"],
                 inherited_conversion_audit=True, new_target_reconstructions=0, tensor_bytes_reused=True)
    c.old.save_new(base/"data_lock.json", value)
    return c.artifact(base/"data_lock.json")


def _lock(binding, base):
    base = Path(base)
    value = c.old.read_json(base/"data_lock.json")
    c.require(value["manifest"] == binding and value["assessment_in_payload"] is False
              and value["historical_replay"] is False and value["new_simulations"] == 0
              and (value["rows"], value["train_rows"], value["validation_rows"]) == (11950, 9461, 2489),
              "Invalid focused data seal")
    freeze = base/"freeze.json"
    if freeze.exists():
        sealed = c.old.read_json(freeze)
        manifest = c.verified(sealed["manifest"])
        c.require(sealed["binding"] == binding == c.old.digest(manifest)
                  and manifest["source_parent"] == value["parent"], "Focused source differs from experiment freeze")
    else:
        c.require(not base.resolve().is_relative_to(c.OUTPUT.resolve()), "Production source lacks freeze")
    c.require(value["schedule"] == value["parent"]["schedule"], "Changed source schedule")
    c.require(value["inherited_data"] == value["parent"]["reused_data"]
              and value["inherited_data"]["sha256"] == LOWRATE_DATA_SHA
              and value["inherited_conversion_audit"] is True
              and value["tensor_bytes_reused"] is True
              and value["new_target_reconstructions"] == 0, "Changed immutable full-corpus reuse")
    original = c.verified(value["inherited_data"])
    c.require(original["manifest"] == LOWRATE_BINDING
              and all(value[key] == original[key] for key in DATA_FIELDS),
              "Higher-rate tensor identities differ from completed low-rate data")
    return value


def load_rows(binding, split, base=c.OUTPUT):
    """Read only the requested old-target tensor partition, never another split."""
    c.require(split in ("train", "validation"), "Unknown fitting split")
    lock = _lock(binding, base)
    rows = torch.load(c.verify_hash(lock["files"][split]), map_location="cpu", weights_only=True)
    expected = []
    for shard in lock["parent"]["corpus"]["shards"]:
        spec = shard["spec"]
        if lock["parent"]["corpus"]["splits"][str(spec["pair"])] == split:
            expected.extend((f"breadth128:{spec['index']}:{root['turn']}", spec["fit"], f"breadth128:{spec['pair']}")
                            for root in shard["roots"])
    c.require(len(rows) == len(expected) == CORPUS_COUNTS[split]
              and row_digest(rows) == lock["row_digests"][split], "Full tensor population changed")
    for row, identity in zip(rows, expected, strict=True):
        validate_row(row)
        c.require(row["split"] == split and (row["id"], row["source_fit"], row["pair"]) == identity,
                  "Full row changed identity or split")
    return rows


def load_panel_rows(binding, base=c.OUTPUT):
    lock = _lock(binding, base)
    rows = torch.load(c.verify_hash(lock["panel_file"]), map_location="cpu", weights_only=True)
    c.require(len(rows) == 336 and row_digest(rows) == lock["panel_digest"], "Assessment panel tensors changed")
    for row, spec in zip(rows, lock["schedule"], strict=True):
        validate_row(row)
        _check_panel_identity(row, spec)
    return rows


def train_probe_ids(binding, base=c.OUTPUT):
    """Read metadata only; never open VALIDATION or assessment tensors."""
    lock = _lock(binding, base)
    ids = [spec["source_rowid"] for spec in lock["schedule"] if spec["split"] == "train"]
    c.require(len(ids) == len(set(ids)) == 168 and ids == lock["original_train_probe_ids"],
              "Original TRAIN probe membership changed")
    return ids


def control_models(binding, base=c.OUTPUT):
    parent = _lock(binding, base)["parent"]
    return {f"{family}_{fit}": parent["models"][f"{family}_{fit}"]
            for family in ("anchor", "specialist", "lowrate") for fit in c.FITS}


def initializer(fit, base=c.OUTPUT):
    c.require(fit in c.FITS, "Unknown initializer fit")
    lock = c.old.read_json(Path(base)/"data_lock.json")
    ref = lock["parent"]["models"][f"anchor_{fit}"]
    c.require(ref["sha256"] == inherited_models.EXPECTED[f"anchor_{fit}"]
              and ref["source_fit"] == fit, "Original initializer differs")
    c.verify_hash(ref)
    return ref


def evaluation_sources(binding, base=c.OUTPUT):
    return _lock(binding, base)["parent"]["evaluation_sources"]


def old_inputs(binding, base=c.OUTPUT):
    """Expose only the original pre-assessment record for locking new predictions."""
    result = []
    for descriptor in evaluation_sources(binding, base):
        saved = c.verified(descriptor["selection"])
        identity, locked = saved["identity"], saved["selection_lock"]
        c.require(saved["manifest"] == c.SOURCE_BINDING and saved["stage"] == "main"
                  and saved["index"] == descriptor["index"]
                  and identity["spec"] == descriptor["spec"]
                  and locked["identity_digest"] == c.old.digest(identity)
                  and locked["old_selection_sha256"] == c.old.digest(saved["old_selection"]),
                  "Changed pre-assessment inputs")
        statistics = locked["selection_statistics"]
        result.append({"index": descriptor["index"], "spec": descriptor["spec"],
                       "snapshot": identity["snapshot"], "candidates": identity["candidates"],
                       "old_sums_wdl": statistics["sums_wdl"],
                       "old_unknown_counts": statistics["unknown_counts"],
                       "old_choices": locked["choices"], "selection_artifact": descriptor["selection"]})
    c.require([item["index"] for item in result] == list(range(336)), "Incomplete old input panel")
    return result


def low_rate_fit(fit_id, binding, base=c.OUTPUT):
    c.require(fit_id in c.FITS, "Unknown low-rate fit")
    return _lock(binding, base)["parent"]["lowrate_fits"][str(fit_id)]["result"]


def low_rate_audit(fit_id, binding, base=c.OUTPUT):
    c.require(fit_id in c.FITS, "Unknown low-rate fit")
    return _lock(binding, base)["parent"]["lowrate_fits"][str(fit_id)]["audit"]


def lowrate_evidence(binding, base=c.OUTPUT):
    return _lock(binding, base)["parent"]["lowrate_evidence"]


def validate(binding, base=c.OUTPUT, authenticate_parent=True):
    """Check reused bytes/splits; callers may authenticate bound evidence once."""
    lock = _lock(binding, base)
    parent = closed_parent() if authenticate_parent else lock["parent"]
    c.require(parent == lock["parent"], "Higher-rate source ancestry changed")
    rows = [*load_rows(binding, "train", base), *load_rows(binding, "validation", base)]
    rows.sort(key=lambda row: row["index"])
    validate_corpus(rows, parent)
    by_id = {row["id"]: row for row in rows}
    expected = [{**by_id[spec["source_rowid"]], "index": spec["index"]} for spec in lock["schedule"]]
    c.require(row_digest(expected) == row_digest(load_panel_rows(binding, base)),
              "Reused panel differs from exact full-corpus rows")
    train_probe_ids(binding, base)
    return lock
