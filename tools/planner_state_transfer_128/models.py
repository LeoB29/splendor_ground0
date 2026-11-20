"""Authenticate the original nine checkpoints through their closed learning audit."""
from pathlib import Path
from . import common as c

EXPECTED = {
    "anchor_161001": "8d83d69b9691569f187beb93f2636635ab419c835442d02f0eb6fc511866fa62",
    "anchor_161002": "b06e37d3640adb846de562daaaae4d526bf586a848a1e1783f2bc4d2c3899c96",
    "anchor_161003": "8d4a258f625003e0fad9241560db0f606c23d1d8fab3a9c1afd3bb0c2f11a66a",
    "broad_161001": "a5096ca32a85a514c7e224208971bdfb2ccaa3efbc6b2987b444aa50a78ea5d3",
    "broad_161002": "6437fad819d821edccc63c7f9ae6f81075df38bea3155f32302ea33fddffb62b",
    "broad_161003": "2954041b2bed4c0b5427ad009cb8f2ef7be0f7be7dfb150d3f42277e74678ce5",
    "narrow_161001": "d5207e87e50c271405ab785fbc23015fdbbd8a262b97d0e8a6c8d5817f5e6b6f",
    "narrow_161002": "8ecac42c1145098c290f90423217e7a08f861638d62dd4b99962dbe73fae1abd",
    "narrow_161003": "120b895f5a2cfee5d725c7df654f96c7875a9a8080762b6c8fc68298b3759441",
}

def closed_parent():
    refs = {}
    def read(ref):
        refs[ref["path"]] = {k: ref[k] for k in ("path", "sha256")}
        return c.verified(ref)
    freeze = read(c.artifact(c.LEARNING / "freeze.json"))
    manifest = read(freeze["manifest"])
    c.require(c.old.digest(manifest) == freeze["binding"] == c.LEARNING_BINDING,
              "Learning binding changed")
    status = read(c.artifact(c.LEARNING / "final_status.json"))
    c.require(status["complete"] and status["all_prescribed_audits_pass"]
              and status["manifest"] == c.LEARNING_BINDING, "Learning parent incomplete")
    c.require(status["result"]["sha256"] == "7aba7182dfb585ddb183fa09e1a0ebcb13b61b3bf3fc6ea20639f1b0b2f75989"
              and status["audit"]["sha256"] == "34a7e1f0f23cccb0cedb486cef4544895dccce57bbad4c3aabd63232c49041e1",
              "Learning result/audit changed")
    result, audit = read(status["result"]), read(status["audit"])
    c.require(audit["passed"] and audit["result"] == status["result"]
              and audit["learning"] == result["learning"]
              and audit["manifest"] == result["manifest"] == c.LEARNING_BINDING,
              "Learning completion linkage")
    learning = read(audit["learning"])
    c.require(learning["manifest"] == c.LEARNING_BINDING, "Wrong learning-audit generation")
    registry = read(learning["models"])
    c.require(registry["manifest"] == c.LEARNING_BINDING and set(registry["models"]) == set(EXPECTED),
              "Wrong original learning registry")
    c.require(registry["matched_dose_and_batches"] and registry["common_broad_validation_selection"],
              "Matched control provenance differs")
    fits, fit_audits = read(learning["fits"]), read(learning["fit_audits"])
    c.require(fits["manifest"] == fit_audits["manifest"] == c.LEARNING_BINDING
              and fits["files"] == registry["fits"] and fit_audits["files"] == registry["audits"]
              and [r["fit"] for r in fits["files"]] == list(c.FITS)
              and [r["fit"] for r in fit_audits["files"]] == list(c.FITS), "Learning fit coverage changed")
    for fit, result_ref, audit_ref in zip(c.FITS, fits["files"], fit_audits["files"], strict=True):
        fit_result, fit_audit = read(result_ref), read(audit_ref)
        c.require(fit_result["manifest"] == fit_audit["manifest"] == c.LEARNING_BINDING
                  and fit_result["fit"] == fit_audit["fit"] == fit
                  and fit_audit["result"] == {k: result_ref[k] for k in ("path", "sha256")}
                  and fit_result["equal_dose_and_batches"] and fit_audit["equal_dose_and_batches"]
                  and fit_audit["complete_fit_bitwise_identical"] and fit_audit["common_broad_validation"]
                  and fit_audit["value_head_frozen"]
                  and fit_result["data_lock"] == registry["data_lock"], "Learning fit audit linkage changed")
        c.require(all(registry["models"][f"anchor_{fit}"][k] == v for k, v in fit_result["initializer"].items()),
                  "Anchor differs from audited initializer")
        for family in ("broad", "narrow"):
            selected = registry["models"][f"{family}_{fit}"]
            update = fit_result["selected_updates"][family]
            c.require(selected["selected_update"] == update == fit_audit["selected_updates"][family],
                      "Changed original checkpoint selection")
            matches = [entry["checkpoint"] for entry in fit_result["trajectories"][family] if entry["update"] == update]
            c.require(len(matches) == 1 and matches[0] == {k: selected[k] for k in ("path", "sha256")},
                      "Selected checkpoint not in audited trajectory")
    for key, ref in registry["models"].items():
        c.require(ref["sha256"] == EXPECTED[key] and ref["source_fit"] == int(key.split("_")[1]),
                  "Frozen selected model changed")
        c.verify_hash(ref)
        refs[ref["path"]] = {k: ref[k] for k in ("path", "sha256")}
    continuation = c.artifact(c.ROOT / c.old.CHECKPOINT)
    c.require(continuation["sha256"] == c.old.CHECKPOINT_SHA, "035 changed")
    refs[continuation["path"]] = continuation
    c.require(registry["data_lock"] == learning["data"], "Training population seal differs")
    data = read(registry["data_lock"])
    data_audit = read(data["audit"])
    c.require(data["manifest"] == data_audit["manifest"] == c.LEARNING_BINDING and data_audit["passed"]
              and data_audit["retained_tensors_targets_splits_unchanged"]
              and data_audit["independent_breadth_tensor_target_reconstruction"]
              and data_audit["source_evidence"] == data["source_evidence"], "Training data audit linkage changed")
    return {"models": registry["models"], "continuation_model": continuation,
            "source_models": learning["models"], "training_data": registry["data_lock"],
            "source_binding": c.LEARNING_BINDING,
            "evidence": [refs[k] for k in sorted(refs)]}

def prepare(binding, base=c.OUTPUT):
    value = {"manifest": binding, **closed_parent(), "training": False,
             "ancestor_learning_replayed": False, "specialist_reselection": False}
    c.old.save_new(Path(base)/"models_lock.json", value)
    return c.artifact(Path(base)/"models_lock.json")

def validate(binding, base=c.OUTPUT):
    value = c.old.read_json(Path(base)/"models_lock.json")
    expected = closed_parent()
    c.require(value["manifest"] == binding and value["training"] is False
              and value["specialist_reselection"] is False
              and value["ancestor_learning_replayed"] is False
              and all(value[k] == v for k, v in expected.items()), "Private model seal changed")
    return value
