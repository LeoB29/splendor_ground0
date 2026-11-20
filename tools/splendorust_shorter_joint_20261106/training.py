"""Fixed saved-corpus diagnostic; no import-time tensor or model execution.

The owner freezes the selection and source closure before calling these workers.
Old qualified mathematics is reused; original source/evidence remains unchanged.
"""
from __future__ import annotations

from collections import defaultdict
from copy import deepcopy
import hashlib
import json
import math
import os
from pathlib import Path
import random
import struct
import time

from tools import future_training_rules_20261021 as guard
from tools.splendorust_policy_head_20261104 import training as accepted
from tools.splendorust_policy_head_20261104 import data as accepted_data
from . import data

ROOT = accepted.ROOT
IDENTITY = accepted_data.IDENTITY
require = accepted.require
artifact = accepted.artifact
save = accepted.save_json
canonical = accepted_data.canonical
digest = accepted_data.digest
STUDY = data.STUDY
STEPS = 96
BATCH = 32
CHECKPOINTS = (0, 95, 96)
EVALUATION_STEPS = (0, 96)
SCOPES = ("joint", "policy")
_STARTUP = None
_ENGINE = None


def startup():
    """Exactly one corrected-source verification per new process, even fit-only."""
    global _STARTUP
    if _STARTUP is None:
        _STARTUP = {"runtime_sources": guard.verify_runtime_sources(), **IDENTITY}
    return deepcopy(_STARTUP)


def engine(folder):
    """Place the fresh owner before tensor work; never a RAM/forecast admission."""
    global _ENGINE
    receipt = startup()
    if _ENGINE is None:
        import psutil
        process = psutil.Process()
        process.cpu_affinity([0])
        process.nice(psutil.NORMAL_PRIORITY_CLASS)
        require(process.cpu_affinity() == [0], "actual CPU0 fitter/evaluator")
        import torch
        torch.set_num_threads(1)
        torch.set_num_interop_threads(1)
        _ENGINE = {"pid": process.pid, "created": process.create_time(),
                   "cpu_affinity": [0], "priority": "NORMAL_PRIORITY_CLASS",
                   "torch_threads": torch.get_num_threads(),
                   "torch_interop_threads": torch.get_num_interop_threads(),
                   "precision": "float32", "device": "cpu"}
    save(Path(folder) / "startup.json", {**receipt, "engine": deepcopy(_ENGINE)})
    return deepcopy(_ENGINE)


def fit_identity(repetition, breadth, scope):
    require(type(repetition) is int and repetition in (0, 1), "two fixed repetitions")
    require(breadth == "broad" and scope == "joint", "two fixed broad joint fits only")
    return {"fit_id": f"r{repetition}_{breadth}_{scope}", "repetition": repetition,
            "breadth": breadth, "scope": scope}


def source_equivalence():
    """Model-free source proof of unchanged updates, RNG and sampling math."""
    import ast
    original_path = ROOT / "tools/splendorust_learning_diagnostic_20261106/training.py"
    original_item = artifact(original_path)
    require(original_item["sha256"] == "89e5d91f07ff7f72c94e37e32dbfe1b34d9153c0bbedfbdcfea9c04cc3a51624",
            "frozen original trainer source")
    def functions(path):
        return {node.name: node for node in ast.parse(Path(path).read_text(encoding="utf-8")).body
                if isinstance(node, ast.FunctionDef)}
    old, new = functions(original_path), functions(__file__)
    names = ("scoped_parameters", "make_optimizer", "step", "rng_state", "restore_rng", "seed_rng")
    for name in names:
        require(ast.dump(old[name], include_attributes=False) == ast.dump(new[name], include_attributes=False),
                "unchanged optimizer/update/RNG function: " + name)
    require(data.artifact(ROOT / data.ORIGINAL_BUILDER["path"]) == data.ORIGINAL_BUILDER,
            "frozen original selector source")
    old_builder = functions(ROOT / data.ORIGINAL_BUILDER["path"])["_build_selection"]
    new_builder = functions(data.__file__)["_build_original_selection"]
    new_builder.name = old_builder.name
    require(ast.dump(old_builder, include_attributes=False) == ast.dump(new_builder, include_attributes=False),
            "unchanged deterministic original sampling builder")
    return {"status": "shorter_schedule_math_rng_sampling_source_verified", "original_trainer": original_item,
            "original_selector": deepcopy(data.ORIGINAL_BUILDER), "trainer": artifact(__file__),
            "selector": artifact(data.__file__), "unchanged_functions": list(names),
            "sampling_domain": data.DOMAIN, "study_id": STUDY, "model_calls": 0, "new_games": 0}


def scoped_parameters(model, scope):
    require(scope in SCOPES, "declared training scope")
    names = dict(model.named_parameters())
    active = [name for name, p in names.items() if p.requires_grad]
    expected = list(names) if scope == "joint" else ["policy.weight", "policy.bias"]
    require(active == expected, "exact ordered trainable tensor set")
    require(all(p.grad is None for name, p in names.items() if name not in active),
            "frozen tensors have no gradients")
    return [names[name] for name in active]


def make_optimizer(model, scope):
    import torch
    require(scope in SCOPES, "fixed scope")
    if scope == "policy":
        optimizer = accepted.make_optimizer(model)
    else:
        for p in model.parameters():
            p.requires_grad_(True)
            p.grad = None
        optimizer = torch.optim.AdamW(model.parameters(), lr=1e-5, betas=(.9, .999),
                                     eps=1e-8, weight_decay=1e-4, amsgrad=False,
                                     foreach=False, fused=False)
    expected = scoped_parameters(model, scope)
    actual = [p for group in optimizer.param_groups for p in group["params"]]
    require([id(p) for p in expected] == [id(p) for p in actual], "optimizer exact parameter ownership")
    return optimizer


def step(model, optimizer, batch, scope, *, boundary=lambda: None):
    """Accepted forward/loss and AdamW recipe; add complete saved update evidence."""
    import torch
    boundary()
    started = time.perf_counter()
    require(len(batch["tokens"]) == BATCH, "every diagnostic update is a full32 batch")
    model.train()
    params = scoped_parameters(model, scope)
    before = accepted.group_digests(model)
    optimizer.zero_grad(set_to_none=True)
    outputs = accepted.tensor_forward(model, batch["tokens"])
    total, policy, value = accepted.loss_components(outputs, batch)
    total.backward()
    boundary()
    require(all(p.grad is not None and bool(torch.isfinite(p.grad).all()) for p in params),
            "finite gradients for every trainable tensor")
    scoped_parameters(model, scope)
    norm = torch.nn.utils.clip_grad_norm_(params, 1., error_if_nonfinite=True, foreach=False)
    optimizer.step()
    boundary()
    require(all(bool(torch.isfinite(p).all()) for p in model.parameters()), "finite optimizer state")
    after = accepted.group_digests(model)
    require(before != after, "actual parameter update")
    if scope == "policy":
        require(all(before[k] == after[k] for k in ("shared", "value")) and
                before["policy"] != after["policy"], "policy-only exact frozen groups")
    scoped_parameters(model, scope)
    return {"batch_examples": BATCH, "batch_denominator": BATCH,
            "loss": float(total.detach()), "policy_mean_unweighted": float(policy.detach().mean()),
            "value_mean_unweighted": float(value.detach().mean()),
            "gradient_norm_before_clip": float(norm), "clipped": float(norm) > 1.,
            "finite_gradients": True, "forward_backward_calls": 1, "optimizer_calls": 1,
            "before_group_sha256": before, "after_group_sha256": after,
            "outputs83": outputs.detach().tolist(), "scope": scope,
            "seconds": time.perf_counter() - started}


def tensor_tree(value):
    """JSON representation with exact tensor bytes; never instantiate a network."""
    import torch
    if isinstance(value, torch.Tensor):
        require(value.device.type == "cpu", "CPU checkpoint tensors only")
        require(not value.is_floating_point() or bool(torch.isfinite(value).all()), "finite checkpoint tensor")
        raw = value.detach().contiguous().numpy().tobytes()
        return {"tensor": True, "dtype": str(value.dtype), "shape": list(value.shape),
                "bytes": len(raw), "sha256": hashlib.sha256(raw).hexdigest()}
    if isinstance(value, dict):
        return {"dict": [[tensor_tree(k), tensor_tree(v)] for k, v in value.items()]}
    if isinstance(value, tuple):
        return {"tuple": [tensor_tree(v) for v in value]}
    if isinstance(value, list):
        return {"list": [tensor_tree(v) for v in value]}
    require(value is None or type(value) in (str, int, bool, float), "strict checkpoint scalar type")
    if type(value) is float:
        require(math.isfinite(value), "finite checkpoint scalar")
        return {"float64": struct.pack("<d", value).hex()}
    return {"type": type(value).__name__, "value": value}


def state_mapping_hashes(state):
    """Same accepted tensor hashes, from a saved mapping without a model object."""
    import torch
    result = {}
    for group in ("all", "shared", "policy", "value"):
        sha = hashlib.sha256()
        for name, value in sorted(state.items()):
            selected = group == "all" or (name.startswith(group + ".") if group in ("policy", "value") else
                                             not name.startswith(("policy.", "value.")))
            if selected:
                require(value.dtype == torch.float32 and value.device.type == "cpu", "saved model float32 tensors")
                sha.update(canonical([name, list(value.shape), str(value.dtype)]).encode())
                sha.update(b"\0")
                sha.update(value.detach().contiguous().numpy().tobytes())
        result[group] = sha.hexdigest()
    return result


def verify_optimizer_payload(payload):
    """Independently check saved AdamW membership, options and exact step counters."""
    import torch
    meta, state, optimizer = payload["metadata"], payload["state_dict"], payload["optimizer"]
    names = list(state) if meta["scope"] == "joint" else ["policy.weight", "policy.bias"]
    require(len(optimizer["param_groups"]) == 1, "one fixed optimizer parameter group")
    group = optimizer["param_groups"][0]
    expected = {"lr": 1e-5, "betas": (.9, .999), "eps": 1e-8, "weight_decay": 1e-4,
                "amsgrad": False, "foreach": False, "fused": False, "maximize": False,
                "capturable": False, "differentiable": False}
    require(all(group.get(k) == v for k, v in expected.items()) and
            group["params"] == list(range(len(names))), "saved exact AdamW recipe and ordered scope")
    index = meta["checkpoint_step"]
    require(set(optimizer["state"]) == (set(range(len(names))) if index else set()), "complete AdamW state inventory")
    for parameter_index, name in enumerate(names):
        if not index:
            continue
        slots = optimizer["state"][parameter_index]
        require(set(slots) == {"step", "exp_avg", "exp_avg_sq"} and
                slots["step"].numel() == 1 and float(slots["step"]) == index,
                "exact completed optimizer update count")
        require(all(slots[k].shape == state[name].shape and slots[k].dtype == torch.float32
                    for k in ("exp_avg", "exp_avg_sq")), "optimizer tensors align to exact named parameter")


def rng_state():
    import numpy as np
    import torch
    state = np.random.get_state()
    return {"torch": torch.get_rng_state().clone(), "python": random.getstate(),
            "numpy": [state[0], state[1].tolist(), state[2], state[3], state[4]]}


def restore_rng(state):
    import numpy as np
    import torch
    torch.set_rng_state(state["torch"])
    random.setstate(state["python"])
    name, words, position, cached, gaussian = state["numpy"]
    np.random.set_state((name, np.asarray(words, dtype=np.uint32), position, cached, gaussian))


def seed_rng(repetition):
    import numpy as np
    import torch
    seed = 202611012 + repetition
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    return seed


def read_json(item, *, boundary=lambda: None):
    return accepted_data.read_artifact(item, boundary=boundary)


def historical_prefix_bindings(*, boundary=lambda: None):
    """Authenticate the two accepted old trajectories without loading tensors."""
    audit_item = artifact(ROOT / "outputs/splendorust_learning_diagnostic_20261106_01/audit/worker/saved_evidence.json")
    require(audit_item["sha256"] == "8707600941066c1284ed5eed916035a3e4d5fc242c6599b16fcc39dd8d231f0a",
            "accepted original saved scientific audit")
    original = read_json(audit_item, boundary=boundary)
    require(original["status"] == "saved_evidence_verified_pending_independent_acceptance" and
            original["total_verified_optimizer_calls"] == 5354 and
            original["selection"] == data.ORIGINAL_SELECTION, "original passed scientific evidence preserved")
    selection_sha = digest(read_json(data.ORIGINAL_SELECTION, boundary=boundary))
    fits = []
    for repetition in (0, 1):
        identity = fit_identity(repetition, "broad", "joint")
        matches = [row for row in original["fits"] if all(row.get(k) == v for k, v in identity.items())]
        require(len(matches) == 1, "one original verified broad joint fit")
        verified = matches[0]
        require(verified["status"] == "fit_saved_verified" and verified["optimizer_calls"] == 668 and
                verified["replay_optimizer_calls"] == 1 and verified["selection_sha256"] == selection_sha,
                "original complete verified fit and selection")
        result = read_json(verified["result"], boundary=boundary)
        require(all(result[k] == v for k, v in identity.items()) and result["status"] == "fit_complete" and
                result["selection_sha256"] == selection_sha and result["optimizer_calls"] == 668 and
                len(result["updates"]) == 668 and result["main_training_started_from_pristine_parent"] is True,
                "authenticated original fixed update trajectory")
        updates = deepcopy(result["updates"][:STEPS])
        parent = Path(verified["result"]["path"]).parent
        require([row["path"] for row in updates] == [(parent / "updates" / f"{index:03d}.json").as_posix()
                                                   for index in range(1, STEPS + 1)],
                "original first96 records, without omission or reordering")
        fits.append({**identity, "fit_result": deepcopy(verified["result"]), "updates": updates})
    return {"status": "accepted_historical_prefix_bindings", "saved_scientific_audit": audit_item,
            "original_selection": deepcopy(data.ORIGINAL_SELECTION), "selection_sha256": selection_sha,
            "fits": fits, "compared_steps_per_fit": STEPS,
            "excluded_comparison_fields": ["seconds", "selection_sha256"], "model_calls": 0}


def verify_historical_update(actual, reference, actual_selection_sha, reference_selection_sha):
    """Only elapsed time and the new selection manifest identity may differ."""
    excluded = {"seconds", "selection_sha256"}
    require(set(actual) == set(reference) and excluded <= set(actual), "same complete update evidence schema")
    require(actual["selection_sha256"] == actual_selection_sha and
            reference["selection_sha256"] == reference_selection_sha, "each trajectory binds its exact selection")
    require(all(type(row["seconds"]) in (int, float) and math.isfinite(row["seconds"]) and row["seconds"] >= 0
                for row in (actual, reference)), "finite nonnegative elapsed update time")
    project = lambda row: {key: value for key, value in row.items() if key not in excluded}
    require(canonical(project(actual)) == canonical(project(reference)),
            "exact historical first-pass mathematics, outputs, gradient evidence and tensor-group chain")
    return digest(project(actual))


def save_checkpoint(model, optimizer, step_index, metadata, folder, *, boundary=lambda: None):
    import torch
    boundary()
    accepted.validate_architecture(model)
    meta = {**metadata, "checkpoint_step": step_index, "weights_sha256": accepted.state_digest(model),
            "group_sha256": accepted.group_digests(model), **IDENTITY}
    payload = {"schema": "learning_diagnostic_checkpoint_v1", "kind": "entity", "metadata": meta,
               "state_dict": model.state_dict(), "optimizer": optimizer.state_dict(), "rng": rng_state()}
    folder = Path(folder)
    folder.mkdir(parents=True, exist_ok=True)
    path = folder / "checkpoint.pt"
    with path.open("xb") as stream:
        torch.save(payload, stream)
        stream.flush()
        os.fsync(stream.fileno())
    trees = {k: tensor_tree(payload[k]) for k in ("state_dict", "optimizer", "rng")}
    tensor_item = save(folder / "tensor_manifest.json", trees)
    manifest = {"schema": "learning_diagnostic_checkpoint_manifest_v1", "checkpoint": artifact(path),
                "tensor_manifest": tensor_item, "metadata": meta,
                "tree_sha256": {k: digest(v) for k, v in trees.items()}}
    result = save(folder / "checkpoint_manifest.json", manifest)
    boundary()
    return result


def checkpoint_payload(item, *, boundary=lambda: None):
    """Saved-only verifier: authenticates archive, metadata and every tensor bit."""
    import torch
    manifest = read_json(item, boundary=boundary)
    require(manifest["schema"] == "learning_diagnostic_checkpoint_manifest_v1", "checkpoint manifest schema")
    descriptor = manifest["checkpoint"]
    path = (ROOT / descriptor["path"]).resolve()
    require(path.is_relative_to(ROOT) and artifact(path) == descriptor and
            0 < descriptor["bytes"] < 256 * 1024**2, "bounded frozen checkpoint bytes")
    before = path.stat()
    with path.open("rb") as stream:
        payload = torch.load(stream, map_location="cpu", weights_only=True)
    after = path.stat()
    require((before.st_size, before.st_mtime_ns) == (after.st_size, after.st_mtime_ns) and
            artifact(path) == descriptor, "checkpoint unchanged during parse")
    require(payload["schema"] == "learning_diagnostic_checkpoint_v1" and payload["kind"] == "entity" and
            payload["metadata"] == manifest["metadata"], "checkpoint metadata binding")
    accepted_data.verify_identity(payload["metadata"])
    require(payload["metadata"]["study_id"] == STUDY and
            payload["metadata"]["historical_warm_start_sha256"] == accepted.PARENT_SHA256 and
            payload["metadata"]["historical_data_relabelled"] is False, "historical warmstart only")
    hashes = state_mapping_hashes(payload["state_dict"])
    require(hashes.pop("all") == manifest["metadata"]["weights_sha256"] and
            hashes == manifest["metadata"]["group_sha256"], "actual saved tensor bytes bind all model hashes")
    verify_optimizer_payload(payload)
    trees = {k: tensor_tree(payload[k]) for k in ("state_dict", "optimizer", "rng")}
    require(trees == read_json(manifest["tensor_manifest"], boundary=boundary) and
            {k: digest(v) for k, v in trees.items()} == manifest["tree_sha256"], "all checkpoint tensor/scalar bits")
    boundary()
    return payload, manifest


def load_checkpoint(item, *, optimizer=False, boundary=lambda: None):
    payload, manifest = checkpoint_payload(item, boundary=boundary)
    model = accepted.load_parent()
    model.load_state_dict(payload["state_dict"], strict=True)
    model.eval()
    accepted.validate_architecture(model)
    require(accepted.state_digest(model) == manifest["metadata"]["weights_sha256"] and
            accepted.group_digests(model) == manifest["metadata"]["group_sha256"], "restored exact model")
    opt = make_optimizer(model, manifest["metadata"]["scope"])
    opt.load_state_dict(payload["optimizer"])
    require(tensor_tree(opt.state_dict()) == tensor_tree(payload["optimizer"]), "restored exact optimizer")
    restore_rng(payload["rng"])
    require(tensor_tree(rng_state()) == tensor_tree(payload["rng"]), "restored exact RNG")
    return model, opt if optimizer else None, manifest


def validate_loaded(loaded):
    require(len(loaded["train_examples"]) == 3072 and loaded["train_weights"] == [1.] * 3072,
            "exact equal-weight3072 selected records")
    require(len(loaded["batches"]) == STEPS and
            all(len(batch) == BATCH and len(set(batch)) == BATCH and
                all(type(i) is int and 0 <= i < 3072 for i in batch) for batch in loaded["batches"]),
            "exact96 predetermined full batches")
    require(sorted(i for batch in loaded["batches"] for i in batch) == list(range(3072)),
            "one complete pass, every selected root once")
    require(loaded["probe_batches"] == [], "no disposable diagnostic gradient probes")
    require(len(loaded["heldout_examples"]) == 3557, "all64 opened heldout games3557 roots")
    return loaded


def selected_batch(loaded, indices):
    return data.collate([loaded["train_examples"][i] for i in indices],
                        [loaded["train_weights"][i] for i in indices])


def metadata(selection, loaded, identity):
    return {**identity, **IDENTITY, "study_id": STUDY, "selection_sha256": digest(selection),
            "source_receipt_sha256": digest(loaded["source_receipt"]),
            "historical_warm_start_sha256": accepted.PARENT_SHA256,
            "historical_data_relabelled": False, "model_source_sha256": accepted.MODEL_SOURCE_SHA256,
            "parameter_count": accepted.PARAMETERS, "fixed_final_step": STEPS,
            "checkpoint_selection": False, "training_records": 3072,
            "optimizer_recipe": {k: v for k, v in accepted.RECIPE.items() if k not in
                                 ("epochs", "shuffle", "example_weight", "batch_reduction")},
            "batch_reduction": "mean32_full_equal_game_record_weights", "record_presentations": STEPS * BATCH,
            "schedule_sha256": digest(loaded["batches"]), "seed": 202611012 + identity["repetition"],
            "original_selection": deepcopy(data.ORIGINAL_SELECTION), "complete_passes": 1,
            "schedule_relation": "exact_first_96_batches_of_original_668",
            "diagnostic_gradient_probes": 0}


def run_qualification(selection, folder, *, boundary=lambda: None):
    folder = Path(folder)
    engine(folder)
    import torch
    qualification = data.qualify_data(selection, folder / "data", boundary=boundary)
    examples, weights = qualification["qualification_examples"], qualification["qualification_weights"]
    require(len(examples) == BATCH and weights == [1.] * BATCH and
            all(x["split"] == "train" for x in examples), "genuine official-target qualification roots")
    loaded = validate_loaded(data.load_selected(selection, 0, "broad", boundary=boundary))
    require(examples == [loaded["train_examples"][i] for i in loaded["batches"][0]],
            "qualification uses frozen first batch, no fabricated labels")
    batch = data.collate(examples, weights)
    results = []
    for scope in SCOPES:
        boundary()
        seed_rng(0)
        model = accepted.load_parent()
        opt = make_optimizer(model, scope)
        # Both disposable scope checks remain, without admitting a policy fit.
        identity = {"fit_id": f"qualification_broad_{scope}", "repetition": 0,
                    "breadth": "broad", "scope": scope}
        meta = {**metadata(selection, loaded, identity), "qualification_only": True}
        before_groups = accepted.group_digests(model)
        with torch.no_grad():
            model.train()
            before = accepted.tensor_forward(model, batch["tokens"]).tolist()
        update = step(model, opt, batch, scope, boundary=boundary)
        require(accepted.full_output_bytes(before) == accepted.full_output_bytes(update["outputs83"]),
                "engineering before-forward equals actual update-forward")
        with torch.no_grad():
            after = accepted.tensor_forward(model, batch["tokens"]).tolist()
        value_proof = accepted.verify_value_pair(before, after, "one-step qualification") if scope == "policy" else None
        item = save_checkpoint(model, opt, 1, meta, folder / scope / "checkpoint", boundary=boundary)
        del model, opt
        model, opt, _ = load_checkpoint(item, optimizer=True, boundary=boundary)
        model.train()
        with torch.no_grad():
            reloaded = accepted.tensor_forward(model, batch["tokens"]).tolist()
        require(accepted.full_output_bytes(after) == accepted.full_output_bytes(reloaded), "save/reload exact outputs")
        results.append({"scope": scope, "trainable_parameters": sum(p.numel() for p in scoped_parameters(model, scope)),
                        "before_group_sha256": before_groups, "step": update,
                        "before_outputs83": before, "after_outputs83": after,
                        "reloaded_outputs83": reloaded, "value_preservation": value_proof,
                        "checkpoint": item, "optimizer_calls": 1, "forward_calls": 4,
                        "backward_calls": 1, "discarded": True})
        del model, opt
    result = {"schema": "learning_diagnostic_qualification_v1", "status": "qualification_passed",
              "selection_sha256": digest(selection), "data_qualification": {k: v for k, v in qualification.items()
                    if k not in ("qualification_examples", "qualification_weights")},
              "root_identities": [[x["game_index"], x["ply"], x["row_sha256"]] for x in examples],
              "scopes": results, "optimizer_calls": 2, "forward_calls": 8, "backward_calls": 2,
              "scientific_updates": 0, "study_id": STUDY, **IDENTITY}
    save(folder / "qualification_result.json", result)
    boundary()
    return result


def run_fit(selection, repetition, breadth, scope, folder, *, boundary=lambda: None):
    folder = Path(folder)
    engine(folder)
    identity = fit_identity(repetition, breadth, scope)
    selection_sha256 = digest(selection)
    loaded = validate_loaded(data.load_selected(selection, repetition, breadth, boundary=boundary))
    meta = {**metadata(selection, loaded, identity), "qualification_only": False}
    seed_rng(repetition)
    model = accepted.load_parent()
    opt = make_optimizer(model, scope)
    initial = accepted.group_digests(model)
    checkpoints = {"0": save_checkpoint(model, opt, 0, meta, folder / "checkpoints/000", boundary=boundary)}
    updates = []
    for index, indices in enumerate(loaded["batches"], start=1):
        boundary()
        update = step(model, opt, selected_batch(loaded, indices), scope, boundary=boundary)
        update.update(**identity, step=index, selection_sha256=selection_sha256, example_indices=indices,
                      batch_source_sha256=digest([[loaded["train_examples"][i]["game_index"],
                          loaded["train_examples"][i]["ply"], loaded["train_examples"][i]["row_sha256"]] for i in indices]))
        updates.append(save(folder / "updates" / f"{index:03d}.json", update))
        if index in CHECKPOINTS:
            checkpoints[str(index)] = save_checkpoint(model, opt, index, meta,
                folder / "checkpoints" / f"{index:03d}", boundary=boundary)
    final_groups = accepted.group_digests(model)
    if scope == "policy":
        require(all(initial[k] == final_groups[k] for k in ("shared", "value")), "entire fit exact frozen groups")
    del model, opt
    replay_model, replay_opt, _ = load_checkpoint(checkpoints[str(STEPS - 1)], optimizer=True, boundary=boundary)
    replay = step(replay_model, replay_opt, selected_batch(loaded, loaded["batches"][STEPS - 1]), scope, boundary=boundary)
    replay_item = save_checkpoint(replay_model, replay_opt, STEPS, meta, folder / "replay/checkpoint", boundary=boundary)
    del replay_model, replay_opt
    final_manifest = read_json(checkpoints[str(STEPS)], boundary=boundary)
    replay_manifest = read_json(replay_item, boundary=boundary)
    require(final_manifest["tree_sha256"] == replay_manifest["tree_sha256"],
            "exact final-step model, optimizer and RNG reproduction")
    final_update = read_json(updates[-1], boundary=boundary)
    require(all(replay[k] == final_update[k] for k in replay if k != "seconds"), "exact final-step outputs/loss/gradients")
    replay_proof = save(folder / "replay/result.json", {"status": "exact_final_update_reproduced",
        "from_checkpoint": checkpoints[str(STEPS - 1)], "expected_checkpoint": checkpoints[str(STEPS)],
        "actual_checkpoint": replay_item, "expected_update": updates[-1], "actual_update": replay,
        "tree_sha256": final_manifest["tree_sha256"], "optimizer_calls": 1})
    result = {"schema": "learning_diagnostic_fit_v1", "status": "fit_complete", **identity, **IDENTITY,
              "selection_sha256": selection_sha256, "source_receipt": loaded["source_receipt"],
              "selection_cell": loaded["selection_cell"], "checkpoints": checkpoints,
              "updates": updates, "replay": replay_proof, "optimizer_calls": STEPS,
              "disposable_optimizer_calls": 1, "record_presentations": STEPS * BATCH,
              "before_group_sha256": initial, "after_group_sha256": final_groups,
              "main_training_started_from_pristine_parent": True, "checkpoint_selection": False,
              "study_id": STUDY}
    save(folder / "fit_result.json", result)
    boundary()
    return result


def prediction_row(example, output):
    return {**accepted.prediction_row(example, output), "row_sha256": example["row_sha256"]}


def parameter_displacement(state, original):
    import torch
    out = {}
    for group in ("all", "shared", "policy", "value"):
        names = [name for name in state if group == "all" or
                 (name.startswith(group + ".") if group in ("policy", "value") else
                  not name.startswith(("policy.", "value.")))]
        square, base, maximum, changed, count = 0., 0., 0., 0, 0
        for name in names:
            x, y = state[name].detach(), original[name]
            require(x.dtype == y.dtype == torch.float32 and x.shape == y.shape, "same fixed parameter shape")
            delta = x.to(torch.float64) - y.to(torch.float64)
            square += float((delta * delta).sum())
            base += float((y.to(torch.float64) ** 2).sum())
            maximum = max(maximum, float(delta.abs().max()))
            changed += int((x.view(torch.int32) != y.view(torch.int32)).sum())
            count += x.numel()
        out[group] = {"parameters": count, "changed_float32_values": changed,
                      "l2": math.sqrt(square), "parent_l2": math.sqrt(base),
                      "relative_l2": math.sqrt(square / base) if base else None, "max_absolute": maximum}
    return out


def gradient_summary(names, gradients):
    import torch
    squares, shared, hashes, missing = [], [], {}, []
    for name, gradient in zip(names, gradients, strict=True):
        if gradient is None:
            missing.append(name)
            continue
        require(bool(torch.isfinite(gradient).all()), "finite per-loss diagnostic gradients")
        square = float((gradient.detach().to(torch.float64) ** 2).sum())
        squares.append(square)
        if not name.startswith(("policy.", "value.")):
            shared.append(square)
        hashes[name] = hashlib.sha256(gradient.detach().contiguous().numpy().tobytes()).hexdigest()
    require(bool(squares), "loss has a genuine gradient path")
    return {"norm": math.sqrt(math.fsum(squares)), "shared_norm": math.sqrt(math.fsum(shared)),
            "gradient_sha256": hashes, "unused_parameters": missing, "finite": True}


def gradient_probe(model, scope, batch, *, boundary=lambda: None):
    """Disposable model, one backward per loss, zero optimizer calls."""
    import torch
    boundary()
    disposable = deepcopy(model)
    disposable.train()
    for name, p in disposable.named_parameters():
        p.requires_grad_(scope == "joint" or name in ("policy.weight", "policy.bias"))
        p.grad = None
    before = accepted.state_digest(disposable)
    rng_before = digest(tensor_tree(rng_state()))
    names = [name for name, p in disposable.named_parameters() if p.requires_grad]
    parameters = scoped_parameters(disposable, scope)
    outputs = accepted.tensor_forward(disposable, batch["tokens"])
    _, policy, value = accepted.loss_components(outputs, batch)
    policy_loss = (policy * batch["weights"]).sum() / 32.
    gradients_p = torch.autograd.grad(policy_loss, parameters, retain_graph=scope == "joint", allow_unused=True)
    p_summary = gradient_summary(names, gradients_p)
    result = {"policy": p_summary, "policy_loss": float(policy_loss.detach()),
              "optimizer_calls": 0, "forward_calls": 1, "backward_probes": 1,
              "checkpoint_weights_sha256": before, "scope": scope}
    if scope == "joint":
        value_loss = (value * batch["weights"]).sum() / 32.
        gradients_v = torch.autograd.grad(value_loss, parameters, allow_unused=True)
        v_summary = gradient_summary(names, gradients_v)
        dot = math.fsum(float((p.detach().to(torch.float64) * v.detach().to(torch.float64)).sum())
                        for name, p, v in zip(names, gradients_p, gradients_v, strict=True)
                        if p is not None and v is not None and not name.startswith(("policy.", "value.")))
        denominator = p_summary["shared_norm"] * v_summary["shared_norm"]
        cosine = dot / denominator if denominator else None
        require(cosine is None or -1.000000001 <= cosine <= 1.000000001, "finite bounded shared gradient cosine")
        result.update(value=v_summary, value_loss=float(value_loss.detach()), shared_dot=dot,
                      shared_cosine=cosine, undefined_cosine=denominator == 0., backward_probes=2)
    require(accepted.state_digest(disposable) == before and all(p.grad is None for p in disposable.parameters()),
            "gradient probes preserve every checkpoint tensor and parameter grad slot")
    require(digest(tensor_tree(rng_state())) == rng_before, "zero-dropout probes preserve RNG")
    result.update(weights_unchanged=True, rng_unchanged=True, zero_optimizer_calls=True)
    del disposable
    boundary()
    return result


def run_diagnostics(selection, repetition, breadth, scope, fit_folder, folder, *, boundary=lambda: None):
    folder = Path(folder)
    engine(folder)
    import torch
    identity = fit_identity(repetition, breadth, scope)
    selection_sha256 = digest(selection)
    loaded = validate_loaded(data.load_selected(selection, repetition, breadth, boundary=boundary))
    fit_item = artifact(Path(fit_folder) / "fit_result.json")
    fit = read_json(fit_item, boundary=boundary)
    require(all(fit[k] == v for k, v in identity.items()) and fit["selection_sha256"] == selection_sha256 and
            fit["status"] == "fit_complete", "fixed completed fit for diagnostics")
    original_payload, _ = checkpoint_payload(fit["checkpoints"]["0"], boundary=boundary)
    original_state = original_payload["state_dict"]
    del original_payload
    results = []
    parent_values = {}
    for index in EVALUATION_STEPS:
        item = fit["checkpoints"][str(index)]
        model, _, manifest = load_checkpoint(item, boundary=boundary)
        before = accepted.state_digest(model)
        model.eval()
        batches, summaries, probes = [], {}, []
        for split in ("train", "validation", "offline"):
            examples = loaded["train_examples"] if split == "train" else [
                x for x in loaded["heldout_examples"] if x["split"] == split]
            rows = []
            for offset in range(0, len(examples), BATCH):
                boundary()
                part = examples[offset:offset + BATCH]
                batch = data.collate(part, [1.] * len(part))
                with torch.inference_mode():
                    outputs = accepted.tensor_forward(model, batch["tokens"]).tolist()
                observations = [prediction_row(example, output) for example, output in zip(part, outputs, strict=True)]
                batch_index = offset // BATCH
                key = (split, batch_index)
                values = accepted.value_bytes(outputs)
                if index == 0:
                    parent_values[key] = values
                elif scope == "policy":
                    require(values == parent_values[key], "all fixed evaluation inputs preserve original value bits")
                batch_item = save(folder / "evaluation" / f"step_{index:03d}" / split / f"{batch_index:06d}.json",
                    {**identity, "checkpoint_step": index, "split": split, "batch_index": batch_index,
                     "selection_sha256": selection_sha256, "weights_sha256": before,
                     "checkpoint_manifest": item, "rows": observations})
                rows.extend(observations)
                batches.append(batch_item)
                boundary()
            summaries[split] = accepted.summarize_predictions(rows, sorted({x["game_index"] for x in examples}))
        require(loaded["probe_batches"] == [], "evaluation has no disposable gradient work")
        require(accepted.state_digest(model) == before, "fixed checkpoint preserved throughout evaluation")
        displacement = parameter_displacement(model.state_dict(), original_state)
        results.append({"checkpoint_step": index, "checkpoint_manifest": item, "batches": batches,
                        "summaries": summaries, "group_displacement": displacement, "gradient_probes": probes,
                        "weights_sha256": before, "forward_calls": len(batches) + len(probes),
                        "backward_probes": 0})
        del model
    result = {"schema": "learning_diagnostic_evaluation_v1", "status": "diagnostics_complete",
              **identity, **IDENTITY, "selection_sha256": selection_sha256, "fit_result": fit_item,
              "checkpoints": results, "forward_calls": sum(x["forward_calls"] for x in results),
              "backward_probes": sum(x["backward_probes"] for x in results), "optimizer_calls": 0,
              "policy_only_value_bits_preserved": True if scope == "policy" else None,
              "opened_heldout_exploratory_only": True, "checkpoint_selection": False, "strength_claim": False,
              "study_id": STUDY}
    save(folder / "diagnostics_result.json", result)
    boundary()
    return result


def verify_qualification(selection, folder, *, boundary=lambda: None):
    """Saved-only scope/reload proof; actual loader qualification audited separately."""
    value = read_json(artifact(Path(folder) / "qualification_result.json"), boundary=boundary)
    require(value["study_id"] == STUDY and value["status"] == "qualification_passed" and value["selection_sha256"] == digest(selection) and
            value["optimizer_calls"] == 2 and value["forward_calls"] == 8 and value["backward_calls"] == 2 and
            value["scientific_updates"] == 0, "exact disposable qualification work inventory")
    require([x["scope"] for x in value["scopes"]] == list(SCOPES), "both fixed scopes qualified once")
    for row in value["scopes"]:
        scope, update = row["scope"], row["step"]
        require(row["optimizer_calls"] == 1 and row["discarded"] is True and
                row["trainable_parameters"] == (accepted.PARAMETERS if scope == "joint" else 20817),
                "exact trainable set size and disposable update")
        require(accepted.full_output_bytes(row["before_outputs83"]) == accepted.full_output_bytes(update["outputs83"]) and
                accepted.full_output_bytes(row["after_outputs83"]) == accepted.full_output_bytes(row["reloaded_outputs83"]),
                "actual forward and save/reload parity")
        payload, manifest = checkpoint_payload(row["checkpoint"], boundary=boundary)
        require(manifest["metadata"]["scope"] == scope and manifest["metadata"]["checkpoint_step"] == 1 and
                manifest["metadata"]["qualification_only"] is True and
                manifest["metadata"]["group_sha256"] == update["after_group_sha256"], "qualification saved model identity")
        if scope == "policy":
            require(all(row["before_group_sha256"][k] == update["after_group_sha256"][k] for k in ("shared", "value")),
                    "qualification exact frozen tensor groups")
            require(row["value_preservation"] == accepted.verify_value_pair(row["before_outputs83"],
                row["after_outputs83"], "saved one-step qualification"), "qualification exact value output bits")
        del payload
    return {"status": "qualification_saved_verified", "optimizer_calls": 2,
            "selection_sha256": digest(selection), "result": artifact(Path(folder) / "qualification_result.json")}


def verify_fit(selection, repetition, breadth, scope, folder, *, loaded=None, boundary=lambda: None):
    """No model construction or execution; validate actual serialized state/replay."""
    folder = Path(folder)
    identity = fit_identity(repetition, breadth, scope)
    selection_sha256 = digest(selection)
    loaded = validate_loaded(loaded if loaded is not None else data.load_selected(
        selection, repetition, breadth, boundary=boundary))
    result_item = artifact(folder / "fit_result.json")
    result = read_json(result_item, boundary=boundary)
    require(result["study_id"] == STUDY and result["schema"] == "learning_diagnostic_fit_v1" and result["status"] == "fit_complete" and
            all(result[k] == v for k, v in identity.items()) and result["selection_sha256"] == selection_sha256 and
            result["optimizer_calls"] == STEPS and result["disposable_optimizer_calls"] == 1 and
            result["record_presentations"] == STEPS * BATCH and result["checkpoint_selection"] is False and
            result["main_training_started_from_pristine_parent"] is True, "fit identity and complete work counts")
    require(result["source_receipt"] == loaded["source_receipt"] and
            result["selection_cell"] == loaded["selection_cell"], "same actual selected source corpus")
    require(result["updates"] == [artifact(folder / "updates" / f"{i:03d}.json") for i in range(1, STEPS + 1)] and
            len(list((folder / "updates").glob("*.json"))) == STEPS, "every update exactly once, no replacement")
    require(set(result["checkpoints"]) == {str(i) for i in CHECKPOINTS}, "fixed checkpoint inventory")
    manifests = {}
    initial_state = None
    for index in CHECKPOINTS:
        item = result["checkpoints"][str(index)]
        payload, manifest = checkpoint_payload(item, boundary=boundary)
        meta = manifest["metadata"]
        require(all(meta[k] == v for k, v in metadata(selection, loaded, identity).items()) and
                meta["checkpoint_step"] == index and meta["qualification_only"] is False,
                "checkpoint exact frozen metadata")
        if index == 0:
            # Original archive hash plus exact original tensors establish warm start.
            import torch
            require(guard.sha256(accepted.PARENT_CHECKPOINT) == accepted.PARENT_SHA256, "original warm start archive")
            parent = torch.load(accepted.PARENT_CHECKPOINT, map_location="cpu", weights_only=True)
            require(tensor_tree(payload["state_dict"]) == tensor_tree(parent["state_dict"]), "all pristine original tensor bytes")
            initial_state = tensor_tree(payload["state_dict"])
            require(payload["optimizer"]["state"] == {}, "fresh empty optimizer at step0")
            del parent
        manifests[index] = manifest
        del payload
    require(initial_state is not None, "parent state checked")
    groups = result["before_group_sha256"]
    require(groups == manifests[0]["metadata"]["group_sha256"], "update chain starts at parent")
    historical = historical_prefix_bindings(boundary=boundary)
    prefix = next(row for row in historical["fits"] if row["fit_id"] == identity["fit_id"])
    matched_mathematics = []
    clipped = 0
    for number, (descriptor, indices) in enumerate(zip(result["updates"], loaded["batches"], strict=True), start=1):
        update = read_json(descriptor, boundary=boundary)
        examples = [loaded["train_examples"][i] for i in indices]
        require(all(update[k] == v for k, v in identity.items()) and update["selection_sha256"] == selection_sha256 and
                update["step"] == number and update["example_indices"] == indices and
                update["batch_source_sha256"] == digest([[x["game_index"], x["ply"], x["row_sha256"]] for x in examples]),
                "every step binds frozen batch and source targets")
        require(update["batch_examples"] == update["batch_denominator"] == BATCH and
                update["forward_backward_calls"] == update["optimizer_calls"] == 1 and
                update["finite_gradients"] is True and math.isfinite(update["gradient_norm_before_clip"]) and
                update["gradient_norm_before_clip"] >= 0 and
                update["clipped"] == (update["gradient_norm_before_clip"] > 1.) and
                update["before_group_sha256"] == groups and update["after_group_sha256"] != groups,
                "complete finite genuine update chain")
        if scope == "policy":
            require(all(update["after_group_sha256"][k] == groups[k] for k in ("shared", "value")),
                    "every update preserves shared/value tensor bits")
        losses = [accepted.prediction_losses(prediction_row(x, y)) for x, y in
                  zip(examples, update["outputs83"], strict=True)]
        expected = {"loss": math.fsum(x["combined"] for x in losses) / BATCH,
                    "policy_mean_unweighted": math.fsum(x["policy"] for x in losses) / BATCH,
                    "value_mean_unweighted": math.fsum(x["value"] for x in losses) / BATCH}
        require(all(math.isclose(update[k], v, rel_tol=1e-6, abs_tol=1e-5) for k, v in expected.items()),
                "saved update outputs reproduce official targets and exact loss recipe")
        reference = read_json(prefix["updates"][number - 1], boundary=boundary)
        matched_mathematics.append(verify_historical_update(update, reference, selection_sha256,
                                                           historical["selection_sha256"]))
        groups = update["after_group_sha256"]
        if number in CHECKPOINTS:
            require(groups == manifests[number]["metadata"]["group_sha256"], "checkpoint lies on exact update chain")
        clipped += int(update["clipped"])
    require(groups == result["after_group_sha256"], "final update chain complete")
    replay = read_json(result["replay"], boundary=boundary)
    replay_payload, replay_manifest = checkpoint_payload(replay["actual_checkpoint"], boundary=boundary)
    require(replay["status"] == "exact_final_update_reproduced" and replay["optimizer_calls"] == 1 and
            replay["from_checkpoint"] == result["checkpoints"][str(STEPS - 1)] and
            replay["expected_checkpoint"] == result["checkpoints"][str(STEPS)] and replay["expected_update"] == result["updates"][-1] and
            replay_manifest["metadata"] == manifests[STEPS]["metadata"] and
            replay_manifest["tree_sha256"] == manifests[STEPS]["tree_sha256"] == replay["tree_sha256"],
            "saved exact final-update model/optimizer/RNG replay")
    require(all(replay["actual_update"][k] == update[k] for k in replay["actual_update"] if k != "seconds"),
            "replay exact output/loss/gradient evidence")
    del replay_payload
    return {"status": "fit_saved_verified", **identity, "result": result_item,
            "optimizer_calls": STEPS, "replay_optimizer_calls": 1,
            "clipped_steps": clipped, "finite_gradient_steps": STEPS,
            "checkpoints": result["checkpoints"], "selection_sha256": selection_sha256,
            "historical_prefix": {"status": "all96_original_mathematical_updates_exact",
                "saved_scientific_audit": historical["saved_scientific_audit"],
                "original_fit_result": prefix["fit_result"], "original_updates": prefix["updates"],
                "compared_steps": len(matched_mathematics), "mathematics_sha256": digest(matched_mathematics),
                "excluded_fields": historical["excluded_comparison_fields"]}}
