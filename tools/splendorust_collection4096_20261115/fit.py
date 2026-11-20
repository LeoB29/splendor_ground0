"""Discarded qualification updates only; unchanged accepted numerical operations.

No tensors/models execute on import. Actual worker admission and clocks are owned
by runtime.bootstrap; direct API callers must provide or obtain startup sources.
"""
from __future__ import annotations
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
from . import common as c
from . import sampling as data
ROOT, IDENTITY, STUDY = c.ROOT, data.IDENTITY, data.STUDY
require, artifact, save, canonical, digest = c.require, c.artifact, c.save, c.canonical, c.digest
STEPS, BATCH = 96, 32
CHECKPOINTS, SCOPES = (0, 95, 96), ("joint", "policy")
_STARTUP = _ENGINE = None
fit_identity = data.fit_identity


def startup(preverified=None):
    """Runtime bootstrap owns the once-per-worker guard; direct calls are guarded."""
    global _STARTUP
    if _STARTUP is None:
        _STARTUP = deepcopy(preverified) if preverified is not None else guard.verify_runtime_sources()
        require(type(_STARTUP) is dict, "actual corrected source verification receipt")
    elif preverified is not None:
        require(canonical(_STARTUP) == canonical(preverified), "one unchanged startup source receipt")
    return deepcopy(_STARTUP)


def engine(folder, *, runtime_sources=None):
    global _ENGINE
    sources = startup(runtime_sources)
    if _ENGINE is None:
        import psutil
        process = psutil.Process(); process.cpu_affinity([0]); process.nice(psutil.NORMAL_PRIORITY_CLASS)
        require(process.cpu_affinity() == [0], "actual CPU0 training/evaluation owner")
        import torch
        torch.set_num_threads(1); torch.set_num_interop_threads(1)
        _ENGINE = dict(pid=process.pid, created=process.create_time(), cpu_affinity=[0],
                       priority="NORMAL_PRIORITY_CLASS", torch_threads=1, torch_interop_threads=1,
                       device="cpu", precision="float32")
    save(Path(folder) / "startup.json", dict(runtime_sources=sources, engine=deepcopy(_ENGINE), **IDENTITY))
    return deepcopy(_ENGINE)


def source_equivalence():
    """Model-free AST proof of the accepted optimizer, loss-call and RNG mechanics."""
    import ast
    original = ROOT / "tools/splendorust_shorter_joint_20261106/training.py"
    expected = "b4e7b45197adb4868589caefee279d03c4972a32299f493364de1715ce0742d0"
    require(artifact(original)["sha256"] == expected, "accepted one-pass numerical source pin")
    def functions(path):
        return {x.name: x for x in ast.parse(Path(path).read_text(encoding="utf8")).body if isinstance(x, ast.FunctionDef)}
    old, new = functions(original), functions(__file__)
    names = ("scoped_parameters", "make_optimizer", "step", "rng_state", "restore_rng",
             "tensor_tree", "state_mapping_hashes", "verify_optimizer_payload")
    for name in names:
        require(ast.dump(old[name], include_attributes=False) == ast.dump(new[name], include_attributes=False),
                "unchanged qualified numerical function: " + name)
    pins = {"tools/splendorust_policy_head_20261104/training.py": "cd822a5bb6642c438ed24224956b8cc9cc5e9557bb4063ac8e7cb6626f007e48",
            "tools/splendorust_policy_head_20261104/data.py": "996d840138e510bd5c2294873149b069731d79742ceaa1e821d95e449a5c2ab5"}
    for path, sha in pins.items(): require(artifact(ROOT/path)["sha256"] == sha, "unchanged accepted loss/forward/collator")
    return dict(status="unchanged_joint_update_math_verified", functions=list(names),
                inputs=[artifact(original), *[artifact(ROOT/p) for p in pins]],
                repetition_rng_seeds=list(data.SEEDS), models=0, optimizer_calls=0)


def seed_rng(repetition):
    import numpy as np
    import torch
    require(type(repetition) is int and repetition in (0, 1), "strict fixed RNG repetition")
    seed = data.SEEDS[repetition]
    random.seed(seed); np.random.seed(seed); torch.manual_seed(seed)
    return seed


def read_json(item, *, boundary=lambda: None):
    c.verify([item], boundary=boundary)
    return c.read(c.resolve(item))


def validate_loaded(loaded):
    require(len(loaded["train_examples"]) == 3072 and loaded["train_weights"] == [1.] * 3072,
            "exact3072 equal-weight selected training roots")
    require(all(x["split"] == "train" for x in loaded["train_examples"]), "held-out targets excluded from fitting")
    require(len(loaded["batches"]) == STEPS and all(type(batch) is list and len(batch) == BATCH and
            all(type(i) is int and 0 <= i < 3072 for i in batch) for batch in loaded["batches"]), "96 strict full batches")
    require(sorted(i for b in loaded["batches"] for i in b) == list(range(3072)), "exactly one pass without replacement")
    require(loaded["probe_batches"] == [] and "heldout_examples" not in loaded, "no hidden evaluation or diagnostic probes")
    return loaded


def metadata(selection, loaded, identity):
    return {**identity, **IDENTITY, "study_id": STUDY, "selection_sha256": digest(selection),
            "source_receipt_sha256": digest(loaded["source_receipt"]),
            "historical_warm_start_sha256": accepted.PARENT_SHA256,
            "historical_data_relabelled": False, "model_source_sha256": accepted.MODEL_SOURCE_SHA256,
            "parameter_count": accepted.PARAMETERS, "fixed_final_step": STEPS,
            "checkpoint_selection": False, "training_records": 3072,
            "source_training_games": len(loaded["train_game_ids"]),
            "roots_per_game": 16 if identity["breadth"] == "narrow" else 3,
            "optimizer_recipe": {k: v for k, v in accepted.RECIPE.items() if k not in
                                 ("epochs", "shuffle", "example_weight", "batch_reduction")},
            "batch_reduction": "mean32_full_equal_game_record_weights", "record_presentations": STEPS * BATCH,
            "schedule_sha256": digest(loaded["batches"]), "seed": data.SEEDS[identity["repetition"]],
            "complete_passes": 1, "schedule_relation": "one_complete_pass_of_fixed3072_fresh_roots",
            "diagnostic_gradient_probes": 0}


def run_qualification(folder, *, boundary=lambda: None, runtime_sources=None):
    """Two discarded updates before collection, on genuine engineering outcomes."""
    from . import corpus
    folder = Path(folder); engine(folder, runtime_sources=runtime_sources)
    import torch
    math_proof = source_equivalence()
    qualification = corpus.qualify_training_inputs(folder/"data", boundary=boundary)
    examples, weights = qualification["qualification_examples"], qualification["qualification_weights"]
    require(len(examples) == BATCH and weights == [1.] * BATCH, "32 genuine engineering targets")
    require(all(x["split"] not in ("validation", "offline") for x in examples), "no scientific held-out qualification")
    batch = data.collate(examples, weights)
    require(batch["value"].tolist() == [x["value"] for x in examples], "actual trainer preserves official actor values")
    results = []
    for scope in SCOPES:
        boundary(); seed_rng(0); model = accepted.load_parent(); opt = make_optimizer(model, scope)
        meta = dict(fit_id=f"qualification_{scope}", repetition=0, breadth="engineering", scope=scope,
                    study_id=STUDY, historical_warm_start_sha256=accepted.PARENT_SHA256,
                    historical_data_relabelled=False, qualification_only=True,
                    source_receipt_sha256=digest(qualification["proof"]), **IDENTITY)
        before_groups = accepted.group_digests(model)
        with torch.no_grad():
            model.train(); before = accepted.tensor_forward(model, batch["tokens"]).tolist()
        update = recorded_step(model, opt, batch, scope, boundary=boundary)
        require(accepted.full_output_bytes(before) == accepted.full_output_bytes(update["outputs83"]), "actual update-forward parity")
        with torch.no_grad(): after = accepted.tensor_forward(model, batch["tokens"]).tolist()
        value_proof = accepted.verify_value_pair(before, after, "one-step qualification") if scope == "policy" else None
        item = save_checkpoint(model, opt, 1, meta, folder/scope/"checkpoint", boundary=boundary)
        del model, opt
        model, opt, _ = load_checkpoint(item, optimizer=True, boundary=boundary); model.train()
        with torch.no_grad(): reloaded = accepted.tensor_forward(model, batch["tokens"]).tolist()
        require(accepted.full_output_bytes(after) == accepted.full_output_bytes(reloaded), "saved reload exact outputs")
        results.append(dict(scope=scope, trainable_parameters=sum(p.numel() for p in scoped_parameters(model, scope)),
                            before_group_sha256=before_groups, step=update, before_outputs83=before,
                            after_outputs83=after, reloaded_outputs83=reloaded, value_preservation=value_proof,
                            checkpoint=item, optimizer_calls=1, forward_calls=4, backward_calls=1, discarded=True))
        del model, opt
    result = dict(schema="independent_data_qualification_v1", status="qualification_passed", study_id=STUDY,
                  data_qualification=qualification["proof"], numerical_equivalence=math_proof,
                  root_identities=[[x["game_index"],x["ply"],x["row_sha256"]] for x in examples],
                  scopes=results, optimizer_calls=2, forward_calls=8, backward_calls=2, scientific_updates=0, **IDENTITY)
    save(folder/"qualification_result.json", result); boundary(); return result


def verify_qualification(folder, *, boundary=lambda: None):
    item = artifact(Path(folder)/"qualification_result.json"); result = read_json(item, boundary=boundary)
    require(result["status"] == "qualification_passed" and result["study_id"] == STUDY and
            result["optimizer_calls"] == 2 and result["forward_calls"] == 8 and result["backward_calls"] == 2 and
            result["scientific_updates"] == 0 and [x["scope"] for x in result["scopes"]] == list(SCOPES), "complete qualification work")
    for row in result["scopes"]:
        require(row["discarded"] is True and row["optimizer_calls"] == 1 and
                row["trainable_parameters"] == (accepted.PARAMETERS if row["scope"] == "joint" else 20817), "exact qualification scope")
        require(accepted.full_output_bytes(row["before_outputs83"]) == accepted.full_output_bytes(row["step"]["outputs83"]) and
                accepted.full_output_bytes(row["after_outputs83"]) == accepted.full_output_bytes(row["reloaded_outputs83"]), "qualification exact outputs")
        payload, manifest = checkpoint_payload(row["checkpoint"], boundary=boundary)
        require(manifest["metadata"]["scope"] == row["scope"] and manifest["metadata"]["qualification_only"] is True and
                manifest["metadata"]["checkpoint_step"] == 1 and manifest["metadata"]["group_sha256"] == row["step"]["after_group_sha256"], "qualification checkpoint")
        if row["scope"] == "policy":
            require(all(row["before_group_sha256"][k] == row["step"]["after_group_sha256"][k] for k in ("shared","value")), "frozen qualification groups")
            require(row["value_preservation"] == accepted.verify_value_pair(row["before_outputs83"], row["after_outputs83"], "saved one-step qualification"), "value parity")
        del payload
    return dict(status="qualification_saved_verified", optimizer_calls=2, result=item)


def create_commitment(selection, fit_folders, destination, *, boundary=lambda: None):
    """All four saved fits must be accepted before the first held-out prediction."""
    require(set(fit_folders) == {x["fit_id"] for x in data.fits()}, "all four fixed fits")
    fits = []
    for spec in data.fits():
        folder = Path(fit_folders[spec["fit_id"]]); boundary()
        exit_item = artifact(folder.parent/"execution.json"); actual_exit = read_json(exit_item, boundary=boundary)
        require(actual_exit["status"] == "passed" and type(actual_exit["worker_exit_code"]) is int and
                actual_exit["worker_exit_code"] == 0 and actual_exit["cleanup_errors"] == [] and
                actual_exit["error"] is None and
                actual_exit["mode"] == "fit" and actual_exit["fit_id"] == spec["fit_id"] and
                actual_exit["plan_sha256"] == c.sha(c.PLAN), "actual clean fixed-fit exit before commitment")
        verified = verify_fit(selection, spec["repetition"], spec["breadth"], "joint", folder, boundary=boundary)
        audit_item = save(folder/"saved_verification.json", verified)
        fits.append(dict(spec=spec, fit_result=verified["result"], saved_verification=audit_item,
                         final_checkpoint=verified["checkpoints"]["96"], actual_exit=exit_item))
    return save(destination, dict(schema="independent_data_all_fits_committed_v1", status="all_four_fits_committed",
                    study_id=STUDY, selection_sha256=digest(selection), source_receipt_sha256=digest(selection["source_receipt"]),
                    freeze=artifact(c.PLAN), selection=artifact(c.SELECTION), source_receipt=selection["source_receipt"],
                    fits=fits, optimizer_calls=384, replay_optimizer_calls=4, checkpoint_selection=False, time=c.now(), **IDENTITY))


def validate_commitment(selection, commitment, *, boundary=lambda: None):
    value = read_json(commitment, boundary=boundary)
    require(value["schema"] == "independent_data_all_fits_committed_v1" and value["status"] == "all_four_fits_committed" and
            value["study_id"] == STUDY and value["selection_sha256"] == digest(selection) and
            value["source_receipt_sha256"] == digest(selection["source_receipt"]) and type(value["optimizer_calls"]) is int and value["optimizer_calls"] == 384 and
            type(value["replay_optimizer_calls"]) is int and value["replay_optimizer_calls"] == 4 and value["checkpoint_selection"] is False and
            canonical([x["spec"] for x in value["fits"]]) == canonical(data.fits()), "all four prospective final checkpoints committed")
    c.verify([value["freeze"], value["selection"]], boundary=boundary)
    require(canonical(value["freeze"]) == canonical(artifact(c.PLAN)) and canonical(value["selection"]) == canonical(artifact(c.SELECTION)) and
            canonical(value["source_receipt"]) == canonical(selection["source_receipt"]), "commitment binds original freeze and accepted corpus")
    for row in value["fits"]:
        proof = read_json(row["saved_verification"], boundary=boundary); result = read_json(row["fit_result"], boundary=boundary)
        actual_exit = read_json(row["actual_exit"], boundary=boundary)
        require(actual_exit["status"] == "passed" and type(actual_exit["worker_exit_code"]) is int and actual_exit["worker_exit_code"] == 0 and
                actual_exit["cleanup_errors"] == [] and actual_exit["error"] is None and actual_exit["fit_id"] == row["spec"]["fit_id"] and
                actual_exit["mode"] == "fit" and actual_exit["plan_sha256"] == value["freeze"]["sha256"], "actual completed fit owner exit")
        manifest = read_json(row["final_checkpoint"], boundary=boundary); meta = manifest["metadata"]
        validate_checkpoint_metadata(meta)
        require(proof["status"] == "fit_saved_verified" and proof["result"] == row["fit_result"] and
                result["status"] == "fit_complete" and result["selection_sha256"] == proof["selection_sha256"] == digest(selection) and
                all(canonical(result[k]) == canonical(v) and canonical(proof[k]) == canonical(v) for k,v in row["spec"].items()) and
                proof["checkpoints"]["96"] == result["checkpoints"]["96"] == row["final_checkpoint"] and
                type(proof["optimizer_calls"]) is int and proof["optimizer_calls"] == 96 and
                type(proof["replay_optimizer_calls"]) is int and proof["replay_optimizer_calls"] == 1 and
                all(canonical(meta[k]) == canonical(v) for k,v in row["spec"].items()) and
                type(meta["checkpoint_step"]) is int and meta["checkpoint_step"] == 96 and
                meta["selection_sha256"] == digest(selection) and meta["qualification_only"] is False,
                "saved accepted exact final checkpoint and fixed identity")
        c.verify([manifest["checkpoint"], manifest["tensor_manifest"]], boundary=boundary)
    return value


def recorded_step(model, optimizer, batch, scope, *, boundary=lambda: None):
    """Preserve exact accepted update math; additionally retain every clipped gradient hash."""
    result = step(model, optimizer, batch, scope, boundary=boundary)
    result["gradient_sha256"] = {name: hashlib.sha256(p.grad.detach().contiguous().numpy().tobytes()).hexdigest()
        for name,p in model.named_parameters() if p.requires_grad}
    result["gradient_evidence"] = "all_trainable_clipped_float32_gradient_bytes"
    return result


def validate_checkpoint_metadata(meta):
    require(type(meta) is dict and meta.get("study_id") == STUDY and
            type(meta.get("checkpoint_step")) is int and type(meta.get("qualification_only")) is bool,
            "strict checkpoint metadata types")
    accepted_data.verify_identity(meta)
    if meta["qualification_only"]:
        require(meta["scope"] in SCOPES and meta["fit_id"] == "qualification_" + meta["scope"] and
                meta["checkpoint_step"] == 1 and meta["breadth"] == "engineering" and
                type(meta["repetition"]) is int and meta["repetition"] == 0, "engineering-only checkpoint")
    else:
        expected = fit_identity(meta["repetition"],meta["breadth"],meta["scope"])
        require(all(canonical(meta[k]) == canonical(v) for k,v in expected.items()), "strict scientific checkpoint identity")
        for field,value in {"training_records":3072,"record_presentations":3072,"fixed_final_step":96,
                            "complete_passes":1,"source_training_games":192 if meta["breadth"]=="narrow" else 1024,
                            "roots_per_game":16 if meta["breadth"]=="narrow" else 3,
                            "parameter_count":accepted.PARAMETERS,"diagnostic_gradient_probes":0}.items():
            require(type(meta.get(field)) is int and meta[field]==value, "strict fixed metadata: "+field)
        require(type(meta.get("seed")) is int and meta["seed"] == data.SEEDS[meta["repetition"]], "registered optimizer RNG seed")
        require(meta["checkpoint_step"] in CHECKPOINTS and meta["checkpoint_selection"] is False,
                "only fixed scientific checkpoints")
    require(meta["historical_data_relabelled"] is False and meta["historical_warm_start_sha256"]==accepted.PARENT_SHA256,
            "explicit historical warm start only")
    return meta


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
    validate_checkpoint_metadata(manifest["metadata"])
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


def selected_batch(loaded, indices):
    return data.collate([loaded["train_examples"][i] for i in indices],
                        [loaded["train_weights"][i] for i in indices])


def run_fit(selection, repetition, breadth, scope, folder, *, boundary=lambda: None, runtime_sources=None):
    folder = Path(folder)
    engine(folder, runtime_sources=runtime_sources)
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
        update = recorded_step(model, opt, selected_batch(loaded, indices), scope, boundary=boundary)
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
    replay = recorded_step(replay_model, replay_opt, selected_batch(loaded, loaded["batches"][STEPS - 1]), scope, boundary=boundary)
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
    gradient_names = None
    for index in CHECKPOINTS:
        item = result["checkpoints"][str(index)]
        payload, manifest = checkpoint_payload(item, boundary=boundary)
        meta = manifest["metadata"]
        require(all(canonical(meta[k]) == canonical(v) for k, v in metadata(selection, loaded, identity).items()) and
                meta["checkpoint_step"] == index and meta["qualification_only"] is False,
                "checkpoint exact frozen metadata")
        if index == 0:
            # Original archive hash plus exact original tensors establish warm start.
            import torch
            require(guard.sha256(accepted.PARENT_CHECKPOINT) == accepted.PARENT_SHA256, "original warm start archive")
            parent = torch.load(accepted.PARENT_CHECKPOINT, map_location="cpu", weights_only=True)
            require(tensor_tree(payload["state_dict"]) == tensor_tree(parent["state_dict"]), "all pristine original tensor bytes")
            initial_state = tensor_tree(payload["state_dict"])
            gradient_names = set(payload["state_dict"])
            require(payload["optimizer"]["state"] == {}, "fresh empty optimizer at step0")
            del parent
        manifests[index] = manifest
        del payload
    require(initial_state is not None, "parent state checked")
    groups = result["before_group_sha256"]
    require(groups == manifests[0]["metadata"]["group_sha256"], "update chain starts at parent")
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
        require(update.get("gradient_evidence") == "all_trainable_clipped_float32_gradient_bytes" and
                type(update.get("gradient_sha256")) is dict and set(update["gradient_sha256"]) == gradient_names,
                "every trainable gradient has exact byte evidence")
        for value in update["gradient_sha256"].values(): data._sha(value)
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
            "fresh_source_games": len(loaded["train_game_ids"]), "historical_prefix_comparison": False}


def main():
    import argparse
    import traceback
    import psutil
    from . import runtime
    parser=argparse.ArgumentParser()
    parser.add_argument("--mode",choices=("qualification",),required=True)
    parser.add_argument("--plan-sha256",required=True);parser.add_argument("--output",required=True)
    args=parser.parse_args();folder=Path(args.output).resolve()
    require(folder==(c.OUT/"training_qualification/worker").resolve(), "exact qualification folder")
    plan,boundary,sources=runtime.bootstrap(folder,args.plan_sha256,args.mode,model=True)
    startup(sources)
    identity=dict(pid=os.getpid(),created=psutil.Process().create_time())
    status,result,error="failed",None,None
    try:
        result=run_qualification(folder,boundary=boundary,runtime_sources=sources)
        boundary();status="passed"
    except BaseException:
        error=traceback.format_exc()
    receipt=dict(status=status,mode=args.mode,fit_id=None,identity=identity,result=result,error=error,
                 plan_sha256=args.plan_sha256,automatic_retries=0,checkpoint_selection=False)
    save(folder/"result.json",receipt)
    if status!="passed":save(folder/"failure.json",receipt);print(error,flush=True);return 1
    print(canonical(dict(status=status,mode=args.mode)),flush=True);return 0


if __name__ == "__main__":
    raise SystemExit(main())
