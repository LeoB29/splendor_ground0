"""One fixed corrected-data update, authenticated save/reload, no import-time model.

Actual model work is permitted only inside the caller's separately frozen owner.
The qualification path is one update, never a substitute for the two-epoch pilot.
"""
from __future__ import annotations

import ast
from copy import deepcopy
import hashlib
import json
import math
import os
from pathlib import Path
import time
import warnings
import zipfile

from . import data
from . import clock_contract as clocks
from tools import future_training_rules_20261021 as guard

ROOT = data.ROOT
MODEL_SOURCE = ROOT / "outputs/external_opponent_search_20261018_02/splendorust/research/architecture_pivots/models.py"
MODEL_SOURCE_SHA256 = "0a554cd2a7b21b76a00add1157ab10b9cbdadd94b2b0e82097da2d9c06054417"
PARENT_CHECKPOINT = ROOT / "outputs/external_opponent_search_20261018_02/splendorust_model.pt"
PARENT_SHA256 = "ef8a4521cd6c03c7075f15efee23f4cde6ec94d1b5398765cf0c24a09ad745cb"
PARAMETERS = 4_780_883
RECIPE = {"optimizer": "AdamW", "learning_rate": 1e-5, "weight_decay": 1e-4,
          "betas": [.9, .999], "epsilon": 1e-8, "amsgrad": False, "foreach": False,
          "fused": False, "batch_size": 32, "gradient_norm_cap": 1.0, "epochs": 2,
          "policy_loss_weight": 1.0, "value_loss_weight": 1.0,
          "value_reduction": "mean_of_two_signed_outputs", "policy_mask": "all_corrected_legal_main_channels",
          "example_weight": "N/(G*n_game)", "batch_reduction": "weighted_sum_divided_by_fixed_32_including_short_batch",
          "shuffle": "ascending_sha256_canonical_[seed,epoch,example_index]_without_replacement",
          "checkpoint_selection": "fixed_final_update_only", "precision": "float32", "device": "cpu"}
_STARTUP_RECEIPT = None
require = data.require


def qualification_plan():
    return {"schema": "selfplay_training_plan_v1", "mode": "qualification", "seed": 202611011,
            "recipe": deepcopy(RECIPE), "training_split": "fixture_train", "optimizer_steps": 1,
            "parity_cases_per_model": 11, "graph_builds": 2, "graph_trace_warm_forwards": 8,
            "parity_forward_calls": 44, "training_forward_backward_calls": 1,
            "base_direct_forward_invocations": 53, "maximum_training_batch": 32,
            "heldout_diagnostics": heldout_policy(),
            "learned_entrypoint_decisions": 6, "learned_entrypoint_simulations_each": 128,
            "direct_forward_count_excludes_entrypoint_callbacks": True,
            "training_game_count": 2, "qualification_only": True, "pilot_data": False}


def pilot_plan():
    return {"schema": "selfplay_training_plan_v1", "mode": "pilot", "seed": 202611012,
            "recipe": deepcopy(RECIPE), "training_split": "train", "epochs": 2,
            "parity_cases_per_model": 11, "graph_builds": 2, "training_game_count": 192,
            "qualification_only": False, "pilot_data": True, "heldout_diagnostics": heldout_policy(),
            "learned_entrypoint_decisions": 6, "learned_entrypoint_simulations_each": 128,
            "direct_forward_count_excludes_entrypoint_callbacks": True,
            "offline_opening": "only_after_final_checkpoint_manifest_is_durably_saved"}


def heldout_policy():
    return {"models": ["parent", "candidate"], "batch_size": 32,
            "all_eligible_roots": True, "opening": "after_candidate_manifest_fixed",
            "reduction": "mean_within_game_then_mean_across_games",
            "losses": ["masked_policy_cross_entropy", "mean_two_signed_value_squared_errors"],
            "checkpoint_selection": False, "outputs_saved_losslessly": True}


def heldout_splits(plan):
    return ["fixture_validation", "fixture_assessment"] if plan["qualification_only"] else ["validation", "offline"]


def direct_forward_budget(plan, example_counts=None):
    """Exact corpus-derived work, or the finite2000-plies-per-source maximum."""
    validate_plan(plan)
    if example_counts is None:
        examples = {plan["training_split"]: plan["training_game_count"] * data.MAX_PLIES}
        examples.update({s: (1 if plan["qualification_only"] else 32) * data.MAX_PLIES for s in heldout_splits(plan)})
    else:
        examples = example_counts
    require(all(type(examples.get(s)) is int and examples[s] > 0 for s in [plan["training_split"], *heldout_splits(plan)]),
            "nonempty complete source counts for forward accounting")
    updates = 1 if plan["qualification_only"] else 2 * math.ceil(examples[plan["training_split"]] / 32)
    heldout = 2 * sum(math.ceil(examples[s] / 32) for s in heldout_splits(plan))
    return {"graph_trace_warm_forwards": 8, "parity_forward_calls": 44,
            "training_forward_backward_calls": updates, "heldout_forward_calls": heldout,
            "direct_forward_invocations": 52 + updates + heldout,
            "learned_entrypoint_decisions": 6, "learned_entrypoint_simulations_each": 128,
            "direct_count_excludes_search_callbacks": True}


def validate_plan(plan):
    expected = qualification_plan() if plan.get("mode") == "qualification" else pilot_plan()
    require(data.canonical(plan) == data.canonical(expected), "exact frozen training recipe and mode")
    return plan


def startup():
    """Every fitter process calls the mandatory guard once, including fit-only."""
    global _STARTUP_RECEIPT
    if _STARTUP_RECEIPT is None:
        _STARTUP_RECEIPT = {"runtime_sources": guard.verify_runtime_sources(), **data.IDENTITY}
    return deepcopy(_STARTUP_RECEIPT)


def artifact(path):
    path = Path(path).resolve()
    require(path.is_relative_to(ROOT), "workspace artifact")
    return {"path": path.relative_to(ROOT).as_posix(), "bytes": path.stat().st_size,
            "sha256": guard.sha256(path)}


def save_json(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("x", encoding="utf-8") as stream:
        stream.write(data.canonical(value) + "\n"); stream.flush(); os.fsync(stream.fileno())
    return artifact(path)


def _source_namespace():
    import torch
    require(guard.sha256(MODEL_SOURCE) == MODEL_SOURCE_SHA256, "authenticated exact historical architecture")
    names = {"normalized", "entities", "entities_fast", "EntityTransformer"}
    tree = ast.parse(MODEL_SOURCE.read_text(encoding="utf-8"))
    nodes = [node for node in tree.body if isinstance(node, (ast.FunctionDef, ast.ClassDef)) and node.name in names]
    require({n.name for n in nodes} == names and len(nodes) == 4, "exact architecture definitions")
    namespace = {"torch": torch, "nn": torch.nn, "_ENTITY_CACHE": {}}
    exec(compile(ast.Module(body=nodes, type_ignores=[]), str(MODEL_SOURCE), "exec"), namespace)
    return namespace


def validate_architecture(model):
    import torch
    require(model.has_history is False and len(model.blocks.layers) == 6 and model.identity.shape == (31, 256),
            "fixed nonhistory six-layer architecture")
    require(model.project.in_features == 48 and model.project.out_features == 256 and
            model.policy.in_features == 256 and model.policy.out_features == 81 and
            model.value.in_features == 256 and model.value.out_features == 2, "unchanged input and both heads")
    for block in model.blocks.layers:
        require(block.self_attn.num_heads == 8 and block.self_attn.batch_first and block.norm_first and
                block.linear1.out_features == 1024 and block.linear2.in_features == 1024 and
                block.dropout.p == block.dropout1.p == block.dropout2.p == 0.0,
                "unchanged attention/FFN/pre-norm/dropout")
    require(sum(p.numel() for p in model.parameters()) == PARAMETERS, "exact parameter count")
    require(all(t.dtype == torch.float32 and t.device.type == "cpu" and bool(torch.isfinite(t).all())
                for t in model.state_dict().values()), "finite unchanged CPU float32 state")


def state_digest(model, prefix=None):
    import torch
    result = hashlib.sha256()
    for name, tensor in sorted(model.state_dict().items()):
        selected = prefix is None or (name.startswith(prefix) if prefix in ("policy.", "value.")
                                      else not name.startswith(("policy.", "value.")))
        if not selected:
            continue
        require(tensor.dtype == torch.float32 and tensor.device.type == "cpu", "float32 state hash")
        result.update(data.canonical([name, list(tensor.shape), str(tensor.dtype)]).encode())
        result.update(b"\0"); result.update(tensor.detach().contiguous().numpy().tobytes())
    return result.hexdigest()


def group_digests(model):
    return {"shared": state_digest(model, "shared"), "policy": state_digest(model, "policy."),
            "value": state_digest(model, "value.")}


def load_parent():
    import torch
    require(guard.sha256(PARENT_CHECKPOINT) == PARENT_SHA256, "immutable historical warm start")
    model = _source_namespace()["EntityTransformer"]()
    payload = torch.load(PARENT_CHECKPOINT, map_location="cpu", weights_only=True)
    require(payload.get("kind") == "entity", "historical checkpoint kind")
    model.load_state_dict(payload["state_dict"], strict=True)
    validate_architecture(model)
    model.eval()
    return model


def tensor_forward(model, tokens):
    """Trainable execution of the exact existing31x48 runtime formula."""
    import torch
    require(tokens.dtype == torch.float32 and tokens.device.type == "cpu" and
            tokens.ndim == 3 and tuple(tokens.shape[1:]) == (31, 48), "actual training tensor contract")
    h = model.norm(model.blocks(model.project(tokens) + model.identity)[:, 0])
    return torch.cat((model.policy(h), model.value(h).tanh()), dim=1)


def compile_inference(model, checkpoint, folder, *, boundary=lambda: None, save=save_json):
    """Same graph math/flags: one trace and exactly three zero warm forwards."""
    import torch
    from tools.splendorust_inference_opt_20261025 import graph
    validate_architecture(model)
    require(torch.get_num_threads() == torch.get_num_interop_threads() == 1, "one-thread inference runtime")
    require(torch.backends.mha.get_fastpath_enabled() and torch.backends.mkldnn.enabled,
            "unchanged accepted MHA and MKLDNN runtime flags")
    model.eval(); before = state_digest(model)
    require(not any(m.training for m in model.modules()), "eval graph only")
    folder = Path(folder); folder.mkdir(parents=True, exist_ok=True)
    with torch.inference_mode():
        forward = graph._make_forward(torch, model)
        boundary()
        with warnings.catch_warnings(record=True) as messages:
            warnings.simplefilter("always")
            traced = torch.jit.trace(forward, torch.zeros((1, 31, 48), dtype=torch.float32),
                                     check_trace=False, strict=True)
            compiled = torch.jit.freeze(traced, optimize_numerics=False)
        boundary()
        for batch in (1, 2, 8):
            output = compiled(torch.zeros((batch, 31, 48), dtype=torch.float32))
            require(output.shape == (batch, 83) and output.dtype == torch.float32 and
                    bool(torch.isfinite(output).all()), "compiled zero-warm shape/finite")
            boundary()
    require(state_digest(model) == before, "compilation preserves weights")
    ir = str(compiled.inlined_graph) + "\n"
    require(len(ir.encode()) <= 256 * 1024, "bounded graph text")
    with (folder / "graph_ir.txt").open("x", encoding="utf-8", newline="\n") as stream:
        stream.write(ir); stream.flush(); os.fsync(stream.fileno())
    identity = {"schema": "selfplay_inference_graph_v1", "checkpoint": deepcopy(checkpoint),
                "weights_sha256": before, "model_source_sha256": MODEL_SOURCE_SHA256,
                "architecture_parameters": PARAMETERS, "backend": graph.BACKEND,
                "graph_options": deepcopy(graph.GRAPH_OPTIONS), "graph_ir": artifact(folder / "graph_ir.txt"),
                "runtime": {"torch_version": str(torch.__version__), "threads": 1, "interop_threads": 1,
                            "mha_fastpath_enabled": True, "mkldnn_enabled": True},
                "warnings": [{"category": x.category.__name__, "message": str(x.message)} for x in messages],
                "graph_forward_calls": 4, "source_checkpoint_claim_is_historical_parent": checkpoint["sha256"] == PARENT_SHA256}
    save(folder / "graph_identity.json", identity)
    return compiled, identity


def verify_parity(model, compiled, fixtures, *, boundary=lambda: None):
    import torch
    from tools.splendorust_inference_opt_20261025 import graph
    require(type(fixtures) is list and len(fixtures) == 11, "exact eleven frozen graph fixtures")
    rows = []
    with torch.inference_mode():
        for fixture in fixtures:
            boundary(); batch = fixture["batch"]
            require(batch in (1, 2, 8) and len(fixture["tokens"]) == batch * 31 * 48, "parity input shapes")
            x = torch.tensor(fixture["tokens"], dtype=torch.float32).reshape(batch, 31, 48)
            require(bool(torch.isfinite(x).all()), "finite parity inputs")
            eager = tensor_forward(model, x).tolist(); boundary()
            actual = compiled(x).tolist(); boundary()
            require(graph.float32_bytes(eager, batch) == graph.float32_bytes(actual, batch),
                    "same-checkpoint eager/compiled exact floats and bits")
            rows.append({"fixture_id": fixture["fixture_id"], "batch": batch,
                         "input_sha256": data.digest(fixture), "eager_outputs": eager,
                         "compiled_outputs": actual, "float32_bits_equal": True})
    return {"status": "exact_eager_compiled_parity", "cases": 11, "forward_calls": 22, "rows": rows}


def select_training_examples(corpus, plan):
    validate_plan(plan)
    require(corpus["qualification_only"] is plan["qualification_only"], "qualification/pilot data separation")
    examples = [x for x in corpus["examples"] if x["split"] == plan["training_split"]]
    game_ids = [g["spec"]["game_index"] for g in corpus["games"] if g["spec"]["split"] == plan["training_split"]]
    require(len(game_ids) == plan["training_game_count"] and len(set(game_ids)) == len(game_ids), "all fixed training games")
    counts = {g: sum(x["game_index"] == g for x in examples) for g in game_ids}
    require(bool(examples) and all(n > 0 for n in counts.values()) and
            set(x["game_index"] for x in examples) == set(game_ids), "no training game silently dropped")
    n, g = len(examples), len(game_ids)
    weights = [n / (g * counts[x["game_index"]]) for x in examples]
    return examples, weights, counts


def epoch_order(count, seed, epoch):
    require(type(count) is int and count > 0 and type(seed) is int and type(epoch) is int and epoch >= 0,
            "fixed finite deterministic epoch inventory")
    return sorted(range(count), key=lambda index: (data.digest([seed, epoch, index]), index))


def collate_inputs(examples, weights):
    import torch
    arrays = data.collate_arrays(examples, weights)
    batch = {k: torch.from_numpy(v.copy()) for k, v in arrays.items()}
    require(all(batch[k].tolist() == arrays[k].tolist() for k in batch), "actual tensor conversion preserves shared collation")
    return batch


def collate(examples, weights):
    import torch
    batch = collate_inputs(examples, weights)
    value = torch.tensor([x["value"] for x in examples], dtype=torch.float32)
    require(value.shape == (len(examples), 2) and bool(torch.isfinite(value).all()) and
            bool(((value == -1) | (value == 0) | (value == 1)).all()) and bool((value.sum(dim=1) == 0).all()),
            "official actor/opponent zero-sum signed targets")
    return {**batch, "value": value}


def loss_components(outputs, batch):
    import torch
    require(outputs.shape == (batch["tokens"].shape[0], 83) and outputs.dtype == torch.float32 and
            bool(torch.isfinite(outputs).all()), "finite two-head forward")
    logits = outputs[:, :81].masked_fill(~batch["mask"], float("-inf"))
    logp = torch.log_softmax(logits, dim=1)
    # Zero outside support before multiplication: 0 * -inf would yield NaN.
    logp = torch.where(batch["mask"], logp, torch.zeros_like(logp))
    policy = -(batch["policy"] * logp).sum(dim=1)
    value = ((outputs[:, 81:] - batch["value"]) ** 2).mean(dim=1)
    total = ((policy + value) * batch["weights"]).sum() / 32.0
    require(bool(torch.isfinite(total)), "finite policy/value objective")
    return total, policy, value


def make_optimizer(model):
    import torch
    return torch.optim.AdamW(model.parameters(), lr=1e-5, betas=(.9, .999), eps=1e-8,
                            weight_decay=1e-4, amsgrad=False, foreach=False, fused=False)


def train_step(model, optimizer, batch, *, boundary=lambda: None):
    import torch
    boundary(); started = time.perf_counter()
    model.train()
    optimizer.zero_grad(set_to_none=True)
    outputs = tensor_forward(model, batch["tokens"])
    total, policy, value = loss_components(outputs, batch)
    total.backward(); boundary()
    require(all(p.grad is not None and bool(torch.isfinite(p.grad).all()) for p in model.parameters()),
            "every trainable parameter has finite gradients")
    norm = torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0, error_if_nonfinite=True, foreach=False)
    optimizer.step(); boundary()
    require(all(bool(torch.isfinite(p).all()) for p in model.parameters()), "finite optimizer result")
    return {"batch_examples": len(batch["tokens"]), "loss": float(total.detach()),
            "policy_mean_unweighted": float(policy.detach().mean()), "value_mean_unweighted": float(value.detach().mean()),
            "gradient_norm_before_clip": float(norm), "seconds": time.perf_counter() - started,
            "forward_backward_calls": 1, "batch_denominator": 32,
            "outputs83": outputs.detach().tolist()}


def write_checkpoint(model, corpus, plan, folder, *, save=save_json):
    import torch
    model.eval(); validate_architecture(model)
    folder = Path(folder); folder.mkdir(parents=True, exist_ok=True)
    metadata = {"schema": "selfplay_checkpoint_metadata_v1", **data.IDENTITY,
                "historical_warm_start_sha256": PARENT_SHA256, "historical_data_relabelled": False,
                "model_source_sha256": MODEL_SOURCE_SHA256, "parameter_count": PARAMETERS,
                "weights_sha256": state_digest(model), "training_corpus_sha256": corpus["manifest_sha256"],
                "training_plan_sha256": data.digest(plan), "qualification_only": plan["qualification_only"],
                "candidate_selected_by_assessment": False}
    path = folder / "candidate.pt"
    with path.open("xb") as stream:
        torch.save({"kind": "entity", "state_dict": model.state_dict(), "metadata": metadata}, stream)
        stream.flush(); os.fsync(stream.fileno())
    # Non-model readers can independently hash the actual serialized storages.
    tensors = []
    with zipfile.ZipFile(path) as archive:
        roots = {name.split("/", 1)[0] for name in archive.namelist()}
        require(len(roots) == 1, "one PyTorch ZIP root")
        root = next(iter(roots))
        require(archive.read(root + "/byteorder") == b"little", "CPU little-endian tensor serialization")
        for index, (name, tensor) in enumerate(model.state_dict().items()):
            require(tensor.is_contiguous(), "fixed model tensor contiguous serialization")
            raw = tensor.detach().numpy().tobytes()
            member = root + "/data/" + str(index)
            require(archive.read(member) == raw, "serialized tensor storage equals actual model state")
            tensors.append({"name": name, "shape": list(tensor.shape), "dtype": "torch.float32",
                            "numel": tensor.numel(), "bytes": len(raw), "sha256": hashlib.sha256(raw).hexdigest(),
                            "zip_member": member})
        require({x for x in archive.namelist() if x.startswith(root + "/data/")} ==
                {x["zip_member"] for x in tensors}, "no unaccounted checkpoint tensors")
    tensor_item = save(folder / "checkpoint_tensors.json", {"schema": "selfplay_checkpoint_tensors_v1",
                       "checkpoint": artifact(path), "tensors": tensors, "weights_sha256": metadata["weights_sha256"]})
    manifest = {"schema": "selfplay_candidate_checkpoint_v1", "checkpoint": artifact(path),
                "model_definition": artifact(MODEL_SOURCE), "tensor_manifest": tensor_item, "metadata": metadata}
    item = save(folder / "checkpoint_manifest.json", manifest)
    require(item == artifact(folder / "checkpoint_manifest.json"), "saved checkpoint manifest descriptor")
    return manifest, item


def load_candidate(manifest_descriptor, *, allow_qualification=False, boundary=lambda: None):
    """Caller must supply an externally frozen/admitted manifest descriptor."""
    import torch
    manifest = data.read_artifact(manifest_descriptor, boundary=boundary)
    require(set(manifest) == {"schema", "checkpoint", "model_definition", "tensor_manifest", "metadata"} and
            manifest["schema"] == "selfplay_candidate_checkpoint_v1", "candidate manifest schema")
    metadata = manifest["metadata"]; data.verify_identity(metadata)
    require(metadata.get("schema") == "selfplay_checkpoint_metadata_v1" and
            metadata.get("historical_warm_start_sha256") == PARENT_SHA256 and
            metadata.get("historical_data_relabelled") is False and metadata.get("parameter_count") == PARAMETERS and
            metadata.get("model_source_sha256") == MODEL_SOURCE_SHA256 and
            metadata.get("candidate_selected_by_assessment") is False, "explicit corrected candidate lineage")
    require(type(metadata.get("qualification_only")) is bool and
            (not metadata["qualification_only"] or allow_qualification), "qualification checkpoint cannot become pilot candidate")
    require(manifest["model_definition"] == artifact(MODEL_SOURCE), "unchanged architecture source descriptor")
    item = manifest["checkpoint"]
    require(set(item) == {"path", "bytes", "sha256"} and type(item["bytes"]) is int and
            0 < item["bytes"] <= 32 * 1024**2, "bounded candidate checkpoint artifact")
    path = (ROOT / item["path"]).resolve()
    require(path.is_relative_to(ROOT) and path.relative_to(ROOT).as_posix() == item["path"], "canonical candidate path")
    boundary(); require(artifact(path) == item, "candidate bytes bound to external manifest")
    payload = torch.load(path, map_location="cpu", weights_only=True)
    require(set(payload) == {"kind", "state_dict", "metadata"} and payload["kind"] == "entity" and
            payload["metadata"] == metadata, "saved checkpoint and manifest identity")
    model = _source_namespace()["EntityTransformer"]()
    model.load_state_dict(payload["state_dict"], strict=True); model.eval()
    validate_architecture(model)
    require(state_digest(model) == metadata["weights_sha256"], "exact reloaded tensor state")
    boundary()
    return model, manifest


def _verify_tensor_archive(path, entries, *, boundary=lambda: None):
    """Pure ZIP/float32-byte verification; no Torch import or model construction."""
    require(type(entries) is list and entries and len({x["name"] for x in entries}) == len(entries) and
            len({x["zip_member"] for x in entries}) == len(entries), "distinct tensor inventory")
    by_name = {}
    with zipfile.ZipFile(path) as archive:
        require(len(archive.namelist()) == len(set(archive.namelist())), "no duplicate ZIP entries")
        for item in entries:
            boundary()
            require(item["dtype"] == "torch.float32" and all(type(n) is int and n > 0 for n in item["shape"]), "float32 tensor shape")
            require(math.prod(item["shape"]) == item["numel"] and item["bytes"] == 4 * item["numel"], "tensor extent")
            info = archive.getinfo(item["zip_member"])
            require(info.file_size == item["bytes"], "bounded stored tensor length")
            raw = archive.read(item["zip_member"])
            require(hashlib.sha256(raw).hexdigest() == item["sha256"], "serialized tensor SHA")
            # Exponent all-ones denotes Inf/NaN; inspect bytes without float conversion.
            require(all((int.from_bytes(raw[i:i + 4], "little") & 0x7f800000) != 0x7f800000
                        for i in range(0, len(raw), 4)), "finite saved float32 tensors")
            by_name[item["name"]] = (item, raw)
    hashes = {}
    for group in ("all", "shared", "policy", "value"):
        result = hashlib.sha256()
        for name, (item, raw) in sorted(by_name.items()):
            selected = group == "all" or (name.startswith(group + ".") if group in ("policy", "value")
                                          else not name.startswith(("policy.", "value.")))
            if selected:
                result.update(data.canonical([name, item["shape"], item["dtype"]]).encode())
                result.update(b"\0"); result.update(raw)
        hashes[group] = result.hexdigest()
    return hashes


def expected_state_shapes():
    result = {"identity": [31, 256], "project.weight": [256, 48], "project.bias": [256]}
    for i in range(6):
        for name, shape in {"self_attn.in_proj_weight": [768, 256], "self_attn.in_proj_bias": [768],
                            "self_attn.out_proj.weight": [256, 256], "self_attn.out_proj.bias": [256],
                            "linear1.weight": [1024, 256], "linear1.bias": [1024],
                            "linear2.weight": [256, 1024], "linear2.bias": [256],
                            "norm1.weight": [256], "norm1.bias": [256], "norm2.weight": [256], "norm2.bias": [256]}.items():
            result[f"blocks.layers.{i}.{name}"] = shape
    result.update({"norm.weight": [256], "norm.bias": [256], "policy.weight": [81, 256], "policy.bias": [81],
                   "value.weight": [2, 256], "value.bias": [2]})
    return result


def verify_parent_tensor_hashes(*, boundary=lambda: None):
    require(guard.sha256(PARENT_CHECKPOINT) == PARENT_SHA256, "original parent bytes retained")
    entries = []
    with zipfile.ZipFile(PARENT_CHECKPOINT) as archive:
        roots = {name.split("/", 1)[0] for name in archive.namelist()}
        require(len(roots) == 1, "parent archive root")
        root = next(iter(roots))
        for index, (name, shape) in enumerate(expected_state_shapes().items()):
            boundary(); member = root + "/data/" + str(index); raw = archive.read(member)
            entries.append({"name": name, "shape": shape, "dtype": "torch.float32", "numel": math.prod(shape),
                            "bytes": len(raw), "sha256": hashlib.sha256(raw).hexdigest(), "zip_member": member})
        require({x for x in archive.namelist() if x.startswith(root + "/data/")} ==
                {x["zip_member"] for x in entries}, "all original parent tensor storages")
    return _verify_tensor_archive(PARENT_CHECKPOINT, entries, boundary=boundary)


def _atomic_json(path, value):
    path = Path(path); pending = path.with_suffix(path.suffix + ".pending")
    require(not path.exists(), "exclusive completion publication")
    save_json(pending, value)
    require(not path.exists(), "exclusive completion destination")
    os.rename(pending, path)
    return artifact(path)


class _TeePipe:
    def __init__(self, stream, log):
        self.stream, self.log = stream, log
    def write(self, value):
        self.log.write(value); self.log.flush()
        return self.stream.write(value)
    def flush(self):
        return self.stream.flush()
    def readline(self, *args):
        value = self.stream.readline(*args)
        self.log.write(value); self.log.flush()
        return value
    def __getattr__(self, name):
        return getattr(self.stream, name)


def qualify_learned_entrypoint(model, compiled, candidate_manifest, declaration, folder, *, boundary=lambda: None):
    """Six actual same-checkpoint decisions, three each eager and compiled.

Each complete response is atomically published and acknowledged by the external
owning supervisor before another request starts. The child alone is not the10s
clock authority. Full native text is retained before parsing.
"""
    import psutil
    from types import SimpleNamespace
    from .capture_adapter import CaptureOpponent, validate_capture
    from tools.external_strength_screen_20261018 import rules, audit
    from tools.splendorust_inference_opt_20261025.graph import FrozenForward
    ids = ["face_T1_05", "private_reserves", "gold_returns"]
    require(declaration["case_ids"] == ids and set(declaration) == {"case_ids", "profile", "native_artifact", "observers"}, "fixed learned entrypoint cases")
    profile = data.read_artifact(declaration["profile"], boundary=boundary)
    require(declaration["profile"]["sha256"] == "048fea7e9d9377f5b3e477d4bafc96d3e96321661d2146e299a95cdc51b36edb" and
            profile["iterations"] == 128, "unchanged128 search-settings template")
    cases = {x["id"]: x for x in data.build_qualification_cases()["cases"]}
    require(declaration["observers"] == [{"id": name, "observer": cases[name]["observer"]} for name in ids], "frozen corrected observers")
    folder = Path(folder); folder.mkdir(parents=True, exist_ok=True)
    rows, closures = [], []
    for mode in ("eager", "compiled"):
        boundary()
        adapter = CaptureOpponent(profile, native_artifact=declaration["native_artifact"],
                                  shared_from=SimpleNamespace(model=model, reference={}), capture_targets=True)
        identity = {"pid": adapter.proc.pid, "created": psutil.Process(adapter.proc.pid).create_time()}
        save_json(folder / f"{mode}_native_started.json", {"identity": identity, "candidate_manifest": candidate_manifest,
                  "search_settings_template": declaration["profile"], "native_artifact": declaration["native_artifact"]})
        logs = []
        try:
            for attr in ("stdin", "stdout"):
                handle = (folder / f"{mode}_native_{attr}.jsonl").open("x", encoding="utf-8", newline="\n")
                logs.append(handle); setattr(adapter.proc, attr, _TeePipe(getattr(adapter.proc, attr), handle))
            adapter.admission_guard = boundary
            if mode == "compiled":
                adapter.forward = FrozenForward(compiled, {"candidate_manifest": candidate_manifest})
            for name in ids:
                boundary(); index = len(rows)
                start = {"schema": "selfplay_learned_request_start_v2", "request_index": index, "case_id": name, "mode": mode,
                         **clocks.start_fields(), "candidate_manifest": candidate_manifest,
                         "native_identity": identity}
                start_item = _atomic_json(folder / f"request_{index:03d}.started.json", start)
                adapter.request_deadline = start["internal_qpc_deadline"]
                action, trace = adapter.choose(cases[name]["observer"], diagnostic_capture=True)
                serialized = rules.serialize_action(action)
                before = cases[name]["snapshot"]
                after = rules.serialize_snapshot(rules.CorrectedEnv().step(guard.decode_training_snapshot(before), action))
                audit.check_step(before, serialized, after)
                capture = adapter.last_training_capture
                validate_capture(capture, cases[name]["observer"], expected_quota=128)
                finished = clocks.completion_fields(start)
                value = {"schema": "selfplay_learned_response_v2", "request_index": index, "case_id": name, "mode": mode,
                         "start": start_item, "candidate_manifest": candidate_manifest,
                         "before": before, "observer": cases[name]["observer"], "after": after,
                         "result": {"action": serialized, "diagnostics": trace}, "training_capture": capture,
                         **finished}
                require(len(data.canonical(value).encode()) <= 8 * 1024**2, "bounded complete learned response")
                item = _atomic_json(folder / f"request_{index:03d}.completed.json", value)
                ack = folder / f"request_{index:03d}.parent_accepted.json"
                while not ack.exists():
                    boundary()
                    require(time.perf_counter() < start["request_qpc_deadline"] and time.time() < start["unix_deadline"], "external response receipt acknowledgement deadline")
                    time.sleep(.01)
                accepted = json.loads(ack.read_text(encoding="utf-8"))
                require(accepted.get("accepted") is True and accepted.get("response") == item and
                        accepted.get("request_index") == index, "actual external complete receipt")
                clocks.validate_ack(start, value, accepted)
                rows.append(value)
        finally:
            adapter.close()
            for log in logs:
                log.flush(); os.fsync(log.fileno()); log.close()
            code = adapter.proc.poll()
            try:
                alive = psutil.Process(identity["pid"]).create_time() == identity["created"]
            except psutil.NoSuchProcess:
                alive = False
            closure = {"mode": mode, "identity": identity, "actual_return_code": code, "same_identity_alive": alive,
                       "intentional_shutdown": True}
            save_json(folder / f"{mode}_native_exit.json", closure); closures.append(closure)
            require(code is not None and not alive, "actual learned native owner cleanup")
    for i in range(3):
        require(rows[i]["result"] == rows[i + 3]["result"] and rows[i]["after"] == rows[i + 3]["after"] and
                rows[i]["training_capture"] == rows[i + 3]["training_capture"], "full eager/compiled same-candidate decisions exact")
    result = {"status": "learned_entrypoint_exact_reload_passed", "case_ids": ids, "decisions": 6, "simulations_each": 128,
              "actual_callback_count": sum(x["result"]["diagnostics"]["inference_calls"] for x in rows),
              "candidate_manifest": candidate_manifest, "search_settings_template_only": declaration["profile"],
              "actual_weights_are_separately_bound_candidate": True, "closed_native_helpers": closures,
              "responses": [artifact(folder / f"request_{i:03d}.completed.json") for i in range(6)]}
    save_json(folder / "result.json", result)
    return result


def qualify_saved_tensor_inputs(folder, binding, *, materialize, boundary=lambda: None):
    """The fitter exercises actual Torch collation; saved review needs no Torch."""
    study = Path(folder).parent
    paths = [study / "capture_fixtures/capture_on" / f"{i:03d}.json" for i in range(16)]
    current_binding = (set(binding) == {"source", "plan", "row_paths", "proof_path"} and binding["source"] == "current_frozen_job" and
            binding["row_paths"] == [x.resolve().relative_to(ROOT).as_posix() for x in paths] and
            binding["proof_path"] == (study / "capture_qualification.json").resolve().relative_to(ROOT).as_posix())
    if not current_binding:
        freeze = _read(study / "freeze.json")
        data.validate_preserved_input_qualification(binding, freeze["source_continuation"], boundary=boundary)
        paths = [ROOT / path for path in binding["row_paths"]]
    proof_path = ROOT / binding["proof_path"]
    cases = data.read_artifact(binding["plan"], boundary=boundary)["loader_cases"]
    require(cases == data.build_qualification_cases(), "frozen actual-loader counterexamples")
    rows = []
    for path in paths:
        boundary(); rows.append(_read(path))
    qualification = data.qualify_captured_rows(cases, rows, boundary=boundary)
    require(_read(proof_path)["actual_loader"] == qualification,
            "same actual precollection corrected-loader evidence")
    decoded = [data.decode_row(row) for row in rows]
    eligible = [x for x in decoded if x["eligible"]]
    arrays = data.collate_arrays(eligible, [1.] * len(eligible))
    if materialize:
        actual = collate_inputs(eligible, [1.] * len(eligible))
        require(all(actual[k].tolist() == v.tolist() for k, v in arrays.items()), "actual fitter tensors preserve all corrected cases")
    return {"status": "actual_fitter_tensor_counterexamples_passed", "sources": [artifact(x) for x in paths],
            "frozen_input_binding": deepcopy(binding),
            "precollection_proof": artifact(proof_path), "rows": len(eligible),
            "shape": list(arrays["tokens"].shape), "arrays_sha256": data.digest({k: v.tolist() for k, v in arrays.items()}),
            "six_corrected_faces": qualification["six_corrected_faces"], "actual_torch_conversion_in_fitter": True,
            "no_network_forward_or_optimizer_update": True, "fabricated_terminal_labels": False}


def prediction_row(example, output):
    require(type(output) is list and len(output) == 83 and all(type(v) in (int, float) and math.isfinite(v) for v in output),
            "finite complete same-input heldout output")
    import struct
    require(all(struct.unpack("<f", struct.pack("<f", x))[0] == x for x in output), "heldout output retains exact float32 values")
    require(all(-1 <= value <= 1 for value in output[81:]), "signed bounded value head")
    return {"game_index": example["game_index"], "ply": example["ply"], "split": example["split"],
            "root_input_sha256": example["root_input_sha256"], "legal_mask81": example["mask"],
            "policy_target81": example["policy"], "value_target2": example["value"], "output83": output}


def prediction_losses(row):
    """Stable float64 report math on saved float32 outputs; no model invocation."""
    mask, target, output, value = row["legal_mask81"], row["policy_target81"], row["output83"], row["value_target2"]
    require(len(mask) == len(target) == 81 and all(type(x) is bool for x in mask) and any(mask), "heldout full legal mask")
    require(all(type(x) in (int, float) and math.isfinite(x) and x >= 0 for x in target) and
            abs(math.fsum(target)-1.) <= 1e-12 and all(ok or t == 0 for ok, t in zip(mask, target)), "heldout normalized legal target")
    require(len(value) == 2 and all(x in (-1., 0., 1.) for x in value) and sum(value) == 0., "heldout official signed target")
    require(len(output) == 83 and all(math.isfinite(x) for x in output), "finite heldout logits and values")
    maximum = max(x for x, ok in zip(output[:81], mask) if ok)
    lognormalizer = maximum + math.log(math.fsum(math.exp(x-maximum) for x, ok in zip(output[:81], mask) if ok))
    policy = math.fsum(t * (lognormalizer-x) for t, x in zip(target, output[:81]) if t)
    value_loss = math.fsum((x-y)**2 for x,y in zip(output[81:], value))/2.
    require(math.isfinite(policy) and policy >= -1e-12 and math.isfinite(value_loss), "finite heldout losses")
    return {"policy": max(0., policy), "value": value_loss, "combined": max(0., policy)+value_loss}


def summarize_predictions(rows, expected_games):
    by_game = {int(game): [] for game in expected_games}
    for row in rows:
        require(row["game_index"] in by_game, "heldout game inventory")
        by_game[row["game_index"]].append(prediction_losses(row))
    require(all(by_game.values()), "every scheduled heldout game contributes")
    games = [{"game_index": game, "roots": len(losses),
              **{key: math.fsum(x[key] for x in losses)/len(losses) for key in ("policy", "value", "combined")}}
             for game, losses in sorted(by_game.items())]
    return {"games": games, "game_count": len(games), "root_count": sum(x["roots"] for x in games),
            "mean_game_losses": {key: math.fsum(x[key] for x in games)/len(games) for key in ("policy", "value", "combined")}}


def run_heldout_diagnostics(candidate, heldout, plan, checkpoint_item, folder, *, boundary=lambda: None, save=save_json):
    import torch
    folder = Path(folder); parent = load_parent(); evidence, summaries, calls = [], {}, 0
    # No optimizer or checkpoint write exists in this phase. Both outputs use
    # the same already-fixed examples and the same eager evaluation formula.
    for name, model in (("parent", parent), ("candidate", candidate)):
        model.eval(); before = state_digest(model); summaries[name] = {}
        for split in heldout_splits(plan):
            examples = [x for x in heldout["examples"] if x["split"] == split]
            games = [x["spec"]["game_index"] for x in heldout["games"] if x["spec"]["split"] == split]
            rows = []
            for offset in range(0, len(examples), 32):
                boundary(); part = examples[offset:offset+32]
                with torch.inference_mode():
                    batch = collate(part, [1.] * len(part))
                    outputs = tensor_forward(model, batch["tokens"]).tolist()
                boundary(); calls += 1
                observations = [prediction_row(x, y) for x, y in zip(part, outputs, strict=True)]
                item = save(folder / name / split / f"{offset//32:06d}.json", {"model": name, "split": split,
                    "batch_index": offset//32, "candidate_manifest_fixed": checkpoint_item,
                    "weights_sha256": before, "rows": observations})
                evidence.append(item); rows.extend(observations)
            summaries[name][split] = summarize_predictions(rows, games)
        require(state_digest(model) == before, "diagnostic forwards preserve fixed weights")
    del parent
    result = {"schema": "selfplay_heldout_diagnostics_v1", "status": "fixed_checkpoint_diagnostics_complete",
              "candidate_manifest": checkpoint_item, "forward_calls": calls, "batches": evidence,
              "summaries": summaries, "candidate_minus_parent_mean_game_losses": {
                  split: {key: summaries["candidate"][split]["mean_game_losses"][key]-summaries["parent"][split]["mean_game_losses"][key]
                          for key in ("policy", "value", "combined")} for split in heldout_splits(plan)},
              "all_eligible_roots": True, "checkpoint_selection": False, "strength_claim": False}
    save(folder / "result.json", result)
    return result


def verify_heldout_diagnostics(folder, heldout, plan, checkpoint_item, weights, *, boundary=lambda: None):
    folder = Path(folder); result = _read(folder / "result.json"); summaries, evidence, calls = {}, [], 0
    for name in ("parent", "candidate"):
        summaries[name] = {}
        for split in heldout_splits(plan):
            examples = [x for x in heldout["examples"] if x["split"] == split]
            games = [x["spec"]["game_index"] for x in heldout["games"] if x["spec"]["split"] == split]
            rows = []
            files = sorted((folder/name/split).glob("*.json"))
            require([p.name for p in files] == [f"{i:06d}.json" for i in range(math.ceil(len(examples)/32))], "exact heldout batch inventory")
            for index,path in enumerate(files):
                boundary(); value = _read(path); part = examples[index*32:(index+1)*32]
                require(value["model"] == name and value["split"] == split and value["batch_index"] == index and
                        value["candidate_manifest_fixed"] == checkpoint_item and value["weights_sha256"] == weights[name], "heldout batch fixed identities")
                require(len(value["rows"]) == len(part) and value["rows"] ==
                        [prediction_row(x, y["output83"]) for x, y in zip(part, value["rows"], strict=True)],
                        "every actual heldout input, legal mask, target and prediction bound")
                rows.extend(value["rows"]); evidence.append(artifact(path)); calls += 1
            summaries[name][split] = summarize_predictions(rows, games)
    expected = {"schema": "selfplay_heldout_diagnostics_v1", "status": "fixed_checkpoint_diagnostics_complete",
                "candidate_manifest": checkpoint_item, "forward_calls": calls, "batches": evidence, "summaries": summaries,
                "candidate_minus_parent_mean_game_losses": {split: {key: summaries["candidate"][split]["mean_game_losses"][key]-summaries["parent"][split]["mean_game_losses"][key]
                    for key in ("policy", "value", "combined")} for split in heldout_splits(plan)},
                "all_eligible_roots": True, "checkpoint_selection": False, "strength_claim": False}
    require(result == expected, "independently reconstructed heldout losses and complete batch coverage")
    return result


def _run_fit(manifest, plan, folder, *, expected_specs, parity_fixtures, learned_entrypoint, input_qualification, boundary=lambda: None, save=save_json):
    import torch
    validate_plan(plan); receipt = startup(); boundary()
    require(torch.get_num_threads() == torch.get_num_interop_threads() == 1, "fitter fixed one-thread runtime")
    folder = Path(folder); folder.mkdir(parents=True, exist_ok=True)
    save(folder / "runtime_sources.json", receipt)
    save(folder / "input_tensor_qualification.json", qualify_saved_tensor_inputs(folder, input_qualification, materialize=True, boundary=boundary))
    save(folder / "corpus_rejections.json", data.qualify_manifest_rejections(manifest, expected_specs, boundary=boundary))
    corpus = data.load_corpus(manifest, expected_specs=expected_specs, include_splits=[plan["training_split"]], boundary=boundary)
    examples, weights, counts = select_training_examples(corpus, plan)
    save(folder / "loaded_corpus.json", {k: v for k, v in corpus.items() if k not in ("examples", "manifest")})
    torch.manual_seed(plan["seed"])
    model = load_parent(); original_groups = group_digests(model); original_weights = state_digest(model)
    parent_graph, parent_identity = compile_inference(model, artifact(PARENT_CHECKPOINT), folder / "parent_graph", boundary=boundary, save=save)
    parent_parity = verify_parity(model, parent_graph, parity_fixtures, boundary=boundary)
    save(folder / "parent_parity.json", parent_parity)
    # Frozen graph owns constant weights; discard it before any training update.
    del parent_graph
    optimizer = make_optimizer(model)
    steps, selected_order = [], []
    epochs = 1 if plan["mode"] == "qualification" else 2
    for epoch in range(epochs):
        order = epoch_order(len(examples), plan["seed"], epoch)
        selected_order.append(data.digest(order))
        save(folder / "orders" / f"{epoch:03d}.json", {"epoch": epoch, "indices": order, "sha256": data.digest(order)})
        for offset in range(0, len(order), 32):
            indices = order[offset:offset + 32]
            batch = collate([examples[i] for i in indices], [weights[i] for i in indices])
            record = train_step(model, optimizer, batch, boundary=boundary)
            record.update(epoch=epoch, step=len(steps), example_indices=indices)
            record["batch_source_sha256"] = data.digest([[examples[i]["game_index"], examples[i]["ply"],
                examples[i]["row_sha256"], weights[i]] for i in indices])
            save(folder / "steps" / f"{len(steps):06d}.json", record); steps.append(record)
            if plan["mode"] == "qualification":
                break
    require(len(steps) == (1 if plan["mode"] == "qualification" else 2 * math.ceil(len(examples) / 32)), "fixed update count")
    final_groups = group_digests(model)
    require(all(final_groups[k] != original_groups[k] for k in original_groups), "shared trunk and both heads genuinely updated")
    require(state_digest(model) != original_weights and guard.sha256(PARENT_CHECKPOINT) == PARENT_SHA256,
            "new weights without changing parent file")
    del optimizer
    checkpoint, checkpoint_item = write_checkpoint(model, corpus, plan, folder, save=save)
    opening_time = clocks.sample()
    opening = {"checkpoint_manifest": checkpoint_item, "checkpoint_fixed_before_heldout_opening": True,
               "training_splits_opened": [plan["training_split"]], "clock_contract": clocks.runtime_contract(),
               "request_qpc_time": opening_time["request_qpc"], "owner_monotonic_time": opening_time["owner_monotonic"], "unix_time": opening_time["unix"]}
    save(folder / "heldout_opening.json", opening)
    heldout = data.load_corpus(manifest, expected_specs=expected_specs, include_splits=heldout_splits(plan), boundary=boundary)
    save(folder / "heldout_loaded_corpus.json", {k: v for k, v in heldout.items() if k not in ("examples", "manifest")})
    trained_digest = state_digest(model)
    del model
    reloaded, loaded_manifest = load_candidate(checkpoint_item, allow_qualification=plan["qualification_only"], boundary=boundary)
    require(loaded_manifest == checkpoint and state_digest(reloaded) == trained_digest, "saved/reloaded exact trainable state")
    compiled, candidate_identity = compile_inference(reloaded, checkpoint["checkpoint"], folder / "candidate_graph", boundary=boundary, save=save)
    candidate_parity = verify_parity(reloaded, compiled, parity_fixtures, boundary=boundary)
    save(folder / "candidate_parity.json", candidate_parity)
    diagnostics = run_heldout_diagnostics(reloaded, heldout, plan, checkpoint_item, folder / "heldout_diagnostics", boundary=boundary, save=save)
    learned = qualify_learned_entrypoint(reloaded, compiled, checkpoint_item, learned_entrypoint,
                                       folder / "learned_entrypoint", boundary=boundary)
    require(state_digest(reloaded) == trained_digest and guard.sha256(PARENT_CHECKPOINT) == PARENT_SHA256,
            "learned adapter inference does not mutate either checkpoint")
    result = {"schema": "selfplay_fit_result_v1", "status": "qualification_update_passed" if plan["qualification_only"] else "fixed_two_epoch_fit_passed",
              "qualification_only": plan["qualification_only"], "checkpoint_manifest": checkpoint_item,
              "training_plan_sha256": data.digest(plan), "corpus_sha256": corpus["manifest_sha256"],
              "training_games": len(counts), "training_roots": len(examples), "roots_per_game": counts,
              "epoch_order_sha256": selected_order, "optimizer_steps": len(steps),
              "parent_weights_sha256": original_weights, "candidate_weights_sha256": trained_digest,
              "before_group_sha256": original_groups, "after_group_sha256": final_groups,
              "parent_checkpoint_unchanged": True, "exact_state_reload": True,
              "parent_parity_cases": 11, "candidate_parity_cases": 11, "graph_builds": 2,
              "graph_trace_warm_forwards": 8, "parity_forward_calls": 44,
              "training_forward_backward_calls": len(steps), "maximum_training_batch": max(x["batch_examples"] for x in steps),
              "heldout_forward_calls": diagnostics["forward_calls"],
              "direct_forward_invocations": 52 + len(steps) + diagnostics["forward_calls"], "direct_count_excludes_search_callbacks": True,
              "learned_entrypoint_decisions": 6, "learned_entrypoint_callback_count": learned["actual_callback_count"],
              "offline_losses_opened": True, "heldout_data_opened_after_checkpoint": True,
              "heldout_diagnostics": artifact(folder / "heldout_diagnostics/result.json"),
              "final_candidate_selected_without_validation_or_assessment": True,
              "parent_graph_identity_sha256": data.digest(parent_identity), "candidate_graph_identity_sha256": data.digest(candidate_identity)}
    save(folder / "fit_result.json", result)
    return result


def qualify_update(manifest, plan, folder, *, expected_specs, parity_fixtures, learned_entrypoint, input_qualification, boundary=lambda: None, save_json=save_json):
    require(plan == qualification_plan(), "qualification-only fixed one-update plan")
    return _run_fit(manifest, plan, folder, expected_specs=expected_specs, parity_fixtures=parity_fixtures,
                    learned_entrypoint=learned_entrypoint, input_qualification=input_qualification, boundary=boundary, save=save_json)


def fit_manifest(manifest, plan, folder, *, expected_specs, parity_fixtures, learned_entrypoint, input_qualification, boundary=lambda: None, save_json=save_json):
    require(plan == pilot_plan(), "later separately frozen full-pilot recipe")
    return _run_fit(manifest, plan, folder, expected_specs=expected_specs, parity_fixtures=parity_fixtures,
                    learned_entrypoint=learned_entrypoint, input_qualification=input_qualification, boundary=boundary, save=save_json)


def _read(path):
    return json.loads(Path(path).read_text(encoding="utf-8"))


def _verify_parity_document(document, fixtures):
    from tools.splendorust_inference_opt_20261025.graph import float32_bytes
    require(document.get("status") == "exact_eager_compiled_parity" and document.get("cases") == 11 and
            document.get("forward_calls") == 22 and len(document["rows"]) == len(fixtures) == 11, "complete parity inventory")
    for row, fixture in zip(document["rows"], fixtures, strict=True):
        require(row["fixture_id"] == fixture["fixture_id"] and row["batch"] == fixture["batch"] and
                row["input_sha256"] == data.digest(fixture), "exact frozen parity input")
        require(row["float32_bits_equal"] is True and float32_bytes(row["eager_outputs"], row["batch"]) ==
                float32_bytes(row["compiled_outputs"], row["batch"]), "saved full float32 exact parity")


def _verify_learned_native_streams(folder, mode, rows, *, boundary=lambda: None):
    """Reconcile every retained native request/callback/result with six proofs."""
    from tools.external_strength_screen_20261018_02 import bridge
    paths = [Path(folder) / f"{mode}_native_{kind}.jsonl" for kind in ("stdin", "stdout")]
    streams = []
    for path in paths:
        require(path.stat().st_size <= 32 * 1024**2, "bounded learned native transcript")
        with path.open(encoding="utf-8") as stream:
            streams.append([json.loads(line) for line in stream])
    incoming, outgoing = streams; ip = op = callbacks = 0
    for row in rows:
        observer = row["observer"]; trace = row["result"]["diagnostics"]
        phases = trace["phases"]
        requests = [(bridge.encode_observation(observer), bridge.main_actions(observer))]
        phase_index = [1]
        def aux(projection, choices):
            requests.append((projection, choices)); chosen = phases[phase_index[0]]["action"]
            phase_index[0] += 1
            return chosen
        bridge.resolve_compound(observer, phases[0]["action"], aux)
        require(len(requests) == len(phases), "all learned auxiliary phases")
        for phase_number, ((projection, choices), phase) in enumerate(zip(requests, phases, strict=True)):
            boundary()
            expected = {"op": "select", "observation": projection, "legal": choices,
                        "seed": int(data.digest(observer)[:16], 16), "iterations": 128, "capture_targets": True}
            require(ip < len(incoming) and incoming[ip] == expected, "actual learned native request")
            ip += 1
            for callback_index, callback in enumerate(phase["inference"]):
                require(op < len(outgoing), "all actual learned callbacks")
                packet = outgoing[op]; op += 1
                require(packet["kind"] == "infer" and packet["callback_index"] == callback_index and
                        packet["batch"] == callback["batch"] and data.digest(packet["tokens"]) == callback["input_sha256"],
                        "native callback input binding")
                require(ip < len(incoming) and incoming[ip]["kind"] == "inference_result" and
                        data.digest(incoming[ip]["outputs"]) == callback["output_sha256"], "actual host callback output binding")
                if callback_index == 0:
                    require(packet["tokens"] == callback["tokens"] and incoming[ip]["outputs"] == callback["outputs"],
                            "full actual root arrays")
                ip += 1; callbacks += 1
            require(op < len(outgoing), "actual learned native result")
            packet = deepcopy(outgoing[op]); op += 1
            require(packet.pop("kind") == "result", "learned native result kind")
            captured = packet.pop("training_capture")
            if phase_number == 0:
                require(captured == row["training_capture"]["native_capture"], "target provenance from actual native response")
            expected_phase = {k: v for k, v in phase.items() if k not in
                              {"inference", "neural_callback_count", "neural_callback_rows", "neural_callbacks"}}
            require(packet == expected_phase, "native result equals full saved scientific phase")
    require(ip == len(incoming) and op == len(outgoing), "no unmatched learned native inputs/outputs")
    return callbacks


def verify_learned_entrypoint(folder, declaration, candidate_manifest, *, boundary=lambda: None):
    folder = Path(folder); result = _read(folder / "result.json")
    require(result["status"] == "learned_entrypoint_exact_reload_passed" and result["decisions"] == 6 and
            result["simulations_each"] == 128 and result["case_ids"] == declaration["case_ids"] and
            result["candidate_manifest"] == candidate_manifest and result["search_settings_template_only"] == declaration["profile"] and
            result["actual_weights_are_separately_bound_candidate"] is True, "complete candidate entrypoint declaration")
    cases = {x["id"]: x for x in data.build_qualification_cases()["cases"]}
    rows, prior_ack, callbacks = [], None, 0
    for i in range(6):
        boundary(); start_path = folder / f"request_{i:03d}.started.json"
        response_path = folder / f"request_{i:03d}.completed.json"
        ack_path = folder / f"request_{i:03d}.parent_accepted.json"
        start, row, ack = _read(start_path), _read(response_path), _read(ack_path)
        name, mode = declaration["case_ids"][i % 3], ("eager" if i < 3 else "compiled")
        require(start["request_index"] == row["request_index"] == ack["request_index"] == i and
                start["case_id"] == row["case_id"] == name and start["mode"] == row["mode"] == mode,
                "fixed learned request order")
        require(row["start"] == artifact(start_path) and result["responses"][i] == artifact(response_path) and
                ack["accepted"] is True and ack["response"] == artifact(response_path), "actual owner complete-response receipt")
        require(row["candidate_manifest"] == start["candidate_manifest"] == candidate_manifest, "candidate identity in every decision")
        clocks.validate_start(start)
        clocks.validate_ack(start, row, ack)
        if prior_ack is not None:
            clocks.validate_sequence(prior_ack, start)
        prior_ack = ack
        case = cases[name]
        require(row["before"] == case["snapshot"] and row["observer"] == case["observer"], "fixed corrected learned fixture")
        decoded = data.decode_row({"spec": {"purpose": "learned_checkpoint_qualification"}, "ply": 0,
            "before": row["before"], "observer": row["observer"], "after": row["after"],
            "action": row["result"]["action"], "trace": row["result"]["diagnostics"],
            "training_capture": row["training_capture"]}, expected_quota=128)
        require(decoded["eligible"] is True, "three genuine learned search decisions")
        callbacks += row["result"]["diagnostics"]["inference_calls"]
        rows.append(row)
    for i in range(3):
        require(rows[i]["result"] == rows[i + 3]["result"] and rows[i]["after"] == rows[i + 3]["after"] and
                rows[i]["training_capture"] == rows[i + 3]["training_capture"], "independently recomputed full same-candidate replay equality")
    raw_callbacks = 0
    for index, mode in enumerate(("eager", "compiled")):
        began, closed = _read(folder / f"{mode}_native_started.json"), _read(folder / f"{mode}_native_exit.json")
        require(began["identity"] == closed["identity"] and closed["actual_return_code"] is not None and
                closed["same_identity_alive"] is False and closed == result["closed_native_helpers"][index], "actual native helper closure")
        require(began["candidate_manifest"] == candidate_manifest and began["native_artifact"] == declaration["native_artifact"], "native helper/candidate binding")
        for i in range(index * 3, (index + 1) * 3):
            started = _read(folder / f"request_{i:03d}.started.json")
            require(started["native_identity"] == began["identity"] and
                    began["identity"]["created"] <= started["unix_start"], "actual request/native process identity")
        raw_callbacks += _verify_learned_native_streams(folder, mode, rows[index * 3:(index + 1) * 3], boundary=boundary)
    require(callbacks == raw_callbacks == result["actual_callback_count"], "all actual learned callback work reconciled")
    return {"status": "passed", "decisions": 6, "same_checkpoint_exact_pairs": 3, "callbacks": callbacks,
            "actual_native_closures": 2, "all9_10_second_receipts": True, "model_calls": 0}


def verify_update_records(folder, examples, weights, plan, result, *, boundary=lambda: None):
    """Reconstruct every ordered batch and loss from preserved actual outputs."""
    folder = Path(folder); steps, orders = [], []
    epochs = 1 if plan["qualification_only"] else 2
    for epoch in range(epochs):
        order = epoch_order(len(examples), plan["seed"], epoch)
        require(_read(folder / "orders" / f"{epoch:03d}.json") == {"epoch": epoch, "indices": order, "sha256": data.digest(order)},
                "independently reconstructed fixed complete epoch")
        orders.append(data.digest(order))
        offsets = [0] if plan["qualification_only"] else range(0, len(order), 32)
        for offset in offsets:
            boundary(); indices = order[offset:offset+32]
            path = folder / "steps" / f"{len(steps):06d}.json"; step = _read(path)
            require(step["epoch"] == epoch and step["step"] == len(steps) and step["example_indices"] == indices and
                    step["batch_examples"] == len(indices) and step["forward_backward_calls"] == 1 and step["batch_denominator"] == 32,
                    "exact full or partial ordered batch with fixed32 reduction")
            require(step["batch_source_sha256"] == data.digest([[examples[i]["game_index"], examples[i]["ply"], examples[i]["row_sha256"], weights[i]] for i in indices]),
                    "actual batch data and game-balanced weights")
            require(len(step["outputs83"]) == len(indices), "one actual forward output per training root")
            losses = [prediction_losses(prediction_row(examples[i], output)) for i, output in zip(indices, step["outputs83"], strict=True)]
            expected = {"loss": math.fsum(value["combined"]*weights[i] for i,value in zip(indices,losses))/32.,
                        "policy_mean_unweighted": math.fsum(v["policy"] for v in losses)/len(losses),
                        "value_mean_unweighted": math.fsum(v["value"] for v in losses)/len(losses)}
            for key,value in expected.items():
                require(math.isfinite(step[key]) and math.isclose(step[key], value, rel_tol=1e-6, abs_tol=1e-5),
                        "independent saved-output reconstruction of actual float32 loss: " + key)
            require(all(type(step[k]) in (int, float) and math.isfinite(step[k]) and step[k] >= 0 for k in
                    ("loss", "policy_mean_unweighted", "value_mean_unweighted", "gradient_norm_before_clip", "seconds")), "finite actual optimizer measurements")
            steps.append(path.name)
    require([p.name for p in sorted((folder / "steps").glob("*.json"))] == steps and
            [p.name for p in sorted((folder / "orders").glob("*.json"))] == [f"{i:03d}.json" for i in range(epochs)],
            "no omitted or added epochs/updates")
    require(result["epoch_order_sha256"] == orders and result["optimizer_steps"] == len(steps), "complete fixed update accounting")
    return len(steps)


def verify_fit(folder, corpus_manifest, planwrapper, parity_fixtures, *, expected_specs, boundary=lambda: None):
    """Independent saved-only numerical, corpus, lifecycle and update accounting."""
    folder = Path(folder)
    require(set(planwrapper) == {"training_plan", "source_specs", "learned_entrypoint", "input_qualification"} and
            planwrapper["source_specs"] == expected_specs, "frozen fit wrapper")
    plan = validate_plan(planwrapper["training_plan"])
    result = _read(folder / "fit_result.json")
    require(result["status"] == ("qualification_update_passed" if plan["qualification_only"] else "fixed_two_epoch_fit_passed") and
            result["qualification_only"] is plan["qualification_only"] and
            result["training_plan_sha256"] == data.digest(plan) and result["corpus_sha256"] == data.digest(corpus_manifest), "qualified fit identities")
    require(_read(folder / "runtime_sources.json") == {"runtime_sources": guard.SOURCE_HASHES, **data.IDENTITY}, "actual fitter startup guard")
    require(_read(folder / "input_tensor_qualification.json") == qualify_saved_tensor_inputs(folder, planwrapper["input_qualification"], materialize=False, boundary=boundary),
            "actual fitter corrected tensor qualification")
    require(_read(folder / "corpus_rejections.json") == data.qualify_manifest_rejections(corpus_manifest, expected_specs, boundary=boundary),
            "actual training entrypoint rejects incompatible corpus metadata and hashes")
    corpus = data.load_corpus(corpus_manifest, expected_specs=expected_specs, include_splits=[plan["training_split"]], boundary=boundary)
    examples, weights, counts = select_training_examples(corpus, plan)
    expected_loaded = {k: v for k, v in corpus.items() if k not in ("examples", "manifest")}
    require(_read(folder / "loaded_corpus.json") == json.loads(data.canonical(expected_loaded)), "only training split decoded before fit")
    require(result["training_games"] == plan["training_game_count"] and result["training_roots"] == len(examples) and
            result["roots_per_game"] == {str(k): v for k, v in counts.items()}, "exact actual minibatch corpus counts")
    updates = verify_update_records(folder, examples, weights, plan, result, boundary=boundary)
    manifest = data.read_artifact(result["checkpoint_manifest"], boundary=boundary)
    require(manifest == _read(folder / "checkpoint_manifest.json"), "exact checkpoint manifest")
    metadata = manifest["metadata"]; data.verify_identity(metadata)
    require(metadata["qualification_only"] is plan["qualification_only"] and metadata["training_corpus_sha256"] == data.digest(corpus_manifest) and
            metadata["training_plan_sha256"] == data.digest(plan) and metadata["historical_warm_start_sha256"] == PARENT_SHA256 and
            metadata["model_source_sha256"] == MODEL_SOURCE_SHA256 and metadata["parameter_count"] == PARAMETERS and
            metadata["historical_data_relabelled"] is False and metadata["candidate_selected_by_assessment"] is False, "candidate corrected lineage")
    tensor_doc = data.read_artifact(manifest["tensor_manifest"], boundary=boundary)
    require(tensor_doc["checkpoint"] == manifest["checkpoint"] == artifact(folder / "candidate.pt") and
            {x["name"]: x["shape"] for x in tensor_doc["tensors"]} == expected_state_shapes() and
            sum(x["numel"] for x in tensor_doc["tensors"]) == PARAMETERS, "same fixed architecture serialized tensors")
    hashes = _verify_tensor_archive(folder / "candidate.pt", tensor_doc["tensors"], boundary=boundary)
    parent_hashes = verify_parent_tensor_hashes(boundary=boundary)
    require(hashes["all"] == metadata["weights_sha256"] == tensor_doc["weights_sha256"] == result["candidate_weights_sha256"] and
            parent_hashes["all"] == result["parent_weights_sha256"], "independent original and learned complete tensor hashes")
    require(result["before_group_sha256"] == {k: parent_hashes[k] for k in ("shared", "policy", "value")} and
            result["after_group_sha256"] == {k: hashes[k] for k in ("shared", "policy", "value")} and
            all(parent_hashes[k] != hashes[k] for k in ("shared", "policy", "value")), "actual shared trunk and both heads changed")
    for mode, expected_checkpoint, weights in (("parent", artifact(PARENT_CHECKPOINT), parent_hashes["all"]),
                                                ("candidate", manifest["checkpoint"], hashes["all"])):
        identity = _read(folder / f"{mode}_graph/graph_identity.json")
        require(identity["checkpoint"] == expected_checkpoint and identity["weights_sha256"] == weights and
                identity["graph_ir"] == artifact(folder / f"{mode}_graph/graph_ir.txt") and identity["graph_forward_calls"] == 4,
                "saved actual graph/checkpoint identity")
        require(identity["runtime"]["threads"] == identity["runtime"]["interop_threads"] == 1 and
                identity["runtime"]["mha_fastpath_enabled"] is True and identity["runtime"]["mkldnn_enabled"] is True,
                "unchanged recorded inference runtime")
        require(data.digest(identity) == result[f"{mode}_graph_identity_sha256"], "fit graph receipt digest")
        _verify_parity_document(_read(folder / f"{mode}_parity.json"), parity_fixtures)
    opening = _read(folder / "heldout_opening.json")
    require(opening["checkpoint_manifest"] == result["checkpoint_manifest"] and
            opening["checkpoint_fixed_before_heldout_opening"] is True and opening["training_splits_opened"] == [plan["training_split"]],
            "heldout data opening bound to fixed candidate")
    heldout = data.load_corpus(corpus_manifest, expected_specs=expected_specs,
                              include_splits=heldout_splits(plan), boundary=boundary)
    require(_read(folder / "heldout_loaded_corpus.json") == json.loads(data.canonical({k: v for k, v in heldout.items() if k not in ("examples", "manifest")})),
            "full heldout loader verification after fixed checkpoint")
    learned = verify_learned_entrypoint(folder / "learned_entrypoint", planwrapper["learned_entrypoint"], result["checkpoint_manifest"], boundary=boundary)
    diagnostics = verify_heldout_diagnostics(folder / "heldout_diagnostics", heldout, plan, result["checkpoint_manifest"],
                                           {"parent": parent_hashes["all"], "candidate": hashes["all"]}, boundary=boundary)
    require(result["heldout_diagnostics"] == artifact(folder / "heldout_diagnostics/result.json"), "complete heldout diagnostic receipt")
    budget = direct_forward_budget(plan, {**corpus["example_counts"], **heldout["example_counts"]})
    for key, expected in {"optimizer_steps": updates, "parent_parity_cases": 11, "candidate_parity_cases": 11, "graph_builds": 2,
            "graph_trace_warm_forwards": 8, "parity_forward_calls": 44, "training_forward_backward_calls": updates,
            "heldout_forward_calls": diagnostics["forward_calls"],
            "direct_forward_invocations": budget["direct_forward_invocations"], "direct_count_excludes_search_callbacks": True, "learned_entrypoint_decisions": 6,
            "learned_entrypoint_callback_count": learned["callbacks"], "parent_checkpoint_unchanged": True, "exact_state_reload": True,
            "offline_losses_opened": True, "heldout_data_opened_after_checkpoint": True,
            "final_candidate_selected_without_validation_or_assessment": True}.items():
        require(result.get(key) == expected, "complete fixed fit accounting " + key)
    return {"status": "saved_qualification_fit_verified" if plan["qualification_only"] else "saved_pilot_fit_verified",
            "actual_training_games": plan["training_game_count"], "heldout_games": 2 if plan["qualification_only"] else 64,
            "optimizer_steps": updates, "direct_forward_invocations": budget["direct_forward_invocations"], "learned_entrypoint": learned,
            "heldout_diagnostics": result["heldout_diagnostics"],
            "candidate_manifest": result["checkpoint_manifest"], "actual_model_calls": 0,
            "all_tensor_bytes_and_two_head_targets_verified": True, "pilot_training_started": not plan["qualification_only"]}


def verify_qualification(folder, corpus_manifest, planwrapper, parity_fixtures, *, expected_specs, boundary=lambda: None):
    require(planwrapper["training_plan"] == qualification_plan(), "engineering-only saved fit")
    return verify_fit(folder, corpus_manifest, planwrapper, parity_fixtures, expected_specs=expected_specs, boundary=boundary)


def verify_pilot_fit(folder, corpus_manifest, planwrapper, parity_fixtures, *, expected_specs, boundary=lambda: None):
    require(planwrapper["training_plan"] == pilot_plan(), "fixed two-epoch pilot saved fit")
    return verify_fit(folder, corpus_manifest, planwrapper, parity_fixtures, expected_specs=expected_specs, boundary=boundary)


def admit_fit_cli(args, *, common, environment):
    """Model-free actual entrypoint binding, independently unit-testable."""
    c = common
    folder = Path(args.out).resolve()
    require(folder == (c.OUT / "fit").resolve(), "exclusive qualified fitter output")
    freeze_path = Path(environment["STUDY_RESEARCH_FREEZE"]).resolve()
    require(freeze_path == (c.OUT / "freeze.json").resolve() and
            guard.sha256(freeze_path) == environment["STUDY_RESEARCH_FREEZE_SHA256"], "owning fitter freeze")
    freeze = _read(freeze_path)
    c.verify([freeze["fit_plan"], freeze["fit_parity_fixtures"]])
    require(Path(args.plan).resolve() == (ROOT / freeze["fit_plan"]["path"]).resolve() and
            Path(args.fixtures).resolve() == (ROOT / freeze["fit_parity_fixtures"]["path"]).resolve(), "frozen fit plan and fixtures")
    require(Path(args.manifest).resolve() == (c.OUT / "corpus_manifest.json").resolve(), "exact new qualified corpus")
    wrapper = _read(args.plan); manifest = _read(args.manifest); fixtures = _read(args.fixtures)["fixtures"]
    expected = qualification_plan() if args.mode == "qualification" else pilot_plan()
    require(set(wrapper) == {"training_plan", "source_specs", "learned_entrypoint", "input_qualification"} and
            wrapper["training_plan"] == expected, "fixed fitter wrapper")
    require(args.mode == c.SCOPE == freeze["phase"] and args.mode in ("qualification", "pilot"), "exact frozen job/CLI scope")
    if args.mode == "pilot":
        data.validate_source_continuation(manifest, wrapper["source_specs"],
            freeze["source_continuation"], current_output=c.OUT)
        data.validate_preserved_input_qualification(wrapper["input_qualification"], freeze["source_continuation"])
    return folder, wrapper, manifest, fixtures


def place_fitter(folder):
    import psutil
    process = psutil.Process()
    process.cpu_affinity([0]); process.nice(psutil.NORMAL_PRIORITY_CLASS)
    require(process.cpu_affinity() == [0] and process.nice() == psutil.NORMAL_PRIORITY_CLASS, "actual CPU0/NORMAL fitter placement")
    reading = clocks.sample()
    value = {"identity": {"pid": process.pid, "created": process.create_time()}, "cpu_affinity": [0],
             "priority": "NORMAL_PRIORITY_CLASS", "before_torch_import": "torch" not in __import__("sys").modules,
             "owner_monotonic_time": reading["owner_monotonic"], "request_qpc_time": reading["request_qpc"],
             "unix_time": reading["unix"], "clock_contract": clocks.runtime_contract()}
    require(value["before_torch_import"], "placement precedes Torch/model work")
    save_json(Path(folder) / "placement.json", value)
    return value


def main():
    import argparse
    import traceback
    from . import common as c
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--mode", choices=["qualification", "pilot"], required=True)
    parser.add_argument("--manifest", required=True); parser.add_argument("--plan", required=True)
    parser.add_argument("--fixtures", required=True); parser.add_argument("--out", required=True)
    args = parser.parse_args()
    monotonic_deadline = float(os.environ["SELFPLAY_MONOTONIC_DEADLINE"])
    unix_deadline = float(os.environ["SELFPLAY_UNIX_DEADLINE"])
    def boundary():
        clocks.require_inherited_deadlines(monotonic_deadline, unix_deadline)
    boundary(); startup()
    folder, wrapper, manifest, fixtures = admit_fit_cli(args, common=c, environment=os.environ)
    place_fitter(folder)
    import torch
    torch.set_num_threads(1); torch.set_num_interop_threads(1)
    try:
        entry = qualify_update if args.mode == "qualification" else fit_manifest
        entry(manifest, wrapper["training_plan"], folder, expected_specs=wrapper["source_specs"],
              parity_fixtures=fixtures, learned_entrypoint=wrapper["learned_entrypoint"],
              input_qualification=wrapper["input_qualification"], boundary=boundary)
    except BaseException as error:
        if not (folder / "fit_failure.json").exists():
            reading = clocks.sample()
            save_json(folder / "fit_failure.json", {"status": "failed", "error": repr(error), "traceback": traceback.format_exc(),
                      "owner_monotonic_time": reading["owner_monotonic"], "request_qpc_time": reading["request_qpc"],
                      "unix_time": reading["unix"], "clock_contract": clocks.runtime_contract(), "no_retry": True})
        raise


if __name__ == "__main__":
    main()
