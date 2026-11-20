"""Bounded, inference-only DirectML probe. Importing this module loads no model.

The GPU candidate is one prospectively specified full-transformer primitive
implementation. CPU timing retains the accepted original frozen TorchScript
forward. Every GPU measurement includes blocking input/output transfers and a
host read of the complete output; it is not a device-only or gameplay timing.
"""
from __future__ import annotations

import argparse
import copy
import hashlib
import importlib.metadata
import json
import math
import os
from pathlib import Path
import statistics
import struct
import time
import traceback
import warnings


ROOT = Path(__file__).resolve().parents[2]
PARENT_SHA256 = "ef8a4521cd6c03c7075f15efee23f4cde6ec94d1b5398765cf0c24a09ad745cb"
MODEL_SHA256 = "0a554cd2a7b21b76a00add1157ab10b9cbdadd94b2b0e82097da2d9c06054417"
WEIGHTS_SHA256 = "32cdd9466befc527d265a356be65200c3666a4fa8a82cfefdc40b327b2f529db"
GPU_IMPLEMENTATION = "full_transformer_primitives_v1"
BATCHES = [1, 2, 8, 32, 64]
NUMERICAL = {"atol": 1e-4, "rtol": 1e-4, "value_abs_atol": 1e-4,
             "policy_probability_atol": 1e-5, "robust_argmax_gap": 1e-3,
             "require_legal_argmax_same": False, "require_robust_argmax_same": True}
TIMING = {"warmups": 5, "repetitions": 30, "order": "paired_alternating_cpu_gpu",
          "gpu_scope": "CPU input tensor to GPU, full forward, output to CPU and readback",
          "cpu_scope": "CPU input tensor, accepted compiled full forward, CPU output readback",
          "input_list_conversion_excluded_equally": True}


class QualificationFailure(ValueError):
    """A retained negative feasibility result, not permission to try another route."""


def qualify(condition, message):
    if not condition:
        raise QualificationFailure(message)


def require(condition, message):
    if not condition:
        raise ValueError("GPU probe: " + message)


def digest(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(",", ":"),
                                     allow_nan=False).encode()).hexdigest()


def file_hash(path):
    with Path(path).open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def save_json(path, value):
    with Path(path).open("x", encoding="utf-8", newline="\n") as stream:
        json.dump(value, stream, sort_keys=True, separators=(",", ":"), allow_nan=False)
        stream.write("\n"); stream.flush(); os.fsync(stream.fileno())


def verify_binding(binding, *, expected=None):
    require(isinstance(binding, dict), "missing artifact binding")
    path = (ROOT / binding["path"]).resolve()
    require(path.is_file() and path.stat().st_size == binding["bytes"] and
            file_hash(path) == binding["sha256"], "artifact identity: " + str(path))
    require(expected is None or binding["sha256"] == expected, "original artifact required")
    return path


def validate_plan(plan):
    require(plan["batches"] == BATCHES, "fixed batch matrix")
    require(plan["numerical"] == NUMERICAL, "prospectively fixed numerical tolerances")
    require(plan["timing"] == TIMING, "fixed timing count, alternating order and transfer scopes")
    require(plan["gpu_implementation"] == GPU_IMPLEMENTATION and
            plan["cpu_primitive_bridge_check"] is True, "single full-network GPU implementation")
    require(plan["device_name_contains"] == "AMD Radeon RX 7800 XT", "explicit selected GPU family")
    require(plan["cpu_affinity"] == [0], "one fixed CPU logical processor")


def float32_bytes(rows):
    require(type(rows) is list and rows, "nonempty model output")
    require(all(type(row) is list and len(row) == 83 and all(
        type(x) in (int, float) and math.isfinite(x) for x in row) for row in rows),
        "finite 83-column model output")
    return b"".join(struct.pack("<f", x) for row in rows for x in row)


def compare_outputs(reference, candidate, legal_policy_indices, numerical=NUMERICAL):
    """Pure-Python diagnostics: no tensor/backend dependency in acceptance logic."""
    ref_bytes, candidate_bytes = float32_bytes(reference), float32_bytes(candidate)
    require(len(reference) == len(candidate), "output row count")
    require(legal_policy_indices is None or len(legal_policy_indices) == len(reference), "legal mask rows")
    errors, policy_errors, value_errors, scaled_errors, probability_errors = [], [], [], [], []
    disagreements, robust_disagreements, raw_disagreements, legal_rows = [], [], [], 0
    for row_index, (ref, out) in enumerate(zip(reference, candidate)):
        for column, (a, b) in enumerate(zip(ref, out)):
            error = abs(a - b)
            errors.append(error)
            scaled_errors.append(error / (numerical["atol"] + numerical["rtol"] * abs(a)))
            if column >= 81:
                value_errors.append(error)
            else:
                policy_errors.append(error)
        raw_ref = min(range(81), key=lambda i: (-ref[i], i))
        raw_candidate = min(range(81), key=lambda i: (-out[i], i))
        if raw_ref != raw_candidate:
            raw_disagreements.append({"row": row_index, "reference": raw_ref, "candidate": raw_candidate})
        if legal_policy_indices is None:
            continue
        legal = legal_policy_indices[row_index]
        require(type(legal) is list and legal and len(set(legal)) == len(legal) and
                all(type(i) is int and 0 <= i < 81 for i in legal), "lawful policy columns")
        legal_rows += 1
        ranked = sorted(legal, key=lambda i: (-ref[i], i))
        candidate_best = min(legal, key=lambda i: (-out[i], i))
        gap = ref[ranked[0]] - ref[ranked[1]] if len(ranked) > 1 else None
        if ranked[0] != candidate_best:
            entry = {"row": row_index, "reference": ranked[0], "candidate": candidate_best, "reference_gap": gap}
            disagreements.append(entry)
            if gap is not None and gap > numerical["robust_argmax_gap"]:
                robust_disagreements.append(entry)
        ref_max, out_max = max(ref[i] for i in legal), max(out[i] for i in legal)
        p, q = [math.exp(ref[i] - ref_max) for i in legal], [math.exp(out[i] - out_max) for i in legal]
        p_sum, q_sum = sum(p), sum(q)
        probability_errors.extend(abs(a / p_sum - b / q_sum) for a, b in zip(p, q))
    passed = (max(scaled_errors) <= 1 and max(value_errors) <= numerical["value_abs_atol"] and
              max(probability_errors, default=0.) <= numerical["policy_probability_atol"] and
              (not numerical["require_legal_argmax_same"] or not disagreements) and
              (not numerical["require_robust_argmax_same"] or not robust_disagreements))
    return {"passed": passed, "rows": len(reference), "float32_bits_equal": ref_bytes == candidate_bytes,
            "max_abs_error": max(errors), "mean_abs_error": statistics.fmean(errors),
            "max_policy_abs_error": max(policy_errors),
            "max_tolerance_fraction": max(scaled_errors), "max_value_abs_error": max(value_errors),
            "max_legal_probability_abs_error": max(probability_errors, default=None),
            "raw_policy_argmax_agreements": len(reference) - len(raw_disagreements),
            "raw_policy_argmax_disagreements": raw_disagreements,
            "legal_rows": legal_rows, "legal_argmax_agreements": legal_rows - len(disagreements),
            "legal_argmax_disagreements": disagreements,
            "robust_legal_argmax_disagreements": robust_disagreements,
            "reference_float32_sha256": hashlib.sha256(ref_bytes).hexdigest(),
            "candidate_float32_sha256": hashlib.sha256(candidate_bytes).hexdigest()}


def is_backend_fallback(message):
    text = str(message).casefold()
    return (any(word in text for word in ("fall back", "fallback", "not supported", "not currently supported", "unsupported", "not implemented")) or
            "could not run" in text and any(word in text for word in ("privateuse1", "directml", "dml")))


class WarningJournal:
    def __init__(self, folder):
        self.folder, self.records, self.phase = Path(folder), [], "startup"
        self.progress = self.folder / "progress.jsonl"
        with self.progress.open("x", encoding="utf-8"):
            pass

    def event(self, state, phase):
        value = {"state": state, "phase": phase, "monotonic": time.monotonic(), "unix": time.time()}
        with self.progress.open("a", encoding="utf-8", newline="\n") as stream:
            stream.write(json.dumps(value, sort_keys=True, separators=(",", ":")) + "\n")
            stream.flush(); os.fsync(stream.fileno())

    def call(self, phase, operation):
        self.phase = phase
        self.event("started", phase)
        with warnings.catch_warnings(record=True) as captured:
            warnings.simplefilter("always")
            try:
                result = operation()
            except (NotImplementedError, RuntimeError) as error:
                self.event("exception", phase)
                if (phase.startswith("directml_") or "directml_roundtrip" in phase) and (
                        isinstance(error, NotImplementedError) or is_backend_fallback(error)):
                    raise QualificationFailure("DirectML unsupported operation: " + str(error)) from error
                raise
            finally:
                items = [{"phase": phase, "category": item.category.__name__,
                          "message": str(item.message)[:4096]} for item in captured]
                self.records.extend(items)
                require(len(self.records) <= 256, "warning evidence bound exceeded")
                if items:
                    save_json(self.folder / ("warnings_%04d.json" % len(self.records)), {"warnings": items})
        qualify(not any(is_backend_fallback(item["message"]) for item in items),
                "unsupported operation or backend fallback warning at " + phase)
        self.event("completed", phase)
        return result


def select_device(provider, expected_name):
    qualify(provider.is_available(), "DirectML provider unavailable")
    devices = [{"index": i, "name": str(provider.device_name(i))} for i in range(provider.device_count())]
    matches = [item for item in devices if expected_name.casefold() in item["name"].casefold()]
    qualify(len(matches) == 1, "exactly one explicit RX 7800 XT adapter match required")
    selected = matches[0]
    device = provider.device(selected["index"])
    return device, {"devices": devices, "selected": selected, "torch_device": str(device),
                    "selection_rule": "unique case-insensitive device-name substring; never default index"}


def validate_architecture(model, torch):
    require(not model.has_history and len(model.blocks.layers) == 6 and
            sum(p.numel() for p in model.parameters()) == 4_780_883, "original full architecture")
    require(model.project.in_features == 48 and model.project.out_features == 256 and
            tuple(model.identity.shape) == (31, 256) and model.policy.out_features == 81 and
            model.value.out_features == 2, "original entity dimensions")
    require(all(not module.training for module in model.modules()), "evaluation-only model")
    for block in model.blocks.layers:
        attn = block.self_attn
        require(block.norm_first and attn.batch_first and attn.embed_dim == 256 and
                attn.num_heads == 8 and attn._qkv_same_embed_dim and
                attn.bias_k is None and attn.bias_v is None and not attn.add_zero_attn and
                attn.dropout == 0 and all(getattr(block, name).p == 0 for name in
                                         ("dropout", "dropout1", "dropout2")) and
                block.linear1.in_features == 256 and block.linear1.out_features == 1024 and
                block.linear2.in_features == 1024 and block.linear2.out_features == 256 and
                block.activation is torch.nn.functional.gelu, "fixed pre-norm attention/GELU/no-dropout architecture")


def make_primitive_forward(torch, model):
    """Same complete network; no fused CPU transformer, pruning, mask or dropout."""
    class FullTransformerPrimitives(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.model = model

        def forward(self, tokens):
            x = self.model.project(tokens) + self.model.identity
            batch, length, width = x.shape
            heads, head_width = 8, 32
            for block in self.model.blocks.layers:
                normalized = block.norm1(x)
                qkv = torch.nn.functional.linear(normalized, block.self_attn.in_proj_weight,
                                                block.self_attn.in_proj_bias)
                q, k, v = qkv.split(width, dim=-1)
                q = q.reshape(batch, length, heads, head_width).transpose(1, 2).reshape(batch * heads, length, head_width)
                k = k.reshape(batch, length, heads, head_width).transpose(1, 2).reshape(batch * heads, length, head_width)
                v = v.reshape(batch, length, heads, head_width).transpose(1, 2).reshape(batch * heads, length, head_width)
                attention = torch.softmax(torch.bmm(q, k.transpose(1, 2)) / math.sqrt(head_width), dim=-1)
                context = torch.bmm(attention, v).reshape(batch, heads, length, head_width)
                context = context.transpose(1, 2).reshape(batch, length, width)
                x = x + block.self_attn.out_proj(context)
                x = x + block.linear2(block.activation(block.linear1(block.norm2(x))))
            h = self.model.norm(x[:, 0])
            return torch.cat((self.model.policy(h), self.model.value(h).tanh()), dim=1)

    return FullTransformerPrimitives().eval()


def state_digest(model, torch):
    hasher, count = hashlib.sha256(), 0
    for name, tensor in sorted(model.state_dict().items()):
        require(tensor.dtype == torch.float32, "unchanged float32 weights")
        descriptor = json.dumps([name, list(tensor.shape), str(tensor.dtype)], separators=(",", ":"))
        hasher.update(descriptor.encode()); hasher.update(b"\0")
        hasher.update(tensor.detach().to("cpu", non_blocking=False).contiguous().numpy().tobytes(order="C"))
        count += tensor.numel()
    require(count == 4_780_883, "complete original weight inventory")
    return hasher.hexdigest()


def build_cpu_model(checkpoint_binding, model_source_binding):
    """Load authenticated original weights without spawning a native game/search."""
    checkpoint = verify_binding(checkpoint_binding, expected=PARENT_SHA256)
    source = verify_binding(model_source_binding, expected=MODEL_SHA256)
    import torch
    from tools.external_strength_screen_20261018_02.splendorust_adapter import load_model
    from tools.splendorust_inference_opt_20261025.graph import _make_forward
    model, _ = load_model(source, checkpoint)
    model.requires_grad_(False)
    validate_architecture(model, torch)
    require(state_digest(model, torch) == WEIGHTS_SHA256, "original full model state fingerprint")
    return {"model": model, "eager": _make_forward(torch, model),
            "primitive": make_primitive_forward(torch, model), "torch": torch}


def output_to_host(output, batch, torch):
    require(output.dtype == torch.float32 and tuple(output.shape) == (batch, 83), "output shape/dtype")
    # Blocking D2H plus a full host read is the synchronization boundary.
    return output.detach().to("cpu", non_blocking=False).contiguous().numpy().copy()


def gpu_roundtrip(gpu, cpu_input, batch, device, torch):
    require(cpu_input.device.type == "cpu", "host input starts on CPU")
    device_input = cpu_input.to(device, non_blocking=False)
    qualify(device_input.device == device and device_input.dtype == torch.float32,
            "actual GPU float32 input device")
    output = gpu(device_input)
    qualify(output.device == device, "actual GPU output device before readback")
    return output_to_host(output, batch, torch)


def percentile(values, fraction):
    ranked = sorted(values); position = (len(ranked) - 1) * fraction
    left, right = math.floor(position), math.ceil(position)
    return ranked[left] + (ranked[right] - ranked[left]) * (position - left)


def summarize_timings(values, batch):
    require(len(values) == 30 and all(type(x) is float and math.isfinite(x) and x > 0 for x in values),
            "thirty positive complete timed samples")
    mean = statistics.fmean(values)
    return {"samples_seconds": values, "mean_seconds": mean, "median_seconds": statistics.median(values),
            "p95_seconds": percentile(values, .95), "min_seconds": min(values), "max_seconds": max(values),
            "positions_per_second": batch / mean}


def evaluate_matrix(bundle, fixtures, plan, folder, *, journal=None, boundary=lambda: None):
    """CPU historical/exact gates, CPU primitive bridge, then one DirectML route."""
    torch = bundle["torch"]; folder = Path(folder)
    journal = journal or WarningJournal(folder)
    require(type(fixtures) is list and fixtures, "nonempty fixed input panel")
    numerical_rows, inputs, eager_outputs = [], {}, {}
    with torch.inference_mode():
        gpu_comparisons = []
        for index, fixture in enumerate(fixtures):
            boundary(); batch, key = fixture["batch"], fixture["fixture_id"]
            require(batch in BATCHES and key not in inputs and len(fixture["tokens"]) == batch * 31 * 48,
                    "unique fixed fixture shape")
            x = torch.tensor(fixture["tokens"], dtype=torch.float32).reshape(batch, 31, 48)
            require(bool(torch.isfinite(x).all()), "finite fixed input")
            inputs[key] = x
            eager = journal.call("cpu_eager_" + key, lambda: output_to_host(bundle["eager"](x), batch, torch)).tolist()
            compiled = journal.call("cpu_compiled_" + key, lambda: output_to_host(bundle["compiled"](x), batch, torch)).tolist()
            primitive = journal.call("cpu_primitive_" + key, lambda: output_to_host(bundle["primitive"](x), batch, torch)).tolist()
            masks = fixture["legal_policy_indices"]
            exact = compare_outputs(eager, compiled, masks, plan["numerical"])
            bridge = compare_outputs(eager, primitive, masks, plan["numerical"])
            historical = fixture.get("historical_expected_cpu_outputs")
            historical_exact = None if historical is None else (
                eager == historical and float32_bytes(eager) == float32_bytes(historical))
            record = {"fixture_id": key, "batch": batch, "input_sha256": digest(fixture["tokens"]),
                      "cpu_eager_outputs": eager, "cpu_compiled_outputs": compiled,
                      "cpu_primitive_outputs": primitive, "cpu_compiled_comparison": exact,
                      "cpu_primitive_comparison": bridge, "historical_cpu_exact": historical_exact}
            save_json(folder / ("cpu_numeric_%03d.json" % index), record)
            qualify(historical_exact is not False and exact["float32_bits_equal"] and eager == compiled,
                    "original CPU historical/compiled exact gate: " + key)
            qualify(bridge["passed"], "full-transformer CPU primitive numerical bridge: " + key)
            numerical_rows.append(record); eager_outputs[key] = eager
        boundary()
        import torch_directml
        device, device_identity = journal.call("directml_adapter_selection", lambda: select_device(torch_directml, plan["device_name_contains"]))
        qualify(device.type == "privateuseone" and device.index == device_identity["selected"]["index"],
                "explicit DirectML GPU device type and index")
        gpu_model = journal.call("directml_weight_transfer", lambda: copy.deepcopy(bundle["model"]).to(device))
        validate_architecture(gpu_model, torch)
        qualify(all(tensor.device == device for tensor in gpu_model.state_dict().values()),
                "every actual model tensor resides on selected GPU")
        device_identity.update(actual_model_tensor_devices=sorted({str(tensor.device) for tensor in gpu_model.state_dict().values()}),
                               gpu_input_output_device_checked_per_call=True)
        save_json(folder / "device_identity.json", device_identity)
        gpu = make_primitive_forward(torch, gpu_model)
        require(journal.call("directml_initial_weight_readback", lambda: state_digest(gpu_model, torch)) == WEIGHTS_SHA256,
                "identical GPU float32 model weights")
        for index, fixture in enumerate(fixtures):
            boundary(); batch, key = fixture["batch"], fixture["fixture_id"]
            x = inputs[key]
            outputs = journal.call("directml_numeric_" + key, lambda: gpu_roundtrip(gpu, x, batch, device, torch)).tolist()
            comparison = compare_outputs(eager_outputs[key], outputs, fixture["legal_policy_indices"], plan["numerical"])
            record = {"fixture_id": key, "batch": batch, "gpu_outputs": outputs, "comparison": comparison}
            save_json(folder / ("gpu_numeric_%03d.json" % index), record)
            gpu_comparisons.append(comparison)
            qualify(comparison["passed"], "DirectML numerical gate: " + key)
        timing_rows = []
        for batch in BATCHES:
            chosen = [fixture for fixture in fixtures if fixture["batch"] == batch and fixture.get("timing_fixture") is True]
            require(len(chosen) == 1 and chosen[0]["legal_policy_indices"] is not None, "one lawful timing fixture per batch")
            key = chosen[0]["fixture_id"]; x = inputs[key]
            operations = {"cpu_compiled": lambda: output_to_host(bundle["compiled"](x), batch, torch),
                          "directml_roundtrip": lambda: gpu_roundtrip(gpu, x, batch, device, torch)}
            samples = {route: [] for route in operations}
            for repetition in range(plan["timing"]["warmups"] + plan["timing"]["repetitions"]):
                boundary()
                order = list(operations) if repetition % 2 == 0 else list(reversed(operations))
                for route in order:
                    def measured():
                        started = time.perf_counter()
                        array = operations[route]()
                        elapsed = time.perf_counter() - started
                        return array, elapsed
                    array, elapsed = journal.call("timing_%d_%d_%s" % (batch, repetition, route), measured)
                    comparison = compare_outputs(eager_outputs[key], array.tolist(), chosen[0]["legal_policy_indices"], plan["numerical"])
                    qualify(comparison["passed"], "timed output numerical gate")
                    if route == "cpu_compiled":
                        qualify(comparison["float32_bits_equal"], "timed exact CPU output")
                    if repetition >= plan["timing"]["warmups"]:
                        samples[route].append(elapsed)
            row = {"batch": batch, "fixture_id": key, "warmups_per_route": 5,
                   "order": "CPU/GPU and GPU/CPU alternating, including warmups",
                   "routes": {route: summarize_timings(values, batch) for route, values in samples.items()}}
            row["cpu_over_gpu_mean_ratio"] = row["routes"]["cpu_compiled"]["mean_seconds"] / row["routes"]["directml_roundtrip"]["mean_seconds"]
            save_json(folder / ("timing_batch_%02d.json" % batch), row)
            timing_rows.append(row)
        require(state_digest(bundle["model"], torch) == WEIGHTS_SHA256 and
                journal.call("directml_final_weight_readback", lambda: state_digest(gpu_model, torch)) == WEIGHTS_SHA256,
                "all weights unchanged after inference-only probe")
    return {"status": "qualified_inference_only", "device": device_identity, "timing": timing_rows,
            "fixtures": len(fixtures), "historical_cpu_cases": sum(row["historical_cpu_exact"] is True for row in numerical_rows),
            "numerical_summary": {"max_policy_abs_error": max(row["max_policy_abs_error"] for row in gpu_comparisons),
                "max_value_abs_error": max(row["max_value_abs_error"] for row in gpu_comparisons),
                "max_legal_probability_abs_error": max(row["max_legal_probability_abs_error"] or 0 for row in gpu_comparisons),
                "raw_policy_argmax_agreements": sum(row["raw_policy_argmax_agreements"] for row in gpu_comparisons),
                "raw_policy_argmax_disagreements": sum(len(row["raw_policy_argmax_disagreements"]) for row in gpu_comparisons),
                "legal_rows": sum(row["legal_rows"] for row in gpu_comparisons),
                "legal_argmax_disagreements": sum(len(row["legal_argmax_disagreements"]) for row in gpu_comparisons),
                "robust_legal_argmax_disagreements": sum(len(row["robust_legal_argmax_disagreements"]) for row in gpu_comparisons),
                "fixture_rows_are_not_independent_samples": True},
            "gpu_implementation": GPU_IMPLEMENTATION, "training_updates": 0, "new_games": 0,
            "gpu_only_timing_measured": False, "end_to_end_player_speed_measured": False,
            "limitations": ["Numerical feasibility only, not lossless search/player qualification.",
                            "Near-tie legal argmax disagreements are recorded; no live move equivalence is established.",
                            "No training, new games, automatic deployment or checkpoint promotion."],
            "fallback_observed": False, "fallback_detection_scope": "Python-visible backend warnings and exceptions; native stderr retained by controller",
            "weights_before_after_sha256": WEIGHTS_SHA256, "warnings": journal.records}


def run_probe(plan, panel, folder, *, boundary=lambda: None):
    folder = Path(folder); folder.mkdir(parents=True, exist_ok=False)
    journal = WarningJournal(folder)
    try:
        from tools.future_training_rules_20261021 import verify_runtime_sources, RULESET_ID, CATALOGUE_SHA256, TURN_CONTRACT
        source_receipt = verify_runtime_sources()
        save_json(folder / "runtime_sources.json", {"sources": source_receipt, "ruleset_id": RULESET_ID,
                  "catalogue_sha256": CATALOGUE_SHA256, "turn_contract": TURN_CONTRACT})
        validate_plan(plan)
        from tools.splendorust_gpu_probe_20261105.panel import validate_panel
        validation = validate_panel(panel)
        save_json(folder / "panel_validation.json", validation)
        for name in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS", "NUMEXPR_NUM_THREADS"):
            os.environ[name] = "1"
        import psutil
        process = psutil.Process(); process.cpu_affinity(plan["cpu_affinity"])
        require(process.cpu_affinity() == plan["cpu_affinity"], "worker CPU affinity applied")
        import torch
        torch.set_num_threads(1); torch.set_num_interop_threads(1)
        require(str(torch.__version__) == plan["torch_version_expected"] == "2.4.1+cpu" and
                importlib.metadata.version("torch-directml") == plan["directml_version_expected"] == "0.2.5.dev240914",
                "existing frozen runtime package versions")
        require(torch.get_num_threads() == torch.get_num_interop_threads() == 1 and
                torch.backends.mha.get_fastpath_enabled() and torch.backends.mkldnn.enabled,
                "accepted one-thread CPU backend flags")
        save_json(folder / "runtime_identity.json", {"torch_version": str(torch.__version__),
                  "torch_directml_version": importlib.metadata.version("torch-directml"),
                  "torch_module": str(Path(torch.__file__).resolve()), "threads": 1, "interop_threads": 1,
                  "cpu_affinity": process.cpu_affinity(), "mha_fastpath_enabled": True, "mkldnn_enabled": True,
                  "gpu_implementation": GPU_IMPLEMENTATION, "plan_sha256": digest(plan), "panel_sha256": digest(panel)})
        boundary()
        bundle = journal.call("load_original_cpu_model", lambda: build_cpu_model(plan["checkpoint"], plan["model_source"]))
        with torch.inference_mode():
            compiled = journal.call("trace_original_cpu_graph", lambda: torch.jit.trace(bundle["eager"], torch.zeros((1, 31, 48), dtype=torch.float32), check_trace=False, strict=True))
            bundle["compiled"] = journal.call("freeze_original_cpu_graph", lambda: torch.jit.freeze(compiled, optimize_numerics=False))
        ir = str(bundle["compiled"].inlined_graph) + "\n"
        require(len(ir.encode()) <= 256 * 1024, "bounded original graph evidence")
        with (folder / "cpu_graph_ir.txt").open("x", encoding="utf-8", newline="\n") as stream:
            stream.write(ir); stream.flush(); os.fsync(stream.fileno())
        result = evaluate_matrix(bundle, panel["fixtures"], plan, folder, journal=journal, boundary=boundary)
        save_json(folder / "worker_result.json", result)
        return result
    except QualificationFailure as error:
        result = {"status": "qualification_failed", "phase": journal.phase, "reason": str(error),
                  "traceback": traceback.format_exc(), "warnings": journal.records,
                  "gpu_implementation": GPU_IMPLEMENTATION, "automatic_retry": False,
                  "alternative_gpu_route_attempted": False, "training_updates": 0, "new_games": 0,
                  "timing_complete": False, "numerical_feasibility_only": True}
        save_json(folder / "worker_result.json", result)
        return result
    except BaseException as error:
        save_json(folder / "worker_failure.json", {"status": "failed_closed", "phase": journal.phase,
                  "error": repr(error), "traceback": traceback.format_exc(), "warnings": journal.records,
                  "automatic_retry": False, "alternative_gpu_route_attempted": False})
        raise


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ("plan", "panel", "output", "plan-sha256", "panel-sha256"):
        parser.add_argument("--" + name, required=True)
    args = parser.parse_args()
    require(file_hash(args.plan) == args.plan_sha256 and file_hash(args.panel) == args.panel_sha256,
            "frozen plan/panel bytes before runtime import")
    plan = json.loads(Path(args.plan).read_text(encoding="utf-8"))
    panel = json.loads(Path(args.panel).read_text(encoding="utf-8"))
    run_probe(plan, panel, args.output)


if __name__ == "__main__":
    main()
