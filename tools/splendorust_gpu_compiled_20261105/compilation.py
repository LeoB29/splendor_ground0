"""One fixed GPU trace/freeze option, with preserved support evidence.

This module imports no tensor framework and performs no model work on import.
An unsupported candidate is distinct from an unrelated resource or I/O failure.
"""
from collections import Counter
import hashlib
import json
import os
from pathlib import Path
import re
import struct
import traceback


OPTIONS = dict(trace_check_trace=False, trace_strict=True, freeze_optimize_numerics=False,
               input_shape=[8, 31, 48], input_dtype="torch.float32", input_device="selected GPU")
MATH_OPS = frozenset(("aten::linear", "aten::add", "aten::layer_norm", "aten::bmm",
                      "aten::div", "aten::softmax", "aten::gelu", "aten::tanh", "aten::cat"))
ALLOWED_OPS = MATH_OPS | frozenset((
    "prim::Constant", "prim::GetAttr", "prim::ListConstruct", "prim::ListUnpack",
    "prim::TupleConstruct", "prim::TupleUnpack", "prim::NumToTensor",
    "aten::size", "aten::Int", "aten::reshape", "aten::transpose", "aten::split",
    "aten::slice", "aten::select", "aten::mul"))


class CompilationRejected(RuntimeError):
    """The single prospective candidate is unsupported or violates exactness."""
    def __init__(self, message, *, code="candidate_exactness_rejected"):
        self.code = code
        super().__init__(message)


GRAPH_REJECTIONS = {
    "unsupported_graph_attribute": "Unsupported graph attribute type: ",
    "unsupported_graph_ivalue": "Unsupported graph IValue type: ",
    "unsupported_graph_operator": "Different or unsupported compiled graph operator: ",
    "changed_math_sequence": "Trace/freeze changed mathematical operator sequence",
    "wrong_primitive_architecture": "Compiled graph does not retain complete six-block primitive mathematics",
}
PHASES = frozenset(("input", "trace", "traced_graph", "freeze", "frozen_graph", "graph_check", "weight_check"))


def rejection_kind(error):
    """Classify only explicit candidate support errors; never resource faults."""
    if isinstance(error, CompilationRejected):
        return "candidate_rejected"
    if isinstance(error, (MemoryError, OSError, TimeoutError, KeyboardInterrupt, SystemExit)):
        return None
    text = str(error).casefold()
    if any(s in text for s in ("out of memory", "not enough memory", "insufficient memory",
                               "allocation failed", "failed to allocate", "bad allocation",
                               "device removed", "device lost", "device hung", "access violation")):
        return None
    if isinstance(error, NotImplementedError):
        return "unsupported_operation"
    if isinstance(error, (RuntimeError, ValueError)) and any(s in text for s in (
            "not supported", "not currently supported", "unsupported", "not implemented",
            "could not run", "fall back", "fallback", "cannot be traced", "could not trace",
            "tracing failed", "trace failed", "tracingcheckerror")):
        return "unsupported_operation"
    return None


def classify_saved_compile_failure(payload):
    """Independently derive a refusal from its original error, not its wrapper."""
    if (payload.get("schema") != "fixed_b8_compile_failure_v1" or payload.get("options") != OPTIONS
            or payload.get("phase") not in PHASES or payload.get("retries") != 0
            or payload.get("alternative_options_attempted") is not False
            or not isinstance(payload.get("traceback"), str) or not payload["traceback"]
            or payload.get("original_weights_unchanged") is not True):
        raise ValueError("Incomplete compilation failure evidence")
    kind = None; typ = payload["exception_type"]; module = payload["exception_module"]
    message = payload["message"]; code = payload.get("exception_code")
    if typ == "CompilationRejected" and module == __name__:
        prefix = GRAPH_REJECTIONS.get(code)
        if prefix and message.startswith(prefix): kind = "candidate_rejected"
    elif module == "builtins" and typ in ("NotImplementedError", "RuntimeError", "ValueError"):
        constructor = {"NotImplementedError": NotImplementedError, "RuntimeError": RuntimeError, "ValueError": ValueError}[typ]
        kind = rejection_kind(constructor(message))
    expected = "candidate_rejected" if kind else "technical_failure"
    if payload.get("classification") != expected or payload.get("rejection_kind") != kind:
        raise ValueError("Saved compilation classification disagrees with original error")
    return expected


def _canonical(value):
    return json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False)


def _digest(value):
    return hashlib.sha256(_canonical(value).encode()).hexdigest()


def _artifact(path):
    from . import common as c
    return c.artifact(path)


def _write(path, text):
    from . import common as c
    path = Path(path).resolve()
    if not path.is_relative_to(c.OUT.resolve()):
        raise ValueError("Compilation evidence must belong to the new study")
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("x", encoding="utf-8", newline="\n") as stream:
        stream.write(text); stream.flush(); os.fsync(stream.fileno())
    return _artifact(path)


def _save(path, value):
    return _write(path, _canonical(value)+"\n")


def _normalized_type(value):
    # Generated module type numbers vary between fresh workers. No tensor dtype,
    # device, dimension or node connectivity is removed from the fingerprint.
    return re.sub(r"\.___torch_mangle_\d+", "", str(value))


def _tensor_identity(tensor):
    host = tensor.detach().to("cpu", non_blocking=False).contiguous()
    return dict(dtype=str(tensor.dtype), shape=list(tensor.shape), device=str(tensor.device),
                sha256=hashlib.sha256(host.numpy().tobytes(order="C")).hexdigest())


def _canonical_ivalue(value):
    """Typed literal constants only; no opaque string/repr fallback.

    Torch represents ordinary int[]/float[] shape constants as IValues. Float
    bits preserve signed zero and nonfinite payloads without nonstandard JSON.
    Tensor attributes already use the explicit byte-bound 't'/'ts' route.
    """
    typ = type(value)
    if value is None:
        return dict(type="none")
    if typ is bool:
        return dict(type="bool", value=value)
    if typ is int:
        return dict(type="int", value=value)
    if typ is float:
        return dict(type="float", ieee754_binary64_hex=struct.pack("<d", value).hex())
    if typ is str:
        return dict(type="str", value=value)
    if typ is list or typ is tuple:
        return dict(type="list" if typ is list else "tuple", items=[_canonical_ivalue(x) for x in value])
    raise CompilationRejected("Unsupported graph IValue type: "+typ.__module__+"."+typ.__qualname__,
                              code="unsupported_graph_ivalue")


def graph_description(graph):
    """Canonical SSA connectivity and attributes, independent of debug names."""
    values = {}; next_id = 0; operators = []

    def bind(value):
        nonlocal next_id
        if value not in values:
            values[value] = next_id; next_id += 1
        return values[value]

    def attribute(node, name):
        kind = node.kindOf(name)
        if kind in ("i", "f", "s"):
            value = getattr(node, kind)(name)
        elif kind in ("is", "fs", "ss"):
            value = list(getattr(node, kind)(name))
        elif kind == "t":
            value = _tensor_identity(node.t(name))
        elif kind == "ts":
            value = [_tensor_identity(x) for x in node.ts(name)]
        elif kind == "g":
            value = describe(node.g(name))
        elif kind == "gs":
            value = [describe(x) for x in node.gs(name)]
        elif kind == "ival":
            value = _canonical_ivalue(node.ival(name))
        else:
            raise CompilationRejected("Unsupported graph attribute type: "+kind, code="unsupported_graph_attribute")
        return dict(kind=kind, value=value)

    def describe(block):
        inputs = [dict(id=bind(v), type=_normalized_type(v.type())) for v in block.inputs()]
        nodes = []
        for node in block.nodes():
            kind = node.kind(); operators.append(kind)
            if kind not in ALLOWED_OPS:
                raise CompilationRejected("Different or unsupported compiled graph operator: "+kind, code="unsupported_graph_operator")
            nodes.append(dict(kind=kind, inputs=[bind(v) for v in node.inputs()],
                outputs=[dict(id=bind(v), type=_normalized_type(v.type())) for v in node.outputs()],
                attributes={name: attribute(node, name) for name in sorted(node.attributeNames())},
                blocks=[describe(x) for x in node.blocks()]))
        return dict(inputs=inputs, nodes=nodes, outputs=[bind(v) for v in block.outputs()])

    body = describe(graph)
    return dict(body=body, operators=operators, operator_inventory=dict(sorted(Counter(operators).items())),
                math_sequence=[x for x in operators if x in MATH_OPS], fingerprint=_digest(body))


def validate_graph_pair(traced, frozen):
    if traced["math_sequence"] != frozen["math_sequence"]:
        raise CompilationRejected("Trace/freeze changed mathematical operator sequence", code="changed_math_sequence")
    # The original complete six-block primitive architecture is mandatory.
    expected = ["aten::linear", "aten::add"]
    block = ["aten::layer_norm", "aten::linear", "aten::bmm", "aten::div", "aten::softmax",
             "aten::bmm", "aten::linear", "aten::add", "aten::layer_norm", "aten::linear",
             "aten::gelu", "aten::linear", "aten::add"]
    expected += block*6 + ["aten::layer_norm", "aten::linear", "aten::linear", "aten::tanh", "aten::cat"]
    if traced["math_sequence"] != expected:
        raise CompilationRejected("Compiled graph does not retain complete six-block primitive mathematics", code="wrong_primitive_architecture")
    return True


def compile_fixed_b8(torch, primitive, device, folder, *, checked, warning_records, weights_sha256, verify_weights):
    """No retry, alternate option, fallback, additional optimization or precision."""
    from . import common as c
    folder = Path(folder).resolve()
    if not folder.is_relative_to(c.OUT.resolve()):
        raise ValueError("Compilation evidence must belong to the new study")
    # Refuse reuse before even constructing the tracing example or running math.
    folder.mkdir(parents=True, exist_ok=False)
    phase = "input"; graphs = {}; descriptions = {}
    warning_start = len(warning_records)
    try:
        # Created outside all panel clocks, solely the one fixed trace example.
        example = checked(lambda: torch.zeros((8, 31, 48), dtype=torch.float32).to(device, non_blocking=False))
        phase = "trace"
        traced = checked(lambda: torch.jit.trace(primitive, example, check_trace=False, strict=True))
        graphs["traced"] = _write(folder/"traced_graph.txt", str(traced.inlined_graph)+"\n")
        phase = "traced_graph"
        descriptions["traced"] = checked(lambda: graph_description(traced.inlined_graph))
        phase = "freeze"
        frozen = checked(lambda: torch.jit.freeze(traced, optimize_numerics=False))
        graphs["frozen"] = _write(folder/"frozen_graph.txt", str(frozen.inlined_graph)+"\n")
        phase = "frozen_graph"
        descriptions["frozen"] = checked(lambda: graph_description(frozen.inlined_graph))
        phase = "graph_check"
        validate_graph_pair(descriptions["traced"], descriptions["frozen"])
        for name, value in descriptions.items():
            graphs[name+"_inventory"] = _save(folder/(name+"_graph_inventory.json"), value)
        binding = dict(options=OPTIONS, weights_sha256=weights_sha256, device=str(device),
                       frozen_graph_fingerprint=descriptions["frozen"]["fingerprint"],
                       mathematical_operator_sequence=descriptions["frozen"]["math_sequence"])
        phase = "weight_check"
        verify_weights()
        receipt = dict(schema="fixed_b8_trace_freeze_v1", status="compiled_supported_pending_exact_qualification",
                       implementation="compiled", binding=binding, compilation_fingerprint=_digest(binding),
                       graphs=graphs, warnings=list(warning_records[warning_start:]), retries=0,
                       exact_outputs_qualified=False, timed_panel=False, original_weights_unchanged=True)
        artifact = _save(folder/"compilation.json", receipt)
        return frozen, dict(compilation_fingerprint=receipt["compilation_fingerprint"], receipt=artifact)
    except BaseException as error:
        error_traceback = traceback.format_exc()
        kind = rejection_kind(error); closure_error = None
        try:
            verify_weights()
        except BaseException as caught:
            closure_error = caught; kind = None
        # Never suppress I/O failure while preserving evidence: it remains a
        # technical failure, rather than turning an unrecorded error into a result.
        _save(folder/"compilation_failure.json", dict(schema="fixed_b8_compile_failure_v1", phase=phase,
            classification="candidate_rejected" if kind else "technical_failure", rejection_kind=kind,
            exception_type=type(error).__name__, exception_module=type(error).__module__, message=str(error),
            exception_code=getattr(error,"code",None), traceback=error_traceback,
            warnings=list(warning_records[warning_start:]), original_weights_unchanged=closure_error is None,
            weight_closure_error=None if closure_error is None else dict(type=type(closure_error).__name__,message=str(closure_error)),
            options=OPTIONS, graphs=graphs, retries=0, alternative_options_attempted=False))
        if closure_error is not None:
            raise closure_error from error
        if kind:
            if isinstance(error, CompilationRejected): raise
            raise CompilationRejected("Fixed compiled candidate refused at "+phase+": "+str(error), code="unsupported_compilation") from error
        raise
