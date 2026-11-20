"""Exact fixed-B8 support qualification, before any candidate speed selection."""
from pathlib import Path
import hashlib
import struct
import traceback

from . import common as c
from tools.splendorust_gpu_throughput_20261105 import qualification as q


class SupportMismatch(ValueError):
    def __init__(self, receipt):
        super().__init__("Compiled fixed-B8 output differs from exact accepted reference")
        self.receipt = receipt


class SupportUnsupported(RuntimeError):
    def __init__(self, receipt):
        super().__init__("Compiled fixed-B8 runtime operation is unsupported")
        self.receipt = receipt


def typed_chain(error):
    rows = []; seen = set()
    while error is not None and id(error) not in seen:
        seen.add(id(error))
        rows.append(dict(type=type(error).__name__, module=type(error).__module__, message=str(error)))
        error = error.__cause__ if error.__cause__ is not None else (
            None if error.__suppress_context__ else error.__context__)
    return rows


def runtime_unsupported_chain(chain):
    """Only an original supported failure type beneath exact service wrappers."""
    from .compilation import rejection_kind
    if not chain:
        return False
    for row in chain[:-1]:
        if row != dict(type="RuntimeError", module="builtins", message="Physical qualification inference failed") and row != dict(
                type="RuntimeError", module="builtins", message="Inference service failed; no retry"):
            return False
    leaf = chain[-1]
    constructors = {"RuntimeError": RuntimeError, "ValueError": ValueError, "NotImplementedError": NotImplementedError}
    if leaf["module"] != "builtins" or leaf["type"] not in constructors:
        return False
    return rejection_kind(constructors[leaf["type"]](leaf["message"])) == "unsupported_operation"


def specification():
    return dict(physical_batch=8, warmup_calls_per_implementation=5,
        warmup_input="all_zero_physical_batch", fixtures=36, repetitions_per_fixture=2,
        order=["eager", "compiled"], full_root_batches=4, single_occupied_batches=32,
        compare_all_float32_rows=True, require_exact_saved_occupied_rows=True)


def packed(outputs):
    c.require(type(outputs) is list and len(outputs) == 8
              and all(type(row) is list and len(row) == 83 for row in outputs), "physical B8 output shape")
    return b"".join(struct.pack("<83f", *row) for row in outputs)


def first_difference(actual, expected, rows=None):
    packed(actual); packed(expected)
    for row in range(8) if rows is None else rows:
        for column in range(83):
            a, e = actual[row][column], expected[row][column]
            ab, eb = struct.pack("<f", a), struct.pack("<f", e)
            if a != e or ab != eb:
                return dict(row=row, column=column, expected=e, actual=a,
                            expected_float32_hex=eb.hex(), actual_float32_hex=ab.hex())
    return None


def build_fixtures(plan):
    from . import cell
    c.require(plan["support"] == specification(), "fixed support inputs and counts")
    cases = cell.validate_cases(plan); fixtures = []
    for index in range(36):
        indexes = list(range(index*8, index*8+8)) if index < 4 else [index-4]
        tokens = [0.]*(8*31*48); expected = [[0.]*83 for _ in range(8)]; sources = []
        for case_index in indexes:
            case = cases[case_index]; slot = case_index % 8
            tokens[slot*31*48:(slot+1)*31*48] = case["expected_root_tokens"]
            record = cell.expected_move(case, "gpu")
            expected[slot] = record["result"]["diagnostics"]["phases"][0]["inference"][0]["output83"]
            sources.append(dict(slot=slot, case_index=case_index, source=case["expected_results"]["gpu"]))
        fixtures.append(dict(index=index, fixture_id=f"b{index:03d}", physical_batch=8,
            kind="full_saved_roots" if index < 4 else "single_occupied_native_row",
            case_indexes=indexes, occupied_slots=[i % 8 for i in indexes], input_tokens=tokens,
            input_sha256=c.digest(tokens), expected_occupied_outputs=expected, expected_sources=sources))
    return fixtures


def compare_and_save(actual_path, expected, expected_artifact, *, implementation,
                     kind, destination, occupied_slots=None, source_bindings=None):
    actual = c.read(actual_path)
    difference = first_difference(actual["outputs"], expected, occupied_slots)
    row = dict(status="passed" if difference is None else "exact_mismatch", kind=kind,
        implementation=implementation, phase=actual["phase"], fixture_id=actual["fixture_id"],
        repetition=actual["repetition"], actual=c.artifact(actual_path), expected=expected_artifact,
        expected_sources=source_bindings or [], occupied_slots=occupied_slots,
        difference=difference, all_rows_compared=occupied_slots is None, no_retry=True)
    receipt = c.save(destination, row)
    if difference is not None:
        if implementation == "compiled":
            raise SupportMismatch(receipt)
        raise ValueError("Eager prerequisite changed from accepted exact output: "+str(destination))
    return receipt


def run_implementation(plan, runtime, folder, fixtures, *, eager_folder=None, boundary=lambda: None):
    implementation = runtime.identity["implementation"]
    c.require(implementation in ("eager", "compiled") and runtime.route == "gpu"
              and runtime.identity["scheduler"] == "row_fifo32" and not runtime.adapters
              and not runtime.service.clients and not runtime.service.pending, "isolated support uses fixed original physical service")
    c.require(plan["support"] == specification() and len(fixtures) == 36,
              "predetermined support requests")
    c.require((eager_folder is None) is (implementation == "eager"), "compiled support binds completed eager records")
    folder = Path(folder); folder.mkdir(parents=True, exist_ok=False)
    eager_folder = None if eager_folder is None else Path(eager_folder)
    rows = []; comparisons = []; zero_tokens = [0.]*(8*31*48)
    requests = [dict(phase="warmup", fixture_id="zero", repetition=i,
        input_tokens=zero_tokens, input_sha256=c.digest(zero_tokens), occupied_slots=[])
        for i in range(5)]
    requests += [dict(fixture, phase="fixtures", repetition=repetition)
                 for fixture in fixtures for repetition in range(2)]
    for request in requests:
        boundary()
        relative = (Path("warmup")/f"{request['repetition']:03d}" if request["phase"] == "warmup"
                    else Path("fixtures")/request["fixture_id"]/f"{request['repetition']:03d}")
        target = folder/relative
        c.save(target/"request.json", request)
        def operation(started):
            with runtime.operation("physical_support", started=started, kind="physical_support"):
                return runtime.service.physical_forward(request["input_tokens"], deadline=started+c.INTERNAL_SECONDS)
        try:
            outputs, clock = q.timed_call(operation, abort=runtime.abort,
                on_receipt=lambda value, path=target: c.save(path/"clock.json", value))
        except BaseException as caught:
            clock_path = target/"clock.json"
            clock = c.read(clock_path) if clock_path.exists() else None
            chain = typed_chain(caught)
            candidate_refusal = (implementation == "compiled" and runtime_unsupported_chain(chain)
                and clock is not None and clock["status"] == "error" and clock["completed"] is False
                and clock["elapsed_seconds"] < c.INTERNAL_SECONDS)
            failure = dict(schema="compiled_physical_support_refusal_v1" if candidate_refusal else "physical_support_technical_failure_v1",
                implementation=implementation, phase=request["phase"], fixture_id=request["fixture_id"],
                repetition=request["repetition"], reason_kind="runtime_unsupported" if candidate_refusal else "technical_failure",
                request=c.artifact(target/"request.json"), clock=c.artifact(clock_path) if clock_path.exists() else None,
                exception_chain=chain, traceback=traceback.format_exc(), no_retry=True)
            receipt = c.save(target/("failure.json" if candidate_refusal else "technical_failure.json"), failure)
            if candidate_refusal:
                raise SupportUnsupported(receipt) from caught
            raise
        row = dict(implementation=implementation, phase=request["phase"], fixture_id=request["fixture_id"],
            repetition=request["repetition"], input_sha256=request["input_sha256"], physical_batch=8,
            occupied_slots=request["occupied_slots"], outputs=outputs,
            output_float32_sha256=hashlib.sha256(packed(outputs)).hexdigest(), clock=clock, no_retry=True)
        actual = c.save(target/"result.json", row); rows.append(actual)
        if eager_folder is not None:
            expected_path = eager_folder/relative/"result.json"
            expected = c.read(expected_path)
            c.require(expected["implementation"] == "eager" and expected["input_sha256"] == row["input_sha256"],
                      "exact corresponding eager support input")
            comparisons.append(compare_and_save(target/"result.json", expected["outputs"], c.artifact(expected_path),
                implementation=implementation, kind="cross_implementation_all_float32_rows",
                destination=target/"against_eager.json"))
        elif request["repetition"] > 0:
            first = folder/(Path("warmup")/"000" if request["phase"] == "warmup"
                            else Path("fixtures")/request["fixture_id"] / "000")/"result.json"
            comparisons.append(compare_and_save(target/"result.json", c.read(first)["outputs"], c.artifact(first),
                implementation=implementation, kind="eager_repeat_all_float32_rows", destination=target/"against_first.json"))
        if request["phase"] == "fixtures":
            comparisons.append(compare_and_save(target/"result.json", request["expected_occupied_outputs"], None,
                implementation=implementation, kind="accepted_saved_occupied_rows", destination=target/"against_saved.json",
                occupied_slots=request["occupied_slots"], source_bindings=request["expected_sources"]))
    c.require(len(rows) == 77, "five warmups and seventy-two fixed fixture requests")
    result = dict(status="exact_physical_support_passed", implementation=implementation, warmup_calls=5,
        fixture_calls=72, complete_calls=len(rows), results=rows, comparisons=comparisons,
        outputs_retained_before_comparison=True, physical_batch=8, no_retry=True,
        ordinary_decisions=0, new_games=0, training_updates=0)
    c.save(folder/"summary.json", result)
    return result
