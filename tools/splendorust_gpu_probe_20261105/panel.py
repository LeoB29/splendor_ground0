"""Model-free, authenticated saved inputs for the bounded GPU feasibility probe.

The normal-position source is an accepted *counterfactual corrected-rules*
parity panel, not newly played games or a representative self-play sample.
No model, optimizer, native process or GPU runtime is imported here.
"""
from __future__ import annotations

from copy import deepcopy
import hashlib
import json
import math
from pathlib import Path
import struct

ROOT = Path(__file__).resolve().parents[2]
STUDY = "outputs/splendorust_policy_head_20261104_01"
SCHEMA = "splendorust_saved_gpu_input_panel_20261105_v1"
BATCHES = (1, 2, 8, 32, 64)
WIDTH = 31 * 48
PINS = {
    "freeze.json": "52641e03dda1a14cb7f4675f97a3c1b1412226679b000db7a78354a6071b81b2",
    "final_closeout.json": "a188767b6479b861e78f05258609e33bf998a488d22d52ca8d5f4a916ff6ca21",
    "independent_review.json": "62812a9e4ebe538dc13712169f2120d1102a3e58bb4031ae27b7d2dc654220af",
}
CHECKPOINT_SHA256 = "ef8a4521cd6c03c7075f15efee23f4cde6ec94d1b5398765cf0c24a09ad745cb"
NATIVE_PANEL = "outputs/external_strength_screen_20261018_02/fixtures.json"
NATIVE_OUTPUT = STUDY + "/assessment/workers/operational_g000_control/entrypoint_clock.json"


def require(value, message):
    if not value:
        raise ValueError("GPU input panel: " + message)


def canonical(value):
    return json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False)


def digest(value):
    return hashlib.sha256(canonical(value).encode()).hexdigest()


class Sources:
    """Read only explicitly named files, with accepted/frozen byte bindings."""

    def __init__(self, root):
        self.root = Path(root).resolve()
        self.used = {}

    def read(self, path, *, binding=None, expected_hash=None, parse=True):
        path = Path(path)
        require(not path.is_absolute() and ".." not in path.parts, "relative source path")
        resolved = (self.root / path).resolve()
        require(resolved.is_relative_to(self.root), "source stays in workspace")
        relative = resolved.relative_to(self.root).as_posix()
        before = resolved.stat()
        require(before.st_size <= 128 * 1024**2, "bounded input artifact")
        raw = resolved.read_bytes()
        after = resolved.stat()
        require((before.st_size, before.st_mtime_ns) == (after.st_size, after.st_mtime_ns),
                "source changed while reading")
        actual = dict(path=relative, bytes=len(raw), sha256=hashlib.sha256(raw).hexdigest())
        if binding is not None:
            require(actual == binding, "accepted source hash/size: " + relative)
        if expected_hash is not None:
            require(actual["sha256"] == expected_hash, "fixed source identity: " + relative)
        self.used[relative] = actual
        return json.loads(raw) if parse else actual


def _indices(mask):
    require(type(mask) is list and len(mask) == 81 and all(type(x) is bool for x in mask),
            "full legal Main mask")
    result = [i for i, enabled in enumerate(mask) if enabled]
    require(bool(result), "nonempty lawful action support")
    return result


def make_fixture(fixture_id, roots, *, timing=False, expected=None, purpose="saved_corrected_root"):
    require(bool(roots), "nonempty fixture roots")
    hashes = [digest(row["tokens"]) for row in roots]
    result = dict(fixture_id=fixture_id, batch=len(roots),
                  tokens=[x for row in roots for x in row["tokens"]],
                  legal_policy_indices=[row["legal_policy_indices"] for row in roots],
                  root_ids=[row["root_id"] for row in roots],
                  unique_input_rows=len(set(hashes)), repeated_input_rows=len(hashes)-len(set(hashes)),
                  timing_fixture=timing, purpose=purpose)
    if expected is not None:
        result["historical_expected_cpu_outputs"] = expected
    return result


def _build_panel(root):
    from tools import future_training_rules_20261021 as guard
    from tools.external_strength_screen_20261018 import rules
    from tools.splendorust_policy_head_20261104 import data
    from tools.splendorust_policy_head_20261104.capture_adapter import public_root_tokens, main_support

    sources = Sources(root)
    accepted = {name: sources.read(STUDY + "/" + name, expected_hash=value)
                for name, value in PINS.items()}
    freeze, closeout, review = (accepted[x] for x in ("freeze.json", "final_closeout.json", "independent_review.json"))
    require(closeout["status"] == "accepted_completed_verified" and review["approved"] is True,
            "source study fully accepted")
    require(closeout["freeze"]["sha256"] == PINS["freeze.json"] and
            review["freeze_sha256"] == PINS["freeze.json"], "acceptance binds exact freeze")
    seal = sources.read(review["seal"]["path"], binding=review["seal"])
    frozen = {x["path"]: x for x in freeze["inputs"]}
    sealed = {x["path"]: x for x in seal["artifacts"]}

    def read_frozen(path):
        require(path in frozen, "source in accepted freeze: " + path)
        return sources.read(path, binding=frozen[path])

    def read_sealed(path):
        require(path in sealed, "evidence in accepted seal: " + path)
        return sources.read(path, binding=sealed[path])

    checkpoint = freeze["historical_parent_checkpoint"]
    require(checkpoint["sha256"] == CHECKPOINT_SHA256, "original parent only")
    sources.read(checkpoint["path"], binding=checkpoint, parse=False)
    # Authenticate every Python source admitted by the accepted freeze. This
    # covers transitive model-free decoder/encoder imports without importing
    # the model or relying on a self-declared catalogue tag.
    for path, binding in frozen.items():
        if path.endswith(".py"):
            sources.read(path, binding=binding, parse=False)

    fit_plan = read_frozen(STUDY + "/fit_plan.json")
    qualification = fit_plan["input_qualification"]
    cases = read_frozen(qualification["plan"]["path"])["loader_cases"]
    capture_rows = [read_frozen(path) for path in qualification["row_paths"]]
    require(len(capture_rows) == 16, "fixed sixteen actual-loader counterexamples")
    actual_checks = data.qualify_captured_rows(cases, capture_rows)
    proof = read_frozen(qualification["proof_path"])
    require(proof["actual_loader"] == actual_checks, "accepted actual corrected-loader proof")
    roots, excluded = [], []
    for index, source in enumerate(capture_rows):
        decoded = data.decode_row(source)
        case = cases["cases"][index] if index < 15 else cases["private_variant"]
        case_id = case.get("id", "private_hidden_repeat")
        if not decoded["eligible"]:
            excluded.append(dict(case_id=case_id, reason=decoded["reason"]))
            continue
        roots.append(dict(root_id="counterexample_" + case_id, tokens=decoded["tokens"],
                          legal_policy_indices=_indices(decoded["mask"]),
                          historical_output83=source["training_capture"]["root_output83"],
                          snapshot=source["before"], encoded_cost_checks=decoded["encoded_cost_checks"],
                          provenance=dict(kind="accepted_actual_collector_counterexample",
                                          source_path=qualification["row_paths"][index], case_id=case_id)))

    native_fixtures = read_frozen(NATIVE_PANEL)
    native_clock = read_sealed(NATIVE_OUTPUT)
    require(native_clock["status"] == "ok" and native_clock["accepted"] is True,
            "accepted original-control native parity request")
    native_suite = native_clock["result"]
    require(native_suite["status"] == "entrypoint_counterexamples_passed", "accepted native qualification")
    native_rows = native_suite["fresh_numeric_parity"]["search"]["rows"]
    require(len(native_fixtures) == len(native_rows) == 23, "fixed23 original native parity roots")
    native_roots = []
    for index, (fixture, result) in enumerate(zip(native_fixtures, native_rows, strict=True)):
        state = guard.decode_training_snapshot(fixture["snapshot"])
        observer = rules.observation_payload(state)
        tokens = public_root_tokens(observer)
        response = result["response"]
        output = response["host_policy"] + response["host_value"]
        require(result["fixture_index"] == index and result["observer_sha256"] == digest(observer),
                "native exact snapshot/observer binding")
        callback_hash = digest([digest(tokens), digest([output]), 1])
        require(response["neural_callback_count"] == response["neural_callback_rows"] == 1 and
                response["neural_callbacks"] == [callback_hash],
                "saved original CPU output binds actual independently encoded tokens")
        native_roots.append(dict(root_id=f"native_saved_{index:02d}", tokens=tokens,
                                legal_policy_indices=_indices(main_support(observer)[2]),
                                historical_output83=output, snapshot=fixture["snapshot"],
                                encoded_cost_checks=data.check_encoded_costs(fixture["snapshot"], tokens),
                                provenance=dict(kind="counterfactual_corrected_rules_parity_fixture",
                                                meaning=fixture["meaning"], source_path=NATIVE_PANEL,
                                                source_index=index, output_source=NATIVE_OUTPUT,
                                                turn_index=state.turn_index)))
    roots = native_roots + roots
    unique = {}
    for row in roots:
        unique.setdefault(digest(row["tokens"]), row)
    distinct = list(unique.values())
    require(len(distinct) >= 32, "at least32 distinct saved inputs for batch32 timing")

    # The historical eleven-case graph tests retain their exact original batch
    # composition; batched CPU arithmetic may differ from singleton arithmetic.
    historical = read_frozen(STUDY + "/fit_parity_fixtures.json")["fixtures"]
    outputs = read_sealed(STUDY + "/fit/parent_parity.json")
    require(outputs["status"] == "exact_eager_compiled_parity" and outputs["cases"] == len(historical) == 11,
            "eleven accepted original-parent graph cases")
    fixtures = []
    for fixture, output in zip(historical, outputs["rows"], strict=True):
        require(output["fixture_id"] == fixture["fixture_id"] and output["batch"] == fixture["batch"] and
                output["input_sha256"] == digest(fixture) and
                output["eager_outputs"] == output["compiled_outputs"] and output["float32_bits_equal"] is True,
                "original-parent historical exact graph evidence")
        masks, ids, hashes = [], [], []
        for offset in range(fixture["batch"]):
            tokens = fixture["tokens"][offset*WIDTH:(offset+1)*WIDTH]
            key = digest(tokens); hashes.append(key)
            if any(tokens):
                require(key in unique, "historical actual root independently decoded under corrected rules")
                masks.append(unique[key]["legal_policy_indices"]); ids.append(unique[key]["root_id"])
            else:
                ids.append("architecture_zero")
        fixtures.append(dict(fixture, legal_policy_indices=masks or None,
                             historical_expected_cpu_outputs=output["eager_outputs"],
                             root_ids=ids, unique_input_rows=len(set(hashes)),
                             repeated_input_rows=fixture["batch"]-len(set(hashes)), timing_fixture=False,
                             purpose="historical_cpu_exact_graph_parity"))
    for row in distinct:
        fixtures.append(make_fixture(row["root_id"], [row], expected=[row["historical_output83"]]))
    timing = {}
    for batch in BATCHES:
        part = [distinct[i % len(distinct)] for i in range(batch)]
        name = "timing_saved_cycle_batch" + str(batch)
        fixtures.append(make_fixture(name, part, timing=True, purpose="fixed_saved_input_timing"))
        timing[str(batch)] = name

    return dict(schema=SCHEMA, ruleset_id=guard.RULESET_ID, catalogue_sha256=guard.CATALOGUE_SHA256,
                turn_contract=guard.TURN_CONTRACT, checkpoint=checkpoint,
                checkpoint_origin="historically_trained_original_reference_not_a_corrected_training_claim",
                input_shape=["batch", 31, 48], output_shape=["batch", 83], dtype="float32",
                new_games=0, training_updates=0, model_calls_in_preparation=0,
                source_study=STUDY, source_acceptance_sha256=PINS["final_closeout.json"],
                source_inputs=sorted(sources.used.values(), key=lambda x:x["path"]),
                corrected_input_checks=actual_checks, excluded_counterexamples=excluded,
                saved_root_count=len(roots), distinct_input_rows=len(distinct),
                roots=roots, fixtures=fixtures, timing_fixtures=timing,
                batch_composition="first distinct roots in saved native23 then structural-case order; cycle only after exhaustion",
                scope="saved counterfactual corrected parity and structural fixtures; not representative selfplay or strength evidence")


def validate_structure(panel):
    require(type(panel) is dict and panel.get("schema") == SCHEMA, "panel schema")
    fixtures = panel.get("fixtures")
    require(type(fixtures) is list and bool(fixtures), "nonempty fixture inventory")
    ids, timing = set(), {}
    for fixture in fixtures:
        name, batch, tokens = fixture["fixture_id"], fixture["batch"], fixture["tokens"]
        require(type(name) is str and name and name not in ids, "unique fixture identity")
        ids.add(name)
        require(type(batch) is int and batch in BATCHES and type(tokens) is list and len(tokens) == batch*WIDTH,
                "fixed batch/input shape")
        require(all(type(x) in (float, int) and math.isfinite(x) and
                    struct.unpack("<f", struct.pack("<f", x))[0] == x for x in tokens), "finite exact float32 inputs")
        legal = fixture["legal_policy_indices"]
        require(legal is None or (type(legal) is list and len(legal) == batch and all(
            type(row) is list and bool(row) and row == sorted(set(row)) and
            all(type(x) is int and 0 <= x < 81 for x in row) for row in legal)), "per-row lawful support")
        require(legal is not None or not any(tokens), "only zero architecture inputs have no lawful mask")
        hashes = {digest(tokens[i*WIDTH:(i+1)*WIDTH]) for i in range(batch)}
        require(fixture["unique_input_rows"] == len(hashes) and
                fixture["repeated_input_rows"] == batch-len(hashes), "exact duplicate disclosure")
        if "historical_expected_cpu_outputs" in fixture:
            out = fixture["historical_expected_cpu_outputs"]
            require(type(out) is list and len(out) == batch and all(type(row) is list and len(row) == 83 and
                all(type(x) is float and math.isfinite(x) for x in row) for row in out), "historical full output shape")
        if fixture["timing_fixture"]:
            require(str(batch) not in timing and legal is not None, "one lawful timing fixture per batch")
            timing[str(batch)] = name
    require(timing == panel["timing_fixtures"] and set(timing) == {str(x) for x in BATCHES}, "exact timing inventory")
    return dict(status="panel_structure_valid", fixture_count=len(fixtures), timing_batches=list(BATCHES))


def validate_panel(panel, *, root=ROOT):
    """Repeat actual saved decoder/encoding/transition checks before model use.

    The evaluator calls verify_runtime_sources() once at its own startup; this
    function deliberately does not add another startup source-guard invocation.
    """
    validate_structure(panel)
    reconstructed = _build_panel(root)
    require(canonical(panel) == canonical(reconstructed), "panel exactly reproduces accepted sources and corrected encoding")
    return dict(status="actual_saved_input_pipeline_qualified", panel_sha256=digest(panel),
                source_count=len(panel["source_inputs"]), distinct_input_rows=panel["distinct_input_rows"],
                six_corrected_faces=panel["corrected_input_checks"]["six_corrected_faces"],
                historical_cpu_cases=sum("historical_expected_cpu_outputs" in x for x in panel["fixtures"]),
                model_calls=0, training_pipeline_qualified=False)


def prepare_panel(output_path=None, *, root=ROOT):
    """Prepare the frozen JSON exclusively from accepted saved evidence."""
    from tools import future_training_rules_20261021 as guard
    guard.verify_runtime_sources(root)
    panel = _build_panel(root)
    validate_structure(panel)
    if output_path is not None:
        destination = Path(output_path)
        if not destination.is_absolute():
            destination = Path(root) / destination
        destination.parent.mkdir(parents=True, exist_ok=True)
        with destination.open("x", encoding="utf-8", newline="\n") as stream:
            stream.write(canonical(panel) + "\n")
    return panel
