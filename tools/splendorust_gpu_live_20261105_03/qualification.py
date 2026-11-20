"""Actual-player qualification for the separately frozen GPU batching study.

Imports/preparation are model-free. Runtime operations are injected explicitly;
the owner authenticates corrected sources once at evaluator startup.
"""
from __future__ import annotations

from copy import deepcopy
from collections import Counter
import hashlib
import json
import math
import os
from pathlib import Path
import queue
import threading
import time
import traceback

from tools import future_training_rules_20261021 as guard
from tools.external_strength_screen_20261018 import audit as ledger, rules
from tools.external_strength_screen_20261018_02 import bridge
from tools.splendorust_search512_20261101 import adapters as original, corpus, scientific
from tools.splendorust_gpu_probe_20261105.worker import compare_outputs, NUMERICAL

ROOT = Path(__file__).resolve().parents[2]
PROBE_PANEL = "outputs/splendorust_gpu_probe_20261105_01/panel.json"
PROBE_PANEL_SHA = "072115bc0df33fb6a60e756dcc75e84afbdce1268e1903c629b600a5b6a551f8"
PROBE_CLOSEOUT = "outputs/splendorust_gpu_probe_20261105_01/final_closeout.json"
PROBE_CLOSEOUT_SHA = "925446af34c0d0fe361bf8b41dd91e5a28661327ac2686d0c3dddfcad8a471d6"
INTERNAL_SECONDS = 9.
OUTER_SECONDS = 10.


def require(condition, message):
    if not condition:
        raise ValueError("GPU live qualification: " + message)


def canonical(value):
    return json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False)


def digest(value):
    return hashlib.sha256(canonical(value).encode()).hexdigest()


def read(path):
    return json.loads(Path(path).read_text(encoding="utf-8"))


def artifact(path):
    path = Path(path).resolve()
    with path.open("rb") as stream:
        sha = hashlib.file_digest(stream, "sha256").hexdigest()
    return dict(path=path.relative_to(ROOT).as_posix(), bytes=path.stat().st_size, sha256=sha)


def save(path, value):
    path = Path(path); path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("x", encoding="utf-8", newline="\n") as stream:
        stream.write(canonical(value)+"\n"); stream.flush(); os.fsync(stream.fileno())
    return artifact(path)


def build_plan():
    """No new games, search calls or model imports during preparation."""
    panel_path, acceptance = ROOT/PROBE_PANEL, ROOT/PROBE_CLOSEOUT
    require(artifact(panel_path)["sha256"] == PROBE_PANEL_SHA and
            artifact(acceptance)["sha256"] == PROBE_CLOSEOUT_SHA, "accepted fixed inference input source")
    panel = read(panel_path)
    unique = {}
    for row in panel["roots"]:
        unique.setdefault(digest(row["tokens"]), row)
    require(len(unique) == 35, "exact35 outcome-blind saved roots")
    roots = []
    for row in unique.values():
        state = guard.decode_training_snapshot(row["snapshot"])
        observer = rules.observation_payload(state)
        roots.append(dict(root_id=row["root_id"], snapshot=row["snapshot"],
                          observer_sha256=digest(observer), expected_root_tokens=row["tokens"],
                          legal_policy_indices=row["legal_policy_indices"], provenance=row["provenance"]))
    return dict(schema="gpu_live_actual_entrypoint_qualification_20261105_v1",
                structural_suite=corpus.build_plan(512), decision_panel=roots,
                prior_panel=artifact(panel_path), prior_closeout=artifact(acceptance),
                source_inputs=panel["source_inputs"], numerical=deepcopy(NUMERICAL),
                required_full_suite_decisions_per_route=67, fixed_root_decisions_per_route=35,
                internal_seconds=INTERNAL_SECONDS, outer_seconds=OUTER_SECONDS,
                cross_backend_action_agreement_is_gate=False,
                within_route_full_result_reproduction_required=True,
                training_pipeline_qualified=False, strength_claim=False)


def profile_bindings():
    result = {}
    for quota in (16, 512):
        path = original.profile_path(quota, route="candidate")
        require(read(path) == original.profile(quota, route="candidate"), "unchanged original profile")
        result[str(quota)] = artifact(path)
    return result


def validate_runtime(runtime):
    require(runtime.route in ("cpu", "gpu") and runtime.runtime_sources == guard.SOURCE_HASHES,
            "actual evaluator startup corrected-source receipt")


def timed_call(operation, *, abort, on_receipt=lambda row: None,
               seconds=OUTER_SECONDS, clock=time.perf_counter):
    """Complete-result observation has an independent outer wall-clock guard.

    A daemon operation thread cannot keep a failed evaluator alive. On timeout,
    abort closes only its owned runtime; the parent also watches published turn
    deadlines and kills the exact job if native I/O cannot return.
    """
    require(type(seconds) in (int, float) and seconds > 0, "positive outer allowance")
    started = clock(); started_wall = time.time(); result_queue = queue.Queue(maxsize=1)

    def work():
        try:
            result_queue.put((True, operation(started)))
        except BaseException as error:
            result_queue.put((False, error))
    thread = threading.Thread(target=work, daemon=True, name="gpu-live-complete-turn")
    thread.start()
    try:
        remaining = seconds-(clock()-started)
        if remaining <= 0:
            raise queue.Empty
        ok, value = result_queue.get(timeout=remaining)
    except queue.Empty:
        receipt = dict(status="timeout", completed=False, elapsed_seconds=clock()-started,
                       started_perf_counter=started, started_unix=started_wall, outer_seconds=seconds,
                       internal_seconds=INTERNAL_SECONDS, no_retry=True)
        on_receipt(receipt)
        abort()
        raise TimeoutError("Complete player response exceeded fixed outer guard")
    ended = clock()
    receipt = dict(status="ok" if ok and ended-started < seconds else "error", completed=bool(ok),
                   elapsed_seconds=ended-started, started_perf_counter=started, ended_perf_counter=ended,
                   started_unix=started_wall, ended_unix=time.time(), outer_seconds=seconds,
                   internal_seconds=INTERNAL_SECONDS, no_retry=True)
    if ended-started >= seconds:
        receipt.update(status="timeout", completed=False)
        on_receipt(receipt); abort()
        raise TimeoutError("Complete response observed after fixed outer guard")
    if not ok:
        receipt["error"] = repr(value)
    on_receipt(receipt)
    if not ok:
        raise value
    return value, receipt


def ordinary(runtime, adapters, request, *, started, client_ids, evidence_callback=lambda row:None):
    """The actual handler used by counterexamples, fixed roots and live games."""
    require(type(request) is dict and set(request) == {"player", "observer"} and
            request["player"] in adapters, "strict ordinary request allowlist")
    arm, observer = request["player"], request["observer"]
    adapter = adapters[arm]
    with runtime.operation(client_ids[arm], started=started, kind="ordinary"):
        state = rules.state_from_observation(observer)
        require(rules.observation_payload(state) == observer and not state.terminal,
                "strict lawful corrected observer roundtrip")
        legal = rules.CorrectedEnv().legal_actions(state)
        require(bool(legal), "nonempty corrected complete legal support")
        deadline = started+INTERNAL_SECONDS
        adapter.request_deadline = deadline
        try:
            action, trace = adapter.choose(observer, diagnostic_capture=True)
            require(action in legal, "actual complete player action legal")
            quota = adapter.profile["iterations"]
            require(quota == (512 if arm == "search" else 16), "fixed scientific quota")
            result = dict(player=arm, action=rules.serialize_action(action), diagnostics=trace,
                          profile_id=adapter.profile["profile_id"],
                          profile_sha256=original.profile_sha256(quota, route="candidate"),
                          wrapper_id=adapter.profile["wrapper_id"], simulations_quota=quota,
                          observer_sha256=digest(observer),
                          legal_actions_sha256=digest([rules.serialize_action(a) for a in legal]), legal_count=len(legal),
                          ruleset=guard.RULESET_ID, catalogue_sha256=guard.CATALOGUE_SHA256,
                          turn_contract=guard.TURN_CONTRACT)
            evidence_callback(dict(event="complete_result_before_worker_validation",arm=arm,
                                   observer=observer,action=result["action"],diagnostics=trace,result=result))
            scientific.validate_result(result, observer, arm, expected_quota=quota)
            require(len(canonical(result).encode()) <= 8*1024**2, "bounded complete scientific response")
            if time.perf_counter() >= deadline:
                raise TimeoutError("Full ordinary response exceeded fixed9s internal guard")
            return result
        finally:
            adapter.request_deadline = None


class Forensics:
    def __init__(self, path):
        self.path = Path(path); self.sequence = 0; self.previous = None
        self.stream = self.path.open("x", encoding="utf-8", newline="\n")

    def record(self, event):
        row = dict(sequence=self.sequence, previous_sha256=self.previous, event=event)
        self.stream.write(canonical(row)+"\n"); self.stream.flush(); os.fsync(self.stream.fileno())
        self.previous = digest(row); self.sequence += 1

    def close(self):
        self.stream.close()


class BoundedAdapter:
    """Give direct/reference/native qualification calls the same outer guard."""
    def __init__(self, adapter, runtime, arm, recorder):
        self.adapter, self.runtime, self.arm, self.recorder = adapter, runtime, arm, recorder

    def __getattr__(self, name):
        value = getattr(self.adapter, name)
        if name not in ("direct_choose", "reference_choose", "exchange"):
            return value
        def bounded(*args, **kwargs):
            def operation(started):
                with self.runtime.operation(self.adapter.client_id,started=started,kind="qualification_"+name):
                    previous = self.adapter.request_deadline
                    self.adapter.request_deadline = started+INTERNAL_SECONDS if previous is None else min(previous,started+INTERNAL_SECONDS)
                    try:
                        return value(*args, **kwargs)
                    finally:
                        self.adapter.request_deadline = previous
            result, _ = timed_call(operation, abort=self.runtime.abort,
                                  on_receipt=lambda row:self.recorder(dict(operation="outer_clock", arm=self.arm,
                                                                         method=name, **row)))
            return result
        return bounded


def validate_fixed_result(case, result):
    state = guard.decode_training_snapshot(case["snapshot"])
    observer = rules.observation_payload(state)
    require(digest(observer) == case["observer_sha256"], "fixed lawful observer identity")
    scientific.validate_result(result, observer, "search", expected_quota=512)
    trace = result["diagnostics"]
    require(not trace["forced_pass"] and trace["phases"] and trace["phases"][0]["neural_callback_count"] > 0,
            "fixed35 roots genuinely reach learned main inference")
    callback = trace["phases"][0]["inference"][0]
    require(callback["batch"] == 1 and callback["tokens"] == case["expected_root_tokens"] and
            callback["input_sha256"] == digest(callback["tokens"]) and
            callback["output_sha256"] == digest(callback["outputs"]) and
            callback["outputs"] == [callback["output83"]], "actual fixed root encoding/output binding")
    encoded = corpus._actual_encoded_costs(state, trace)
    require(encoded["neural_input"], "actual model corrected card encoding")
    return dict(root_input_sha256=callback["input_sha256"], output83=callback["output83"],
                encoded_cost_checks=encoded, legal_policy_indices=case["legal_policy_indices"])


def run_qualification(plan, runtime, folder, boundary=lambda: None):
    validate_runtime(runtime)
    selected = plan["qualification"]
    require(selected == build_plan(), "frozen actual qualification plan")
    folder = Path(folder); folder.mkdir(parents=True, exist_ok=False)
    raw = Forensics(folder/"entrypoint_forensics.jsonl")
    adapters, clients = {}, {"search":"qualification_search", "reference":"qualification_reference"}
    try:
        adapters["search"] = runtime.make_adapter(clients["search"], 0, 512)
        adapters["reference"] = runtime.make_adapter(clients["reference"], 1, 16)
        def handle(request):
            boundary()
            value, _ = timed_call(lambda started:ordinary(runtime, adapters, request, started=started, client_ids=clients,
                                                        evidence_callback=raw.record),
                                  abort=runtime.abort, on_receipt=lambda row:raw.record(dict(operation="outer_ordinary_clock",
                                                                                         request_sha256=digest(request), **row)))
            return value
        bounded = {arm:BoundedAdapter(adapter, runtime, arm, raw.record) for arm, adapter in adapters.items()}
        suite = corpus.run_entrypoint_checks(bounded, boundary=boundary, request_callback=handle,
                                             evidence_callback=raw.record, route="candidate")
        require(suite["counts"] == {"complete_decisions":67} and suite["challenger_quota"] == 512,
                "complete actual512 entrypoint counterexamples")
        save(folder/"entrypoint_suite.json", suite)
        rows = []
        for index, case in enumerate(selected["decision_panel"]):
            boundary(); observer = rules.observation_payload(guard.decode_training_snapshot(case["snapshot"]))
            request = dict(player="search", observer=observer)
            value, clock = timed_call(lambda started:ordinary(runtime, adapters, request, started=started, client_ids=clients),
                                      abort=runtime.abort, on_receipt=lambda row:save(folder/"fixed_root_clocks"/f"{index:03d}.json",row))
            save(folder/"fixed_roots_raw"/f"{index:03d}.json",dict(root_id=case["root_id"],result=value,clock=clock))
            row = dict(root_id=case["root_id"], route=runtime.route, case_sha256=digest(case),
                       result=value, clock=clock, root=validate_fixed_result(case,value))
            save(folder/"fixed_roots"/f"{index:03d}.json", row); rows.append(row)
        result = dict(status="actual_entrypoint_qualified", route=runtime.route, complete_entrypoint_decisions=67,
                      fixed_root_decisions=35, ruleset_id=guard.RULESET_ID, catalogue_sha256=guard.CATALOGUE_SHA256,
                      turn_contract=guard.TURN_CONTRACT, suite_sha256=digest(suite), fixed_roots_sha256=digest(rows),
                      elapsed_root_seconds=[row["clock"]["elapsed_seconds"] for row in rows],
                      games=0, training_updates=0, cross_backend_actions_not_gated=True)
        save(folder/"qualification.json", result)
        raw.close()
        verification = verify_qualification(plan,folder)
        save(folder/"saved_qualification_audit.json",verification)
        return dict(summary=result, suite=suite, fixed_roots=rows)
    except BaseException:
        save(folder/"failure.json", dict(route=runtime.route, error=traceback.format_exc(), no_retry=True))
        runtime.abort(); raise
    finally:
        raw.close()
        for adapter in adapters.values():
            adapter.close()


def compare_qualification(cpu, gpu, plan):
    cases = plan["qualification"]["decision_panel"]
    require(cpu["summary"]["route"] == "cpu" and gpu["summary"]["route"] == "gpu" and
            len(cpu["fixed_roots"]) == len(gpu["fixed_roots"]) == len(cases) == 35, "both complete fixed35 routes")
    rows = []
    for case, left, right in zip(cases,cpu["fixed_roots"],gpu["fixed_roots"],strict=True):
        require(left["root_id"] == right["root_id"] == case["root_id"] and
                left["case_sha256"] == right["case_sha256"] == digest(case), "same exact saved root across routes")
        for row in (left,right):
            require(row["root"] == validate_fixed_result(case,row["result"]), "recomputed actual root qualification")
        comparison = compare_outputs([left["root"]["output83"]], [right["root"]["output83"]],
                                     [case["legal_policy_indices"]],plan["qualification"]["numerical"])
        require(comparison["passed"], "actual-player CPU/GPU root numerical tolerance")
        rows.append(dict(root_id=case["root_id"], full_action_equal=left["result"]["action"] == right["result"]["action"],
                         main_action_equal=bridge._main(rules.deserialize_action(left["result"]["action"])) ==
                                           bridge._main(rules.deserialize_action(right["result"]["action"])),
                         cpu_action=left["result"]["action"], gpu_action=right["result"]["action"], numerical=comparison,
                         cpu_seconds=left["clock"]["elapsed_seconds"], gpu_seconds=right["clock"]["elapsed_seconds"]))
    return dict(status="corrected_actual_player_numerics_qualified", roots=35, rows=rows,
                full_action_agreements=sum(x["full_action_equal"] for x in rows),
                main_action_agreements=sum(x["main_action_equal"] for x in rows),
                action_agreement_is_gate=False, lossless_player_replacement=False,
                total_cpu_seconds=sum(x["cpu_seconds"] for x in rows), total_gpu_seconds=sum(x["gpu_seconds"] for x in rows))


def read_forensics(path):
    events=[]; previous=None
    with Path(path).open(encoding="utf-8") as stream:
        for index,line in enumerate(stream):
            require(index < 4096 and len(line.encode()) <= 8*1024**2, "bounded complete forensic inventory")
            value=json.loads(line)
            require(set(value)=={"sequence","previous_sha256","event"} and value["sequence"]==index and
                    value["previous_sha256"]==previous, "exact durable forensic chain")
            previous=digest(value); events.append(value["event"])
    return events,previous


def verify_qualification(plan,folder):
    """Reconstruct actual entrypoint and fixed-root evidence without model calls."""
    folder=Path(folder); selected=plan["qualification"]
    require(selected==build_plan(), "immutable corrected qualification inputs")
    summary,suite=read(folder/"qualification.json"),read(folder/"entrypoint_suite.json")
    require(summary["status"]=="actual_entrypoint_qualified" and summary["route"] in ("cpu","gpu") and
            summary["suite_sha256"]==digest(suite) and suite["status"]=="entrypoint_counterexamples_passed" and
            suite["challenger_quota"]==512 and suite["reference_quota"]==16 and suite["counts"]=={"complete_decisions":67},
            "complete actual route qualification")
    events,terminal=read_forensics(folder/"entrypoint_forensics.jsonl")
    require(not any(e.get("event") in ("failed","entrypoint_failed","worker_decision_failed") for e in events),
            "failed invocation cannot be hidden by a later summary")
    ordinary=[e for e in events if e.get("operation")=="ordinary_worker_request" and e.get("event")=="result"]
    validated=[e for e in events if e.get("operation")=="ordinary_worker_request" and e.get("event")=="validated"]
    rejected=[e for e in events if e.get("operation")=="ordinary_worker_request" and e.get("event")=="rejected"]
    direct=[e for e in events if e.get("operation")=="reference_choose" and e.get("event")=="result"]
    same=[e for e in events if e.get("operation")=="direct_same_quota_choose" and e.get("event")=="result"]
    raw=[e for e in events if e.get("event")=="complete_result_before_worker_validation"]
    require(len(ordinary)==len(validated)==len(raw)==37 and len(rejected)==26 and len(direct)==len(same)==15,
            "all67 actual decisions and26 rejected requests retained")
    require(Counter((e["arm"],e["case_id"]) for e in rejected)==
            Counter((arm,"worker_rejection/"+name) for arm in corpus.ARMS for name in corpus.rejection_case_names()),
            "all strict identity/private-input negative cases")
    after={(e["arm"],e["case_id"]):e for e in validated}
    require(len(after)==37, "unique validated actual decision identities")
    for event,actual in zip(ordinary,raw,strict=True):
        arm=event["arm"]; result=event["result"]; observer=event["request"]["observer"]
        scientific.validate_result(result,observer,arm,expected_quota=512 if arm=="search" else 16)
        require(actual["result"]==result and actual["action"]==result["action"] and
                actual["diagnostics"]==result["diagnostics"] and actual["observer"]==observer,
                "actual prevalidation result binds completed handler evidence")
        ledger.check_observer(event["before_snapshot"],observer)
        guard.check_training_transition(event["before_snapshot"],result["action"],after[arm,event["case_id"]]["after_snapshot"])
    cases=selected["structural_suite"]["counterexamples"]
    fixtures={row["id"]:row for row in suite["fixtures"]}
    require(set(fixtures)=={x["id"] for x in cases} and len(fixtures)==15, "complete structural case inventory")
    for case in cases:
        state=guard.decode_training_snapshot(case["snapshot"])
        for arm in corpus.ARMS:
            matching=[e for e in ordinary if e["case_id"]==case["id"] and e["arm"]==arm]
            require(len(matching)==1 and matching[0]["before_snapshot"]==case["snapshot"] and
                    matching[0]["result"]==fixtures[case["id"]]["results"][arm], "actual case snapshot/result binding")
            encoded=corpus._actual_encoded_costs(state,matching[0]["result"]["diagnostics"],
                                                corrected_face="corrected_face" in case["tags"])
            require(dict(case_id=case["id"],arm=arm,**encoded) in suite["actual_encoded_costs"],
                    "recomputed actual corrected cost offsets")
    references={x["id"]:x["trace"] for x in suite["reference_reproductions"]}
    direct512={x["id"]:x for x in suite["direct_same_quota"]}
    require(set(references)==set(direct512)==set(fixtures), "full direct16/direct512 inventory")
    for event in direct:
        source=fixtures[event["case_id"]]["results"]["reference"]
        require(event["result"]==dict(action=source["action"],diagnostics=source["diagnostics"]) and
                event["result"]["diagnostics"]==references[event["case_id"]], "exact within-route direct16 reproduction")
    for event in same:
        source=fixtures[event["case_id"]]["results"]["search"]
        require(event["result"]==direct512[event["case_id"]]["result"]==dict(action=source["action"],diagnostics=source["diagnostics"])
                and direct512[event["case_id"]]["exact_callback_arrays_and_hashes"] is True, "exact within-route full512 reproduction")
    contract=suite["native_contract"]
    require(contract["status"]=="passed" and contract["count"]==len(contract["checks"])==28, "native support count")
    for check in contract["checks"]:
        snapshot=next(x["snapshot"] for x in cases if x["id"]==check["case_id"])
        require(check["inspection"]["legal"]==bridge.main_actions(rules.observation_payload(guard.decode_training_snapshot(snapshot))),
                "native ordered legal actions rederived")
    for value in suite["fresh_numeric_parity"].values():
        require(value["status"]=="passed" and value["checks"]==len(value["rows"])==23, "native-host parity23 cases")
        for row in value["rows"]:
            response=row["response"]; left=response["host_policy"]+response["host_value"]; right=response["native_policy"]+response["native_value"]
            require(len(left)==len(right)==83 and all(math.isfinite(x) for x in left+right) and
                    max(abs(a-b) for a,b in zip(left,right))<=.0005 and 0<=response["max_abs_delta"]<=.0005,
                    "saved native-host original tolerance")
    clocks=[e for e in events if e.get("operation") in ("outer_clock","outer_ordinary_clock")]
    require(len([e for e in clocks if e["operation"]=="outer_ordinary_clock"])==63 and
            all(e["status"] in ("ok","error") and 0<=e["elapsed_seconds"]<10. and
                e["outer_seconds"]==10. and e["internal_seconds"]==9. for e in clocks), "every actual qualification outer clock")
    roots=[]
    require([x.name for x in sorted((folder/"fixed_roots").glob("*.json"))]==[f"{i:03d}.json" for i in range(35)],
            "complete fixed35 saved decision inventory")
    for index,case in enumerate(selected["decision_panel"]):
        row=read(folder/"fixed_roots"/f"{index:03d}.json")
        require(row["root_id"]==case["root_id"] and row["route"]==summary["route"] and row["case_sha256"]==digest(case) and
                row["root"]==validate_fixed_result(case,row["result"]) and
                read(folder/"fixed_roots_raw"/f"{index:03d}.json")==dict(root_id=row["root_id"],result=row["result"],clock=row["clock"]),
                "full fixed-root raw result, corrected encoding and metadata")
        require(row["clock"]["status"]=="ok" and row["clock"]["completed"] is True and
                0<=row["clock"]["elapsed_seconds"]<10. and
                row["clock"]==read(folder/"fixed_root_clocks"/f"{index:03d}.json"), "fixed-root complete response guard")
        roots.append(row)
    require(summary["fixed_roots_sha256"]==digest(roots) and
            summary["elapsed_root_seconds"]==[x["clock"]["elapsed_seconds"] for x in roots], "fixed-root timing/source summary")
    return dict(status="saved_actual_entrypoint_verified",route=summary["route"],actual_decisions=67,fixed_root_decisions=35,
                ordinary_transitions=37,negative_requests=26,encoded_cost_checks=30,native_support_checks=28,
                direct_exact_reproductions=30,native_host_cases=46,forensic_events=len(events),
                forensic_terminal_sha256=terminal,model_calls=0)
