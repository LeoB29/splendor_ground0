"""Independent corrected-game audit for Dom PPO512 versus unchanged32.

The external adapter checks every atomic policy phase and the complete action.
The control retains its original search trace and verifier. No learned forward,
native process or referee transition is executed by this evidence reader.
"""
from __future__ import annotations

from collections import Counter
from copy import deepcopy
import hashlib
from itertools import product
import json
import math

from tools.external_strength_screen_20261018 import audit as ledger
from tools.external_strength_screen_20261018.common import digest
from tools.future_training_rules_20261021 import decode_training_snapshot
from tools.splendorust_search_scale_20261024 import audit as original
from .schedule import validate_spec, validate_schedule, make_schedule
from tools.splendorust_boundary_timing_20261025 import analyze as boundaries

ROLES = ("challenger", "opponent")
QUOTAS = {"challenger": None, "opponent": 32}
MODES = {"challenger": "dom", "opponent": "baseline"}
ARMS = {"challenger":"dom", "opponent":"search"}
PLACEMENT_MODE = "isolated"
AUDIT_DOMAIN = "dom_screen_1s_20261027_audit_v1"
check_clock = original.check_clock
classify_failure = original.classify_failure
validate_result = original.validate_result
audit_decision = original.audit_decision


def require(value, message):
    if not value:
        raise ValueError("DOM_SCREEN_AUDIT: " + message)


def _same(left, right):
    return json.dumps(left, sort_keys=True, separators=(",", ":"), allow_nan=False) == \
        json.dumps(right, sort_keys=True, separators=(",", ":"), allow_nan=False)


def _require_isolated(row):
    require(row.get('placement_mode') == PLACEMENT_MODE, 'qualified isolated supervision required')
    require('placement_error' not in row and 'profile_capture_error' not in row,
            'placement/observer capture failure is an infrastructure error')


def _spec_identity(spec):
    validate_spec(spec)
    require(type(spec) is dict and spec.get("stage") in ("main", "operational"), "game stage")
    require(type(spec.get("challenger_seat")) is int and spec["challenger_seat"] in (0, 1), "challenger seat")
    for role, quota in QUOTAS.items():
        require(spec.get(role+"_quota") == quota and type(spec.get(role+"_quota")) is type(quota), "externally frozen quota")
    for key in ("pair_index", "game_index"):
        require(type(spec.get(key)) is int and spec[key] >= 0, "game schedule identity")
    require(spec["game_index"] // 2 == spec["pair_index"], "paired game index")


def audit_game(initial_snapshot, rows, ending, spec, *, boundary=lambda: None):
    """Replay every actual move with the dictionary ledger; never call env.step.

    The runner additionally binds the initial snapshot to the frozen seed and
    each row's clock/input files. This checker binds all subsequent transitions.
    """
    _spec_identity(spec)
    require(all(key in ending and _same(ending[key], value) for key, value in spec.items()), "ending schedule binding")
    require(ending.get("complete") is True and type(ending.get("plies")) is int and ending["plies"] == len(rows),
            "complete game move count")
    require(type(rows) is list and 0 < len(rows) <= 2000, "bounded nonempty move inventory")
    decode_training_snapshot(initial_snapshot)
    current = deepcopy(initial_snapshot)
    failure, transitions = None, 0
    latency = {role: [] for role in ROLES}
    for ply, row in enumerate(rows):
        boundary()
        require(failure is None and not current["state"]["terminal"], "moves after final ending")
        require(type(row.get("ply")) is int and row["ply"] == ply and _same(row.get("before"), current),
                "noncontiguous move evidence")
        require(all(key in row and _same(row[key], spec[key]) for key in ("stage", "pair_index", "game_index")),
                "move schedule binding")
        seat = current["state"]["current_player"]
        role = "challenger" if seat == spec["challenger_seat"] else "opponent"
        quota = QUOTAS[role]
        require(type(row.get("seat")) is int and row["seat"] == seat and row.get("player_role") == role and
                row.get("arm") == ARMS[role] and row.get("mode") == MODES[role] and
                type(row.get("quota")) is type(quota) and row["quota"] == quota, "acting role/scientific arm/quota")
        ledger.check_observer(current, row["observer"])
        receipt = row["clock"]
        received = check_clock(receipt)
        require(receipt["deadline_seconds"] == 1., "full one-second move allowance required; stage clamp is not a loss")
        validate_boundary_row(row)
        if receipt["status"] == "ok":
            require(row.get("completed") is True and type(row.get("after")) is dict,
                    "completed move requires a concrete successor snapshot")
            validate_science(receipt["result"],row["observer"],role)
            require(_same(row["action"], receipt["result"]["action"]) and
                    _same(row["trace"], receipt["result"]["diagnostics"]), "move/response evidence coupling")
            successor = ledger.transition(current, row["action"])
            require(_same(successor, row["after"]), "independent dictionary transition differs")
            decode_training_snapshot(successor)
            # Check all phase-work/callback evidence, separately from the ledger.
            checked = audit_science(row,current,successor)
            require(checked["transition_checked"] is True, "transition replay was skipped")
            current = successor
            latency[role].append(received)
            transitions += 1
        else:
            require(row.get("completed") is False and row.get("action") is None and row.get("trace") is None and
                    row.get("after") is None, "failed move must not invent a successor or played action")
            failure = {"seat": seat, "role": role, "status": classify_failure(receipt)}
    boundary()
    require(type(ending.get("final")) is dict and _same(ending["final"], current), "saved final state")
    if failure is None:
        require(ending.get("reason") == "natural" and current["state"]["terminal"] and
                _same(ending.get("winner"), current["state"]["winner"]) and ending.get("loser") is None and
                ending.get("failed_role") is None and ending.get("failed_seat") is None,
                "official natural ending")
    else:
        require(ending.get("reason") == "forfeit" and not current["state"]["terminal"] and
                type(ending.get("loser")) is int and ending["loser"] == failure["seat"] and
                type(ending.get("winner")) is int and ending["winner"] == 1-failure["seat"] and
                ending.get("failed_role") == failure["role"] and
                type(ending.get("failed_seat")) is int and ending["failed_seat"] == failure["seat"],
                "forfeit attribution")
        saved_failure = ending.get("failure")
        require(saved_failure == failure["status"] or
                type(saved_failure) is dict and saved_failure.get("status") == failure["status"],
                "forfeit classifier identity")
    return {"status": "passed", "transitions": transitions, "failed_moves": int(failure is not None),
            "final_sha256": digest(current), "latencies": latency,
            "transition_oracle": "frozen independent dictionary ledger; no env.step",
            "scientific_arms": {role: {"arm": ARMS[role], "quota": quota, "mode": MODES[role]} for role, quota in QUOTAS.items()},
            "trust_boundary": "shared corrected catalogue/atomic bridge and pinned native numerical search; initial seed and raw clock binding checked by runner"}


def select_replay_sample(rows, maximum=48, *, allow_operational=False):
    """Fixed-hash minimum per role/seat/phase/private-reserve cell, at most48."""
    require(type(maximum) is int and maximum == 48, "frozen48 replay-cell bound")
    require(type(allow_operational) is bool, "explicit stopped-prefix selector mode")
    selected, seen = {}, set()
    for row in rows:
        stage = row.get("stage")
        require(stage in (("operational", "main") if allow_operational else ("main",)), "approved replay sample stage")
        _require_isolated(row)
        require(type(row.get("completed")) is bool, "explicit completed flag")
        if not row["completed"]:
            continue
        pair, game, ply, role, seat = (row[key] for key in ("pair_index", "game_index", "ply", "player_role", "seat"))
        require(all(type(value) is int for value in (pair, game, ply, seat)) and
                0 <= pair < (3 if stage == "operational" else 30) and 0 <= game < (6 if stage == "operational" else 60) and game//2 == pair and 0 <= ply < 2000 and seat in (0, 1) and
                role in ROLES and row.get("arm") == ARMS[role] and row.get("mode") == MODES[role] and
                type(row.get("quota")) is type(QUOTAS[role]) and row["quota"] == QUOTAS[role], "sample role/quota identity")
        identity = stage, pair, game, ply
        require(identity not in seen, "duplicate completed decision")
        seen.add(identity)
        view = row["observer"]["observation"]
        require(view["current_player"] == seat, "sample observation seat")
        private = [any(reserve["source"] == "deck" for reserve in view["players"][actor]["reserves"])
                   for actor in (seat, 1-seat)]
        category = "both" if all(private) else "own" if private[0] else "opponent" if private[1] else "neither"
        phase = "0..19" if ply < 20 else "20..39" if ply < 40 else "40+"
        cell = role, phase, seat, category
        encoded = json.dumps([AUDIT_DOMAIN, stage, pair, game, ply, role], separators=(",", ":"), ensure_ascii=True)
        key = hashlib.sha256(encoded.encode()).hexdigest()
        item = {"stage": stage, "pair_index": pair, "game_index": game, "ply": ply, "player_role": role,
                "arm": ARMS[role], "quota": QUOTAS[role], "mode": MODES[role], "seat": seat,
                "placement_mode": PLACEMENT_MODE,
                "phase": phase, "private_category": category, "selection_sha256": key,
                "observer_sha256": digest(row["observer"])}
        if cell not in selected or key < selected[cell]["selection_sha256"]:
            selected[cell] = item
    cells = list(product(ROLES, ("0..19", "20..39", "40+"), (0, 1), ("neither", "own", "opponent", "both")))
    return {"schema": "dom_screen_replay_sample_v1", "domain": AUDIT_DOMAIN,
            "placement_mode": PLACEMENT_MODE,
            "maximum_decisions": maximum, "decisions": [selected[cell] for cell in cells if cell in selected],
            "empty_cells": [{"player_role": role, "phase": phase, "seat": seat, "private_category": private}
                            for role, phase, seat, private in cells if (role, phase, seat, private) not in selected],
            "selection_uses_outcomes": False, "selection_uses_timing": False,
            "scope": "all_completed_played_prefix_cells" if allow_operational else "main_completed_cells",
            "failed_moves_excluded": True, "all_available_cells_selected": True}


def _distribution(values):
    require(all(type(value) in (int, float) and math.isfinite(value) and value >= 0 for value in values),
            "performance finite nonnegative values")
    ordered = sorted(values)
    def quantile(fraction):
        if not ordered:
            return None
        position = (len(ordered)-1)*fraction
        lo, hi = math.floor(position), math.ceil(position)
        return ordered[lo] + (ordered[hi]-ordered[lo])*(position-lo)
    return {"count": len(values), "mean": sum(values)/len(values) if values else None,
            "min": ordered[0] if values else None, "median": quantile(.5), "p95": quantile(.95),
            "max": ordered[-1] if values else None}


def performance(rows, games=None):
    """Descriptive played-prefix timing; no strength estimate or optional-stop CI."""
    games = [] if games is None else games
    require(all(row.get("stage") in ("operational", "main") for row in rows) and
            all(game.get("stage") in ("operational", "main") for game in games), "live performance stage")
    for row in rows:
        _require_isolated(row)
        role = row.get("player_role")
        require(role in ROLES and row.get("arm") == ARMS[role] and row.get("mode") == MODES[role] and
                type(row.get("quota")) is type(QUOTAS[role]) and row["quota"] == QUOTAS[role], "performance role/quota")
        check_clock(row["clock"])
    players = {}
    fields = ("request_cpu_seconds", "rss_bytes", "peak_sampled_rss_bytes")
    for role in ROLES:
        receipts = [row["clock"] for row in rows if row["player_role"] == role]
        completed = [receipt["receipt_elapsed_seconds"] for receipt in receipts if receipt["status"] == "ok"]
        arrived = [receipt["receipt_elapsed_seconds"] for receipt in receipts
                   if receipt["receipt_elapsed_seconds"] is not None and receipt["receipt_elapsed_seconds"] >= 0]
        players[role] = {"scientific_arm": ARMS[role], "quota": QUOTAS[role], "mode": MODES[role],
            "attempted_decisions": len(receipts), "completed_decisions": len(completed),
            "statuses": dict(Counter(receipt["status"] for receipt in receipts)),
            "completed_arrival_seconds": _distribution(completed), "all_observed_arrival_seconds": _distribution(arrived),
            "missing_arrival_timestamps": sum(receipt["receipt_elapsed_seconds"] is None for receipt in receipts),
            "negative_arrival_timestamps": sum(receipt["receipt_elapsed_seconds"] is not None and
                                               receipt["receipt_elapsed_seconds"] < 0 for receipt in receipts),
            "completed_above_engineering_margin": sum(value > .7 for value in completed),
            "completed_at_or_above_engineering_margin": sum(value >= .7 for value in completed),
            "decision_cpu_seconds": _distribution([receipt["telemetry"][fields[0]] for receipt in receipts if fields[0] in receipt.get("telemetry", {})]),
            "sampled_tree_rss_bytes": _distribution([receipt["telemetry"][fields[1]] for receipt in receipts if fields[1] in receipt.get("telemetry", {})]),
            "cumulative_sampled_tree_peak_rss_bytes": _distribution([receipt["telemetry"][fields[2]] for receipt in receipts if fields[2] in receipt.get("telemetry", {})]),
            "missing_telemetry": {field: sum(field not in receipt.get("telemetry", {}) for receipt in receipts) for field in fields}}
    natural = [game for game in games if game["reason"] == "natural"]
    return {"schema": "dom_screen_performance_v1", "players": players,
            "scope": "collected_live_prefix_only", "games": len(games),
            "no_strength_claim": True, "failure_rate_confidence_interval": None,
            "placement_mode": PLACEMENT_MODE, "engineering_margin_seconds": .7,
            "engineering_margin_is_descriptive_only": True,
            "engineering_margin_changes_completion_or_stopping": False,
            "engineering_margin_counter_definitions": {
                "completed_above_engineering_margin": "strictly greater than0.700 seconds",
                "completed_at_or_above_engineering_margin": "greater than or equal to0.700 seconds"},
            "zero_failures_is_not_zero_risk": True,
            "stage_counts": dict(Counter(game["stage"] for game in games)),
            "natural_games": len(natural), "natural_game_length_plies": _distribution([game["plies"] for game in natural]),
            "natural_game_length_plies_by_challenger_seat": {str(seat): _distribution([game["plies"] for game in natural if game["challenger_seat"] == seat]) for seat in (0, 1)},
            "forfeits": [{key: game.get(key) for key in ("pair_index", "game_index", "challenger_seat", "loser", "winner", "failed_role", "failed_seat", "failure", "plies")}
                         for game in games if game["reason"] == "forfeit"],
            "memory_scope": "sampled worker plus owned descendants; no continuous-peak guarantee",
            "cpu_scope": "sampled worker plus owned descendants; excludes supervising parent",
            "log_integrity_scope": "verified separately by the runner; not inferred from timing receipts"}


summarise_performance = performance


def boundary_row(row):
    """Project diagnostic mode without relabelling the scientific player."""
    _require_isolated(row)
    role = row.get("player_role")
    require(role in ROLES and row.get("arm") == ARMS[role] and row.get("mode") == MODES[role] and
            type(row.get("quota")) is type(QUOTAS[role]) and row["quota"] == QUOTAS[role], "boundary scientific role/quota identity")
    require(row.get("observation_mode") == "profiled", "qualified observer required for both players")
    require(not row.get("profile_capture_error"), "timing evidence capture failure is not a valid diagnostic completion")
    return {**row, "mode": "profiled"}


def validate_boundary_row(row):
    """Preserve partial failure evidence; authenticate every available marker."""
    return validate_profiles(boundary_row(row))


def validate_stopped_prefix(schedule, game_results, stop=None):
    """Accept either every natural game or the exact prefix ending in one loss.

    The live runner separately binds the stop clock to the final saved move and
    verifies raw logs, game replay, graph proofs, reproduction and evidence seals.
    A resource or source failure never becomes a successfully captured player
    fault merely because some earlier games have completed.
    """
    validate_schedule(schedule)
    require(type(game_results) is list and 0 < len(game_results) <= len(schedule), "bounded nonempty played prefix")
    for index, game in enumerate(game_results):
        expected = schedule[index]
        require(type(game) is dict and all(key in game and _same(game[key], value) for key, value in expected.items()),
                "played games must be exact scheduled prefix")
        require(game.get("complete") is True and type(game.get("plies")) is int and 0 < game["plies"] <= 2000,
                "only complete natural/forfeit game records in verified prefix")
        require(game.get("reason") in ("natural", "forfeit"), "unclassified game ending")
        require(game["reason"] != "forfeit" or index == len(game_results)-1,
                "no game may follow the first attributable player failure")
    last = game_results[-1]
    if stop is None:
        require(len(game_results) == len(schedule) and all(game["reason"] == "natural" for game in game_results),
                "without a captured fault all66 scheduled games are required")
        status = "full_workload_verified"
        failure = None
    else:
        require(type(stop) is dict and last["reason"] == "forfeit", "fault stop requires final forfeit")
        for key in ("stage", "pair_index", "game_index"):
            require(key in stop and _same(stop[key], last[key]), "stop game identity")
        require(type(stop.get("ply")) is int and stop["ply"] == last["plies"]-1, "stop must name final failed ply")
        require(stop.get("player_role") == last.get("failed_role") in ROLES and
                type(stop.get("seat")) is int and stop["seat"] == last.get("failed_seat") == last.get("loser") and
                last.get("winner") == 1-stop["seat"], "stop forfeit role/seat attribution")
        receipt = stop.get("clock")
        require(type(receipt) is dict and receipt.get("deadline_seconds") == 1., "original full-clock failure receipt required")
        classification = classify_failure(receipt)
        require(stop.get("clock_status") == receipt["status"] and
                stop.get("failure_classification") == classification and stop.get("request_id") == receipt["request_id"],
                "stop original failure receipt binding")
        ending_failure = last.get("failure")
        require(ending_failure == classification or type(ending_failure) is dict and
                ending_failure.get("status") == classification, "stop ending failure classification")
        status = "captured_fault_verified_prefix"
        timed = receipt["status"] == "timeout" or "one_second_internal_deadline" in str(receipt.get("detail", "")).lower()
        failure = {key: stop[key] for key in ("stage", "pair_index", "game_index", "ply", "player_role", "seat",
                                             "clock_status", "failure_classification", "request_id")}
        failure["kind"] = "deadline_failure" if timed else "attributable_player_protocol_or_legality_failure"
        failure["missing_boundary_sidecar_is_not_a_successful_boundary_capture"] = True
    counts = Counter(game["stage"] for game in game_results)
    return {"schema": "dom_screen_coverage_v1", "status": status,
            "planned_games": len(schedule), "verified_games": len(game_results),
            "unplayed_games": len(schedule)-len(game_results), "verified_games_by_stage": dict(counts),
            "complete_planned_workload": stop is None, "stopped_on_first_attributable_failure": stop is not None,
            "failure": failure, "no_strength_claim": True, "stopped_prefix_is_not_full_sample": stop is not None,
            "optional_stopping_confidence_interval": None,
            "placement_mode": PLACEMENT_MODE,
            "engineering_margin_is_descriptive_only": True,
            "no_strength_inference_or_player_promotion": True,
            "requires_runner_raw_replay_graph_reproduction_seal_and_cleanup": True}


def summarize_boundary_rows(rows, *, boundary=lambda: None):
    """Describe elapsed boundaries of every collected live row, including fault.

    The immutable observer analyzer expects its earlier panel-stage names. This
    summary calls its row validator directly and preserves the actual live stage,
    game identity and scientific mode in all recorded summaries.
    """
    groups, failures = {}, []
    for row in rows:
        boundary()
        require(row.get("stage") in ("operational", "main"), "live boundary stage")
        check_clock(row["clock"])
        checked = validate_boundary_row(row)
        role = row["player_role"]
        group = groups.setdefault((row["stage"], role), {"attempted": 0, "completed": 0,
            "statuses": Counter(), "arrivals": [], "metrics": {}, "missing_parent": 0, "missing_child": 0})
        group["attempted"] += 1
        group["statuses"][row["clock"]["status"]] += 1
        group["missing_parent"] += int(checked.get("missing_parent_profile", False))
        group["missing_child"] += int(checked.get("missing_child_profile", False))
        identity = {key: row[key] for key in ("stage", "pair_index", "game_index", "ply", "player_role", "seat", "mode", "quota")}
        if not row["completed"]:
            classification = classify_failure(row["clock"])
            failures.append({**identity, "request_id": row["clock"]["request_id"], "classification": classification,
                "receipt_elapsed_seconds": row["clock"].get("receipt_elapsed_seconds"), "boundary_evidence": checked,
                "partial_or_missing_evidence_not_promoted": True})
            continue
        group["completed"] += 1
        group["arrivals"].append(row["clock"]["receipt_elapsed_seconds"])
        def append(prefix, value):
            for name, item in value.items():
                field = prefix+"."+name if prefix else name
                if type(item) is dict:
                    append(field, item)
                elif type(item) is int:
                    group["metrics"].setdefault(field, []).append(item)
        append("", checked)
    summaries = []
    for (stage, role), group in sorted(groups.items()):
        summaries.append({"stage": stage, "player_role": role, "quota": QUOTAS[role], "mode": MODES[role],
            "observation_mode": "profiled", "attempted_decisions": group["attempted"],
            "completed_decisions": group["completed"], "statuses": dict(group["statuses"]),
            "missing_parent_profiles": group["missing_parent"], "missing_child_profiles": group["missing_child"],
            "completed_arrival_seconds": _distribution(group["arrivals"]),
            "completed_above_700ms": sum(value > .7 for value in group["arrivals"]),
            "completed_at_or_above_700ms": sum(value >= .7 for value in group["arrivals"]),
            "completed_boundary_metrics": {name: {"unit": "nanoseconds" if name.endswith("_ns") else "count",
                "distribution": boundaries._distribution(values)} for name, values in sorted(group["metrics"].items())}})
    return {"schema": "dom_screen_analysis_v1", "rows": len(rows), "groups": summaries,
        "failures": failures, "observed_faults": len(failures), "all_collected_decisions_completed": not failures,
        "complete_planned_collection_inferred": False, "deadline_adjustment_seconds": 0.,
        "causal_bottleneck_proven": False, "no_strength_claim": True,
        "interpretation": ["Only unchanged trusted raw receipt clocks determine completion and forfeits.",
            "Missing failure sidecars remain missing; they cannot localize a delay without boundary evidence.",
            "Boundary intervals partition elapsed wall time, not CPU, inference or scheduling resource cost.",
            "Pipe operations overlap across processes; signed gaps are retained rather than clamped.",
            "A first-fault stop yields a censored workload prefix, not a complete strength sample or a fixed-sample rate estimate.",
            "No-fault completion does not establish zero timeout risk, and700ms is descriptive only."]}


# The external sidecar retains the exact transport schema, with truthful backend identity.
from . import common as c, placement, clock as transport, adapter
from pathlib import Path
from tools.splendorust_search_scale_20261024.log_audit import verify_worker_logs
from tools.splendorust_live_boundary_20261025.raw_transport import _packet_evidence, _bind_parent_wire, _bind_child_wire, REQUEST_ID, MAX_FAILED_LINES
from tools.splendorust_search_scale_20261024.compression import verify_log
from tools.splendorust_boundary_timing_20261025.analyze import CHILD_SCHEMA, _int, _digest, _sha

def validate_science(result,observer,role):
    if role=='challenger':return adapter.validate_result(result,observer)
    return original.validate_result(result,observer,'search',expected_quota=32)

def audit_science(row,before,after):
    if row['player_role']=='opponent':
        return original.audit_decision(row['observer'],row['action'],row['trace'],'search',before=before,after=after,expected_quota=32)
    adapter.validate_result(row['clock']['result'],row['observer'])
    require(ledger.transition(before,row['action'])==after,'External independent transition differs')
    return dict(transition_checked=True)

def validate_profiles(row):
    if row['player_role']=='opponent':return boundaries.validate_profile_row(row)
    receipt=row['clock'];completed=row['completed'];check_clock(receipt)
    require(row['quota'] is None and row['mode']=='profiled' and receipt['deadline_seconds']==1. and completed==(receipt['status']=='ok'),'External observer scope')
    return external_profiles(row.get('parent_profile'),row.get('child_profile'),receipt,{'player':'dom','observer':row['observer']},completed=completed)

def external_profiles(parent,child,receipt,payload,*,completed,startup=False,worker_pid=None):
    require(parent is not None,'Original parent boundary evidence required')
    start,arrival,metrics=boundaries._validate_parent(parent,receipt,completed=completed)
    output=dict(complete_decision=completed,instrumented=True,missing_parent_profile=False,missing_child_profile=child is None,parent=metrics)
    if child is None:
        require(not completed,'Successful external child sidecar missing');return output
    _external_child_identity(child,parent,receipt,payload,None,completed=completed,startup=startup,worker_pid=worker_pid)
    profiles=child.get('profiles');require(type(profiles) is list and len(profiles)<=(0 if startup else 1),'External handle inventory')
    if completed and not startup:require(len(profiles)==1,'External ordinary handle absent')
    begin=end=None
    if profiles:_,begin,end=boundaries._validate_handle(profiles[0],child,completed=completed)
    output['child']=boundaries._boundaries(child,parent,start,arrival,begin=begin,end=end,startup=startup)
    return output
def _external_child_identity(child, parent, receipt, payload, quota, *, completed, startup=False, worker_pid=None):
    require(type(child) is dict and child.get("schema") == CHILD_SCHEMA, "child sidecar schema")
    require(len(json.dumps(child, sort_keys=True, separators=(",", ":"), allow_nan=False).encode()) <=
            (512*1024 if startup else 8192), "bounded child sidecar")
    require(child.get("request_id") == receipt["request_id"] and child.get("quota") == quota, "child request/quota")
    require(child.get("backend") == "dom_numpy_policy" and
            child.get("graph_identity_sha256") == None, "child backend identity")
    require(child.get("pid") == parent["child_pid"] and (worker_pid is None or child["pid"] == worker_pid),
            "parent/child supervised PID binding")
    python_pids = {p["pid"] for p in receipt.get("telemetry", {}).get("process_tree", []) if p.get("role") == "python"}
    require(python_pids == {child["pid"]}, "sampled Python PID identity differs")
    require(child.get("clock") == "perf_counter_ns" and child.get("clock_implementation") == "QueryPerformanceCounter()",
            "shared cross-process QPC runtime")
    for field in ("times_are_diagnostic_not_trusted_deadline_receipts", "success_sidecar_written_after_response_flush",
                  "post_flush_may_follow_parent_receipt", "child_read_may_precede_parent_dispatch_flush_completion",
                  "killed_worker_sidecar_may_be_missing", "text_transport_may_expand_newlines"):
        require(child.get(field) is True, "child sidecar scope " + field)
    require(child.get("outcome") in ("response_flushed", "failed_before_successful_emission"), "child outcome")
    require(not completed or child["outcome"] == "response_flushed", "successful request has failed sidecar")
    require(child.get("payload_sha256") == _digest(payload), "child input content binding")
    require(parent["request_raw_complete"] and not parent["request_raw_overflow"] and
            child.get("request_raw_sha256") == parent["request_raw_sha256"], "actual transported request hash differs")
    _sha(child.get("result_sha256"), "scientific result", nullable=not completed)
    if completed:
        require(child["result_sha256"] == _digest(receipt["result"]), "child scientific result binding")
        require(child.get("error") is None, "successful child error")
    else:
        require(child.get("error") is None or type(child.get("error")) is str and len(child["error"]) <= 4096,
                "bounded child error")



def verify_raw_transport(folder,freeze,envelope):
    folder=Path(folder);cleanup=c.read(folder/'cleanup.json');init=c.read(folder/'initialization.json')
    metadata=init['result'];external=metadata.get('player')=='dom';quota=None if external else metadata['search_simulations'];mode='dom' if external else 'baseline'
    c.require(external or quota==32,'Only Dom/eager32 worker modes')
    paths=sorted(path for path in folder.glob('request-*.json') if REQUEST_ID.fullmatch(path.stem))
    c.require(len(paths)<=2002,'Bounded request inventory')
    c.require([p.stem for p in paths]==[f'request-{i:08d}' for i in range(1,len(paths)+1)],
              'Contiguous original worker request inventory')
    receipts={path.stem:c.read(path) for path in paths};ids=list(receipts)
    failures=[rid for rid,receipt in receipts.items() if receipt['status']!='ok']
    c.require(not failures or failures==[ids[-1]],'At most one final failed request per worker')
    for rid in failures:
        original.classify_failure(receipts[rid])
    failed_id=failures[0] if failures else None
    packets={rid:[] for rid in ids};pending=bytearray();ready=False;index=0;partial=None
    def packet_line(raw):
        nonlocal ready,index
        envelope.guard();c.require(len(raw)<=4*c.MiB and raw.endswith(b'\n'),'Raw packet frame bound')
        item=_packet_evidence(raw);packet=item['packet']
        if not ready:
            c.require(packet=={'type':'ready','metadata':metadata},'Exact raw READY metadata changed')
            ready=True;return
        c.require(index<len(ids),'Unsolicited extra raw packet after successful final request')
        rid=ids[index];receipt=receipts[rid]
        if receipt['status']=='ok':
            c.require(type(packet) is dict and set(packet)=={'type','request_id','result'} and
                packet['type']=='response' and packet['request_id']==rid and packet['result']==receipt['result'],
                'Completed raw response differs from original trusted receipt')
            packets[rid].append(item);index+=1
        else:
            c.require(rid==failed_id and len(packets[rid])<MAX_FAILED_LINES,'Bounded failed raw packet suffix')
            packets[rid].append(item)
    def sink(chunk):
        pending.extend(chunk)
        while True:
            end=pending.find(b'\n')
            if end<0:break
            raw=bytes(pending[:end+1]);del pending[:end+1];packet_line(raw)
        c.require(len(pending)<=4*c.MiB,'Raw pending response frame bound');envelope.guard()
    verified=verify_log(folder/'stdout.log.gz',cleanup['log_compression']['stdout.log'],
        max_raw_bytes=16*c.MiB,max_compressed_bytes=16*c.MiB*16+128,
        deadline=envelope.effective_deadline(),sink=sink)
    c.require(ready,'Closed worker raw READY missing')
    if pending:
        c.require(failed_id is not None and index==len(ids)-1,'Partial output without final failed request')
        partial=dict(bytes=len(pending),sha256=hashlib.sha256(pending).hexdigest(),complete_packet=False)
    c.require(index==len(ids) if failed_id is None else index==len(ids)-1,
              'Missing completed raw response')
    input_ids=[p.stem for directory in ('inputs','startup_inputs') for p in (folder/directory).glob('*.json')]
    c.require(len(input_ids)==len(ids) and set(input_ids)==set(ids),'Unaccounted input request inventory')
    expected_profiles={rid+'_child_profile.json' for rid in ids}
    c.require({p.name for p in folder.glob('request-*_child_profile.json')}<=expected_profiles,
              'Unexpected child sidecar inventory')
    c.require({p.stem for p in (folder/'parent_profiles').glob('*.json')}==set(ids),
              'Complete parent sidecar inventory required')
    startup_count=0;checks=[]
    for rid,receipt in receipts.items():
        envelope.guard();c.require(receipt['request_id']==rid,'Raw trusted receipt identity')
        startup=folder/'startup_inputs'/(rid+'.json');ordinary=folder/'inputs'/(rid+'.json')
        c.require(startup.is_file()!=ordinary.is_file(),'Exactly one saved request payload type required')
        if startup.is_file():
            payload=c.read(startup);startup_count+=1
            c.require(receipt['status']=='ok','Admission failure cannot be accepted as a live-player failure')
            if payload=={'operation':'entrypoint_checks'}:
                c.require(folder.name.startswith('qualification_'),'Unexpected entrypoint request during play')
            else:raise ValueError('Unapproved startup operation')
        else:
            saved=c.read(ordinary);payload={'player':saved['player'],'observer':saved['observer']}
            c.require(receipt['deadline_seconds']==1.,'Unchanged full move allowance required')
        parent=c.read(folder/'parent_profiles'/(rid+'.json'))
        child_path=folder/(rid+'_child_profile.json')
        if child_path.exists():
            c.require(child_path.stat().st_size<=(512*1024+1 if startup.is_file() else 8193),
                      'Original bounded child sidecar size')
            child=c.read(child_path)
        else:
            c.require(receipt['status']!='ok','Completed child sidecar missing');child=None
        c.require(parent['request_id']==rid,'Parent sidecar request binding')
        parent_observed=_bind_parent_wire(parent,packets[rid],completed=receipt['status']=='ok')
        if child is not None:
            c.require(child['request_id']==rid and child['payload_sha256']==c.digest(payload) and
                parent['request_raw_sha256']==child['request_raw_sha256'] and
                type(parent['request_raw_bytes']) is int and 0<parent['request_raw_bytes']<=c.MiB and
                parent['request_raw_complete'] is True and parent['request_raw_overflow'] is False,
                'Actual parent written bytes differ from child raw request')
        binding=_bind_child_wire(child,packets[rid],receipt)
        if startup.is_file():
            external_profiles(parent,child,receipt,payload,completed=True,startup=True,worker_pid=init['identity']['pid']) if external else boundaries.validate_startup_profiles(parent,child,receipt,payload,32,worker_pid=init['identity']['pid'])
        else:
            profile_row=dict(mode='profiled',quota=quota,player_role='challenger' if external else 'opponent',
                observer=payload['observer'],clock=receipt,completed=receipt['status']=='ok',
                parent_profile=parent,child_profile=child)
            validate_profiles(profile_row)
        if receipt['status']=='worker_error':
            errors=[p['packet'] for p in packets[rid] if type(p['packet']) is dict and p['packet'].get('type')=='error']
            c.require(len(errors)==1 and errors[0].get('request_id')==rid and
                str(errors[0].get('error'))[:4096]==receipt['detail'],'Original worker error/raw packet binding')
        checks.append(dict(request_id=rid,status=receipt['status'],startup=startup.is_file(),
            parent_trusted_packet_observed=parent_observed,raw_packets=len(packets[rid]),**binding,
            failed_packets=[{key:item[key] for key in ('raw_sha256','raw_bytes','decode_error')}
                            for item in packets[rid]] if receipt['status']!='ok' else []))
    expected_startups=1 if folder.name.startswith('qualification_') else 0
    c.require(startup_count==expected_startups,'Original startup request inventory differs')
    return dict(status='passed',worker=folder.name,requests=len(ids),startup_requests=startup_count,
        observed=True,quota=quota,scientific_mode=mode,stdout_log=verified,exact_bytes_checked=True,
        failed_requests=len(failures),failed_request_id=failed_id,partial_failed_output=partial,requests_checked=checks,
        newline_normalization='terminal CRLF to LF only; JSON bytes never reserialized',
        failed_packets_remain_unaccepted=True,original_trusted_receipts_unchanged=True)


def check_row_integrity(row):
    c.require('profile_capture_error' not in row and 'placement_error' not in row,
        'Saved observer/placement evidence error; infrastructure stop')


def verify_placement_request(folder,request_id,freeze,mode,parent_profile):
    proof=transport.verify_worker_placement(folder,freeze,mode)
    value=c.read(folder/'placement_requests'/f'{request_id}.json')
    placement.verify_saved_request(value,freeze,mode,request_id=request_id,
        player_pid=proof['player_pid'],supervisor_identity=proof['supervisor'],parent_profile=parent_profile)
    launch=c.read(c.OUT/'launch.json')
    c.require((proof['supervisor']['pid'],proof['supervisor']['created'])==(launch['pid'],launch['created']),
        'Worker admission supervisor differs from owning runner')
    return value


def verify_saved_decision(row,freeze):
    ledger.check_observer(row['before'],row['observer'])
    role=row['player_role'];require(row.get('player')==('dom' if role=='challenger' else 'control32') and row.get('arm')==ARMS[role],'Player/scientific arm binding')
    require(row['mode']==MODES[row['player_role']] and row['quota']==QUOTAS[row['player_role']],'Player identity');check_row_integrity(row)
    c.require(row['placement_mode']=='isolated','Every collected decision must use isolated supervision')
    c.require(row['mode']==MODES[row['player_role']] and
        row.get('observation_mode')=='profiled','Saved role/backend/observer binding changed')
    path=c.ROOT/row['worker_clock_file']
    c.require(path.resolve().is_relative_to((c.OUT/'workers').resolve()),'Clock outside owned evidence')
    c.require(c.read(path)==row['clock'] and path.name==row['clock']['request_id']+'.json','Trusted clock record changed')
    c.require(c.read(path.parent/'inputs'/path.name)==dict(player=ARMS[row['player_role']],observer=row['observer'],before=row['before']),
        'Saved player input changed')
    for field,expected in (('parent_profile_file',path.parent/'parent_profiles'/path.name),
        ('child_profile_file',path.parent/(row['clock']['request_id']+'_child_profile.json'))):
        value=row.get(field);profile=row.get(field.removesuffix('_file'))
        if value is None:c.require(profile is None and not row['completed'],'Successful timing sidecar missing')
        else:
            actual=c.ROOT/value
            c.require(actual.resolve()==expected.resolve() and c.read(actual)==profile,'Saved timing sidecar binding changed')
    c.require(verify_placement_request(path.parent,row['clock']['request_id'],freeze,row['placement_mode'],row['parent_profile'])==
        row['placement_evidence'],'Saved placement receipt differs from move row')
    validate_boundary_row(row)
    original.check_clock(row['clock'])
    c.require(row['clock']['deadline_seconds']==1.,'Original full one-second decision required')
    if row['completed']:
        validate_science(row['clock']['result'],row['observer'],row['player_role'])
        c.require(row['action']==row['clock']['result']['action'] and row['trace']==row['clock']['result']['diagnostics'],
            'Scientific response and saved decision differ')
        audit_science(row,row['before'],row['after'])
    else:
        c.require(row['action'] is row['trace'] is row['after'] is None,'Failed request invents completed science')
        classify_failure(row['clock'])


def verify_closed_worker(folder,freeze,envelope):
    # A deadline can close the worker inside the trusted request, before the
    # caller persists its placement/profile sidecars. Refresh that early cache.
    envelope.close_worker(folder)
    envelope.guard()
    logs=verify_worker_logs(folder,boundary=envelope.guard,deadline=envelope.effective_deadline())
    raw=verify_raw_transport(folder,freeze,envelope)
    metadata=c.read(folder/'initialization.json')['result'];external=metadata.get('player')=='dom';quota=None if external else metadata['search_simulations']
    if not external:transport.verify_instrumentation_identity(folder,freeze,32,metadata)
    else:require(metadata==dict(**freeze['dom_ready_contract'],pid=c.read(folder/'initialization.json')['identity']['pid']),'External ready identity differs')
    wrapper=c.read(folder/'wrapper_identity.json')
    require(wrapper==dict(player='dom' if external else 'control32',metadata=metadata,placement_mode='isolated',inherited_reader=True,inherited_deadline=True,unchanged32=not external),'Wrapper identity changed')
    if external:
        identity=c.read(folder/'boundary_timing_identity.json')
        require(identity==dict(schema=CHILD_SCHEMA,backend='dom_numpy_policy',quota=None,graph_identity_sha256=None,metadata=metadata,scientific_response_fields_unchanged=True,clock='perf_counter_ns',clock_implementation='QueryPerformanceCounter()',evidence='atomic exclusive request sidecars; successful sidecar after stdout flush'),'External observer startup identity differs')
        require(c.read(folder/'runtime_sources.json')==metadata and c.read(folder/'player_identity.json')==metadata,'External source metadata differs')
        numerical=c.read(folder/'numerical_runtime.json')
        require(numerical.get('status')=='passed' and numerical.get('contract')==freeze['numpy_blas_runtime'] and
                numerical.get('require_one_thread') is True and numerical.get('no_model_forward') is True,
                'External numerical runtime receipt differs')
        require(numerical.get('observed_threads')==[
            dict(absolute_path=item['absolute_path'],num_threads=1) for item in freeze['numpy_blas_runtime']['libraries']],
            'External BLAS thread count or library inventory differs')
    proof=c.read(folder/'placement_admission.json');mode=proof['placement_mode']
    c.require(mode=='isolated','Every worker must use isolated supervision')
    transport.verify_worker_placement(folder,freeze,mode)
    requests=sorted((folder/'parent_profiles').glob('request-*.json'))
    placement_paths=sorted((folder/'placement_requests').glob('request-*.json'))
    c.require([p.name for p in requests]==[p.name for p in placement_paths],'Per-request placement inventory differs')
    for path in requests:
        envelope.guard();verify_placement_request(folder,path.stem,freeze,mode,c.read(path))
    graph=None
    if quota==32:c.require('inference_backend' not in metadata and not any(folder.glob('graph_*.json')),'Eager32 backend changed')
    envelope.close_worker(folder)
    envelope.guard()
    return dict(logs=logs,raw=raw,graph=graph,placement_mode=mode,quota=quota,placement_requests=len(requests))



def screen_assessment(games):
    main=[game for game in games if game['stage']=='main']
    faults=[game for game in games if game['reason']!='natural']
    natural_points=sum(.5 if game['winner'] is None else float(game['winner']==game['challenger_seat']) for game in main if game['reason']=='natural')
    full=len(main)==60 and len(games)==66 and not faults
    return dict(status='competitive' if full and natural_points>=15 else 'not_competitive' if full else 'not_admitted_incomplete_or_fault',
        complete_screen=full,natural_points=natural_points,denominator=60,minimum_natural_points=15,
        qualified_for_future_confirmation=full and natural_points>=15,external_faults=sum(game.get('failed_role')=='challenger' for game in faults),
        control_faults=sum(game.get('failed_role')=='opponent' for game in faults),optimized64_seen=False,confirmatory_strength_claim=False)
