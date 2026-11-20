"""Independent live reliability replay for fixed graph64 versus eager32.

Both players retain the original scientific arm ``search``. Matchup roles are
separate labels and never imply the old PUCT16 ``reference`` scientific arm.
No model, native process or environment transition is invoked by this module.
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
QUOTAS = {"challenger": 64, "opponent": 32}
MODES = {"challenger": "candidate", "opponent": "baseline"}
AUDIT_DOMAIN = "splendorust_live_boundary_1s_20261025_audit_v1"
check_clock = original.check_clock
classify_failure = original.classify_failure
validate_result = original.validate_result
audit_decision = original.audit_decision


def require(value, message):
    if not value:
        raise ValueError("LIVE_BOUNDARY_AUDIT: " + message)


def _same(left, right):
    return json.dumps(left, sort_keys=True, separators=(",", ":"), allow_nan=False) == \
        json.dumps(right, sort_keys=True, separators=(",", ":"), allow_nan=False)


def _spec_identity(spec):
    validate_spec(spec)
    require(type(spec) is dict and spec.get("stage") in ("main", "operational"), "game stage")
    require(type(spec.get("challenger_seat")) is int and spec["challenger_seat"] in (0, 1), "challenger seat")
    for role, quota in QUOTAS.items():
        require(type(spec.get(role+"_quota")) is int and spec[role+"_quota"] == quota, "externally frozen quota")
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
                row.get("arm") == "search" and row.get("mode") == MODES[role] and
                type(row.get("quota")) is int and row["quota"] == quota, "acting role/scientific arm/quota")
        ledger.check_observer(current, row["observer"])
        receipt = row["clock"]
        received = check_clock(receipt)
        require(receipt["deadline_seconds"] == 1., "full one-second move allowance required; stage clamp is not a loss")
        validate_boundary_row(row)
        if receipt["status"] == "ok":
            require(row.get("completed") is True and type(row.get("after")) is dict,
                    "completed move requires a concrete successor snapshot")
            validate_result(receipt["result"], row["observer"], "search", expected_quota=quota)
            require(_same(row["action"], receipt["result"]["action"]) and
                    _same(row["trace"], receipt["result"]["diagnostics"]), "move/response evidence coupling")
            successor = ledger.transition(current, row["action"])
            require(_same(successor, row["after"]), "independent dictionary transition differs")
            decode_training_snapshot(successor)
            # Check all phase-work/callback evidence, separately from the ledger.
            checked = audit_decision(row["observer"], row["action"], row["trace"], "search",
                                     before=current, after=successor, expected_quota=quota)
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
            "scientific_arms": {role: {"arm": "search", "quota": quota, "mode": MODES[role]} for role, quota in QUOTAS.items()},
            "trust_boundary": "shared corrected catalogue/atomic bridge and pinned native numerical search; initial seed and raw clock binding checked by runner"}


def select_replay_sample(rows, maximum=48, *, allow_operational=False):
    """Fixed-hash minimum per role/seat/phase/private-reserve cell, at most48."""
    require(type(maximum) is int and maximum == 48, "frozen48 replay-cell bound")
    require(type(allow_operational) is bool, "explicit stopped-prefix selector mode")
    selected, seen = {}, set()
    for row in rows:
        stage = row.get("stage")
        require(stage in (("operational", "main") if allow_operational else ("main",)), "approved replay sample stage")
        require(type(row.get("completed")) is bool, "explicit completed flag")
        if not row["completed"]:
            continue
        pair, game, ply, role, seat = (row[key] for key in ("pair_index", "game_index", "ply", "player_role", "seat"))
        require(all(type(value) is int for value in (pair, game, ply, seat)) and
                0 <= pair < (3 if stage == "operational" else 100) and 0 <= game < (6 if stage == "operational" else 200) and game//2 == pair and 0 <= ply < 2000 and seat in (0, 1) and
                role in ROLES and row.get("arm") == "search" and row.get("mode") == MODES[role] and
                type(row.get("quota")) is int and row["quota"] == QUOTAS[role], "sample role/quota identity")
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
                "arm": "search", "quota": QUOTAS[role], "mode": MODES[role], "seat": seat,
                "phase": phase, "private_category": category, "selection_sha256": key,
                "observer_sha256": digest(row["observer"])}
        if cell not in selected or key < selected[cell]["selection_sha256"]:
            selected[cell] = item
    cells = list(product(ROLES, ("0..19", "20..39", "40+"), (0, 1), ("neither", "own", "opponent", "both")))
    return {"schema": "splendorust_live_boundary_replay_sample_v1", "domain": AUDIT_DOMAIN,
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
        role = row.get("player_role")
        require(role in ROLES and row.get("arm") == "search" and row.get("mode") == MODES[role] and
                type(row.get("quota")) is int and row["quota"] == QUOTAS[role], "performance role/quota")
        check_clock(row["clock"])
    players = {}
    fields = ("request_cpu_seconds", "rss_bytes", "peak_sampled_rss_bytes")
    for role in ROLES:
        receipts = [row["clock"] for row in rows if row["player_role"] == role]
        completed = [receipt["receipt_elapsed_seconds"] for receipt in receipts if receipt["status"] == "ok"]
        arrived = [receipt["receipt_elapsed_seconds"] for receipt in receipts
                   if receipt["receipt_elapsed_seconds"] is not None and receipt["receipt_elapsed_seconds"] >= 0]
        players[role] = {"scientific_arm": "search", "quota": QUOTAS[role], "mode": MODES[role],
            "attempted_decisions": len(receipts), "completed_decisions": len(completed),
            "statuses": dict(Counter(receipt["status"] for receipt in receipts)),
            "completed_arrival_seconds": _distribution(completed), "all_observed_arrival_seconds": _distribution(arrived),
            "missing_arrival_timestamps": sum(receipt["receipt_elapsed_seconds"] is None for receipt in receipts),
            "negative_arrival_timestamps": sum(receipt["receipt_elapsed_seconds"] is not None and
                                               receipt["receipt_elapsed_seconds"] < 0 for receipt in receipts),
            "completed_above_engineering_margin": sum(value > .7 for value in completed),
            "decision_cpu_seconds": _distribution([receipt["telemetry"][fields[0]] for receipt in receipts if fields[0] in receipt.get("telemetry", {})]),
            "sampled_tree_rss_bytes": _distribution([receipt["telemetry"][fields[1]] for receipt in receipts if fields[1] in receipt.get("telemetry", {})]),
            "cumulative_sampled_tree_peak_rss_bytes": _distribution([receipt["telemetry"][fields[2]] for receipt in receipts if fields[2] in receipt.get("telemetry", {})]),
            "missing_telemetry": {field: sum(field not in receipt.get("telemetry", {}) for receipt in receipts) for field in fields}}
    natural = [game for game in games if game["reason"] == "natural"]
    return {"schema": "splendorust_live_boundary_performance_v1", "players": players,
            "scope": "collected_live_prefix_only", "games": len(games),
            "no_strength_claim": True, "failure_rate_confidence_interval": None,
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
    role = row.get("player_role")
    require(role in ROLES and row.get("arm") == "search" and row.get("mode") == MODES[role] and
            type(row.get("quota")) is int and row["quota"] == QUOTAS[role], "boundary scientific role/quota identity")
    require(row.get("observation_mode") == "profiled", "qualified observer required for both players")
    require(not row.get("profile_capture_error"), "timing evidence capture failure is not a valid diagnostic completion")
    return {**row, "mode": "profiled"}


def validate_boundary_row(row):
    """Preserve partial failure evidence; authenticate every available marker."""
    return boundaries.validate_profile_row(boundary_row(row))


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
                "without a captured fault all206 scheduled games are required")
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
    return {"schema": "splendorust_live_boundary_coverage_v1", "status": status,
            "planned_games": len(schedule), "verified_games": len(game_results),
            "unplayed_games": len(schedule)-len(game_results), "verified_games_by_stage": dict(counts),
            "complete_planned_workload": stop is None, "stopped_on_first_attributable_failure": stop is not None,
            "failure": failure, "no_strength_claim": True, "stopped_prefix_is_not_full_sample": stop is not None,
            "optional_stopping_confidence_interval": None,
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
            "completed_boundary_metrics": {name: {"unit": "nanoseconds" if name.endswith("_ns") else "count",
                "distribution": boundaries._distribution(values)} for name, values in sorted(group["metrics"].items())}})
    return {"schema": "splendorust_live_boundary_analysis_v1", "rows": len(rows), "groups": summaries,
        "failures": failures, "observed_faults": len(failures), "all_collected_decisions_completed": not failures,
        "complete_planned_collection_inferred": False, "deadline_adjustment_seconds": 0.,
        "causal_bottleneck_proven": False, "no_strength_claim": True,
        "interpretation": ["Only unchanged trusted raw receipt clocks determine completion and forfeits.",
            "Missing failure sidecars remain missing; they cannot localize a delay without boundary evidence.",
            "Boundary intervals partition elapsed wall time, not CPU, inference or scheduling resource cost.",
            "Pipe operations overlap across processes; signed gaps are retained rather than clamped.",
            "A first-fault stop yields a censored workload prefix, not a complete strength sample or a fixed-sample rate estimate.",
            "No-fault completion does not establish zero timeout risk, and700ms is descriptive only."]}
