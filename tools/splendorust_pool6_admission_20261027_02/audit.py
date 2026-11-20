"""Independent live replay for the fixed pool6 candidate versus unchanged64.

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
QUOTAS = {"challenger": 64, "opponent": 64}
MODES = {"challenger": "research", "opponent": "frozen"}
PLACEMENT_MODE = "isolated"
AUDIT_DOMAIN = "splendorust_pool6_admission_20261027_02_audit_v1"
from tools.splendorust_sampling_stability_20261027 import audit as saved_science
from . import common as c

def check_clock(receipt, *, limit=1.):
    return saved_science.check_clock(receipt,limit=limit)

validate_research_result = saved_science.validate_research_result

def profile_identity(freeze, quota, world_pool, *, boundary=lambda:None):
    require(quota==64 and world_pool==6,'Fixed live profile')
    item=freeze['research_profiles']['64-6'];boundary();c.verify([item]);profile=c.read(c.ROOT/item['path'])
    require(profile['iterations']==64 and profile['world_pool']==6 and
        profile['profile_id']=='live_puct64_pool6_20261027_v1' and
        profile['wrapper_id']==profile['profile_id']+'_1s' and profile['internal_budget_seconds']==.9,'Exact candidate profile')
    return dict(profile_id=profile['profile_id'],profile_sha256=item['sha256'],wrapper_id=profile['wrapper_id'],
        native_binary_sha256=profile.get('native_binary_sha256',freeze.get('research_native_sha256')))

def validate_result(result,observer,route,*,identity=None):
    if route=='frozen':return original.validate_result(result,observer,'search',expected_quota=64)
    require(route=='research' and identity is not None,'Externally bound candidate identity')
    return validate_research_result(result,observer,quota=64,world_pool=6,seed_label='A',
        telemetry_required=False,schema='splendorust_pool6_live_v1',**identity)

def classify_failure(receipt):
    check_clock(receipt)
    status=receipt['status'];detail=str(receipt.get('detail','')).lower()
    require(status!='ok','Successful packet is not failure')
    require(not any(x in detail for x in ('pool6_live_audit','decision_quality_audit','sampling_stability',
        'source changed','ruleset','catalogue','resource','adapter_prerequisite')),'Integrity/prerequisite failure is not a game loss')
    if status=='worker_error' and ('sampling_internal_deadline:' in detail or 'pool6_live_internal_deadline:' in detail):
        return status
    if status=='timeout':
        require(receipt['elapsed_seconds']>=1. and receipt['received_before_deadline'] is False and
            (receipt['receipt_elapsed_seconds'] is None or receipt['receipt_elapsed_seconds']>1.),'Authentic full-clock timeout')
    return original.classify_failure(receipt)



def require(value, message):
    if not value:
        raise ValueError("POOL6_LIVE_AUDIT: " + message)


def _same(left, right):
    return json.dumps(left, sort_keys=True, separators=(",", ":"), allow_nan=False) == \
        json.dumps(right, sort_keys=True, separators=(",", ":"), allow_nan=False)


def _require_isolated(row):
    require(row.get('placement_mode') == PLACEMENT_MODE, 'qualified isolated supervision required')
    require('placement_error' not in row and 'profile_capture_error' not in row,
            'placement/observer capture failure is an infrastructure error')


def _spec_identity(spec):
    validate_spec(spec)
    require(type(spec) is dict and spec.get("stage") == "operational", "game stage")
    require(type(spec.get("challenger_seat")) is int and spec["challenger_seat"] in (0, 1), "challenger seat")
    for role, quota in QUOTAS.items():
        require(type(spec.get(role+"_quota")) is int and spec[role+"_quota"] == quota, "externally frozen quota")
    for key in ("pair_index", "game_index"):
        require(type(spec.get(key)) is int and spec[key] >= 0, "game schedule identity")
    require(spec["game_index"] // 2 == spec["pair_index"], "paired game index")


def audit_game(initial_snapshot, rows, ending, spec, *, freeze, boundary=lambda: None):
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
    identity=profile_identity(freeze,64,6,boundary=boundary)
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
            validate_result(receipt["result"], row["observer"], MODES[role], identity=identity)
            require(_same(row["action"], receipt["result"]["action"]) and
                    _same(row["trace"], receipt["result"]["diagnostics"]), "move/response evidence coupling")
            successor = ledger.transition(current, row["action"])
            require(_same(successor, row["after"]), "independent dictionary transition differs")
            decode_training_snapshot(successor)
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
        _require_isolated(row)
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
                "placement_mode": PLACEMENT_MODE,
                "phase": phase, "private_category": category, "selection_sha256": key,
                "observer_sha256": digest(row["observer"])}
        if cell not in selected or key < selected[cell]["selection_sha256"]:
            selected[cell] = item
    cells = list(product(ROLES, ("0..19", "20..39", "40+"), (0, 1), ("neither", "own", "opponent", "both")))
    return {"schema": "splendorust_pool6_live_replay_sample_v1", "domain": AUDIT_DOMAIN,
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
            "completed_at_or_above_engineering_margin": sum(value >= .7 for value in completed),
            "decision_cpu_seconds": _distribution([receipt["telemetry"][fields[0]] for receipt in receipts if fields[0] in receipt.get("telemetry", {})]),
            "sampled_tree_rss_bytes": _distribution([receipt["telemetry"][fields[1]] for receipt in receipts if fields[1] in receipt.get("telemetry", {})]),
            "cumulative_sampled_tree_peak_rss_bytes": _distribution([receipt["telemetry"][fields[2]] for receipt in receipts if fields[2] in receipt.get("telemetry", {})]),
            "missing_telemetry": {field: sum(field not in receipt.get("telemetry", {}) for receipt in receipts) for field in fields}}
    natural = [game for game in games if game["reason"] == "natural"]
    return {"schema": "splendorust_pool6_live_performance_v1", "players": players,
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
    require(role in ROLES and row.get("arm") == "search" and row.get("mode") == MODES[role] and
            type(row.get("quota")) is int and row["quota"] == QUOTAS[role], "boundary scientific role/quota identity")
    require(row.get("observation_mode") == "profiled", "qualified observer required for both players")
    require(not row.get("profile_capture_error"), "timing evidence capture failure is not a valid diagnostic completion")
    return {**row, "mode": "profiled"}


def validate_boundary_row(row):
    """Preserve partial failure evidence; authenticate every available marker."""
    from .runner import validate_profiles
    boundary_row(row)
    return validate_profiles(row['parent_profile'],row['child_profile'],row['clock'],dict(player='search',observer=row['observer']),row['mode'])


def validate_stopped_prefix(schedule,game_results,stop=None):
    validate_schedule(schedule)
    require(type(game_results) is list and len(game_results)<=len(schedule),'Bounded scheduled prefix')
    for index,game in enumerate(game_results):
        require(all(_same(game.get(k),v) for k,v in schedule[index].items()),'Exact scheduled prefix')
        require(game.get('complete') is True and game.get('reason') in ('natural','forfeit') and
            type(game.get('plies')) is int and 0<game['plies']<=2000,'Official ending')
        if c.PHASE=='qualification':require(game['reason']!='forfeit' or index==len(game_results)-1,'Ops stop first player fault')
    complete=len(game_results)==len(schedule)
    if stop is not None:
        require(c.PHASE=='qualification' and game_results and game_results[-1]['reason']=='forfeit','Only ops fault stops collection')
        last=game_results[-1]
        require(all(stop.get(k)==last[k] for k in ('stage','pair_index','game_index')) and
            stop['ply']==last['plies']-1 and stop['seat']==last['loser'] and stop['player_role']==last['failed_role'],
            'Stop exact failed game/ply/role')
        require(classify_failure(stop['clock'])==stop['failure_classification'],'Preserved fault classification')
    return dict(status='full_workload_verified' if complete else 'verified_prefix',planned_games=len(schedule),
        verified_games=len(game_results),unplayed_games=len(schedule)-len(game_results),
        complete_planned_workload=complete,forfeits=sum(g['reason']=='forfeit' for g in game_results),
        continuation_after_forfeit=c.PHASE=='assessment',no_strength_claim=not complete or c.PHASE!='assessment')


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
    return {"schema": "splendorust_pool6_live_analysis_v1", "rows": len(rows), "groups": summaries,
        "failures": failures, "observed_faults": len(failures), "all_collected_decisions_completed": not failures,
        "complete_planned_collection_inferred": False, "deadline_adjustment_seconds": 0.,
        "causal_bottleneck_proven": False, "no_strength_claim": True,
        "interpretation": ["Only unchanged trusted raw receipt clocks determine completion and forfeits.",
            "Missing failure sidecars remain missing; they cannot localize a delay without boundary evidence.",
            "Boundary intervals partition elapsed wall time, not CPU, inference or scheduling resource cost.",
            "Pipe operations overlap across processes; signed gaps are retained rather than clamped.",
            "This admission stops at its first fault; no assessment games are authorized by this runner.",
            "No-fault completion does not establish zero timeout risk, and700ms is descriptive only."]}
