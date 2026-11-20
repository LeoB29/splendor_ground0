"""Independent replay/accounting for frozen graph64 versus unchanged eager32.

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
from .schedule import validate_spec

ROLES = ("challenger", "opponent")
QUOTAS = {"challenger": 64, "opponent": 32}
MODES = {"challenger": "candidate", "opponent": "baseline"}
AUDIT_DOMAIN = "splendorust_graph64_vs_eager32_1s_20261025_audit_v1"
check_clock = original.check_clock
classify_failure = original.classify_failure
validate_result = original.validate_result
audit_decision = original.audit_decision


def require(value, message):
    if not value:
        raise ValueError("GRAPH64_STRENGTH_AUDIT: " + message)


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


def select_replay_sample(rows, maximum=48):
    """Fixed-hash minimum per role/seat/phase/private-reserve cell, at most48."""
    require(type(maximum) is int and maximum == 48, "frozen48 replay-cell bound")
    selected, seen = {}, set()
    for row in rows:
        require(row.get("stage") == "main", "main-only replay sample")
        require(type(row.get("completed")) is bool, "explicit completed flag")
        if not row["completed"]:
            continue
        pair, game, ply, role, seat = (row[key] for key in ("pair_index", "game_index", "ply", "player_role", "seat"))
        require(all(type(value) is int for value in (pair, game, ply, seat)) and
                0 <= pair < 100 and 0 <= game < 200 and game//2 == pair and 0 <= ply < 2000 and seat in (0, 1) and
                role in ROLES and row.get("arm") == "search" and row.get("mode") == MODES[role] and
                type(row.get("quota")) is int and row["quota"] == QUOTAS[role], "sample role/quota identity")
        identity = pair, game, ply
        require(identity not in seen, "duplicate completed decision")
        seen.add(identity)
        view = row["observer"]["observation"]
        require(view["current_player"] == seat, "sample observation seat")
        private = [any(reserve["source"] == "deck" for reserve in view["players"][actor]["reserves"])
                   for actor in (seat, 1-seat)]
        category = "both" if all(private) else "own" if private[0] else "opponent" if private[1] else "neither"
        phase = "0..19" if ply < 20 else "20..39" if ply < 40 else "40+"
        cell = role, phase, seat, category
        encoded = json.dumps([AUDIT_DOMAIN, pair, game, ply, role], separators=(",", ":"), ensure_ascii=True)
        key = hashlib.sha256(encoded.encode()).hexdigest()
        item = {"stage": "main", "pair_index": pair, "game_index": game, "ply": ply, "player_role": role,
                "arm": "search", "quota": QUOTAS[role], "mode": MODES[role], "seat": seat,
                "phase": phase, "private_category": category, "selection_sha256": key,
                "observer_sha256": digest(row["observer"])}
        if cell not in selected or key < selected[cell]["selection_sha256"]:
            selected[cell] = item
    cells = list(product(ROLES, ("0..19", "20..39", "40+"), (0, 1), ("neither", "own", "opponent", "both")))
    return {"schema": "splendorust_graph64_strength_replay_sample_v1", "domain": AUDIT_DOMAIN,
            "maximum_decisions": maximum, "decisions": [selected[cell] for cell in cells if cell in selected],
            "empty_cells": [{"player_role": role, "phase": phase, "seat": seat, "private_category": private}
                            for role, phase, seat, private in cells if (role, phase, seat, private) not in selected],
            "selection_uses_outcomes": False, "selection_uses_timing": False}


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
    """Descriptive verified-main timing, retaining failed/missing measurements."""
    games = [] if games is None else games
    require(all(row.get("stage") == "main" for row in rows) and all(game.get("stage") == "main" for game in games),
            "main-only performance")
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
    return {"schema": "splendorust_graph64_strength_performance_v1", "players": players,
            "natural_games": len(natural), "natural_game_length_plies": _distribution([game["plies"] for game in natural]),
            "natural_game_length_plies_by_challenger_seat": {str(seat): _distribution([game["plies"] for game in natural if game["challenger_seat"] == seat]) for seat in (0, 1)},
            "forfeits": [{key: game.get(key) for key in ("pair_index", "game_index", "challenger_seat", "loser", "winner", "failed_role", "failed_seat", "failure", "plies")}
                         for game in games if game["reason"] == "forfeit"],
            "memory_scope": "sampled worker plus owned descendants; no continuous-peak guarantee",
            "cpu_scope": "sampled worker plus owned descendants; excludes supervising parent",
            "log_integrity_scope": "verified separately by the runner; not inferred from timing receipts"}


summarise_performance = performance
