"""Independent live replay for preselected256 candidate and unchanged128 reference.

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
QUOTAS = {"challenger": 256, "opponent": 128}
PLACEMENT_MODE = "isolated"
AUDIT_DOMAIN = "splendorust_search256_strength_20261029_02_audit_v1"
from tools.splendorust_sampling_stability_20261027 import audit as saved_science
from . import common as c
from .scientific import check_clock, classify_failure
from .adapters import validate_result
from .performance import performance, boundary_row, validate_boundary_row, summarize_boundary_rows


validate_research_result = saved_science.validate_research_result






def require(value, message):
    if not value:
        raise ValueError("SEARCH256_AUDIT: " + message)


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
        require(type(spec.get(role+"_quota")) is int and spec[role+"_quota"] == quota, "externally frozen quota")
    for key in ("pair_index", "game_index"):
        require(type(spec.get(key)) is int and spec[key] >= 0, "game schedule identity")


def modes_for(candidate):
    require(candidate == 'research', 'exact candidate route')
    return {'challenger': candidate, 'opponent': 'frozen'}


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
    modes = modes_for(spec['candidate'])
    identities={r:profile_identity(freeze,c.ROUTE_QUOTAS[r],3,route=r,boundary=boundary)
                for r in ('frozen',spec['candidate'])}
    current = deepcopy(initial_snapshot)
    failure, transitions = None, 0
    latency = {role: [] for role in ROLES}
    for ply, row in enumerate(rows):
        boundary()
        require(failure is None and not current["state"]["terminal"], "moves after final ending")
        require(type(row.get("ply")) is int and row["ply"] == ply and _same(row.get("before"), current),
                "noncontiguous move evidence")
        require(all(key in row and _same(row[key], spec[key]) for key in
                    ("stage", "candidate", "pair_index", "game_index")),
                "move schedule binding")
        seat = current["state"]["current_player"]
        role = "challenger" if seat == spec["challenger_seat"] else "opponent"
        quota = QUOTAS[role]
        require(type(row.get("seat")) is int and row["seat"] == seat and row.get("player_role") == role and
                row.get("arm") == "search" and row.get("mode") == modes[role] and
                type(row.get("quota")) is int and row["quota"] == quota, "acting role/scientific arm/quota")
        ledger.check_observer(current, row["observer"])
        receipt = row["clock"]
        received = check_clock(receipt)
        require(receipt["deadline_seconds"] == c.SECONDS_PER_MOVE, "full research move allowance required; stage clamp is not a loss")
        validate_boundary_row(row)
        if receipt["status"] == "ok":
            require(row.get("completed") is True and type(row.get("after")) is dict,
                    "completed move requires a concrete successor snapshot")
            validate_result(receipt["result"], row["observer"], modes[role], identity=identities[modes[role]])
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
                ending.get("failed_role") is None and ending.get("failed_seat") is None and
                not ending.get("administrative_forfeit",False) and ending.get("failure_policy") is None and ending.get("failure") is None,
                "official natural ending")
    else:
        require(ending.get("reason") == "forfeit" and ending.get("administrative_forfeit") is True and
                ending.get("failure_policy")=="research_hard_guard_9s_internal_10s_complete_response" and not current["state"]["terminal"] and
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
            "scientific_arms": {role: {"arm": "search", "quota": quota, "mode": modes[role]} for role, quota in QUOTAS.items()},
            "trust_boundary": "shared corrected catalogue/atomic bridge and pinned native numerical search; initial seed and raw clock binding checked by runner"}


def select_replay_sample(rows, maximum=48, *, stage, candidate, allow_operational=False):
    """Fixed-hash minimum per role/seat/phase/private-reserve cell, at most48."""
    require(type(maximum) is int and maximum == 48, "frozen48 replay-cell bound")
    require(type(allow_operational) is bool, "explicit stopped-prefix selector mode")
    require(stage in (('operational','main') if allow_operational else ('main',)), 'approved replay sample stage')
    modes = modes_for(candidate)
    selected, seen = {}, set()
    for row in rows:
        require(row.get('stage') == stage and row.get('candidate') == candidate,
                'replay caller must supply exactly one stage/candidate')
        _require_isolated(row)
        require(type(row.get("completed")) is bool, "explicit completed flag")
        if not row["completed"]:
            continue
        pair, game, ply, role, seat = (row[key] for key in ("pair_index", "game_index", "ply", "player_role", "seat"))
        from .schedule import PAIRS
        pair_limit = PAIRS[stage]
        game_base = 2*pair
        require(all(type(value) is int for value in (pair, game, ply, seat)) and
                0 <= pair < pair_limit and game in (game_base,game_base+1) and 0 <= ply < 2000 and seat in (0, 1) and
                role in ROLES and row.get("arm") == "search" and row.get("mode") == modes[role] and
                type(row.get("quota")) is int and row["quota"] == QUOTAS[role], "sample role/quota identity")
        identity = stage, candidate, pair, game, ply
        require(identity not in seen, "duplicate completed decision")
        seen.add(identity)
        view = row["observer"]["observation"]
        require(view["current_player"] == seat, "sample observation seat")
        private = [any(reserve["source"] == "deck" for reserve in view["players"][actor]["reserves"])
                   for actor in (seat, 1-seat)]
        category = "both" if all(private) else "own" if private[0] else "opponent" if private[1] else "neither"
        phase = "0..19" if ply < 20 else "20..39" if ply < 40 else "40+"
        cell = role, phase, seat, category
        encoded = json.dumps([AUDIT_DOMAIN, stage, candidate, pair, game, ply, role], separators=(",", ":"), ensure_ascii=True)
        key = hashlib.sha256(encoded.encode()).hexdigest()
        item = {"stage": stage, "candidate": candidate, "pair_index": pair, "game_index": game, "ply": ply, "player_role": role,
                "arm": "search", "quota": QUOTAS[role], "mode": modes[role], "seat": seat,
                "placement_mode": PLACEMENT_MODE,
                "phase": phase, "private_category": category, "selection_sha256": key,
                "observer_sha256": digest(row["observer"])}
        if cell not in selected or key < selected[cell]["selection_sha256"]:
            selected[cell] = item
    cells = list(product(ROLES, ("0..19", "20..39", "40+"), (0, 1), ("neither", "own", "opponent", "both")))
    return {"schema": "splendorust_search256_replay_sample_v1", "domain": AUDIT_DOMAIN,
            "stage": stage, "candidate": candidate,
            "placement_mode": PLACEMENT_MODE,
            "maximum_decisions": maximum, "decisions": [selected[cell] for cell in cells if cell in selected],
            "empty_cells": [{"player_role": role, "phase": phase, "seat": seat, "private_category": private}
                            for role, phase, seat, private in cells if (role, phase, seat, private) not in selected],
            "selection_uses_outcomes": False, "selection_uses_timing": False,
            "scope": "all_completed_played_prefix_cells" if allow_operational else "fixed_stage_candidate_completed_cells",
            "failed_moves_excluded": True, "all_available_cells_selected": True}






summarise_performance = performance






def validate_stopped_prefix(schedule,game_results,stop=None):
    validate_schedule(schedule)
    require(type(game_results) is list and len(game_results)<=len(schedule),'Bounded scheduled prefix')
    for index,game in enumerate(game_results):
        require(all(_same(game.get(k),v) for k,v in schedule[index].items()),'Exact scheduled prefix')
        require(game.get('complete') is True and game.get('reason') in ('natural','forfeit') and
            type(game.get('plies')) is int and 0<game['plies']<=2000,'Official ending')
        if game['stage']=='operational':require(game['reason']!='forfeit' or index==len(game_results)-1,'Ops stop first player fault')
    complete=len(game_results)==len(schedule)
    if stop is not None:
        require(game_results and game_results[-1]['stage']=='operational' and game_results[-1]['reason']=='forfeit','Only ops fault stops collection')
        last=game_results[-1]
        require(all(stop.get(k)==last[k] for k in ('stage','candidate','pair_index','game_index')) and
            stop['ply']==last['plies']-1 and stop['seat']==last['loser'] and stop['player_role']==last['failed_role'],
            'Stop exact failed game/ply/role')
        require(classify_failure(stop['clock'])==stop['failure_classification'],'Preserved fault classification')
    return dict(status='full_workload_verified' if complete else 'verified_prefix',planned_games=len(schedule),
        verified_games=len(game_results),unplayed_games=len(schedule)-len(game_results),
        complete_planned_workload=complete,forfeits=sum(g['reason']=='forfeit' for g in game_results),
        continuation_after_forfeit='main_only',no_strength_claim=not complete)




def profile_identity(freeze, quota, world_pool=3, *, route=None, boundary=lambda:None):
    from . import adapters
    require(route in ('research','frozen'), 'explicit256/128 route required')
    require(quota==c.ROUTE_QUOTAS[route] and world_pool==3,'Fixed three-world numerical player')
    boundary()
    return adapters.profile_identity(freeze,route,quota=quota)
