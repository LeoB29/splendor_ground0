"""Independent evidence checks for the one-second Entity PUCT search-budget study.

Actual transitions use the frozen dictionary ledger, never env.step. Legal
enumeration and atomic phase projections remain disclosed shared dependencies;
the public safe-choice filter and 81-to-67 map are independently expressed here.
No model or native process is loaded by this module.
"""
from __future__ import annotations

from collections import Counter
from copy import deepcopy
import hashlib
from itertools import combinations, product
import json
import math
import struct

from tools.external_strength_screen_20261018 import audit as ledger, rules
from tools.external_strength_screen_20261018.common import digest
from tools.external_strength_screen_20261018_02 import bridge
from tools.future_training_rules_20261021 import TURN_CONTRACT, decode_training_snapshot

from . import adapters as c

AUDIT_DOMAIN = "splendorust_puct_budget_1s_20261024_audit_v1"
PLAYER_FAILURES = frozenset(("timeout", "worker_exit", "worker_error", "malformed", "wrong_request",
                            "illegal", "unsolicited", "write_error"))
RESULT_FIELDS = {"player", "action", "diagnostics", "observer_sha256", "legal_actions_sha256",
                 "legal_count", "ruleset", "catalogue_sha256", "turn_contract", "profile_sha256",
                 "profile_id", "wrapper_id", "simulations_quota"}


def require(value, message):
    if not value:
        raise ValueError("SEARCH_SCALE_AUDIT: " + message)


def _same(left, right):
    # JSON comparison distinguishes bool from integer and excludes NaN.
    return json.dumps(left, sort_keys=True, separators=(",", ":"), allow_nan=False) == \
        json.dumps(right, sort_keys=True, separators=(",", ":"), allow_nan=False)


def canonical_index(action):
    kind = action["kind"]
    if kind == "Take":
        values = action["tokens"]
        require(len(values) == 5 and all(type(v) is int and 0 <= v <= 2 for v in values), "Take vector")
        if 2 in values:
            require(sum(values) == 2, "double Take")
            return 32 + values.index(2)
        return sum(1 << c for c, v in enumerate(values) if v)
    offsets = {"ReserveVisible": (37, "index", 12), "ReserveDeck": (49, "tier", 3),
               "BuyVisible": (52, "index", 12), "BuyReserved": (64, "index", 3)}
    require(kind in offsets, "canonical Main action")
    offset, field, count = offsets[kind]
    value = action[field]
    require(type(value) is int and 0 <= value < count, "canonical index bounds")
    return offset + value


def canonical_logits(policy81):
    require(type(policy81) is list and len(policy81) == 81 and
            all(type(x) in (int, float) and math.isfinite(x) for x in policy81), "raw policy logits")
    values = [-1e8] * 67
    masks = [sum(1 << c for c in cells) for count in (1, 2, 3)
             for cells in combinations(range(5), count)]
    for index, mask in enumerate(masks):
        values[mask] = policy81[30 + index]
    for target, source in ((32, 55), (37, 12), (49, 24), (52, 0), (64, 27)):
        count = {32: 5, 37: 12, 49: 3, 52: 12, 64: 3}[target]
        values[target:target + count] = policy81[source:source + count]
    return values


def _f32(values):
    return b"".join(struct.pack("<f", x) for x in values)


def safe_choices(observation, legal):
    """Independent Python statement of the pinned public sufficient conditions."""
    require(type(legal) is list and bool(legal), "nonempty native support")
    o = observation
    actor = o["players"][o["current"]]
    cards = rules.corrected_cards_by_id()
    def affordable(player, card_id):
        card = cards[bridge.RUST_CARD_IDS[card_id]]
        return sum(max(0, card.cost.get(color, 0) - player["bonuses"][i] - player["tokens"][i])
                   for i, color in enumerate(bridge.COLORS)) <= player["tokens"][5]
    def preserves(player, bank):
        opponents = o["count"] - 1
        if bank > 3 * opponents or any(r["card"] != 255 and affordable(player, r["card"])
                                       for r in player["reserved"]):
            return True
        affordable_market = sum(card != 255 and affordable(player, card) for card in o["market"])
        return affordable_market > opponents - math.ceil(bank / 3)
    choices = legal
    if o["reserved_counts"][o["current"]] == 3:
        retained = None
        if o["phase"]["kind"] == "Return":
            retained = []
            for action in legal:
                if action["kind"] != "Return":
                    continue
                player = deepcopy(actor)
                for color, amount in enumerate(action["tokens"]):
                    player["tokens"][color] -= amount
                bank = sum(o["bank"][c] + action["tokens"][c] for c in range(5))
                if preserves(player, bank):
                    retained.append(action)
        elif all(action["kind"] == "Take" for action in legal):
            retained = []
            for action in legal:
                player = deepcopy(actor)
                for color, amount in enumerate(action["tokens"]):
                    player["tokens"][color] += amount
                bank = sum(o["bank"][c] - action["tokens"][c] for c in range(5))
                if sum(player["tokens"]) <= 10 and preserves(player, bank):
                    retained.append(action)
        if retained and len(retained) < len(legal):
            choices = retained
    nobles = rules.corrected_nobles_by_id()
    noble_finishes = actor["score"] >= 12 and any(
        all(actor["bonuses"][c] >= nobles[bridge.RUST_NOBLE_IDS[n]].requirement.get(color, 0)
            for c, color in enumerate(bridge.COLORS)) for n in o["nobles"])
    if o["current"] + 1 == o["count"] and (o["final_round"] or noble_finishes):
        return choices
    other = (o["current"] + 1) % o["count"]
    player = o["players"][other]
    def blocks(action):
        if action["kind"] != "Take" or sum(actor["tokens"]) + sum(action["tokens"]) > 10:
            return False
        if any(o["bank"][c] != action["tokens"][c] for c in range(5)):
            return False
        return (o["reserved_counts"][other] == 3 and
                all(r["card"] != 255 for r in player["reserved"]) and
                all(not affordable(player, r["card"]) for r in player["reserved"]) and
                all(card == 255 or not affordable(player, card) for card in o["market"]))
    safe = [action for action in choices if not blocks(action)]
    return safe if safe and len(safe) < len(choices) else choices


def _expected_quota(arm, expected_quota=None, quota=None):
    """Quota comes from the frozen caller, never the worker's declaration."""
    require(arm in ("search", "reference"), "response arm")
    if quota is not None:
        require(expected_quota is None or expected_quota == quota, "conflicting expected quota")
        expected_quota = quota
    if arm == "reference":
        require(expected_quota is None or type(expected_quota) is int and expected_quota == 16,
                "reference must remain PUCT16")
        return 16
    require(type(expected_quota) is int and expected_quota in (128, 256),
            "search requires externally frozen challenger quota")
    return expected_quota


def validate_result(result, observer, arm, expected_quota=None, *, quota=None):
    """Strict completed response binding; no acceptance based on action alone."""
    q = _expected_quota(arm, expected_quota, quota)
    require(type(result) is dict and set(result) == RESULT_FIELDS, "response envelope fields")
    require(result["player"] == arm, "response arm")
    state = rules.state_from_observation(observer)
    require(rules.observation_payload(state) == observer and not state.terminal, "lawful observation decoder")
    legal = [rules.serialize_action(a) for a in rules.legal_actions(state)]
    require(result["observer_sha256"] == digest(observer) and result["legal_actions_sha256"] == digest(legal)
            and type(result["legal_count"]) is int and result["legal_count"] == len(legal), "response observer/legal binding")
    require(result["ruleset"] == rules.RULESET_ID and result["catalogue_sha256"] == rules.CATALOGUE_SHA256
            and result["turn_contract"] == TURN_CONTRACT, "response corrected identity")
    require(type(result["simulations_quota"]) is int and result["simulations_quota"] == q and
            result["profile_id"] == c.profile_id(q,route=c.route_from_profile_id(result["profile_id"],q)) and result["profile_sha256"] == c.profile_sha256(q,route=c.route_from_profile_id(result["profile_id"],q)) and
            result["wrapper_id"] == c.wrapper_id(q,route=c.route_from_profile_id(result["profile_id"],q)), "response exact player/quota")
    require(any(_same(result["action"], a) for a in legal), "response action legality")
    audit_decision(observer, result["action"], result["diagnostics"], arm, expected_quota=q)
    return True


def validate_search_work(trace, observer, quota=16):
    """Validate all native phase work and its externally specified profile."""
    require(type(quota) is int and quota in (16, 128, 256), "supported search quota")
    require(type(trace) is dict and trace.get('profile_id') == c.profile_id(quota,route=c.route_from_profile_id(trace.get("profile_id"),quota)), "search work exact profile")
    phases = trace['phases']
    require(type(phases) is list, 'search phase list')
    mains = bridge.main_actions(observer)
    forced = mains == [{'kind': 'Pass'}]
    require(type(trace['forced_pass']) is bool and trace['forced_pass'] == forced, 'forced pass identity')
    totals = {'inference_calls': 0, 'leaf_evaluations': 0,
              'heuristic_leaf_evaluations': 0, 'simulations': 0}
    index = 0
    def select(native_observer, choices):
        nonlocal index
        require(index < len(phases), 'missing phase record')
        phase = phases[index]
        index += 1
        require(type(phase) is dict and any(_same(phase.get('action'), action) for action in choices),
                'search phase action legality')
        work = check_search_phase(native_observer, choices, phase, quota=quota)
        for field in totals:
            totals[field] += work[field]
        return phase['action']
    chosen = mains[0] if forced else select(bridge.encode_observation(observer), mains)
    bridge.resolve_compound(observer, chosen, select)
    require(index == len(phases), 'all search phase records consumed')
    return check_search_totals(trace, totals)


def audit_decision(observer, action, trace, arm, before=None, after=None,
                   expected_quota=None, *, quota=None):
    q = _expected_quota(arm, expected_quota, quota)
    state = rules.state_from_observation(observer)
    legal = [rules.serialize_action(a) for a in rules.legal_actions(state)]
    require(any(_same(action, a) for a in legal), "complete action outside legal support")
    if before is not None:
        decode_training_snapshot(before)
        ledger.check_observer(before, observer)
    if after is not None:
        require(before is not None, "after requires before")
        decode_training_snapshot(after)
        ledger.check_step(before, action, after)
    search_fields = {"opponent_id", "profile_id", "observer_seed", "phases", "inference_calls", "inference_rows", "forced_pass"}
    require(type(trace) is dict and set(trace) == search_fields, "scientific trace fields")
    require(trace["profile_id"] == c.profile_id(q,route=c.route_from_profile_id(trace["profile_id"],q)) and type(trace["observer_seed"]) is int and
            trace["observer_seed"] == int(digest(observer)[:16], 16), "scientific identity/seed")
    require(trace["opponent_id"] == "splendorust", "search identity")
    phases = trace["phases"]
    require(type(phases) is list, "phase list")
    main = bridge.main_actions(observer)
    forced = main == [{"kind": "Pass"}]
    require(type(trace["forced_pass"]) is bool and trace["forced_pass"] == forced, "forced pass identity")
    index, calls, simulations, leaves, heuristic_leaves = 0, 0, 0, 0, 0
    branches = Counter()
    def select(native_observer, choices):
        nonlocal index, calls, simulations, leaves, heuristic_leaves
        require(index < len(phases), "missing phase record")
        row = phases[index]
        index += 1
        fields = {"action", "simulations", "leaf_calls", "native_legal_count", "heuristic_phase",
                  "neural_callbacks", "neural_callback_count", "neural_callback_rows"}
        require(type(row) is dict and set(row) in (fields, fields | {"inference"}), "search phase fields")
        retained = safe_choices(native_observer, choices)
        require(type(row["native_legal_count"]) is int and row["native_legal_count"] == len(choices), "phase legal count")
        require(any(_same(row["action"], a) for a in retained), "selected native action outside safe legal support")
        work = check_search_phase(native_observer, choices, row, quota=q)
        calls += work['inference_calls']
        simulations += work['simulations']
        leaves += work['leaf_evaluations']
        heuristic_leaves += work['heuristic_leaf_evaluations']
        branches[work['branch']] += 1
        return row["action"]
    selected = main[0] if forced else select(bridge.encode_observation(observer), main)
    resolved = rules.serialize_action(bridge.resolve_compound(observer, selected, select))
    require(index == len(phases) and _same(resolved, action), "all atomic phase choices/action agreement")
    require(type(trace["inference_calls"]) is int and trace["inference_calls"] == calls, "aggregate inference count")
    require(type(trace["inference_rows"]) is int and trace["inference_rows"] == calls, "batch-one inference rows")
    return {"status": "passed", "arm": arm, "simulations_quota": q, "phases": index, "inference_calls": calls,
            "leaf_evaluations": leaves, "heuristic_leaf_evaluations": heuristic_leaves,
            "simulations": simulations, "branches": dict(branches), "observer_sha256": digest(observer),
            "action_sha256": digest(action), "transition_checked": after is not None}



import re
SUPPORTED_QUOTAS = (16, 128, 256)
SEARCH_DEPTH = 64
FALLBACK_TURN = 124
SHA256 = re.compile(r"[0-9a-f]{64}\Z")


def require(value, message):
    if not value:
        raise ValueError("SEARCH_SCALE_AUDIT: " + message)


def check_search_phase(native_observer, choices, phase, quota=16):
    """Check one phase against branch semantics and actual callback receipts."""
    require(type(quota) is int and quota in SUPPORTED_QUOTAS, "supported simulation quota")
    require(type(phase) is dict, "search phase object")
    turns = native_observer['turns']
    require(type(turns) is int and 0 <= turns <= 2**32 - 1, "native turn count")
    require(type(choices) is list and bool(choices), "search phase legal support")
    heuristic = native_observer['phase']['kind'] != 'Main' or len(choices) == 1 or turns >= FALLBACK_TURN
    require(type(phase.get('heuristic_phase')) is bool and phase['heuristic_phase'] == heuristic,
            "search branch")
    simulations, leaves = phase.get('simulations'), phase.get('leaf_calls')
    require(type(simulations) is int and simulations == (0 if heuristic else quota),
            "exact configured simulation quota")
    require(type(leaves) is int and 0 <= leaves <= quota + 1, "native leaf evaluation bounds")
    callbacks = phase.get('neural_callbacks')
    count, rows = phase.get('neural_callback_count'), phase.get('neural_callback_rows')
    require(type(callbacks) is list and all(type(s) is str and SHA256.fullmatch(s) for s in callbacks),
            "neural callback digest receipts")
    require(type(count) is int and count == len(callbacks), "neural callback receipt count")
    require(type(rows) is int and rows == count, "per-phase batch-one callback rows")
    if heuristic:
        require(leaves == count == 0, "heuristic phase must use zero leaf evaluations and neural callbacks")
        branch = 'heuristic'
    else:
        require(1 <= count <= leaves <= quota + 1, "searched root neural/evaluation bounds")
        if turns + min(quota, SEARCH_DEPTH) < FALLBACK_TURN:
            require(count == leaves, "heuristic leaf unreachable at this root and simulation quota")
        elif turns == FALLBACK_TURN - 1:
            require(count == 1, "turn123 has exactly one root neural callback")
        branch = 'puct' + str(quota)
    if 'inference' in phase:
        capture = phase['inference']
        require(type(capture) is list and len(capture) == count, "diagnostic callback count")
        for index, (item, receipt) in enumerate(zip(capture, callbacks, strict=True)):
            base = {'input_sha256', 'output_sha256', 'batch'}
            full = base | {'tokens', 'outputs', 'output83'}
            require(type(item) is dict and set(item) in (base, full), "diagnostic callback fields")
            require(type(item['batch']) is int and item['batch'] == 1 and
                    all(type(item[k]) is str and SHA256.fullmatch(item[k])
                        for k in ('input_sha256', 'output_sha256')), "diagnostic callback identities")
            require(receipt == digest([item['input_sha256'], item['output_sha256'], item['batch']]),
                    "diagnostic/host callback digest mismatch")
            if set(item) == full:
                require(index == 0, "only root diagnostic stores arrays")
                tokens, outputs = item['tokens'], item['outputs']
                require(type(tokens) is list and len(tokens) == 31 * 48 and
                        all(type(x) in (int, float) and math.isfinite(x) for x in tokens),
                        "diagnostic root token shape")
                require(type(outputs) is list and len(outputs) == 1 and type(outputs[0]) is list and
                        len(outputs[0]) == 83 and all(type(x) in (int, float) and math.isfinite(x)
                                                    for x in outputs[0]), "diagnostic root output shape")
                require(item['input_sha256'] == digest(tokens) and item['output_sha256'] == digest(outputs)
                        and digest(item['output83']) == digest(outputs[0]), "diagnostic root array hashes")
    return {'simulations': simulations, 'inference_calls': count, 'leaf_evaluations': leaves,
            'heuristic_leaf_evaluations': leaves - count, 'branch': branch}


def check_search_totals(trace, totals):
    for field in ('inference_calls', 'inference_rows'):
        require(type(trace.get(field)) is int and trace[field] == totals['inference_calls'],
                'aggregate host ' + field)
    return totals


from tools.splendorust_pool6_strengthfirst_20261028.clock_policy import check_clock, boundary, INTERNAL_SECONDS, SECONDS_PER_MOVE

def profile_identity(freeze, quota=None, world_pool=3, *, route=None, boundary=lambda:None):
    require(world_pool==3 and route in c.ROUTES,'Explicit unchanged pool3 route')
    boundary()
    return c.profile_identity(freeze,route,quota)

def validate_execution_result(result,observer,route,*,identity):
    c.validate_execution_identity(result,observer,route,identity=identity)
    return validate_result(result,observer,result['player'],expected_quota=identity['quota'])

def classify_failure(receipt):
    check_clock(receipt,limit=10.)
    status=receipt['status'];detail=str(receipt.get('detail',''))
    require(status!='ok' and receipt['result'] is None,'Only unsuccessful undecoded deadline receipts can forfeit')
    if status=='timeout':
        require(receipt['elapsed_seconds']>=10. and receipt['received_before_deadline'] is False and
          (receipt['receipt_elapsed_seconds'] is None or receipt['receipt_elapsed_seconds']>10.),'Full ten-second timeout')
        return 'research_outer_hard_guard'
    require(status=='worker_error' and 'STRENGTH_INTERNAL_DEADLINE:' in detail and
        not any(x in detail.lower() for x in ('source changed','ruleset','catalogue','resource','prerequisite','audit:')),
        'Only explicit nine-second player deadline is a forfeit')
    return 'research_internal_hard_guard'
