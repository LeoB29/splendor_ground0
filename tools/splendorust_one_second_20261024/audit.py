"""Independent evidence checks for the one-second Entity policy/search study.

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

SEARCH_PROFILE = "entity_onehot_puct16_canonical_atomic_prereveal_cpu_v1"
GREEDY_PROFILE = "entity_onehot_greedy_canonical_atomic_prereveal_cpu_20261023_v1"
SEARCH_WRAPPER = "entity_onehot_puct16_canonical_atomic_prereveal_cpu_1s_20261024_v1"
GREEDY_WRAPPER = "entity_onehot_greedy_canonical_atomic_prereveal_cpu_1s_20261024_v1"
PROFILE_SHA = "cd915908df1a7c6f39e55cfc9df59d88002a2b5ce167ce9512803e7c63d64424"
AUDIT_DOMAIN = "splendorust_policy_puct16_1s_20261024_audit_v1"
PLAYER_FAILURES = frozenset(("timeout", "worker_exit", "worker_error", "malformed", "wrong_request",
                            "illegal", "unsolicited", "write_error"))
RESULT_FIELDS = {"player", "action", "diagnostics", "observer_sha256", "legal_actions_sha256",
                 "legal_count", "ruleset", "catalogue_sha256", "turn_contract", "profile_sha256",
                 "profile_id", "wrapper_id"}


def require(value, message):
    if not value:
        raise ValueError("ONE_SECOND_AUDIT: " + message)


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


def validate_result(result, observer, arm):
    """Strict completed response binding; no acceptance based on action alone."""
    require(type(result) is dict and set(result) == RESULT_FIELDS, "response envelope fields")
    require(arm in ("search", "greedy") and result["player"] == arm, "response arm")
    state = rules.state_from_observation(observer)
    require(rules.observation_payload(state) == observer and not state.terminal, "lawful observation decoder")
    legal = [rules.serialize_action(a) for a in rules.legal_actions(state)]
    require(result["observer_sha256"] == digest(observer) and result["legal_actions_sha256"] == digest(legal)
            and type(result["legal_count"]) is int and result["legal_count"] == len(legal), "response observer/legal binding")
    require(result["ruleset"] == rules.RULESET_ID and result["catalogue_sha256"] == rules.CATALOGUE_SHA256
            and result["turn_contract"] == TURN_CONTRACT, "response corrected identity")
    require(result["profile_id"] == (SEARCH_PROFILE if arm == "search" else GREEDY_PROFILE) and
            result["profile_sha256"] == (PROFILE_SHA if arm == "search" else None) and
            result["wrapper_id"] == (SEARCH_WRAPPER if arm == "search" else GREEDY_WRAPPER), "response exact player")
    require(any(_same(result["action"], a) for a in legal), "response action legality")
    audit_decision(observer, result["action"], result["diagnostics"], arm)
    return True


def audit_decision(observer, action, trace, arm, before=None, after=None):
    require(arm in ("search", "greedy"), "decision arm")
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
    require(type(trace) is dict, "scientific trace")
    greedy_fields = {"schema", "profile_id", "observer_sha256", "observer_seed", "action_sha256",
                     "forced_pass", "phases", "inference_calls", "simulations"}
    search_fields = {"opponent_id", "profile_id", "observer_seed", "phases", "inference_calls", "inference_rows", "forced_pass"}
    require(set(trace) == (greedy_fields if arm == "greedy" else search_fields), "scientific trace fields")
    require(trace["profile_id"] == (GREEDY_PROFILE if arm == "greedy" else SEARCH_PROFILE) and
            type(trace["observer_seed"]) is int and trace["observer_seed"] == int(digest(observer)[:16], 16), "scientific identity/seed")
    if arm == "greedy":
        require(trace["schema"] == "entity_greedy_decision_v1" and trace["observer_sha256"] == digest(observer)
                and trace["action_sha256"] == digest(action) and type(trace["simulations"]) is int
                and trace["simulations"] == 0, "greedy identity/zero search")
    else:
        require(trace["opponent_id"] == "splendorust", "search identity")
    phases = trace["phases"]
    require(type(phases) is list, "phase list")
    main = bridge.main_actions(observer)
    forced = main == [{"kind": "Pass"}]
    require(type(trace["forced_pass"]) is bool and trace["forced_pass"] == forced, "forced pass identity")
    index, calls, simulations = 0, 0, 0
    branches = Counter()
    def select(native_observer, choices):
        nonlocal index, calls, simulations
        require(index < len(phases), "missing phase record")
        row = phases[index]
        index += 1
        require(type(row) is dict, "phase object")
        is_main = native_observer["phase"]["kind"] == "Main"
        retained = safe_choices(native_observer, choices)
        require(type(row.get("native_legal_count")) is int and row["native_legal_count"] == len(choices), "phase legal count")
        require(any(_same(row.get("action"), a) for a in retained), "selected native action outside safe legal support")
        if arm == "greedy":
            fields = {"phase", "observation_sha256", "legal_sha256", "branch", "native_legal_count", "retained_indices",
                      "canonical_indices", "selected_index", "action", "policy_logits_sha256", "inference", "canonical_logits"}
            require(set(row) == fields, "greedy phase fields")
            require(row["phase"] == native_observer["phase"]["kind"] and
                    row["observation_sha256"] == digest(native_observer) and row["legal_sha256"] == digest(choices), "phase observer/legal hashes")
            wanted_indices = [choices.index(a) for a in retained]
            require(_same(row["retained_indices"], wanted_indices), "independent safe support/order")
            require(_same(row["canonical_indices"], [canonical_index(a) for a in retained] if is_main else []), "canonical legal indices")
            selected = row["selected_index"]
            require(type(selected) is int and 0 <= selected < len(retained) and _same(row["action"], retained[selected]), "selected retained index")
            branch = "auxiliary_heuristic" if not is_main else "single_choice" if len(retained) == 1 else \
                "late_heuristic" if native_observer["turns"] >= 124 else "neural_greedy"
            require(row["branch"] == branch, "greedy branch")
            neural = branch == "neural_greedy"
            require(type(row["inference"]) is list and len(row["inference"]) == int(neural), "one greedy forward or declared exception")
            calls += int(neural)
            if neural:
                logits = row["canonical_logits"]
                require(type(logits) is list and len(logits) == 67 and
                        all(type(x) in (int, float) and math.isfinite(x) for x in logits), "canonical logits")
                require(row["policy_logits_sha256"] == digest(logits), "greedy logit hash")
                inference = row["inference"][0]
                require(set(inference) in ({"input_sha256", "output_sha256", "output83"},
                                          {"input_sha256", "output_sha256", "output83", "tokens"}), "greedy inference evidence fields")
                output = inference["output83"]
                require(type(output) is list and len(output) == 83 and
                        all(type(x) in (int, float) and math.isfinite(x) for x in output) and
                        inference["output_sha256"] == digest(output), "greedy raw output evidence")
                if "tokens" in inference:
                    require(inference["input_sha256"] == digest(inference["tokens"]), "encoder input hash")
                require(_f32(logits) == _f32(canonical_logits(output[:81])), "independent float32 raw/canonical mapping")
                best = max(range(len(retained)), key=lambda i: logits[canonical_index(retained[i])])
                require(selected == best, "greedy first maximum including ties")
            else:
                require(row["canonical_logits"] is None and row["policy_logits_sha256"] is None, "heuristic carries neural logits")
        else:
            fields = {"action", "simulations", "leaf_calls", "native_legal_count", "heuristic_phase"}
            require(set(row) in (fields, fields | {"inference"}), "search phase fields")
            heuristic = not is_main or len(choices) == 1 or native_observer["turns"] >= 124
            require(type(row["heuristic_phase"]) is bool and row["heuristic_phase"] == heuristic, "search branch")
            require(type(row["simulations"]) is int and row["simulations"] == (0 if heuristic else 16), "exact PUCT16 simulation quota")
            require(type(row["leaf_calls"]) is int and row["leaf_calls"] >= 0 and
                    (row["leaf_calls"] == 0 if heuristic else row["leaf_calls"] > 0), "search neural work branch")
            calls += row["leaf_calls"]
            simulations += row["simulations"]
            branch = "heuristic" if heuristic else "puct16"
        branches[branch] += 1
        return row["action"]
    selected = main[0] if forced else select(bridge.encode_observation(observer), main)
    resolved = rules.serialize_action(bridge.resolve_compound(observer, selected, select))
    require(index == len(phases) and _same(resolved, action), "all atomic phase choices/action agreement")
    require(type(trace["inference_calls"]) is int and trace["inference_calls"] == calls, "aggregate inference count")
    if arm == "search":
        require(type(trace["inference_rows"]) is int and trace["inference_rows"] == calls, "batch-one inference rows")
    return {"status": "passed", "arm": arm, "phases": index, "inference_calls": calls,
            "simulations": simulations, "branches": dict(branches), "observer_sha256": digest(observer),
            "action_sha256": digest(action), "transition_checked": after is not None}


def check_clock(receipt, *, limit=1.0):
    require(type(receipt) is dict and receipt.get("clock") == "perf_counter_ns" and
            isinstance(receipt.get("request_id"), str) and receipt["request_id"].startswith("request-"), "trusted clock identity")
    deadline, elapsed, received = (receipt[k] for k in ("deadline_seconds", "elapsed_seconds", "receipt_elapsed_seconds"))
    require(all(type(v) in (int, float) and math.isfinite(v) for v in (deadline, elapsed)) and
            0 < deadline <= limit and elapsed >= 0, "clock numeric bounds")
    require(received is None or type(received) in (int, float) and math.isfinite(received) and received <= elapsed,
            "trusted receipt time")
    timely = received is not None and 0 <= received <= deadline
    require(receipt["received_before_deadline"] is timely and receipt["accepted"] is (receipt["status"] == "ok"), "clock flags")
    require(receipt["status"] != "ok" or timely, "late packet accepted")
    return received


def classify_failure(receipt):
    """One shared prospective adjudication rule for collection and verification.

    A successful packet is never a failure. Resource/supervisor/source/adapter
    prerequisite faults stop the study. Explicit player deadline/illegal-action
    exceptions and attributable protocol failures can produce a forfeit.
    """
    check_clock(receipt)
    status, detail = receipt["status"], str(receipt.get("detail", "")).lower()
    require(status != "ok", "successful response is not a failure")
    require(status in PLAYER_FAILURES, "infrastructure failure is not a game loss")
    require(not any(x in detail for x in ("adapter_prerequisite", "one_second_adapter", "greedy_adapter",
                                          "one_second_audit", "source changed", "ruleset", "catalogue",
                                          "resource", "empty support", "deadlock", "frozen input")),
            "prerequisite/resource fault cannot be scored as player failure")
    if status == "worker_error":
        require("one_second_internal_deadline" in detail or "player_illegal_action" in detail,
                "unclassified worker exception stops incomplete")
    if status == "timeout" or "one_second_internal_deadline" in detail:
        require(receipt["deadline_seconds"] == 1.0,
                "stage-clamped request cannot be a player deadline loss")
    return status


def audit_game(initial_snapshot, rows, ending, *, search_seat, expected_identity=None, boundary=lambda: None):
    require(type(search_seat) is int and search_seat in (0, 1), "search seat")
    decode_training_snapshot(initial_snapshot)
    current = deepcopy(initial_snapshot)
    require(type(rows) is list and 0 < len(rows) <= 2000, "bounded nonempty game moves")
    counts = Counter()
    latency = {"search": [], "greedy": []}
    failure = None
    for ply, row in enumerate(rows):
        boundary()
        require(type(row) is dict and type(row.get("ply")) is int and row["ply"] == ply,
                "ordered actual game plies")
        if expected_identity is not None:
            require(all(row.get(k) == v for k, v in expected_identity.items()), "game/schedule move identity")
        require(failure is None and not current["state"]["terminal"], "moves after final ending")
        actor = current["state"]["current_player"]
        arm = "search" if actor == search_seat else "greedy"
        require(type(row.get("seat")) is int and row["seat"] == actor and row.get("arm") == arm, "acting seat/arm")
        require(_same(row["before"], current), "contiguous corrected pre-state")
        ledger.check_observer(current, row["observer"])
        receipt = row["clock"]
        received = check_clock(receipt)
        if receipt["status"] == "ok":
            require(row.get("completed") is True, "completed flag")
            validate_result(receipt["result"], row["observer"], arm)
            require(_same(row["action"], receipt["result"]["action"]) and
                    _same(row["trace"], receipt["result"]["diagnostics"]), "move/response evidence coupling")
            successor = ledger.transition(current, row["action"])
            require(_same(successor, row["after"]), "independent dictionary transition")
            decode_training_snapshot(row["after"])
            current = successor
            latency[arm].append(received)
            counts["transitions"] += 1
        else:
            require(row.get("completed") is False and row.get("action") is None and row.get("trace") is None
                    and row.get("after") in (None, current), "failed move must not invent a successor")
            failure = (actor, classify_failure(receipt))
    boundary()
    if failure is None:
        require(ending.get("reason") == "natural" and current["state"]["terminal"] and
                ending.get("winner") == current["state"]["winner"] and ending.get("loser") is None, "official natural ending")
    else:
        loser, status = failure
        require(ending.get("reason") == "forfeit" and not current["state"]["terminal"] and
                type(ending.get("loser")) is int and ending["loser"] == loser and
                type(ending.get("winner")) is int and ending["winner"] == 1-loser, "forfeit attribution")
        saved = ending.get("failure")
        require(saved == status or isinstance(saved, dict) and saved.get("status") == status, "forfeit failure receipt")
    if "final" in ending:
        require(_same(ending["final"], current), "saved final state")
    return {"status": "passed", "transitions": counts["transitions"], "failed_moves": int(failure is not None),
            "final_sha256": digest(current), "latencies": latency,
            "transition_oracle": "frozen independent dictionary ledger; no env.step",
            "trust_boundary": "shared corrected catalogue/legal enumeration/atomic phase bridge; pinned native search internals and numerical parity"}


def select_replay_sample(rows):
    selected = {}
    seen = set()
    for row in rows:
        if row.get("completed", True) is not True:
            continue
        require(row.get("stage", "main") == "main", "sample must contain main games only")
        pair, game, ply, arm, seat = (row[k] for k in ("pair_index", "game_index", "ply", "arm", "seat"))
        require(all(type(x) is int for x in (pair, game, ply, seat)) and 0 <= pair < 100 and
                0 <= game < 200 and game // 2 == pair and 0 <= ply < 2000 and seat in (0, 1)
                and arm in ("search", "greedy"), "sample row identity")
        identity = pair, game, ply, arm
        require(identity not in seen, "duplicate completed decision")
        seen.add(identity)
        view = row["observer"]["observation"]
        require(view["current_player"] == seat, "sample observation acting seat")
        private = [any(r["source"] == "deck" for r in view["players"][s]["reserves"])
                   for s in (seat, 1-seat)]
        category = "both" if all(private) else "own" if private[0] else "opponent" if private[1] else "neither"
        phase = "0..19" if ply < 20 else "20..39" if ply < 40 else "40+"
        cell = arm, phase, seat, category
        encoded = json.dumps([AUDIT_DOMAIN, pair, game, ply, arm], separators=(",", ":"), ensure_ascii=True)
        key = hashlib.sha256(encoded.encode("utf-8")).hexdigest()
        candidate = {"pair_index": pair, "game_index": game, "ply": ply, "arm": arm, "seat": seat,
                     "phase": phase, "private_category": category, "selection_sha256": key,
                     "observer_sha256": digest(row["observer"])}
        if cell not in selected or key < selected[cell]["selection_sha256"]:
            selected[cell] = candidate
    all_cells = list(product(("search", "greedy"), ("0..19", "20..39", "40+"), (0, 1),
                             ("neither", "own", "opponent", "both")))
    return {"schema": "splendorust_one_second_replay_sample_v1", "domain": AUDIT_DOMAIN,
            "maximum_decisions": 48, "decisions": [selected[cell] for cell in all_cells if cell in selected],
            "empty_cells": [{"arm": a, "phase": p, "seat": s, "private_category": r}
                            for a, p, s, r in all_cells if (a, p, s, r) not in selected],
            "selection_uses_outcomes": False}


def summarise_performance(rows, games):
    """Descriptive performance from verified main evidence, including failures.

    Worker CPU/RSS are sampled whole-tree receipts, not continuous peaks. A
    missing complete packet has no invented receipt latency.
    """
    def distribution(values):
        require(all(type(x) in (int, float) and math.isfinite(x) and x >= 0 for x in values),
                "performance numeric values")
        ordered = sorted(values)
        def q(p):
            if not ordered:
                return None
            x = (len(ordered) - 1) * p
            lo, hi = math.floor(x), math.ceil(x)
            return ordered[lo] + (ordered[hi] - ordered[lo]) * (x - lo)
        return {"count": len(values), "mean": sum(values) / len(values) if values else None,
                "min": min(values) if values else None, "median": q(.5), "p95": q(.95),
                "max": max(values) if values else None}
    require(all(r.get("stage", "main") == "main" for r in rows) and
            all(g.get("stage", "main") == "main" for g in games), "main-only performance")
    arms = {}
    for arm in ("search", "greedy"):
        selected = [row for row in rows if row["arm"] == arm]
        receipts = [r["clock"] for r in selected]
        timely = [r["receipt_elapsed_seconds"] for r in receipts if r["status"] == "ok"]
        arrived = [r["receipt_elapsed_seconds"] for r in receipts if r["receipt_elapsed_seconds"] is not None
                   and r["receipt_elapsed_seconds"] >= 0]
        missing = {key: sum(key not in r.get("telemetry", {}) for r in receipts)
                   for key in ("request_cpu_seconds", "rss_bytes", "peak_sampled_rss_bytes")}
        arms[arm] = {"attempted_decisions": len(selected), "completed_decisions": len(timely),
                     "statuses": dict(Counter(r["status"] for r in receipts)),
                     "completed_arrival_seconds": distribution(timely), "all_observed_arrival_seconds": distribution(arrived),
                     "missing_arrival_timestamps": sum(r["receipt_elapsed_seconds"] is None for r in receipts),
                     "completed_above_engineering_margin": sum(x > .7 for x in timely),
                     "decision_cpu_seconds": distribution([r["telemetry"]["request_cpu_seconds"] for r in receipts
                                                           if "request_cpu_seconds" in r.get("telemetry", {})]),
                     "sampled_tree_rss_bytes": distribution([r["telemetry"]["rss_bytes"] for r in receipts
                                                           if "rss_bytes" in r.get("telemetry", {})]),
                     "cumulative_sampled_tree_peak_rss_bytes": distribution([r["telemetry"]["peak_sampled_rss_bytes"] for r in receipts
                                                                            if "peak_sampled_rss_bytes" in r.get("telemetry", {})]),
                     "missing_telemetry": missing}
    natural = [g for g in games if g["reason"] == "natural"]
    lengths = [g["plies"] for g in natural]
    return {"schema": "splendorust_one_second_performance_v1", "arms": arms,
            "natural_games": len(natural), "natural_game_length_plies": distribution(lengths),
            "natural_game_length_plies_by_search_seat": {str(s): distribution([g["plies"] for g in natural
                                                                            if g["search_seat"] == s]) for s in (0, 1)},
            "forfeits": [{k: g.get(k) for k in ("pair_index", "game_index", "search_seat", "loser", "winner", "failure", "plies")}
                         for g in games if g["reason"] == "forfeit"],
            "memory_scope": "Sampled worker plus owned descendant RSS; no guaranteed continuous peak",
            "cpu_scope": "Sampled worker plus owned descendant CPU delta per complete request"}
