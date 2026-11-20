"""Model-free search-budget corruption, clock, replay and statistics checks."""
from copy import deepcopy
import hashlib
import json
import pytest

from tools.external_strength_screen_20261018 import rules
from tools.external_strength_screen_20261018.common import digest
from tools.external_strength_screen_20261018_02 import bridge
from tools.move_evaluation_20261023.qualification import build_fixtures
from tools.splendorust_search_scale_20261024 import audit, common as c, statistics as stats
from tools.splendorust_search_scale_20261024.work_accounting import check_search_phase


def clock(status="ok", received=.2):
    return {"status": status, "accepted": status == "ok", "request_id": "request-00000001",
            "deadline_seconds": 1., "elapsed_seconds": max(received or 1., 1.2),
            "receipt_elapsed_seconds": received, "received_before_deadline": received is not None and 0 <= received <= 1.,
            "clock": "perf_counter_ns", "result": None, "detail": None, "telemetry": {}}


def synthetic_work(turn, quota, *, count, leaves, phase="Main", choices=2):
    view = {"turns": turn, "phase": {"kind": phase}}
    legal = [{"kind": "fake", "index": i} for i in range(choices)]
    heuristic = phase != "Main" or choices == 1 or turn >= 124
    row = {"simulations": 0 if heuristic else quota, "leaf_calls": leaves,
           "heuristic_phase": heuristic, "neural_callback_count": count,
           "neural_callback_rows": count,
           "neural_callbacks": [digest(["synthetic callback", i]) for i in range(count)]}
    return view, legal, row


@pytest.mark.parametrize("quota", [16, 32, 64])
def test_quota_specific_late_leaf_bounds(quota):
    check = lambda *args, **kw: check_search_phase(*synthetic_work(*args, quota=quota, **kw), quota=quota)
    assert check(123, count=1, leaves=quota+1)["heuristic_leaf_evaluations"] == quota
    assert check(124-quota, count=quota, leaves=quota+1)["heuristic_leaf_evaluations"] == 1
    assert check(123-quota, count=quota+1, leaves=quota+1)["heuristic_leaf_evaluations"] == 0
    for turn, count, leaves, match in ((123-quota, quota, quota+1, "unreachable"),
        (123, 2, quota+1, "exactly one"), (123, 0, quota+1, "bounds"),
        (123, 1, quota+2, "bounds"), (124, 0, 1, "zero")):
        with pytest.raises(ValueError, match=match):
            check(turn, count=count, leaves=leaves)
    for phase, choices in (("Main", 1), ("Return", 3), ("Payment", 2), ("Noble", 2)):
        assert check(0, count=0, leaves=0, phase=phase, choices=choices)["simulations"] == 0


@pytest.mark.parametrize("quota", [16, 32, 64])
@pytest.mark.parametrize("field,value", [("simulations", True), ("simulations", 15),
    ("leaf_calls", True), ("neural_callback_count", True), ("neural_callback_count", 2),
    ("neural_callback_rows", True), ("neural_callbacks", ["bad hash"]), ("neural_callbacks", [])])
def test_work_rejects_false_receipts(quota, field, value):
    view, choices, row = synthetic_work(123, quota, count=1, leaves=quota+1)
    row[field] = value
    with pytest.raises(ValueError):
        check_search_phase(view, choices, row, quota=quota)


@pytest.mark.parametrize("quota", [16, 32, 64])
def test_quota_cannot_be_self_declared_at_late_root(quota):
    view, choices, row = synthetic_work(123, quota, count=1, leaves=quota+1)
    other = 32 if quota != 32 else 64
    with pytest.raises(ValueError, match="quota"):
        check_search_phase(view, choices, row, quota=other)
    with pytest.raises(ValueError, match="quota"):
        check_search_phase(view, choices, row, quota=True)


@pytest.mark.parametrize("quota", [16, 32, 64])
def test_diagnostic_hashes_bind_to_host_callbacks(quota):
    view, choices, row = synthetic_work(120, quota, count=3, leaves=quota+1)
    row["inference"] = [{"input_sha256": digest(["in", i]), "output_sha256": digest(["out", i]),
                         "batch": 1} for i in range(3)]
    row["neural_callbacks"] = [digest([x["input_sha256"], x["output_sha256"], 1]) for x in row["inference"]]
    assert check_search_phase(view, choices, row, quota=quota)["inference_calls"] == 3
    row["inference"][2]["output_sha256"] = "0" * 64
    with pytest.raises(ValueError, match="digest mismatch"):
        check_search_phase(view, choices, row, quota=quota)


def transcript(quota, fixture=None):
    state = rules.CorrectedEnv(seed=84515214).initial_state() if fixture is None else rules.deserialize_snapshot(fixture)
    observer = rules.observation_payload(state)
    phases = []
    def select(view, choices):
        heuristic = view["phase"]["kind"] != "Main" or len(choices) == 1 or view["turns"] >= 124
        calls = 0 if heuristic else 1 if view["turns"] == 123 else quota+1
        leaves = 0 if heuristic else quota+1
        action = audit.safe_choices(view, choices)[0]
        phases.append({"action": action, "native_legal_count": len(choices),
                       "simulations": 0 if heuristic else quota, "leaf_calls": leaves,
                       "heuristic_phase": heuristic, "neural_callback_count": calls, "neural_callback_rows": calls,
                       "neural_callbacks": [digest(["synthetic", i]) for i in range(calls)]})
        return action
    mains = bridge.main_actions(observer)
    forced = mains == [{"kind": "Pass"}]
    chosen = mains[0] if forced else select(bridge.encode_observation(observer), mains)
    action = rules.serialize_action(bridge.resolve_compound(observer, chosen, select))
    calls = sum(p["neural_callback_count"] for p in phases)
    trace = {"opponent_id": "splendorust", "profile_id": c.profile_id(quota),
             "observer_seed": int(digest(observer)[:16], 16), "phases": phases,
             "inference_calls": calls, "inference_rows": calls, "forced_pass": forced}
    return state, observer, action, trace


def response(observer, action, trace, quota):
    legal = [rules.serialize_action(a) for a in rules.legal_actions(rules.state_from_observation(observer))]
    return {"player": "reference" if quota == 16 else "search", "action": action, "diagnostics": trace,
            "observer_sha256": digest(observer), "legal_actions_sha256": digest(legal), "legal_count": len(legal),
            "ruleset": rules.RULESET_ID, "catalogue_sha256": rules.CATALOGUE_SHA256,
            "turn_contract": audit.TURN_CONTRACT, "profile_id": c.profile_id(quota),
            "wrapper_id": c.wrapper_id(quota), "profile_sha256": c.profile_sha256(quota), "simulations_quota": quota}


@pytest.fixture(autouse=True)
def synthetic_profile_hash(monkeypatch):
    # Actual profile files are bound by launch; no preparation/output mutation here.
    monkeypatch.setattr(c, "profile_sha256", lambda q: digest(["synthetic profile", q]))


@pytest.mark.parametrize("quota", [16, 32, 64])
def test_response_identity_quota_and_independent_transition(quota):
    state, observer, action, trace = transcript(quota)
    arm = "reference" if quota == 16 else "search"
    before = rules.serialize_snapshot(state)
    after = rules.serialize_snapshot(rules.CorrectedEnv().step(state, rules.deserialize_action(action)))
    checked = audit.audit_decision(observer, action, trace, arm, before, after, expected_quota=quota)
    assert checked["transition_checked"] and checked["simulations"] == quota
    assert audit.validate_search_work(trace, observer, quota=quota)["simulations"] == quota
    original = response(observer, action, trace, quota)
    assert audit.validate_result(original, observer, arm, expected_quota=quota)
    for key, value in (("simulations_quota", True), ("simulations_quota", 64 if quota != 64 else 32),
        ("profile_id", "forged"), ("wrapper_id", "forged"), ("profile_sha256", "0"*64),
        ("legal_count", True), ("ruleset", "legacy"), ("deck_order", [])):
        bad = deepcopy(original); bad[key] = value
        with pytest.raises(ValueError):
            audit.validate_result(bad, observer, arm, expected_quota=quota)
    after["state"]["players"][0]["score"] += 1
    with pytest.raises(ValueError):
        audit.audit_decision(observer, action, trace, arm, before, after, expected_quota=quota)
    if arm == "search":
        with pytest.raises(ValueError, match="externally frozen"):
            audit.validate_result(original, observer, arm)
    else:
        assert audit.validate_result(original, observer, arm)
        with pytest.raises(ValueError, match="PUCT16"):
            audit.validate_result(original, observer, arm, expected_quota=32)


@pytest.mark.parametrize("quota", [16, 32, 64])
def test_fixed_rule_fixtures_and_search_trace_profile(quota):
    for fixture in build_fixtures():
        _, observer, action, trace = transcript(quota, fixture["snapshot"])
        arm = "reference" if quota == 16 else "search"
        checked = audit.audit_decision(observer, action, trace, arm, expected_quota=quota)
        assert checked["inference_calls"] <= checked["leaf_evaluations"] <= quota+1
        if fixture["id"] in ("turn124", "single_buy", "forced_pass"):
            assert checked["simulations"] == checked["inference_calls"] == checked["leaf_evaluations"] == 0
        bad = deepcopy(trace); bad["profile_id"] = "untrusted"
        with pytest.raises(ValueError, match="profile"):
            audit.validate_search_work(bad, observer, quota=quota)


def test_clock_charges_delivery_and_preserves_failures():
    assert audit.check_clock(clock()) == .2
    with pytest.raises(ValueError, match="late"):
        audit.check_clock(clock(received=1.000001))
    receipt = clock(); receipt["received_before_deadline"] = 1
    with pytest.raises(ValueError):
        audit.check_clock(receipt)
    receipt = clock("worker_error")
    receipt["detail"] = "TimeoutError: ONE_SECOND_INTERNAL_DEADLINE: complete response exceeded 0.900s"
    assert audit.classify_failure(receipt) == "worker_error"
    for detail in ("SEARCH_SCALE_ADAPTER: source changed", "SEARCH_SCALE_AUDIT: malformed trace", "unclassified failure"):
        receipt["detail"] = detail
        with pytest.raises(ValueError):
            audit.classify_failure(receipt)


def test_forfeit_requires_frozen_quota_and_full_clock():
    state = rules.CorrectedEnv(seed=84515214).initial_state()
    before = rules.serialize_snapshot(state)
    identity = {"challenger_simulations": 64}
    row = {**identity, "ply": 0, "seat": 0, "arm": "reference", "before": before,
           "observer": rules.observation_payload(state), "clock": clock("timeout", None),
           "completed": False, "action": None, "trace": None, "after": None}
    ending = {**identity, "reason": "forfeit", "winner": 1, "loser": 0, "failure": "timeout", "final": before}
    check = lambda: audit.audit_game(before, [row], ending, search_seat=1, expected_identity=identity)
    assert check()["failed_moves"] == 1
    ending["challenger_simulations"] = 32
    with pytest.raises(ValueError, match="quota"): check()
    ending["challenger_simulations"] = 64
    row["clock"]["deadline_seconds"] = .5
    with pytest.raises(ValueError, match="stage-clamped"): check()
    row["clock"] = clock("resource_guard", None)
    with pytest.raises(ValueError, match="infrastructure"): check()
    with pytest.raises(ValueError, match="externally frozen"):
        audit.audit_game(before, [row], ending, search_seat=1)


def test_replay_sample_independent_of_order_and_outcomes():
    observer = rules.observation_payload(rules.CorrectedEnv(seed=84515214).initial_state())
    rows = [{"pair_index": p, "game_index": 2*p, "ply": 2*p, "arm": "reference", "seat": 0,
             "observer": observer, "completed": True, "winner": p%2} for p in range(10)]
    original = audit.select_replay_sample(rows)
    changed = list(reversed(deepcopy(rows)))
    for row in changed: row["winner"] = None
    assert audit.select_replay_sample(changed) == original
    assert len(original["decisions"]) == 1 and len(original["empty_cells"]) == 47
    assert original["domain"] == "splendorust_puct_budget_1s_20261024_audit_v1"
    with pytest.raises(ValueError, match="duplicate"):
        audit.select_replay_sample(rows+rows[:1])


def games(outcome, quota=64):
    rows = []
    for pair in range(100):
        for game in (0, 1):
            seat = (pair+game)%2; halfpoints = outcome(pair, game, seat)
            winner = None if halfpoints == 1 else seat if halfpoints == 2 else 1-seat
            rows.append({"pair_index": pair, "game_index": 2*pair+game, "search_seat": seat,
                         "winner": winner, "reason": "natural", "stage": "main", "challenger_simulations": quota,
                         "seed_decimal": stats.expected_seed("main", pair)})
    return rows


def test_fixed_sample_stats_and_independent_bootstrap():
    rows = games(lambda p,g,s: (p+g)%3)
    result = stats.analyze_games(rows, True)
    assert result["challenger_simulations"] == 64 and result["reference_simulations"] == 16
    assert result["bootstrap"]["seed"] == 2026102402
    assert stats.verify_analysis(rows, result)["status"] == "passed"
    result["bootstrap"]["indices_sha256"] = "0"*64
    with pytest.raises(ValueError, match="stream"):
        stats.verify_analysis(rows, result)


def test_strength_thresholds_seat_gate_and_draws():
    assert stats.analyze_games(games(lambda *a: 2), True)["classification"] == "useful_evidence_for_more_search"
    assert stats.analyze_games(games(lambda *a: 0), True)["classification"] == "useful_evidence_for_reference_puct16"
    result = stats.analyze_games(games(lambda *a: 1), True)
    assert result["draws"] == 200 and result["classification"] == "inconclusive_or_small"
    result = stats.analyze_games(games(lambda p,g,s: 2 if s == 0 or p < 40 else 0), True)
    assert result["by_search_seat"] == {"0": 1., "1": .4} and result["classification"] == "inconclusive_or_small"


def test_fixed_schedule_seed_quota_and_incomplete_rejection():
    rows = games(lambda *a: 2)
    with pytest.raises(ValueError, match="gates"): stats.analyze_games(rows, False)
    with pytest.raises(ValueError, match="200"): stats.validated_games(rows[:-1])
    for field, value in (("game_index", True), ("search_seat", 1), ("stage", "operational"),
                         ("seed_decimal", "1"), ("winner", True), ("reason", "max_turns"),
                         ("challenger_simulations", 32), ("challenger_simulations", True)):
        bad=deepcopy(rows); bad[0][field]=value
        with pytest.raises(ValueError): stats.validated_games(bad)
    with pytest.raises(ValueError, match="duplicate"):
        stats.validated_games(rows[:-1]+rows[:1])
    expected = str(int.from_bytes(hashlib.sha256(json.dumps(
        ["splendorust_puct_budget_1s_20261024_v1", "main", 99], separators=(",", ":")).encode()).digest(), "big"))
    assert rows[-1]["seed_decimal"] == expected
    rows[0].update(reason="forfeit", loser=0, winner=1)
    assert stats.validated_games(rows)[0]["search_score_halfpoints"] == 0
    rows[0]["winner"] = 0
    with pytest.raises(ValueError, match="forfeit"): stats.validated_games(rows)


def test_performance_preserves_timeout_and_missing_latency():
    completed = clock()
    completed["telemetry"] = {"request_cpu_seconds": .1, "rss_bytes": 100, "peak_sampled_rss_bytes": 120}
    rows = [{"arm": "reference", "clock": completed}, {"arm": "search", "clock": clock("timeout", None)}]
    games = [{"reason": "natural", "plies": 58, "search_seat": 0},
             {"reason": "forfeit", "plies": 11, "search_seat": 1, "loser": 0, "winner": 1}]
    result = audit.summarise_performance(rows, games)
    assert result["natural_game_length_plies"]["mean"] == 58
    assert result["arms"]["reference"]["decision_cpu_seconds"]["mean"] == .1
    assert result["arms"]["search"]["completed_arrival_seconds"]["count"] == 0
    assert result["arms"]["search"]["missing_arrival_timestamps"] == 1
    assert len(result["forfeits"]) == 1


def test_independent_seeds_match_both_new_schedule_namespaces():
    for row in c.make_schedule():
        assert stats.expected_seed(row["stage"], row["pair_index"]) == row["seed_decimal"]
