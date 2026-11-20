"""Model-free corruption, replay, attribution and selection checks."""
from copy import deepcopy
import pytest

from tools.external_strength_screen_20261018 import rules
from tools.external_strength_screen_20261018.common import digest
from tools.external_strength_screen_20261018_02 import bridge
from tools.move_evaluation_20261023.qualification import build_fixtures
from tools.splendorust_search_scale_20261024 import common as scientific
from tools.splendorust_graph64_strength_20261025 import audit, schedule as frozen_schedule


def clock(status="ok", received=.2, detail=None):
    return {"status": status, "accepted": status == "ok", "request_id": "request-00000001",
            "deadline_seconds": 1., "elapsed_seconds": 1.02 if received is None else received+.01,
            "receipt_elapsed_seconds": received, "received_before_deadline": received is not None and 0 <= received <= 1.,
            "clock": "perf_counter_ns", "result": None, "detail": detail, "telemetry": {}}


def spec(challenger_seat=0):
    return frozen_schedule.stage_schedule("main")[challenger_seat]


def trace_for(state, quota):
    observer, phases = rules.observation_payload(state), []
    def select(view, choices):
        heuristic = view["phase"]["kind"] != "Main" or len(choices) == 1 or view["turns"] >= 124
        calls = 0 if heuristic else 1 if view["turns"] == 123 else quota+1
        selected = audit.original.safe_choices(view, choices)[0]
        phases.append(dict(action=selected, native_legal_count=len(choices), simulations=0 if heuristic else quota,
                           leaf_calls=0 if heuristic else quota+1, heuristic_phase=heuristic,
                           neural_callback_count=calls, neural_callback_rows=calls,
                           neural_callbacks=[digest(["synthetic test callback", index]) for index in range(calls)]))
        return selected
    mains = bridge.main_actions(observer)
    forced = mains == [{"kind": "Pass"}]
    chosen = mains[0] if forced else select(bridge.encode_observation(observer), mains)
    action = rules.serialize_action(bridge.resolve_compound(observer, chosen, select))
    calls = sum(phase["neural_callback_count"] for phase in phases)
    trace = dict(opponent_id="splendorust", profile_id=scientific.profile_id(quota),
                 observer_seed=int(digest(observer)[:16], 16), phases=phases,
                 inference_calls=calls, inference_rows=calls, forced_pass=forced)
    legal = [rules.serialize_action(item) for item in rules.legal_actions(state)]
    response = dict(player="search", action=action, diagnostics=trace, observer_sha256=digest(observer),
                    legal_actions_sha256=digest(legal), legal_count=len(legal), ruleset=rules.RULESET_ID,
                    catalogue_sha256=rules.CATALOGUE_SHA256, turn_contract=audit.original.TURN_CONTRACT,
                    profile_id=scientific.profile_id(quota), wrapper_id=scientific.wrapper_id(quota),
                    profile_sha256=scientific.profile_sha256(quota), simulations_quota=quota)
    return observer, action, trace, response


def natural_game(challenger_seat=0):
    snapshot = next(item["snapshot"] for item in build_fixtures() if item["id"] == "noble_final_round")
    state = rules.deserialize_snapshot(snapshot)
    schedule, rows = spec(challenger_seat), []
    for ply in range(4):
        if state.terminal: break
        role = "challenger" if state.current_player == challenger_seat else "opponent"
        quota = audit.QUOTAS[role]
        observer, action, trace, response = trace_for(state, quota)
        before = rules.serialize_snapshot(state)
        state = rules.CorrectedEnv().step(state, rules.deserialize_action(action))
        receipt = clock(); receipt["result"] = response
        rows.append(dict(schedule, ply=ply, seat=before["state"]["current_player"],
                         player_role=role, arm="search", mode=audit.MODES[role], quota=quota, completed=True,
                         observer=observer, before=before, after=rules.serialize_snapshot(state),
                         action=action, trace=trace, clock=receipt))
    assert state.terminal
    ending = dict(schedule, complete=True, plies=len(rows), reason="natural", winner=state.winner,
                  loser=None, final=rules.serialize_snapshot(state))
    return snapshot, rows, ending, schedule


@pytest.mark.parametrize("seat", [0, 1])
def test_both_roles_are_search_at_exact64_and32_with_real_ledger(seat):
    initial, rows, ending, schedule = natural_game(seat)
    result = audit.audit_game(initial, rows, ending, schedule)
    assert result["transitions"] == len(rows) and result["failed_moves"] == 0
    assert result["scientific_arms"] == {"challenger": {"arm": "search", "quota": 64, "mode": "candidate"}, "opponent": {"arm": "search", "quota": 32, "mode": "baseline"}}


@pytest.mark.parametrize("field,value", [("arm", "reference"), ("player_role", "opponent"), ("quota", 32),
                                        ("quota", True), ("mode", "baseline"), ("after", None)])
def test_wrong_role_quota_parent_or_missing_transition_is_rejected(field, value):
    initial, rows, ending, schedule = natural_game()
    rows[0][field] = value
    with pytest.raises(ValueError): audit.audit_game(initial, rows, ending, schedule)


def forfeit_game(role="challenger", status="timeout", received=None, detail=None):
    state = rules.CorrectedEnv(seed=84515214).initial_state()
    initial, schedule = rules.serialize_snapshot(state), spec(0 if role == "challenger" else 1)
    row = dict(schedule, ply=0, seat=0, player_role=role,
               arm="search", mode=audit.MODES[role], quota=audit.QUOTAS[role], completed=False, before=initial, after=None,
               observer=rules.observation_payload(state), action=None, trace=None, clock=clock(status, received, detail))
    ending = dict(schedule, complete=True, plies=1, reason="forfeit", winner=1, loser=0,
                  failed_role=role, failed_seat=0, failure=status, final=initial)
    return initial, [row], ending, schedule


def test_original_timeout_loss_policy_and_infrastructure_stop():
    initial, rows, ending, schedule = forfeit_game()
    assert audit.audit_game(initial, rows, ending, schedule)["failed_moves"] == 1
    rows[0]["clock"]["deadline_seconds"] = .5
    with pytest.raises(ValueError, match="stage clamp"): audit.audit_game(initial, rows, ending, schedule)
    rows[0]["clock"] = clock("resource_guard", None)
    with pytest.raises(ValueError, match="infrastructure"): audit.audit_game(initial, rows, ending, schedule)


@pytest.mark.parametrize("field,value", [("winner", 0), ("loser", 1), ("failed_role", "opponent"),
                                        ("failed_seat", 1), ("complete", False), ("plies", 2)])
def test_forfeit_attribution_and_inventory_are_mandatory(field, value):
    initial, rows, ending, schedule = forfeit_game()
    ending[field] = value
    with pytest.raises(ValueError): audit.audit_game(initial, rows, ending, schedule)


def test_trace_coupling_and_independent_terminal_result_are_checked():
    initial, rows, ending, schedule = natural_game()
    changed = deepcopy(rows)
    changed[0]["trace"]["inference_calls"] += 1
    with pytest.raises(ValueError): audit.audit_game(initial, changed, ending, schedule)
    ending["winner"] = 1-ending["winner"]
    with pytest.raises(ValueError): audit.audit_game(initial, rows, ending, schedule)


def test_attributable_malformed_response_retains_original_classifier():
    initial, rows, ending, schedule = forfeit_game()
    rows[0]["clock"] = clock("malformed", .2)
    ending["failure"] = "malformed"
    assert audit.audit_game(initial, rows, ending, schedule)["failed_moves"] == 1


def selection_rows():
    observer = rules.observation_payload(rules.CorrectedEnv(seed=84515214).initial_state())
    return [dict(stage="main", pair_index=pair, game_index=2*pair, ply=2*pair,
                 player_role="challenger", arm="search", quota=64, mode="candidate", seat=0,
                 observer=observer, completed=True, winner=pair%2, timing=pair*.1) for pair in range(10)]


def test_replay_selection_is_order_outcome_timing_blind_and_distinct():
    rows = selection_rows()
    expected = audit.select_replay_sample(rows)
    altered = list(reversed(deepcopy(rows)))
    for row in altered: row.update(winner=None, timing=999)
    assert audit.select_replay_sample(altered) == expected
    assert len(expected["decisions"]) == 1 and len(expected["empty_cells"]) == 47
    assert expected["decisions"][0]["quota"] == 64 and expected["decisions"][0]["stage"] == "main"
    assert expected["domain"] != audit.original.AUDIT_DOMAIN
    with pytest.raises(ValueError, match="duplicate"): audit.select_replay_sample(rows+rows[:1])


def test_performance_retains_timeout_missing_arrival_and_exact_role():
    good = clock()
    good["telemetry"] = dict(request_cpu_seconds=.1, rss_bytes=100, peak_sampled_rss_bytes=120)
    rows = [dict(stage="main", player_role="challenger", arm="search", mode="candidate", quota=64, clock=good),
            dict(stage="main", player_role="opponent", arm="search", mode="baseline", quota=32, clock=clock("timeout", None))]
    result = audit.performance(rows)
    assert result["players"]["challenger"]["decision_cpu_seconds"]["mean"] == .1
    assert result["players"]["opponent"]["missing_arrival_timestamps"] == 1
    assert result["players"]["opponent"]["completed_arrival_seconds"]["count"] == 0
    assert result["players"]["opponent"]["statuses"] == {"timeout": 1}



@pytest.mark.parametrize("role", ["challenger", "opponent"])
@pytest.mark.parametrize("status,received,detail", [
    ("timeout", None, "No response before the full one-second deadline"),
    ("timeout", 1.01, "Response arrived after the full one-second deadline"),
    ("worker_error", .91, "RuntimeError: ONE_SECOND_INTERNAL_DEADLINE exhausted"),
    ("worker_error", .2, "RuntimeError: PLAYER_ILLEGAL_ACTION"),
    ("malformed", .2, "Invalid JSON response"),
    ("worker_exit", None, "Worker exited before response"),
])
def test_faithful_attributable_failure_receipts_are_losses_for_exact_runtime(role, status, received, detail):
    initial, rows, ending, schedule = forfeit_game(role, status, received, detail)
    result = audit.audit_game(initial, rows, ending, schedule)
    assert result["failed_moves"] == 1 and result["transitions"] == 0
    assert ending["loser"] == 0 and ending["winner"] == 1
    assert rows[0]["mode"] == audit.MODES[role]


@pytest.mark.parametrize("status,detail", [
    ("resource_guard", "Combined RSS exceeded"),
    ("worker_error", "source changed"),
    ("worker_error", "ruleset mismatch"),
    ("worker_error", "catalogue mismatch"),
    ("worker_error", "resource shortage"),
    ("worker_error", "unexpected exception"),
    ("malformed", "adapter_prerequisite mismatch"),
])
def test_infrastructure_source_resource_and_unclassified_failures_stop_incomplete(status, detail):
    initial, rows, ending, schedule = forfeit_game("challenger", status, None, detail)
    with pytest.raises(ValueError): audit.audit_game(initial, rows, ending, schedule)


@pytest.mark.parametrize("role,wrong_mode", [("challenger", "baseline"), ("opponent", "candidate")])
def test_mode_switch_forbidden_in_game_replay_sample_and_performance(role, wrong_mode):
    initial, rows, ending, schedule = forfeit_game(role)
    rows[0]["mode"] = wrong_mode
    with pytest.raises(ValueError): audit.audit_game(initial, rows, ending, schedule)
    completed = {**rows[0], "completed": True}
    with pytest.raises(ValueError): audit.select_replay_sample([completed])
    with pytest.raises(ValueError): audit.performance(rows)


def test_reproduction_selection_retains_eager32_mode():
    observer = rules.observation_payload(rules.CorrectedEnv(seed=84515214).initial_state())
    row = dict(stage="main",pair_index=0,game_index=1,ply=0,seat=0,player_role="opponent",
               arm="search",mode="baseline",quota=32,observer=observer,completed=True)
    chosen = audit.select_replay_sample([row])["decisions"][0]
    assert chosen["mode"] == "baseline" and chosen["quota"] == 32 and chosen["player_role"] == "opponent"


def test_missing_complete_flag_and_changed_mode_schedule_are_rejected():
    initial, rows, ending, schedule = natural_game()
    del ending["complete"]
    with pytest.raises(ValueError, match="complete game"): audit.audit_game(initial, rows, ending, schedule)
    initial, rows, ending, schedule = natural_game()
    schedule["opponent_mode"] = "candidate"
    with pytest.raises(ValueError): audit.audit_game(initial, rows, ending, schedule)


def test_failure_classifier_is_same_original_function_without_wrapper_or_policy_changes():
    assert audit.classify_failure is audit.original.classify_failure
    assert audit.check_clock is audit.original.check_clock
