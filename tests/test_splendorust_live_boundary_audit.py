"""Model-free corruption, replay, attribution and selection checks."""
from copy import deepcopy
import pytest

from tools.external_strength_screen_20261018 import rules
from tools.external_strength_screen_20261018.common import digest
from tools.external_strength_screen_20261018_02 import bridge
from tools.move_evaluation_20261023.qualification import build_fixtures
from tools.splendorust_search_scale_20261024 import common as scientific
from tools.splendorust_live_boundary_20261025 import audit, schedule as frozen_schedule


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
    for row in rows: attach_boundaries(row)
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
    attach_boundaries(row)
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


def attach_boundaries(row):
    """Synthetic transport evidence paired with independently replayed science."""
    row.update(observation_mode="profiled", parent_profile=None, child_profile=None)
    if not row["completed"]:
        return row
    a = audit.boundaries
    quota = row["quota"]
    parent = dict(schema=a.PARENT_SCHEMA, request_id=row["clock"]["request_id"], pid=10, child_pid=42,
        clock="perf_counter_ns", clock_implementation="QueryPerformanceCounter()", request_start_ns=1_000_000_000,
        receipt_arrival_ns=1_200_000_000, receipt_reconstructed_arrival_ns=1_200_000_000, capture_snapshot_ns=1_210_000_000,
        raw_arrivals_ns=[1_200_000_000], maximum_arrival_events=16, dropped_arrival_events=0,
        request_capture_maximum_bytes=1024*1024, request_raw_complete=True, request_raw_overflow=False,
        request_raw_bytes=100, request_raw_sha256="0"*64,
        response_raw_sha256="1"*64, response_raw_bytes=100, response_capture_overflow=False,
        response_capture_maximum_bytes=4*1024*1024,
        stdin_write_start_ns=1_010_000_000, stdin_write_end_ns=1_018_000_000, stdin_flush_start_ns=1_018_000_000,
        stdin_flush_end_ns=1_030_000_000, dispatch_complete_ns=1_030_000_000, dispatch_write_calls=1, dispatch_flush_calls=1,
        dispatch_write_failures=0, dispatch_flush_failures=0, pending_writes=0, pending_flushes=0,
        times_are_diagnostic_not_trusted_deadline_receipts=True, measurement_units="nanoseconds",
        raw_reader_semantics="unchanged raw FileIO.readline then immediate trusted perf_counter_ns",
        child_read_may_precede_dispatch_completion=True, child_post_flush_may_follow_trusted_arrival=True,
        hashes_calculated_after_original_request_returns=True, dispatch_markers_may_be_partial_if_pending=True)
    inner = dict(schema=a.CHILD_SCHEMA, pid=42, clock="perf_counter_ns", measurement_units="nanoseconds",
        request_read_complete_ns=1_020_000_000, handle_start_ns=1_025_000_000, handle_end_ns=1_150_000_000,
        handle_failed=False, times_are_diagnostic_not_trusted_deadline_receipts=True)
    child = dict(schema=a.CHILD_SCHEMA, request_id=row["clock"]["request_id"], pid=42, quota=quota,
        backend="qualified_graph64" if quota == 64 else "unchanged_eager32",
        graph_identity_sha256=a.GRAPH_IDENTITY if quota == 64 else None,
        clock="perf_counter_ns", clock_implementation="QueryPerformanceCounter()",
        request_read_complete_ns=1_020_000_000, request_started_seconds=1.019,
        pre_write_ns=1_160_000_000, post_flush_ns=1_190_000_000, outcome="response_flushed", error=None,
        profiles=[inner], request_raw_sha256="0"*64,
        payload_sha256=a._digest({"player": "search", "observer": row["observer"]}),
        result_sha256=a._digest(row["clock"]["result"]), serialized_packet_utf8_sha256="1"*64,
        serialized_packet_utf8_bytes=100, times_are_diagnostic_not_trusted_deadline_receipts=True,
        success_sidecar_written_after_response_flush=True, post_flush_may_follow_parent_receipt=True,
        child_read_may_precede_parent_dispatch_flush_completion=True, killed_worker_sidecar_may_be_missing=True,
        text_transport_may_expand_newlines=True)
    row["clock"]["telemetry"] = {"process_tree": [{"role": "python", "pid": 42}]}
    row.update(parent_profile=parent, child_profile=child)
    return row


def natural_prefix(count=206):
    return [dict(spec,complete=True,plies=2,reason="natural",winner=None,loser=None)
            for spec in frozen_schedule.make_schedule()[:count]]


def stopped_prefix(count=1, *, status="timeout", detail=None):
    games = natural_prefix(count)
    game = games[-1]
    role = "challenger" if game["challenger_seat"] == 0 else "opponent"
    game.update(reason="forfeit",winner=1,loser=0,failed_role=role,failed_seat=0,failure=status)
    receipt = clock(status,None,detail)
    stop = {key: game[key] for key in ("stage", "pair_index", "game_index")}
    stop.update(ply=1,player_role=role,seat=0,clock_status=status,failure_classification=status,
                request_id=receipt["request_id"],clock=receipt)
    return games, stop


def test_all206_naturals_required_for_full_reliability_workload():
    result = audit.validate_stopped_prefix(frozen_schedule.make_schedule(),natural_prefix())
    assert result["status"] == "full_workload_verified" and result["complete_planned_workload"]
    assert result["verified_games"] == 206 and result["unplayed_games"] == 0 and result["no_strength_claim"]
    with pytest.raises(ValueError,match="all206"):
        audit.validate_stopped_prefix(frozen_schedule.make_schedule(),natural_prefix(8))


@pytest.mark.parametrize("count", [1,6,7,206])
def test_first_fault_verified_prefix_is_not_full_sample(count):
    games, stop = stopped_prefix(count)
    result = audit.validate_stopped_prefix(frozen_schedule.make_schedule(),games,stop)
    assert result["status"] == "captured_fault_verified_prefix" and result["stopped_prefix_is_not_full_sample"]
    assert not result["complete_planned_workload"] and result["verified_games"] == count
    assert result["failure"]["kind"] == "deadline_failure"
    assert result["optional_stopping_confidence_interval"] is None


@pytest.mark.parametrize("mutation", [
    lambda g,s:g.reverse(), lambda g,s:g[0].update(reason="forfeit"),
    lambda g,s:s.update(ply=0), lambda g,s:s.update(request_id="wrong"),
    lambda g,s:s.update(player_role="bad"), lambda g,s:s.update(clock_status="worker_exit"),
    lambda g,s:s.update(failure_classification="malformed"), lambda g,s:g[-1].update(winner=0),
    lambda g,s:s["clock"].update(deadline_seconds=.5),
])
def test_fault_prefix_rejects_moved_omitted_or_reclassified_evidence(mutation):
    games,stop=stopped_prefix(3);mutation(games,stop)
    with pytest.raises(ValueError):
        audit.validate_stopped_prefix(frozen_schedule.make_schedule(),games,stop)


@pytest.mark.parametrize("status,detail", [("resource_guard","RAM"),("worker_error","source changed"),
    ("worker_error","unexpected exception"),("malformed","adapter_prerequisite mismatch")])
def test_infrastructure_never_becomes_successful_fault_capture(status,detail):
    games,stop=stopped_prefix(2,status=status,detail=detail)
    with pytest.raises(ValueError):
        audit.validate_stopped_prefix(frozen_schedule.make_schedule(),games,stop)


def test_non_timing_attributable_fault_is_labelled_accurately():
    games,stop=stopped_prefix(1,status="worker_error",detail="PLAYER_ILLEGAL_ACTION")
    result=audit.validate_stopped_prefix(frozen_schedule.make_schedule(),games,stop)
    assert result["failure"]["kind"] == "attributable_player_protocol_or_legality_failure"


def test_partial_prefix_selector_admits_ops_without_game_id_collision():
    rows=selection_rows()[:2]
    operations=deepcopy(rows)
    for row in operations:row["stage"]="operational"
    selected=audit.select_replay_sample(rows+operations,allow_operational=True)
    assert selected["all_available_cells_selected"] and len(selected["decisions"])==1
    assert selected["scope"]=="all_completed_played_prefix_cells"
    with pytest.raises(ValueError,match="stage"):
        audit.select_replay_sample(operations)
    for row in rows+operations:row["completed"]=False
    assert audit.select_replay_sample(rows+operations,allow_operational=True)["decisions"]==[]


def test_boundary_projection_cannot_change_scientific_modes_or_source_row():
    _,rows,_,_=natural_game();row=rows[0];before=deepcopy(row)
    projected=audit.boundary_row(row)
    assert projected["mode"]=="profiled" and row==before and row["mode"]=="candidate"
    assert projected["clock"] is row["clock"]
    row["observation_mode"]="baseline"
    with pytest.raises(ValueError,match="observer required"):
        audit.validate_boundary_row(row)


def test_boundary_failure_missing_sidecar_is_partial_but_corruption_is_error():
    _,rows,_,_=forfeit_game()
    result=audit.summarize_boundary_rows(rows)
    assert result["observed_faults"]==1 and not result["all_collected_decisions_completed"]
    assert result["failures"][0]["boundary_evidence"]["missing_child_profile"]
    rows[0]["profile_capture_error"]="disk write failed"
    with pytest.raises(ValueError,match="capture failure"):
        audit.summarize_boundary_rows(rows)


def test_boundary_summary_uses_actual_stages_roles_and_no_strength_claims():
    _,rows,_,_=natural_game()
    result=audit.summarize_boundary_rows(rows)
    assert result["rows"]==len(rows) and result["no_strength_claim"]
    assert result["deadline_adjustment_seconds"]==0 and not result["complete_planned_collection_inferred"]
    for group in result["groups"]:
        assert group["stage"]=="main" and group["mode"]==audit.MODES[group["player_role"]]
        assert group["completed_boundary_metrics"]["child.handle_wall_ns"]["distribution"]["mean"]==125_000_000
    rows[0]["child_profile"]["result_sha256"]="f"*64
    with pytest.raises(ValueError,match="result binding"):
        audit.summarize_boundary_rows(rows)
