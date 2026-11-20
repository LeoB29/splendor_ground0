"""Model-free corruption, branch, independent ledger and sampling regressions."""
from copy import deepcopy
import pytest
from tools.external_strength_screen_20261018 import rules
from tools.external_strength_screen_20261018.common import digest
from tools.external_strength_screen_20261018_02 import bridge
from tools.move_evaluation_20261023.greedy_adapter import NativeOnly
from tools.move_evaluation_20261023.qualification import build_fixtures
from tools.splendorust_one_second_v3_20261024 import audit


def clock(status="ok", received=.2):
    return {"status": status, "accepted": status == "ok", "request_id": "request-00000001",
            "deadline_seconds": 1., "elapsed_seconds": max(received or 1., 1.2),
            "receipt_elapsed_seconds": received, "received_before_deadline": received is not None and 0 <= received <= 1.,
            "clock": "perf_counter_ns", "result": None, "detail": None, "telemetry": {}}


@pytest.fixture
def transcript(tmp_path):
    """Synthetic zero logits, authenticated native mapping; no model inference."""
    state = rules.CorrectedEnv(seed=84515214).initial_state()
    observer = rules.observation_payload(state)
    native = NativeOnly(tmp_path / "native")
    phases, raw = [], [0.] * 83
    def select(view, choices):
        request = {"observation": view, "legal": choices, "seed": int(digest(observer)[:16], 16)}
        if view["phase"]["kind"] == "Main":
            result = native.exchange({**request, "op": "reference", "policy81": raw[:81]})
            logits, branch = result["canonical_logits"], "neural_greedy"
            inference = [{"input_sha256": "0"*64, "output_sha256": digest(raw), "output83": raw}]
        else:
            result = native.exchange({**request, "op": "greedy"})
            logits, branch, inference = None, "auxiliary_heuristic", []
        phases.append({"phase": view["phase"]["kind"], "observation_sha256": digest(view),
                       "legal_sha256": digest(choices), "branch": branch, "native_legal_count": len(choices),
                       "retained_indices": [choices.index(a) for a in result["filtered"]],
                       "canonical_indices": result["canonical_indices"], "selected_index": result["selected_index"],
                       "action": result["action"], "policy_logits_sha256": digest(logits) if logits is not None else None,
                       "inference": inference, "canonical_logits": logits})
        return result["action"]
    try:
        selected = select(bridge.encode_observation(observer), bridge.main_actions(observer))
        action = rules.serialize_action(bridge.resolve_compound(observer, selected, select))
    finally:
        native.close()
    trace = {"schema": "entity_greedy_decision_v1", "profile_id": audit.GREEDY_PROFILE,
             "observer_sha256": digest(observer), "observer_seed": int(digest(observer)[:16], 16),
             "action_sha256": digest(action), "forced_pass": False, "phases": phases,
             "inference_calls": 1, "simulations": 0}
    return state, observer, action, trace


def response(observer, action, trace):
    legal = [rules.serialize_action(a) for a in rules.legal_actions(rules.state_from_observation(observer))]
    return {"player": "greedy", "action": action, "diagnostics": trace, "observer_sha256": digest(observer),
            "legal_actions_sha256": digest(legal), "legal_count": len(legal), "ruleset": rules.RULESET_ID,
            "catalogue_sha256": rules.CATALOGUE_SHA256, "turn_contract": audit.TURN_CONTRACT,
            "profile_id": audit.GREEDY_PROFILE, "wrapper_id": audit.GREEDY_WRAPPER, "profile_sha256": None}


def test_raw_to_canonical_permutation_and_float32_rounding():
    out = audit.canonical_logits(list(range(81)))
    assert out[52:64] == list(range(12)) and out[37:49] == list(range(12, 24))
    assert out[49:52] == [24, 25, 26] and out[64:67] == [27, 28, 29]
    assert out[32:37] == list(range(55, 60))
    assert out[1] == 30 and out[2] == 31 and out[3] == 35 and out[7] == 45
    assert out[0] == -1e8 and out[31] == -1e8
    assert audit._f32([.1]) == audit._f32([.10000000149011612])


def test_independent_safe_filter_matches_native_contract(tmp_path):
    native = NativeOnly(tmp_path / "native")
    try:
        count = 0
        for fixture in build_fixtures():
            observer = rules.observation_payload(rules.deserialize_snapshot(fixture["snapshot"]))
            choices = bridge.main_actions(observer)
            if choices == [{"kind": "Pass"}]:
                continue
            result = native.inspect(observer)
            assert audit.safe_choices(bridge.encode_observation(observer), choices) == result["filtered"]
            count += 1
        assert count == 14
    finally:
        native.close()


def test_greedy_tie_ledger_and_corruption(transcript):
    state, observer, action, trace = transcript
    before = rules.serialize_snapshot(state)
    after = rules.serialize_snapshot(rules.CorrectedEnv().step(state, rules.deserialize_action(action)))
    assert trace["phases"][0]["selected_index"] == 0
    assert audit.audit_decision(observer, action, trace, "greedy", before, after)["status"] == "passed"
    assert audit.validate_result(response(observer, action, trace), observer, "greedy")
    bad = deepcopy(after)
    bad["state"]["players"][0]["score"] += 1
    with pytest.raises(ValueError):
        audit.audit_decision(observer, action, trace, "greedy", before, bad)
    bad = deepcopy(trace)
    bad["phases"][0]["retained_indices"].pop()
    with pytest.raises(ValueError, match="safe support"):
        audit.audit_decision(observer, action, bad, "greedy")
    bad = deepcopy(trace)
    bad["phases"][0]["canonical_indices"][0] += 1
    with pytest.raises(ValueError, match="canonical legal"):
        audit.audit_decision(observer, action, bad, "greedy")
    bad = deepcopy(trace)
    bad["phases"][0]["canonical_logits"][0] = 0.
    bad["phases"][0]["policy_logits_sha256"] = digest(bad["phases"][0]["canonical_logits"])
    with pytest.raises(ValueError, match="raw/canonical"):
        audit.audit_decision(observer, action, bad, "greedy")
    bad = deepcopy(trace)
    phase = bad["phases"][0]
    phase["selected_index"] = 1
    phase["action"] = bridge.main_actions(observer)[phase["retained_indices"][1]]
    with pytest.raises(ValueError, match="first maximum"):
        audit.audit_decision(observer, action, bad, "greedy")


def test_envelope_private_field_profile_and_bool_rejections(transcript):
    _, observer, action, trace = transcript
    original = response(observer, action, trace)
    for key, value in [("deck_order", []), ("profile_id", audit.SEARCH_PROFILE), ("legal_count", True),
                       ("ruleset", "legacy")]:
        bad = deepcopy(original)
        bad[key] = value
        with pytest.raises(ValueError):
            audit.validate_result(bad, observer, "greedy")


def test_puct_simulations_and_neural_calls_can_differ(transcript):
    _, observer, action, greedy = transcript
    phases = [{"action": p["action"], "native_legal_count": p["native_legal_count"],
               "simulations": 16 if p["phase"] == "Main" else 0,
               "leaf_calls": 12 if p["phase"] == "Main" else 0,
               "neural_callbacks": [digest(['synthetic', i]) for i in range(12)] if p["phase"] == "Main" else [],
               "neural_callback_count": 12 if p["phase"] == "Main" else 0,
               "neural_callback_rows": 12 if p["phase"] == "Main" else 0,
               "heuristic_phase": p["phase"] != "Main"} for p in greedy["phases"]]
    trace = {"opponent_id": "splendorust", "profile_id": audit.SEARCH_PROFILE,
             "observer_seed": greedy["observer_seed"], "phases": phases,
             "inference_calls": 12, "inference_rows": 12, "forced_pass": False}
    assert audit.audit_decision(observer, action, trace, "search")["simulations"] == 16
    trace["phases"][0]["simulations"] = 0
    with pytest.raises(ValueError, match="quota"):
        audit.audit_decision(observer, action, trace, "search")


def _synthetic_work(turn, *, count, leaves, phase='Main', choices=2):
    view = {'turns': turn, 'phase': {'kind': phase}}
    legal = [{'kind': 'fake', 'index': i} for i in range(choices)]
    is_heuristic = phase != 'Main' or choices == 1 or turn >= 124
    row = {'simulations': 0 if is_heuristic else 16, 'leaf_calls': leaves,
           'heuristic_phase': is_heuristic, 'neural_callback_count': count,
           'neural_callback_rows': count,
           'neural_callbacks': [digest(['synthetic callback', i]) for i in range(count)]}
    return view, legal, row


def test_source_branch_accounting_accepts_late_hypothetical_heuristics():
    from tools.splendorust_one_second_v3_20261024.work_accounting import check_search_phase
    assert check_search_phase(*_synthetic_work(123, count=1, leaves=17))['heuristic_leaf_evaluations'] == 16
    assert check_search_phase(*_synthetic_work(122, count=5, leaves=17))['heuristic_leaf_evaluations'] == 12
    assert check_search_phase(*_synthetic_work(108, count=16, leaves=17))['heuristic_leaf_evaluations'] == 1
    assert check_search_phase(*_synthetic_work(107, count=17, leaves=17))['heuristic_leaf_evaluations'] == 0
    for turn, phase, choices in ((124, 'Main', 2), (0, 'Payment', 3), (0, 'Return', 4),
                                 (0, 'Noble', 2), (0, 'Main', 1)):
        assert check_search_phase(*_synthetic_work(turn, count=0, leaves=0, phase=phase, choices=choices))['simulations'] == 0


@pytest.mark.parametrize('turn,count,leaves,match', [
    (0, 16, 17, 'unreachable'), (107, 16, 17, 'unreachable'),
    (123, 2, 17, 'exactly one'), (123, 0, 17, 'bounds'),
    (123, 2, 1, 'bounds'), (123, 1, 18, 'bounds'), (124, 0, 1, 'zero'),
])
def test_rejects_false_callback_or_leaf_counts(turn, count, leaves, match):
    from tools.splendorust_one_second_v3_20261024.work_accounting import check_search_phase
    with pytest.raises(ValueError, match=match):
        check_search_phase(*_synthetic_work(turn, count=count, leaves=leaves))


@pytest.mark.parametrize('field,value', [
    ('simulations', 15), ('simulations', True), ('leaf_calls', True),
    ('neural_callback_count', True), ('neural_callback_count', 2),
    ('neural_callback_rows', 2), ('neural_callback_rows', True),
    ('neural_callbacks', ['not a hash']), ('neural_callbacks', []),
])
def test_corrupt_receipts_still_fail(field, value):
    from tools.splendorust_one_second_v3_20261024.work_accounting import check_search_phase
    view, choices, row = _synthetic_work(123, count=1, leaves=17)
    row[field] = value
    with pytest.raises(ValueError):
        check_search_phase(view, choices, row)


def test_all_optional_diagnostic_callback_summaries_bind_to_receipts():
    from tools.splendorust_one_second_v3_20261024.work_accounting import check_search_phase
    view, choices, row = _synthetic_work(120, count=3, leaves=17)
    row['inference'] = [{'input_sha256': digest(['input', i]), 'output_sha256': digest(['output', i]),
                         'batch': 1} for i in range(3)]
    row['neural_callbacks'] = [digest([item['input_sha256'], item['output_sha256'], 1]) for item in row['inference']]
    assert check_search_phase(view, choices, row)['inference_calls'] == 3
    row['inference'][2]['output_sha256'] = '0' * 64
    with pytest.raises(ValueError, match='digest mismatch'):
        check_search_phase(view, choices, row)


@pytest.fixture
def fixed_output_native_search(monkeypatch):
    """Exact pinned binary/new wrapper, deterministic constants instead of a model.

    This runs no checkpoint, training or study games. A ten-second owned-child
    timer bounds a broken native transport even if a blocking read would hang.
    """
    import hashlib
    import json
    from pathlib import Path
    import subprocess
    import threading
    from tools.future_training_rules_20261021 import verify_runtime_sources
    from tools.splendorust_one_second_v3_20261024 import adapters
    verify_runtime_sources()
    root = Path(__file__).resolve().parents[1]
    profile = json.loads(adapters.PROFILE_PATH.read_text(encoding='utf-8'))
    executable = root / profile['native_binary']
    assert hashlib.sha256(executable.read_bytes()).hexdigest() == adapters.SEARCH_BINARY_SHA256
    opponent = object.__new__(adapters.SearchOpponent)
    opponent.profile = profile
    opponent.capture, opponent.callbacks = False, []
    opponent.deadline = None
    opponent.proc = subprocess.Popen([str(executable)], cwd=root, stdin=subprocess.PIPE,
        stdout=subprocess.PIPE, stderr=subprocess.DEVNULL, text=True, encoding='utf-8', bufsize=1,
        creationflags=getattr(subprocess, 'CREATE_NO_WINDOW', 0))
    timer = threading.Timer(10, lambda: opponent.proc.kill() if opponent.proc.poll() is None else None)
    timer.daemon = True
    timer.start()
    try:
        ready = json.loads(opponent.proc.stdout.readline())
        assert ready == {'kind': 'ready', 'bridge': 'observer_only_fixed_work_v1',
                         'source_pin': '1828507b8d2f0e02bc03d102b9c6d29a54fa8d57'}
        def constants(self, tokens, batch):
            assert batch == 1 and len(tokens) == 31 * 48
            return [[0.] * 83]
        monkeypatch.setattr(adapters.Opponent, '_infer', constants)
        yield opponent
    finally:
        timer.cancel()
        if opponent.proc.poll() is None:
            opponent.proc.terminate()
        opponent.proc.wait(timeout=2)
        opponent.proc.stdin.close()
        opponent.proc.stdout.close()


def test_actual_frozen_native_turn123_regression_and_exception_branches(fixed_output_native_search):
    """The old equality fails on a real native continuation, with no model load."""
    from tools.splendorust_one_second_20261024 import audit as failed_attempt_audit
    native = fixed_output_native_search
    fixtures = {row['id']: row for row in build_fixtures()}
    recorded = {}
    for identity in ('turn123', 'turn124', 'single_buy', 'forced_pass', 'private_reserves'):
        state = rules.deserialize_snapshot(fixtures[identity]['snapshot'])
        observer = rules.observation_payload(state)
        action, trace = native.choose(observer, diagnostic_capture=True)
        action = rules.serialize_action(action)
        checked = audit.audit_decision(observer, action, trace, 'search')
        assert audit.validate_search_work(trace, observer)['inference_calls'] == checked['inference_calls']
        recorded[identity] = checked
        if identity == 'turn123':
            assert checked['simulations'] == 16
            assert checked['inference_calls'] == 1
            assert checked['leaf_evaluations'] > 1
            old_trace = deepcopy(trace)
            for phase in old_trace['phases']:
                for key in ('neural_callback_count', 'neural_callback_rows', 'neural_callbacks'):
                    phase.pop(key)
            with pytest.raises(ValueError, match='aggregate inference count'):
                failed_attempt_audit.audit_decision(observer, action, old_trace, 'search')
            wrong = deepcopy(trace)
            wrong['inference_calls'] = wrong['inference_rows'] = checked['leaf_evaluations']
            with pytest.raises(ValueError, match='aggregate inference count'):
                audit.audit_decision(observer, action, wrong, 'search')
        elif identity in ('turn124', 'single_buy', 'forced_pass'):
            assert checked['inference_calls'] == checked['simulations'] == checked['leaf_evaluations'] == 0
        else:
            assert checked['inference_calls'] == checked['leaf_evaluations']
    assert recorded['turn123']['heuristic_leaf_evaluations'] > 0


def test_actual_native_all_fixed_counterexamples_work_counts(fixed_output_native_search):
    native = fixed_output_native_search
    for fixture in build_fixtures():
        observer = rules.observation_payload(rules.deserialize_snapshot(fixture['snapshot']))
        action, trace = native.choose(observer, diagnostic_capture=False)
        checked = audit.audit_decision(observer, rules.serialize_action(action), trace, 'search')
        assert checked['inference_calls'] <= checked['leaf_evaluations'] <= 17
        for phase in trace['phases']:
            assert 'neural_callbacks' in phase and 'inference' not in phase


def test_clock_charges_delivery_not_later_validation():
    assert audit.check_clock(clock()) == .2
    with pytest.raises(ValueError, match="late"):
        audit.check_clock(clock(received=1.000001))
    bad = clock()
    bad["received_before_deadline"] = 1
    with pytest.raises(ValueError):
        audit.check_clock(bad)


def test_forfeit_rejects_clamped_clock_and_infrastructure():
    state = rules.CorrectedEnv(seed=84515214).initial_state()
    before = rules.serialize_snapshot(state)
    row = {"ply": 0, "seat": 0, "arm": "greedy", "before": before,
           "observer": rules.observation_payload(state), "clock": clock("timeout", None),
           "completed": False, "action": None, "trace": None, "after": None}
    ending = {"reason": "forfeit", "winner": 1, "loser": 0, "failure": "timeout", "final": before}
    assert audit.audit_game(before, [row], ending, search_seat=1)["failed_moves"] == 1
    row["clock"]["deadline_seconds"] = .5
    with pytest.raises(ValueError, match="stage-clamped"):
        audit.audit_game(before, [row], ending, search_seat=1)
    row["clock"] = clock("resource_guard", None)
    with pytest.raises(ValueError, match="infrastructure"):
        audit.audit_game(before, [row], ending, search_seat=1)


def test_replay_sample_independent_of_order_and_outcome():
    observer = rules.observation_payload(rules.CorrectedEnv(seed=84515214).initial_state())
    rows = [{"pair_index": p, "game_index": 2*p, "ply": 2*p, "arm": "search", "seat": 0,
             "observer": observer, "completed": True, "winner": p % 2} for p in range(10)]
    manifest = audit.select_replay_sample(rows)
    changed = list(reversed(deepcopy(rows)))
    for row in changed:
        row["winner"] = None
    assert audit.select_replay_sample(changed) == manifest
    assert len(manifest["decisions"]) == 1 and len(manifest["empty_cells"]) == 47
    with pytest.raises(ValueError, match="duplicate"):
        audit.select_replay_sample(rows + rows[:1])


def test_private_category_uses_own_known_blind_reserve_provenance():
    fixture = next(f for f in build_fixtures() if f["id"] == "private_reserves")
    observer = rules.observation_payload(rules.deserialize_snapshot(fixture["snapshot"]))
    row = {"pair_index": 0, "game_index": 0, "ply": 42, "arm": "greedy", "seat": 0, "observer": observer}
    choice = audit.select_replay_sample([row])["decisions"][0]
    assert choice["private_category"] == "both" and choice["phase"] == "40+"


def test_classification_preserves_internal_guard_losses_and_infrastructure_stop():
    receipt = clock("worker_error")
    receipt["detail"] = "TimeoutError: ONE_SECOND_INTERNAL_DEADLINE: complete response exceeded 0.900s"
    assert audit.classify_failure(receipt) == "worker_error"
    receipt["detail"] = "ValueError: PLAYER_ILLEGAL_ACTION: returned action is not legal"
    assert audit.classify_failure(receipt) == "worker_error"
    for detail in ("ONE_SECOND_ADAPTER: source changed", "GREEDY_ADAPTER: native legal mismatch", "unclassified failure"):
        receipt["detail"] = detail
        with pytest.raises(ValueError):
            audit.classify_failure(receipt)


def test_descriptive_performance_keeps_missing_receipt_and_forfeit_separate():
    completed = clock()
    completed["telemetry"] = {"request_cpu_seconds": .1, "rss_bytes": 100, "peak_sampled_rss_bytes": 120}
    missing = clock("timeout", None)
    rows = [{"arm": "greedy", "clock": completed}, {"arm": "search", "clock": missing}]
    games = [{"reason": "natural", "plies": 58, "search_seat": 0},
             {"reason": "forfeit", "plies": 11, "search_seat": 1, "loser": 0, "winner": 1}]
    summary = audit.summarise_performance(rows, games)
    assert summary["natural_game_length_plies"]["mean"] == 58
    assert summary["arms"]["greedy"]["decision_cpu_seconds"]["mean"] == .1
    assert summary["arms"]["search"]["completed_arrival_seconds"]["count"] == 0
    assert summary["arms"]["search"]["missing_arrival_timestamps"] == 1
    assert len(summary["forfeits"]) == 1


def test_replay_sample_hashes_real_prepared_global_game_identity():
    import hashlib
    import json
    from pathlib import Path
    root = Path(__file__).resolve().parents[1]
    prepared = json.loads((root / "outputs/splendorust_roadmap_20261024_preparation_01/prospective_schedule.json").read_text())
    observer = rules.observation_payload(rules.CorrectedEnv(seed=84515214).initial_state())
    rows = [{"stage": "main", "pair_index": g["pair_index"], "game_index": g["game_index"],
             "ply": 0, "seat": 0, "arm": "search" if g["puct16_seat"] == 0 else "greedy",
             "observer": observer, "completed": True}
            for g in prepared["games"] if g["stage"] == "main"]
    sample = audit.select_replay_sample(rows)
    assert len(sample["decisions"]) == 2
    for arm in ("search", "greedy"):
        candidates = []
        for r in rows:
            if r["arm"] == arm:
                encoded = json.dumps([audit.AUDIT_DOMAIN, r["pair_index"], r["game_index"], 0, arm],
                                     ensure_ascii=True, separators=(",", ":"))
                candidates.append((hashlib.sha256(encoded.encode("utf-8")).hexdigest(), r["pair_index"], r["game_index"]))
        expected = min(candidates)
        selected = next(r for r in sample["decisions"] if r["arm"] == arm)
        assert (selected["selection_sha256"], selected["pair_index"], selected["game_index"]) == expected
    rows[2]["game_index"] = 0
    with pytest.raises(ValueError, match="sample row identity"):
        audit.select_replay_sample(rows)
