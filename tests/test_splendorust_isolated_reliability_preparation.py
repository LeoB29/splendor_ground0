"""Model-free qualification preservation and complete resource accounting."""
from copy import deepcopy
from types import SimpleNamespace

import pytest

from tools.splendorust_isolated_reliability_20261026 import preparation as p


@pytest.fixture(scope='module')
def accepted():
    # Read the finite summary documents, never models or the full evidence seal.
    return {name: p.c.read(p.QUALIFIED/name) for name in p.ACCEPTANCE_NAMES}


@pytest.fixture(scope='module')
def forecast():
    return p.resource_forecast()


def test_exact_latest_qualification_is_accepted_without_rerunning_it(accepted):
    assert p.validate_accepted_documents(accepted)


@pytest.fixture(scope='module')
def bindings():
    return ({name: p.c.artifact(p.QUALIFIED/name) for name in p.ACCEPTANCE_NAMES},
            p.c.artifact(p.c.ROOT/'tools/splendorust_supervisor_qualification_review_20261026.py'))


def test_summary_bytes_and_review_execution_source_are_pinned(accepted, bindings):
    artifacts, source = bindings
    result = p.validate_review_bindings(accepted, artifacts, source)
    assert set(result['independently_reviewed_summary_names']) == set(p.REVIEW_COVERED_NAMES)
    assert 'all_placement_integrity.json' in result['preserved_additional_summary_names']
    assert result['additional_summaries_not_claimed_independently_read'] is True


def test_modified_summary_bytes_or_missing_reviewed_summary_are_rejected(accepted, bindings):
    artifacts, source = deepcopy(bindings)
    artifacts['all_worker_integrity.json']['sha256'] = 'different'
    with pytest.raises(ValueError, match='reviewed summary bytes changed'):
        p.validate_review_bindings(accepted, artifacts, source)
    documents = deepcopy(accepted)
    target = bindings[0]['all_worker_integrity.json']['path']
    documents['independent_results_review.json']['consumed_inputs'] = [
        item for item in documents['independent_results_review.json']['consumed_inputs'] if item['path'] != target]
    with pytest.raises(ValueError, match='summary absent'):
        p.validate_review_bindings(documents, *bindings)


def test_review_script_and_execution_receipt_cannot_be_replaced(accepted, bindings):
    artifacts, source = deepcopy(bindings)
    source['sha256'] = 'different'
    with pytest.raises(ValueError, match='review source'):
        p.validate_review_bindings(accepted, artifacts, source)
    artifacts, source = deepcopy(bindings)
    artifacts['independent_review_execution.json']['sha256'] = 'different'
    with pytest.raises(ValueError, match='execution receipt changed'):
        p.validate_review_bindings(accepted, artifacts, source)


@pytest.mark.parametrize('mutate', [
    lambda x: x['completion.json'].__setitem__('freeze_sha256', 'different'),
    lambda x: x['execution.json'].__setitem__('exit_code', 1),
    lambda x: x['independent_review_execution.json'].__setitem__('stdout_sha256', 'different'),
    lambda x: x['qualification_summary.json']['suites'][0].__setitem__('complete_decisions', 51),
    lambda x: x['timing_validation.json'].__setitem__('requests', 191),
    lambda x: x['timing_validation.json'].__setitem__('fully_replicated_groups', 47),
    lambda x: x['timing_gate.json']['per_quota']['32']['distributions']['isolated'].__setitem__('max', .7),
    lambda x: x['timing_lifecycle_verification.json'].__setitem__('continuation_workers', 1),
    lambda x: x['repeated_decision_summary.json'].__setitem__('requests', 48),
    lambda x: x['verification_summary.json'].__setitem__('live_requests', 337),
    lambda x: x['freeze.json']['previous_attempt_acceptance'].__setitem__('old_qualification_reclassified', True),
    lambda x: x['freeze.json']['history_acceptance'].__setitem__('freeze_sha256', 'different'),
    lambda x: x['freeze.json']['placement']['role_masks'].__setitem__('isolated', [0]),
])
def test_partial_changed_or_relabelled_qualification_cannot_be_accepted(accepted, mutate):
    changed = deepcopy(accepted)
    mutate(changed)
    with pytest.raises(ValueError):
        p.validate_accepted_documents(changed)


def test_actual_forecast_charges_exact_200_not_206_games_and_all_workers(forecast):
    assert p.validate_resource_forecast(forecast) == forecast
    assert forecast['inventories'] == dict(total_games=200, operational_games=6, main_games=194,
        actual_entrypoint_workers=2, actual_entrypoint_decisions=104, timing_requests=0,
        game_workers_maximum=400, reproduction_requests_maximum=48,
        workers_maximum=450, graph_workers_maximum=249)
    assert forecast['outcomes_used_for_selection_or_resource_estimation'] is False
    assert forecast['projected_final_bytes'] < p.c.FINAL_RESERVE_BYTES


def test_main_projection_charges_exact_remaining194_complete_game_lifecycles(forecast):
    measured = forecast['measured']
    assert forecast['projected_stage_seconds']['main'] == pytest.approx(
        1.5*194*measured['maximum_complete_game_seconds']+30.)
    assert forecast['projected_stage_seconds']['operational'] == pytest.approx(
        1.5*6*measured['maximum_complete_game_seconds']+15.)
    # Current game cost already includes workers, rather than measuring moves only.
    assert measured['maximum_complete_game_seconds'] > 30


def test_final_projection_includes_all200_saved_game_checks_and48_new_reproductions(forecast):
    m = forecast['measured']
    required = 1.5*(200*m['maximum_game_independent_seconds']+
        48*(m['maximum_startup_and_parity_seconds']+1.+m['per_decision_validation_seconds']+
            m['maximum_closed_worker_check_seconds']+.015)+
        2*m['maximum_closed_worker_check_seconds']+120.)
    assert forecast['projected_stage_seconds']['final'] >= required
    slower = deepcopy(m)
    slower['maximum_startup_and_parity_seconds'] += 1.
    assert p.project_resources(slower)['projected_stage_seconds']['final'] >= required+72.


@pytest.mark.parametrize('change', [
    lambda x: x['inventories'].__setitem__('total_games', 206),
    lambda x: x['inventories'].__setitem__('main_games', 200),
    lambda x: x['inventories'].__setitem__('workers_maximum', 72),
    lambda x: x['inventories'].__setitem__('graph_workers_maximum', 31),
    lambda x: x['inventories'].__setitem__('reproduction_requests_maximum', 38),
    lambda x: x.__setitem__('outcomes_used_for_selection_or_resource_estimation', True),
    lambda x: x.__setitem__('model_calls', 1),
    lambda x: x['measured'].__setitem__('maximum_complete_game_seconds', float('nan')),
    lambda x: x['projected_stage_seconds'].__setitem__('final', p.c.STAGES['final']),
    lambda x: x.__setitem__('projected_prefinal_bytes', p.c.PREFINAL_EVIDENCE_BYTES),
    lambda x: x.__setitem__('projected_final_bytes', p.c.FINAL_RESERVE_BYTES),
    lambda x: x.__setitem__('projected_total_seconds', 0),
])
def test_omitted_inventory_nonfinite_cost_or_exhausted_bound_is_rejected(forecast, change):
    changed = deepcopy(forecast)
    change(changed)
    with pytest.raises(ValueError):
        p.validate_resource_forecast(changed)


def test_stage_duration_uses_larger_real_clock_and_rejects_backwards_time():
    assert p._duration(dict(monotonic=10., unix_time=100.), dict(monotonic=12., unix_time=103.)) == 3.
    with pytest.raises(ValueError):
        p._duration(dict(monotonic=10., unix_time=100.), dict(monotonic=9., unix_time=99.))


def test_seed_scan_rejects_registered_collision_and_excludes_own_new_schedule(tmp_path, monkeypatch):
    monkeypatch.setattr(p.c, 'ROOT', tmp_path)
    monkeypatch.setattr(p.c, 'OUT', tmp_path/'outputs/new')
    old = tmp_path/'outputs/old/schedule.json'; old.parent.mkdir(parents=True)
    old.write_text('{"nested":[{"seed_decimal":"1234"}]}')
    current = p.c.OUT/'schedule.json'; current.parent.mkdir(parents=True)
    current.write_text('[{"seed_decimal":"5678"}]')
    monkeypatch.setattr(p.subprocess, 'run', lambda *args, **kwargs: SimpleNamespace(
        stdout='outputs/old/schedule.json\noutputs/new/schedule.json\n'))
    with pytest.raises(ValueError, match='seed collision'):
        p.seed_inventory([dict(seed_decimal='1234')])
    receipt = p.seed_inventory([dict(seed_decimal='5678')])
    assert receipt['collisions'] == []
    assert [x['path'] for x in receipt['checked']] == ['outputs/old/schedule.json']


def test_prior_output_dependencies_cannot_be_dropped_or_replaced(tmp_path, monkeypatch):
    monkeypatch.setattr(p.c, 'ROOT', tmp_path)
    profile = tmp_path/'profile.json'; profile.write_text('{}')
    monkeypatch.setattr(p.c, 'profile_path', lambda quota: profile)
    monkeypatch.setattr(p.c, 'profile_for', lambda quota: dict(artifacts=[]))
    old = dict(path='outputs/old/failure.json', bytes=10, sha256='oldhash')
    prior = dict(inputs=[old], native_artifacts=[])
    assert old in p.dependencies(prior, [], [])
    with pytest.raises(ValueError, match='Frozen input changed'):
        p.dependencies(prior, [dict(evidence=[dict(old, sha256='replacement')])], [])
    with pytest.raises(ValueError, match='Repository-relative'):
        p.dependencies(prior, [dict(evidence=[dict(path='../escape', bytes=1, sha256='bad')])], [])
