"""Model-free repeat-wrapper checks; saved fixtures never become new model runs."""
import ast
from copy import deepcopy
import inspect
import json
from pathlib import Path
import textwrap
from types import SimpleNamespace
from unittest.mock import patch

import pytest

from tools.splendorust_search128_repeat_20261029 import (
    adapters as a, common as c, corpus, scientific as s, transport as t,
    worker as w, qualification as q, placement,
)
from tools.splendorust_search128_resume_20261029 import (
    adapters as old_a, worker as old_w, transport as old_t,
    qualification as old_q,
)
from tools.external_strength_screen_20261018_02.splendorust_adapter import Opponent
from tools.splendorust_boundary_timing_20261025.parent_profile import ProfiledStudyWorker

OLD = c.ROOT / 'outputs/splendorust_search128_strength_20261028_01'
SOURCE = c.ROOT / 'tools/splendorust_search128_resume_20261029'


def syntax(obj):
    return ast.dump(ast.parse(textwrap.dedent(inspect.getsource(obj))), include_attributes=False)


@pytest.mark.parametrize('name', ['_ordinary', 'handle', 'serve', 'response_packet', 'close_context'])
def test_worker_numerical_clock_and_emission_path_unchanged(name):
    assert syntax(getattr(w, name)) == syntax(getattr(old_w, name))


def test_initializer_only_changes_explicit_execution_phase():
    new = inspect.getsource(w.initialize).replace(
        "os.environ.get('SPLENDOR_SEARCH128_REPEAT_PHASE','repeat')",
        "os.environ.get('SPLENDOR_SEARCH128_PHASE','strength')")
    assert ast.dump(ast.parse(new), include_attributes=False) == syntax(old_w.initialize)
    assert syntax(a.BaselineOpponent) == syntax(old_a.BaselineOpponent)


@pytest.mark.parametrize('module', [s, corpus, placement])
def test_frozen_science_and_placement_byte_identity(module):
    assert Path(module.__file__).read_bytes() == (SOURCE / Path(module.__file__).name).read_bytes()


@pytest.mark.parametrize('name', ['request', 'collect_profiles', 'ordinary', 'validate_profiles',
    'verify_saved_record', 'verify_placement_request', 'verify_worker_logs', 'verify_closed_worker',
    'verify_graph_evidence', 'research_child_identity', 'research_boundaries'])
def test_parent_clock_capture_and_evidence_paths_unchanged(name):
    assert syntax(getattr(t, name)) == syntax(getattr(old_t, name))


def test_raw_verifier_only_tightens_finite_request_count():
    new = inspect.getsource(t.verify_raw_transport).replace(
        "c.require(len(paths)<=62,'Bounded repeat request inventory')",
        "c.require(len(paths)<=2002,'Bounded request inventory')")
    assert ast.dump(ast.parse(new), include_attributes=False) == syntax(old_t.verify_raw_transport)
    assert syntax(t._PlacementConstructor) == syntax(old_t._PlacementConstructor)
    assert syntax(t.PlacedProfiledStudyWorker) == syntax(old_t.PlacedProfiledStudyWorker)
    assert t.PlacedProfiledStudyWorker.request is ProfiledStudyWorker.request
    for name in ('_receive', '_stdout_reader', '_stderr_reader'):
        assert getattr(t.PlacedProfiledStudyWorker, name) is getattr(ProfiledStudyWorker, name)


@pytest.mark.parametrize('quota,route', [(16, 'frozen'), (64, 'frozen'), (128, 'research')])
def test_immutable_profile_content_and_path(quota, route):
    assert a.profile_path(quota, route=route) == OLD / 'profiles' / f'puct{quota}.json'
    saved = c.read(a.profile_path(quota, route=route))
    assert a.profile(quota, route=route) == saved == old_a.profile(quota, route=route)
    assert (saved['iterations'], saved['world_pool'], saved['depth'], saved['internal_budget_seconds']) == (quota, 3, 64, 9.)
    assert a.profile_sha256(quota) == old_a.profile_sha256(quota)


def test_once_only_source_guard_precedes_models():
    observed = []
    def stop():
        observed.append('source guard')
        raise ValueError('stop before learned model')
    with patch.object(w, 'verify_runtime_sources', stop), patch.object(a, 'initialize_adapters', side_effect=AssertionError('model forbidden')):
        with pytest.raises(ValueError, match='stop before learned model'):
            w.initialize('research')
    assert observed == ['source guard']
    nodes = ast.walk(ast.parse(inspect.getsource(w.initialize)))
    assert sum(isinstance(n, ast.Call) and isinstance(n.func, ast.Name) and n.func.id == 'verify_runtime_sources' for n in nodes) == 1


def test_wrong_new_phase_rejected_before_models(tmp_path, monkeypatch):
    folder = tmp_path / 'workers' / 'fixture'; folder.mkdir(parents=True)
    freeze = tmp_path / 'freeze.json'; freeze.write_text(json.dumps({'phase': 'strength'}))
    monkeypatch.setattr(c, 'OUT', tmp_path)
    for key, value in dict(STUDY_WORKER_DIR=str(folder), STUDY_BOUNDARY_PROFILE='1',
            STUDY_RESEARCH_FREEZE=str(freeze), STUDY_RESEARCH_FREEZE_SHA256=c.file_hash(freeze),
            SPLENDOR_SEARCH128_REPEAT_PHASE='repeat').items(): monkeypatch.setenv(key, value)
    with patch.object(w, 'verify_runtime_sources', return_value={}), patch.object(a, 'initialize_adapters', side_effect=AssertionError('model forbidden')):
        with pytest.raises(ValueError, match='phase'):
            w.initialize('research')


@pytest.mark.parametrize('route,quota', [('research', 128), ('frozen', 64)])
def test_actual_saved_result_and_full_callback_validator(route, quota):
    receipt = c.read(OLD / 'entrypoint_checks' / f'{route}.json')
    result = next(x for x in receipt['result']['fixtures'] if x['id'] == 'private_reserves')['results']['search']
    fixture = next(x for x in corpus.build_plan(quota)['counterexamples'] if x['id'] == 'private_reserves')
    observer = corpus.rules.observation_payload(corpus.guard.decode_training_snapshot(fixture['snapshot']))
    freeze = c.read(OLD / 'freeze.json'); identity = a.profile_identity(freeze, route)
    original = deepcopy(result)
    assert a.validate_result(result, observer, route, identity=identity) and result == original
    corrupt = deepcopy(result); corrupt['simulations_quota'] = 64 if quota == 128 else 128
    with pytest.raises(ValueError): a.validate_result(corrupt, observer, route, identity=identity)
    if result['diagnostics']['phases'][0]['neural_callback_count']:
        corrupt = deepcopy(result); corrupt['diagnostics']['phases'][0]['inference'][0].pop('tokens')
        with pytest.raises(ValueError): a.validate_result(corrupt, observer, route, identity=identity)


def test_ten_second_receipts_not_deployment_relabelled():
    for seconds in (1.1, 8.99, 9.5, 10.):
        value = dict(clock='perf_counter_ns', request_id='request-00000003',
            deadline_seconds=10., elapsed_seconds=seconds, receipt_elapsed_seconds=seconds,
            received_before_deadline=True, accepted=True, status='ok', result={'fixture': True})
        before = deepcopy(value)
        assert s.check_clock(value) == seconds and value == before
    value['receipt_elapsed_seconds'] = 10.01
    with pytest.raises(ValueError): s.check_clock(value)


@pytest.mark.parametrize('now,success', [(101.2, True), (108.999, True), (109., False)])
def test_existing_nine_second_origin_not_reset(now, success):
    obj = object.__new__(a.BaselineOpponent)
    obj.profile = dict(iterations=128, internal_budget_seconds=9.)
    obj.request_deadline = 109.; obj.active_deadline = None; obj.admission_guard = None; obj.capture = False
    with patch.object(a.time, 'perf_counter', return_value=now), patch.object(Opponent, 'choose', return_value=('a', {})):
        if success: assert obj.choose({}) == ('a', {})
        else:
            with pytest.raises(TimeoutError): obj.choose({})
    assert obj.request_deadline == 109.


def test_two_field_request_allowlist_unchanged():
    with pytest.raises(ValueError, match='allowlist'):
        w._ordinary(dict(route='research', request_started=100., diagnostic_capture=False),
                    dict(player='search', observer={}, seed=1))


def test_actual119_suite_plan_and_unchanged_forensic_checker():
    plan = q.build_plan()
    assert (plan['scientific_decisions'], plan['graph_cases'], plan['ordinary_requests'], plan['total_parent_requests']) == (119, 22, 120, 124)
    assert [x['complete_decisions'] for x in plan['entrypoint_requests']] == [67, 52]
    assert syntax(q.verify_entrypoint) == syntax(old_q.verify_entrypoint)
    for route in ('research', 'frozen'):
        receipt = c.read(OLD / 'entrypoint_checks' / f'{route}.json')
        proof = q.verify_entrypoint(OLD / 'workers' / f'qualification_{route}', receipt, plan, c.read(OLD / 'freeze.json'))
        assert proof['status'] == 'passed' and proof['raw_execution_decisions_bound'] == (67 if route == 'research' else 52)


def make_lifecycle_fixture(tmp_path, monkeypatch):
    monkeypatch.setattr(c, 'OUT', tmp_path)
    freeze = dict(declared_worker_lifecycles=t.lifecycle_inventory())
    receipts = []
    for route in ('research', 'frozen'):
        folder = tmp_path / 'workers' / f'qualification_repeat_{route}'
        folder.mkdir(parents=True)
        for sub in ('startup_inputs', 'inputs'): (folder / sub).mkdir()
        (folder/'startup_inputs/request-00000001.json').write_text(json.dumps({'operation': 'graph_parity', 'fixtures': []}))
        (folder/'startup_inputs/request-00000002.json').write_text(json.dumps({'operation': 'entrypoint_checks'}))
        clock = dict(status='ok', request_id='request-00000002', result=dict(counts=dict(complete_decisions=67 if route=='research' else 52)))
        (folder/'request-00000002.json').write_text(json.dumps(clock))
        path = tmp_path/'entrypoint_checks'/f'{route}.json'; path.parent.mkdir(exist_ok=True); path.write_text(json.dumps(clock))
        receipts.append(dict(route=route, worker_folder=folder.relative_to(c.ROOT).as_posix(), clock=c.artifact(path),
            verification=dict(status='passed', route=route, raw_execution_decisions_bound=67 if route=='research' else 52)))
    (tmp_path/'qualification_summary.json').write_text(json.dumps(dict(status='passed',scientific_decisions=119,
        parent_requests=2,graph_cases=22,same_workers_retained_for_repeats=True,receipts=receipts)))
    return freeze


def test_strict_prospective_startup_then_sixty_request_inventory(tmp_path, monkeypatch):
    monkeypatch.setattr(c, 'ROOT', tmp_path)
    freeze = make_lifecycle_fixture(tmp_path, monkeypatch)
    folder = tmp_path/'workers/qualification_repeat_research'
    assert t.authenticated_lifecycle(folder, freeze, 'research')['startup_requests'] == 2
    for i in range(3, 63): (folder/'inputs'/f'request-{i:08d}.json').write_text('{}')
    assert t.authenticated_lifecycle(folder, freeze, 'research')['ordinary_requests'] == 60
    (folder/'inputs/request-00000063.json').write_text('{}')
    with pytest.raises(ValueError, match='bounded ordinary'): t.authenticated_lifecycle(folder, freeze, 'research')


@pytest.mark.parametrize('mutation', ['missing', 'repeat', 'swapped', 'wrong_route', 'wrong_summary'])
def test_illegal_startup_lifecycle_never_accepted(tmp_path, monkeypatch, mutation):
    monkeypatch.setattr(c, 'ROOT', tmp_path)
    freeze = make_lifecycle_fixture(tmp_path, monkeypatch)
    folder = tmp_path/'workers/qualification_repeat_research'
    if mutation=='missing': (folder/'startup_inputs/request-00000002.json').unlink()
    elif mutation=='repeat': (folder/'startup_inputs/request-00000003.json').write_text('{"operation":"entrypoint_checks"}')
    elif mutation=='swapped': (folder/'startup_inputs/request-00000001.json').write_text('{"operation":"entrypoint_checks"}')
    elif mutation=='wrong_route': freeze['declared_worker_lifecycles'][folder.name]['route']='frozen'
    else:
        path=tmp_path/'qualification_summary.json'; value=json.loads(path.read_text()); value['scientific_decisions']=118; path.write_text(json.dumps(value))
    with pytest.raises(ValueError): t.authenticated_lifecycle(folder, freeze, 'research')


def test_no_frozen_module_or_class_mutation():
    for module in (a,w,s,corpus,t,q):
        for node in ast.walk(ast.parse(inspect.getsource(module))):
            if isinstance(node, ast.Assign):
                for target in node.targets:
                    if isinstance(target, ast.Attribute) and isinstance(target.value, ast.Name):
                        assert target.value.id not in {'Opponent','SearchOpponent','original','old','graph','frozen'}
