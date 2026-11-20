"""Model-free route, native-configuration, exact-proof and lifecycle regressions."""
import ast
from copy import deepcopy
import inspect
import json
from pathlib import Path
import struct
import textwrap
from unittest.mock import patch

import pytest

from tools.splendorust_exploration128_20261029 import (
    adapters as a, common as c, corpus, scientific as s, transport as t,
    worker as w, qualification as q, placement,
)
from tools.splendorust_search128_repeat_20261029 import adapters as old_a, worker as old_w, transport as old_t
from tools.external_strength_screen_20261018_02.splendorust_adapter import Opponent
from tools.splendorust_boundary_timing_20261025.parent_profile import ProfiledStudyWorker

OLD = c.ROOT / 'outputs/splendorust_search128_strength_20261028_01'


def syntax(obj):
    return ast.dump(ast.parse(textwrap.dedent(inspect.getsource(obj))), include_attributes=False)


@pytest.mark.parametrize('name', ['_ordinary', 'serve', 'response_packet', 'close_context'])
def test_original_completed_decision_clock_and_emission_body(name):
    new = inspect.getsource(getattr(w, name)).replace("route='frozen'", "route='research'")
    assert ast.dump(ast.parse(textwrap.dedent(new)), include_attributes=False) == syntax(getattr(old_w, name))


def test_incumbent_and_reference_adapters_numerically_unchanged():
    assert syntax(a.BaselineOpponent) == syntax(old_a.BaselineOpponent)
    for quota in (16, 128):
        saved = c.read(OLD/'profiles'/f'puct{quota}.json')
        assert a.profile(quota, route='frozen') == saved
        assert a.profile_path(quota, route='frozen') == OLD/'profiles'/f'puct{quota}.json'
    for route in a.ROUTES:
        assert a.profile(16, route=route) == a.profile(16, route='frozen')
        assert a.profile(16, route=route)['cpuct'] == .4


@pytest.mark.parametrize('route,coefficient', [('low', .2), ('high', .8), ('parity', .4)])
def test_only_declared_scientific_coefficient_changes(route, coefficient):
    value = a.profile(128, route=route)
    original = a.profile(128, route='frozen')
    ignored = {'profile_id', 'wrapper_id', 'native_binary', 'artifacts', 'execution_policy', 'cpuct'}
    assert {k:v for k,v in value.items() if k not in ignored} == {k:v for k,v in original.items() if k not in ignored}
    assert value['cpuct'] == coefficient
    assert value['iterations'] == 128 and value['world_pool'] == 3
    assert a.native_descriptor(route, 16)['sha256'] == a.SEARCH_BINARY_SHA256
    assert a.native_descriptor(route, 128)['sha256'] != a.SEARCH_BINARY_SHA256


@pytest.mark.parametrize('route', ['research', 'frozen64', '0.3', None])
def test_undeclared_routes_rejected(route):
    with pytest.raises(ValueError): a.route_cpuct(route)


@pytest.mark.parametrize('name', ['request', 'collect_profiles', 'ordinary', 'verify_saved_record',
    'verify_placement_request', 'verify_worker_logs', 'verify_graph_evidence', 'research_child_identity', 'research_boundaries'])
def test_original_clock_capture_and_observer_unchanged(name):
    assert syntax(getattr(t, name)) == syntax(getattr(old_t, name))


def test_original_process_reader_and_constructor_unchanged():
    assert syntax(t._PlacementConstructor) == syntax(old_t._PlacementConstructor)
    assert syntax(t.PlacedProfiledStudyWorker) == syntax(old_t.PlacedProfiledStudyWorker)
    for name in ('request', '_receive', '_stdout_reader', '_stderr_reader'):
        assert getattr(t.PlacedProfiledStudyWorker, name) is getattr(ProfiledStudyWorker, name)


def test_actual_guard_is_once_before_model_creation():
    observed = []
    def stop():
        observed.append('guard')
        raise ValueError('stop before model')
    with patch.object(w, 'verify_runtime_sources', stop), patch.object(a, 'initialize_adapters', side_effect=AssertionError('model forbidden')):
        with pytest.raises(ValueError, match='stop before model'): w.initialize('low')
    assert observed == ['guard']
    nodes = ast.walk(ast.parse(inspect.getsource(w.initialize)))
    assert sum(isinstance(n, ast.Call) and isinstance(n.func, ast.Name) and n.func.id == 'verify_runtime_sources' for n in nodes) == 1


def test_extra_worker_request_parameters_rejected():
    with pytest.raises(ValueError, match='allowlist'):
        w._ordinary(dict(route='low', request_started=100., diagnostic_capture=False),
                    dict(player='search', observer={}, cpuct=.8))


def test_deployment_thresholds_do_not_rewrite_clocks():
    for seconds in (.71, 1.1, 8.99, 9.5, 10.):
        value = dict(clock='perf_counter_ns', request_id='request-00000003', deadline_seconds=10.,
            elapsed_seconds=seconds, receipt_elapsed_seconds=seconds, received_before_deadline=True,
            accepted=True, status='ok', result={'fixture':True})
        assert s.check_clock(value) == seconds
    value['receipt_elapsed_seconds'] = 10.01
    with pytest.raises(ValueError): s.check_clock(value)


def config(route):
    cpuct = a.route_cpuct(route)
    return dict(schema=a.NATIVE_SCHEMA, cpuct=cpuct, cpuct_bits=struct.unpack('<Q', struct.pack('<d', cpuct))[0],
        world_pool=3, chance_universes=3, dynamic_fpu=True, fpu_reduction=.02965, uniform_prior=0., inference_batch_size=1)


@pytest.mark.parametrize('field,value', [('cpuct', .4), ('cpuct_bits', 0), ('world_pool', 6), ('inference_batch_size', 2), ('fpu_reduction', 0.)])
def test_wrong_native_actual_configuration_rejected(field, value):
    actual = config('low'); actual[field] = value
    with pytest.raises(ValueError): a.validate_native_configuration(actual, 'low')


def configured_fake(tmp_path, route='low'):
    obj = object.__new__(a.ConfiguredOpponent)
    obj.route=route; obj.attestation_folder=tmp_path; obj.attestation_sequence=0; obj.attestation_previous=None
    return obj


def test_select_injects_exact_coefficient_and_preserves_native_receipt(tmp_path):
    obj = configured_fake(tmp_path)
    request = dict(op='select', observation={'fixture':True}, legal=[{'kind':'Take','tokens':[1,1,1,0,0]}], seed=17, iterations=128)
    native = dict(kind='result', action=request['legal'][0], simulations=128, leaf_calls=129, native_legal_count=1,
        heuristic_phase=False, native_configuration=config('low'), neural_callback_count=0, neural_callback_rows=0, neural_callbacks=[])
    with patch.object(a.BaselineOpponent, 'exchange', return_value=native) as original:
        result = obj.exchange(request)
    assert original.call_args.args[0] == dict(request, cpuct=.2)
    assert 'cpuct' not in request and 'native_configuration' not in result
    event = json.loads((tmp_path/'native_configuration_receipts.jsonl').read_text())
    assert event['native_result']['native_configuration'] == config('low')
    assert event['request_sha256'] == c.digest(dict(request, cpuct=.2))
    assert event['sequence'] == 0 and event['previous_sha256'] is None


@pytest.mark.parametrize('change', [{'iterations':16}, {'cpuct':.8}])
def test_configuration_cannot_leak_to_reference_or_be_overridden(tmp_path, change):
    obj = configured_fake(tmp_path)
    request = dict(op='select', observation={}, legal=[], seed=17, iterations=128); request.update(change)
    with patch.object(a.BaselineOpponent, 'exchange', side_effect=AssertionError('must reject before native')):
        with pytest.raises(ValueError): obj.exchange(request)


def test_native_build_is_separate_and_core_search_is_byte_identical():
    receipt = a.native_build_receipt()
    assert receipt['model_inference_executed'] is False and receipt['core_search_unchanged'] is True
    for row in receipt['core_and_workspace_sources']:
        c.verify([row['original'], row['copy']])
        assert row['original']['sha256'] == row['copy']['sha256']
    source=(c.ROOT/receipt['bridge_source']['path']).read_text()
    assert 'agent.cpuct=value;' in source
    assert 'apply_exploration(&mut agent,selected_cpuct.unwrap())' in source
    core = c.ROOT/'outputs/external_opponent_search_20261018_02/native_build_v2/crates/splendor-agents/src/neural_search.rs'
    assert core.read_text().count('self.cpuct') == 2
    assert c.read(c.ROOT/receipt['test_attempt']['path'])['returncode'] == 0
    assert len(receipt['smoke']) == 6


def test_fixed_qualification_inventory_and_panel_balance():
    plan=q.build_plan()
    assert (plan['scientific_decisions'],plan['graph_cases'],plan['ordinary_requests'],plan['total_parent_requests']) == (364,44,96,104)
    assert [row['route'] for row in plan['entrypoint_requests']] == ['frozen','parity','low','high']
    assert all(row['complete_decisions']==67 for row in plan['entrypoint_requests'])
    for route in q.QUAL_ROUTES:
        rows=[r for r in plan['panel_requests'] if r['route']==route]
        assert len(rows)==24 and len({r['root_id'] for r in rows})==18 and sum(r['repeat'] for r in rows)==6


def test_saved_actual128_counterexamples_through_new_checker():
    # Historical receipt is read-only and explicitly re-labelled only as a checker fixture.
    profiles=a.prepare_profiles(); freeze={'profiles':profiles['profiles']}
    receipt=deepcopy(c.read(OLD/'entrypoint_checks/research.json'))
    receipt['result']['execution_route']='frozen';receipt['result']['execution_schema']=a.EXECUTION_SCHEMA
    proof=q.verify_entrypoint(OLD/'workers/qualification_research',receipt,q.build_plan(),freeze)
    assert proof['raw_execution_decisions_bound']==67 and proof['independent_saved_transitions']==37
    changed=deepcopy(receipt);changed['result']['counts']['complete_decisions']=66
    with pytest.raises(ValueError):q.verify_entrypoint(OLD/'workers/qualification_research',changed,q.build_plan(),freeze)


def test_default_parity_compares_full_callbacks_not_just_action():
    old=deepcopy(c.read(OLD/'entrypoint_checks/research.json')['result']);old['execution_route']='frozen'
    new=deepcopy(old);new['execution_route']='parity'
    for fixture in new['fixtures']:
        result=fixture['results']['search']
        result['profile_id']=a.profile_id(128,route='parity');result['profile_sha256']='f'*64
        result['wrapper_id']=a.wrapper_id(128,route='parity');result['diagnostics']['profile_id']=result['profile_id']
    assert q.compare_default_parity(old,new)['complete_results']==30
    target=next(f['results']['search'] for f in new['fixtures'] if f['results']['search']['diagnostics']['phases'] and f['results']['search']['diagnostics']['phases'][0]['neural_callback_count'])
    target['diagnostics']['phases'][0]['neural_callbacks'][0]='0'*64
    with pytest.raises(ValueError):q.compare_default_parity(old,new)


@pytest.mark.parametrize('mutation', [None, 'coefficient', 'chain', 'action', 'missing'])
def test_native_attestation_readback_binds_every_completed_phase(tmp_path, monkeypatch, mutation):
    receipt=c.read(OLD/'entrypoint_checks/research.json')
    result=next(row['results']['search'] for row in receipt['result']['fixtures'] if row['id']=='private_reserves')
    fixture=next(row for row in corpus.build_fixtures() if row['id']=='private_reserves')
    observer=corpus.rules.observation_payload(corpus.guard.decode_training_snapshot(fixture['snapshot']))
    expected=a._phase_configuration_expected(observer,result['diagnostics'],'low')
    assert expected
    contract=dict(kind='result',schema=a.NATIVE_SCHEMA,allowed_cpuct=[.2,.4,.8],allowed_iterations=[128],
        selection_field='NeuralAgent.cpuct',core_search_unchanged=True,neural_callback_count=0,neural_callback_rows=0,neural_callbacks=[])
    (tmp_path/'native_configuration_contract.json').write_text(json.dumps(contract))
    (tmp_path/'inputs').mkdir()
    (tmp_path/'inputs/request-00000002.json').write_text(json.dumps(dict(player='search',observer=observer)))
    (tmp_path/'request-00000002.json').write_text(json.dumps(dict(status='ok',result=result)))
    records=[];previous=None
    for index,value in enumerate(expected):
        value=deepcopy(value);value['native_result']['native_configuration']=config('low')
        event=dict(sequence=index,previous_sha256=previous,**value);previous=c.digest(event);records.append(event)
    if mutation=='coefficient':records[0]['native_result']['native_configuration']['cpuct']=.4
    elif mutation=='chain':records[0]['previous_sha256']='f'*64
    elif mutation=='action':records[0]['native_result']['action']={'kind':'Pass'}
    elif mutation=='missing':records.pop()
    (tmp_path/'native_configuration_receipts.jsonl').write_text(''.join(json.dumps(row)+'\n' for row in records))
    monkeypatch.setattr(c,'artifact',lambda path:dict(path=str(path),bytes=Path(path).stat().st_size,sha256=c.file_hash(path)))
    if mutation is None:
        proof=a.verify_native_configuration_receipts(tmp_path,'low')
        assert proof['phase_receipts']==proof['bound_successful_phases']==len(expected)
    else:
        with pytest.raises(ValueError):a.verify_native_configuration_receipts(tmp_path,'low')


def test_no_frozen_class_or_module_mutation():
    for module in (a,w,s,corpus,t,q):
        for node in ast.walk(ast.parse(inspect.getsource(module))):
            if isinstance(node,ast.Assign):
                for target in node.targets:
                    if isinstance(target,ast.Attribute) and isinstance(target.value,ast.Name):
                        assert target.value.id not in {'Opponent','SearchOpponent','original','old','graph','frozen'}
