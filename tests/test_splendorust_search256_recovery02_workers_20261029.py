"""Model-free quota, historical-science, corrected-entrypoint and clock checks."""
import ast
from copy import deepcopy
import inspect
import json
from pathlib import Path
import sys
import textwrap
from unittest.mock import patch

import pytest

from tools.splendorust_search256_strength_20261029_02 import (
    adapters as a, common as c, corpus, qualification as q, scientific as s,
    transport as t, worker as w, performance, placement,
)
from tools.splendorust_exploration128_20261029_02 import (
    adapters as old_a, worker as old_w, transport as old_t, placement as old_placement,
)
from tools.splendorust_boundary_timing_20261025.parent_profile import ProfiledStudyWorker
from tools.external_strength_screen_20261018_02.splendorust_adapter import Opponent


def syntax(obj):
    return ast.dump(ast.parse(textwrap.dedent(inspect.getsource(obj))), include_attributes=False)


def prepared():
    return a.prepare_profiles()


@pytest.mark.parametrize('name', ['_ordinary', 'serve', 'response_packet', 'close_context', 'handle'])
def test_complete_clock_capture_and_worker_handler_unchanged(name):
    assert syntax(getattr(w, name)) == syntax(getattr(old_w, name))


@pytest.mark.parametrize('name', ['request', 'collect_profiles', 'ordinary', 'verify_saved_record',
    'verify_placement_request', 'verify_worker_logs', 'verify_graph_evidence', 'research_child_identity', 'research_boundaries'])
def test_original_trusted_clock_reader_and_evidence_paths_unchanged(name):
    assert syntax(getattr(t, name)) == syntax(getattr(old_t, name))


def test_original_search_adapter_and_native_transport_unchanged():
    assert syntax(a.BaselineOpponent) == syntax(old_a.BaselineOpponent)
    assert syntax(t._PlacementConstructor) == syntax(old_t._PlacementConstructor)
    assert syntax(t.PlacedProfiledStudyWorker) == syntax(old_t.PlacedProfiledStudyWorker)
    assert Path(placement.__file__).read_bytes() == Path(old_placement.__file__).read_bytes()
    for name in ('request', '_receive', '_stdout_reader', '_stderr_reader'):
        assert getattr(t.PlacedProfiledStudyWorker, name) is getattr(ProfiledStudyWorker, name)


def test_only_scientific_quota_changes_and_original_profiles_stay_exact():
    values = prepared()
    original = a.profile(128, route='frozen')
    candidate = a.profile(256, route='research')
    labels = {'profile_id', 'wrapper_id', 'execution_policy', 'iterations'}
    assert {k:v for k,v in original.items() if k not in labels} == {k:v for k,v in candidate.items() if k not in labels}
    assert candidate['iterations'] == 256 and original['iterations'] == 128
    assert candidate['execution_policy']['changed_scientific_fields'] == ['iterations']
    assert candidate['execution_policy']['original_profile'] == values['profiles']['frozen']['128']
    assert a.profile_sha256(128, route='frozen') == a.ORIGINAL_PROFILE_SHAS[128]
    for route in a.ROUTES:
        assert a.profile_sha256(16, route=route) == a.ORIGINAL_PROFILE_SHAS[16]
        assert a.profile(16, route=route) == a.profile(16, route='frozen')
        assert a.native_descriptor(route)['sha256'] == a.SEARCH_BINARY_SHA256
        assert a.native_descriptor(route, 16)['sha256'] == a.SEARCH_BINARY_SHA256
        assert len(a.native_roles(route)) == 2
    assert set(values) == {'profiles', 'native_artifacts_by_route', 'native_roles_by_route'}


@pytest.mark.parametrize('quota,route', [(128,'research'), (256,'frozen'), (64,'frozen'), (64,'research'), (True,'frozen')])
def test_crossed_or_unsupported_quotas_rejected(quota, route):
    with pytest.raises(ValueError): a.profile_id(quota, route=route)
    with pytest.raises(ValueError): a.profile(quota, route=route)


@pytest.mark.parametrize('route', ['low', 'high', 'parity', None])
def test_prior_exploration_routes_rejected(route):
    with pytest.raises(ValueError): a.route_quota(route)


def test_bound_original_native_uses_request_quota_and_fixed_other_settings():
    profile = a.profile(256, route='research')
    descriptor = next(x for x in profile['artifacts'] if x['path'].endswith('native_bridge/src/main.rs'))
    c.verify([descriptor])
    source = (c.ROOT/descriptor['path']).read_text()
    assert 'iterations:u32::try_from(v["iterations"].as_u64().unwrap()).unwrap()' in source
    assert 'agent.cpuct=0.4;' in source and 'agent.world_pool=3;' in source
    assert 'agent.chance_universes=3;' in source and 'agent.inference_batch_size=1;' in source
    assert "'iterations': self.profile['iterations']" in inspect.getsource(Opponent.choose)
    assert not hasattr(a, 'ConfiguredOpponent')


def test_guard_runs_once_before_model_or_output_initialization(monkeypatch):
    calls=[]
    def stop():
        calls.append('guard');raise ValueError('stop before model')
    with patch.object(w, 'verify_runtime_sources', stop), patch.object(a, 'initialize_adapters', side_effect=AssertionError('model forbidden')):
        with pytest.raises(ValueError, match='stop before model'): w.initialize('research')
    assert calls == ['guard']
    tree=ast.walk(ast.parse(inspect.getsource(w.initialize)))
    assert sum(isinstance(n, ast.Call) and isinstance(n.func, ast.Name) and n.func.id=='verify_runtime_sources' for n in tree)==1
    assert 'torch' not in sys.modules


def test_wrong_output_root_rejected_before_models(monkeypatch):
    monkeypatch.setenv('STUDY_WORKER_DIR', str(q.BASELINE_EVIDENCE/'workers/qualification_frozen'))
    with patch.object(w, 'verify_runtime_sources', return_value={}), patch.object(a, 'initialize_adapters', side_effect=AssertionError('model forbidden')):
        with pytest.raises(ValueError, match='evidence boundary'): w.initialize('research')


def test_new_worker_and_phase_bound_to_new_attempt():
    assert t.OBSERVER_MODULE == 'tools.splendorust_search256_strength_20261029_02.worker'
    assert c.PHASE == 'strength'
    assert c.OUT == c.ROOT/'outputs/splendorust_search256_strength_20261029_02'
    assert c.ROUTE_QUOTAS == {'frozen':128,'research':256}
    assert a.profile_path(256).is_relative_to(c.OUT)


@pytest.mark.parametrize('extra', [{'iterations':128}, {'quota':128}, {'cpuct':.8}, {'world_pool':6}])
def test_worker_request_cannot_override_frozen_science(extra):
    with pytest.raises(ValueError, match='allowlist'):
        w._ordinary(dict(route='research', request_started=100., diagnostic_capture=False),
                    dict(player='search', observer={}, **extra))


def test_fixed_corrected_suite_panel_and_parity_inventory():
    plan=q.build_plan()
    assert (plan['scientific_decisions'],plan['graph_cases'],plan['ordinary_requests'],plan['total_parent_requests'])==(182,22,48,52)
    assert set(plan['actual_suites'])=={'128','256'}
    assert all(len(x['counterexamples'])==15 for x in plan['actual_suites'].values())
    for route in a.ROUTES:
        panel=[x for x in plan['panel_requests'] if x['route']==route]
        assert len(panel)==24 and len({x['root_id'] for x in panel})==18 and sum(x['repeat'] for x in panel)==6
        assert all(x['quota']==a.route_quota(route) for x in panel)
    for index, spec in enumerate(plan['panel_requests'][:24]):
        assert spec == c.read(c.ROOT/plan['baseline_parity_sources']['panel'][index]['path'])['spec']


def test_saved_actual_original128_entrypoint_passes_new_checker_without_model():
    # Saved checker fixture only; it does not qualify a new live worker.
    freeze=prepared();plan=q.build_plan()
    folder=q.BASELINE_EVIDENCE/'workers/qualification_frozen'
    receipt=deepcopy(c.read(folder/'entrypoint_clock.json'))
    receipt['result']['execution_schema']=a.EXECUTION_SCHEMA
    proof=q.verify_entrypoint(folder,receipt,plan,freeze)
    assert proof['complete_decisions']==67 and proof['ordinary_handlers']==37
    assert proof['negative_handler_cases']==26 and proof['direct_same_quota']==15
    assert q.compare_baseline_parity(receipt['result'],plan)['complete_scientific_decisions']==67


@pytest.mark.parametrize('mutation', ['callback', 'quota', 'private_repeat', 'cost'])
def test_baseline_parity_checks_all_science_without_quota_normalization(mutation):
    plan=q.build_plan();suite=deepcopy(c.read(c.ROOT/plan['baseline_parity_sources']['entrypoint']['path'])['result'])
    suite['execution_schema']=a.EXECUTION_SCHEMA
    if mutation=='callback':
        target=next(x['results']['search'] for x in suite['fixtures'] if x['results']['search']['diagnostics']['inference_calls'])
        target['diagnostics']['phases'][0]['neural_callbacks'][0]='0'*64
    elif mutation=='quota':suite['challenger_quota']=256
    elif mutation=='private_repeat':suite['private_repeats']={'corrupted':True}
    else:suite['actual_encoded_costs'][0]['corrupted']=True
    with pytest.raises(ValueError):q.compare_baseline_parity(suite,plan)


@pytest.mark.parametrize('quota', [128,256])
def test_actual_work_limits_use_external_quota_and_reject_false_counts(quota):
    native={'turns':0,'phase':{'kind':'Main'}}
    choices=[{'kind':'ReserveDeck','tier':0},{'kind':'ReserveDeck','tier':1}]
    phase=dict(heuristic_phase=False,simulations=quota,leaf_calls=quota+1,
        neural_callbacks=['a'*64]*(quota+1),neural_callback_count=quota+1,neural_callback_rows=quota+1)
    assert s.check_search_phase(native,choices,phase,quota=quota)['simulations']==quota
    with pytest.raises(ValueError):s.check_search_phase(native,choices,phase,quota=128 if quota==256 else 256)
    phase['neural_callback_count']-=1
    with pytest.raises(ValueError):s.check_search_phase(native,choices,phase,quota=quota)


def test_late_completed_moves_preserved_and_performance_labels_actual_quotas():
    for seconds in (.71,1.1,8.99,9.5,10.):
        value=dict(clock='perf_counter_ns',request_id='request-00000003',deadline_seconds=10.,
            elapsed_seconds=seconds,receipt_elapsed_seconds=seconds,received_before_deadline=True,
            accepted=True,status='ok',result={'fixture':True})
        assert s.check_clock(value)==seconds
    value['receipt_elapsed_seconds']=10.01
    with pytest.raises(ValueError):s.check_clock(value)
    assert performance.STAGES==('operational','main')
    assert performance._thresholds([.7,.70001,1.,1.1])['completed_over_one_second']==1
    assert performance._player_summary([], 'research')['quota']==256
    assert performance._player_summary([], 'frozen')['quota']==128
    for role,quota,mode in [('challenger',256,'research'),('opponent',128,'frozen')]:
        row=dict(placement_mode='isolated',player_role=role,arm='search',mode=mode,candidate='research',quota=quota,seat=0)
        performance._require_isolated(row)
        row['quota']=128 if quota==256 else 256
        with pytest.raises(ValueError):performance._require_isolated(row)


def test_four_mib_packet_envelope_handles_conservative_doubled_suite():
    # Packing-only stress: enlarge every callback list in the accepted128 suite,
    # even reference16 and heuristic branches. This is not synthetic science.
    suite=deepcopy(c.read(q.BASELINE_EVIDENCE/'workers/qualification_frozen/entrypoint_clock.json')['result'])
    def expand(value):
        if isinstance(value,dict):
            for key,item in list(value.items()):
                if key in ('neural_callbacks','inference') and isinstance(item,list):value[key]=item+deepcopy(item)
                else:expand(item)
        elif isinstance(value,list):
            for item in value:expand(item)
    expand(suite)
    packet=w.response_packet(dict(type='result',request_id='request-00000002',result=suite))
    assert len(packet.encode()) < w.MAX_RESPONSE_BYTES==4*c.MiB
    with pytest.raises(ValueError):w.response_packet({'oversize':'x'*(4*c.MiB)})


def test_no_configured_native_side_channel_or_predecessor_mutation(tmp_path):
    for route in a.ROUTES:
        assert a.verify_native_configuration_receipts(tmp_path,route)['configured_native'] is False
    (tmp_path/'native_configuration_contract.json').write_text('{}')
    with pytest.raises(ValueError):a.verify_native_configuration_receipts(tmp_path,'research')
    old_freeze=c.read(q.BASELINE_EVIDENCE/'freeze.json')
    names={str(Path(m.__file__).name) for m in (old_a,old_w,old_t,old_placement)}
    prior=[x for x in old_freeze['inputs'] if x['path'].startswith('tools/splendorust_exploration128_20261029_02/') and Path(x['path']).name in names]
    assert len(prior)==len(names)
    c.verify(prior)
