"""Model-free quota, actual saved science, startup and immutable-network checks."""
import ast
from copy import deepcopy
import inspect
import json
from pathlib import Path
from types import SimpleNamespace
import pytest

from tools.splendorust_search512_20261101 import common as c,adapters as a,worker as w,corpus,scientific as s,qualification as q
from tools.splendorust_search256_recovery_20261030_02 import adapters as old_a,worker as old_w,qualification as old_q
from tools.external_strength_screen_20261018 import rules
from tools.future_training_rules_20261021 import decode_training_snapshot,SOURCE_HASHES

PRIOR=c.ROOT/'outputs/splendorust_network_live_20261031_01'
CONTROL=PRIOR/'workers/live_control'
GRAPH_CONTROL=c.OLD_OUT/'workers/qualification_research'


def node(source,name):
    tree=ast.parse(inspect.getsource(source))
    return ast.dump(next(x for x in tree.body if isinstance(x,(ast.FunctionDef,ast.ClassDef)) and x.name==name),include_attributes=False)


def declared(name='main_b000_s000_candidate',route='candidate',ordinary=10000):
    return {name:dict(route=route,quota=a.route_quota(route),startup_requests=2,
        graph_request_id='request-00000001',entrypoint_request_id='request-00000002',
        ordinary_first_request_id='request-00000003',ordinary_requests=ordinary)}


def control_freeze():
    return dict(profiles={'control':{'16':c.artifact(a.profile_path(16)),'256':c.artifact(a.profile_path(256))}})


def test_only_supported_quota_routes_and_original_reference():
    assert a.ROUTES==('control','candidate') and a.QUOTAS==(256,512)
    assert [a.route_quota(r) for r in a.ROUTES]==[256,512]
    for quota in (16,256,512):assert a.pool_for(quota)==3
    assert a.profile_id(256)==old_a.profile_id(256,route='research')
    assert a.profile_id(16)==old_a.profile_id(16,route='research')
    assert a.route_from_profile_id(a.profile_id(512),512)=='candidate'
    assert a.route_from_profile_id(a.profile_id(256),256)=='control'


@pytest.mark.parametrize('value',[True,False,0,64,128,1024,512.,'512'])
def test_invalid_quota_cannot_enter_profile_or_scientific_path(value):
    with pytest.raises(ValueError):a.pool_for(value)
    with pytest.raises(ValueError):s._expected_quota('search',value)


def test_profiles_original_bytes_and_only_declared_scientific_delta():
    control=a.profile(256,route='control');candidate=a.profile(512,route='candidate')
    assert control==old_a.profile(256,route='research')==c.read(a.ORIGINAL256_PROFILE)
    assert a.profile_sha256(256,route='control')==a.ORIGINAL256_PROFILE_SHA
    assert a.profile(16,route='candidate')==old_a.profile(16,route='research')
    left=deepcopy(control);right=deepcopy(candidate)
    for key in ('profile_id','wrapper_id','execution_policy','iterations'):left.pop(key);right.pop(key)
    assert left==right and (control['iterations'],candidate['iterations'])==(256,512)
    assert candidate['execution_policy']['changed_scientific_fields']==['iterations']
    assert candidate['execution_policy']['original_profile']==c.artifact(a.ORIGINAL256_PROFILE)
    assert candidate['execution_policy']['numerical_implementation_unchanged'] is True
    assert a.profile_path(512)==c.OUT/'profiles/puct512.json'
    assert a.native_descriptor('control')==a.native_descriptor('candidate')
    assert a.native_descriptor('candidate',16)==a.native_descriptor('control',16)
    assert control['world_pool']==candidate['world_pool']==3 and control['cpuct']==candidate['cpuct']==.4


def test_route_cannot_select_its_opponents_quota_or_profile():
    for route,quota in (('control',512),('candidate',256),('frozen',256),('research',512)):
        with pytest.raises(ValueError):a.profile_id(quota,route=route)
    with pytest.raises(ValueError):a.route_from_profile_id(a.profile_id(256),512)


@pytest.mark.parametrize('route',['control','candidate'])
def test_exact_declared_worker_folder_and_prefix(route):
    name='main_b003_s036_'+route;freeze={'declared_worker_lifecycles':declared(name,route)}
    assert w.lifecycle_route(c.OUT/'workers'/name,freeze,route=route)==route
    with pytest.raises(ValueError):w.lifecycle_route(c.OUT/'workers'/name,freeze,route='candidate' if route=='control' else 'control')
    with pytest.raises(ValueError):w.lifecycle_route(PRIOR/'workers'/name,freeze,route=route)
    with pytest.raises(ValueError):w.lifecycle_route(c.OUT/'workers'/'undeclared',freeze,route=route)


@pytest.mark.parametrize('key,value',[('quota',256),('quota',True),('startup_requests',3),
    ('graph_request_id','request-00000002'),('entrypoint_request_id','request-00000003'),
    ('ordinary_first_request_id','request-00000004'),('ordinary_requests',-1),('ordinary_requests',True),
    ('ordinary_requests',812001)])
def test_lifecycle_rejects_changed_bounds(key,value):
    planned=declared();name=next(iter(planned));planned[name][key]=value
    with pytest.raises(ValueError):w.lifecycle_route(c.OUT/'workers'/name,{'declared_worker_lifecycles':planned})


def test_source_guard_is_once_before_any_model_or_profile_load(monkeypatch):
    calls=[]
    monkeypatch.setattr(w,'verify_runtime_sources',lambda:calls.append('guard') or SOURCE_HASHES)
    with pytest.raises(ValueError,match='route'):w.initialize('unknown')
    assert calls==['guard']
    function=next(n for n in ast.parse(inspect.getsource(w)).body if isinstance(n,ast.FunctionDef) and n.name=='initialize')
    calls=[n for n in ast.walk(function) if isinstance(n,ast.Call) and isinstance(n.func,ast.Name) and n.func.id=='verify_runtime_sources']
    assert len(calls)==1 and calls[0] is function.body[0].value


def test_no_network_rewrite_or_precision_change_and_original_scientific_bodies():
    assert node(w,'_ordinary')==node(old_w,'_ordinary')
    assert node(a,'BaselineOpponent')==node(old_a,'BaselineOpponent')
    assert node(a,'initialize_adapters')==node(old_a,'initialize_adapters')
    assert node(q,'verify_entrypoint')==node(old_q,'verify_entrypoint')
    text=inspect.getsource(w)
    assert 'candidate_final_token' not in text and 'build_candidate' not in text and 'install_execution' not in text
    assert 'numerical_checks' not in [n.name for n in ast.parse(text).body if isinstance(n,ast.FunctionDef)]


def test_corpus_and_scientific_changes_are_only_explicit_quota_allowlists():
    base=c.ROOT/'tools/splendorust_search256_recovery_20261030_02'
    expected=(base/'corpus.py').read_text().replace('def build_plan(quota=128):','def build_plan(quota=256):')
    expected=expected.replace('(128, 256)','(256, 512)').replace('challenger quota128or256','challenger quota256or512')
    expected=expected.replace('if quota in (128,256):','if quota in (256,512):')
    expected=expected.replace("execution_schema='splendorust_search256_strength_execution_v1'","execution_schema='splendorust_search512_strength_execution_v1'")
    assert inspect.getsource(corpus)==expected
    expected=(base/'scientific.py').read_text().replace('(128, 256)','(256, 512)').replace('(16, 128, 256)','(16, 256, 512)')
    assert inspect.getsource(s)==expected


def test_complete_actual_counterexample_plan_both_real_quotas():
    plan=q.build_plan()
    assert set(plan['actual_suites'])=={'256','512'} and plan['scientific_decisions']==134
    assert plan['ordinary_requests']==0 and plan['total_parent_requests']==4 and plan['graph_cases']==22
    assert plan['per_worker']==dict(graph_cases=11,scientific_decisions=67,ordinary_requests=0,parent_requests=2)
    for quota in (256,512):
        suite=plan['actual_suites'][str(quota)]
        assert suite['challenger_quota']==quota and suite['reference_quota']==16
        assert len(suite['counterexamples'])==15 and suite['actual_complete_decisions']['total']==52
        assert suite['worker_rejection_attempts']==26 and suite['native_operations']['total']==82
    assert plan['actual_suites']['256']==c.read(c.OLD_OUT/'qualification_plan.json')['actual_suites']['256']


def current_control_suite():
    receipt=c.read(CONTROL/'entrypoint_clock.json')
    receipt['result']['execution_route']='control';receipt['result']['execution_schema']=a.EXECUTION_SCHEMA
    return receipt


def test_saved_original256_full_suite_parity_and_actual_forensics():
    # This only rechecks already saved evidence, never qualifies a new live worker.
    plan=q.build_plan();receipt=current_control_suite()
    assert q.compare_control_parity(receipt['result'],plan)['complete_scientific_decisions']==67
    result=q.verify_entrypoint(CONTROL,receipt,plan,control_freeze())
    assert result['complete_decisions']==67 and result['ordinary_handlers']==37 and result['negative_handler_cases']==26
    assert result['native_numeric_parity']==46 and result['direct_same_quota']==15 and result['independent_saved_transitions']==37


@pytest.mark.parametrize('fault',['action','callback','quota','reference','route','schema'])
def test_control_parity_cannot_hide_changed_science(fault):
    suite=current_control_suite()['result']
    if fault=='action':suite['fixtures'][0]['results']['search']['action']={}
    elif fault=='callback':suite['fixtures'][0]['results']['search']['diagnostics']['inference_calls']+=1
    elif fault=='quota':suite['challenger_quota']=512
    elif fault=='reference':suite['reference_quota']=256
    elif fault=='route':suite['execution_route']='candidate'
    else:suite['execution_schema']='legacy'
    with pytest.raises(ValueError):q.compare_control_parity(suite,q.build_plan())


def test_actual_saved_graph_full_arrays_bits_and_historical_anchors():
    freeze=c.read(c.OLD_OUT/'freeze.json');result=c.read(GRAPH_CONTROL/'graph_parity_clock.json')['result']
    proof=w.verify_graph_admission(GRAPH_CONTROL,freeze,result)
    assert proof['accepted'] and proof['fixtures']==11


def test_practical_tolerance_wrapper_is_not_an_exact_graph_admission():
    with pytest.raises(ValueError,match='Exact graph parity gate'):
        w.verify_graph_admission(CONTROL,c.read(PRIOR/'freeze.json'),c.read(CONTROL/'graph_parity_clock.json')['result'])


@pytest.mark.parametrize('fault',['bits','fixture','summary'])
def test_graph_readback_rejects_forged_arrays_or_identity(fault,tmp_path):
    saved=c.read(GRAPH_CONTROL/'graph_parity.json');result=deepcopy(saved['summary'])
    if fault=='bits':saved['evidence'][0]['candidate_outputs'][0][0]+=1e-5
    elif fault=='fixture':saved['evidence'][0]['fixture_id']='replacement'
    else:saved['summary']['passed']=False
    (tmp_path/'graph_parity.json').write_text(json.dumps(saved))
    with pytest.raises(ValueError):w.verify_graph_admission(tmp_path,c.read(c.OLD_OUT/'freeze.json'),result)


def test_512_forced_pass_validates_real_corrected_support_without_a_model(monkeypatch):
    fixture=next(x for x in corpus.build_fixtures() if x['id']=='forced_pass')
    observer=rules.observation_payload(decode_training_snapshot(fixture['snapshot']))
    result=deepcopy(next(x for x in current_control_suite()['result']['fixtures'] if x['id']=='forced_pass')['results']['search'])
    result.update(simulations_quota=512,profile_id=a.profile_id(512),wrapper_id=a.wrapper_id(512),profile_sha256='f'*64)
    result['diagnostics']['profile_id']=a.profile_id(512)
    monkeypatch.setattr(a,'profile_sha256',lambda quota,**kwargs:'f'*64)
    assert s.validate_result(result,observer,'search',expected_quota=512)
    with pytest.raises(ValueError):s.validate_result(result,observer,'search',expected_quota=256)
    result['simulations_quota']=256
    with pytest.raises(ValueError):s.validate_result(result,observer,'search',expected_quota=512)


def test_startup_order_cannot_be_bypassed():
    context=dict(graph_parity_passed=False)
    with pytest.raises(ValueError,match='graph parity'):w.handle(context,{'player':'search','observer':{}})
    context.update(graph_parity_passed=True,entrypoint_passed=False,diagnostic_capture=False)
    with pytest.raises(ValueError,match='qualification before play'):w.handle(context,{'player':'search','observer':{}})
    context.update(entrypoint_attempted=True)
    with pytest.raises(ValueError,match='Single actual'):w.handle(context,{'operation':'entrypoint_checks'})


def test_actual_suite_callback_and_negative_handlers_use_same_ordinary_entrypoint(tmp_path,monkeypatch):
    calls=[];instance=SimpleNamespace(admission_guard=None,request_deadline=None)
    context=dict(folder=tmp_path,adapters={'search':instance,'reference':instance},route='candidate',
        graph_parity_passed=True,entrypoint_passed=False,entrypoint_attempted=False,diagnostic_capture=False,
        boundary_handle_active=False,component_profiles=[],request_started=10.,quota=512)
    class Stream:
        def close(self):calls.append('closed')
    monkeypatch.setattr(w,'_forensic_recorder',lambda _: (lambda e:None,Stream()))
    import psutil
    monkeypatch.setattr(psutil,'virtual_memory',lambda:SimpleNamespace(available=4*1024**3))
    def ordinary(ctx,request):
        calls.append(request)
        if request.get('player')!='search':raise ValueError('negative fixture')
        return dict(simulations_quota=512)
    monkeypatch.setattr(w,'_ordinary',ordinary)
    def suite(adapters,*,request_callback,boundary,evidence_callback,route):
        assert route=='candidate' and context['diagnostic_capture']
        assert request_callback({'player':'search','observer':{}})==dict(simulations_quota=512)
        with pytest.raises(ValueError,match='negative fixture'):request_callback({'player':'invalid','observer':{}})
        return dict(counts={'complete_decisions':67})
    monkeypatch.setattr(corpus,'run_entrypoint_checks',suite)
    assert w.handle(context,{'operation':'entrypoint_checks'})['counts']['complete_decisions']==67
    assert context['entrypoint_passed'] and context['entrypoint_attempted'] and not context['diagnostic_capture']
    assert context['quota']==512 and context['request_started']==10. and instance.admission_guard is None
    assert len(context['component_profiles'])==2 and context['component_profiles'][1]['handle_failed']


def test_response_capacity_is_symmetric_eight_mib_and_bounded():
    assert w.MAX_RESPONSE_BYTES==8*1024**2
    assert len(w.response_packet({'data':'x'*(4*1024**2)}))>4*1024**2
    with pytest.raises(ValueError,match='8MiB'):w.response_packet({'data':'x'*(8*1024**2)})


def test_adapter_does_not_publish_startup_requests_out_of_order():
    with pytest.raises(ValueError,match='sole first'):
        a.graph_parity(SimpleNamespace(_counter=1),None,None,request_fn=None,collect_fn=None)
    with pytest.raises(ValueError,match='sole second'):
        a.actual_entrypoint(SimpleNamespace(_counter=0),None,None,request_fn=None,collect_fn=None)


def test_ordinary_request_is_only_legal_observer_and_search_arm():
    observer={'lawful':'fixture'}
    assert a.ordinary_payload({'variant':'candidate','quota':512},observer)==dict(player='search',observer=observer)


@pytest.mark.parametrize('fault',[None,'quota','lifecycle','graph','weights','alternative'])
def test_execution_identity_binds_original_graph_and_exact_lifetime(tmp_path,monkeypatch,fault):
    monkeypatch.setattr(c,'OUT',tmp_path)
    name='main_b000_s000_candidate';folder=tmp_path/'workers'/name;folder.mkdir(parents=True)
    graph=dict(weights_before_sha256='a'*64,weights_after_sha256='a'*64)
    freeze=dict(declared_worker_lifecycles=declared(name),unchanged_graph_backend_identity={
        'graph_identity_sha256':c.digest(graph),'weights_sha256':'a'*64})
    metadata={'route':'candidate'}
    context=dict(folder=folder,route='candidate',quota=512,component_backend_identity=c.digest(graph))
    execution=w.execution_identity(context,metadata)
    if fault=='quota':execution['quota']=256
    elif fault=='lifecycle':execution['worker_lifecycle']='other_candidate'
    elif fault=='graph':freeze['unchanged_graph_backend_identity']['graph_identity_sha256']='0'*64
    elif fault=='weights':freeze['unchanged_graph_backend_identity']['weights_sha256']='b'*64
    elif fault=='alternative':(folder/'candidate_identity.json').write_text('{}')
    (folder/'graph_identity.json').write_text(json.dumps(graph))
    (folder/'search_execution_identity.json').write_text(json.dumps(execution))
    if fault is None:assert w.verify_execution_identity(folder,freeze,metadata)['original_full_network']
    else:
        with pytest.raises(ValueError):w.verify_execution_identity(folder,freeze,metadata)


def test_ready_metadata_keeps_original_network_and_exact_route_profile_hashes():
    profiles={'control':{'16':{'sha256':'a'*64},'256':{'sha256':'b'*64}},
        'candidate':{'16':{'sha256':'a'*64},'512':{'sha256':'c'*64}}}
    for route,quota in (('control',256),('candidate',512)):
        value=a.ready_metadata(route,{'profiles':profiles},pid=1,native_pids={'search':2,'reference':3},sources=SOURCE_HASHES,freeze_sha256='d'*64)
        assert value['route']==route and value['request_quotas']==[quota] and value['qualification_reference_quota']==16
        assert value['world_pools']==[3] and value['cpuct']==.4 and value['threads']==1
        assert value['execution_profiles']=={key:item['sha256'] for key,item in profiles[route].items()}
        assert value['inference_backend']=='torchscript_frozen_exact_forward_v1'
        assert value['seconds_per_move']==10. and value['internal_seconds']==9.
