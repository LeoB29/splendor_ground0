"""Model-free research routing, native additive diff and failure regressions."""
import ast
from copy import deepcopy
import hashlib
import json
from pathlib import Path
from types import SimpleNamespace

import pytest

from tools.splendorust_decision_quality_20261027 import adapters as a, qualification as q, worker as w, native_build as nb
from tools.splendorust_search_scale_20261024 import worker as frozen_worker, common as oldc
from tools.splendorust_search_scale_20261024 import qualification as oldq
from tools.external_strength_screen_20261018 import rules
from tools.future_training_rules_20261021 import decode_training_snapshot

ROOT=Path(__file__).resolve().parents[1]

def root():
    return json.loads((ROOT/'outputs/decision_quality_preparation_20261027_01/panel.json').read_text())['roots'][0]

def context(tmp_path=None):
    return dict(adapters={},profiles={},profile_artifacts={},folder=tmp_path,quota=64,
        diagnostic_capture=False,request_started=None,component_profiles=[],boundary_handle_active=False,
        graph_parity_passed=True,component_backend='research_graph',component_backend_identity='graph',
        request_read_complete_ns=None)

def test_additive_bridge_only_two_substitutions_and_core_files_exact():
    original=(nb.SOURCE/'bridge/src/main.rs').read_text()
    assert (nb.BUILD/'bridge/src/main.rs').read_text()==nb.bridge_source(original)
    assert 'agent.capture_search_targets=' not in original
    for source in (nb.SOURCE/'crates').rglob('*'):
        if source.is_file(): assert source.read_bytes()==(nb.BUILD/source.relative_to(nb.SOURCE)).read_bytes()
    receipt=json.loads((nb.BUILD/'build_receipt.json').read_text())
    assert not receipt['model_inference_executed'] and not receipt['search_executed']

def test_native_transform_rejects_unknown_source():
    with pytest.raises(ValueError,match='substitution'): nb.bridge_source('different source')

@pytest.mark.parametrize('label',['A','B','C'])
def test_seed_is_canonical_lawful_observer_only(label):
    observer=root()['observer']
    shuffled=dict(reversed(list(observer.items())))
    assert a.observer_seed(observer,label)==a.observer_seed(shuffled,label)
    encoded=json.dumps(observer,sort_keys=True,separators=(',',':'),allow_nan=False)
    expected=hashlib.sha256(encoded.encode()).hexdigest() if label=='A' else hashlib.sha256((a.DOMAIN+'|'+label+'|'+encoded).encode()).hexdigest()
    assert a.observer_seed(observer,label)==int(expected[:16],16)

def test_three_seed_families_are_distinct():
    assert len({a.observer_seed(root()['observer'],label) for label in ('A','B','C')})==3

@pytest.mark.parametrize('label',['D','a',0,None,True])
def test_invalid_seed_family_rejected(label):
    with pytest.raises(ValueError):a.observer_seed(root()['observer'],label)

def test_exact_fixed_qualification_inventory():
    plan=q.build_plan()
    assert plan['complete_decisions']==123 and plan['graph_cases']==22 and plan['global_graph_cases']==44
    assert plan['maximum_initialized_workers']==4 and plan['maximum_simultaneous_worker_trees']==1
    assert len(plan['parity_requests'])==19 and plan['parity_requests'][0]['route']=='frozen'
    assert {(x['quota'],x['seed_label'],x['telemetry']) for x in plan['parity_requests'][1:]}=={
        (quota,seed,capture) for quota in (64,256,1024) for seed in 'ABC' for capture in (False,True)}
    assert {x['root_id'] for x in plan['parity_requests']}=={'root_00'}
    assert plan['original_suite']['actual_complete_decisions']['total']==52

@pytest.mark.parametrize('quota',[16,64,256,1024])
def test_profiles_change_only_declared_research_choices(quota):
    p=a.profile(quota)
    assert p['iterations']==quota and p['internal_budget_seconds']==29
    assert p['checkpoint']==oldc.profile_for(16)['checkpoint']
    assert p['model_definition']==oldc.profile_for(16)['model_definition']
    assert p['native_binary_sha256']==json.loads((nb.BUILD/'build_receipt.json').read_text())['executable']['sha256']
    assert p['profile_id']==a.profile_id(quota)

def test_ordinary_reference_quota_is_rejected():
    with pytest.raises(ValueError):q.ordinary_payload(dict(route='research',quota=16,seed_label='A'),root()['observer'])
    with pytest.raises(ValueError,match='arm/quota'):
        w._ordinary(context(),dict(player='reference',observer=root()['observer'],quota=16,seed_label='A',telemetry=True))

def test_actual_decode_rejects_all_original26_cases_without_model():
    private=next(x for x in q.build_plan()['original_suite']['counterexamples'] if x['id']=='private_reserves')
    state=decode_training_snapshot(private['snapshot']);ctx=context();ctx['diagnostic_capture']=True
    def route(payload):
        return w.handle(ctx,dict(payload,quota=16 if payload.get('player')=='reference' else 64,seed_label='A',telemetry=True))
    rejected=oldq.worker_rejections(route,state)
    assert len(rejected)==26 and len(ctx['component_profiles'])==26
    assert all(x['handle_failed'] for x in ctx['component_profiles'])

def test_boundary_failure_is_truthful_and_does_not_change_frozen_module(monkeypatch):
    before=frozen_worker.handle;ctx=context()
    def fail(*args):raise TimeoutError('specific original failure')
    monkeypatch.setattr(w,'_ordinary',fail)
    with pytest.raises(TimeoutError,match='specific'):w.handle(ctx,{})
    assert ctx['component_profiles'][0]['handle_failed']
    assert frozen_worker.handle is before and not ctx['boundary_handle_active']

def test_bundled_callback_reentry_preserves_37handlers_26rejects15direct(monkeypatch,tmp_path):
    ctx=context(tmp_path);before=frozen_worker.handle
    def ordinary(c,request):
        if 'reject' in request:raise ValueError('synthetic reject')
        return {'simulations_quota':64}
    monkeypatch.setattr(w,'_ordinary',ordinary)
    def checks(c,callback,**kwargs):
        for n in range(37):callback({'case':n})
        for n in range(26):
            with pytest.raises(ValueError):callback({'reject':n})
        return {'counts':{'complete_decisions':37+15}}
    monkeypatch.setattr(q,'run_entrypoint_checks',checks)
    result=w.handle(ctx,{'operation':'entrypoint_checks'})
    assert result['counts']['complete_decisions']==52 and len(ctx['component_profiles'])==63
    assert sum(x['handle_failed'] for x in ctx['component_profiles'])==26
    assert frozen_worker.handle is before and ctx['request_started'] is None and not ctx['diagnostic_capture']

def test_inference_instance_dispatch_preserves_float_values_and_hashes():
    instance=a.ResearchOpponent.__new__(a.ResearchOpponent)
    outputs=[[float(i)/8 for i in range(83)]];tokens=[0.0]*1488
    instance.active_deadline=None;instance.capture=True;instance.callbacks=[];instance.callback_bits=[]
    instance.forward=SimpleNamespace(infer=lambda adapter,t,b:deepcopy(outputs))
    assert instance._infer(tokens,1)==outputs
    record=instance.callbacks[0]
    assert record['tokens']==tokens and record['outputs']==outputs and record['output83']==outputs[0]
    assert record['callback_sha256']==a.digest([a.digest(tokens),a.digest(outputs),1])
    assert instance.callback_bits==[dict(input_float32_sha256=a._bits(tokens),output_float32_sha256=a._bits(outputs[0]))]

def test_deadline_fails_before_synthetic_inference(monkeypatch):
    instance=a.ResearchOpponent.__new__(a.ResearchOpponent);instance.active_deadline=1
    instance.forward=SimpleNamespace(infer=lambda *args:pytest.fail('late forward'))
    monkeypatch.setattr(a.time,'perf_counter',lambda:2)
    with pytest.raises(TimeoutError,match='29.000'):instance._infer([],1)

def test_research_packet_cap_and_nonfinite_reject():
    assert json.loads(w.response_packet({'value':'safe'}))=={'value':'safe'}
    with pytest.raises(ValueError,match='4MiB'):w.response_packet({'huge':'x'*w.MAX_RESPONSE_BYTES})
    with pytest.raises(ValueError):w.response_packet({'value':float('nan')})

def test_startup_guard_precedes_model_and_no_global_inference_patch():
    tree=ast.parse(Path(w.__file__).read_text())
    init=next(x for x in tree.body if isinstance(x,ast.FunctionDef) and x.name=='initialize')
    calls=[x for x in ast.walk(init) if isinstance(x,ast.Call)]
    guards=[x for x in calls if ast.unparse(x.func)=='verify_runtime_sources']
    assert len(guards)==1 and guards[0].lineno<min(x.lineno for x in calls if ast.unparse(x.func)=='adapters.initialize_adapters')
    source=Path(w.__file__).read_text()+Path(a.__file__).read_text()
    assert 'InferenceBinding().install' not in source and 'frozen.handle =' not in source and 'Opponent._infer =' not in source

def test_projection_only_declared_identity_does_not_modify_raw():
    result={k:None for k in q.original_audit.RESULT_FIELDS}
    result.update(simulations_quota=64,diagnostics={'profile_id':'research','observer_seed':7},
        research={'seed_label':'A','seed':7},action={'test':1})
    before=deepcopy(result);projected=q.frozen_projection(result)
    assert result==before and projected['action']==before['action']
    assert projected['profile_id']==oldc.profile_id(64) and projected['diagnostics']['observer_seed']==7
    result['simulations_quota']=1024
    with pytest.raises(ValueError):q.frozen_projection(result)

def test_core_retains_work_and_callback_identity():
    value=dict(profile_id='id',profile_sha256='sha',wrapper_id='wrapper',research={'other':1},
        action={'x':1},diagnostics={'profile_id':'id','observer_seed':7,'phases':[
            {'simulations':64,'neural_callbacks':['abc'],'inference':['capture']} ]})
    result=q.scientific_core(value)
    assert result['diagnostics']['phases']==[{'simulations':64,'neural_callbacks':['abc']}]
    assert result['diagnostics']['observer_seed']==7 and result['action']==value['action']

def test_finite_saved_original_entrypoint_fixture_verifier():
    base=ROOT/'outputs/splendorust_isolated_reliability_20261026_01'
    receipt=json.loads((base/'entrypoint_checks_64.json').read_text())
    result=q.verify_entrypoint(base/'workers/qualification_64',receipt,q.build_plan(),{})
    assert result['complete_decisions']==52 and result['ordinary_handlers']==37
    assert result['negative_handler_cases']==26 and result['native_numeric_parity']==46
    assert result['model_calls']==0 and not result['qualification_rerun']

def test_entrypoint_forensic_chain_corruption_fails_before_science(tmp_path):
    (tmp_path/'entrypoint_forensics.jsonl').write_text(json.dumps({'sequence':1,'previous_sha256':None,'event':{}})+'\n')
    suite={'status':'entrypoint_counterexamples_passed','challenger_quota':64,'reference_quota':16,
           'counts':{'complete_decisions':52}}
    with pytest.raises(ValueError,match='chain/sequence'):
        q.verify_entrypoint(tmp_path,{'status':'ok','result':suite},{},{})

def raw_annotation_fixture():
    fixture=next(x for x in q.build_plan()['original_suite']['counterexamples'] if x['id']=='forced_pass')
    observer=rules.observation_payload(decode_training_snapshot(fixture['snapshot']))
    seed=a.observer_seed(observer,'A')
    result=dict(simulations_quota=64,diagnostics=dict(observer_seed=seed,phases=[]),research=dict(
        schema=a.SCHEMA,quota=64,seed_label='A',seed=seed,telemetry_enabled=True,
        internal_seconds=29.,native_binary_sha256='1'*64,phases=[]))
    return observer,result,{'research_native_sha256':'1'*64}

def test_raw_qualification_annotation_identity_and_empty_forced_phase():
    observer,result,freeze=raw_annotation_fixture()
    q.validate_raw_qualification_annotations(result,observer,freeze)

@pytest.mark.parametrize('field,value',[('schema','wrong'),('quota',16),('seed_label','B'),('seed',0),
    ('telemetry_enabled',False),('internal_seconds',.9),('native_binary_sha256','2'*64)])
def test_raw_qualification_annotation_corruption_rejected(field,value):
    observer,result,freeze=raw_annotation_fixture();result['research'][field]=value
    with pytest.raises(ValueError,match='identity/clock/native/capture'):
        q.validate_raw_qualification_annotations(result,observer,freeze)

def test_raw_qualification_phase_inventory_corruption_rejected():
    observer,result,freeze=raw_annotation_fixture();result['research']['phases']=[{}]
    with pytest.raises(ValueError,match='phase inventory'):
        q.validate_raw_qualification_annotations(result,observer,freeze)

