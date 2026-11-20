"""Saved-failure and fake-runtime checks; no checkpoint or model invocation."""
import ast
from copy import deepcopy
import inspect
import json
from pathlib import Path
import struct
import subprocess
import sys
import os
from types import SimpleNamespace

import pytest

from tools.splendorust_network_live_20261031 import common as c,worker as w
from tools.splendorust_inference_opt_20261025 import graph


ROOT=Path(__file__).resolve().parents[1]
SAVED=ROOT/'outputs/splendorust_network_execution_20261031_01/workers/candidate/graph_parity.json'


def saved():return json.loads(SAVED.read_text())


def f32(value):return struct.unpack('<f',struct.pack('<f',value))[0]


def fixture():return dict(fixture_id='test',batch=1,tokens=[0.]*1488)


def outputs():return [[1.]*81+[.5,-.5]]


def lifecycle_plan():
    return {name:dict(execution_variant=variant,route='research',startup_requests=3,
        graph_request_id='request-00000001',numerical_request_id='request-00000002',entrypoint_request_id='request-00000003',
        ordinary_first_request_id='request-00000004',ordinary_requests=c.MAX_GAMES*c.MAX_PLIES//2 if name.startswith('live_') else c.MAX_REPLAY_DECISIONS)
        for name,variant in w.WORKER_LIFECYCLES.items()}


def test_saved_exactness_failure_is_not_relabelled_as_exact():
    document=saved();freeze={'numerical_tolerances':dict(w.TOLERANCES)}
    candidate=w.summarize_admission('graph_parity',document['fixtures'],document['evidence'],
        document['summary'],'candidate',freeze)
    control=w.summarize_admission('graph_parity',document['fixtures'],document['evidence'],
        document['summary'],'control',freeze)
    assert candidate['accepted'] and candidate['status']=='passed'
    assert not candidate['exact_summary']['passed'] and all(not x['comparison']['float32_bits_equal'] for x in candidate['rows'])
    assert not control['accepted'] and control['status']=='numerically_rejected'
    assert sum(x['policy']['unequal_float32_bits']+x['value']['unequal_float32_bits'] for x in candidate['rows'])==1425
    assert candidate['decisions_compared'] is False


@pytest.mark.parametrize('index,delta,accepted',[(0,2**-17,True),(0,2**-16,False),(81,2**-21,True),(81,2**-19,False)])
def test_frozen_policy_and_value_limits_are_separate(index,delta,accepted):
    eager=outputs();candidate=deepcopy(eager);candidate[0][index]=f32(eager[0][index]+delta)
    result=w.assess_outputs(fixture(),eager,candidate,'candidate')
    assert result['accepted'] is accepted
    assert w.assess_outputs(fixture(),eager,candidate,'control')['accepted'] is False


def test_policy_relative_limit_uses_reference_not_observed():
    eager=outputs();eager[0][0]=1024.;candidate=deepcopy(eager);candidate[0][0]=f32(1024.+.0009765625)
    row=w.assess_outputs(fixture(),eager,candidate,'candidate')
    assert row['accepted'] and row['policy']['maximum_error_to_allowed_ratio']==pytest.approx(.0009765625/(1e-5+1e-6*1024.))


def test_signed_zero_remains_an_exact_control_difference():
    eager=outputs();candidate=deepcopy(eager);eager[0][0]=0.;candidate[0][0]=-0.
    assert w.assess_outputs(fixture(),eager,candidate,'candidate')['accepted']
    row=w.assess_outputs(fixture(),eager,candidate,'control')
    assert not row['accepted'] and row['comparison']['exact_float_equal'] and not row['comparison']['float32_bits_equal']


@pytest.mark.parametrize('mutation',['nan','infinity','integer','double_only','missing','extra','wrong_batch'])
def test_malformed_numerical_outputs_are_errors_not_valid_rejections(mutation):
    eager=outputs();candidate=deepcopy(eager)
    if mutation=='nan':candidate[0][0]=float('nan')
    elif mutation=='infinity':candidate[0][0]=float('inf')
    elif mutation=='integer':candidate[0][0]=1
    elif mutation=='double_only':candidate[0][0]=1.+1e-12
    elif mutation=='missing':candidate[0].pop()
    elif mutation=='extra':candidate[0].append(0.)
    else:candidate.append(candidate[0][:])
    with pytest.raises(ValueError):w.assess_outputs(fixture(),eager,candidate,'candidate')


@pytest.mark.parametrize('mutation',['policy_atol','policy_rtol','value_atol','value_rtol','missing'])
def test_no_tolerance_relaxation_or_runtime_defaults(mutation):
    tolerances=dict(w.TOLERANCES)
    if mutation=='missing':tolerances.pop('value_rtol')
    else:tolerances[mutation]+=1e-8
    with pytest.raises(ValueError):w.numerical_contract({'numerical_tolerances':tolerances})


def numerical_fixtures():
    values=[dict(fixture_id=f'fresh_{i}',batch=1,tokens=[float(i)]+[0.]*1487) for i in range(36)]
    for batch in (2,8):values.append(dict(fixture_id=f'batch{batch}',batch=batch,tokens=sum([f['tokens'] for f in values[:batch]],[])))
    return values


@pytest.mark.parametrize('mutation',[None,'duplicate_input','duplicate_id','bad_shape','bad_batch','nonfinite'])
def test_exact_fresh_numerical_inventory(mutation):
    fixtures=numerical_fixtures()
    if mutation=='duplicate_input':fixtures[1]['tokens']=fixtures[0]['tokens'][:]
    elif mutation=='duplicate_id':fixtures[1]['fixture_id']=fixtures[0]['fixture_id']
    elif mutation=='bad_shape':fixtures[0]['tokens'].pop()
    elif mutation=='bad_batch':fixtures[-1]['batch']=2
    elif mutation=='nonfinite':fixtures[0]['tokens'][0]=float('nan')
    if mutation:
        with pytest.raises(ValueError):w.validate_numerical_fixtures(fixtures)
    else:assert w.validate_numerical_fixtures(fixtures)==fixtures


def setup_saved(tmp_path,monkeypatch,variant='candidate'):
    monkeypatch.setattr(c,'ROOT',tmp_path);monkeypatch.setattr(c,'OUT',tmp_path/'out')
    folder=c.OUT/'workers'/('live_'+variant);folder.mkdir(parents=True)
    document=saved();freeze=dict(declared_worker_lifecycles=lifecycle_plan(),numerical_tolerances=dict(w.TOLERANCES),graph_parity_fixtures=document['fixtures'],
        unchanged_graph_backend_identity={'graph_identity_sha256':document['summary']['graph_identity_sha256']})
    result=w.summarize_admission('graph_parity',document['fixtures'],document['evidence'],document['summary'],variant,freeze)
    c.save(folder/'graph_parity.json',document);c.save(folder/'graph_admission.json',result)
    return folder,freeze,document,result


@pytest.mark.parametrize('variant',['control','candidate'])
def test_saved_verifier_returns_valid_negative_without_throwing(tmp_path,monkeypatch,variant):
    folder,freeze,document,result=setup_saved(tmp_path,monkeypatch,variant)
    proof=w.verify_graph_admission(folder,freeze,result)
    assert proof['accepted'] is (variant=='candidate') and proof['fixtures_count']==11 and proof['exact_fixtures']==0


@pytest.mark.parametrize('mutation',['output','hash','input','result','summary','partial'])
def test_saved_verifier_rejects_unbound_or_partial_evidence(tmp_path,monkeypatch,mutation):
    folder,freeze,document,result=setup_saved(tmp_path,monkeypatch)
    if mutation=='output':document['evidence'][0]['candidate_outputs'][0][0]+=1.
    elif mutation=='hash':document['evidence'][0]['comparison']['input_sha256']='0'*64
    elif mutation=='input':document['fixtures'][0]['tokens'][0]+=1.
    elif mutation=='result':result['accepted']=False
    elif mutation=='summary':document['summary']['passed']=True
    else:document['error']='interrupted numerical panel'
    (folder/'graph_parity.json').write_text(json.dumps(document))
    with pytest.raises(ValueError):w.verify_graph_admission(folder,freeze,result)


def fake_context(tmp_path,monkeypatch,*,raise_on=None,large_error=False):
    monkeypatch.setattr(c,'ROOT',tmp_path);monkeypatch.setattr(c,'OUT',tmp_path/'out')
    folder=c.OUT/'workers/live_candidate';folder.mkdir(parents=True)
    fixtures=numerical_fixtures();c.save(c.OUT/'numerical_plan.json',{'fixtures':fixtures,
        'provenance':[dict(fixture_id=f['fixture_id'],input_sha256=w.digest(f['tokens']),
            historical_output83=outputs()[0]) for f in fixtures if f['batch']==1]})
    identity={'original':True};freeze=dict(declared_worker_lifecycles=lifecycle_plan(),numerical_tolerances=dict(w.TOLERANCES),
        numerical_plan=c.artifact(c.OUT/'numerical_plan.json'),
        unchanged_graph_backend_identity={'graph_identity_sha256':w.digest(identity)})
    calls=[]
    def eager(adapter,tokens,batch):
        calls.append(('eager',tokens[0],batch));return outputs()*batch
    def infer(adapter,tokens,batch):
        calls.append(('candidate',tokens[0],batch))
        if raise_on is not None and tokens[0]==raise_on:raise RuntimeError('saved failure')
        value=[outputs()[0] for _ in range(batch)]
        for row in value:row[0]=f32(row[0]+(2**-16 if large_error else 2**-17))
        return value
    context=dict(folder=folder,freeze=freeze,graph_parity_passed=True,numerical_attempted=False,numerical_passed=False,
        entrypoint_attempted=False,request_started=w.time.perf_counter(),adapters={'search':object()},
        inference_binding=SimpleNamespace(original=eager),frozen_forward=SimpleNamespace(infer=infer,identity=identity),execution_variant='candidate')
    return context,calls


@pytest.mark.parametrize('large_error',[False,True])
def test_full_numerical_panel_runs_once_and_preserves_rejected_arrays(tmp_path,monkeypatch,large_error):
    context,calls=fake_context(tmp_path,monkeypatch,large_error=large_error)
    result=w.run_numerical_checks(context)
    assert len(calls)==76 and [row[0] for row in calls]==['eager','candidate']*38
    assert result['accepted'] is (not large_error) and context['numerical_passed'] is result['accepted']
    proof=w.verify_numerical_admission(context['folder'],context['freeze'],result)
    assert proof['accepted'] is result['accepted'] and proof['fixtures_count']==38
    before=len(calls)
    with pytest.raises(ValueError):w.run_numerical_checks(context)
    assert len(calls)==before


def test_numeric_runtime_failure_preserves_partial_arrays_and_is_not_retried(tmp_path,monkeypatch):
    context,calls=fake_context(tmp_path,monkeypatch,raise_on=3.)
    with pytest.raises(RuntimeError,match='saved failure'):w.run_numerical_checks(context)
    document=c.read(context['folder']/'numerical_checks.json')
    assert len(document['evidence'])==4 and document['evidence'][-1]['eager_outputs'] and document['evidence'][-1]['candidate_outputs'] is None
    assert document['summary']['fixtures_count']==3 and 'saved failure' in document['error']
    assert not (context['folder']/'numerical_admission.json').exists()
    with pytest.raises(ValueError):w.run_numerical_checks(context)
    assert len(calls)==8


@pytest.mark.parametrize('mutation',['value','signed_zero','input_hash','fixture_id','order','count'])
def test_saved_historical_anchor_is_exact_and_bound(tmp_path,monkeypatch,mutation):
    context,calls=fake_context(tmp_path,monkeypatch)
    result=w.run_numerical_checks(context)
    plan_path=c.OUT/'numerical_plan.json';plan=c.read(plan_path);origins=plan['provenance']
    if mutation=='value':origins[0]['historical_output83'][0]=f32(1.+2**-20)
    elif mutation=='signed_zero':
        # Numeric equality is insufficient; even a sign-bit mismatch is rejected.
        evidence_path=context['folder']/'numerical_checks.json';document=c.read(evidence_path)
        document['evidence'][0]['eager_outputs'][0][0]=0.
        evidence_path.write_text(json.dumps(document));origins[0]['historical_output83'][0]=-0.
    elif mutation=='input_hash':origins[0]['input_sha256']='0'*64
    elif mutation=='fixture_id':origins[0]['fixture_id']='different'
    elif mutation=='order':origins[0],origins[1]=origins[1],origins[0]
    else:origins.pop()
    plan_path.write_text(json.dumps(plan));context['freeze']['numerical_plan']=c.artifact(plan_path)
    with pytest.raises(ValueError,match='historical|Historical'):w.verify_numerical_admission(context['folder'],context['freeze'],result)
    assert len(calls)==76


def test_live_reference_drift_preserves_full_arrays_without_admission(tmp_path,monkeypatch):
    context,calls=fake_context(tmp_path,monkeypatch)
    original=context['inference_binding'].original
    def changed(adapter,tokens,batch):
        value=original(adapter,tokens,batch)
        if batch==1:value[0][0]=f32(value[0][0]+2**-20)
        return value
    context['inference_binding'].original=changed
    with pytest.raises(ValueError,match='historical'):w.run_numerical_checks(context)
    assert len(calls)==76 and len(c.read(context['folder']/'numerical_checks.json')['evidence'])==38
    assert not context['numerical_passed'] and not (context['folder']/'numerical_admission.json').exists()


def test_composite_eager_outputs_are_not_derived_from_single_outputs(tmp_path,monkeypatch):
    context,calls=fake_context(tmp_path,monkeypatch)
    for owner,name in ((context['inference_binding'],'original'),(context['frozen_forward'],'infer')):
        original=getattr(owner,name)
        def composite(adapter,tokens,batch,original=original):
            value=original(adapter,tokens,batch)
            if batch>1:value=[row[:81]+[.25,-.25] for row in value]
            return value
        setattr(owner,name,composite)
    result=w.run_numerical_checks(context)
    assert result['accepted'] and w.verify_numerical_admission(context['folder'],context['freeze'],result)['accepted']
    assert len(calls)==76


def test_actual_suite_delegates_unchanged_and_decision_differences_are_returned(monkeypatch):
    context=dict(graph_parity_passed=True,numerical_passed=True,entrypoint_attempted=False,entrypoint_passed=False)
    calls=[]
    def handler(ctx,request):
        calls.append((ctx,request))
        return {'counts':{'complete_decisions':67}} if request=={'operation':'entrypoint_checks'} else {'action':'a legitimately different action'}
    monkeypatch.setattr(w.accepted,'handle',handler)
    assert w.handle(context,{'operation':'entrypoint_checks'})['counts']['complete_decisions']==67
    assert context['entrypoint_passed']
    result=w.handle(context,{'player':'search','observer':{'fixture':True}})
    assert result['action']=='a legitimately different action' and len(calls)==2
    with pytest.raises(ValueError):w.handle(context,{'operation':'entrypoint_checks'})


@pytest.mark.parametrize('which',['graph','numerical','suite'])
def test_no_scientific_calls_bypass_required_admissions(monkeypatch,which):
    context=dict(graph_parity_passed=which!='graph',numerical_passed=which!='numerical',entrypoint_passed=which!='suite')
    monkeypatch.setattr(w.accepted,'handle',lambda *args:pytest.fail('Unqualified call'))
    with pytest.raises(ValueError):w.handle(context,{'player':'search','observer':{}})


def test_graph_admission_sets_practical_flag_without_rewriting_original_failure(tmp_path,monkeypatch):
    folder,freeze,document,_=setup_saved(tmp_path,monkeypatch)
    (folder/'graph_admission.json').unlink()
    context=dict(folder=folder,freeze=freeze,execution_variant='candidate',graph_parity_passed=False)
    monkeypatch.setattr(w,'original_graph_parity',lambda ctx,request:document['summary'])
    result=w.practical_graph_parity(context,dict(operation='graph_parity',fixtures=freeze['graph_parity_fixtures']))
    assert result['accepted'] and context['graph_parity_passed'] and not c.read(folder/'graph_parity.json')['summary']['passed']


def test_source_guard_first_once_and_unmodified_actual_corpus_route():
    source=inspect.getsource(w.initialize);tree=ast.parse(source);body=tree.body[0].body
    assert isinstance(body[0],ast.Assign) and ast.unparse(body[0].value)=='verify_runtime_sources()'
    assert source.count('verify_runtime_sources(')==1
    assert w.accepted.corpus.run_entrypoint_checks.__module__=='tools.splendorust_search256_recovery_20261030_02.corpus'
    assert w.INTERNAL_SECONDS==9. and w.NUMERICAL_INTERNAL_SECONDS==45.


def test_import_does_not_load_torch_or_candidate_module():
    env=os.environ.copy();env.update(PYTHONDONTWRITEBYTECODE='1',PYTHONPATH=str(ROOT/'src')+os.pathsep+str(ROOT))
    code='import sys;from tools.splendorust_network_live_20261031 import worker;assert "torch" not in sys.modules;assert "tools.splendorust_network_execution_20261031.candidate_final_token_ffn" not in sys.modules'
    result=subprocess.run([sys.executable,'-B','-c',code],cwd=ROOT,env=env,capture_output=True,text=True,timeout=30)
    assert result.returncode==0,result.stderr


@pytest.mark.parametrize('name',list(w.WORKER_LIFECYCLES))
def test_each_live_and_replay_lifecycle_binds_scientific_variant(name):
    freeze={'declared_worker_lifecycles':lifecycle_plan()}
    assert w.lifecycle_variant(c.OUT/'workers'/name,freeze)==w.WORKER_LIFECYCLES[name]
    with pytest.raises(ValueError,match='variant must agree'):
        w.lifecycle_variant(c.OUT/'workers'/name,freeze,variant='candidate' if name.endswith('control') else 'control')


@pytest.mark.parametrize('mutation',['name','old_root','variant','route','extra_lifecycle','missing_lifecycle','count','graph','numeric','suite','first','startup'])
def test_lifecycle_aliases_prefixes_and_limits_fail_closed(mutation):
    freeze={'declared_worker_lifecycles':lifecycle_plan()};folder=c.OUT/'workers/live_candidate'
    spec=freeze['declared_worker_lifecycles']['live_candidate']
    if mutation=='name':folder=c.OUT/'workers/candidate'
    elif mutation=='old_root':folder=ROOT/'outputs/splendorust_network_practical_20261031_02/workers/live_candidate'
    elif mutation=='variant':spec['execution_variant']='control'
    elif mutation=='route':spec['route']='frozen'
    elif mutation=='extra_lifecycle':freeze['declared_worker_lifecycles']['retry_candidate']=spec
    elif mutation=='missing_lifecycle':freeze['declared_worker_lifecycles'].pop('replay_control')
    elif mutation=='count':spec['ordinary_requests']+=1
    elif mutation=='startup':spec['startup_requests']=2
    else:spec[{'graph':'graph_request_id','numeric':'numerical_request_id','suite':'entrypoint_request_id','first':'ordinary_first_request_id'}[mutation]]='request-00000009'
    with pytest.raises(ValueError):w.lifecycle_variant(folder,freeze)


@pytest.mark.parametrize('name',list(w.WORKER_LIFECYCLES))
def test_each_actual_startup_guards_once_before_rejecting_wrong_variant(monkeypatch,name):
    calls=[];monkeypatch.setattr(w,'verify_runtime_sources',lambda:calls.append('guard') or {})
    monkeypatch.setenv('STUDY_EXECUTION_VARIANT','candidate' if name.endswith('control') else 'control')
    monkeypatch.setenv('STUDY_WORKER_DIR',str(c.OUT/'workers'/name))
    with pytest.raises(ValueError,match='variant must agree'):w.initialize()
    assert calls==['guard']


@pytest.mark.parametrize('name',list(w.WORKER_LIFECYCLES))
def test_actual_saved_numerical_verdicts_apply_to_each_lifecycle(tmp_path,monkeypatch,name):
    old_out=ROOT/'outputs/splendorust_network_practical_20261031_02';variant=w.WORKER_LIFECYCLES[name]
    freeze=json.loads((old_out/'freeze.json').read_text());freeze['declared_worker_lifecycles']=lifecycle_plan()
    # Keep original ROOT for authentic immutable numerical-plan provenance; only new output is temporary.
    monkeypatch.setattr(c,'OUT',tmp_path/'out')
    monkeypatch.setattr(c,'artifact',lambda path:dict(path=str(Path(path)),bytes=Path(path).stat().st_size,sha256=c.file_hash(path)))
    folder=c.OUT/'workers'/name;folder.mkdir(parents=True)
    for operation,admission in [('graph_parity','graph_admission'),('numerical_checks','numerical_admission')]:
        for file in (operation,admission):
            (folder/(file+'.json')).write_bytes((old_out/'workers'/variant/(file+'.json')).read_bytes())
        result=json.loads((folder/(admission+'.json')).read_text())
        verifier=w.verify_graph_admission if operation=='graph_parity' else w.verify_numerical_admission
        proof=verifier(folder,freeze,result)
        assert proof['accepted'] and proof['execution_variant']==variant
        assert proof['fixtures_count']==(11 if operation=='graph_parity' else 38)


def test_unchanged_science_candidate_factory_clocks_and_numerical_code():
    from tools.splendorust_network_practical_v2_20261031 import worker as old
    for name in ('response_packet','numerical_contract','candidate_source','install_execution','validate_numerical_fixtures',
                 'numerical_fixtures','verify_historical_numerical_shadows','assess_outputs','summarize_admission',
                 'practical_graph_parity','run_numerical_checks','handle','close_context','serve'):
        assert ast.dump(ast.parse(inspect.getsource(getattr(w,name))))==ast.dump(ast.parse(inspect.getsource(getattr(old,name))))
    assert w.TOLERANCES==old.TOLERANCES and w.INTERNAL_SECONDS==old.INTERNAL_SECONDS==9.
    assert w.MAX_RESPONSE_BYTES==old.MAX_RESPONSE_BYTES and c.CANDIDATE_SOURCE_SHA=='e10ff696e66dfe1cbc788c7968e053d7c07e2b0be3bf8f5ae411734c2906f870'


@pytest.mark.parametrize('name',list(w.WORKER_LIFECYCLES))
def test_lifecycle_identity_cannot_be_transplanted(tmp_path,monkeypatch,name):
    monkeypatch.setattr(c,'ROOT',tmp_path);monkeypatch.setattr(c,'OUT',tmp_path/'out')
    source=tmp_path/'candidate.py';source.write_text('same held candidate fixture')
    monkeypatch.setattr(c,'CANDIDATE_SOURCE',source);monkeypatch.setattr(c,'CANDIDATE_SOURCE_SHA',c.file_hash(source))
    descriptor=c.artifact(source);variant=w.WORKER_LIFECYCLES[name];folder=c.OUT/'workers'/name;folder.mkdir(parents=True)
    original={'graph_ir_sha256':'a'*64};metadata={'pid':123,'route':'research'}
    freeze=dict(declared_worker_lifecycles=lifecycle_plan(),numerical_tolerances=dict(w.TOLERANCES),inputs=[descriptor],
        unchanged_graph_backend_identity={'graph_identity_sha256':w.digest(original)})
    context=dict(folder=folder,freeze=freeze,execution_variant=variant,frozen_forward=SimpleNamespace(identity=original,compiled=object()),
        candidate_descriptor=descriptor if variant=='candidate' else None)
    value=w.execution_identity(context,metadata);c.save(folder/'practical_execution_identity.json',value);c.save(folder/'graph_identity.json',original)
    assert value['worker_lifecycle']==name and w.verify_execution_identity(folder,freeze,metadata)['execution_variant']==variant
    value['worker_lifecycle']='replay_'+variant if name.startswith('live_') else 'live_'+variant
    (folder/'practical_execution_identity.json').write_text(json.dumps(value))
    with pytest.raises(ValueError,match='identity fields'):w.verify_execution_identity(folder,freeze,metadata)


def test_new_ready_adapter_requires_declared_lifecycle_before_any_telemetry(monkeypatch):
    from tools.splendorust_network_live_20261031 import adapters as a
    child=SimpleNamespace(output_dir=c.OUT/'workers/candidate',_sample=lambda:pytest.fail('Invalid lifecycle reached telemetry'))
    with pytest.raises(ValueError,match='worker folder'):a.validate_ready({},child,{'declared_worker_lifecycles':lifecycle_plan()})
