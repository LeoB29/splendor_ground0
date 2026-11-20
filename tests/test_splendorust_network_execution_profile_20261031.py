"""Model-free checks of fixed operator localization and the candidate import gate."""
from copy import deepcopy
from pathlib import Path
from types import SimpleNamespace
import ast
import hashlib
import inspect
import json
import sys
import pytest

from tools.splendorust_network_execution_20261031 import profile_worker as w,common as c


def plan():
    return w.profiling_plan(c.read(c.PREVIOUS_OUT/'freeze.json'))


def output():return [[float(i)/8 for i in range(83)]]


def operators():
    events=[]
    for i in range(6):
        events.append(dict(event_id=i,operator='aten::_transformer_encoder_layer_fwd',thread_id=10,
            parent_id=None,children_ids=[],start_us=float(i*10),end_us=float(i*10+8),
            inclusive_elapsed_us=8.,self_elapsed_us=8.))
    return [dict(operator='aten::_transformer_encoder_layer_fwd',calls=6,
        inclusive_elapsed_us=48.,self_elapsed_us=48.)],events


def measurement(enabled):
    rows,events=operators() if enabled else ([],[])
    return dict(forward_wall_ns=100,outer_wall_ns=150,profiler_setup_teardown_wall_ns=50,
        operators=rows,events=events,raw_event_count=len(events))


def test_fixed_lawful_fixture_selection_and_exact60_counterbalanced_calls():
    p=plan()
    assert [x['fixture_id'] for x in p['fixtures']]==list(w.PROFILE_FIXTURE_IDS)
    assert len(p['calls'])==60 and sum(x['mode']=='on' for x in p['calls'])==30
    assert len(p['input_hashes'])==6 and all(x['batch']==1 for x in p['fixtures'])
    for i in range(6):
        rows=p['calls'][i*10:(i+1)*10]
        assert [x['mode'] for x in rows]==['off','on','off','on','on','off','off','on','on','off']
        assert rows[0]['purpose']==rows[1]['purpose']=='profiler_exactness'
    assert p['profiling_overhead_is_descriptive'] and p['final_speed_requires_observer_off']


def test_loop_preserves_all_outputs_and_calls_once():
    p=plan();saved=[];invocations=[]
    def invoke(fixture,enabled):
        invocations.append((fixture['fixture_id'],enabled));return output(),measurement(enabled)
    rows=w.run_profile_loop(p,{f['fixture_id']:output() for f in p['fixtures']},invoke,
        lambda x:saved.append(deepcopy(x)),lambda:None)
    assert rows==saved and len(saved)==len(invocations)==60


@pytest.mark.parametrize('fault',['float','negative_zero','fused_count','elapsed','event_parent','aggregate'])
def test_profile_failure_retains_actual_bad_row_and_stops_without_retry(fault):
    p=plan();saved=[];calls=[]
    expected={f['fixture_id']:output() for f in p['fixtures']}
    def invoke(fixture,enabled):
        calls.append(enabled);value=output();metrics=measurement(enabled)
        if enabled:
            if fault=='float':value[0][0]=.00001
            if fault=='negative_zero':value[0][0]=-.0
            if fault=='fused_count':metrics['operators'][0]['calls']=5
            if fault=='elapsed':metrics['outer_wall_ns']=100_000_000_000
            if fault=='event_parent':metrics['events'][0]['parent_id']=999
            if fault=='aggregate':metrics['operators'][0]['self_elapsed_us']=47.
        return value,metrics
    with pytest.raises(ValueError):w.run_profile_loop(p,expected,invoke,lambda x:saved.append(deepcopy(x)),lambda:None)
    assert calls==[False,True] and len(saved)==2 and saved[-1]['error']
    assert saved[-1]['outputs'] is not None and saved[-1]['measurement'] is not None


def test_primary_invocation_exception_saved_and_rethrown():
    p=plan();saved=[];error=RuntimeError('profiler dispatcher failed')
    def invoke(*args):raise error
    with pytest.raises(RuntimeError) as caught:
        w.run_profile_loop(p,{f['fixture_id']:output() for f in p['fixtures']},invoke,saved.append,lambda:None)
    assert caught.value is error and len(saved)==1 and 'dispatcher' in saved[0]['error']


def test_measured_forward_off_has_no_profiler_or_extra_model_call():
    token=object();result=object();calls=[];ticks=iter([10,20,60,80])
    def compiled(x):calls.append(x);return result
    def forbidden(*args):raise AssertionError('OFF constructed profiler')
    actual,m=w.measured_forward(compiled,token,False,None,clock=lambda:next(ticks),factory=forbidden)
    assert actual is result and calls==[token]
    assert m['forward_wall_ns']==40 and m['outer_wall_ns']==70 and m['profiler_setup_teardown_wall_ns']==30
    assert m['operators']==m['events']==[]


def test_measured_forward_on_keeps_exact_object_and_lifecycle(monkeypatch):
    sequence=[];result=object();ticks=iter([10,20,60,80]);rows,events=operators()
    class Profile:
        def __enter__(self):sequence.append('enter');return self
        def __exit__(self,*args):sequence.append('exit')
    def factory(torch):sequence.append('factory');return Profile()
    def compiled(x):sequence.append('forward');return result
    monkeypatch.setattr(w,'operator_rows',lambda p:(rows,events))
    actual,m=w.measured_forward(compiled,object(),True,None,clock=lambda:next(ticks),factory=factory)
    assert actual is result and sequence==['factory','enter','forward','exit'] and m['raw_event_count']==6


def test_hierarchy_recomputes_inclusive_self_and_aggregates():
    rows,events=operators()
    assert w.verify_operator_hierarchy(rows,events)['events']==6
    parent=dict(event_id=10,operator='forward',thread_id=10,parent_id=None,children_ids=list(range(6)),
        start_us=0.,end_us=60.,inclusive_elapsed_us=60.,self_elapsed_us=12.)
    for e in events:e['parent_id']=10
    events.append(parent);rows.append(dict(operator='forward',calls=1,inclusive_elapsed_us=60.,self_elapsed_us=12.))
    assert w.verify_operator_hierarchy(rows,events)['operators']==2


@pytest.mark.parametrize('fault',['overlap','wrong_thread','unknown_child','self_time','cycle','nan','duplicate'])
def test_hierarchy_rejects_unreconciled_intervals(fault):
    rows,events=operators()
    if fault=='overlap':events[0].update(children_ids=[1,2],end_us=30.,inclusive_elapsed_us=30.,self_elapsed_us=14.);events[1]['parent_id']=0;events[2]['parent_id']=0;events[2].update(start_us=17.,end_us=25.)
    if fault=='wrong_thread':events[0]['children_ids']=[1];events[1]['parent_id']=0;events[1]['thread_id']=11
    if fault=='unknown_child':events[0]['children_ids']=[999]
    if fault=='self_time':events[0]['self_elapsed_us']=7.
    if fault=='cycle':events[0]['parent_id']=0;events[0]['children_ids']=[0]
    if fault=='nan':events[0]['start_us']=float('nan')
    if fault=='duplicate':events.append(deepcopy(events[0]))
    with pytest.raises(ValueError):w.verify_operator_hierarchy(rows,events)


def test_operator_inventory_ignores_graph_comments_and_counts_six_fusions():
    text='\n'.join(f'%{i} = aten::_transformer_encoder_layer_fwd(%x)' for i in range(6))
    assert w.graph_operator_inventory(text)=={'aten::_transformer_encoder_layer_fwd':6}


def test_source_guard_is_first_and_once_before_any_profile_or_model(monkeypatch):
    calls=[];monkeypatch.setattr(w,'verify_runtime_sources',lambda:calls.append('guard') or {})
    monkeypatch.setenv('STUDY_EXECUTION_VARIANT','forbidden')
    with pytest.raises(ValueError,match='variant'):w.initialize()
    assert calls==['guard']
    source=ast.parse(inspect.getsource(w.initialize))
    assert sum(isinstance(n,ast.Call) and isinstance(n.func,ast.Name) and n.func.id=='verify_runtime_sources'
        for n in ast.walk(source))==1


def test_actual_handler_graph_suite_and_ordinary_delegate_exactly(monkeypatch):
    seen=[];result={'counts':{'complete_decisions':67}}
    monkeypatch.setattr(w.accepted,'handle',lambda ctx,payload:seen.append(payload) or result)
    context=dict(graph_parity_passed=True,entrypoint_attempted=False,entrypoint_passed=False,execution_variant='control')
    assert w.handle(context,{'operation':'entrypoint_checks'}) is result
    assert context['entrypoint_passed'] is True
    request={'player':'search','observer':{'fixture':True}}
    assert w.handle(context,request) is result and seen[-1] is request
    with pytest.raises(ValueError,match='retried'):w.handle(context,{'operation':'entrypoint_checks'})


def test_localization_cannot_play_or_profile_before_full_qualification():
    context=dict(graph_parity_passed=True,entrypoint_passed=False,execution_variant='localization')
    with pytest.raises(ValueError,match='qualification'):w.handle(context,{'player':'search','observer':{}})
    with pytest.raises(ValueError,match='qualification'):w.handle(context,{'operation':'operator_profile'})


def write(path,value):
    path.parent.mkdir(parents=True,exist_ok=True);path.write_text(json.dumps(value),encoding='utf-8')


def supplement(tmp_path,monkeypatch):
    monkeypatch.setattr(c,'ROOT',tmp_path);monkeypatch.setattr(c,'OUT',tmp_path/'out')
    mod=tmp_path/'tools/splendorust_network_execution_20261031/candidate_fixture.py'
    mod.parent.mkdir(parents=True);mod.write_text('def build_candidate(compiled,model,folder,save_json): return compiled\n')
    refs={}
    for name in ('localization_result','localization_verification','decision','model_free_tests','independent_review'):
        path=c.OUT/(name+'.json');write(path,dict(exit_code=0,models=0,games=0,approved=True,blocking_findings=[]));refs[name]=c.artifact(path)
    return dict(schema='splendorust_network_execution_candidate_v1',phase1_freeze_sha256='a'*64,
        candidate_id='fixture',candidate_function='build_candidate',candidate_module=c.artifact(mod),
        candidate_import_module='tools.splendorust_network_execution_20261031.candidate_fixture',
        inputs=[c.artifact(mod)],**refs)


def test_candidate_contract_validates_before_loading_code(tmp_path,monkeypatch):
    value=supplement(tmp_path,monkeypatch)
    assert w.validate_candidate_supplement(value,'a'*64)==value['candidate_module']
    assert value['candidate_import_module'] not in sys.modules


@pytest.mark.parametrize('fault',['freeze','factory','path','module','unlisted','tests','review','source'])
def test_candidate_import_contract_rejects_unapproved_or_changed_sources(tmp_path,monkeypatch,fault):
    value=supplement(tmp_path,monkeypatch)
    if fault=='freeze':value['phase1_freeze_sha256']='b'*64
    if fault=='factory':value['candidate_function']='another_factory'
    if fault=='path':value['candidate_module']['path']='outside.py'
    if fault=='module':value['candidate_import_module']='tools.foreign.candidate_fixture'
    if fault=='unlisted':value['inputs']=[]
    if fault in ('tests','review'):
        key='model_free_tests' if fault=='tests' else 'independent_review';p=c.ROOT/value[key]['path']
        write(p,dict(exit_code=1,approved=False));value[key]=c.artifact(p)
    if fault=='source':(c.ROOT/value['candidate_module']['path']).write_text('changed')
    with pytest.raises((ValueError,KeyError)):w.validate_candidate_supplement(value,'a'*64)


def saved_profile(tmp_path,monkeypatch):
    p=plan();freeze=c.read(c.PREVIOUS_OUT/'freeze.json')
    monkeypatch.setattr(c,'ROOT',tmp_path);monkeypatch.setattr(c,'OUT',tmp_path/'out')
    folder=c.OUT/'workers/localization';base=folder/'operator_profile'
    expected={f['fixture_id']:output() for f in p['fixtures']};pins=[]
    write(folder/'graph_parity.json',dict(summary=dict(passed=True),evidence=[dict(fixture_id=k,candidate_outputs=v) for k,v in expected.items()]))
    write(base/'plan.json',p)
    def save(row):
        path=base/'calls'/f"{row['spec']['index']:04d}.json";write(path,row);pins.append(c.artifact(path))
    rows=w.run_profile_loop(p,expected,lambda f,on:(output(),measurement(on)),save,lambda:None)
    result=w.profile_summary(p,rows,pins);write(base/'summary.json',result)
    return folder,freeze,result


def test_full_saved_verifier_reconciles60_outputs_and30_hierarchies(tmp_path,monkeypatch):
    folder,freeze,result=saved_profile(tmp_path,monkeypatch)
    proof=w.verify_operator_profile(folder,freeze,result)
    assert proof['exact_outputs']==60 and proof['complete_saved_event_hierarchies']==30


@pytest.mark.parametrize('fault',['missing','extra','hash','changed_output','changed_operator','summary'])
def test_saved_verifier_rejects_laundered_profile_evidence(tmp_path,monkeypatch,fault):
    folder,freeze,result=saved_profile(tmp_path,monkeypatch);base=folder/'operator_profile';path=base/'calls/0001.json'
    if fault=='missing':path.unlink()
    if fault=='extra':write(base/'calls/retry.json',{})
    if fault=='hash':path.write_text('{}')
    if fault in ('changed_output','changed_operator'):
        row=json.loads(path.read_text())
        if fault=='changed_output':row['outputs'][0][0]=.1
        else:row['measurement']['operators'][0]['calls']=5
        write(path,row);result['call_artifacts'][1]=c.artifact(path)
    if fault=='summary':result['median_forward_ratio']=999.
    with pytest.raises((ValueError,FileNotFoundError)):w.verify_operator_profile(folder,freeze,result)
