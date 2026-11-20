"""Actual owner/ack/readback orchestration with a deliberately fast fake fitter.

The fake Popen publishes placement before returning to the parent, then emits
six real-format clock records. No model, native process or game is invoked.
"""
import ast
import inspect
import json
from types import SimpleNamespace

import psutil
import pytest

from tools.splendorust_policy_head_20261104 import common as c
from tools.splendorust_policy_head_20261104 import qualification_control as owner
from tools.splendorust_policy_head_20261104 import qualification_watchdog as watchdog
from tools.splendorust_policy_head_20261104 import training


def write(path,value):
    path.parent.mkdir(parents=True,exist_ok=True)
    path.write_text(json.dumps(value),encoding='utf-8')


def old_postspawn_sampling_function():
    tree=ast.parse(inspect.getsource(owner.run_fit))
    body=next(n for n in ast.walk(tree) if isinstance(n,ast.Try)).body
    assignment=next(n for n in body if isinstance(n,ast.Assign) and
        any(isinstance(t,ast.Name) and t.id=='owner_reading' for t in n.targets))
    body.remove(assignment)
    spawn=next(i for i,n in enumerate(body) if isinstance(n,ast.Assign) and
        isinstance(n.value,ast.Call) and isinstance(n.value.func,ast.Attribute) and n.value.func.attr=='Popen')
    body.insert(spawn+1,assignment)
    namespace=dict(vars(owner));exec(compile(ast.fix_missing_locations(tree),'<old-postspawn-order>','exec'),namespace)
    return namespace['run_fit']


def complete_fast_fitter(monkeypatch,tmp_path,*,legacy=False):
    from tools.local_player_qualification_128 import clock as process_clock
    import subprocess
    out=tmp_path/'scope';out.mkdir()
    monkeypatch.setattr(c,'ROOT',tmp_path);monkeypatch.setattr(c,'OUT',out);monkeypatch.setattr(c,'WRITE_OBSERVER',None)
    epoch,qpc_offset=1_800_000_000.,10_000_000.
    state={'mono':100.,'done':False};calls=[]
    def sample():return dict(owner_monotonic=state['mono'],request_qpc=qpc_offset+state['mono'],unix=epoch+state['mono'])
    monkeypatch.setattr(owner.clocks,'sample',sample)
    monkeypatch.setattr(owner,'time',SimpleNamespace(monotonic=lambda:state['mono'],time=lambda:epoch+state['mono'],sleep=lambda _:None))
    monkeypatch.setattr(owner,'fit_command_environment',lambda *args:(['fake-fitter'],{}))
    monkeypatch.setattr(process_clock,'_WindowsJob',lambda pid:None)
    monkeypatch.setattr(watchdog,'cleanup_owned',lambda *args,**kwargs:{'errors':[]})
    identity=dict(pid=54321,created=epoch+100.)
    contract=owner.clocks.runtime_contract()
    freeze=dict(clock_contract=contract,learned_entrypoint={'case_ids':['a','b','c']},
        fit_plan={'path':'fit_plan.json'},fit_parity_fixtures={'path':'parity.json'})
    write(tmp_path/'fit_plan.json',{});write(tmp_path/'parity.json',{'fixtures':[]});write(out/'corpus_manifest.json',{})
    write(out/'stages/fit_start.json',dict(monotonic=90.,unix_time=epoch+90.,monotonic_deadline=200.,unix_deadline=epoch+200.))
    def process(pid=None):
        if pid==identity['pid'] and state['done']:raise psutil.NoSuchProcess(pid)
        actual=identity if pid is not None else dict(pid=12345,created=epoch-20.)
        return SimpleNamespace(pid=actual['pid'],create_time=lambda:actual['created'],cpu_affinity=lambda:[0],
            nice=lambda:owner.placement.NORMAL_PRIORITY,children=lambda recursive=True:[],memory_info=lambda:SimpleNamespace(rss=100))
    monkeypatch.setattr(owner,'psutil',SimpleNamespace(Process=process,NoSuchProcess=psutil.NoSuchProcess))
    class FastChild:
        pid=identity['pid'];returncode=None
        def __init__(self,*args,**kwargs):
            # This event is permitted by Popen: the child runs before it returns.
            state['mono']=100.5;self.index=0
            folder=out/'fit';reading=sample()
            write(folder/'placement.json',dict(identity=identity,cpu_affinity=[0],priority='NORMAL_PRIORITY_CLASS',
                before_torch_import=True,clock_contract=contract,owner_monotonic_time=reading['owner_monotonic'],
                request_qpc_time=reading['request_qpc'],unix_time=reading['unix']))
            write(folder/'checkpoint_manifest.json',{'synthetic_checkpoint_identity':True})
            state['mono']=101.
        def poll(self):
            if self.index==6:
                state['mono']+=.1;state['done']=True;self.returncode=0;return 0
            i=self.index;self.index+=1;state['mono']+=.2
            directory=out/'fit/learned_entrypoint'
            start=dict(schema='selfplay_learned_request_start_v2',request_index=i,case_id=['a','b','c'][i%3],
                mode='eager' if i<3 else 'compiled',candidate_manifest=c.artifact(out/'fit/checkpoint_manifest.json'),
                **owner.clocks.start_fields())
            path=directory/f'request_{i:03d}.started.json';write(path,start)
            state['mono']+=.1
            response=dict(schema='selfplay_learned_response_v2',request_index=i,case_id=start['case_id'],mode=start['mode'],
                candidate_manifest=start['candidate_manifest'],start=c.artifact(path),**owner.clocks.completion_fields(start))
            write(directory/f'request_{i:03d}.completed.json',response);state['mono']+=.1
            return None
    monkeypatch.setattr(subprocess,'Popen',FastChild)
    monkeypatch.setattr(training,'verify_pilot_fit',lambda *args,**kwargs:calls.append('science-verifier') or {'passed':True})
    monkeypatch.setattr(owner,'load_inputs',lambda *a:{'manifest':{},'source_specs':[],'proof':{}})
    env=SimpleNamespace(resident={},start=0.,wall_start=epoch,deadline=200.,wall_deadline=epoch+200.,admit=lambda:None,guard=lambda:None)
    (old_postspawn_sampling_function() if legacy else owner.run_fit)(env,freeze)
    return out,env,freeze,calls


def test_actual_owner_prespawn_sample_accepts_fast_child_and_all_six_acks(monkeypatch,tmp_path):
    out,env,freeze,calls=complete_fast_fitter(monkeypatch,tmp_path)
    launched=c.read(out/'fit/owner.json');placed=c.read(out/'fit/placement.json')
    assert launched['owner_monotonic_start']==100.<placed['owner_monotonic_time']==100.5
    assert launched['request_qpc_start']<placed['request_qpc_time']
    assert c.read(out/'fit_execution.json')['exit_code']==0
    assert len(list((out/'fit/learned_entrypoint').glob('*.parent_accepted.json')))==6
    assert owner.verify_fit(env,freeze)=={'passed':True} and calls==['science-verifier']


def test_old_postspawn_sampling_reproduces_saved_verifier_false_stop(monkeypatch,tmp_path):
    out,env,freeze,calls=complete_fast_fitter(monkeypatch,tmp_path,legacy=True)
    assert c.read(out/'fit/owner.json')['owner_monotonic_start']==101.
    assert c.read(out/'fit/placement.json')['owner_monotonic_time']==100.5
    with pytest.raises(ValueError,match='placement before Torch'):owner.verify_fit(env,freeze)
    assert calls==[]


@pytest.mark.parametrize('field',['owner_monotonic_start','request_qpc_start','monotonic_deadline','unix_deadline'])
def test_saved_reader_rejects_future_owner_or_reset_deadline(monkeypatch,tmp_path,field):
    out,env,freeze,calls=complete_fast_fitter(monkeypatch,tmp_path)
    path=out/'fit/owner.json';value=c.read(path);value[field]+=1000.;write(path,value)
    with pytest.raises(ValueError):owner.verify_fit(env,freeze)
    assert calls==[]


from tools import splendorust_policy_head_owner_20261104 as retained
from tools.splendorust_policy_head_20261104 import independent_review as reader


@pytest.fixture
def owner_plan(monkeypatch,tmp_path):
    monkeypatch.setattr(c,'ROOT',tmp_path);monkeypatch.setattr(c,'OUT',tmp_path/'study')
    monkeypatch.setattr(c,'PILOT_OUT',c.OUT);monkeypatch.setattr(c,'WRITE_OBSERVER',None)
    monkeypatch.setattr(retained,'OUT',tmp_path/'owner');monkeypatch.setattr(watchdog,'OWNER_OUT',retained.OUT)
    source=tmp_path/'owner.py';source.write_text('immutable_owner_source')
    test=tmp_path/'test_owner.py';test.write_text('immutable_test_source')
    monkeypatch.setattr(retained,'SOURCE',source);monkeypatch.setattr(retained,'TEST_SOURCE',test)
    monkeypatch.setattr(retained,'AUTHORIZATION',c.OUT/'prelaunch/user_authorization.json')
    write(retained.AUTHORIZATION,dict(authorized=True,study='policy_head_only',new_source_games=0,reference_promoted=False))
    freeze={**owner.contract(),'inputs':[c.artifact(source)]};write(c.OUT/'freeze.json',freeze)
    log=tmp_path/'tests.log';log.write_text('passed')
    tests=tmp_path/'tests.json';write(tests,dict(exit_code=0,inputs=[c.artifact(source),c.artifact(test)],log=c.artifact(log)))
    review=tmp_path/'review.json';write(review,dict(approved=True,blocking_findings=[],inputs=[c.artifact(source),c.artifact(tests)]))
    plan=dict(schema='policy_head_retained_owner_v1',authorization=c.artifact(retained.AUTHORIZATION),
        pilot_freeze=c.artifact(c.OUT/'freeze.json'),owner_source=c.artifact(source),tests=c.artifact(tests),
        reviews=[c.artifact(review)],automatic_retries=0)
    write(retained.OUT/'plan.json',plan)
    return plan,freeze


def test_owner_starts_one_fresh_shared_clock_and_rejects_duplicate(owner_plan):
    plan,freeze=owner_plan
    assert retained.validate_plan(plan)==plan
    instance=retained.Owner(plan)
    claim=c.read(retained.OUT/'claim.json')
    assert instance.remaining()>c.JOB_SECONDS-10
    assert watchdog.fresh_origins(freeze,c.file_hash(retained.OUT/'claim.json'))==(instance.mono,instance.wall)
    assert claim['fresh_owned_clock'] and claim['job_seconds']==72*3600
    assert claim['model_cutoff_seconds']==60*3600
    with pytest.raises(ValueError,match='No prior owner'):retained.validate_plan(plan)


@pytest.mark.parametrize('field', ['authorization','scope','forecast','source','tests','review','retry'])
def test_wrong_owner_authority_source_contract_or_review_stops(owner_plan,field):
    plan,freeze=owner_plan
    if field=='authorization':
        value=c.read(retained.AUTHORIZATION);value['authorized']=False;write(retained.AUTHORIZATION,value)
        plan['authorization']=c.artifact(retained.AUTHORIZATION)
    elif field in ('scope','forecast'):
        freeze['phase' if field=='scope' else 'resource_forecasts_may_stop']='qualification' if field=='scope' else True
        write(c.OUT/'freeze.json',freeze);plan['pilot_freeze']=c.artifact(c.OUT/'freeze.json')
    elif field=='source':retained.SOURCE.write_text('changed')
    elif field=='tests':
        p=c.ROOT/plan['tests']['path'];value=c.read(p);value['exit_code']=1;write(p,value);plan['tests']=c.artifact(p)
    elif field=='review':
        p=c.ROOT/plan['reviews'][0]['path'];value=c.read(p);value['approved']=False;write(p,value);plan['reviews']=[c.artifact(p)]
    else:plan['automatic_retries']=1
    with pytest.raises(ValueError):retained.validate_plan(plan)


@pytest.mark.parametrize('key', ['monotonic_origin','unix_origin','job_seconds','model_cutoff_seconds','created'])
def test_guardian_rejects_invalid_clock_or_process_identity(owner_plan,key):
    plan,freeze=owner_plan;retained.Owner(plan)
    p=retained.OUT/'claim.json';value=c.read(p)
    if key=='monotonic_origin':value[key]+=2*c.JOB_SECONDS
    elif key=='unix_origin':value[key]-=2*c.JOB_SECONDS
    else:value[key]+=1
    write(p,value)
    with pytest.raises(ValueError):watchdog.fresh_origins(freeze,c.file_hash(p))


def test_real_low_ram_is_observed_without_wait_or_reserve_gate(owner_plan,monkeypatch):
    plan,_=owner_plan;instance=retained.Owner(plan)
    monkeypatch.setattr(retained.shutil,'disk_usage',lambda _:SimpleNamespace(free=6*c.GiB))
    monkeypatch.setattr(retained.psutil,'virtual_memory',lambda:SimpleNamespace(available=1))
    monkeypatch.setattr(retained.time,'sleep',lambda _:pytest.fail('RAM reserve must never cause waiting'))
    instance.admit('pilot')
    assert c.read(retained.OUT/'resource_admission.json')['ram_wait_seconds']==0
    env=owner.Envelope(dict(monotonic_origin=instance.mono,unix_origin=instance.wall))
    monkeypatch.setattr(env,'guard',lambda:None)
    env.admit();env.admit()
    assert env.waited==0 and env.admissions==2
    assert c.read(c.OUT/'memory_admissions/000001.json')['observed_available_bytes']==1


def test_actual_disk_floor_still_applies(owner_plan,monkeypatch):
    plan,_=owner_plan;instance=retained.Owner(plan)
    monkeypatch.setattr(retained.shutil,'disk_usage',lambda _:SimpleNamespace(free=c.GiB))
    with pytest.raises(ValueError,match='free-disk'):instance.admit('pilot')


def test_process_order_no_requalification_no_source_games(owner_plan,monkeypatch):
    plan,_=owner_plan;retained.Owner(plan);calls=[]
    class FakeOwner:
        def admit(self,scope):calls.append('admit')
        def guard(self):pass
        def process(self,scope,operation,args,maximum,**kwargs):calls.append((operation,args));return {'exit_code':0}
        def event(self,*a,**k):calls.append('accepted')
    def accept(sha,receipt,**kwargs):write(c.OUT/'final_closeout.json',{'passed':True});return {'passed':True}
    monkeypatch.setattr(retained,'accept_study',accept)
    assert retained.run_sequence(plan,FakeOwner())=={'passed':True}
    assert calls[0]=='admit' and calls[-1]=='accepted'
    assert [x[0] for x in calls if isinstance(x,tuple)]==['admission','guardian']
    assert '--owner-claim-sha256' in calls[2][1]


@pytest.mark.parametrize('fail_at',['admission','guardian'])
def test_owned_failure_never_retries_or_accepts(owner_plan,monkeypatch,fail_at):
    plan,_=owner_plan;retained.Owner(plan);seen=[]
    class FailedOwner:
        def admit(self,*a):pass
        def guard(self):pass
        def process(self,scope,operation,*args,**kwargs):
            seen.append(operation)
            if operation==fail_at:raise ValueError('saved failure')
            return {}
        def event(self,*a,**k):pytest.fail('No acceptance')
    monkeypatch.setattr(retained,'accept_study',lambda *a,**k:pytest.fail('No acceptance'))
    with pytest.raises(ValueError,match='saved failure'):retained.run_sequence(plan,FailedOwner())
    assert seen==(['admission'] if fail_at=='admission' else ['admission','guardian'])


def test_fit_command_uses_original_reused_manifest_and_new_scope(monkeypatch,tmp_path):
    monkeypatch.setattr(c,'ROOT',tmp_path);monkeypatch.setattr(c,'OUT',tmp_path/'new')
    write(c.OUT/'freeze.json',{'frozen':True})
    freeze=dict(reused_corpus={'path':'old/corpus_manifest.json'},fit_plan={'path':'new/fit_plan.json'},
                fit_parity_fixtures={'path':'new/fit_parity.json'})
    command,env=owner.fit_command_environment(SimpleNamespace(deadline=500.,wall_deadline=1800000500.),freeze,c.OUT/'fit')
    assert command[command.index('--manifest')+1]==str(tmp_path/'old/corpus_manifest.json')
    assert command[command.index('--mode')+1]=='pilot'
    assert command[4]=='tools.splendorust_policy_head_20261104.training'
    assert env['STUDY_RESEARCH_FREEZE_SHA256']==c.file_hash(c.OUT/'freeze.json')
    assert env['SELFPLAY_MONOTONIC_DEADLINE']=='500.0'
    assert env['SELFPLAY_UNIX_DEADLINE']=='1800000500.0'


def test_guardian_and_reader_final_clocks_keep_original_origin():
    assert watchdog.model_clock_open(0.,1000.,59*3600,1000.+59*3600,final_started=False,owned_children=[])
    assert not watchdog.model_clock_open(0.,1000.,61*3600,1000.+61*3600,final_started=False,owned_children=[])
    assert watchdog.model_clock_open(0.,1000.,61*3600,1000.+61*3600,final_started=True,owned_children=[])
    with pytest.raises(ValueError,match='Saved-only final'):
        watchdog.model_clock_open(0.,1000.,61*3600,1000.+61*3600,final_started=True,owned_children=[1])
    mono,wall=reader.review_deadlines(dict(monotonic_origin=0.,unix_origin=1000.),
        dict(monotonic_deadline=72*3600,unix_deadline=1000+72*3600),71*3600,1000+71*3600)
    assert mono==72*3600 and wall==1000+72*3600


def test_same_namespace_child_process_is_owned_and_exits_cleanly(owner_plan,monkeypatch):
    # Actual tiny standard-library child; no model imports or study execution.
    plan,_=owner_plan;instance=retained.Owner(plan)
    monkeypatch.setattr(retained,'PACKAGE','json')
    monkeypatch.setattr(retained,'child_environment',lambda scope:dict(__import__('os').environ))
    result=instance.process('pilot','smoke',['tool','--help'],10,freeze_sha='fixture')
    assert result['exit_code']==0 and result['cleanup_errors']==[]
    retained.require_absent([result['identity']])


def test_root_runner_only_fit_then_fixed_assessment_then_saved_final(monkeypatch,tmp_path):
    from tools.splendorust_policy_head_20261104.assessment import control as assessment
    monkeypatch.setattr(c,'ROOT',tmp_path);monkeypatch.setattr(c,'OUT',tmp_path/'new');monkeypatch.setattr(c,'WRITE_OBSERVER',None)
    freeze={'reused_corpus':{'path':'old/corpus.json'},'source_reuse_proof':{'path':'proof.json'}}
    write(c.OUT/'freeze.json',freeze);sha=c.file_hash(c.OUT/'freeze.json')
    write(c.OUT/'watchdog_claim.json',dict(freeze_sha256=sha,monotonic_origin=0.,unix_origin=0.))
    calls=[]
    class Env:
        def __init__(self,*a):self.start=self.wall_start=0.;self.stage=None;self.resident={}
        def enable_evidence_accounting(self):calls.append('accounting')
        def begin(self,stage):self.stage=stage;calls.append('begin:'+stage)
        def admit(self):calls.append('admit')
        def finish(self):calls.append('finish:'+self.stage)
    monkeypatch.setattr(owner,'Envelope',Env)
    monkeypatch.setattr(owner.placement,'configure_execution',lambda *a:None)
    monkeypatch.setattr(owner,'verify_runtime_sources',lambda:{'fixture':True})
    monkeypatch.setattr(owner,'verify_runtime',lambda *a:calls.append('runtime'))
    monkeypatch.setattr(owner,'load_inputs',lambda *a:calls.append('reuse'))
    monkeypatch.setattr(owner,'switch_placement',lambda *a:calls.append('placement'))
    def fit(*args):calls.append('fit');write(c.OUT/'fit/fit_result.json',dict(checkpoint_manifest={'candidate':True}))
    monkeypatch.setattr(owner,'run_fit',fit)
    monkeypatch.setattr(owner,'verify_fit',lambda *a:calls.append('verifyfit'))
    monkeypatch.setattr(assessment,'run',lambda *a:calls.append('assessment'))
    monkeypatch.setattr(owner,'final_checks',lambda *a:({'passed':True} if not calls.append('savedfinal') else None))
    assert owner.run(sha)==0
    assert calls==['accounting','begin:fit','runtime','reuse','admit','placement','fit','verifyfit','finish:fit',
                   'assessment','begin:final','savedfinal','finish:final']
    result=c.read(c.OUT/'completion.json')
    assert result['new_source_games']==0 and result['reused_source_games']==256
