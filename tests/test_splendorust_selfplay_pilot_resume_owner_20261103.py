"""Original clocks, no reserve waiting, and retained process ownership."""
import ast
from copy import deepcopy
from pathlib import Path
from types import SimpleNamespace
import pytest
from tools import splendorust_selfplay_pilot_resume_owner_20261103 as owner
from tools.splendorust_selfplay_pilot_resume_20261103 import qualification_watchdog as wd

REAL_ROOT=owner.c.ROOT


def write(path,value):
    path.parent.mkdir(parents=True,exist_ok=True)
    path.write_text(owner.c.canonical(value)+'\n',encoding='utf-8')
    return owner.c.artifact(path)


@pytest.fixture
def fixture(monkeypatch,tmp_path):
    c=owner.c
    source=owner.SOURCE.read_bytes();test=owner.TEST_SOURCE.read_bytes()
    monkeypatch.setattr(c,'ROOT',tmp_path);monkeypatch.setattr(c,'OUT',tmp_path/'pilot')
    monkeypatch.setattr(c,'PILOT_OUT',tmp_path/'pilot');monkeypatch.setattr(owner,'OUT',tmp_path/'retained_owner')
    monkeypatch.setattr(owner,'SOURCE',tmp_path/'owner.py');owner.SOURCE.write_bytes(source)
    monkeypatch.setattr(owner,'TEST_SOURCE',tmp_path/'test_owner.py');owner.TEST_SOURCE.write_bytes(test)
    monkeypatch.setattr(owner,'AUTHORIZATION',tmp_path/'pilot/prelaunch/user_authorization.json')
    monkeypatch.setattr(owner.continuation,'OLD_OUT',tmp_path/'old')
    monkeypatch.setattr(owner.continuation,'OLD_LAUNCH',tmp_path/'old_owner')
    monkeypatch.setattr(owner.continuation,'MANIFEST',tmp_path/'pilot/continuation.json')
    monkeypatch.setattr(owner.continuation,'PREFIX_REVIEW',tmp_path/'pilot/prefix_review.json')
    original=write(tmp_path/'old/guardian_claim.json',dict(monotonic_origin=owner.PILOT_MONO,unix_origin=owner.PILOT_WALL))
    oldplan=write(tmp_path/'old_owner/plan.json',{'old':True})
    oldouter=write(tmp_path/'old_owner/claim.json',dict(monotonic_origin=owner.OUTER_MONO,unix_origin=owner.OUTER_WALL,
        maximum_seconds=76*3600,automatic_retries=0,plan=oldplan))
    manifest=dict(original_job_seconds=72*3600,original_model_seconds=60*3600,next_game_index=32,
        failed_decision_retries=0,original_status_remains_incomplete=True,future_forecasts_advisory=True,
        original_monotonic_origin=owner.PILOT_MONO,original_unix_origin=owner.PILOT_WALL,
        inputs=[original,oldouter],original_launch_plan=oldplan,source_continuation={'prefix':32},exact_owner_identities_absent=[])
    md=write(owner.continuation.MANIFEST,manifest);pr=write(owner.continuation.PREFIX_REVIEW,{'passed':True})
    monkeypatch.setattr(owner.continuation,'verify_continuation',lambda descriptor,**k:c.read(c.ROOT/descriptor['path']))
    monkeypatch.setattr(owner.continuation,'verify_prefix_review',lambda descriptor,**k:c.read(c.ROOT/descriptor['path']))
    freeze=dict(phase='pilot',scope='pilot',continuation=md,saved_prefix_review=pr,
        source_continuation=manifest['source_continuation'],job_seconds=72*3600,collection_cutoff_seconds=60*3600,
        resource_forecast_policy='advisory_only',resource_forecasts_may_stop=False)
    fd=write(c.OUT/'freeze.json',freeze)
    authorization=write(owner.AUTHORIZATION,dict(schema='selfplay_pilot_resume_authorization_v1',resume_authorized=True,
        resource_forecasts_advisory=True,science_unchanged=True,actual_original_clocks_retained=True,
        failed_decision_retries=0,original_completed_games_preserved=32))
    log=tmp_path/'tests.log';log.write_text('model-free passed\n')
    tests=write(tmp_path/'tests.json',dict(exit_code=0,models=0,games=0,inputs=[c.artifact(owner.SOURCE),c.artifact(owner.TEST_SOURCE)],log=c.artifact(log)))
    review=write(tmp_path/'review.json',dict(approved=True,blocking_findings=[],inputs=[c.artifact(owner.SOURCE),tests]))
    plan=dict(schema='selfplay_pilot_retained_resume_v1',authorization=authorization,continuation=md,prefix_review=pr,
        pilot_freeze=fd,owner_source=c.artifact(owner.SOURCE),tests=tests,reviews=[review],automatic_retries=0)
    write(owner.OUT/'plan.json',plan)
    return plan,freeze,manifest


def test_actual_inherited_origin_is_not_new_owner_start(fixture):
    plan,freeze,_=fixture
    assert wd.inherited_origins(freeze)==(904900.656,1793649048.764625)
    assert owner.validate_plan(plan)==plan
    instance=owner.Owner(plan);claim=owner.c.read(owner.OUT/'claim.json')
    assert (instance.mono,instance.wall)==(900586.578,1793644734.6388845)
    assert claim['current_monotonic_start']>claim['original_pilot_monotonic_origin']
    assert instance.remaining()<72*3600


@pytest.mark.parametrize('key,value',[
    ('original_monotonic_origin',904901.656),('original_unix_origin',1793649049.764625),
    ('original_job_seconds',73*3600),('original_model_seconds',61*3600),('next_game_index',31),
    ('failed_decision_retries',1),('future_forecasts_advisory',False)])
def test_guardian_rejects_clock_reset_or_changed_continuation(fixture,key,value):
    _,freeze,manifest=fixture;manifest[key]=value
    freeze['continuation']=write(owner.continuation.MANIFEST,manifest)
    with pytest.raises(ValueError):wd.inherited_origins(freeze)


@pytest.mark.parametrize('provider,offset,model_required',[
    ('monotonic',60*3600,True),('time',60*3600,True),
    ('monotonic',72*3600,False),('time',72*3600,False),
    ('monotonic',-1,True),('time',-1,True)])
def test_actual_original_deadlines_and_reversed_origin_reject(fixture,monkeypatch,provider,offset,model_required):
    _,freeze,_=fixture
    base=owner.PILOT_MONO if provider=='monotonic' else owner.PILOT_WALL
    monkeypatch.setattr(wd.time,provider,lambda:base+offset)
    with pytest.raises(ValueError):wd.inherited_origins(freeze,model_required=model_required)


def test_saved_only_work_can_follow_model_cutoff_within_original_global(fixture,monkeypatch):
    _,freeze,_=fixture
    monkeypatch.setattr(wd.time,'monotonic',lambda:owner.PILOT_MONO+61*3600)
    monkeypatch.setattr(wd.time,'time',lambda:owner.PILOT_WALL+61*3600)
    assert wd.inherited_origins(freeze,model_required=False)==(owner.PILOT_MONO,owner.PILOT_WALL)
    with pytest.raises(ValueError):wd.inherited_origins(freeze,model_required=True)


def test_production_guardian_claim_passes_original_clocks_to_reader(fixture,monkeypatch):
    _,freeze,_=fixture;c=owner.c
    write(c.OUT/'prelaunch_review.json',dict(approved=True,freeze_sha256=c.file_hash(c.OUT/'freeze.json')))
    class Process:
        pid=123456;returncode=0
        def poll(self):return 0
    monkeypatch.setattr(wd.subprocess,'Popen',lambda *a,**k:Process())
    monkeypatch.setattr(wd.psutil,'Process',lambda *a:SimpleNamespace(create_time=lambda:1.))
    monkeypatch.setattr(wd,'_WindowsJob',lambda pid:object())
    monkeypatch.setattr(wd.placement,'configure_execution',lambda *a,**k:None)
    monkeypatch.setattr(wd,'cleanup_owned',lambda *a,**k:dict(errors=[],unconfirmed=[]))
    observed=[]
    def reader(args,claim,freeze):observed.append(claim);return 0
    monkeypatch.setattr(wd,'independent_main',reader)
    args=SimpleNamespace(freeze_sha256=c.file_hash(c.OUT/'freeze.json'))
    assert wd.guardian_main(args)==0
    claim=c.read(c.OUT/'guardian_claim.json')
    assert claim['monotonic_origin']==owner.PILOT_MONO and claim['unix_origin']==owner.PILOT_WALL
    assert claim['owner_started_monotonic']>claim['monotonic_origin']
    assert observed==[claim]
    result=c.read(c.OUT/'watchdog_execution.json')
    assert result['monotonic_origin']==owner.PILOT_MONO and result['charged_job_seconds']>0


def test_inner_watchdog_rejects_new_clock_before_any_process(fixture,monkeypatch):
    _,_,_=fixture;c=owner.c
    sha=c.file_hash(c.OUT/'freeze.json')
    write(c.OUT/'guardian_claim.json',dict(freeze_sha256=sha,monotonic_origin=owner.PILOT_MONO+1,
        unix_origin=owner.PILOT_WALL,inherited_original_clocks=True,
        owner_started_monotonic=owner.PILOT_MONO+10,owner_started_unix_time=owner.PILOT_WALL+10))
    def forbidden(*a,**k):raise AssertionError('No process may start')
    monkeypatch.setattr(wd.subprocess,'Popen',forbidden)
    with pytest.raises(ValueError,match='inherits old clocks'):wd.inner_main(SimpleNamespace(freeze_sha256=sha))


@pytest.mark.parametrize('which',['authorization','scope','duplicate','proof','forecast_policy'])
def test_owner_approval_continuation_and_duplicate_gates(fixture,monkeypatch,which):
    plan,freeze,_=fixture;c=owner.c
    if which=='authorization':
        v=c.read(owner.AUTHORIZATION);v['resume_authorized']=False;plan['authorization']=write(owner.AUTHORIZATION,v)
    elif which=='scope':monkeypatch.setattr(c,'SCOPE','qualification')
    elif which=='duplicate':write(owner.OUT/'claim.json',{})
    elif which=='proof':
        def fail(*a,**k):raise ValueError('Saved prefix failed')
        monkeypatch.setattr(owner.continuation,'verify_prefix_review',fail)
    else:
        freeze['resource_forecasts_may_stop']=True;plan['pilot_freeze']=write(c.OUT/'freeze.json',freeze)
    with pytest.raises(ValueError):owner.validate_plan(plan)


def test_ram_admission_observes_immediately_without_sleep(fixture,monkeypatch):
    plan,_,_=fixture;instance=owner.Owner(plan)
    monkeypatch.setattr(owner.psutil,'virtual_memory',lambda:SimpleNamespace(available=1))
    monkeypatch.setattr(owner.shutil,'disk_usage',lambda _:SimpleNamespace(free=owner.c.MINIMUM_FREE_DISK_BYTES))
    def forbidden(_):raise AssertionError('No RAM reserve sleep')
    monkeypatch.setattr(owner.time,'sleep',forbidden)
    instance.admit('pilot')
    result=owner.c.read(owner.OUT/'pilot_resource_admission.json')
    assert result['observed_available_ram_bytes']==1 and result['ram_wait_seconds']==0 and result['stable_seconds']==0


def test_actual_disk_floor_is_retained(fixture,monkeypatch):
    plan,_,_=fixture;instance=owner.Owner(plan)
    monkeypatch.setattr(owner.shutil,'disk_usage',lambda _:SimpleNamespace(free=owner.c.MINIMUM_FREE_DISK_BYTES-1))
    with pytest.raises(ValueError,match='disk floor'):instance.admit('pilot')


@pytest.mark.parametrize('failure',[None,'admission','guardian','acceptance'])
def test_sequence_has_no_future_fit_gate_and_stops_failed_checks(fixture,monkeypatch,failure):
    plan,_,_=fixture;events=[]
    class Owner:
        def guard(self):pass
        def admit(self,scope):events.append('ram_observation')
        def remaining(self):raise AssertionError('No future workload fit admission')
        def process(self,scope,operation,args,maximum,**k):
            assert scope=='pilot';events.append(operation)
            if operation==failure:raise ValueError(operation)
            return {'exit_code':0}
        def event(self,status,**k):events.append(status)
    def accept(*a,**k):
        events.append('acceptance')
        if failure=='acceptance':raise ValueError('acceptance')
        write(owner.c.PILOT_OUT/'final_closeout.json',{'passed':True});return {'passed':True}
    monkeypatch.setattr(owner.w,'accept_scope',accept)
    if failure:
        with pytest.raises(ValueError):owner.run_sequence(plan,Owner())
        assert events[-1]==failure
    else:
        assert owner.run_sequence(plan,Owner())=={'passed':True}
        assert events==['ram_observation','admission','guardian','acceptance','pilot_accepted']


@pytest.mark.parametrize('failed',[False,True])
def test_real_detached_model_free_owner_lifecycle(fixture,monkeypatch,failed):
    plan,_,_=fixture;instance=owner.Owner(plan)
    monkeypatch.setattr(owner,'PACKAGE','json')
    if failed:
        with pytest.raises(ValueError):instance.process('pilot','probe',['tool','--not-valid'],10)
    else:assert instance.process('pilot','probe',['tool','--help'],10)['exit_code']==0
    receipt=owner.c.read(owner.OUT/'pilot_probe/exit.json')
    assert not receipt['cleanup_errors']
    owner.require_absent([receipt['identity']])


def test_process_cleanup_and_reader_bodies_remain_unchanged():
    old=ast.parse((REAL_ROOT/'tools/splendorust_selfplay_pilot_budget_20261102/workflow.py').read_text())
    new=ast.parse((REAL_ROOT/'tools/splendorust_selfplay_pilot_resume_owner_20261103.py').read_text())
    classes=[next(n for n in tree.body if isinstance(n,ast.ClassDef) and n.name=='Owner') for tree in (old,new)]
    methods=[{n.name:n for n in tree.body if isinstance(n,ast.FunctionDef)} for tree in classes]
    for name in ('process','event'):assert ast.dump(methods[0][name])==ast.dump(methods[1][name])
    a=ast.parse((REAL_ROOT/'tools/splendorust_selfplay_pilot_budget_20261102/qualification_watchdog.py').read_text().replace('splendorust_selfplay_pilot_budget_20261102','splendorust_selfplay_pilot_resume_20261103'))
    b=ast.parse((REAL_ROOT/'tools/splendorust_selfplay_pilot_resume_20261103/qualification_watchdog.py').read_text())
    functions=[{n.name:n for n in tree.body if isinstance(n,ast.FunctionDef)} for tree in (a,b)]
    for name in ('cleanup_owned','model_clock_open','stage_clock_open','independent_main'):
        assert ast.dump(functions[0][name])==ast.dump(functions[1][name]),name
