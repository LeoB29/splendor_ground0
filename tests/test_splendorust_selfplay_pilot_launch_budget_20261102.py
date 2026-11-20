"""Saved-only launch gates and real tiny Windows ownership regressions."""
import ast
from copy import deepcopy
from pathlib import Path
from types import SimpleNamespace
import pytest
from tools import splendorust_selfplay_pilot_launch_budget_20261102 as launch


REAL_ROOT=launch.c.ROOT


def write(path,value):
    path.parent.mkdir(parents=True,exist_ok=True)
    path.write_text(launch.c.canonical(value)+'\n',encoding='utf-8')
    return launch.c.artifact(path)


@pytest.fixture
def setup(monkeypatch,tmp_path):
    c=launch.c
    source=launch.SOURCE.read_bytes();workflow=launch.WORKFLOW_SOURCE.read_bytes()
    authorization=c.read(launch.BUDGET_AUTHORIZATION)
    old=launch.OLD_OUT
    prior={key:c.read(path) for key,path in launch.prior_paths().items()}
    acceptance=c.read(c.QUALIFICATION_OUT/'qualification_acceptance.json')
    monkeypatch.setattr(c,'ROOT',tmp_path)
    monkeypatch.setattr(c,'QUALIFICATION_OUT',tmp_path/'qualification')
    monkeypatch.setattr(c,'PILOT_OUT',tmp_path/'pilot')
    monkeypatch.setattr(launch,'OUT',tmp_path/'launcher')
    monkeypatch.setattr(launch,'OLD_OUT',tmp_path/'old_workflow')
    monkeypatch.setattr(launch,'SOURCE',tmp_path/'launcher.py')
    monkeypatch.setattr(launch,'TEST_SOURCE',tmp_path/'test_launcher.py')
    monkeypatch.setattr(launch,'WORKFLOW_SOURCE',tmp_path/'original_workflow.py')
    monkeypatch.setattr(launch,'BUDGET_AUTHORIZATION',tmp_path/'authorization.json')
    launch.SOURCE.write_bytes(source);launch.WORKFLOW_SOURCE.write_bytes(workflow)
    launch.TEST_SOURCE.write_text('fixed test fixture source\n')
    qfreeze=write(c.QUALIFICATION_OUT/'freeze.json',{'scope':'qualification'})
    qcompletion=write(c.QUALIFICATION_OUT/'completion.json',dict(status='completed_verified',failure=None,coverage=acceptance['coverage']))
    acceptance.update(freeze=qfreeze,completion=qcompletion)
    for key in ('qualification_guardian_exit','qualification_admission_exit'):
        prior[key]['freeze_sha256']=qfreeze['sha256']
    acceptance['guardian_exit_observation']=prior['qualification_guardian_exit']
    qa=write(c.QUALIFICATION_OUT/'qualification_acceptance.json',acceptance)
    monkeypatch.setattr(launch,'QUALIFICATION_ACCEPTANCE_SHA',qa['sha256'])
    prior['plan']['qualification_freeze']=qfreeze
    paths=launch.prior_paths()
    plan_item=write(paths['plan'],prior['plan'])
    prior['claim']['plan']=plan_item
    entries={key:write(paths[key],value) for key,value in prior.items()}
    log=tmp_path/'tests.log';log.write_text('passed model-free fixture\n')
    tests=write(tmp_path/'tests.json',dict(exit_code=0,models=0,games=0,inputs=[c.artifact(launch.SOURCE),c.artifact(launch.TEST_SOURCE)],log=c.artifact(log)))
    review=write(tmp_path/'review.json',dict(approved=True,blocking_findings=[],inputs=[c.artifact(launch.SOURCE),tests]))
    science_tests=write(tmp_path/'science_tests.json',c.read(tmp_path/'tests.json'))
    resource_review=write(tmp_path/'resource_review.json',dict(approved=True,blocking_findings=[],inputs=[c.artifact(launch.SOURCE)]))
    authorization['proposal']=write(tmp_path/'proposal.json',{'fixture':'measured20hproposal'})
    approval=write(launch.BUDGET_AUTHORIZATION,authorization)
    monkeypatch.setattr(launch,'BUDGET_AUTHORIZATION_SHA',approval['sha256'])
    proof=write(tmp_path/'resource_equivalence.json',{'schema':'selfplay_resource_only_revision_v1','fixture':True})
    monkeypatch.setattr(launch.preparation,'verify_resource_only_equivalence',lambda item:c.read(c.ROOT/item['path']),raising=False)
    preflight=dict(acceptance=qa,freeze=qfreeze,reused_training_games=0,new_source_required=True,
        resource_only_equivalence=proof,resource_only_review=resource_review,budget_authorization=approval)
    monkeypatch.setattr(launch.preparation,'accepted_new_entrypoint',lambda:deepcopy(preflight))
    pfreeze=write(c.PILOT_OUT/'freeze.json',dict(scope='pilot',phase='pilot',schema=c.SCHEMA,
        new_entrypoint_preflight=preflight,model_free_tests=science_tests,implementation_reviews=[resource_review]*3,
        resource_only_equivalence=proof,resource_only_review=resource_review,budget_authorization=approval,
        job_seconds=72*3600,collection_cutoff_seconds=60*3600,stage_seconds={'assessment':20*3600,'final':12*3600}))
    plan=dict(schema='selfplay_pilot_retained_budget_launch_v1',authorization=launch.AUTHORIZATION,
        prior_workflow=entries,qualification_acceptance=qa,pilot_freeze=pfreeze,
        workflow_source=c.artifact(launch.WORKFLOW_SOURCE),launcher_source=c.artifact(launch.SOURCE),
        tests=tests,reviews=[review],workflow_seconds=launch.WORKFLOW_SECONDS,pilot_seconds=launch.JOB_SECONDS['pilot'],
        acceptance_reserve_seconds=600,automatic_retries=0,budget_authorization=approval,resource_only_equivalence=proof)
    write(launch.OUT/'plan.json',plan)
    return plan


def repin(plan,key,value):
    item=plan['prior_workflow'][key]
    plan['prior_workflow'][key]=write(launch.c.ROOT/item['path'],value)


def test_read_only_accepted_qualification_and_held_workflow_admitted(setup):
    plan=setup
    old={p:p.read_bytes() for p in launch.OLD_OUT.rglob('*') if p.is_file()}
    assert launch.validate_plan(plan)==plan
    assert all(p.read_bytes()==b for p,b in old.items())
    assert not (launch.OUT/'claim.json').exists()
    assert not (launch.c.PILOT_OUT/'guardian_claim.json').exists()


@pytest.mark.parametrize('key,value',[
    ('authorization','older automatic approval'),('workflow_seconds',76*3600+1),
    ('pilot_seconds',72*3600+1),('acceptance_reserve_seconds',0),('automatic_retries',1),
    ('schema','selfplay_pilot_automatic_workflow_v1')])
def test_no_changed_authorization_or_clocks(setup,key,value):
    setup[key]=value
    with pytest.raises(ValueError):launch.validate_plan(setup)


@pytest.mark.parametrize('key,value',[
    ('monotonic_origin',900587.578),('unix_origin',1793644735.6388845),
    ('maximum_seconds',76*3600+1),('automatic_retries',1)])
def test_original_outer_claim_is_not_reset(setup,key,value):
    claim=launch.c.read(launch.OLD_OUT/'claim.json');claim[key]=value;repin(setup,'claim',claim)
    with pytest.raises(ValueError,match='original76-hour'):launch.validate_plan(setup)


@pytest.mark.parametrize('provider,value',[
    ('monotonic',900585.578),('time',1793644733.6388845),
    ('monotonic',900586.578+76*3600),('time',1793644734.6388845+76*3600)])
def test_expired_or_reversed_original_clock_rejected(setup,monkeypatch,provider,value):
    monkeypatch.setattr(launch.time,provider,lambda:value)
    with pytest.raises(ValueError,match='outer clock valid'):launch.validate_plan(setup)


@pytest.mark.parametrize('which',[
    'launcher_claim','launcher_failure','pilot_claim','pilot_launch','pilot_admission','pilot_closeout'])
def test_duplicate_claim_or_implicit_resume_rejected(setup,which):
    paths=dict(launcher_claim=launch.OUT/'claim.json',launcher_failure=launch.OUT/'failure.json',
        pilot_claim=launch.c.PILOT_OUT/'guardian_claim.json',pilot_launch=launch.c.PILOT_OUT/'launch.json',
        pilot_admission=launch.c.PILOT_OUT/'prelaunch_review.json',pilot_closeout=launch.c.PILOT_OUT/'final_closeout.json')
    write(paths[which],{})
    with pytest.raises(ValueError,match='duplicate'):launch.validate_plan(setup)


@pytest.mark.parametrize('kind',['parent_scope','pilot_scope','wrong_preflight','failed_test','failed_review','active_owner','missing_freeze'])
def test_wrong_scope_or_approval_cannot_advance(setup,monkeypatch,kind):
    if kind=='parent_scope':monkeypatch.setattr(launch.c,'SCOPE','pilot')
    elif kind in ('pilot_scope','wrong_preflight'):
        p=launch.c.PILOT_OUT/'freeze.json';v=launch.c.read(p)
        if kind=='pilot_scope':v['scope']='qualification'
        else:v['new_entrypoint_preflight']['new_source_required']=False
        setup['pilot_freeze']=write(p,v)
    elif kind=='failed_test':
        p=launch.c.ROOT/setup['tests']['path'];v=launch.c.read(p);v['exit_code']=1;setup['tests']=write(p,v)
    elif kind=='failed_review':
        p=launch.c.ROOT/setup['reviews'][0]['path'];v=launch.c.read(p);v['approved']=False;setup['reviews']=[write(p,v)]
    elif kind=='active_owner':
        def active(items):raise ValueError('Exact previous owned process still alive')
        monkeypatch.setattr(launch,'require_absent',active)
    else:(launch.c.PILOT_OUT/'freeze.json').unlink()
    with pytest.raises((ValueError,FileNotFoundError)):launch.validate_plan(setup)
    assert not (launch.OUT/'claim.json').exists()


@pytest.mark.parametrize('kind',['wrong_failure','no_acceptance_progress','failed_old_exit','helper_started','missing_hold'])
def test_only_intentional_post_qualification_hold_is_eligible(setup,kind):
    if kind=='wrong_failure':
        v=launch.c.read(launch.OLD_OUT/'failure.json');v['error']='RuntimeError: another operation failed';repin(setup,'failure',v)
    elif kind=='no_acceptance_progress':
        v=launch.c.read(launch.OLD_OUT/'progress/003.json');v['status']='running';repin(setup,'qualification_accepted_progress',v)
    elif kind=='failed_old_exit':
        v=launch.c.read(launch.OLD_OUT/'qualification_admission/exit.json');v['exit_code']=1;repin(setup,'qualification_admission_exit',v)
    elif kind=='helper_started':write(launch.OLD_OUT/'pilot_freeze/owner.json',{})
    else:(launch.OLD_OUT/'pilot_freeze/USER_REQUESTED_HOLD.json').unlink()
    with pytest.raises((ValueError,FileNotFoundError)):launch.validate_plan(setup)


def test_owner_inherits_exact_original_clock_and_only_writes_new_claim(setup):
    owner=launch.Owner(setup)
    claim=launch.c.read(launch.OUT/'claim.json')
    assert owner.mono==900586.578 and owner.wall==1793644734.6388845
    assert claim['inherited_original_workflow_clock']
    assert claim['monotonic_origin']==owner.mono and claim['unix_origin']==owner.wall
    assert claim['current_monotonic_start']>owner.mono
    with pytest.raises(FileExistsError):launch.Owner(setup)


def test_new_save_cannot_write_old_evidence(setup):
    with pytest.raises(ValueError):launch.save(launch.OLD_OUT/'new.json',{})
    with pytest.raises(ValueError):launch.save(launch.c.QUALIFICATION_OUT/'new.json',{})
    with pytest.raises(ValueError):launch.save(launch.c.PILOT_OUT/'new.json',{})


@pytest.mark.parametrize('kind',['test_source','test_file','review_source','review_receipt'])
def test_generic_passed_receipts_do_not_replace_exact_wrapper_review(setup,kind):
    c=launch.c
    if kind.startswith('test_'):
        p=c.ROOT/setup['tests']['path'];value=c.read(p)
        value['inputs'].pop(0 if kind=='test_source' else 1)
        setup['tests']=write(p,value)
        review_path=c.ROOT/setup['reviews'][0]['path'];review=c.read(review_path)
        review['inputs'][1]=setup['tests'];setup['reviews']=[write(review_path,review)]
    else:
        p=c.ROOT/setup['reviews'][0]['path'];value=c.read(p)
        value['inputs'].pop(0 if kind=='review_source' else 1)
        setup['reviews']=[write(p,value)]
    with pytest.raises(ValueError,match='bind'):launch.validate_plan(setup)


@pytest.mark.parametrize('key,value',[
    ('approval','Approve another experiment'),('authorized',False),('assessment_stage_seconds',16*3600),
    ('total_job_seconds',76*3600),('model_cutoff_seconds',64*3600),('other_scientific_settings_unchanged',False),
    ('same_fresh_samples',False),('prior_hold_preserved_as_history',False),('requires_separate_version_and_freeze',False)])
def test_budget_approval_is_exact_and_does_not_extend_other_limits(setup,monkeypatch,key,value):
    approval=launch.c.read(launch.BUDGET_AUTHORIZATION);approval[key]=value
    setup['budget_authorization']=write(launch.BUDGET_AUTHORIZATION,approval)
    monkeypatch.setattr(launch,'BUDGET_AUTHORIZATION_SHA',setup['budget_authorization']['sha256'])
    with pytest.raises(ValueError,match='Only the unlaunched'):launch.validate_plan(setup)


@pytest.mark.parametrize('kind',['unproved','wrong_equivalence','wrong_approval','wrong_review','job','model','assessment','final'])
def test_actual_equivalence_check_and_freeze_budget_bindings_are_mandatory(setup,monkeypatch,kind):
    if kind=='unproved':
        def rejected(item):raise ValueError('scientific source differs')
        monkeypatch.setattr(launch.preparation,'verify_resource_only_equivalence',rejected)
    else:
        path=launch.c.PILOT_OUT/'freeze.json';v=launch.c.read(path)
        if kind=='wrong_equivalence':v['resource_only_equivalence']=v['model_free_tests']
        elif kind=='wrong_approval':v['budget_authorization']=v['model_free_tests']
        elif kind=='wrong_review':v['resource_only_review']=v['model_free_tests']
        elif kind=='job':v['job_seconds']+=1
        elif kind=='model':v['collection_cutoff_seconds']+=1
        else:v['stage_seconds'][kind]+=1
        setup['pilot_freeze']=write(path,v)
    with pytest.raises(ValueError):launch.validate_plan(setup)
    assert not (launch.OUT/'claim.json').exists()


def test_budget_child_uses_additive_package_and_original_hold_source():
    assert launch.PACKAGE=='tools.splendorust_selfplay_pilot_budget_20261102'
    assert 'pilot_v2_20261102' in launch.WORKFLOW_SOURCE.as_posix()


def test_ownership_methods_are_exact_original_ast_without_global_rebinding():
    old=ast.parse((REAL_ROOT/'tools/splendorust_selfplay_pilot_v2_20261102/workflow.py').read_text())
    new=ast.parse((REAL_ROOT/'tools/splendorust_selfplay_pilot_launch_budget_20261102.py').read_text())
    oc=next(n for n in old.body if isinstance(n,ast.ClassDef) and n.name=='Owner')
    nc=next(n for n in new.body if isinstance(n,ast.ClassDef) and n.name=='Owner')
    oldmethods={n.name:n for n in oc.body if isinstance(n,ast.FunctionDef)}
    newmethods={n.name:n for n in nc.body if isinstance(n,ast.FunctionDef)}
    assert set(oldmethods)==set(newmethods)
    for name in ('guard','remaining','admit','process','event'):
        assert ast.dump(oldmethods[name])==ast.dump(newmethods[name]),name
    for n in ast.walk(new):
        if isinstance(n,(ast.Assign,ast.AnnAssign,ast.AugAssign)):
            targets=n.targets if isinstance(n,ast.Assign) else [n.target]
            assert not any(isinstance(t,ast.Attribute) and isinstance(t.value,ast.Name) and t.value.id in ('w','c') for t in targets)


@pytest.mark.parametrize('failure',[None,'resources','admission','remaining','guardian','acceptance'])
def test_true_pilot_only_order_and_no_advance_after_failure(setup,monkeypatch,failure):
    events=[]
    class Owner:
        def guard(self):pass
        def admit(self,scope):
            assert scope=='pilot';events.append('resources')
            if failure=='resources':raise ValueError('resources')
        def process(self,scope,operation,args,maximum,**kwargs):
            assert scope=='pilot';events.append(operation)
            assert operation in ('admission','guardian')
            assert args[0]==('preparation' if operation=='admission' else 'qualification_watchdog')
            assert maximum==(600 if operation=='admission' else 72*3600)
            if failure==operation:raise ValueError(operation)
            return {'exit_code':0}
        def remaining(self):
            events.append('remaining')
            return 72*3600+599 if failure=='remaining' else 72*3600+600
        def event(self,status,**kwargs):events.append(status)
    def accept(scope,sha,receipt,**kwargs):
        assert scope=='pilot' and sha==setup['pilot_freeze']['sha256'];events.append('acceptance')
        if failure=='acceptance':raise ValueError('acceptance')
        write(launch.c.PILOT_OUT/'final_closeout.json',{'accepted':True})
        return {'accepted':True}
    monkeypatch.setattr(launch.w,'accept_scope',accept)
    if failure:
        with pytest.raises(ValueError):launch.run_sequence(setup,Owner())
        assert events[-1]==failure
        assert 'pilot_accepted' not in events
    else:
        assert launch.run_sequence(setup,Owner())=={'accepted':True}
        assert events==['resources','admission','remaining','guardian','acceptance','pilot_accepted']


@pytest.mark.parametrize('failed',[False,True])
def test_actual_detached_windows_child_exit_and_cleanup(setup,monkeypatch,failed):
    owner=launch.Owner(setup)
    monkeypatch.setattr(launch,'PACKAGE','json')
    arguments=['tool','--not-a-valid-option' if failed else '--help']
    if failed:
        with pytest.raises(ValueError):owner.process('pilot','probe',arguments,10)
    else:
        result=owner.process('pilot','probe',arguments,10)
        assert result['status']=='completed' and result['exit_code']==0 and result['cleanup_errors']==[]
    receipt=launch.c.read(launch.OUT/'pilot_probe/exit.json')
    assert receipt['status']==('failed' if failed else 'completed')
    assert receipt['observed_by']=='retained workflow Popen handle'
    assert receipt['cleanup_errors']==[]
    launch.w.require_absent([receipt['identity']])


def test_parent_scope_and_original_workflow_out_stay_unchanged():
    before=launch.w.OUT
    env=launch.child_environment('pilot')
    assert env['SELFPLAY_JOB_SCOPE']=='pilot' and launch.c.SCOPE=='qualification'
    assert launch.w.OUT==before and launch.OUT!=before
