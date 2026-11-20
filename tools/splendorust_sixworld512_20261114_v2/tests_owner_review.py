"""Independent model-free tests of qualification gates and bounded ownership."""
from copy import deepcopy
import pytest
from . import common as c, control, owner, worker, schedule, sequential


@pytest.fixture
def workspace(tmp_path,monkeypatch):
    monkeypatch.setattr(c,'ROOT',tmp_path)
    monkeypatch.setattr(c,'OUT',tmp_path/'new_attempt')
    monkeypatch.setattr(c,'PLAN',c.OUT/'freeze.json')
    return c.OUT


def test_dual_clocks_expire_independently(monkeypatch):
    limits=dict(monotonic_origin=10.,unix_origin=100.,monotonic_deadline=20.,unix_deadline=110.)
    monkeypatch.setattr(c.time,'monotonic',lambda:19.)
    monkeypatch.setattr(c.time,'time',lambda:109.)
    c.check_deadlines(limits)
    monkeypatch.setattr(c.time,'time',lambda:110.)
    with pytest.raises(ValueError):c.check_deadlines(limits)
    monkeypatch.setattr(c.time,'time',lambda:109.)
    monkeypatch.setattr(c.time,'monotonic',lambda:20.)
    with pytest.raises(ValueError):c.check_deadlines(limits)


def test_child_stage_never_extends_original_clocks_or_model_cutoff(monkeypatch):
    parent=dict(monotonic_origin=10.,unix_origin=100.,monotonic_deadline=1000.,unix_deadline=1090.)
    monkeypatch.setattr(c,'MODEL_SECONDS',100)
    monkeypatch.setattr(control.time,'monotonic',lambda:90.)
    monkeypatch.setattr(control.time,'time',lambda:180.)
    model=control.clipped_limits(parent,500,model=True)
    assert model['monotonic_deadline']==110 and model['unix_deadline']==200
    audit=control.clipped_limits(parent,2000,model=False)
    assert audit['monotonic_deadline']==1000 and audit['unix_deadline']==1090


def test_role_schedule_has_no_overlap_or_extra_games(workspace):
    plan={'schedule':c.save(c.OUT/'schedule.json',schedule.build_schedule())}
    assert len(worker.stage_specs(plan,'operational'))==6
    selected=[worker.stage_specs(plan,role) for role in c.BLOCKS]
    assert all(len(rows)==400 for rows in selected)
    assert [row['game_index'] for rows in selected for row in rows]==list(range(3600))
    with pytest.raises(ValueError):worker.stage_specs(plan,'analysis')


@pytest.mark.parametrize('change',[
    {'exit_code':1},{'actual_exit_observed':False},{'exact_owned_absence':False},
    {'freeze_sha256':'other'},{'cleanup_errors':['saved failure']},{'error':'saved failure'},
])
def test_passed_job_requires_actual_clean_owned_exit(workspace,monkeypatch,change):
    monkeypatch.setattr(c,'exact_alive',lambda identity:False)
    ident={'pid':123,'created':456.};role='operational';expected='frozen'
    result=dict(status='passed',error=None,identity=ident,role=role,freeze_sha256=expected,result={})
    item=c.save(c.OUT/'jobs'/role/'worker/result.json',result)
    receipt=dict(status='passed',exit_code=0,actual_exit_observed=True,exact_owned_absence=True,
        freeze_sha256=expected,error=None,cleanup_errors=[],identity=ident,
        observed_owned_identities=[],result_artifact=item)
    receipt.update(change);c.save(c.OUT/'jobs'/role/'actual_exit.json',receipt)
    with pytest.raises(ValueError):worker.passed_job(role,expected)


def _mock_owner(workspace,monkeypatch,*,failing_role=None):
    from tools import future_training_rules_20261021 as guard
    plan=dict(corrected_source_receipt={},worker_inputs=[],data_inputs=[],calibration={})
    c.save(c.PLAN,plan);expected=c.sha(c.PLAN)
    original_read=c.read
    monkeypatch.setattr(c,'read',lambda path:{} if path=={} else original_read(path))
    original_resolve=c.resolve
    monkeypatch.setattr(c,'resolve',lambda item:{} if item=={} else original_resolve(item))
    monkeypatch.setattr(guard,'verify_runtime_sources',lambda:{})
    bounds=dict(monotonic_origin=0,unix_origin=0,monotonic_deadline=10**12,unix_deadline=10**12)
    monkeypatch.setattr(control,'handshake',lambda *args:{'limits':bounds})
    monkeypatch.setattr(control,'bounds',lambda limits:None)
    monkeypatch.setattr(control,'clipped_limits',lambda *args,**kwargs:bounds)
    monkeypatch.setattr(owner,'qualified_calibration',lambda path:{})
    seen=[];results={};verifications=[]
    def launch(command,folder,limits,metadata,**kwargs):
        role=metadata['role'];seen.append(role)
        if role=='block_02':assert (c.OUT/'interims/look_000400.json').exists()
        if role=='analysis':assert (c.OUT/'interims/look_000800.json').exists()
        if role==failing_role:raise ValueError('preserved qualification failure')
        value={}
        if role in c.BLOCKS:value={'block':c.save(folder/'block.json',{'pair_scores':[.5]*200})}
        results[role]=value
        return c.save(folder/'worker/result.json',value)
    monkeypatch.setattr(control,'launch',launch)
    monkeypatch.setattr(worker,'passed_job',lambda role,expected:(results[role],{}))
    monkeypatch.setattr(sequential,'analyze_pairs',lambda pairs,calibration:{'stop':len(pairs)>=400})
    monkeypatch.setattr(sequential,'verify_analysis',lambda pairs,*args:verifications.append(len(pairs)))
    return expected,seen,verifications


def test_root_gates_and_stops_exactly_after_verified_interim(workspace,monkeypatch):
    expected,seen,checks=_mock_owner(workspace,monkeypatch)
    assert owner.run(expected)==0
    assert seen==['qualification','operational','block_01','block_02','analysis','acceptance']
    assert checks==[200,400]
    assert c.read(c.OUT/'completion.json')['status']=='verified_pending_actual_owner_exit'
    assert not (c.OUT/'jobs/block_03').exists()


def test_failed_qualification_cannot_launch_operational_or_main_games(workspace,monkeypatch):
    expected,seen,checks=_mock_owner(workspace,monkeypatch,failing_role='qualification')
    assert owner.run(expected)==1
    assert seen==['qualification'] and checks==[]
    assert c.read(c.OUT/'failure.json')['status']=='incomplete'
    assert not (c.OUT/'completion.json').exists()


@pytest.mark.parametrize('status',['failed','passed'])
def test_failed_calibration_cannot_be_admitted_by_recovery(workspace,monkeypatch,status):
    from . import statistical_qualification as stats
    calib=c.OUT/'inputs/calibration.json';c.save(calib,{'status':status})
    monkeypatch.setattr(stats,'verify_qualification',lambda report:{'status':'passed','qualification_passed':False})
    with pytest.raises(ValueError):owner.qualified_calibration(calib)


def test_final_charts_allow_valid_forfeits_with_no_completed_timings(workspace):
    from . import charts
    interim=c.save(c.OUT/'interim.json',dict(games=400,look=dict(score=.5,lower=.4,upper=.6)))
    report=dict(interims=[interim],timings={'3':[],'6':[]},failed_decisions=[{'kind':'timeout'}])
    result=charts.render(report,c.OUT/'charts')
    assert len(result)==2 and all(c.resolve(item).suffix=='.png' and item['bytes']>0 for item in result)
    c.verify(result)


@pytest.mark.parametrize('extend',[False,True])
def test_recovery_launcher_preserves_original_global_clocks(workspace,monkeypatch,extend):
    from . import launcher
    monkeypatch.setattr(c,'FAILED_OUT',c.OUT/'preserved_attempt')
    limits=dict(monotonic_origin=10.,unix_origin=100.,monotonic_deadline=1000.,unix_deadline=1090.)
    c.save(c.FAILED_OUT/'launcher_claim.json',{'limits':limits})
    proposed=dict(limits)
    if extend:proposed['monotonic_deadline']+=1
    c.save(c.PLAN,{'original_limits':proposed})
    monkeypatch.setattr(c.time,'monotonic',lambda:90.)
    monkeypatch.setattr(c.time,'time',lambda:180.)
    monkeypatch.setattr(launcher.sys,'argv',['launcher','--freeze-sha256',c.sha(c.PLAN)])
    calls=[];monkeypatch.setattr(control,'launch',lambda *args,**kwargs:calls.append((args,kwargs)))
    if extend:
        with pytest.raises(ValueError,match='clocks never reset'):launcher.main()
        assert not calls and not (c.OUT/'launcher_claim.json').exists()
    else:
        assert launcher.main()==0 and calls[0][0][2]==limits
        assert c.read(c.OUT/'launcher_claim.json')['limits']==limits
        assert calls[0][0][0][4]==c.PACKAGE+'.owner'


def recovery_fixture(workspace,monkeypatch):
    from tools import future_training_rules_20261021 as guard
    monkeypatch.setattr(c,'FAILED_OUT',c.OUT/'preserved_attempt')
    old=c.FAILED_OUT;ident={'pid':999,'created':123.0}
    calibration=c.save(old/'calibration.json',{'status':'passed'})
    original=dict(design=c.design(),corrected_source_receipt={},worker_inputs=[],data_inputs=[],calibration=calibration)
    for key in ('schedule','initials','qualification_plan','research_profiles','control_profiles','graph_parity_plan','console_executable'):
        original[key]={'preserved':key}
    c.save(old/'freeze.json',original);monkeypatch.setattr(c,'FAILED_FREEZE_SHA',c.sha(old/'freeze.json'))
    receipt=dict(status='failed',exit_code=1,actual_exit_observed=True,exact_owned_absence=True,
        cleanup_errors=[],identity=ident,observed_owned_identities=[])
    for relative in ('launch/actual_exit.json','jobs/qualification/actual_exit.json'):c.save(old/relative,receipt)
    limits=dict(monotonic_origin=10,unix_origin=100,monotonic_deadline=1000,unix_deadline=1090)
    c.save(old/'launcher_claim.json',dict(identity=ident,limits=limits))
    c.save(old/'jobs/qualification/worker/result.json',dict(result=None,error='Worker must use one allowed logical processor'))
    transport=old/'jobs/qualification/worker/live/services/life_0000/transport'
    transport.mkdir(parents=True);(transport/'stdout.log').write_bytes(b'')
    monkeypatch.setattr(c,'exact_alive',lambda ident:False)
    monkeypatch.setattr(guard,'verify_runtime_sources',lambda:{})
    monkeypatch.setattr(owner,'qualified_calibration',lambda path:{'status':'passed'})
    monkeypatch.setattr(owner,'source_inputs',lambda:[])
    return original,transport,limits


def test_recovery_reuses_exact_original_sample_and_limits_after_zero_model_failure(workspace,monkeypatch):
    original,transport,limits=recovery_fixture(workspace,monkeypatch)
    prepared=c.read(c.resolve(owner.prepare(None)))
    for key in ('schedule','initials','qualification_plan','calibration','research_profiles','control_profiles','design'):
        assert prepared[key]==original[key]
    assert prepared['original_limits']==limits
    preserved=c.read(c.resolve(prepared['startup_correction']))
    assert preserved['models']==preserved['new_games']==0
    assert preserved['original_limits']==limits and preserved['inputs']


@pytest.mark.parametrize('evidence',['initialization','service','operational','started_result','live_owner'])
def test_recovery_refuses_ambiguous_or_started_prior_execution(workspace,monkeypatch,evidence):
    original,transport,limits=recovery_fixture(workspace,monkeypatch)
    if evidence=='initialization':c.save(transport/'initialization.json',{})
    elif evidence=='service':(transport.parent/'service').mkdir()
    elif evidence=='operational':(c.FAILED_OUT/'jobs/operational').mkdir()
    elif evidence=='started_result':
        path=c.FAILED_OUT/'jobs/qualification/worker/result.json'
        path.write_text(c.canonical(dict(result={},error='Worker must use one allowed logical processor')))
    else:monkeypatch.setattr(c,'exact_alive',lambda ident:True)
    with pytest.raises(ValueError):owner.prepare(None)
    assert not (c.OUT/'prepared.json').exists()
