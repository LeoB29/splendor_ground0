"""Independent model-free review of recovery isolation and inherited clocks."""
import ast
from pathlib import Path
import pytest
from . import common as c
from . import control,charts
from tools.splendorust_proxy_calibration_20261113 import analysis as original_analysis


def _function(path,name):
    tree=ast.parse(Path(path).read_text(encoding='utf8'))
    return next(node for node in tree.body if isinstance(node,ast.FunctionDef) and node.name==name)


def test_chart_function_is_identical_with_separate_output_globals():
    assert ast.dump(_function(charts.__file__,'render_charts'),include_attributes=False)==ast.dump(
        _function(original_analysis.__file__,'render_charts'),include_attributes=False)
    assert charts.c is c and charts.c is not original_analysis.c
    assert charts.c.OUT!=original_analysis.c.OUT


def test_recovery_reuses_original_origins_and_clips_both_clocks(monkeypatch):
    old=dict(monotonic_origin=20.,unix_origin=1000.,monotonic_deadline=5020.,unix_deadline=6000.)
    monkeypatch.setattr(c,'read',lambda path:dict(limits=old))
    monkeypatch.setattr(c,'DISCOVERED_UNIX',1500.)
    monkeypatch.setattr(c,'TOTAL_SECONDS',1200.)
    actual=c.recovery_limits()
    assert actual==dict(monotonic_origin=20.,unix_origin=1000.,monotonic_deadline=1720.,unix_deadline=2700.)
    monkeypatch.setattr(c,'DISCOVERED_UNIX',5900.)
    assert c.recovery_limits()==old


def test_analysis_retains_original_stage_deadlines(monkeypatch):
    monkeypatch.setattr(control.time,'monotonic',lambda:200.)
    monkeypatch.setattr(control.time,'time',lambda:1200.)
    original=dict(monotonic_origin=100.,unix_origin=1100.,monotonic_deadline=240.,unix_deadline=1240.)
    monkeypatch.setattr(c,'read',lambda path:dict(limits=original))
    parent=dict(monotonic_origin=50.,unix_origin=1050.,monotonic_deadline=400.,unix_deadline=1400.)
    actual=control.stage_limits(parent,'analysis')
    assert actual['monotonic_deadline']==240. and actual['unix_deadline']==1240.
    accepted=control.stage_limits(parent,'acceptance')
    assert accepted['monotonic_deadline']==400. and accepted['unix_deadline']==1400.


def test_expired_original_analysis_clock_cannot_restart(monkeypatch):
    monkeypatch.setattr(control.time,'monotonic',lambda:200.)
    monkeypatch.setattr(control.time,'time',lambda:1200.)
    monkeypatch.setattr(c,'read',lambda path:dict(limits=dict(monotonic_deadline=199.,unix_deadline=1199.)))
    with pytest.raises(ValueError,match='clock expired'):
        control.stage_limits(dict(monotonic_deadline=400.,unix_deadline=1400.),'analysis')


@pytest.mark.parametrize('role',['qualification','diagnostic','reproduction','fit'])
def test_no_model_or_decision_stage_can_enter(role):
    with pytest.raises(ValueError,match='no models'):
        control.stage_limits({},role)


def test_model_stage_registration_invalidates_recovery(monkeypatch):
    monkeypatch.setattr(c,'MODEL_JOBS',('analysis',))
    with pytest.raises(ValueError,match='no models'):
        control.stage_limits({},'analysis')


def test_new_writer_cannot_change_original_failed_receipt(tmp_path):
    with pytest.raises(ValueError,match='New output boundary'):
        c.save(c.OLD_OUT/'jobs/reproduction/worker/service_actual_exit.json',{'status':'passed'})
    assert not (tmp_path/'unexpected.json').exists()


@pytest.mark.parametrize('reject_reproduction',[False,True])
def test_saved_inputs_require_full_original_records_and_reproduction_comparison(monkeypatch,reject_reproduction):
    from . import worker
    calls=[];panel={'roots':[]};records=['saved_record'];report={'rows':5120}
    def item(path):return {'path':str(path),'bytes':1,'sha256':'a'*64}
    monkeypatch.setattr(worker.c,'read',lambda path:panel)
    monkeypatch.setattr(worker.c,'artifact',item)
    monkeypatch.setattr(worker.oldworker,'check_qualification',lambda *a:calls.append('qualification') or {'passed':True})
    monkeypatch.setattr(worker.panel,'validate_panel',lambda value:value)
    monkeypatch.setattr(worker.panel,'request_matrix',lambda value:['fixed_request'])
    monkeypatch.setattr(worker.oldworker,'passed_job',lambda role:({}, {'result':{'index':item(worker.oldc.OUT/'jobs'/role/'worker/index.json')}}))
    monkeypatch.setattr(worker.checks,'check_service',lambda role,*a:calls.append('service:'+role) or {'passed':True})
    def ordinary(*args,**kwargs):
        calls.append('ordinary_records');assert args[2]==['fixed_request'];return records,report
    def reproduce(*args,**kwargs):
        calls.append('exact_reproduction_comparison')
        if reject_reproduction:raise ValueError('saved replay differs')
        return {'rows':32,'all_exact':True}
    monkeypatch.setattr(worker.verifier,'verify_records',ordinary)
    monkeypatch.setattr(worker.verifier,'verify_reproductions',reproduce)
    if reject_reproduction:
        with pytest.raises(ValueError,match='saved replay differs'):worker.inputs({}, {}, lambda:None)
    else:
        saved,result,proof=worker.inputs({}, {}, lambda:None)
        assert saved is panel and result is records and proof['ordinary'] is report
        assert proof['reproduction']=={'rows':32,'all_exact':True}
    assert calls==['qualification','service:diagnostic','ordinary_records','service:reproduction','exact_reproduction_comparison']


def test_each_actual_worker_guards_sources_once_before_saved_evaluation(monkeypatch,tmp_path):
    from . import worker
    from tools import future_training_rules_20261021 as guard
    monkeypatch.setattr(c,'OUT',tmp_path);monkeypatch.setattr(c,'PLAN',tmp_path/'freeze.json')
    c.save(c.PLAN,{'design':c.design(),'worker_inputs':[],'data_inputs':[]})
    expected=c.sha(c.PLAN);calls=[]
    monkeypatch.setattr(control,'set_own_cpu',lambda:None)
    monkeypatch.setattr(control,'handshake',lambda *a:{})
    monkeypatch.setattr(control,'Boundary',lambda *a:lambda:None)
    def source_guard():calls.append('guard');return guard.SOURCE_HASHES
    monkeypatch.setattr(guard,'verify_runtime_sources',source_guard)
    def stop(*a):
        assert calls==['guard'];calls.append('saved_inputs');raise ValueError('stop before analysis')
    monkeypatch.setattr(worker,'inputs',stop)
    assert worker.run('analysis',expected)==1
    assert calls==['guard','saved_inputs']
    assert c.read(tmp_path/'jobs/analysis/worker/runtime_sources.json')['sources']==guard.SOURCE_HASHES
    assert c.read(tmp_path/'jobs/analysis/worker/result.json')['status']=='failed'


def test_current_live_exact_descendant_still_blocks_recovered_service(monkeypatch):
    from . import checks
    bundle={key:checks.c.read(checks.c.resolve(item)) for key,item in checks.PINS.items() if key!='freeze'}
    live=bundle['premature_exit']['owned_native_identities'][-1]
    recovery=dict(original_freeze=checks.PINS['freeze'],original_limits=bundle['root_owner']['limits'],cleanup_exception=checks.exception_contract())
    monkeypatch.setattr(checks.c,'check_deadlines',lambda limits:None)
    monkeypatch.setattr(checks.c,'exact_alive',lambda ident:ident==live)
    with pytest.raises(ValueError,match='Prior owned processes absent'):
        checks.check_service('reproduction',checks.c.read(checks.c.PLAN),lambda:None,recovery)



def test_real_saved_service_passes_complete_recovery_route_without_models(monkeypatch):
    from . import checks
    import sys
    owner=checks.c.read(checks.c.resolve(checks.PINS['root_owner']))
    recovery=dict(original_freeze=checks.PINS['freeze'],original_limits=owner['limits'],cleanup_exception=checks.exception_contract())
    # This regression exercises immutable historical evidence. The independent
    # clock tests and prospective launch retain real current deadline admission.
    monkeypatch.setattr(checks.c,'check_deadlines',lambda limits:None)
    result=checks.check_service('reproduction',checks.c.read(checks.c.PLAN),lambda:None,recovery)
    assert result['recovered_cleanup']['cleanup_interval_microseconds']==233332
    assert result['recovered_cleanup']['complete_decisions']==32
    assert result['recovered_cleanup']['exact_reproduction_comparisons_still_required'] is True
    assert 'torch' not in sys.modules



@pytest.mark.parametrize('evidence_case',['matching','different','missing','foreign'])
def test_acceptance_binds_analysis_saved_evidence_to_recomputed_proof(monkeypatch,tmp_path,evidence_case):
    from . import worker
    from . import checks
    from tools import future_training_rules_20261021 as guard
    monkeypatch.setattr(c,'OUT',tmp_path);monkeypatch.setattr(c,'PLAN',tmp_path/'freeze.json')
    c.save(c.PLAN,{'design':c.design(),'worker_inputs':[],'data_inputs':[],
        'original_freeze':checks.PINS['freeze'],'cleanup_exception':checks.exception_contract()})
    expected=c.sha(c.PLAN);proof={'ordinary':5120,'reproduction':32,'cleanup':'original_outer_receipt'}
    analysis_folder=tmp_path/'jobs/analysis/worker'
    report=c.save(analysis_folder/'analysis.json',{'diagnostic':'fixed'})
    proof_path=analysis_folder/('other.json' if evidence_case=='foreign' else 'saved_evidence.json')
    if evidence_case=='missing':
        saved={'path':str(proof_path),'bytes':1,'sha256':'0'*64}
    else:
        saved=c.save(proof_path,dict(proof,reproduction=31) if evidence_case=='different' else proof)
    job={'result':{'analysis':report,'verification':saved,'charts':[]}}
    monkeypatch.setattr(control,'set_own_cpu',lambda:None)
    monkeypatch.setattr(control,'handshake',lambda *a:{})
    monkeypatch.setattr(control,'Boundary',lambda *a:lambda:None)
    monkeypatch.setattr(guard,'verify_runtime_sources',lambda:guard.SOURCE_HASHES)
    monkeypatch.setattr(worker,'inputs',lambda *a:({'roots':[]},[],proof))
    monkeypatch.setattr(worker,'passed_job',lambda role:({},job))
    monkeypatch.setattr(worker.analysis,'verify_arithmetic',lambda *a,**k:{'passed':True})
    monkeypatch.setattr(worker.panel,'validate_panel',lambda *a,**k:None)
    result=worker.run('acceptance',expected)
    receipt=c.read(tmp_path/'jobs/acceptance/worker/result.json')
    if evidence_case=='matching':
        assert result==0 and receipt['status']=='passed'
    else:
        assert result==1 and receipt['status']=='failed'
        assert not(tmp_path/'jobs/acceptance/worker/independent_acceptance.json').exists()
