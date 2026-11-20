"""Synthetic timeout, admission and freeze boundaries without player execution."""
from copy import deepcopy
import json
from pathlib import Path
from types import SimpleNamespace

import pytest

from tools.matched_strength_20261020 import common as c, runner as r, memory as m
from test_matched_strength_adapter_20261020 import CONFIG, MATERIAL


@pytest.fixture(autouse=True)
def configured(monkeypatch):
    monkeypatch.setattr(c,'config',lambda:deepcopy(CONFIG))


def test_twelve_saved_transfer_fixtures_without_inference():
    cases=r.build_validation()
    assert len(cases)==12 and [cases[i]['observer']['context']['game'] for i in (2,6,10)]==[1,2,5]
    for i in range(0,12,2):
        assert cases[i]['snapshot']!=cases[i+1]['snapshot']
        assert cases[i]['observer']==cases[i+1]['observer']
    assert {row['fit'] for row in cases}==set(c.FITS)


def test_protected_analysis_budget_survives_forward_wall_jump(monkeypatch):
    monkeypatch.setattr(r.sys,'modules',{key:value for key,value in r.sys.modules.items() if key!='torch'})
    monkeypatch.setattr(r.time,'monotonic',lambda:100.)
    monkeypatch.setattr(c,'boundary',lambda *a,**kw:{'remaining_seconds':4300.})
    with pytest.raises(c.ResourceHold,match='dual-clock worker admission'):
        r.worker_options(100000,0,reserve=4200,mode='audit')
    monkeypatch.setattr(c,'boundary',lambda *a,**kw:{'remaining_seconds':6000.})
    options=r.worker_options(100000,0,reserve=4200,mode='audit')
    assert options['absolute_deadline']==1900.
    monkeypatch.setattr(c,'boundary',lambda *a,**kw:{'remaining_seconds':4200.})
    with pytest.raises(c.ResourceHold,match='protected reserve'):options['resource_guard']()


@pytest.mark.parametrize('status',sorted(r.PLAYER_FAILURES)+['worker_error'])
def test_player_forfeits_retained_but_scientific_errors_hold(tmp_path,monkeypatch,status):
    case=r.build_validation()[0]
    monkeypatch.setattr(c,'OUTPUT',tmp_path)
    monkeypatch.setattr(c,'artifact',lambda p:{'path':str(p),'sha256':c.file_hash(p),'bytes':Path(p).stat().st_size})
    calls=[]
    def admission(base,deadline,*,name,**kw):
        calls.append('admission');c.save_new(base/'memory_admissions'/(name+'.json'),{'status':'passed'})
    monkeypatch.setattr(m,'wait_for_memory',admission)
    monkeypatch.setattr(r,'worker_options',lambda *a,**kw:{})
    class Worker:
        def __init__(self,module,folder,**kw):
            calls.append('worker');self.folder=folder;folder.mkdir(parents=True)
            self.initialization={'status':'ok'};c.save_new(folder/'initialization.json',self.initialization)
        def __enter__(self):return self
        def __exit__(self,*args):pass
        def request(self,payload,seconds,validate=None):
            assert seconds==120 and set(payload)=={'observer'}
            receipt={'status':status,'accepted':False,'result':None,'receipt_elapsed_seconds':None}
            c.save_new(self.folder/'request-00000001.json',receipt);return receipt
    monkeypatch.setattr(r,'PlannerWorker',Worker)
    if status in ('worker_error','illegal'):
        with pytest.raises(ValueError,match='Hard worker|Unrecognized illegal-action'):r.decision_attempt('games/g000/t0000','local',case['snapshot'],case['observer'],1e12,0,reserve=64800)
    else:
        result=r.decision_attempt('games/g000/t0000','local',case['snapshot'],case['observer'],1e12,0,reserve=64800)
        assert result['status']=='forfeit_'+status and not result['correct']
    assert calls==['admission','worker']
    assert (tmp_path/'decision_attempts/games/g000/t0000.json').exists()


def illegal_response_fixture(tmp_path,player):
    from splendor_ai.engine.actions import Action, ActionType
    case=r.build_validation()[0];observer=case['observer'] if player=='local' else c.opponent_observer(case['observer'])
    if player=='opponent':
        state=r.rules.CorrectedEnv(seed=11).initial_state()
        case={'snapshot':r.rules.serialize_snapshot(state)};observer=r.rules.observation_payload(state)
    legal=[c.serialize_action(a) for a in r.rules.legal_actions(r.rules.deserialize_snapshot(case['snapshot']))]
    action=c.serialize_action(Action(ActionType.PASS));assert action not in legal
    trace=tmp_path/'continuations.jsonl.gz';trace.write_bytes(b'synthetic complete trace')
    if player=='local':
        record={'root_identity':observer['context']['identity'],'planner':'reference',
            'profile':c.LOCAL_PROFILE_ID,'ruleset_id':c.RULESET_ID,'catalogue_sha256':c.CATALOGUE_SHA256,
            'proposal':{'fit':observer['context']['fit'],'model_id':'anchor_'+str(observer['context']['fit'])},
            'choice':0,'candidates':{'actions':[action]},'worlds':[{'outcomes':[{}]} for _ in range(128)]}
        result={'decision':{'action':action,'record':record},'trace':{'sha256':c.file_hash(trace),
            'bytes':trace.stat().st_size,'records':128},'ruleset_id':c.RULESET_ID,
            'catalogue_sha256':c.CATALOGUE_SHA256,'profile_id':c.LOCAL_PROFILE_ID,'observer_sha256':c.digest(observer)}
    else:
        from tools.analyze_external_strength_screen_20261018_02 import main_action
        profile=c.read(c.EXTERNAL_PROFILE_PATH)
        action=deepcopy(legal[0]);action['metadata']=[['unsupported','synthetic']]
        count=len({c.canonical(main_action(a)) for a in legal})
        result={'player':'opponent','observer_sha256':c.digest(observer),'legal_actions_sha256':c.digest(legal),
            'legal_count':len(legal),'ruleset':c.RULESET_ID,'catalogue_sha256':c.CATALOGUE_SHA256,
            'profile_sha256':c.EXTERNAL_PROFILE_SHA256,'status':'completed','action':action,
            'diagnostics':{'opponent_id':'splendorust','profile_id':profile['profile_id'],
                'observer_seed':int(c.digest(observer)[:16],16),'phases':[{'action':main_action(action),
                    'simulations':profile['iterations'],'leaf_calls':1,'native_legal_count':count,'heuristic_phase':False}],
                'inference_calls':1,'inference_rows':1,'forced_pass':False}}
    return case,observer,legal,trace,result


@pytest.mark.parametrize('player',['local','opponent'])
def test_recognized_illegal_response_is_distinct_from_unknown_or_legal(tmp_path,player):
    _,observer,legal,trace,result=illegal_response_fixture(tmp_path,player)
    assert r.response_has_recognized_illegal_action(result,player,observer,legal,trace)
    action=result['decision']['action'] if player=='local' else result['action']
    assert not r.response_has_recognized_illegal_action(result,player,observer,legal+[action],trace)
    result['observer_sha256']='0'*64
    assert not r.response_has_recognized_illegal_action(result,player,observer,legal,trace)
    assert not r.response_has_recognized_illegal_action(None,player,observer,legal,trace)


@pytest.mark.parametrize('damage',['record_identity','fit','choice','bool_choice','work','trace','action_types'])
def test_broken_local_work_never_becomes_illegal_forfeit(tmp_path,damage):
    _,observer,legal,trace,result=illegal_response_fixture(tmp_path,'local')
    record=result['decision']['record']
    if damage=='record_identity':record['profile']='wrong'
    elif damage=='fit':record['proposal']['fit']=161002
    elif damage=='choice':record['choice']=1
    elif damage=='bool_choice':record['choice']=False
    elif damage=='work':record['worlds'].pop()
    elif damage=='trace':trace.write_bytes(b'changed')
    elif damage=='action_types':result['decision']['action']['tier']=False
    assert not r.response_has_recognized_illegal_action(result,'local',observer,legal,trace)


@pytest.mark.parametrize('damage',['empty_phases','wrong_count','wrong_work','wrong_selected_phase','wrong_inference'])
def test_broken_opponent_work_never_becomes_illegal_forfeit(tmp_path,damage):
    _,observer,legal,trace,result=illegal_response_fixture(tmp_path,'opponent')
    d=result['diagnostics']
    if damage=='empty_phases':d.update(phases=[],inference_calls=0,inference_rows=0)
    elif damage=='wrong_count':d['phases'][0]['native_legal_count']+=1
    elif damage=='wrong_work':d['phases'][0]['simulations']-=1
    elif damage=='wrong_selected_phase':d['phases'][0]['action']={'kind':'Pass'}
    elif damage=='wrong_inference':d['inference_rows']+=1
    assert not r.response_has_recognized_illegal_action(result,'opponent',observer,legal,trace)


def test_complete_illegal_clock_response_is_retained_as_loss(tmp_path,monkeypatch):
    case,observer,legal,trace,result=illegal_response_fixture(tmp_path,'local')
    monkeypatch.setattr(c,'OUTPUT',tmp_path/'output')
    monkeypatch.setattr(c,'artifact',lambda p:{'path':str(p),'sha256':c.file_hash(p),'bytes':Path(p).stat().st_size})
    monkeypatch.setattr(r,'worker_options',lambda *a,**kw:{})
    def admission(base,deadline,*,name,**kw):c.save_new(base/'memory_admissions'/(name+'.json'),{'status':'passed'})
    monkeypatch.setattr(m,'wait_for_memory',admission)
    class Worker:
        def __init__(self,module,folder,**kw):
            self.folder=folder;folder.mkdir(parents=True)
            (folder/'continuations.jsonl.gz').write_bytes(trace.read_bytes())
            self.initialization={'status':'ok'};c.save_new(folder/'initialization.json',self.initialization)
        def __enter__(self):return self
        def __exit__(self,*args):pass
        def request(self,payload,seconds,validate=None):
            assert seconds==120 and validate(result) is False
            receipt={'status':'illegal','accepted':False,'result':result,'receipt_elapsed_seconds':1.}
            c.save_new(self.folder/'request-00000001.json',receipt);return receipt
    monkeypatch.setattr(r,'PlannerWorker',Worker)
    row=r.decision_attempt('games/g000/t0000','local',case['snapshot'],observer,1e12,0,reserve=64800)
    assert row['status']=='forfeit_illegal' and not row['correct']
    assert row['result_sha256']==c.digest(result) and row['trace']['sha256']==c.file_hash(trace)


def test_audit_admission_preserves_analysis_and_finalization_reserve(tmp_path,monkeypatch):
    calls=[]
    def stop(base,deadline,**kw):calls.append(kw);raise c.ResourceHold('synthetic stop')
    monkeypatch.setattr(m,'wait_for_memory',stop)
    monkeypatch.setattr(r,'PlannerWorker',lambda *a,**kw:pytest.fail('No worker'))
    with pytest.raises(c.ResourceHold):r.audit_attempt('g000-t0000',{},'sample',1e12,0)
    assert calls==[{'name':'audits/g000-t0000','reserve':4200,'request_seconds':600}]


def test_run_rejects_wrong_expected_freeze_before_launch(tmp_path,monkeypatch):
    monkeypatch.setattr(c,'OUTPUT',tmp_path);(tmp_path/'freeze.json').write_text('{}')
    with pytest.raises(ValueError,match='Expected prepared freeze'):r.run('0'*64)
    assert not (tmp_path/'launch.json').exists()


def test_prepare_requires_exclusive_new_output_before_reading_history(tmp_path,monkeypatch):
    path=tmp_path/'config.json';path.write_text(json.dumps(CONFIG))
    base=tmp_path/'already';base.mkdir();(base/'preserve').write_bytes(b'original')
    monkeypatch.setattr(c,'OUTPUT',base)
    monkeypatch.setattr(r,'accepted_history',lambda:pytest.fail('No historical scan'))
    with pytest.raises(FileExistsError):r.prepare(path)
    assert (base/'preserve').read_bytes()==b'original'


class Clock:
    def __init__(self):self.value=0.
    def monotonic(self):return 100.+self.value
    def time(self):return 1000.+self.value
    def sleep(self,seconds):self.value+=seconds


def memory_fixture(tmp_path,monkeypatch,low=False):
    clock=Clock()
    monkeypatch.setattr(m.time,'time',clock.time);monkeypatch.setattr(m.time,'monotonic',clock.monotonic)
    monkeypatch.setattr(m.time,'sleep',clock.sleep)
    monkeypatch.setattr(m,'guard',lambda *a,**kw:{'remaining_seconds':200000.})
    def reading():
        return {'available_physical':1 if low and 10<=clock.value<20 else 2*1024**3,'available_page':4*1024**3}
    monkeypatch.setattr(m,'memory',reading);monkeypatch.setattr(m,'emit',lambda row:None)
    return clock


@pytest.mark.parametrize('low',[False,True])
def test_initial_memory_gate_has_complete_stable_window_and_configured_total(tmp_path,monkeypatch,low):
    clock=memory_fixture(tmp_path,monkeypatch,low)
    result=m.wait_for_memory(tmp_path,200100,name='initial',reserve=600,request_seconds=0,initial=True)
    assert result['status']=='passed' and result['charged_seconds']==(140. if low else 120.)
    assert result['maximum_cumulative_wait_seconds']==21600 and result['required_consecutive_samples']==13
    assert result['readings'][-1]['consecutive_admissible_samples']==13
    with pytest.raises(c.ResourceHold,match='already attempted'):
        m.wait_for_memory(tmp_path,200100,name='initial',reserve=600,request_seconds=0,initial=True)


def test_later_memory_gate_quick_pass_and_analysis_gate_names(tmp_path,monkeypatch):
    memory_fixture(tmp_path,monkeypatch)
    m.wait_for_memory(tmp_path,200100,name='initial',reserve=600,request_seconds=0,initial=True)
    for name in ('validation/v000','audits/validation/v000','games/g000/t0000','audits/g000-t0000','analysis'):
        result=m.wait_for_memory(tmp_path,200100,name=name,reserve=600,request_seconds=3600 if name=='analysis' else 120)
        assert result['status']=='passed' and result['charged_seconds']==0 and not result['stability_required']


def test_cumulative_memory_cap_not_reset_by_new_gate(tmp_path,monkeypatch):
    memory_fixture(tmp_path,monkeypatch)
    c.save_new(tmp_path/'memory_admissions/initial.json',{'status':'passed','charged_seconds':21600.})
    with pytest.raises(c.ResourceHold,match='allowance exhausted'):
        m.wait_for_memory(tmp_path,200100,name='analysis',reserve=600,request_seconds=3600)
    assert c.read(tmp_path/'memory_admissions/analysis.json')['status']=='preserved_hold'


def test_real_memory_guard_protects_analysis_start_budget(monkeypatch):
    monkeypatch.setattr(c,'boundary',lambda *a,**kw:{'remaining_seconds':4259.})
    with pytest.raises(c.ResourceHold,match='global work reserve'):
        m.guard(Path('synthetic'),1e12,600,3600)
    monkeypatch.setattr(c,'boundary',lambda *a,**kw:{'remaining_seconds':4260.})
    assert m.guard(Path('synthetic'),1e12,600,3600)['remaining_seconds']==4260.


@pytest.mark.parametrize('damage',['wall','mono','disk','size','physical','commit'])
def test_actual_global_boundary_rejects_each_bound(tmp_path,monkeypatch,damage):
    c.save_new(tmp_path/'launch.json',{'epoch_seconds':1000.,'maximum_seconds':CONFIG['execution_cap_seconds']})
    monkeypatch.setattr(c.time,'time',lambda:1000.+(259200 if damage=='wall' else 1))
    monkeypatch.setattr(c.time,'monotonic',lambda:100.+(1000 if damage=='mono' else 1))
    monkeypatch.setattr(c.shutil,'disk_usage',lambda _:SimpleNamespace(free=c.MIN_FREE-(1 if damage=='disk' else 0)))
    monkeypatch.setattr(c,'directory_bytes',lambda _:CONFIG['evidence_cap_bytes']-c.EVIDENCE_RESERVE if damage=='size' else 0)
    monkeypatch.setattr(c,'memory',lambda:{'available_physical':1024**3-(1 if damage=='physical' else 0),
                                         'available_page':3*1024**3-(1 if damage=='commit' else 0)})
    with pytest.raises(c.ResourceHold):c.boundary(tmp_path,1100.)


def test_ram_episode_limit_preserves_failure_without_worker(tmp_path,monkeypatch):
    clock=memory_fixture(tmp_path,monkeypatch)
    monkeypatch.setattr(m,'memory',lambda:{'available_physical':1,'available_page':4*1024**3})
    with pytest.raises(c.ResourceHold,match='allowance exhausted'):
        m.wait_for_memory(tmp_path,200100,name='initial',reserve=600,request_seconds=0,initial=True)
    saved=c.read(tmp_path/'memory_admissions/initial.json')
    assert saved['status']=='preserved_hold' and saved['charged_seconds']==7200.
    assert saved['no_worker_during_admission'] is True
