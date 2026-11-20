"""Model-free live qualification, fixed configuration and shortcut regressions."""
import ast
from copy import deepcopy
import json
from pathlib import Path
from types import SimpleNamespace

import pytest

from tools.splendorust_pool6_live_20261027 import adapters as a,worker as w,qualification as q
from tools.splendorust_sampling_stability_20261027 import adapters as accepted
from tools.splendorust_search_scale_20261024 import qualification as original,worker as oldworker
from tools.future_training_rules_20261021 import decode_training_snapshot
from tools.external_strength_screen_20261018 import rules

ROOT=Path(__file__).resolve().parents[1]

def fixtures():return original.build_plan(64)['counterexamples']

def fixture(case):return next(x for x in fixtures() if x['id']==case)

def observer(case='turn124'):
    return rules.observation_payload(decode_training_snapshot(fixture(case)['snapshot']))

def context():
    return dict(diagnostic_capture=False,quota=64,graph_parity_passed=True,
        component_profiles=[],boundary_handle_active=False,request_started=None,
        adapters={},profiles={},profile_artifacts={},request_read_complete_ns=None)

@pytest.mark.parametrize('quota,pool',[(16,3),(64,6)])
def test_profiles_preserve_all_accepted_scientific_settings(quota,pool):
    live=a.profile(quota,pool)
    prior=json.loads((a.PRIOR/'freeze.json').read_text())['research_profiles'][f'{quota}-{pool}']
    old=json.loads((ROOT/prior['path']).read_text())
    changes={'profile_id','wrapper_id','purpose','seed_family','telemetry','artifacts',
        'source_sampling_profile_sha256','original_late_game_and_compound_behavior_preserved'}
    assert {k:v for k,v in live.items() if k not in changes}=={k:v for k,v in old.items() if k not in changes}
    assert live['iterations']==quota and live['world_pool']==pool and live['internal_budget_seconds']==.9
    assert live['source_sampling_profile_sha256']==prior['sha256']
    assert live['artifacts']==old['artifacts']+[prior]
    assert a.profile_path(quota,pool).parent.parent.name=='splendorust_pool6_live_20261027_qualification_01'

@pytest.mark.parametrize('quota,pool',[(64,3),(16,6),(32,3),(1024,6),(True,6),(64,6.)])
def test_only_the_two_frozen_live_role_profiles_exist(quota,pool):
    with pytest.raises(ValueError):a.profile_id(quota,pool)

def test_all_numerical_exchange_and_deadline_methods_are_unchanged_objects():
    for name in ('_infer','exchange','boundary','close'):
        assert getattr(a.LiveOpponent,name) is getattr(accepted.ResearchOpponent,name)
    for name in ('inspect','native_reference'):
        assert not hasattr(a.LiveOpponent,name) and not hasattr(accepted.ResearchOpponent,name)

@pytest.mark.parametrize('quota,pool',[(16,3),(64,6)])
def test_live_choose_forces_exact_configuration_and_only_changes_identity(quota,pool,monkeypatch):
    inst=a.LiveOpponent.__new__(a.LiveOpponent);inst.profile={'iterations':quota,'world_pool':pool}
    calls=[];action=object()
    trace={'profile_id':'old','observer_seed':7,'phases':[{'action':'unchanged'}]}
    def choose(self,view,**kwargs):calls.append((view,kwargs));return action,deepcopy(trace)
    monkeypatch.setattr(accepted.ResearchOpponent,'choose',choose)
    returned,recorded=inst.choose({'visible':'only'},diagnostic_capture=False)
    assert returned is action
    assert calls==[({'visible':'only'},dict(diagnostic_capture=True,seed_label='A',telemetry=False,quota=quota,world_pool=pool))]
    assert recorded==dict(trace,profile_id=a.profile_id(quota,pool))

@pytest.mark.parametrize('label',['B','C','D','F',None,True])
def test_no_alternative_seed_can_enter_live_player(label):
    with pytest.raises(ValueError):a.observer_seed(observer(),label)

@pytest.mark.parametrize('case',['turn123','turn124','forced_pass','gold_returns','multiple_nobles','noble_final_round'])
def test_actual_live_handler_accepts_late_compound_and_forced_cases_without_models(case,monkeypatch):
    view=observer(case);state=w.decode_observer(view)
    choice=rules.CorrectedEnv().legal_actions(state)[0]
    calls=[]
    def choose(actual,**kwargs):
        calls.append((actual,kwargs));return choice,dict(profile_id=a.profile_id(64),observer_seed=a.observer_seed(view),phases=[])
    inst=SimpleNamespace(choose=choose,last_research_phases=[],request_deadline=None)
    ctx=context();ctx.update(adapters={'search':inst},profiles={'64-6':{'native_binary_sha256':'f'*64}},
        profile_artifacts={'64-6':{'sha256':'e'*64}},request_started=10.)
    monkeypatch.setattr(w.time,'perf_counter',lambda:10.5)
    result=w._ordinary(ctx,{'player':'search','observer':view})
    assert calls==[(view,{'diagnostic_capture':True})]
    assert result['research']['world_pool']==6 and result['research']['seed_label']=='A'
    assert result['research']['telemetry_enabled'] is False
    assert result['action']==rules.serialize_action(choice) and inst.request_deadline is None

@pytest.mark.parametrize('field,value',[('quota',64),('world_pool',3),('seed_label','B'),('telemetry',True)])
def test_caller_cannot_change_live_configuration(field,value):
    with pytest.raises(ValueError,match='request allowlist'):
        w._ordinary(context(),{'player':'search','observer':observer(),field:value})

def test_reference_role_is_qualification_only():
    with pytest.raises(ValueError,match='arm allowlist'):
        w._ordinary(context(),{'player':'reference','observer':observer()})

def test_actual26_original_reject_paths_remain_end_to_end_without_model():
    ctx=context();ctx['diagnostic_capture']=True
    state=decode_training_snapshot(fixture('private_reserves')['snapshot'])
    rows=original.worker_rejections(lambda request:w.handle(ctx,request),state)
    assert len(rows)==26 and len(ctx['component_profiles'])==26
    assert all(p['handle_failed'] for p in ctx['component_profiles'])

def test_startup_guard_before_models_and_no_imported_global_mutation():
    source=Path(w.__file__).read_text();tree=ast.parse(source)
    init=next(n for n in tree.body if isinstance(n,ast.FunctionDef) and n.name=='initialize')
    calls=[n for n in ast.walk(init) if isinstance(n,ast.Call)]
    guards=[n for n in calls if ast.unparse(n.func)=='verify_runtime_sources']
    assert len(guards)==1
    assert guards[0].lineno<min(n.lineno for n in calls if ast.unparse(n.func)=='adapters.initialize_adapters')
    for prohibited in ('Opponent._infer =','accepted.ResearchOpponent.choose =','state.turn_index<124'):
        assert prohibited not in source+Path(a.__file__).read_text()

@pytest.mark.parametrize('bad',['freeze','environment'])
def test_phase_identity_fails_before_model_and_guard_runs_once(bad,monkeypatch,tmp_path):
    from tools.splendorust_pool6_live_20261027 import common as c
    out=tmp_path/'qualification';out.mkdir();path=out/'freeze.json'
    path.write_text(json.dumps({'phase':'assessment' if bad=='freeze' else 'qualification'}))
    monkeypatch.setattr(c,'OUT',out);monkeypatch.setattr(c,'PHASE','qualification')
    monkeypatch.setenv('STUDY_WORKER_DIR',str(out/'worker'))
    monkeypatch.setenv('STUDY_BOUNDARY_PROFILE','1')
    monkeypatch.setenv('STUDY_RESEARCH_FREEZE',str(path))
    monkeypatch.setenv('STUDY_RESEARCH_FREEZE_SHA256',c.file_hash(path))
    monkeypatch.setenv('SPLENDOR_POOL6_PHASE','assessment' if bad=='environment' else 'qualification')
    guards=[]
    monkeypatch.setattr(w,'verify_runtime_sources',lambda:guards.append('checked') or {})
    monkeypatch.setattr(a,'initialize_adapters',lambda *args:pytest.fail('model initialization must remain unreachable'))
    with pytest.raises(ValueError,match='phase/freeze/environment'):w.initialize()
    assert guards==['checked']

def test_original_boundary_clock_failure_keeps_partial_evidence(monkeypatch):
    original_handle=oldworker.handle;ctx=context()
    monkeypatch.setattr(w,'_ordinary',lambda *args:(_ for _ in ()).throw(TimeoutError('SAMPLING_INTERNAL_DEADLINE: original')))
    with pytest.raises(TimeoutError):w.handle(ctx,{'player':'search','observer':observer()})
    assert len(ctx['component_profiles'])==1 and ctx['component_profiles'][0]['handle_failed']
    assert oldworker.handle is original_handle and not ctx['boundary_handle_active']

def test_full_callback_capture_is_retained_in_exact_reproduction_projection():
    result=dict(profile_id='live',profile_sha256='a',wrapper_id='id',research={'telemetry':False},
        diagnostics=dict(profile_id='live',phases=[{'inference':[{'output83':[1.,2.]}],'neural_callbacks':['sha']}]))
    view=q.exact_scientific_outputs(result)
    assert view['diagnostics']['phases'][0]['inference']==[{'output83':[1.,2.]}]
    assert result['profile_id']=='live' and 'research' not in view

def test_bundled_handler_count_and_reference_calls_are_not_fabricated(monkeypatch,tmp_path):
    ctx=context();ctx['folder']=tmp_path
    def ordinary(c,request):
        if request.get('bad'):raise ValueError('reject')
        return {'simulations_quota':64}
    def checks(c,callback,**kwargs):
        for i in range(37):callback({'case':i})
        for i in range(26):
            with pytest.raises(ValueError):callback({'bad':i+1})
        return {'counts':{'complete_decisions':52}}
    monkeypatch.setattr(w,'_ordinary',ordinary);monkeypatch.setattr(q,'run_entrypoint_checks',checks)
    result=w.handle(ctx,{'operation':'entrypoint_checks'})
    assert result['counts']['complete_decisions']==52 and len(ctx['component_profiles'])==63
    assert sum(p['handle_failed'] for p in ctx['component_profiles'])==26

def test_new_raw_live_annotations_reject_telemetry_or_pool_drift():
    view=observer('forced_pass');seed=a.observer_seed(view)
    result=dict(simulations_quota=64,diagnostics=dict(observer_seed=seed,phases=[]),research=dict(
        schema=a.SCHEMA,quota=64,world_pool=6,seed_label='A',seed=seed,telemetry_enabled=False,
        internal_seconds=.9,native_binary_sha256='f'*64,phases=[]))
    q.validate_raw_qualification_annotations(result,view,{'research_native_sha256':'f'*64})
    for field,value in [('world_pool',3),('seed_label','B'),('telemetry_enabled',True),('internal_seconds',29.)]:
        bad=deepcopy(result);bad['research'][field]=value
        with pytest.raises(ValueError):q.validate_raw_qualification_annotations(bad,view,{'research_native_sha256':'f'*64})

def test_fixed_plan_194_science90_timed48reproductions6edges36reserved():
    plan=q.build_plan();rows=plan['parity_requests']
    assert plan['complete_decisions']==194 and plan['graph_cases']==22 and len(rows)==90
    assert [r['index'] for r in rows]==list(range(90))
    assert sum(r['route']=='frozen' for r in rows)==42 and sum(r['route']=='research' for r in rows)==48
    assert sum(r['kind']=='saved_reproduction' for r in rows)==48
    assert sum(r['kind']=='live_edge' for r in rows)==6
    assert sum(r['kind']=='fresh_reserved' for r in rows)==36
    assert sum(r.get('exact_frozen_shortcut',False) for r in rows)==1
    assert {r['seed_label'] for r in rows}=={'A'} and all(r['telemetry'] is False for r in rows)

def test_original_saved_entrypoint_remains_valid_with_same52_actual_checks():
    receipt=json.loads((a.PRIOR/'entrypoint_checks/frozen.json').read_text())
    summary=q.verify_entrypoint(a.PRIOR/'workers/qualification_frozen',receipt,q.build_plan(),{})
    assert summary['complete_decisions']==52 and summary['ordinary_handlers']==37
    assert summary['negative_handler_cases']==26 and summary['native_numeric_parity']==46

def test_response_bound_and_nonfinite_rejection():
    assert json.loads(w.response_packet({'safe':True}))=={'safe':True}
    with pytest.raises(ValueError):w.response_packet({'too_large':'x'*w.MAX_RESPONSE_BYTES})
    with pytest.raises(ValueError):w.response_packet({'nan':float('nan')})

def qualification_rows_without_models():
    """Authentic stored output fixtures plus explicit synthetic reserved receipts.

    The latter exercise inventory/observer binding only; online scientific and
    raw-clock validation is a separate mandatory parent responsibility.
    """
    plan=q.build_plan();rows=[]
    for spec in plan['parity_requests']:
        if spec['kind']=='saved_reproduction':
            result=json.loads((ROOT/spec['expected_record']['path']).read_text())['clock']['result']
        elif spec['kind']=='live_edge':
            previous=json.loads((ROOT/spec['expected_entrypoint']['path']).read_text())
            result=deepcopy(next(x for x in previous['result']['normalized_suite']['fixtures'] if x['id']==spec['case_id'])['results']['search'])
        else:
            result=dict(observer_sha256=spec['observer_sha256'],simulations_quota=64,
                diagnostics=dict(observer_seed=a.observer_seed(spec['observer']),phases=[]))
        if spec['route']=='research':
            result['profile_id']=a.profile_id(64);result['wrapper_id']=a.wrapper_id(64)
            result['diagnostics']['profile_id']=a.profile_id(64)
            if 'research' not in result:
                result['research']=dict(phases=[{'telemetry':None} for _ in result['diagnostics']['phases']])
            result['research'].update(world_pool=6,seed_label='A',telemetry_enabled=False)
            for phase in result['research']['phases']:phase['telemetry']=None
        rows.append(dict(spec=spec,clock=dict(status='ok',result=result)))
    current_frozen=json.loads((a.PRIOR/'entrypoint_checks/frozen.json').read_text())
    return rows,plan,current_frozen

def test_saved_full_outputs_and_late_heuristic_are_independently_rechecked():
    rows,plan,baseline=qualification_rows_without_models()
    result=q.validate_parity_rows(rows,plan,frozen_entrypoint=baseline)
    assert result['counts']==dict(frozen_saved=24,candidate_saved=24,live_edges=6,fresh_reserved=36,exact_frozen_late_shortcuts=1)
    assert result['exact_recorded_numeric_outputs'] and result['exact_candidate_callback_bits']

@pytest.mark.parametrize('change',['array','bits','late','pool','telemetry','missing','order'])
def test_live_parity_rejects_changed_saved_outputs_configuration_and_inventory(change):
    rows,plan,baseline=qualification_rows_without_models()
    candidate=next(r for r in rows if r['spec']['route']=='research' and r['spec']['kind']=='saved_reproduction')
    result=candidate['clock']['result']
    if change=='array':result['diagnostics']['phases'][0]['inference'][0]['output83'][0]+=.01
    if change=='bits':result['research']['phases'][0]['callback_bits'][0]['output_float32_sha256']='0'*64
    if change=='late':
        next(x for x in baseline['result']['fixtures'] if x['id']=='turn124')['results']['search']['diagnostics']['forced_pass']=True
    if change=='pool':result['research']['world_pool']=3
    if change=='telemetry':result['research']['telemetry_enabled']=True
    if change=='missing':rows.pop()
    if change=='order':rows[0],rows[1]=rows[1],rows[0]
    with pytest.raises(ValueError):q.validate_parity_rows(rows,plan,frozen_entrypoint=baseline)
