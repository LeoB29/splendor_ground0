"""Model-free fixed128 worker/transport/correctness regressions.

Saved outputs below are copied and explicitly reskinned as test fixtures only;
no learned network, native search process or evaluator is initialized.
"""
import ast
from collections import Counter
from copy import deepcopy
import inspect
import json
import textwrap
from types import SimpleNamespace
from unittest.mock import patch
import pytest
from tools.splendorust_search128_strength_20261028 import adapters as a,worker as w,scientific as s,corpus,qualification as q,transport as t,common as c
from tools.external_strength_screen_20261018_02.splendorust_adapter import Opponent
from tools.splendorust_search_scale_20261024.adapters import SearchOpponent
from tools.local_player_qualification_128.clock import PersistentWorker
from tools.splendorust_boundary_timing_20261025.parent_profile import ProfiledStudyWorker

def clock(elapsed=2.,status='ok'):
    return dict(clock='perf_counter_ns',request_id='request-00000001',deadline_seconds=10.,elapsed_seconds=elapsed,
        receipt_elapsed_seconds=elapsed,received_before_deadline=elapsed<=10.,accepted=status=='ok',status=status,
        result={'test_fixture':'raw'} if status=='ok' else None)

@pytest.mark.parametrize('elapsed',[.2,1.1,2.,8.9,9.5,10.])
def test_completed_clock_is_ten_without_projection(elapsed):
    value=clock(elapsed);before=deepcopy(value)
    assert s.check_clock(value)==elapsed and value==before

def test_late_or_clamped_clock_rejected():
    with pytest.raises(ValueError):s.check_clock(clock(10.001))
    value=clock(.2);value['deadline_seconds']=1.
    with pytest.raises(ValueError):s.check_clock(value)

@pytest.mark.parametrize('qv',[16,64,128])
def test_profiles_only_declared_scientific_and_clock_changes(qv):
    route='research' if qv==128 else 'frozen';item=a.original_profile_descriptor(route,qv)
    old=c.read(c.ROOT/item['path']);new=a.profile(qv,route=route)
    removed={'profile_id','wrapper_id','internal_budget_seconds','execution_policy','iterations'}
    assert {k:v for k,v in old.items() if k not in removed}=={k:v for k,v in new.items() if k not in removed}
    assert new['iterations']==qv and new['world_pool']==3 and new['depth']==64 and new['internal_budget_seconds']==9.
    assert new['execution_policy']['changed_scientific_fields']==(['iterations'] if qv==128 else [])
    assert new['execution_policy']['full_callback_capture_for_both_arms'] is True

@pytest.mark.parametrize('quota,pool,route',[(True,3,'frozen'),(32,3,'frozen'),(256,3,'research'),(64,6,'frozen'),(128,3,'frozen'),(64,3,'research')])
def test_unknown_or_mislabeled_scientific_condition_rejected(quota,pool,route):
    with pytest.raises(ValueError):a.profile_id(quota,pool,route=route)

def bare(quota=64):
    obj=object.__new__(a.BaselineOpponent);obj.profile={'iterations':quota,'internal_budget_seconds':9.}
    obj.request_deadline=None;obj.active_deadline=None;obj.admission_guard=None;obj.capture=False;obj.callbacks=[];obj.forward=None
    return obj

@pytest.mark.parametrize('quota',[16,64,128])
def test_direct_real_quota_and_guard_share_numerical_method(quota):
    obj=bare(quota);seen=[]
    def choose(self,observer):
        seen.append((self.profile['iterations'],self.active_deadline,self.capture));return 'action',{'science':'unchanged'}
    with patch.object(a.time,'perf_counter',return_value=100.),patch.object(Opponent,'choose',choose):
        assert obj.direct_choose({})==('action',{'science':'unchanged'})
    assert seen==[(quota,109.,True)] and obj.active_deadline is None and not obj.capture

@pytest.mark.parametrize('now,success',[(101.2,True),(108.999,True),(109.,False),(111.,False)])
def test_existing_request_origin_never_reset(now,success):
    obj=bare(128);obj.request_deadline=109.
    with patch.object(a.time,'perf_counter',return_value=now),patch.object(Opponent,'choose',lambda *args:('a',{})):
        if success:assert obj.choose({})==('a',{})
        else:
            with pytest.raises(TimeoutError):obj.choose({})
    assert obj.request_deadline==109.

def test_inference_ast_changes_only_local_graph_dispatch():
    old=ast.parse(textwrap.dedent(inspect.getsource(SearchOpponent._infer))).body[0]
    new=ast.parse(textwrap.dedent(inspect.getsource(a.BaselineOpponent._infer))).body[0]
    class Normalize(ast.NodeTransformer):
        def visit_Attribute(self,node):
            if isinstance(node.value,ast.Name) and node.value.id=='copy' and node.attr=='deepcopy':return ast.Name(id='deepcopy',ctx=ast.Load())
            return self.generic_visit(node)
    old=Normalize().visit(old);new.body[1]=deepcopy(old.body[1])
    assert ast.dump(old,include_attributes=False)==ast.dump(new,include_attributes=False)

def saved_result(quota=64,case='private_reserves'):
    path=c.ROOT/'outputs/splendorust_pool6_strengthfirst_20261028_qualification_01/entrypoint_checks/frozen.json'
    suite=c.read(path)['result']['normalized_suite'];raw=next(x for x in suite['fixtures'] if x['id']==case)['results']['search']
    result=deepcopy(raw);profile=a.profile(quota);route='frozen' if quota==64 else 'research'
    identity=dict(route=route,quota=quota,world_pool=3,profile_id=a.profile_id(quota),profile_sha256=a.profile_sha256(quota),
        wrapper_id=a.wrapper_id(quota),native_binary_sha256=a.SEARCH_BINARY_SHA256,internal_seconds=9.,seconds_per_move=10.)
    result.update(profile_id=identity['profile_id'],profile_sha256=identity['profile_sha256'],wrapper_id=identity['wrapper_id'],simulations_quota=quota)
    result['diagnostics']['profile_id']=identity['profile_id']
    observer=next(x for x in corpus.build_fixtures() if x['id']==case)
    observer=corpus.rules.observation_payload(corpus.guard.decode_training_snapshot(observer['snapshot']))
    return raw,result,observer,identity

def test_saved64_science_exact_identity_only_normalization():
    old,new,observer,identity=saved_result();before=deepcopy(new)
    assert a.validate_result(new,observer,'frozen',identity=identity)
    assert q.scientific_core(old)==q.scientific_core(new) and new==before

@pytest.mark.parametrize('field',['profile_id','profile_sha256','wrapper_id','simulations_quota'])
def test_corrupted_identity_rejected_before_scientific_checks(field):
    _,result,observer,identity=saved_result();result[field]='corrupt'
    with pytest.raises(ValueError):a.validate_result(result,observer,'frozen',identity=identity)

def test_synthetic128_work_is_checked_as128_not_relabelled64():
    _,result,observer,identity=saved_result(128)
    trace=result['diagnostics'];main=trace['phases'][0]
    # Explicit synthetic bookkeeping fixture, never actual128 model evidence.
    first=deepcopy(main['inference'][0]);short={k:first[k] for k in ('input_sha256','output_sha256','batch')}
    main['inference']=[first]+[deepcopy(short) for _ in range(128)]
    main.update(simulations=128,leaf_calls=129,neural_callback_count=129,neural_callback_rows=129,
        neural_callbacks=[c.digest([first['input_sha256'],first['output_sha256'],1])]*129)
    trace['inference_calls']=trace['inference_rows']=sum(p['neural_callback_count'] for p in trace['phases'])
    assert a.validate_result(result,observer,'research',identity=identity)
    main['simulations']=64
    with pytest.raises(ValueError,match='simulation quota'):a.validate_result(result,observer,'research',identity=identity)

def test_missing_full_root_arrays_is_not_symmetric_evidence():
    _,result,observer,identity=saved_result();first=result['diagnostics']['phases'][0]['inference'][0]
    for name in ('tokens','outputs','output83'):first.pop(name)
    with pytest.raises(ValueError,match='Full first'):a.validate_result(result,observer,'frozen',identity=identity)

def test_historical_projection_never_changes_quota_or_callback_hashes():
    _,result,_,_=saved_result();before=deepcopy(result);core=q.scientific_core(result,capture=False)
    assert result==before and core['simulations_quota']==64
    assert core['diagnostics']['phases'][0]['neural_callbacks']==result['diagnostics']['phases'][0]['neural_callbacks']
    assert 'inference' not in core['diagnostics']['phases'][0]
    other=deepcopy(result);other['simulations_quota']=128
    assert q.scientific_core(other,capture=False)!=core

def test_startup_exactly_once_guard_before_model():
    seen=[]
    def guard():seen.append('guard');raise ValueError('stop before model')
    with patch.object(w,'verify_runtime_sources',guard),patch.object(a,'initialize_adapters',side_effect=AssertionError('model forbidden')):
        with pytest.raises(ValueError,match='stop before model'):w.initialize('research')
    assert seen==['guard']
    nodes=ast.walk(ast.parse(inspect.getsource(w.initialize)))
    assert sum(isinstance(n,ast.Call) and isinstance(n.func,ast.Name) and n.func.id=='verify_runtime_sources' for n in nodes)==1

def test_phase_rejection_before_model(tmp_path,monkeypatch):
    folder=tmp_path/'worker';folder.mkdir();path=tmp_path/'freeze.json';path.write_text(json.dumps({'phase':'wrong'}))
    monkeypatch.setattr(c,'OUT',tmp_path);monkeypatch.setenv('STUDY_WORKER_DIR',str(folder));monkeypatch.setenv('STUDY_BOUNDARY_PROFILE','1')
    monkeypatch.setenv('STUDY_RESEARCH_FREEZE',str(path));monkeypatch.setenv('STUDY_RESEARCH_FREEZE_SHA256',c.file_hash(path))
    monkeypatch.setenv('SPLENDOR_SEARCH128_PHASE','strength')
    with patch.object(w,'verify_runtime_sources',return_value={}),patch.object(a,'initialize_adapters',side_effect=AssertionError('model forbidden')):
        with pytest.raises(ValueError,match='phase'):w.initialize('research')

def test_original_two_field_parser_rejects_extra_seed():
    with pytest.raises(ValueError,match='allowlist'):w._ordinary(dict(route='research',request_started=100.,diagnostic_capture=False),dict(player='search',observer={},seed=1))

@pytest.mark.parametrize('route,quota',[('frozen',64),('research',128)])
def test_ordinary_captures_both_routes_and_accepts_slow_completion(route,quota):
    _,result,observer,identity=saved_result(quota);obj=SimpleNamespace(request_deadline=None)
    def choose(obs,diagnostic_capture):
        assert obs==observer and diagnostic_capture is True and obj.request_deadline==109.
        return corpus.rules.deserialize_action(result['action']),result['diagnostics']
    obj.choose=choose;context=dict(route=route,request_started=100.,diagnostic_capture=False,
        adapters={'search':obj},execution_identities={quota:identity})
    with patch.object(w.time,'perf_counter',return_value=102.):
        assert w._ordinary(context,dict(player='search',observer=observer))==result
    with patch.object(w.time,'perf_counter',return_value=109.):
        with pytest.raises(TimeoutError,match='complete response'):w._ordinary(context,dict(player='search',observer=observer))
    assert obj.request_deadline is None

def test_transport_inherits_trusted_clock_and_reader_methods():
    assert t.PlacedProfiledStudyWorker.request is ProfiledStudyWorker.request
    for name in ('_receive','_stdout_reader','_stderr_reader'):
        assert getattr(t.PlacedProfiledStudyWorker,name) is getattr(ProfiledStudyWorker,name)
    src=inspect.getsource(t.make_worker)
    assert 'max_message_bytes=4*c.MiB' in src and 'max_log_bytes=128*c.MiB' in src
    assert "module = OBSERVER_MODULE" in src and 'str(adapters.route_quota(route))' in src

def test_no_frozen_global_rebinding():
    for mod in (a,w,s,corpus,q,t):
        for node in ast.walk(ast.parse(inspect.getsource(mod))):
            if isinstance(node,ast.Assign):
                for target in node.targets:
                    if isinstance(target,ast.Attribute) and isinstance(target.value,ast.Name):
                        assert target.value.id not in {'Opponent','SearchOpponent','frozen','original','graph'}

def test_qualification_fixed_inventory_and_repeats():
    plan=q.build_plan();rows=plan['parity_requests']
    assert (plan['complete_decisions'],plan['ordinary_requests'],plan['total_parent_requests'],plan['graph_cases'])==(203,84,88,22)
    assert plan['entrypoint_requests'][1]['direct_same_quota']==15
    assert Counter(r['kind'] for r in rows)==dict(saved_reproduction=24,fresh_reserved=36,live_edge=12,exact_repeat=12)
    for row in rows:
        assert row['quota']==a.route_quota(row['route']) and row['world_pool']==3
        if row['kind']=='exact_repeat':assert rows[row['repeat_of']]['observer']==row['observer'] and rows[row['repeat_of']]['route']==row['route']
    assert plan['timing_margins_are_descriptive'] and not hasattr(corpus,'validate_timed_panel')

def test_corrupt_full_prefix_inventory_rejected_without_model():
    plan=q.build_plan()
    with pytest.raises(ValueError,match='Complete84'):q.validate_parity_rows([],plan)
    proof=q.validate_parity_rows([],plan,allow_prefix=True)
    assert proof['status']=='verified_prefix' and not proof['complete']

def test_original52_counterexample_inventory_and_direct128_definition():
    for quota in (64,128):
        plan=corpus.build_plan(quota)
        assert len(plan['counterexamples'])==15 and plan['actual_complete_decisions']['total']==52
        assert plan['worker_rejection_attempts']==26 and plan['native_operations']['fresh_numeric_parity']==46
    source=inspect.getsource(corpus.run_entrypoint_checks)
    assert "if quota==128:" in source and "direct_choose(observer,diagnostic_capture=True)" in source
    assert "'Direct actual128 native/host action and full callback output parity'" in source

def test_all26_actual_handler_rejections_reach_new_parser_model_free():
    private=next(x for x in corpus.build_fixtures() if x['id']=='private_reserves')
    state=corpus.guard.decode_training_snapshot(private['snapshot'])
    context=dict(route='research',request_started=None,diagnostic_capture=True,graph_parity_passed=True,
        boundary_handle_active=False,component_profiles=[],adapters={})
    evidence=[]
    rows=corpus.worker_rejections(lambda request:w.handle(context,request),state,evidence_callback=evidence.append)
    assert len(rows)==26 and len(context['component_profiles'])==26
    assert all(x['handle_failed'] for x in context['component_profiles'])
    assert sum(e['event']=='rejected' for e in evidence)==26

@pytest.mark.parametrize('turn,quota,calls,leaves,simulations',[(0,128,129,129,128),(123,128,1,129,128),(124,128,0,0,0)])
def test128_branch_aware_work_preserves_late_threshold(turn,quota,calls,leaves,simulations):
    phase=dict(heuristic_phase=turn>=124,simulations=simulations,leaf_calls=leaves,
        neural_callbacks=['0'*64]*calls,neural_callback_count=calls,neural_callback_rows=calls)
    proof=s.check_search_phase(dict(turns=turn,phase={'kind':'Main'}),[{},{}],phase,quota)
    assert proof['inference_calls']==calls and proof['simulations']==simulations
