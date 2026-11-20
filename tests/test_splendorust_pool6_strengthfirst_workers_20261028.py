"""Model-free execution-policy tests; never initialize a learned model/native process."""
import ast
from copy import deepcopy
import inspect
import json
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch
import pytest
from tools.splendorust_pool6_strengthfirst_20261028 import adapters as a, worker as w, qualification as q, clock_policy as cp, common as c
from tools.splendorust_search_scale_20261024.adapters import SearchOpponent
from tools.external_strength_screen_20261018_02.splendorust_adapter import Opponent


def clock(elapsed=2.,status='ok'):
    return dict(clock='perf_counter_ns',request_id='request-00000001',deadline_seconds=10.,elapsed_seconds=elapsed,
        receipt_elapsed_seconds=elapsed,received_before_deadline=elapsed<=10.,accepted=status=='ok',status=status,
        result={'kept':'raw'} if status=='ok' else None)


@pytest.mark.parametrize('elapsed',[.2,1.1,2.,8.9,9.5,10.])
def test_parent_accepts_completed_before_ten_without_projection(elapsed):
    value=clock(elapsed);before=deepcopy(value)
    assert cp.check_clock(value)==elapsed and value==before


def test_parent_never_accepts_after_ten_or_clamped_clock():
    with pytest.raises(ValueError):cp.check_clock(clock(10.001))
    value=clock(.2);value['deadline_seconds']=1.
    with pytest.raises(ValueError):cp.check_clock(value)
    value=clock(10.01,'timeout');value['receipt_elapsed_seconds']=None
    assert cp.check_clock(value) is None


@pytest.mark.parametrize('now,accepted',[(101.1,True),(108.999,True),(109.,False),(110.,False)])
def test_original_origin_nine_second_internal_guard(now,accepted):
    with patch.object(cp.time,'perf_counter',return_value=now):
        if accepted:cp.boundary(109.)
        else:
            with pytest.raises(TimeoutError,match='STRENGTH_INTERNAL_DEADLINE'):cp.boundary(109.)


@pytest.mark.parametrize('route,quota',[('frozen',16),('frozen',64),('research',16),('research',64)])
def test_profiles_change_only_declared_execution_fields(route,quota):
    item=a.original_profile_descriptor(route,quota);previous=c.read(c.ROOT/item['path'])
    current=a.profile(quota,route=route)
    assert current['iterations']==quota and current['internal_budget_seconds']==9.
    assert current['execution_policy']['additional_search_when_time_remains'] is False
    removed={'profile_id','wrapper_id','internal_budget_seconds','execution_policy'}
    assert {k:v for k,v in current.items() if k not in removed}=={k:v for k,v in previous.items() if k not in removed}
    assert current['execution_policy']['original_profile']==item
    assert previous['internal_budget_seconds']==.9


@pytest.mark.parametrize('quota,pool,route',[(32,3,'frozen'),(256,6,'research'),(64,3,'research'),(16,6,'research'),(True,3,'frozen')])
def test_unapproved_quota_or_pool_rejected(quota,pool,route):
    with pytest.raises(ValueError):a.profile_id(quota,pool,route=route)


def bare_baseline(quota=64):
    obj=object.__new__(a.BaselineOpponent);obj.profile={'iterations':quota,'internal_budget_seconds':9.}
    obj.request_deadline=None;obj.active_deadline=None;obj.admission_guard=None;obj.capture=False;obj.callbacks=[];obj.forward=None
    return obj


def test_baseline_nested_profile_and_direct_reference_use_nine():
    obj=bare_baseline(16);seen=[]
    def choose(self,observer):
        seen.append((self.active_deadline,self.profile['internal_budget_seconds'],self.capture))
        return 'action',{'unchanged':'science'}
    with patch.object(a.time,'perf_counter',return_value=100.),patch.object(Opponent,'choose',choose):
        assert obj.reference_choose({})==('action',{'unchanged':'science'})
    assert seen==[(109.,9.,True)] and obj.active_deadline is None and not obj.capture


def test_baseline_does_not_reset_existing_request_origin():
    obj=bare_baseline();obj.request_deadline=109.
    with patch.object(a.time,'perf_counter',return_value=108.),patch.object(Opponent,'choose',lambda s,o:('a',{})):
        assert obj.choose({})==('a',{})
    with patch.object(a.time,'perf_counter',return_value=109.):
        with pytest.raises(TimeoutError):obj.choose({})
    assert obj.request_deadline==109.


@pytest.mark.parametrize('prior',[None,109.])
def test_candidate_direct_and_nested_deadline_fallback(prior):
    obj=object.__new__(a.LiveOpponent);obj.request_deadline=prior;obj.profile={'iterations':64,'world_pool':6}
    seen=[]
    def old_choose(self,observer,diagnostic_capture=True):
        seen.append((self.request_deadline,diagnostic_capture));return 'a',{'profile_id':'old'}
    with patch.object(a.time,'perf_counter',return_value=100.),patch.object(a.live.LiveOpponent,'choose',old_choose):
        result=obj.choose({})
    assert seen==[(109.,True)] and obj.request_deadline is prior
    assert result[1]['profile_id']==a.profile_id(64,6)


def test_baseline_local_graph_has_exact_callback_science_without_global_binding():
    obj=bare_baseline();obj.capture=True;obj.forward=SimpleNamespace(infer=lambda *args:[[.25]*83])
    original=Opponent._infer
    got=obj._infer([[1.,2.]],1);new=deepcopy(obj.callbacks)
    obj.callbacks=[]
    with patch.object(Opponent,'_infer',lambda *args:[[.25]*83]):
        expected=SearchOpponent._infer(obj,[[1.,2.]],1)
    assert got==expected and obj.callbacks==new and Opponent._infer is original


def test_infer_ast_diff_is_only_instance_local_graph_dispatch():
    import textwrap
    original=ast.parse(textwrap.dedent(inspect.getsource(SearchOpponent._infer))).body[0]
    new=ast.parse(textwrap.dedent(inspect.getsource(a.BaselineOpponent._infer))).body[0]
    class Normalize(ast.NodeTransformer):
        def visit_Attribute(self,node):
            if isinstance(node.value,ast.Name) and node.value.id=='copy' and node.attr=='deepcopy':return ast.Name(id='deepcopy',ctx=ast.Load())
            return self.generic_visit(node)
    original=Normalize().visit(original);new.body[1]=deepcopy(original.body[1])
    assert ast.dump(original,include_attributes=False)==ast.dump(new,include_attributes=False)


def saved_result(route):
    matrix=c.read(q.MATRIX)
    spec=next(x for x in matrix['requests'] if x['route']==route and x['world_pool']==(3 if route=='frozen' else 6) and x['seed_label']=='A' and x['stage']!='research_reproduction')
    row=c.read(q.PRIOR/'records'/f"{spec['sequence']:04d}.json")
    old=deepcopy(row['clock']['result']);observer=row['observer']
    new=deepcopy(old);profile=a.profile(64,route=route)
    identity=dict(route=route,quota=64,world_pool=3 if route=='frozen' else 6,profile_id=profile['profile_id'],
        profile_sha256='0'*64,wrapper_id=profile['wrapper_id'],native_binary_sha256=profile.get('native_binary_sha256','unused'),
        internal_seconds=9.,seconds_per_move=10.)
    new.update(profile_id=identity['profile_id'],profile_sha256=identity['profile_sha256'],wrapper_id=identity['wrapper_id'])
    new['diagnostics']['profile_id']=identity['profile_id']
    if route=='research':
        # Explicit test-only policy projection of accepted sampling telemetryON.
        new['research'].update(schema=a.SCHEMA,internal_seconds=9.,telemetry_enabled=False)
        for phase in new['research']['phases']:phase['telemetry']=None
    return old,new,observer,identity


@pytest.mark.parametrize('route',['frozen','research'])
def test_saved_science_identity_projection_preserves_every_callback(route):
    old,new,observer,identity=saved_result(route);before=deepcopy(new)
    projected=a.original_projection(new,observer,route,identity=identity)
    assert q.exact_scientific_outputs(projected)==q.exact_scientific_outputs(old)
    assert new==before
    assert a.validate_result(new,observer,route,identity=identity)
    if route=='research':assert [x['callback_bits'] for x in new['research']['phases']]==[x['callback_bits'] for x in old['research']['phases']]


@pytest.mark.parametrize('field',['profile_id','profile_sha256','wrapper_id','simulations_quota'])
def test_corrupted_identity_rejected_before_projection(field):
    old,new,observer,identity=saved_result('research');new[field]='corrupt'
    with pytest.raises(ValueError):a.original_projection(new,observer,'research',identity=identity)


@pytest.mark.parametrize('field,value',[('internal_seconds',.9),('quota',256),('world_pool',3),('telemetry_enabled',True),('schema','old')])
def test_raw_candidate_policy_corruption_rejected(field,value):
    _,new,observer,identity=saved_result('research');new['research'][field]=value
    with pytest.raises(ValueError):a.original_projection(new,observer,'research',identity=identity)


def test_startup_source_guard_runs_once_before_any_model():
    events=[]
    def guard():events.append('guard');raise ValueError('fixture stops before model')
    with patch.object(w,'verify_runtime_sources',guard),patch.object(a,'initialize_adapters',side_effect=AssertionError('model forbidden')):
        with pytest.raises(ValueError,match='fixture stops'):w.initialize('frozen')
    assert events==['guard']
    source=ast.parse(inspect.getsource(w.initialize))
    calls=[n for n in ast.walk(source) if isinstance(n,ast.Call) and isinstance(n.func,ast.Name) and n.func.id=='verify_runtime_sources']
    assert len(calls)==1


def test_new_worker_does_not_patch_frozen_globals_or_install_global_graph():
    for module in (w,a,q):
        tree=ast.parse(inspect.getsource(module))
        for node in ast.walk(tree):
            if isinstance(node,ast.Assign):
                for target in node.targets:
                    if isinstance(target,ast.Attribute) and isinstance(target.value,ast.Name):
                        assert target.value.id not in {'Opponent','SearchOpponent','frozen','live','original','old','graph'}
            if isinstance(node,ast.Call) and isinstance(node.func,ast.Attribute):assert node.func.attr!='install'


def test_full_interleaved_plan_and_deadlines_without_model():
    plan=q.build_plan();rows=plan['parity_requests']
    assert len(rows)==90 and plan['complete_decisions']==194 and plan['graph_cases']==22
    assert plan['maximum_simultaneous_qualification_trees']==2
    assert plan['ordinary_frozen_external_seconds']==plan['ordinary_research_external_seconds']==10
    assert plan['ordinary_internal_seconds']==9.
    pairs=[x for x in rows if x['kind']!='live_edge']
    assert len(pairs)==84
    for index in range(42):
        left,right=pairs[index*2:index*2+2]
        assert left['observer']==right['observer'] and {left['route'],right['route']}=={'frozen','research'}
        assert left['route']==('frozen' if index%2==0 else 'research')
    assert {x['source_plan_index'] for x in rows}==set(range(90))


def test_ordinary_negative_fields_reach_actual_parser():
    context={'route':'frozen','request_started':100.,'diagnostic_capture':False}
    with pytest.raises(ValueError,match='allowlist'):
        w._ordinary(context,{'player':'search','observer':{},'surplus':True})


def test_bad_phase_stops_before_model(tmp_path,monkeypatch):
    freeze={'phase':'assessment'};folder=tmp_path/'worker';folder.mkdir()
    path=tmp_path/'freeze.json';path.write_text(json.dumps(freeze))
    monkeypatch.setattr(c,'OUT',tmp_path);monkeypatch.setattr(c,'PHASE','qualification')
    monkeypatch.setenv('STUDY_WORKER_DIR',str(folder));monkeypatch.setenv('STUDY_BOUNDARY_PROFILE','1')
    monkeypatch.setenv('STUDY_RESEARCH_FREEZE',str(path));monkeypatch.setenv('STUDY_RESEARCH_FREEZE_SHA256',c.file_hash(path))
    monkeypatch.setenv('SPLENDOR_POOL6_STRENGTH_PHASE','qualification')
    with patch.object(w,'verify_runtime_sources',return_value={}),patch.object(a,'initialize_adapters',side_effect=AssertionError('model forbidden')):
        with pytest.raises(ValueError,match='phase'):w.initialize('research')
