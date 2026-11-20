"""Model-free fresh admission facade, unchanged science and prospective ordering."""
import ast
from copy import deepcopy
import json
from pathlib import Path
from types import SimpleNamespace

import pytest

from tools.splendorust_pool6_admission_20261027_02 import adapters as a, worker as w, qualification as q, common as c, design
from tools.splendorust_pool6_live_20261027 import adapters as olda, worker as oldw, qualification as oldq
from tools.splendorust_search_scale_20261024 import qualification as original
from tools.future_training_rules_20261021 import decode_training_snapshot
from tools.external_strength_screen_20261018 import rules


def functions(module):
    return {n.name: n for n in ast.parse(Path(module.__file__).read_text()).body
            if isinstance(n, ast.FunctionDef)}


def fixture(case):
    return next(x for x in original.build_plan(64)['counterexamples'] if x['id'] == case)


def observer(case='turn124'):
    return rules.observation_payload(decode_training_snapshot(fixture(case)['snapshot']))


def context():
    return dict(diagnostic_capture=False, quota=64, graph_parity_passed=True,
        component_profiles=[], boundary_handle_active=False, request_started=None,
        adapters={}, profiles={}, profile_artifacts={}, request_read_complete_ns=None)


def test_worker_only_changes_entrypoint_phase_binding():
    old, new = functions(oldw), functions(w)
    assert old.keys() == new.keys()
    for name in old:
        if name == 'initialize':
            text = ast.unparse(old[name]).replace("'SPLENDOR_POOL6_PHASE', 'qualification'",
                "'SPLENDOR_POOL6_ADMISSION_PHASE', 'qualification'")
            assert ast.dump(ast.parse(text), include_attributes=False) == ast.dump(
                ast.parse(ast.unparse(new[name])), include_attributes=False)
        else:
            assert ast.dump(old[name], include_attributes=False) == ast.dump(new[name], include_attributes=False)


def test_exact_class_and_scientific_functions_are_reused_without_mutation():
    for name in ('LiveOpponent', 'initialize_adapters', 'ordinary_payload', 'observer_seed',
                 'pool_for', 'profile_id', 'wrapper_id'):
        assert getattr(a, name) is getattr(olda, name)
    for name in ('run_entrypoint_checks', 'verify_entrypoint', 'validate_raw_qualification_annotations'):
        assert getattr(q, name) is getattr(oldq, name)
    for name in ('validate_ready', 'verify_saved_ready', 'graph_parity', 'actual_entrypoint'):
        assert ast.dump(functions(a)[name], include_attributes=False) == ast.dump(functions(olda)[name], include_attributes=False)


@pytest.mark.parametrize('quota,pool', [(16, 3), (64, 6)])
def test_profiles_are_exact_immutable_prior_files(quota, pool):
    item = c.read(a.PRIOR/'freeze.json')['research_profiles'][f'{quota}-{pool}']
    assert a.profile_descriptor(quota, pool) == item
    assert a.profile_path(quota, pool) == c.ROOT/item['path']
    assert a.profile_sha256(quota, pool) == item['sha256']
    assert a.profile(quota, pool) == c.read(c.ROOT/item['path'])
    assert a.SCHEMA == olda.SCHEMA and a.INTERNAL_SECONDS == .9


@pytest.mark.parametrize('bad', ['freeze_phase', 'environment', 'output_boundary', 'freeze_hash'])
def test_new_startup_guard_fails_closed_before_models(bad, monkeypatch, tmp_path):
    out=tmp_path/'study';out.mkdir();path=out/'freeze.json'
    path.write_text(json.dumps({'phase': 'assessment' if bad=='freeze_phase' else 'qualification'}))
    monkeypatch.setattr(c,'OUT',out)
    monkeypatch.setenv('STUDY_WORKER_DIR',str(tmp_path/'other' if bad=='output_boundary' else out/'worker'))
    monkeypatch.setenv('STUDY_BOUNDARY_PROFILE','1')
    monkeypatch.setenv('STUDY_RESEARCH_FREEZE',str(path))
    monkeypatch.setenv('STUDY_RESEARCH_FREEZE_SHA256','0'*64 if bad=='freeze_hash' else c.file_hash(path))
    monkeypatch.setenv('SPLENDOR_POOL6_ADMISSION_PHASE','assessment' if bad=='environment' else 'qualification')
    calls=[]
    monkeypatch.setattr(w,'verify_runtime_sources',lambda:calls.append('guard') or {})
    monkeypatch.setattr(a,'initialize_adapters',lambda *args:pytest.fail('No model construction permitted'))
    with pytest.raises(ValueError):w.initialize()
    assert calls==['guard']


def test_guard_exactly_once_before_model_and_no_old_global_rebinding():
    node=functions(w)['initialize']
    calls=[n for n in ast.walk(node) if isinstance(n,ast.Call)]
    guard=[n for n in calls if ast.unparse(n.func)=='verify_runtime_sources']
    assert len(guard)==1 and guard[0].lineno < next(n.lineno for n in calls if ast.unparse(n.func)=='adapters.initialize_adapters')
    for module in (a,w,q):
        for n in ast.walk(ast.parse(Path(module.__file__).read_text())):
            if isinstance(n,ast.Assign):
                assert not any(isinstance(t,ast.Attribute) and isinstance(t.value,ast.Name)
                    and t.value.id in ('immutable','olda','oldw','oldq') for t in n.targets)


@pytest.mark.parametrize('case', ['turn123','turn124','forced_pass','gold_returns','multiple_nobles','noble_final_round'])
def test_additive_actual_handler_keeps_compound_and_late_routes(case,monkeypatch):
    view=observer(case);choice=rules.CorrectedEnv().legal_actions(w.decode_observer(view))[0]
    calls=[]
    def choose(actual,**kwargs):
        calls.append((actual,kwargs));return choice,dict(profile_id=a.profile_id(64),observer_seed=a.observer_seed(view),phases=[])
    instance=SimpleNamespace(choose=choose,last_research_phases=[],request_deadline=None)
    ctx=context();ctx.update(adapters={'search':instance},profiles={'64-6':{'native_binary_sha256':'f'*64}},
        profile_artifacts={'64-6':{'sha256':'e'*64}},request_started=10.)
    monkeypatch.setattr(w.time,'perf_counter',lambda:10.5)
    result=w.handle(ctx,{'player':'search','observer':view})
    assert calls==[(view,{'diagnostic_capture':True})]
    assert result['action']==rules.serialize_action(choice)
    assert result['research']['world_pool']==6 and result['research']['telemetry_enabled'] is False
    assert instance.request_deadline is None and len(ctx['component_profiles'])==1


def test_actual26_negative_entrypoint_routes_remain_unchanged():
    ctx=context();ctx['diagnostic_capture']=True
    state=decode_training_snapshot(fixture('private_reserves')['snapshot'])
    failures=original.worker_rejections(lambda request:w.handle(ctx,request),state)
    assert len(failures)==26 and len(ctx['component_profiles'])==26
    assert all(p['handle_failed'] for p in ctx['component_profiles'])


@pytest.mark.parametrize('extra', [{'quota':64},{'world_pool':3},{'seed_label':'B'},{'telemetry':True}])
def test_live_payload_cannot_change_players(extra):
    with pytest.raises(ValueError,match='request allowlist'):
        w.handle(context(),dict(player='search',observer=observer(),**extra))


def test_original_failure_marker_and_partial_sidecar_are_preserved(monkeypatch):
    ctx=context()
    def fail(*args):raise TimeoutError('SAMPLING_INTERNAL_DEADLINE: fixed guard')
    monkeypatch.setattr(w,'_ordinary',fail)
    with pytest.raises(TimeoutError,match='fixed guard'):w.handle(ctx,{'player':'search','observer':observer()})
    assert ctx['component_profiles'][0]['handle_failed'] and not ctx['boundary_handle_active']


def test_plan_preserves_old_science_and_replaces_only_fresh_inputs_and_order():
    plan=q.build_plan();old,_=q._old_plan();rows=plan['parity_requests']
    assert plan['complete_decisions']==194 and plan['graph_cases']==22 and len(rows)==90
    assert plan['maximum_simultaneous_qualification_trees']==2
    assert [r['index'] for r in rows]==list(range(90))
    assert sorted(r['historical_plan_index'] for r in rows)==list(range(90))
    assert sum(r['route']=='frozen' for r in rows)==42
    assert sum(r['route']=='research' for r in rows)==48
    changed=0
    for r in rows:
        prior=deepcopy(old['parity_requests'][r['historical_plan_index']])
        new=deepcopy(r);new.pop('historical_plan_index');new['index']=prior['index']
        if r['kind']=='fresh_reserved':
            fields=('observer','observer_sha256','before_snapshot','reserved_seed_decimal')
            assert all(new[k]!=prior[k] for k in fields)
            for key in fields:new[key]=prior[key]
            changed+=1
        assert new==prior
    assert changed==36
    assert [i for i,r in enumerate(rows) if r['kind']=='live_edge']==[14,29,44,59,74,89]
    paired=[r for r in rows if r['kind']!='live_edge']
    for pair_index in range(42):
        left,right=paired[2*pair_index:2*pair_index+2]
        assert [left['route'],right['route']]==(['frozen','research'] if pair_index%2==0 else ['research','frozen'])
        assert left['observer']==right['observer'] and left['before_snapshot']==right['before_snapshot']


def test_old_reserved_panel_cannot_be_silently_reused(monkeypatch):
    read=c.read;fresh=c.OUT/'reserved_panel.json'
    old=c.read(a.PRIOR/'reserved_panel.json')
    monkeypatch.setattr(c,'read',lambda path:old if Path(path)==fresh else read(path))
    with pytest.raises(ValueError,match='Fresh reserved position/seed'):q.build_plan()


def qualification_rows_without_models():
    """Prior actual saved/edge results; synthetic fresh receipts test bindings only.

    Mandatory online/final full scientific validation of fresh decisions remains
    in the parent. No new neural output is invented or executed by these tests.
    """
    plan=q.build_plan();rows=[]
    for spec in plan['parity_requests']:
        if spec['kind']=='fresh_reserved':
            result=dict(observer_sha256=spec['observer_sha256'],simulations_quota=64,
                diagnostics=dict(observer_seed=a.observer_seed(spec['observer']),phases=[]))
            if spec['route']=='research':
                result['research']=dict(world_pool=6,seed_label='A',telemetry_enabled=False,phases=[])
        else:
            prior=c.read(a.PRIOR/'qualification_rows'/f"{spec['historical_plan_index']:04d}.json")
            result=deepcopy(prior['clock']['result'])
        rows.append(dict(spec=deepcopy(spec),completed=True,clock=dict(status='ok',accepted=True,result=result)))
    current_frozen=c.read(a.PRIOR/'entrypoint_checks/frozen.json')
    return rows,plan,current_frozen


def test_all_saved_bits_edge_and_current_frozen_late_checks_remain():
    rows,plan,baseline=qualification_rows_without_models()
    result=q.validate_parity_rows(rows,plan,frozen_entrypoint=baseline)
    assert result['complete'] and result['complete_decisions']==90
    assert result['counts']==dict(frozen_saved=24,candidate_saved=24,live_edges=6,fresh_reserved=36,exact_frozen_late_shortcuts=1)
    assert result['exact_candidate_callback_bits'] and result['exact_recorded_numeric_outputs']


@pytest.mark.parametrize('change',['array','bits','late','pool','telemetry','observer','missing','order'])
def test_exact_science_or_inventory_corruption_fails_before_continuation(change):
    rows,plan,baseline=qualification_rows_without_models()
    row=next(r for r in rows if r['spec']['route']=='research' and r['spec']['kind']=='saved_reproduction')
    result=row['clock']['result']
    if change=='array':result['diagnostics']['phases'][0]['inference'][0]['output83'][0]+=.01
    elif change=='bits':result['research']['phases'][0]['callback_bits'][0]['output_float32_sha256']='0'*64
    elif change=='late':
        next(x for x in baseline['result']['fixtures'] if x['id']=='turn124')['results']['search']['diagnostics']['forced_pass']=True
    elif change=='pool':result['research']['world_pool']=3
    elif change=='telemetry':result['research']['telemetry_enabled']=True
    elif change=='observer':result['observer_sha256']='0'*64
    elif change=='missing':rows.pop()
    elif change=='order':rows[0],rows[1]=rows[1],rows[0]
    with pytest.raises(ValueError):q.validate_parity_rows(rows,plan,frozen_entrypoint=baseline)
    if change in ('array','bits','pool','telemetry','observer'):
        with pytest.raises(ValueError):
            q.validate_admission_row(row,plan['parity_requests'][row['spec']['index']],{},frozen_entrypoint=baseline)


def test_exact_source_cache_is_local_and_final_readback_is_independent(monkeypatch):
    rows,plan,baseline=qualification_rows_without_models()
    cache={};calls=[];verify=c.verify
    monkeypatch.setattr(c,'verify',lambda items:(calls.extend(items),verify(items))[1])
    for row,spec in zip(rows,plan['parity_requests']):
        q.validate_admission_row(row,spec,cache,frozen_entrypoint=baseline)
    assert len(calls)==49  # 48 distinct saved records and one shared edge suite.
    q.validate_parity_rows(rows,plan,frozen_entrypoint=baseline)
    assert len(calls)==98


def test_final_original_failure_prefix_is_preserved_without_scientific_credit():
    rows,plan,baseline=qualification_rows_without_models();rows=rows[:8]
    rows[-1]['completed']=False
    rows[-1]['clock'].update(status='timeout',accepted=False,result=None,detail='original clock')
    result=q.validate_parity_rows(rows,plan,allow_prefix=True,frozen_entrypoint=baseline)
    assert not result['complete'] and result['complete_decisions']==7 and result['attempted']==8
    assert result['failures']==[dict(index=7,status='timeout',detail='original clock')]
    rows.append(qualification_rows_without_models()[0][8])
    with pytest.raises(ValueError):q.validate_parity_rows(rows,plan,allow_prefix=True,frozen_entrypoint=baseline)


def test_actual52_forensic_checker_remains_same_and_accepts_both_immutable_receipts():
    plan=q.build_plan();freeze=c.read(a.PRIOR/'freeze.json')
    for route in ('frozen','research'):
        summary=q.verify_entrypoint(a.PRIOR/'workers'/f'qualification_{route}',
            c.read(a.PRIOR/'entrypoint_checks'/f'{route}.json'),plan,freeze)
        assert summary['complete_decisions']==52 and summary['ordinary_handlers']==37
        assert summary['negative_handler_cases']==26 and summary['native_numeric_parity']==46


def test_700ms_is_not_a_changed_scientific_or_real_clock_rule():
    rows,plan,baseline=qualification_rows_without_models()
    row=rows[0];row['clock']['receipt_elapsed_seconds']=.75
    assert q.validate_admission_row(row,plan['parity_requests'][0],{},frozen_entrypoint=baseline)['status']=='passed'
    # The parent must separately reject this completed scientific response at
    # the prospective engineering gate; it is not a one-second timeout.


def test_response_bounds_stay_exact():
    assert w.MAX_RESPONSE_BYTES==oldw.MAX_RESPONSE_BYTES==4*1024**2
    assert w.response_packet({'safe':True})==oldw.response_packet({'safe':True})
    with pytest.raises(ValueError):w.response_packet({'large':'x'*w.MAX_RESPONSE_BYTES})
    with pytest.raises(ValueError):w.response_packet({'value':float('nan')})


