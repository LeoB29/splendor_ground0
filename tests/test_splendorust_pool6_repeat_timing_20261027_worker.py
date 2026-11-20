"""Model-free facade, startup guards and exact saved repeat regressions."""
import ast
from copy import deepcopy
import json
from pathlib import Path
from types import SimpleNamespace

import pytest

from tools.splendorust_pool6_repeat_timing_20261027 import adapters as a, worker as w, qualification as q, common as c, design
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
                "'SPLENDOR_POOL6_REPEAT_PHASE', 'repeat_timing'")
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
    path.write_text(json.dumps({'phase': 'qualification' if bad=='freeze_phase' else 'repeat_timing'}))
    monkeypatch.setattr(c,'OUT',out)
    monkeypatch.setenv('STUDY_WORKER_DIR',str(tmp_path/'other' if bad=='output_boundary' else out/'worker'))
    monkeypatch.setenv('STUDY_BOUNDARY_PROFILE','1')
    monkeypatch.setenv('STUDY_RESEARCH_FREEZE',str(path))
    monkeypatch.setenv('STUDY_RESEARCH_FREEZE_SHA256','0'*64 if bad=='freeze_hash' else c.file_hash(path))
    monkeypatch.setenv('SPLENDOR_POOL6_REPEAT_PHASE','qualification' if bad=='environment' else 'repeat_timing')
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


@pytest.fixture
def prepared(monkeypatch,tmp_path):
    panel=design.build_panel();matrix=design.build_matrix(panel)
    monkeypatch.setattr(c,'OUT',tmp_path)
    (tmp_path/'saved_panel.json').write_text(c.canonical(panel))
    (tmp_path/'request_matrix.json').write_text(c.canonical(matrix))
    # Artifacts in this fixture are temporary metadata only; immutable targets
    # are still authenticated by their real hashes in historical_result.
    real_artifact=c.artifact
    monkeypatch.setattr(c,'artifact',lambda p:dict(path='test/'+Path(p).name,sha256=c.file_hash(p),bytes=Path(p).stat().st_size)
        if Path(p).is_relative_to(tmp_path) else real_artifact(p))
    return q.build_plan()


def rows_from_saved_results(plan):
    """Synthetic clocks around authentic saved outputs, not new execution."""
    baselines={};rows=[]
    for spec in plan['repeat_requests']:
        key=(spec['position_index'],spec['route'])
        if key not in baselines:baselines[key]=q.historical_result(spec)['result']
        rows.append(dict(spec=deepcopy(spec),completed=True,
            clock=dict(status='ok',accepted=True,result=deepcopy(baselines[key]))))
    return rows


def test_complete_inventory_and_all144_exact_historical_comparisons(prepared):
    assert prepared['complete_decisions']==104 and prepared['total_scientific_decisions']==248
    assert prepared['graph_cases']==22 and prepared['maximum_parent_requests']==148
    result=q.validate_repeat_rows(rows_from_saved_results(prepared),prepared)
    assert result['complete'] and result['exact_historical_comparisons']==144
    assert result['exact_first_repeat_comparisons']==132
    assert len(result['cells'])==12 and all(x['successful']==12 for x in result['cells'])
    assert result['admission_decision'] is False and result['margin_700ms']=='descriptive_only'


def test_frozen_gold_removes_only_startup_arrays_not_any_scientific_identity(prepared):
    spec=next(x for x in prepared['repeat_requests'] if x['case_id']=='gold_returns' and x['route']=='frozen')
    target=q.historical_result(spec)
    saved=c.read(c.ROOT/spec['expected_entrypoint']['path'])
    raw=next(x for x in saved['result']['fixtures'] if x['id']=='gold_returns')['results']['search']
    projected=deepcopy(raw)
    assert any('inference' in p for p in raw['diagnostics']['phases'])
    for phase in projected['diagnostics']['phases']:phase.pop('inference',None)
    assert target['result']==projected
    assert target['projection']=='remove_phase_inference_from_frozen_startup_target_only'
    assert 'inference' in raw['diagnostics']['phases'][0]


@pytest.mark.parametrize('change',['action','callback','bits','profile','missing','order','late_failure_continuation'])
def test_repeat_validation_rejects_changed_outputs_and_inventory(prepared,change):
    rows=rows_from_saved_results(prepared)
    if change=='missing':rows.pop()
    elif change=='order':rows[0],rows[1]=rows[1],rows[0]
    elif change=='late_failure_continuation':
        rows[1]['clock'].update(status='timeout',accepted=False,result=None);rows[1]['completed']=False
    else:
        row=next(x for x in rows if x['spec']['route']=='research' and x['spec']['round']==1)
        result=row['clock']['result']
        if change=='action':result['action']['market_index']=999
        if change=='callback':result['diagnostics']['phases'][0]['neural_callbacks'][0]='0'*64
        if change=='bits':result['research']['phases'][0]['callback_bits'][0]['output_float32_sha256']='0'*64
        if change=='profile':result['profile_sha256']='0'*64
    with pytest.raises(ValueError):q.validate_repeat_rows(rows,prepared,allow_prefix=change=='late_failure_continuation')


def test_failed_final_prefix_is_retained_not_counted_as_science(prepared):
    rows=rows_from_saved_results(prepared)[:8]
    rows[-1]['clock'].update(status='timeout',accepted=False,result=None,detail='original deadline')
    rows[-1]['completed']=False
    result=q.validate_repeat_rows(rows,prepared,allow_prefix=True)
    assert result['status']=='verified_prefix' and not result['complete']
    assert result['attempted']==8 and result['successful']==result['exact_historical_comparisons']==7
    assert result['failures']==[dict(index=7,route=rows[-1]['spec']['route'],case_id=rows[-1]['spec']['case_id'],status='timeout',detail='original deadline')]


def test_full_original_saved_entrypoint_validator_is_reused(prepared):
    frozen=c.read(a.PRIOR/'freeze.json')
    for role in ('frozen','research'):
        receipt=c.read(a.PRIOR/'entrypoint_checks'/f'{role}.json')
        result=q.verify_entrypoint(a.PRIOR/'workers'/f'qualification_{role}',receipt,prepared,frozen)
        assert result['complete_decisions']==52 and result['negative_handler_cases']==26


def test_response_bounds_stay_exact():
    assert w.MAX_RESPONSE_BYTES==oldw.MAX_RESPONSE_BYTES==4*1024**2
    assert w.response_packet({'safe':True})==oldw.response_packet({'safe':True})
    with pytest.raises(ValueError):w.response_packet({'large':'x'*w.MAX_RESPONSE_BYTES})
    with pytest.raises(ValueError):w.response_packet({'value':float('nan')})
