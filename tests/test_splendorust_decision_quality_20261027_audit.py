"""Model-free corrected-loader, evidence-integrity and descriptive-analysis tests."""
from copy import deepcopy
import hashlib
import json
from pathlib import Path

import pytest

from tools.splendorust_decision_quality_20261027 import audit as a, analysis


def read(path):
    return json.loads(Path(path).read_text(encoding='utf-8'))


def artifact(path, root):
    data = Path(path).read_bytes()
    return dict(path=Path(path).relative_to(root).as_posix(), bytes=len(data),
                sha256=hashlib.sha256(data).hexdigest())


@pytest.fixture(scope='module')
def source():
    panel = read(a.ROOT/'outputs/decision_quality_preparation_20261027_01/panel.json')
    descriptor = panel['roots'][0]
    row = read(a.ROOT/descriptor['source']['path'])
    clock = artifact(a.ROOT/descriptor['source_worker_clock_file'], a.ROOT)
    freeze = dict(inputs=[descriptor['source'], clock])
    return descriptor, row, freeze


def rebound(tmp_path, source, change):
    descriptor, row, _ = deepcopy(source)
    change(descriptor, row)
    path = tmp_path/descriptor['source']['path']; path.parent.mkdir(parents=True)
    path.write_text(json.dumps(row), encoding='utf-8')
    descriptor['source'] = artifact(path, tmp_path)
    clock_path = tmp_path/descriptor['source_worker_clock_file']; clock_path.parent.mkdir(parents=True)
    clock_path.write_text(json.dumps(row['clock']), encoding='utf-8')
    return descriptor, dict(inputs=[descriptor['source'], artifact(clock_path,tmp_path)])


def test_actual_corrected_loader_preserves_historical_science_exposes_observer_only(source):
    descriptor, row, freeze = source
    loaded = a.load_saved_root(descriptor, freeze)
    assert loaded['observer'] == row['observer']
    assert loaded['original_result'] == row['clock']['result']
    assert loaded['independent_saved_transition_checked']
    assert loaded['model_input_private_state'] is False
    assert 'before' not in loaded and 'after' not in loaded


@pytest.mark.parametrize('change', [
    lambda d,r: r['before'].pop('ruleset_id'),
    lambda d,r: r['before'].__setitem__('catalogue_sha256', '0'*64),
    lambda d,r: r['observer'].__setitem__('private_deck', ['forbidden']),
    lambda d,r: d.__setitem__('before_sha256', '0'*64),
    lambda d,r: r.__setitem__('seat', 1-r['seat']),
    lambda d,r: r['after']['state'].__setitem__('turn_index', 999),
    lambda d,r: r['clock']['result'].__setitem__('profile_sha256', '0'*64),
    lambda d,r: r.__setitem__('completed', False),
])
def test_actual_corrected_loader_rejects_semantics_even_after_file_repin(tmp_path, source, change):
    descriptor, freeze = rebound(tmp_path, source, change)
    with pytest.raises((ValueError,KeyError)):
        a.load_saved_root(descriptor, freeze, root=tmp_path)


def test_loader_rejects_raw_clock_changed_independently(tmp_path, source):
    descriptor, freeze = rebound(tmp_path, source, lambda d,r: None)
    path = tmp_path/descriptor['source_worker_clock_file']; value = read(path)
    value['elapsed_seconds'] += .01; path.write_text(json.dumps(value), encoding='utf-8')
    freeze['inputs'][1] = artifact(path,tmp_path)
    with pytest.raises(ValueError, match='clock-file/row'):
        a.load_saved_root(descriptor,freeze,root=tmp_path)


def synthetic_result(source, quota=64, seed='A'):
    descriptor,row,_ = source
    result = deepcopy(row['clock']['result']); observer=deepcopy(row['observer'])
    pid = f'research_puct{quota}_saved_diagnostic_20261027_v1'
    identity=dict(profile_id=pid,profile_sha256='1'*64,wrapper_id=pid+'_30s',native_binary_sha256='2'*64)
    result.update({k:v for k,v in identity.items() if k != 'native_binary_sha256'},simulations_quota=quota)
    trace=result['diagnostics'];trace.update(profile_id=pid,observer_seed=a.seed_for(observer,seed))
    extras=[]
    for index, phase in enumerate(trace['phases']):
        count=phase['neural_callback_count']; phase['inference']=[]; bits=[]
        if not phase['heuristic_phase']: phase['simulations']=quota
        for j in range(count):
            tokens=[0.]*1488; outputs=[[0.]*83]
            item=dict(input_sha256=a.digest(tokens), output_sha256=a.digest(outputs),batch=1)
            if j==0: item.update(tokens=tokens,outputs=outputs,output83=outputs[0])
            phase['inference'].append(item)
            bits.append(dict(input_float32_sha256=a.float32_sha(tokens), output_float32_sha256=a.float32_sha(outputs[0])))
        phase['neural_callbacks']=[a.digest([x['input_sha256'],x['output_sha256'],1]) for x in phase['inference']]
        telemetry=None
        if not phase['heuristic_phase']:
            selected=a.original.canonical_index(phase['action']); visits=[0]*67;qs=[None]*67
            visits[selected]=quota;qs[selected]=.5
            telemetry=dict(visits=visits,q=qs,root_value=.5,network_value=.5,selected_edge_value=.5)
        extras.append(dict(phase_index=index,phase_kind='Main',telemetry=telemetry,callback_bits=bits))
    result['research']=dict(schema=a.RESEARCH_SCHEMA,quota=quota,seed_label=seed,seed=a.seed_for(observer,seed),
        telemetry_enabled=True,internal_seconds=29.,native_binary_sha256='2'*64,phases=extras)
    return result,observer,identity


def check(result,observer,identity):
    return a.validate_research_result(result,observer,quota=result['simulations_quota'],
        seed_label=result['research']['seed_label'],**identity)


@pytest.mark.parametrize('quota', [64,256,1024])
def test_synthetic_finite_science_all_ordinary_quotas(source,quota):
    result,observer,identity=synthetic_result(source,quota)
    assert check(result,observer,identity)['simulations']==quota


@pytest.mark.parametrize('case', ['missing_q','invented_q','quota_visits','selected_q','value_sign','root_mean',
                                  'callback_bits','callback_canonical','illegal_visits','private_observer','seed','quota16'])
def test_research_integrity_counterexamples(source,case):
    result,observer,identity=synthetic_result(source)
    telemetry=result['research']['phases'][0]['telemetry'];selected=a.original.canonical_index(result['diagnostics']['phases'][0]['action'])
    empty=next(i for i,n in enumerate(telemetry['visits']) if n==0)
    if case=='missing_q': telemetry['q'][selected]=None
    elif case=='invented_q': telemetry['q'][empty]=0.
    elif case=='quota_visits': telemetry['visits'][selected]-=1
    elif case=='selected_q': telemetry['selected_edge_value']=.25
    elif case=='value_sign': telemetry['network_value']=.75
    elif case=='root_mean': telemetry['root_value']=.25
    elif case=='callback_bits': result['research']['phases'][0]['callback_bits'][0]['output_float32_sha256']='0'*64
    elif case=='callback_canonical': result['diagnostics']['phases'][0]['neural_callbacks'][0]='0'*64
    elif case=='illegal_visits': telemetry['visits'][0]=1;telemetry['q'][0]=.5;telemetry['visits'][selected]-=1
    elif case=='private_observer': observer['secret_deck']=[]
    elif case=='seed': result['research']['seed']+=1
    elif case=='quota16': result['simulations_quota']=16
    with pytest.raises((ValueError,KeyError)):
        check(result,observer,identity)


def test_q_comparison_requires_both_actions_in_both_references():
    old=dict(kind='ReserveDeck',tier=0);alt=dict(kind='ReserveDeck',tier=1)
    visits=[0]*67; qs=[None]*67;visits[49]=2;visits[50]=2;qs[49]=.4;qs[50]=.6
    b=dict(root_visits=visits,root_q=qs)
    c=deepcopy(b);c['root_visits'][49]=0;c['root_q'][49]=None
    compared=analysis.fixed_pair_comparison(old,alt,{'B':b,'C':c})
    assert compared['eligible'] is False and compared['consensus_preference'] is None
    assert compared['references']['C']['original_q'] is None
    assert compared['references']['C']['alternative_minus_original_q'] is None
    c=deepcopy(b)
    assert analysis.fixed_pair_comparison(old,alt,{'B':b,'C':c})['consensus_preference']=='alternative'
    assert analysis.fixed_pair_comparison(old,old,{'B':b,'C':c})['eligible'] is False


def test_partial_analysis_retains_missing_conditions_and_no_strength_claim(source):
    descriptor,_,_=source
    result=analysis.analyze([],{'roots':[descriptor]}, {'requests':[]},complete=False)
    assert result['all_five_conditions_roots']==0
    assert len(result['roots'][0]['missing_conditions'])==5
    assert result['roots'][0]['fixed_pair'] is None
    assert result['timing']=={}
    assert 'incomplete' in result['next_study_recommendation']


def test_exact_reproduction_excludes_only_declared_capture_identity(source):
    result,_,_=synthetic_result(source)
    original=deepcopy(result)
    result['profile_id']='other';result['research']['seed_label']='C'
    assert a.scientific_core(result)==a.scientific_core(original)
    result['diagnostics']['phases'][0]['neural_callbacks'][0]='0'*64
    assert a.scientific_core(result)!=a.scientific_core(original)


def prefix_fixture(monkeypatch):
    specs=[dict(sequence=i,root_id='r',stage='research_main',quota=64,seed_family='A') for i in range(156)]
    rows=[dict(spec=deepcopy(specs[i]),completed=True,worker_clock_file=f'worker/request-{i}.json',
               clock=dict(request_id=f'request-{i}',status='ok',result=dict(value=1))) for i in range(2)]
    monkeypatch.setattr(a,'audit_record',lambda row,spec,freeze,**kwargs: dict(status='passed',failure_kind='operational'))
    return specs,rows,dict(roots=[dict(root_id='r')]),dict(r={})


def test_exact_prefix_rejects_duplicate_request_file_even_with_different_id(monkeypatch):
    specs,rows,panel,loaded=prefix_fixture(monkeypatch)
    rows[1]['worker_clock_file']=rows[0]['worker_clock_file']
    with pytest.raises(ValueError,match='duplicated'):
        a.audit_prefix(rows,{},dict(requests=specs),panel,saved_roots=loaded,allow_partial=True)


def test_exact_prefix_stops_after_first_failure(monkeypatch):
    specs,rows,panel,loaded=prefix_fixture(monkeypatch)
    rows[0]['completed']=False;rows[0]['clock']['status']='timeout'
    with pytest.raises(ValueError,match='after first failure'):
        a.audit_prefix(rows,{},dict(requests=specs),panel,saved_roots=loaded,allow_partial=True)
    result=a.audit_prefix(rows[:1],{},dict(requests=specs),panel,saved_roots=loaded,allow_partial=True)
    assert not result['complete'] and result['completed_requests']==0
    assert result['preserved_failure']['no_retry']


def test_exact_research_reproduction_compares_full_telemetry(monkeypatch):
    specs,rows,panel,loaded=prefix_fixture(monkeypatch)
    specs[1]['stage']='research_reproduction';rows[1]['spec']=deepcopy(specs[1])
    rows[1]['clock']['result']['value']=2
    with pytest.raises(ValueError,match='full research reproduction'):
        a.audit_prefix(rows,{},dict(requests=specs),panel,saved_roots=loaded,allow_partial=True)


def test_analysis_pair_selection_does_not_look_at_b_or_c(source):
    descriptor,historical,_=source
    records=[dict(spec=dict(root_id=descriptor['root_id'],stage='frozen_reference_reproduction'),
                  completed=True,clock=dict(result=historical['clock']['result']))]
    for quota,seed in analysis.CONDITIONS:
        result,_,_=synthetic_result(source,quota,seed)
        records.append(dict(spec=dict(root_id=descriptor['root_id'],stage='research_main',quota=quota,seed_family=seed),
                            completed=True,clock=dict(result=result,receipt_elapsed_seconds=.1,elapsed_seconds=.11)))
    before=analysis.analyze(records,dict(roots=[descriptor]),{},complete=True)
    records[-1]['clock']['result']['research']['phases'][0]['telemetry']['q']=[None]*67
    after=analysis.analyze(records,dict(roots=[descriptor]),{},complete=True)
    for key in ('original_index','alternative_index','same_action','alternative_selected_from'):
        assert before['roots'][0]['fixed_pair'][key]==after['roots'][0]['fixed_pair'][key]
    assert before['roots'][0]['conditions']['64-A']['raw_logits_are_normalized_priors'] is False
    assert 'No automatic numerical trigger' in before['next_study_recommendation']


def test_failed_request_cannot_receive_shorter_clock(source):
    receipt=deepcopy(source[1]['clock'])
    receipt.update(status='timeout',accepted=False,result=None,received_before_deadline=False,
                   receipt_elapsed_seconds=None,deadline_seconds=.5,elapsed_seconds=.51)
    with pytest.raises(ValueError,match='unshortened'):
        a.check_clock(receipt,limit=1.)
