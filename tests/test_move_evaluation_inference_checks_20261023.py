"""Synthetic Stage A checks; no model construction, inference or game execution."""
from copy import deepcopy
import hashlib
import json
import pytest
from tools.move_evaluation_inference_20261023 import checks as c
REAL_PANEL_IDENTITIES = c._panel_identities

def measurement(counts):
    rows = []
    for label, count in counts.items():
        rows.append({'label': label, 'calls': count, 'returns': count, 'unwinds': 0,
                     'inclusive_ns': 1000 if count else 0, 'exclusive_ns': 100 if count else 0})
    root = next(r for r in rows if r['label'] == 'scientific_body')
    root['inclusive_ns'] = 1_000_000_000
    root['exclusive_ns'] = root['inclusive_ns'] - sum(r['exclusive_ns'] for r in rows if r is not root)
    return {'schema': 'selected_python_regions_timing_v1', 'body_ns': 1_000_001_000,
            'body_seconds': 1.000001, 'root_region_ns': 1_000_000_000,
            'unassigned_body_ns': 1000, 'exclusive_sum_ns': 1_000_000_000,
            'balanced': True, 'events': 2 * sum(counts.values()), 'rows': rows,
            'boundary_tolerance_ns': 10_000_010,
            'callback_count': 2 * sum(counts.values()), 'callback_body_ns': 100,
            'manifest': [{'label': label, 'source_path': 'C:/synthetic/source.py',
                          'source_sha256': 'a' * 64, 'qualname': label, 'line': i + 1,
                          'code_sha256': 'b' * 64} for i, label in enumerate(counts)]}

def fixture():
    ids = ['face_' + str(i) for i in range(6)] + ['fixture_' + str(i) for i in range(9)]
    return {'status': 'entrypoint_counterexamples_passed',
            'fixtures': [{'id': name} for name in ids],
            'proposal_encoder_checks': [
                {'id': name, 'fit': fit, 'model_inputs': [{'tensor_digest': 'synthetic'}]}
                for name in ids[:6] for fit in c.frozen.FITS],
            'actual_mixed_trajectory': {'status': 'passed', 'independent_transitions': 3,
                'trace': [{'policy': p} for p in ('forced', 'greedy_entity', 'raw035')],
                'result': {'reason': 'completed', 'turns': 3}}}


def fine_measurement(counts):
    value = measurement(counts)
    for row in value['rows']:
        if row['label'] in c.FINE_LABELS:
            row['inclusive_ns'] = row['exclusive_ns']
    return value


def check_fixture(value=None, timing=None):
    value = fixture() if value is None else value
    timing = fine_measurement(c.FIXTURE_CALLS) if timing is None else timing
    return c.validate_fixture(value, deepcopy(value), deepcopy(value), timing)


def identities():
    return [{'id': f'games/g{i:03d}/t0001', 'fit': c.frozen.FITS[i//6],
             'observer_sha256': c.digest({'public_index': i})} for i in range(18)]


def panel(*, neural=True, forced_pass=False, noble=False):
    rows = []
    for rep in range(8):
        for i, identity in enumerate(identities()):
            action = {'action_type': 'PASS' if forced_pass else 'TAKE_TOKENS'}
            logits = [0]*67 if neural else None
            tokens, output = [0]*(31*48), [0]*83
            callback = {'tokens': tokens, 'output83': output, 'input_sha256': c.digest(tokens),
                        'output_sha256': c.digest(output)}
            phases = [] if forced_pass else [{
                'phase': 'Main', 'branch': 'neural_greedy' if neural else 'single_choice',
                'inference': [callback] if neural else [], 'canonical_logits': logits,
                'policy_logits_sha256': c.digest(logits) if neural else None}]
            if noble:
                phases.append({'phase': 'Noble', 'branch': 'auxiliary_heuristic', 'inference': [],
                               'canonical_logits': None, 'policy_logits_sha256': None})
            trace = {'schema': 'entity_greedy_decision_v1', 'profile_id': c.PROFILE,
                     'observer_sha256': identity['observer_sha256'],
                     'observer_seed': int(identity['observer_sha256'][:16],16),
                     'action_sha256': c.digest(action), 'forced_pass': forced_pass,
                     'phases': phases, 'inference_calls': int(neural), 'simulations': 0}
            rows.append({'repetition': rep, 'index': i, **identity, 'action': action, 'trace': trace})
    return {'schema': 'entity_public_inference_panel_v1', 'repetitions': 8, 'panel_size': 18, 'rows': rows}


def panel_measurement(*, neural=True):
    n = 144*int(neural)
    counts = {'scientific_body': 1, 'probe_panel': 1, 'greedy_choose': 144,
              'observer_decode': 144, 'legal': 144, 'entity_infer': n,
              'reference_greedy_choose': 0, **dict.fromkeys(c.reuse.PREPARED_LABELS,144),
              **dict.fromkeys(c.reuse.LEGACY_LABELS,0), **dict.fromkeys(c.FINE_LABELS,n),
              **dict.fromkeys(c.OPTIONAL_ZERO_LABELS,0)}
    return fine_measurement(counts)


@pytest.fixture(autouse=True)
def fake_panel_lock(monkeypatch):
    monkeypatch.setattr(c, '_panel_identities', lambda: (identities(),
        {'path': 'synthetic_lock.json', 'sha256': 'a'*64, 'bytes': 1}))


def test_fixture_retains_original_science_and_exact_fourteen_callbacks():
    result = check_fixture()
    assert result['status'] == 'passed' and result['fixture_count'] == 15
    assert result['proposal_encoder_checks'] == 18
    scopes = result['inference_scopes']
    assert scopes['inference_callbacks'] == 14
    assert scopes['helper_exclusive_sum_ns'] + scopes['parent_residual_exclusive_ns'] == scopes['inference_inclusive_ns']
    assert sum(r['fraction_of_inference_inclusive'] for r in scopes['helper_scopes']) == pytest.approx(.9)
    assert not scopes['scope']['pure_kernel_measurement']
    assert not scopes['scope']['predicted_latency_savings']


@pytest.mark.parametrize('label', ('entity_infer',)+c.FINE_LABELS)
def test_fixture_rejects_changed_inference_count(label):
    counts = dict(c.FIXTURE_CALLS); counts[label] += 1
    with pytest.raises(ValueError, match=label): check_fixture(timing=fine_measurement(counts))


@pytest.mark.parametrize('which', range(3))
def test_fixture_exact_equality(which):
    values = [fixture() for _ in range(3)]; values[which]['changed'] = True
    with pytest.raises(ValueError, match='scientific output'):
        c.validate_fixture(*values, fine_measurement(c.FIXTURE_CALLS))


@pytest.mark.parametrize('damage', ('missing', 'balanced', 'gap', 'partition', 'nested', 'parent'))
def test_fine_measurement_rejects_timing_damage(damage):
    value = fine_measurement(c.FIXTURE_CALLS)
    fine = next(r for r in value['rows'] if r['label'] == c.FINE_LABELS[0])
    if damage == 'missing': value['rows'].remove(fine)
    elif damage == 'balanced': value['balanced'] = False
    elif damage == 'gap': value['unassigned_body_ns'] += 1
    elif damage == 'partition': value['exclusive_sum_ns'] += 1
    elif damage == 'nested': fine['inclusive_ns'] += 1
    else: next(r for r in value['rows'] if r['label']=='entity_infer')['inclusive_ns'] += 1
    with pytest.raises(ValueError): check_fixture(timing=value)


@pytest.mark.parametrize('neural,forced,noble', ((True,False,False),(True,False,True),
    (False,False,False),(False,True,False),(False,True,True)))
def test_panel_accepts_exact_repeated_science(neural,forced,noble):
    value = panel(neural=neural,forced_pass=forced,noble=noble)
    result = c.validate_panel(value,deepcopy(value),deepcopy(value),panel_measurement(neural=neural))
    assert result['status']=='passed' and result['decisions']==144
    assert result['entity_inference_calls']==144*int(neural)
    assert result['candidate_lock']['path']=='synthetic_lock.json'
    assert result['repeated_inputs_scientifically_identical']
    if not neural:
        assert result['inference_scopes']['inference_inclusive_ns']==0
        assert all(row['fraction_of_inference_inclusive'] is None for row in result['inference_scopes']['helper_scopes'])


@pytest.mark.parametrize('which', range(3))
def test_panel_exact_threeway_equality(which):
    values = [panel(neural=False) for _ in range(3)]; values[which]['rows'][0]['action']['changed']=True
    with pytest.raises(ValueError, match='scientific output'):
        c.validate_panel(*values,panel_measurement(neural=False))


@pytest.mark.parametrize('damage', ('schema','extra','repetitions','panel_size','missing_row','extra_row',
    'row_extra','bool_repetition','order','id','fit','bool_fit','observer','trace_schema','profile','seed',
    'action','forced','simulations','inference_total','callback_hash','callback_shape','callback_bool',
    'logits_hash','logits_shape','missing_capture','repeat'))
def test_panel_rejects_corrupted_evidence_even_if_three_copies_match(damage):
    value=panel(); row=value['rows'][0]; trace=row['trace']; phase=trace['phases'][0]
    if damage=='schema': value['schema']='wrong'
    elif damage=='extra': value['extra']=1
    elif damage=='repetitions': value['repetitions']=9
    elif damage=='panel_size': value['panel_size']=True
    elif damage=='missing_row': value['rows'].pop()
    elif damage=='extra_row': value['rows'].append(deepcopy(row))
    elif damage=='row_extra': row['extra']=True
    elif damage=='bool_repetition': row['repetition']=False
    elif damage=='order': row['index']=1
    elif damage=='id': row['id']='other'
    elif damage=='fit': row['fit']=161003
    elif damage=='bool_fit': row['fit']=True
    elif damage=='observer': row['observer_sha256']='0'*64
    elif damage=='trace_schema': trace['schema']='other'
    elif damage=='profile': trace['profile_id']='other'
    elif damage=='seed': trace['observer_seed']+=1
    elif damage=='action': row['action']['other']=True
    elif damage=='forced': trace['forced_pass']=True
    elif damage=='simulations': trace['simulations']=1
    elif damage=='inference_total': trace['inference_calls']=2
    elif damage=='callback_hash': phase['inference'][0]['output_sha256']='0'*64
    elif damage=='callback_shape': phase['inference'][0]['tokens'].pop()
    elif damage=='callback_bool': phase['inference'][0]['tokens'][0]=True
    elif damage=='logits_hash': phase['policy_logits_sha256']='0'*64
    elif damage=='logits_shape': phase['canonical_logits'].pop()
    elif damage=='missing_capture': phase.pop('canonical_logits')
    else:
        row=value['rows'][18]; row['action']['different']=True
        row['trace']['action_sha256']=c.digest(row['action'])
    with pytest.raises(ValueError): c.validate_panel(value,deepcopy(value),deepcopy(value),panel_measurement())


@pytest.mark.parametrize('label', c.FINE_LABELS + ('entity_infer','greedy_choose','observer_decode','legal',
    'reference_greedy_choose') + c.reuse.PREPARED_LABELS + c.reuse.LEGACY_LABELS + c.OPTIONAL_ZERO_LABELS)
def test_panel_rejects_wrong_work_count(label):
    value=panel(neural=False)
    m=panel_measurement(neural=False)
    counts={r['label']:r['calls'] for r in m['rows']}; counts[label]+=1
    with pytest.raises(ValueError,match=label):
        c.validate_panel(value,deepcopy(value),deepcopy(value),fine_measurement(counts))


def test_optional_absent_targets_are_allowed_but_extra_unwinds_are_not():
    value=panel(neural=False)
    counts={r['label']:r['calls'] for r in panel_measurement(neural=False)['rows']
            if r['label'] not in c.OPTIONAL_ZERO_LABELS}
    c.validate_panel(value,deepcopy(value),deepcopy(value),fine_measurement(counts))
    counts['optional']=1; timing=fine_measurement(counts)
    row=next(r for r in timing['rows'] if r['label']=='optional')
    row['returns']=0; row['unwinds']=1
    with pytest.raises(ValueError,match='unwind'):
        c.validate_panel(value,deepcopy(value),deepcopy(value),timing)


def test_science_is_not_modified():
    value=panel(neural=False); before=deepcopy(value)
    c.validate_panel(value,value,value,panel_measurement(neural=False))
    assert value==before


@pytest.mark.parametrize('damage', (None,'hash','fit','context_fit','envelope','size'))
def test_public_panel_projection_is_bound_to_candidate_lock(tmp_path,monkeypatch,damage):
    roots=[]
    for i,identity in enumerate(identities()):
        observer={'schema':'corrected_live_planner_observer_v1','context':{'fit':identity['fit'],'private_seed_not_for_entity':'sentinel'},
                  'ruleset_id':'corrected','catalogue_sha256':'a'*64,'observation':{'index':i}}
        roots.append({'id':identity['id'],'fit':identity['fit'],'observer':observer})
    if damage=='fit': roots[0]['fit']=161003
    elif damage=='context_fit': roots[0]['observer']['context']['fit']=161003
    elif damage=='envelope': roots[0]['observer']['extra']=1
    elif damage=='size': roots.pop()
    path=tmp_path/'candidate_lock.json'; data=c.canonical({'roots':roots}).encode(); path.write_bytes(data)
    monkeypatch.setattr(c,'LOCK',path); monkeypatch.setattr(c,'ROOT',tmp_path)
    monkeypatch.setattr(c,'LOCK_SHA256','0'*64 if damage=='hash' else hashlib.sha256(data).hexdigest())
    if damage is not None:
        with pytest.raises(ValueError): REAL_PANEL_IDENTITIES()
    else:
        got,artifact=REAL_PANEL_IDENTITIES()
        expected={k:v for k,v in roots[0]['observer'].items() if k!='context'}
        expected['schema']='external_screen_observation_v1'
        assert got[0]['observer_sha256']==c.digest(expected)
        assert got[0]['observer_sha256']!=c.digest(roots[0]['observer'])
        assert artifact['sha256']==hashlib.sha256(data).hexdigest()
