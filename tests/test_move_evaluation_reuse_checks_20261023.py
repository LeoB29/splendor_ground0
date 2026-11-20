"""Model-free selected-region/count regressions; no project model imports."""
from copy import deepcopy
import gzip
import json
import math

import pytest

from tools.move_evaluation_reuse_20261023 import checks as c


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
                for name in ids[:6] for fit in c.FITS],
            'actual_mixed_trajectory': {'status': 'passed', 'independent_transitions': 3,
                'trace': [{'policy': p} for p in ('forced', 'greedy_entity', 'raw035')],
                'result': {'reason': 'completed', 'turns': 3}}}


def check_fixture(m=None, value=None):
    value = fixture() if value is None else value
    return c.validate_fixture(value, deepcopy(value), deepcopy(value),
                              measurement(c.FIXTURE_CALLS) if m is None else m)


def test_fixture_accepts_exact_science_counts_and_disjoint_shares():
    got = check_fixture()
    assert got['status'] == 'passed' and got['expected_calls'] == c.FIXTURE_CALLS
    assert got['proposal_checks_per_fit'] == {'161001': 6, '161002': 6, '161003': 6}
    assert math.isclose(sum(row['selected_root_fraction'] for row in got['cost_shares']), 1.)
    assert got['scope']['inclusive_times_overlap_do_not_sum']
    assert not got['scope']['policy_or_transition_replay_performed_here']
    assert not got['scope']['native_child_cpu_attributed']


@pytest.mark.parametrize('which', ('ordinary', 'timed', 'reference'))
def test_exact_fixture_science_compares_all_three_objects(which):
    objects = [fixture() for _ in range(3)]
    objects[('ordinary', 'timed', 'reference').index(which)]['changed'] = True
    with pytest.raises(ValueError, match='scientific output'):
        c.validate_fixture(*objects, measurement(c.FIXTURE_CALLS))


@pytest.mark.parametrize('damage', ('status', 'fixtures', 'duplicate_fixture', 'fit', 'encoder_empty',
                                   'encoder_duplicate', 'panel', 'mixed_policy', 'mixed_reason', 'mixed_turns_bool'))
def test_fixture_reference_structure_is_not_only_equality(damage):
    value = fixture()
    if damage == 'status': value['status'] = 'administrative_pass'
    elif damage == 'fixtures': value['fixtures'].pop()
    elif damage == 'duplicate_fixture': value['fixtures'][1] = value['fixtures'][0]
    elif damage == 'fit': value['proposal_encoder_checks'][0]['fit'] = 161004
    elif damage == 'encoder_empty': value['proposal_encoder_checks'][0]['model_inputs'] = []
    elif damage == 'encoder_duplicate': value['proposal_encoder_checks'][1] = value['proposal_encoder_checks'][0]
    elif damage == 'panel': value['proposal_encoder_checks'][0]['id'] = 'fixture_0'
    elif damage == 'mixed_policy': value['actual_mixed_trajectory']['trace'][1]['policy'] = 'raw035'
    elif damage == 'mixed_reason': value['actual_mixed_trajectory']['result']['reason'] = 'cutoff'
    else: value['actual_mixed_trajectory']['result']['turns'] = True
    with pytest.raises(ValueError): check_fixture(value=value)


@pytest.mark.parametrize('damage', ('schema', 'balanced', 'body_bool', 'body_seconds', 'body_nan',
    'row_bool', 'negative', 'unbalanced', 'duplicate_label', 'missing_label', 'wrong_calls',
    'root_unwind', 'root_inclusive', 'exclusive_sum', 'exclusive_partition', 'inclusive_small',
    'events', 'negative_gap', 'large_gap', 'gap_field', 'tolerance', 'callback_count', 'callback_ns',
    'manifest_missing', 'manifest_duplicate', 'manifest_hash', 'manifest_line_bool', 'uncalled_time'))
def test_timing_integrity_fail_closed(damage):
    m = measurement(c.FIXTURE_CALLS); row = m['rows'][1]
    if damage == 'schema': m['schema'] = 'old'
    elif damage == 'balanced': m['balanced'] = 1
    elif damage == 'body_bool': m['body_ns'] = True
    elif damage == 'body_seconds': m['body_seconds'] += .01
    elif damage == 'body_nan': m['body_seconds'] = float('nan')
    elif damage == 'row_bool': row['calls'] = True
    elif damage == 'negative': row['exclusive_ns'] = -1
    elif damage == 'unbalanced': row['returns'] += 1
    elif damage == 'duplicate_label': m['rows'].append(deepcopy(row))
    elif damage == 'missing_label': m['rows'].pop()
    elif damage == 'wrong_calls': row['calls'] += 1; row['returns'] += 1; m['events'] += 2
    elif damage == 'root_unwind': m['rows'][0]['returns'] = 0; m['rows'][0]['unwinds'] = 1
    elif damage == 'root_inclusive': m['rows'][0]['inclusive_ns'] += 1
    elif damage == 'exclusive_sum': m['exclusive_sum_ns'] += 1
    elif damage == 'exclusive_partition': row['exclusive_ns'] += 1
    elif damage == 'inclusive_small': row['inclusive_ns'] = 0
    elif damage == 'events': m['events'] += 1
    elif damage == 'negative_gap': m['body_ns'] = 999_999_999; m['body_seconds'] = .999999999
    elif damage == 'large_gap': m['body_ns'] += 20_000_000; m['body_seconds'] = m['body_ns']/1e9; m['unassigned_body_ns'] += 20_000_000
    elif damage == 'gap_field': m['unassigned_body_ns'] = 0
    elif damage == 'tolerance': m['boundary_tolerance_ns'] += 1
    elif damage == 'callback_count': m['callback_count'] = 0
    elif damage == 'callback_ns': m['callback_body_ns'] = m['body_ns'] + 1
    elif damage == 'manifest_missing': m['manifest'].pop()
    elif damage == 'manifest_duplicate': m['manifest'][1] = m['manifest'][0]
    elif damage == 'manifest_hash': m['manifest'][0]['source_sha256'] = 'bad'
    elif damage == 'manifest_line_bool': m['manifest'][0]['line'] = True
    else: row['calls'] = row['returns'] = 0
    with pytest.raises(ValueError): check_fixture(m)


def test_unknown_labels_must_balance_but_fixture_may_have_expected_caught_unwinds():
    counts = {**c.FIXTURE_CALLS, 'optional_decoder': 2, 'unused_region': 0}
    m = measurement(counts)
    row = next(r for r in m['rows'] if r['label'] == 'optional_decoder')
    row['returns'] = row['unwinds'] = 1
    assert check_fixture(m)['observed_calls']['optional_decoder'] == 2
    row['unwinds'] = 2
    with pytest.raises(ValueError): check_fixture(m)


def trace_rows(*, one_ply=False, neural=True):
    rows = []
    for n in range(512):
        steps = [{'policy': 'forced', 'forced': True, 'policy_trace': None,
                  'legal_count': 3, 'selected_index': 0, 'actor': 1, 'turn': 1}]
        if not one_ply:
            diag = {'inference_calls': int(neural), 'simulations': 0, 'forced_pass': False,
                    'phases': [{'phase': 'Main', 'branch': 'neural_greedy' if neural else 'late_heuristic',
                                'inference': [{'input_sha256': 'a'*64}] if neural else []},
                               {'phase': 'Noble', 'branch': 'auxiliary_heuristic', 'inference': []}]}
            steps += [{'policy': 'greedy_entity', 'forced': False, 'policy_trace': diag,
                       'legal_count': 4, 'selected_index': 1, 'actor': 0, 'turn': 2},
                      {'policy': 'raw035', 'forced': False, 'policy_trace': None,
                       'legal_count': 5, 'selected_index': 2, 'actor': 1, 'turn': 3}]
        rows.append({'world_index': n//4, 'candidate_index': n%4, 'seed': 'seed'+str(n//4),
                     'turns': len(steps), 'steps': steps})
    return rows


def selection_measurement(*, one_ply=False, neural=True):
    g = 0 if one_ply else 512
    moves = 512 if one_ply else 1536
    return measurement({'scientific_body': 1, 'choose': 1, 'trajectory': 512,
        'sample_world': 128, 'proposal': 1, 'raw_score': g+2, 'greedy_choose': g,
        'reference_greedy_choose': 0, 'entity_infer': g*int(neural),
        **{label: g for label in c.PREPARED_LABELS},
        **{label: 0 for label in c.LEGACY_LABELS},
        'observer_decode': g+1, 'step': moves, 'trace_sink': 512,
        'trace_finalize': 1, 'shared_attention_forward': g+3, 'legal': 2*moves+g+129})


def save_trace(tmp_path, rows):
    path = tmp_path/'trace.jsonl.gz'
    path.write_bytes(gzip.compress(''.join(c.canonical(row)+'\n' for row in rows).encode(), mtime=0))
    return path


@pytest.mark.parametrize('one_ply,neural', ((False, True), (False, False), (True, False)))
def test_selection_derives_known_work_from_complete_trace(tmp_path, one_ply, neural):
    rows = trace_rows(one_ply=one_ply, neural=neural)
    path = save_trace(tmp_path, rows)
    got = c.validate_selection(selection_measurement(one_ply=one_ply, neural=neural), path)
    assert got['status'] == 'passed' and got['trajectory_records'] == 512 and got['worlds'] == 128
    assert got['moves'] == (512 if one_ply else 1536)
    assert got['entity_inference_calls'] == (512 if neural and not one_ply else 0)
    assert got['trace']['bytes'] == path.stat().st_size


@pytest.mark.parametrize('noble_choice', (False, True))
def test_forced_pass_allows_only_optional_noble_auxiliary(tmp_path, noble_choice):
    rows = trace_rows(neural=False)
    for row in rows:
        row['steps'][1]['policy_trace'] = {'inference_calls': 0, 'simulations': 0, 'forced_pass': True,
            'phases': [{'phase': 'Noble', 'branch': 'auxiliary_heuristic', 'inference': []}] if noble_choice else []}
    got = c.validate_selection(selection_measurement(neural=False), save_trace(tmp_path, rows))
    assert got['entity_inference_calls'] == 0 and got['observed_calls']['greedy_choose'] == 512


@pytest.mark.parametrize('phases', (
    [{'phase': 'Return', 'branch': 'auxiliary_heuristic', 'inference': []}],
    [{'phase': 'Noble', 'branch': 'neural_greedy', 'inference': [{}]}],
    [{'phase': 'Noble', 'branch': 'auxiliary_heuristic', 'inference': []}]*2,
))
def test_forced_pass_rejects_other_auxiliary_or_neural_work(phases):
    with pytest.raises(ValueError):
        c.frozen._inferences({'inference_calls': 0, 'simulations': 0, 'forced_pass': True, 'phases': phases})


@pytest.mark.parametrize('damage', ('short', 'extra', 'order', 'bool_index', 'seed', 'duplicate_seed',
    'int_seed', 'turn_count', 'policy', 'forced', 'legal_count', 'selected_index', 'actor', 'turn',
    'diagnostic', 'inference_total', 'inference_list', 'simulations', 'forced_pass', 'phase', 'nonentity_diag',
    'missing_newline', 'truncated_gzip', 'duplicate_json_key', 'nan', 'unwind', 'wrong_legal_calls', 'changed_file'))
def test_selection_rejects_corrupt_trace_or_count(tmp_path, monkeypatch, damage):
    rows = trace_rows(); m = selection_measurement(); row = rows[0]; step = row['steps'][1]
    if damage == 'short': rows.pop()
    elif damage == 'extra': rows.append(deepcopy(rows[-1]))
    elif damage == 'order': rows[1]['candidate_index'] = 0
    elif damage == 'bool_index': row['world_index'] = False
    elif damage == 'seed': rows[1]['seed'] = 'different'
    elif damage == 'duplicate_seed':
        for r in rows[4:8]: r['seed'] = 'seed0'
    elif damage == 'int_seed': row['seed'] = 0
    elif damage == 'turn_count': row['turns'] = 2
    elif damage == 'policy': step['policy'] = 'raw035'
    elif damage == 'forced': step['forced'] = True
    elif damage == 'legal_count': step['legal_count'] = False
    elif damage == 'selected_index': step['selected_index'] = 4
    elif damage == 'actor': step['actor'] = 1
    elif damage == 'turn': step['turn'] = 3
    elif damage == 'diagnostic': step['policy_trace'] = None
    elif damage == 'inference_total': step['policy_trace']['inference_calls'] = 2
    elif damage == 'inference_list': step['policy_trace']['phases'][0]['inference'] = []
    elif damage == 'simulations': step['policy_trace']['simulations'] = 1
    elif damage == 'forced_pass': step['policy_trace']['forced_pass'] = True
    elif damage == 'phase': step['policy_trace']['phases'][0]['phase'] = 'Noble'
    elif damage == 'nonentity_diag': row['steps'][0]['policy_trace'] = {}
    elif damage == 'unwind': m['rows'][1]['returns'] = 0; m['rows'][1]['unwinds'] = 1
    elif damage == 'wrong_legal_calls':
        r = next(r for r in m['rows'] if r['label'] == 'legal')
        r['calls'] -= 1; r['returns'] -= 1; m['events'] -= 2
    path = save_trace(tmp_path, rows)
    if damage in ('missing_newline', 'duplicate_json_key', 'nan'):
        raw = gzip.decompress(path.read_bytes())
        if damage == 'missing_newline': raw = raw[:-1]
        elif damage == 'duplicate_json_key': raw = raw.replace(b'"world_index":0', b'"world_index":0,"world_index":0', 1)
        else: raw = raw.replace(b'"turns":3', b'"turns":NaN', 1)
        path.write_bytes(gzip.compress(raw, mtime=0))
    elif damage == 'truncated_gzip': path.write_bytes(path.read_bytes()[:-8])
    elif damage == 'changed_file':
        original = c.frozen._trace_identity; calls = []
        def changed(p):
            item = original(p); calls.append(1)
            if len(calls) > 1: item['sha256'] = '0'*64
            return item
        monkeypatch.setattr(c.frozen, '_trace_identity', changed)
    with pytest.raises((ValueError, EOFError, gzip.BadGzipFile)):
        c.validate_selection(m, path)


def test_real_timer_schema_with_model_free_functions():
    from tools.move_evaluation_timing_20261023.monitor import SelectedTimer
    def outer():
        return helper()
    def helper():
        return 7
    value, m = SelectedTimer({'scientific_body': outer, 'helper': helper}).run(outer)
    assert value == 7
    result = c.frozen._measurement(m, {'scientific_body': 1, 'helper': 1}, no_unwinds=True)
    assert result['status'] == 'passed'


def test_fixture_allows_original_bridge_contract_probes_but_not_original_choice():
    counts = {**c.FIXTURE_CALLS, **dict.fromkeys(c.LEGACY_LABELS, 3)}
    got = check_fixture(measurement(counts))
    assert got['legacy_helper_fixture_counts_unconstrained'] is True
    assert got['expected_calls']['reference_greedy_choose'] == 0
    assert got['implementation_manifest']['administrative_only'] is True
    counts['reference_greedy_choose'] = 1
    with pytest.raises(ValueError, match='reference_greedy_choose'):
        check_fixture(measurement(counts))


@pytest.mark.parametrize('label', c.PREPARED_LABELS + ('reference_greedy_choose',))
def test_fixture_requires_exact_new_preparation_census(label):
    counts = dict(c.FIXTURE_CALLS)
    counts[label] += 1
    with pytest.raises(ValueError, match=label):
        check_fixture(measurement(counts))


@pytest.mark.parametrize('label', c.PREPARED_LABELS + c.LEGACY_LABELS + (
    'reference_greedy_choose', 'observer_decode', 'legal', 'step',
    'shared_attention_forward', 'entity_infer', 'raw_score', 'trajectory',
    'sample_world', 'trace_sink', 'trace_finalize'))
def test_selection_rejects_preparation_loss_duplicate_or_legacy_work(tmp_path, label):
    counts = {r['label']: r['calls'] for r in selection_measurement()['rows']}
    counts[label] += 1
    with pytest.raises(ValueError, match=label):
        c.validate_selection(measurement(counts), save_trace(tmp_path, trace_rows()))


@pytest.mark.parametrize('label', c.PREPARED_LABELS + c.LEGACY_LABELS + ('observer_decode', 'step'))
def test_selection_rejects_missing_selected_instrumentation(tmp_path, label):
    counts = {r['label']: r['calls'] for r in selection_measurement()['rows']}
    del counts[label]
    with pytest.raises(ValueError, match='Missing required'):
        c.validate_selection(measurement(counts), save_trace(tmp_path, trace_rows()))


def test_selection_reconciles_approved_count_reductions_without_claiming_savings(tmp_path):
    got = c.validate_selection(selection_measurement(), save_trace(tmp_path, trace_rows()))
    reduction = got['preparation_call_reduction']
    assert reduction['observer_decode'] == {'reference_expected': 3073, 'reuse_expected': 513, 'fewer_calls': 2560}
    assert reduction['legal'] == {'reference_expected': 6273, 'reuse_expected': 3713, 'fewer_calls': 2560}
    assert reduction['legacy_bridge_prepare_expected'] == 2560
    assert reduction['new_prepare_once_expected'] == 512
    assert got['scope']['original_policy_replay_required_separately']
    assert got['scope']['count_reduction_is_not_a_latency_saving_estimate']
    assert not reduction['cross_decision_cache_tested_here']
    manifest = got['implementation_manifest']
    assert {t['label'] for t in manifest['targets']} == set(c.IMPLEMENTATION_LABELS)
    assert manifest['sha256'] == c.digest(manifest['targets'])
    assert manifest['scientific_policy_label_changed'] is False


def test_scientific_objects_remain_unmodified():
    objects = [fixture() for _ in range(3)]
    before = deepcopy(objects)
    c.validate_fixture(*objects, measurement(c.FIXTURE_CALLS))
    assert objects == before
