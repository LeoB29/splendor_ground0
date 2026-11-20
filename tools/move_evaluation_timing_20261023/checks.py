"""Independent, model-free count and timing checks for selected Python regions.

This does not replay a game or infer a policy. Scientific equivalence and a
trace-derived invocation census are checked separately from wall attribution.
"""
from collections import Counter
import gzip
import hashlib
import json
import math
from pathlib import Path


FITS = (161001, 161002, 161003)
WORLDS = 128
SUPPORT = 4
RECORDS = WORLDS * SUPPORT
MAX_LINE = 8 * 1024**2
FIXTURE_CALLS = {
    'scientific_body': 1, 'entrypoint_checks': 1,
    'actual_trajectory_check': 1, 'trajectory': 1, 'raw_select': 15,
    'raw_score': 34, 'proposal': 18, 'sample_world': 16, 'greedy_choose': 17,
    'shared_attention_forward': 52,
}


def require(value, message):
    if not value:
        raise ValueError('Independent selected-timing check: ' + message)


def canonical(value):
    return json.dumps(value, sort_keys=True, separators=(',', ':'), allow_nan=False)


def digest(value):
    return hashlib.sha256(canonical(value).encode()).hexdigest()


def _integer(value, minimum=0):
    return type(value) is int and value >= minimum


def _hash(value):
    return (type(value) is str and len(value) == 64 and
            all(c in '0123456789abcdef' for c in value))


def _measurement(value, expected, *, no_unwinds):
    require(type(value) is dict and value.get('schema') == 'selected_python_regions_timing_v1',
            'Timing schema')
    require(value.get('balanced') is True, 'Timer did not certify balanced events')
    for key in ('body_ns', 'root_region_ns', 'exclusive_sum_ns'):
        require(_integer(value.get(key), 1), 'Invalid positive timing field: ' + key)
    require(_integer(value.get('unassigned_body_ns')) and _integer(value.get('events'), 2),
            'Invalid gap or event count')
    seconds = value.get('body_seconds')
    require(type(seconds) in (int, float) and math.isfinite(seconds) and
            seconds == value['body_ns'] / 1_000_000_000, 'Body seconds/nanoseconds disagree')
    rows = value.get('rows')
    require(type(rows) is list and rows, 'Missing timing rows')
    by_label = {}
    for row in rows:
        require(type(row) is dict and set(row) == {
            'label', 'calls', 'returns', 'unwinds', 'inclusive_ns', 'exclusive_ns'}, 'Timing row schema')
        label = row['label']
        require(type(label) is str and label and label not in by_label, 'Duplicate/invalid timing label')
        require(all(_integer(row[key]) for key in ('calls', 'returns', 'unwinds', 'inclusive_ns', 'exclusive_ns')),
                'Timing row numbers must be nonnegative integers')
        require(row['calls'] == row['returns'] + row['unwinds'], 'Unbalanced row: ' + label)
        require(row['inclusive_ns'] >= row['exclusive_ns'], 'Exclusive time exceeds inclusive: ' + label)
        require(row['calls'] > 0 or row['inclusive_ns'] == row['exclusive_ns'] == 0,
                'Time assigned to an uncalled function: ' + label)
        require(not no_unwinds or row['unwinds'] == 0, 'Unexpected selection unwind: ' + label)
        by_label[label] = row
    require(set(expected) <= set(by_label), 'Missing required selected functions')
    for label, count in expected.items():
        require(by_label[label]['calls'] == count, 'Scientific call count differs: ' + label)
    root = by_label['scientific_body']
    require(root['calls'] == root['returns'] == 1 and root['unwinds'] == 0,
            'Exactly one successful scientific body required')
    require(root['inclusive_ns'] == value['root_region_ns'], 'Root inclusive duration differs')
    total = sum(row['exclusive_ns'] for row in rows)
    require(total == value['exclusive_sum_ns'] == value['root_region_ns'],
            'Exclusive regions do not partition root duration')
    require(value['events'] == sum(row['calls'] + row['returns'] + row['unwinds'] for row in rows),
            'Event census differs from rows')
    gap = value['body_ns'] - value['root_region_ns']
    tolerance = max(5_000_000, value['body_ns'] // 100)
    require(0 <= gap == value['unassigned_body_ns'] <= tolerance, 'Root/body gap exceeds max(5ms,1%)')
    if 'boundary_tolerance_ns' in value:
        require(type(value['boundary_tolerance_ns']) is int and value['boundary_tolerance_ns'] == tolerance,
                'Declared boundary tolerance differs')
    if 'callback_count' in value:
        require(_integer(value['callback_count']) and value['callback_count'] >= value['events'],
                'Callback count below selected event count')
    if 'callback_body_ns' in value:
        require(_integer(value['callback_body_ns']) and value['callback_body_ns'] <= value['body_ns'],
                'Invalid callback diagnostic duration')
    manifest = value.get('manifest')
    require(type(manifest) is list and len(manifest) == len(rows), 'Missing target manifest entries')
    targets = {}
    for target in manifest:
        require(type(target) is dict and set(target) == {
            'label', 'source_path', 'source_sha256', 'qualname', 'line', 'code_sha256'}, 'Target manifest schema')
        label = target['label']
        require(type(label) is str and label in by_label and label not in targets, 'Manifest label mismatch')
        require(type(target['source_path']) is str and target['source_path'] and
                type(target['qualname']) is str and target['qualname'] and _integer(target['line'], 1) and
                _hash(target['source_sha256']) and _hash(target['code_sha256']), 'Malformed target identity')
        targets[label] = target
    require(set(targets) == set(by_label), 'Target manifest and measured labels differ')
    return {
        'schema': 'move_evaluation_selected_timing_check_v1', 'status': 'passed',
        'expected_calls': dict(expected), 'observed_calls': {k: by_label[k]['calls'] for k in sorted(by_label)},
        'timing_measurement_sha256': digest(value), 'target_manifest_sha256': digest(manifest),
        'body_seconds': seconds, 'root_region_ns': value['root_region_ns'],
        'unassigned_body_ns': gap, 'boundary_tolerance_ns': tolerance,
        'exclusive_sum_ns': total, 'events': value['events'],
        'cost_shares': [{
            'label': row['label'], 'calls': row['calls'], 'exclusive_ns': row['exclusive_ns'],
            'selected_root_fraction': row['exclusive_ns'] / total,
            'inclusive_ns_nonadditive': row['inclusive_ns'],
        } for row in sorted(rows, key=lambda r: (-r['exclusive_ns'], r['label']))],
        'scope': {
            'inclusive_times_overlap_do_not_sum': True,
            'exclusive_scope': 'Nearest selected region; includes unselected descendants, native work, waiting and instrumentation overhead.',
            'share_denominator': 'Measured selected root region, not uninstrumented response time.',
            'native_child_cpu_attributed': False, 'instrumentation_overhead_subtracted': False,
            'target_source_hashes_reverified_here': False,
            'policy_or_transition_replay_performed_here': False,
        },
    }


def validate_fixture(ordinary_scientific, timed_scientific, original_reference, measurement):
    """Require exact accepted fixture science and independently expected counts."""
    require(canonical(ordinary_scientific) == canonical(timed_scientific) == canonical(original_reference),
            'Fixture scientific output differs from accepted reference')
    value = original_reference
    require(type(value) is dict and value.get('status') == 'entrypoint_counterexamples_passed',
            'Actual scientific fixture reference required')
    fixtures = value.get('fixtures')
    require(type(fixtures) is list and len(fixtures) == 15 and
            all(type(f) is dict and type(f.get('id')) is str for f in fixtures) and
            len({f['id'] for f in fixtures}) == 15, 'Accepted fifteen fixture identities')
    proposals = value.get('proposal_encoder_checks')
    require(type(proposals) is list and len(proposals) == 18, 'Expected eighteen proposal encoder checks')
    seen = set()
    counts = Counter()
    for row in proposals:
        require(type(row) is dict and type(row.get('fit')) is int and row['fit'] in FITS and
                type(row.get('id')) is str and row['id'] in {f['id'] for f in fixtures} and
                type(row.get('model_inputs')) is list and bool(row['model_inputs']), 'Actual proposal encoder evidence')
        identity = (row['id'], row['fit'])
        require(identity not in seen, 'Duplicate proposal fixture/fit')
        seen.add(identity); counts[row['fit']] += 1
    require(counts == Counter({fit: 6 for fit in FITS}), 'All three fits require six encoder checks')
    require(len({identity[0] for identity in seen}) == 6, 'Different proposal fixture panels across fits')
    mixed = value.get('actual_mixed_trajectory')
    require(type(mixed) is dict and mixed.get('status') == 'passed' and
            type(mixed.get('independent_transitions')) is int and mixed['independent_transitions'] == 3,
            'Actual mixed trajectory proof')
    steps = mixed.get('trace')
    require(type(steps) is list and len(steps) == 3 and
            [step.get('policy') for step in steps] == ['forced', 'greedy_entity', 'raw035'],
            'Known three-step mixed trajectory differs')
    require(type(mixed.get('result')) is dict and mixed['result'].get('reason') == 'completed' and
            type(mixed['result'].get('turns')) is int and mixed['result']['turns'] == 3,
            'Mixed fixture did not complete naturally')
    result = _measurement(measurement, FIXTURE_CALLS, no_unwinds=False)
    result.update(mode='actual_model_fixture', scientific_sha256=digest(value), fixture_count=15,
                  proposal_encoder_checks=18, proposal_checks_per_fit={str(k): counts[k] for k in FITS},
                  mixed_trajectory_moves=3, exact_original_scientific_match=True)
    return result


def _unique_object(pairs):
    result = {}
    for key, value in pairs:
        require(key not in result, 'Duplicate JSON key in scientific trace')
        result[key] = value
    return result


def _trace_identity(path):
    with path.open('rb') as stream:
        sha = hashlib.file_digest(stream, 'sha256').hexdigest()
    return {'path': str(path), 'bytes': path.stat().st_size, 'sha256': sha}


def _inferences(diag):
    require(type(diag) is dict and _integer(diag.get('inference_calls')) and
            type(diag.get('phases')) is list and type(diag.get('simulations')) is int and
            diag['simulations'] == 0 and type(diag.get('forced_pass')) is bool, 'Entity diagnostic work schema')
    phases = diag['phases']
    if diag['forced_pass']:
        # The Main PASS is forced, but corrected rules can still require a
        # choice between eligible nobles. Only that auxiliary can remain.
        require(len(phases) <= 1, 'Forced PASS has unexpected auxiliary phases')
    else:
        require(bool(phases), 'Nonforced Entity decision has no Main phase')
    calls = 0
    for i, phase in enumerate(phases):
        main = i == 0 and not diag['forced_pass']
        require(type(phase) is dict and phase.get('phase') in ('Main', 'Payment', 'Return', 'Noble') and
                (phase['phase'] == 'Main') is main and type(phase.get('inference')) is list and
                (not diag['forced_pass'] or phase['phase'] == 'Noble'),
                'Entity phase schema/order')
        branch = phase.get('branch')
        require(branch in ('neural_greedy', 'auxiliary_heuristic', 'single_choice', 'late_heuristic') and
                (branch == 'auxiliary_heuristic') is (not main), 'Entity phase branch')
        require(len(phase['inference']) == int(branch == 'neural_greedy'), 'Entity phase inference count')
        calls += len(phase['inference'])
    require(calls == diag['inference_calls'], 'Entity total inference count differs')
    return calls


def validate_selection(measurement, trace_path):
    """Census the complete fixed128x4 trace; no policy or transition reexecution."""
    path = Path(trace_path).resolve(strict=True)
    before = _trace_identity(path)
    records = moves = inferences = 0
    policies, seeds = Counter(), []
    with gzip.open(path, 'rb') as stream:
        while True:
            line = stream.readline(MAX_LINE + 1)
            if not line:
                break
            require(len(line) <= MAX_LINE and line.endswith(b'\n'), 'Oversized/incomplete scientific record')
            row = json.loads(line, object_pairs_hook=_unique_object,
                             parse_constant=lambda x: require(False, 'Nonfinite trace JSON: ' + x))
            require(type(row) is dict and records < RECORDS, 'Extra/nonobject scientific record')
            wi, ci = divmod(records, SUPPORT)
            require(type(row.get('world_index')) is int and row['world_index'] == wi and
                    type(row.get('candidate_index')) is int and row['candidate_index'] == ci,
                    'Fixed world/candidate trace order differs')
            seed = row.get('seed')
            require(type(seed) is str and seed, 'Selection seed must retain original string representation')
            if ci == 0:
                require(seed not in seeds, 'Duplicate selection world seed')
                seeds.append(seed)
            require(seed == seeds[wi], 'Candidate branches do not share the world seed')
            steps = row.get('steps')
            require(type(steps) is list and steps and type(row.get('turns')) is int and row['turns'] == len(steps),
                    'Scientific trajectory step count')
            for index, step in enumerate(steps):
                require(type(step) is dict, 'Trace step must be an object')
                expected = 'forced' if index == 0 else 'greedy_entity' if index % 2 else 'raw035'
                require(step.get('policy') == expected and step.get('forced') is (index == 0),
                        'Forced/root/opponent policy dispatch differs')
                require(_integer(step.get('legal_count'), 1) and _integer(step.get('selected_index')) and
                        step['selected_index'] < step['legal_count'], 'Trace legal-action census')
                require(type(step.get('actor')) is int and step['actor'] in (0, 1) and
                        _integer(step.get('turn')), 'Trace actor/turn identity')
                if index:
                    require(step['actor'] == 1 - steps[index-1]['actor'] and
                            step['turn'] == steps[index-1]['turn'] + 1, 'Trace actor/turn progression')
                if expected == 'greedy_entity':
                    inferences += _inferences(step.get('policy_trace'))
                else:
                    require(step.get('policy_trace') is None, 'Unexpected non-Entity diagnostic')
                policies[expected] += 1
                moves += 1
            records += 1
    require(records == RECORDS and len(seeds) == WORLDS and policies['forced'] == RECORDS,
            'Incomplete fixed128x4 trace')
    require(_trace_identity(path) == before, 'Trace changed during independent census')
    greedy = policies['greedy_entity']
    expected = {
        'scientific_body': 1, 'choose': 1, 'trajectory': RECORDS, 'sample_world': WORLDS,
        'proposal': 1, 'raw_score': 2 + policies['raw035'], 'greedy_choose': greedy,
        'entity_infer': inferences, 'bridge_prepare': 5 * greedy,
        'bridge_encode': 2 * greedy, 'bridge_actions': 2 * greedy,
        'bridge_resolve': greedy, 'trace_sink': RECORDS, 'trace_finalize': 1,
        'shared_attention_forward': 3 + policies['raw035'],
        'legal': 2 * moves + 6 * greedy + WORLDS + 1,
    }
    result = _measurement(measurement, expected, no_unwinds=True)
    result.update(mode='full_fixed_selection', trace=before, worlds=WORLDS,
                  candidate_support=SUPPORT, trajectory_records=records, moves=moves,
                  policy_moves=dict(policies), entity_inference_calls=inferences,
                  legal_count_formula='2*all_moves + 6*Entity_moves + 128_world_checks + 1_root_check')
    return result
