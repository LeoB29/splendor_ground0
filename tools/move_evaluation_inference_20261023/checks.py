"""Model-free Stage A scientific, invocation and inference-scope checks."""
from copy import deepcopy
import hashlib
import json
import math
from pathlib import Path

from tools.move_evaluation_reuse_20261023 import checks as reuse
from tools.move_evaluation_timing_20261023 import checks as frozen

ROOT = Path(__file__).resolve().parents[2]
LOCK = ROOT / 'outputs/move_evaluation_study_20261023_preparation_01/candidate_lock.json'
LOCK_SHA256 = '8782b2704686111fac7b0f5e06a5dda6d6d1ea740a7448e5758308c7ce0ac4a7'
PROFILE = 'entity_onehot_greedy_canonical_atomic_prereveal_cpu_20261023_v1'
FINE_LABELS = ('infer_input_shape', 'infer_tensor_input', 'infer_input_finite',
               'infer_project', 'infer_transformer', 'infer_pool_norm',
               'infer_heads', 'infer_output_finite', 'infer_output_list')
OPTIONAL_ZERO_LABELS = ('entrypoint_checks', 'actual_trajectory_check', 'choose',
                        'trajectory', 'sample_world', 'decode_observer', 'raw_select',
                        'raw_score', 'proposal', 'shared_attention_forward',
                        'observer_encode', 'step', 'trace_sink', 'trace_finalize')
FIXTURE_CALLS = {**reuse.FIXTURE_CALLS, 'entity_infer': 14,
                 **dict.fromkeys(FINE_LABELS, 14)}
require, canonical, digest = frozen.require, frozen.canonical, frozen.digest


def _fine_summary(measurement):
    rows = {r['label']: r for r in measurement['rows']}
    entity = rows['entity_infer']
    denominator = entity['inclusive_ns']
    scopes = [rows[label] for label in FINE_LABELS]
    # These nine nonnested helpers are the only selected descendants of _infer.
    require(all(row['inclusive_ns'] == row['exclusive_ns'] for row in scopes),
            'Fine inference helper contains unexpected selected descendants')
    require(sum(row['inclusive_ns'] for row in scopes) + entity['exclusive_ns'] == denominator,
            'Fine inference helpers and parent residual do not partition inference')
    return {
        'schema': 'entity_inference_selected_scope_summary_v1',
        'inference_callbacks': entity['calls'], 'inference_inclusive_ns': denominator,
        'parent_residual_exclusive_ns': entity['exclusive_ns'],
        'helper_exclusive_sum_ns': sum(row['exclusive_ns'] for row in scopes),
        'helper_scopes': [{
            'label': row['label'], 'calls': row['calls'],
            'exclusive_ns': row['exclusive_ns'], 'inclusive_ns': row['inclusive_ns'],
            'fraction_of_inference_inclusive': row['exclusive_ns']/denominator if denominator else None,
        } for row in scopes],
        'target_manifest': [row.copy() for row in measurement['manifest']
                            if row['label'] in FINE_LABELS or row['label'] == 'entity_infer'],
        'scope': {
            'denominator': 'Instrumented entity_infer inclusive duration over the recorded callbacks.',
            'regions': 'Selected Python scopes include unselected descendants, native kernels, waits and timer overhead.',
            'inclusive_parents_and_children_must_not_be_added': True,
            'pure_kernel_measurement': False, 'instrumentation_overhead_subtracted': False,
            'predicted_latency_savings': False, 'model_execution_performed_here': False,
        },
    }


def validate_fixture(plain_scientific, timed_scientific, original_reference, measurement):
    result = reuse.validate_fixture(plain_scientific, timed_scientific, original_reference, measurement)
    result.update(frozen._measurement(measurement, FIXTURE_CALLS, no_unwinds=False))
    result['schema'] = 'move_evaluation_inference_fixture_check_v1'
    result['inference_scopes'] = _fine_summary(measurement)
    return result


def _panel_identities():
    data = LOCK.read_bytes()
    require(hashlib.sha256(data).hexdigest() == LOCK_SHA256, 'Frozen candidate lock changed')
    roots = json.loads(data)['roots']
    require(type(roots) is list and len(roots) == 18, 'Fixed eighteen-root panel required')
    result = []
    for index, row in enumerate(roots):
        require(type(row['fit']) is int and row['fit'] == frozen.FITS[index//6], 'Fixed panel fit assignment')
        observer = deepcopy(row['observer'])
        require(set(observer) == {'schema', 'context', 'observation', 'ruleset_id', 'catalogue_sha256'} and
                observer['schema'] == 'corrected_live_planner_observer_v1' and
                observer['context']['fit'] == row['fit'], 'Fixed panel observer envelope')
        del observer['context']
        observer['schema'] = 'external_screen_observation_v1'
        result.append({'id': row['id'], 'fit': row['fit'], 'observer_sha256': digest(observer)})
    return result, {'path': LOCK.relative_to(ROOT).as_posix(), 'bytes': len(data), 'sha256': LOCK_SHA256}


def _vector(value, size):
    return type(value) is list and len(value) == size and all(
        type(x) in (int, float) and math.isfinite(x) for x in value)


def _captured_trace(row):
    trace = row['trace']
    require(type(trace) is dict and set(trace) == {
        'schema', 'profile_id', 'observer_sha256', 'observer_seed', 'action_sha256',
        'forced_pass', 'phases', 'inference_calls', 'simulations'}, 'Panel decision trace schema')
    require(trace['schema'] == 'entity_greedy_decision_v1' and trace['profile_id'] == PROFILE,
            'Unchanged Entity policy identity')
    require(trace['observer_sha256'] == row['observer_sha256'] and
            type(trace['observer_seed']) is int and trace['observer_seed'] == int(row['observer_sha256'][:16], 16),
            'Panel observer seed/hash binding')
    require(type(row['action']) is dict and trace['action_sha256'] == digest(row['action']) and
            (row['action'].get('action_type') == 'PASS') is trace['forced_pass'], 'Panel action/trace binding')
    calls = frozen._inferences(trace)
    for phase in trace['phases']:
        neural = phase['branch'] == 'neural_greedy'
        logits = phase.get('canonical_logits')
        require('canonical_logits' in phase and
                (_vector(logits, 67) if neural else logits is None), 'Captured policy logits missing/invalid')
        require(phase.get('policy_logits_sha256') == (digest(logits) if neural else None),
                'Captured policy logits digest')
        for callback in phase['inference']:
            require(type(callback) is dict and set(callback) == {
                'input_sha256', 'output_sha256', 'tokens', 'output83'} and
                _vector(callback['tokens'], 31*48) and _vector(callback['output83'], 83),
                'Captured full inference input/output required')
            require(callback['input_sha256'] == digest(callback['tokens']) and
                    callback['output_sha256'] == digest(callback['output83']), 'Captured callback digest mismatch')
    return calls


def validate_panel(reference_scientific, plain_scientific, timed_scientific, measurement):
    require(canonical(reference_scientific) == canonical(plain_scientific) == canonical(timed_scientific),
            'Panel scientific output differs across reference/plain/timed')
    value = reference_scientific
    require(type(value) is dict and set(value) == {'schema', 'repetitions', 'panel_size', 'rows'} and
            value['schema'] == 'entity_public_inference_panel_v1' and
            type(value['repetitions']) is int and value['repetitions'] == 8 and
            type(value['panel_size']) is int and value['panel_size'] == 18,
            'Fixed public inference panel schema')
    rows = value['rows']
    require(type(rows) is list and len(rows) == 144, 'Complete eight-by-eighteen panel required')
    identities, lock = _panel_identities()
    expected_keys = {'repetition', 'index', 'id', 'fit', 'observer_sha256', 'action', 'trace'}
    calls, first = 0, {}
    for ordinal, row in enumerate(rows):
        repetition, index = divmod(ordinal, 18)
        require(type(row) is dict and set(row) == expected_keys and
                type(row['repetition']) is int and row['repetition'] == repetition and
                type(row['index']) is int and row['index'] == index, 'Fixed panel row order/schema')
        require(type(row['fit']) is int and row['fit'] == frozen.FITS[index//6] and
                all(row[k] == identities[index][k] for k in ('id', 'fit', 'observer_sha256')),
                'Panel identity differs from frozen candidate lock')
        calls += _captured_trace(row)
        scientific = canonical({'action': row['action'], 'trace': row['trace']})
        if repetition == 0:
            first[index] = scientific
        require(first[index] == scientific, 'Repeated public input changed scientific result')
    expected = {'scientific_body': 1, 'probe_panel': 1, 'greedy_choose': 144,
                'observer_decode': 144, 'legal': 144, 'entity_infer': calls,
                'reference_greedy_choose': 0, **dict.fromkeys(reuse.PREPARED_LABELS, 144),
                **dict.fromkeys(reuse.LEGACY_LABELS, 0), **dict.fromkeys(FINE_LABELS, calls)}
    labels = {row.get('label') for row in measurement.get('rows', []) if type(row) is dict}
    expected.update({label: 0 for label in OPTIONAL_ZERO_LABELS if label in labels})
    result = frozen._measurement(measurement, expected, no_unwinds=True)
    result.update(schema='move_evaluation_inference_panel_check_v1', mode='fixed_public_inference_panel',
                  panel_size=18, repetitions=8, decisions=144, entity_inference_calls=calls,
                  scientific_sha256=digest(value), candidate_lock=lock,
                  exact_reference_scientific_match=True, repeated_inputs_scientifically_identical=True,
                  inference_scopes=_fine_summary(measurement))
    return result
