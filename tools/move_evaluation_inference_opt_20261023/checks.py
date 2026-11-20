"""Model-free exact-science and selected-scope checks for the one Stage B candidate.

Worker eligibility is a separate authenticated initialization/response receipt,
checked by the runner. It is deliberately not inserted into scientific output.
"""
from pathlib import Path

from tools.move_evaluation_inference_20261023 import checks as stage_a
from tools.move_evaluation_reuse_20261023 import checks as reuse
from tools.move_evaluation_timing_20261023 import checks as frozen

ROOT = Path(__file__).resolve().parents[2]
OUT = ROOT / 'outputs/move_evaluation_inference_opt_20261023_01'
FINE_LABELS = stage_a.FINE_LABELS
FIXTURE_CALLS = stage_a.FIXTURE_CALLS
require = frozen.require


def validate_fixture(plain_scientific, timed_scientific, original_reference, measurement):
    result = stage_a.validate_fixture(plain_scientific, timed_scientific, original_reference, measurement)
    result['schema'] = 'move_evaluation_inference_opt_fixture_check_v1'
    result['scope']['eligibility_receipt_authenticated_separately_by_runner'] = True
    return result


def validate_panel(reference_scientific, plain_scientific, timed_scientific, measurement):
    result = stage_a.validate_panel(reference_scientific, plain_scientific, timed_scientific, measurement)
    result['schema'] = 'move_evaluation_inference_opt_panel_check_v1'
    result['scope']['eligibility_receipt_authenticated_separately_by_runner'] = True
    return result


def _trace_path(science):
    require(type(science) is dict and set(science) == {
        'decision', 'trace', 'profile_id', 'observer_sha256', 'ruleset_id', 'catalogue_sha256'},
        'Complete scientific selection wrapper required')
    require(science['profile_id'] == 'fixed4_raw035_own_greedy_entity_opponent_20261023_v1' and
            science['ruleset_id'] == 'splendor_base_qualified_rules_20261018_v1' and
            science['catalogue_sha256'] == 'db4776918e612eee5a6f2617719c85ec5f413d592bbab0a64058babc83a70312' and
            frozen._hash(science['observer_sha256']), 'Unchanged scientific selection identities')
    item = science['trace']
    require(type(item) is dict and set(item) == {'path', 'sha256', 'bytes', 'records'} and
            type(item['path']) is str and not Path(item['path']).is_absolute() and
            '..' not in Path(item['path']).parts and frozen._hash(item['sha256']) and
            frozen._integer(item['bytes'], 1) and type(item['records']) is int and item['records'] == 512,
            'Complete trace descriptor required')
    path = (ROOT/item['path']).resolve(strict=True)
    require(path.is_relative_to(OUT.resolve()) and path.name == 'continuations.jsonl.gz',
            'Trace must belong to this versioned output')
    actual = frozen._trace_identity(path)
    require(actual['sha256'] == item['sha256'] and actual['bytes'] == item['bytes'],
            'Scientific trace hash/size binding')
    return path


def validate_selection(science, measurement):
    """Authenticate trace and retain full512 census with one fine-scope call per callback."""
    path = _trace_path(science)
    result = reuse.validate_selection(measurement, path)
    callbacks = result['entity_inference_calls']
    expected = {**result['expected_calls'], **dict.fromkeys(FINE_LABELS, callbacks)}
    result.update(frozen._measurement(measurement, expected, no_unwinds=True))
    result.update(schema='move_evaluation_inference_opt_selection_check_v1',
                  scientific_sha256=frozen.digest(science),
                  inference_scopes=stage_a._fine_summary(measurement))
    result['scope']['eligibility_receipt_authenticated_separately_by_runner'] = True
    result['scope']['exact_previous_science_comparison_required_separately'] = True
    result['scope']['original_policy_replay_required_separately'] = True
    result['scope']['count_reduction_is_not_a_latency_saving_estimate'] = True
    return result
