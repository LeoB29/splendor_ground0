"""Model-free invocation census for exact per-decision public preparation reuse.

Timing integrity and fixture science retain the frozen validated checker. The
new census checks the deliberately reduced preparation work, not a renamed
version of the old redundant call counts. Policy replay remains a separate gate.
"""
from collections import Counter
import gzip
import json
from pathlib import Path

from tools.move_evaluation_timing_20261023 import checks as frozen


FITS, WORLDS, SUPPORT, RECORDS = frozen.FITS, frozen.WORLDS, frozen.SUPPORT, frozen.RECORDS
MAX_LINE = frozen.MAX_LINE
require, canonical, digest = frozen.require, frozen.canonical, frozen.digest
PREPARED_LABELS = ('prepare_once', 'encode_prepared', 'main_prepared', 'resolve_prepared')
LEGACY_LABELS = ('legacy_bridge_prepare', 'legacy_bridge_encode',
                 'legacy_bridge_actions', 'legacy_bridge_resolve')
IMPLEMENTATION_LABELS = ('greedy_choose', 'reference_greedy_choose') + PREPARED_LABELS + LEGACY_LABELS
FIXTURE_CALLS = {**frozen.FIXTURE_CALLS, 'reference_greedy_choose': 0,
                 **{label: 17 for label in PREPARED_LABELS}}


def _implementation(result, measurement):
    """Keep measured implementation provenance outside all scientific payloads."""
    result['schema'] = 'move_evaluation_reuse_timing_check_v1'
    entries = [row.copy() for row in measurement['manifest']
               if row['label'] in IMPLEMENTATION_LABELS]
    result['implementation_manifest'] = {
        'schema': 'per_decision_reuse_timing_targets_v1',
        'targets': entries, 'sha256': digest(entries),
        'administrative_only': True, 'scientific_policy_label_changed': False,
    }
    result['scope']['original_policy_replay_required_separately'] = True
    result['scope']['count_reduction_is_not_a_latency_saving_estimate'] = True
    return result


def validate_fixture(ordinary_scientific, timed_scientific, original_reference, measurement):
    """Exact old fixture science; seventeen new preparations, zero old choices."""
    # This validates all original fifteen fixtures, eighteen proposal encoders,
    # three fits, known mixed terminal trajectory, and unchanged model counts.
    result = frozen.validate_fixture(ordinary_scientific, timed_scientific,
                                     original_reference, measurement)
    result.update(frozen._measurement(measurement, FIXTURE_CALLS, no_unwinds=False))
    result['legacy_helper_fixture_counts_unconstrained'] = True
    return _implementation(result, measurement)


def _census(trace_path):
    """Independent ordered trace census; no inference or transition execution."""
    path = Path(trace_path).resolve(strict=True)
    before = frozen._trace_identity(path)
    records = moves = inferences = 0
    policies, seeds = Counter(), []
    with gzip.open(path, 'rb') as stream:
        while True:
            line = stream.readline(MAX_LINE + 1)
            if not line:
                break
            require(len(line) <= MAX_LINE and line.endswith(b'\n'), 'Oversized/incomplete scientific record')
            row = json.loads(line, object_pairs_hook=frozen._unique_object,
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
                policy = 'forced' if index == 0 else 'greedy_entity' if index % 2 else 'raw035'
                require(step.get('policy') == policy and step.get('forced') is (index == 0),
                        'Forced/root/opponent policy dispatch differs')
                require(frozen._integer(step.get('legal_count'), 1) and
                        frozen._integer(step.get('selected_index')) and
                        step['selected_index'] < step['legal_count'], 'Trace legal-action census')
                require(type(step.get('actor')) is int and step['actor'] in (0, 1) and
                        frozen._integer(step.get('turn')), 'Trace actor/turn identity')
                if index:
                    require(step['actor'] == 1 - steps[index-1]['actor'] and
                            step['turn'] == steps[index-1]['turn'] + 1, 'Trace actor/turn progression')
                if policy == 'greedy_entity':
                    inferences += frozen._inferences(step.get('policy_trace'))
                else:
                    require(step.get('policy_trace') is None, 'Unexpected non-Entity diagnostic')
                policies[policy] += 1
                moves += 1
            records += 1
    require(records == RECORDS and len(seeds) == WORLDS and policies['forced'] == RECORDS,
            'Incomplete fixed128x4 trace')
    require(frozen._trace_identity(path) == before, 'Trace changed during independent census')
    return before, moves, policies, inferences


def validate_selection(measurement, trace_path):
    """Check unchanged full work and exactly one preparation per Entity move."""
    trace, moves, policies, inferences = _census(trace_path)
    greedy, raw = policies['greedy_entity'], policies['raw035']
    expected = {
        'scientific_body': 1, 'choose': 1, 'trajectory': RECORDS,
        'sample_world': WORLDS, 'proposal': 1, 'raw_score': 2 + raw,
        'greedy_choose': greedy, 'reference_greedy_choose': 0,
        'entity_infer': inferences, 'shared_attention_forward': raw + 3,
        'observer_decode': greedy + 1, 'legal': 2 * moves + greedy + WORLDS + 1,
        'step': moves, 'trace_sink': RECORDS, 'trace_finalize': 1,
        **{label: greedy for label in PREPARED_LABELS},
        **{label: 0 for label in LEGACY_LABELS},
    }
    result = frozen._measurement(measurement, expected, no_unwinds=True)
    result.update(mode='full_fixed_selection', trace=trace, worlds=WORLDS,
                  candidate_support=SUPPORT, trajectory_records=RECORDS, moves=moves,
                  policy_moves=dict(policies), entity_inference_calls=inferences,
                  legal_count_formula='2*all_moves + Entity_moves + 128_world_checks + 1_root_check',
                  observer_decode_count_formula='Entity_moves + 1_root_observer')
    result['preparation_call_reduction'] = {
        'scope': 'Expected source-path census, not measured wall-time savings.',
        'entity_choices': greedy,
        'observer_decode': {'reference_expected': 6 * greedy + 1,
                            'reuse_expected': greedy + 1, 'fewer_calls': 5 * greedy},
        'legal': {'reference_expected': 2 * moves + 6 * greedy + WORLDS + 1,
                  'reuse_expected': expected['legal'], 'fewer_calls': 5 * greedy},
        'legacy_bridge_prepare_expected': 5 * greedy,
        'new_prepare_once_expected': greedy,
        'cross_decision_cache_tested_here': False,
    }
    return _implementation(result, measurement)
