"""Fixed, descriptive qualification-logging measurements; no strength inference."""
import math
import statistics
from collections import Counter

from . import common as c

ARM_IDS = ('frozen_cpu0', 'frozen_cpu2', 'research_cpu0', 'research_cpu2')
CASE_IDS = ('private_reserves', 'gold_returns')
ACTUAL_GATE = dict(maximum_mean_ratio=1.05, maximum_added_mean_seconds=.010,
                   maximum_added_p95_seconds=.025)
REPLAY_GATE = dict(maximum_added_mean_seconds=.005, maximum_added_p95_seconds=.025)
TIMESTAMPS = ('start_ns', 'serialized_ns', 'written_ns', 'flushed_ns', 'synced_ns', 'digested_ns')
COMPONENTS = ('serialization', 'write', 'flush', 'fsync', 'chain_digest')


def numeric(value):
    return type(value) in (float, int) and math.isfinite(value)


def distribution(values):
    values = list(values)
    c.require(all(numeric(x) for x in values), 'Finite observations; missing is not zero')
    values.sort()
    if not values:
        return dict(n=0, minimum=None, median=None, mean=None, p95=None, maximum=None)
    return dict(n=len(values), minimum=values[0], median=statistics.median(values),
                mean=statistics.mean(values), p95=values[math.ceil(.95 * len(values)) - 1], maximum=values[-1])


def _difference_gate(off, on, ready, contract):
    left, right = distribution(off), distribution(on)
    available = bool(off and on)
    mean_delta = right['mean'] - left['mean'] if available else None
    p95_delta = right['p95'] - left['p95'] if available else None
    ratio = right['mean'] / left['mean'] if available and left['mean'] > 0 else None
    checks = dict(mean=ready and right['mean'] <= left['mean'] + contract['maximum_added_mean_seconds'],
                  p95=ready and right['p95'] <= left['p95'] + contract['maximum_added_p95_seconds'])
    if 'maximum_mean_ratio' in contract:
        checks['mean_ratio'] = ready and right['mean'] <= contract['maximum_mean_ratio'] * left['mean']
    return dict(off=left, on=right, evaluated=bool(ready), passed=bool(ready and all(checks.values())),
                on_minus_off_mean_seconds=mean_delta, on_minus_off_p95_seconds=p95_delta,
                on_over_off_mean=ratio, checks={k: bool(v) if ready else None for k, v in checks.items()})


def overhead_gate(rows):
    """Only the64 prospectively assigned overhead rows decide this gate."""
    selected = [r for r in rows if r['spec']['stage'] == 'overhead']
    c.require(len(selected) <= 64, 'Bounded overhead inventory')
    counts = Counter()
    for row in selected:
        spec = row['spec']; mode = spec['logging_timing_enabled']
        c.require(type(mode) is bool and spec['arm_id'] in ARM_IDS and spec['case_id'] in CASE_IDS,
                  'Explicit fixed arm, fixture and instrumentation mode')
        c.require(row['arm_id'] == spec['arm_id'], 'Original arm binding')
        counts[spec['arm_id'], spec['case_id'], mode] += 1
    c.require(all(n <= 4 for n in counts.values()), 'No replacement or overrepresented overhead cell')
    complete = len(selected) == 64 and all(counts[a, p, m] == 4 for a in ARM_IDS for p in CASE_IDS for m in (False, True))
    arms = {}
    for arm in ARM_IDS:
        group = [r for r in selected if r['spec']['arm_id'] == arm]
        by_mode = {mode: [] for mode in (False, True)}
        all_success = True
        for row in group:
            receipt = row['clock']; elapsed = receipt.get('receipt_elapsed_seconds')
            c.require(receipt['deadline_seconds'] == 1., 'Unchanged full one-second request clock')
            if elapsed is not None:
                c.require(numeric(elapsed) and elapsed >= 0, 'Valid original response arrival')
            success = receipt['status'] == 'ok' and receipt['accepted'] is True and numeric(elapsed) and 0 <= elapsed <= 1.
            evidence = logging_measurements(row)
            success = success and evidence['measurement_accepted'] and evidence.get('record_count', 0) > 0
            all_success = all_success and success
            if success:
                by_mode[row['spec']['logging_timing_enabled']].append(elapsed)
        ready = complete and len(group) == 16 and all_success and all(len(v) == 8 for v in by_mode.values())
        arms[arm] = dict(attempted=len(group), all_original_responses_accepted=all_success,
                         **_difference_gate(by_mode[False], by_mode[True], ready, ACTUAL_GATE))
    return dict(schema='startup_logging_actual_overhead_gate_v1', attempted=len(selected), planned=64,
                complete=complete, evaluated=complete, passed=complete and all(v['passed'] for v in arms.values()),
                per_arm=arms, contract=ACTUAL_GATE, p95_definition='nearest_rank',
                p95_small_sample_note='With8 observations per condition and arm, nearest-rank p95 is the maximum.',
                margin_700ms='descriptive_only', fixture_counts_per_arm_condition=4,
                paired_or_population_inference=False, automatic_player_change=False)


def replay_overhead_gate(rows):
    """Saved record replay: no model; each independently byte-verified row is timed."""
    c.require(len(rows) <= 64, 'Bounded saved-record replay inventory')
    counts = Counter(); groups = {}
    for row in rows:
        c.require(row['case_id'] in CASE_IDS and type(row['logging_timing_enabled']) is bool,
                  'Frozen replay fixture/mode')
        c.require(numeric(row['elapsed_seconds']) and row['elapsed_seconds'] >= 0, 'Replay timing')
        counts[row['case_id'], row['logging_timing_enabled']] += 1
    c.require(all(n <= 16 for n in counts.values()), 'No replay replacement or extra observation')
    complete = len(rows) == 64 and all(counts[p, m] == 16 for p in CASE_IDS for m in (False, True))
    for case in CASE_IDS:
        selected = [r for r in rows if r['case_id'] == case]
        verified = all(r['bytes_verified'] is True for r in selected)
        modes = {mode: [r['elapsed_seconds'] for r in selected if r['logging_timing_enabled'] is mode] for mode in (False, True)}
        groups[case] = dict(bytes_verified=verified, **_difference_gate(modes[False], modes[True], complete and verified, REPLAY_GATE))
    return dict(schema='startup_logging_saved_replay_overhead_gate_v1', attempted=len(rows), planned=64,
                complete=complete, passed=complete and all(g['passed'] for g in groups.values()),
                per_fixture=groups, contract=REPLAY_GATE, threshold_origin='New prospective absolute instrumentation allowance',
                p95_definition='nearest_rank; with16 observations p95 is the maximum', model_calls=0,
                establishes_actual_handler_overhead=False)


def logging_measurements(row):
    """Derive spans from raw clock markers without calling a model or reading a log."""
    profile = row.get('logging_profile'); expected = row['spec']['logging_timing_enabled']
    if profile is None:
        return dict(available=False, measurement_accepted=False, instrumented=expected,
                    reason='missing_saved_logging_profile', records=[])
    from .logging_probe import validate_profile
    proof = validate_profile(profile, expected_enabled=expected, child_profile=row.get('child_profile'))
    c.require(profile['summary'] == proof, 'Logging summary must derive from raw original markers')
    c.require(type(profile['instrumented']) is bool and profile['instrumented'] is expected,
              'Logging sidecar mode must match original request')
    records = profile['records']; components = {k: [] for k in COMPONENTS}; measured = []
    handlers = (row.get('child_profile') or {}).get('profiles', [])
    handler_spans = [(p['handle_start_ns'], p['handle_end_ns']) for p in handlers
                     if type(p.get('handle_start_ns')) is int and type(p.get('handle_end_ns')) is int]
    handler_bounds_available = bool(handlers) and len(handler_spans) == len(handlers)
    for record in records:
        markers = record['timestamps']
        c.require(set(markers) == set(TIMESTAMPS), 'Complete explicit logging timestamp shape')
        stamps = [markers[k] for k in TIMESTAMPS]
        if not expected:
            c.require(all(x is None for x in stamps), 'OFF logging cannot claim measured durations')
            continue
        present = [x for x in stamps if x is not None]
        c.require(all(type(x) is int and x >= 0 for x in present) and present == sorted(present), 'Ordered native clock markers')
        c.require(stamps == present + [None] * (len(stamps) - len(present)), 'Only a retained trailing partial logging span')
        complete = len(present) == len(stamps)
        if record['completed'] is True:
            c.require(complete and record['error'] is None, 'Completed recorder has all markers and no error')
        durations = {key: (stamps[j + 1] - stamps[j]) / 1e9 if stamps[j] is not None and stamps[j + 1] is not None else None
                     for j, key in enumerate(COMPONENTS)}
        total = (stamps[-1] - stamps[0]) / 1e9 if complete else None
        inside = complete and any(a <= stamps[0] <= stamps[-1] <= b for a, b in handler_spans)
        after_failure = record['event'] in ('worker_decision_failed', 'worker_research_decision_failed', 'entrypoint_failed', 'failed')
        measured.append(dict(sequence=record['sequence'], event=record['event'], completed=record['completed'],
                             error=record['error'], total_seconds=total, components_seconds=durations,
                             wholly_inside_handler=inside, failure_evidence=after_failure))
        for key, value in durations.items():
            if value is not None: components[key].append(value)
    inside = [v for v in measured if v['wholly_inside_handler'] and v['total_seconds'] is not None]
    return dict(available=expected and bool(measured), measurement_accepted=proof['measurement_accepted'], instrumented=expected,
                reason=None if expected else 'timers_disabled_not_zero_cost', record_count=len(records),
                timed_records=len(measured), complete_timed_records=sum(x['completed'] for x in measured),
                total_recording_seconds=sum(x['total_seconds'] for x in measured if x['total_seconds'] is not None) if expected else None,
                handler_bounds_available=handler_bounds_available,
                wholly_in_handler_recording_seconds=sum(x['total_seconds'] for x in inside) if expected and handler_bounds_available else None,
                in_handler_prevalidation_recording_seconds=sum(x['total_seconds'] for x in inside if not x['failure_evidence']) if expected and handler_bounds_available else None,
                in_handler_failure_evidence_seconds=sum(x['total_seconds'] for x in inside if x['failure_evidence']) if expected and handler_bounds_available else None,
                component_distributions={k: distribution(v) for k, v in components.items()} if expected else None,
                records=measured,
                interpretation='Elapsed serialization/write/flush/fsync/digest spans include scheduling and storage waits. They are not CPU-only costs. Failure logging may follow the guard rejection.')


def measurements(row):
    from tools.splendorust_pool6_repeat_timing_20261027.analysis import measurements as immutable
    value = immutable(row); spec = row['spec']
    value.update(stage=spec['stage'], arm_id=spec['arm_id'], case_id=spec['case_id'],
                 route=spec['route'], logical_cpu=spec['logical_cpu'],
                 logging_timing_enabled=spec['logging_timing_enabled'], logging=logging_measurements(row))
    logging_seconds = value['logging'].get('wholly_in_handler_recording_seconds')
    handler = value['handler_seconds']
    value['handler_residual_after_measured_logging_seconds'] = handler - logging_seconds if handler is not None and logging_seconds is not None else None
    c.require(value['handler_residual_after_measured_logging_seconds'] is None or value['handler_residual_after_measured_logging_seconds'] >= -1e-9,
              'Disjoint logging spans must fit inside original handler')
    return value


def analyze(records, panel, matrix, complete, boundary=lambda: None):
    c.require(type(complete) is bool and len(matrix['requests']) == 96 and len(records) <= 96, 'Fixed96-request analysis')
    if complete:
        c.require(len(records) == 96 and all(r['clock']['status'] == 'ok' and r['clock']['accepted'] is True for r in records),
                  'Complete means the full accepted original-clock workload')
    values = []
    for row, spec in zip(records, matrix['requests']):
        boundary(); c.require(row['spec'] == spec, 'Exact fixed ordered analysis prefix')
        values.append(measurements(row))
    if complete:
        c.require(all(v['logging']['measurement_accepted'] and
                      (not v['logging_timing_enabled'] or v['logging']['available']) for v in values),
                  'Complete diagnostic requires all mandatory logging evidence')
    gate = overhead_gate(records)
    if complete: c.require(gate['passed'], 'Detailed diagnostic requires the unchanged overhead gate')
    c.require({p['case_id'] for p in panel['positions']} == set(CASE_IDS), 'Exact two predeclared fixtures')
    groups = []
    for stage in ('overhead', 'diagnostic'):
        for arm in ARM_IDS:
            for case in CASE_IDS:
                for mode in ((False, True) if stage == 'overhead' else (True,)):
                    selected = [v for v in values if v['stage'] == stage and v['arm_id'] == arm and v['case_id'] == case and v['logging_timing_enabled'] is mode]
                    groups.append(dict(stage=stage, arm_id=arm, case_id=case, logging_timing_enabled=mode,
                        attempted=len(selected), completed=sum(v['status'] == 'ok' for v in selected),
                        receipt=distribution([v['receipt_seconds'] for v in selected if v['receipt_seconds'] is not None]),
                        handler=distribution([v['handler_seconds'] for v in selected if v['handler_seconds'] is not None]),
                        measured_logging=distribution([v['logging']['wholly_in_handler_recording_seconds'] for v in selected if v['logging'].get('wholly_in_handler_recording_seconds') is not None]),
                        observations=selected))
    return dict(schema='startup_logging_descriptive_analysis_v1', complete=complete, attempted=len(records), planned=96,
                overhead_gate=gate, groups=groups, logging_timing_available=sum(v['logging']['available'] for v in values),
                failed_requests=sum(v['failed_request'] for v in values), deadline_failures=sum(v['deadline_failed'] for v in values),
                margin_crossings=sum(v['margin_crossing'] for v in values),
                statistical_inference=False, causal_attribution=False, core_placement_verdict=None,
                admission_passed=False, automatic_player_change=False, new_games=0,
                limitations=['A targeted qualification-logging diagnostic, not a deployment or core-placement benchmark.',
                    'TimerOFF still writes and synchronously persists every original forensic event.',
                    'The residual after measured logging includes search, validation, other serialization and scheduling; it is not pure search time.',
                    'Small repeated samples are temporally dependent; p95 is the sample maximum for the frozen overhead cells.',
                    'Failure records remain failures; a saved action does not rehabilitate its original clock.'])
