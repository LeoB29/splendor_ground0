"""Bounded timing diagnosis only; no optimized player or strength games here."""
import argparse
from contextlib import ExitStack
import math
import os
from pathlib import Path
import shutil
import statistics
import sys
import time
import traceback

import psutil
from . import common as c
from tools.external_strength_screen_20261018 import rules, audit as ledger
from tools.external_strength_screen_20261018.common import serialize_action, deserialize_action, legal_digest
from tools.future_training_rules_20261021 import decode_training_snapshot, verify_runtime_sources
from tools.local_player_qualification_128.clock import processor_topology
from tools.splendorust_search_scale_20261024 import audit, qualification as q
from tools.splendorust_search_scale_20261024.log_audit import verify_worker_logs

MODES = ('baseline', 'profiled')


def bytes_used():
    return sum(path.stat().st_size for path in c.OUT.rglob('*') if path.is_file())


class Envelope:
    """One non-resettable job clock, aggregate resident-worker resource bound."""
    def __init__(self, claim):
        self.start, self.wall_start = claim['monotonic_origin'], claim['unix_origin']
        self.end = self.start + c.JOB_SECONDS
        self.stage = None
        self.deadline, self.wall_deadline = self.end, self.wall_start + c.JOB_SECONDS
        self.waited, self.admitted, self.admissions = 0., False, 0
        self.evidence = None
        self.resident = {}
        self.peak_combined_rss = 0

    def enable_evidence_accounting(self):
        self.evidence = dict(baseline=bytes_used(), parent_bytes=0, workers={})
        self.log_baseline = {name: (c.OUT/name).stat().st_size if (c.OUT/name).exists() else 0
                             for name in ('runner_stdout.log', 'runner_stderr.log')}
        c.WRITE_OBSERVER = self.note_write

    def note_write(self, path, size):
        if not path.resolve().is_relative_to((c.OUT/'workers').resolve()):
            self.evidence['parent_bytes'] += size

    def register_worker(self, folder):
        self.evidence['workers'][folder] = None

    def own_worker(self, child):
        proc = psutil.Process(child.process.pid)
        identity = (proc.pid, proc.create_time())
        c.require(identity not in self.resident, 'Duplicate resident worker identity')
        self.resident[identity] = proc

    def close_worker(self, folder, child=None):
        if child is not None:
            identities = [identity for identity in self.resident if identity[0] == child.process.pid]
            for identity in identities:
                self.resident.pop(identity)
        self.evidence['workers'][folder] = sum(path.stat().st_size for path in folder.rglob('*') if path.is_file())

    def aggregate_worker_rss(self, exclude_pid=None, additional_rss=0):
        seen, total = set(), additional_rss
        for identity, root in self.resident.items():
            if identity[0] == exclude_pid:
                continue
            try:
                c.require(root.create_time() == identity[1], 'Resident root PID identity changed')
                processes = [root] + root.children(recursive=True)
            except psutil.NoSuchProcess:
                raise ValueError('Resident worker disappeared before owned cleanup')
            for process in processes:
                try:
                    child_identity = (process.pid, process.create_time())
                    if child_identity not in seen:
                        total += process.memory_info().rss
                        seen.add(child_identity)
                except psutil.NoSuchProcess:
                    # Helpers may exit between enumeration and observation; the
                    # owning StudyWorker independently checks required roles.
                    continue
        self.peak_combined_rss = max(self.peak_combined_rss, total)
        c.require(total <= 2*c.GiB, 'Combined resident workers/helpers RSS exceeded 2GiB')
        return total

    def guard_worker_sample(self, child, telemetry):
        """Include a new tree before its constructor has returned readiness."""
        self.aggregate_worker_rss(exclude_pid=child.process.pid, additional_rss=telemetry['rss_bytes'])

    def accounted_bytes(self):
        if self.evidence is None:
            return bytes_used()
        total = self.evidence['baseline'] + self.evidence['parent_bytes']
        for folder, closed in self.evidence['workers'].items():
            total += closed if closed is not None else sum(path.stat().st_size for path in folder.rglob('*') if path.is_file())
        for name, initial in self.log_baseline.items():
            path = c.OUT/name
            total += max(0, (path.stat().st_size if path.exists() else 0)-initial)
        return total + 65536

    def begin(self, stage):
        mono, wall = time.monotonic(), time.time()
        self.stage = stage
        cap = c.STAGES[stage]
        cutoff = c.JOB_SECONDS if stage == 'final' else c.COLLECTION_CUTOFF_SECONDS
        self.deadline = min(mono+cap, self.start+cutoff, self.end)
        self.wall_deadline = min(wall+cap, self.wall_start+cutoff, self.wall_start+c.JOB_SECONDS)
        self.guard()
        c.save(c.OUT/'stages'/f'{stage}_start.json', dict(time=c.now(), monotonic=mono, unix_time=wall,
            monotonic_deadline=self.deadline, unix_deadline=self.wall_deadline, cap_seconds=cap))

    def remaining(self):
        return min(self.deadline-time.monotonic(), self.wall_deadline-time.time(),
                   self.end-time.monotonic(), self.wall_start+c.JOB_SECONDS-time.time())

    def effective_deadline(self):
        return time.monotonic()+self.remaining()

    def guard(self):
        c.require(self.remaining() > 0, 'Stage/global real-clock deadline')
        c.require(shutil.disk_usage(c.ROOT).free >= 5*c.GiB, 'Free disk reserve')
        cap = (512 if self.stage == 'final' else 448)*c.MiB
        c.require(self.accounted_bytes() < cap, 'Study evidence ceiling')
        self.aggregate_worker_rss()

    def timeout(self, maximum):
        self.guard()
        return min(maximum, self.remaining())

    def admit(self):
        self.guard()
        begun = time.monotonic()
        available = psutil.virtual_memory().available
        stable = begun if available >= 2*c.GiB else None
        minimum = available
        fast = self.admitted and stable is not None
        while not fast:
            self.guard()
            current, available = time.monotonic(), psutil.virtual_memory().available
            minimum = min(minimum, available)
            if available < 2*c.GiB:
                stable = None
            elif stable is None:
                stable = current
            c.require(self.waited+current-begun < 600, 'Cumulative RAM wait exceeded ten minutes')
            if stable is not None and (self.admitted or current-stable >= 60):
                break
            time.sleep(.5)
        elapsed = time.monotonic()-begun
        self.waited += elapsed
        self.admitted = True
        self.admissions += 1
        c.require(self.waited < 600, 'Cumulative RAM admission bound')
        if not fast:
            c.save(c.OUT/'memory_admissions'/f'{self.admissions:06d}.json', dict(time=c.now(), wait_seconds=elapsed,
                cumulative_wait_seconds=self.waited, minimum_available_bytes=minimum,
                stable_seconds=60 if self.admissions == 1 else 0))

    def finish(self):
        self.guard()
        if self.evidence is not None:
            c.require(bytes_used() <= self.accounted_bytes(), 'Unaccounted study evidence')
        c.save(c.OUT/'stages'/f'{self.stage}_end.json', dict(time=c.now(), monotonic=time.monotonic(), unix_time=time.time()))


def verify_runtime(freeze, envelope=None):
    for item in freeze['inputs']:
        if envelope:
            envelope.guard()
        c.verify([item])
    c.require(sys.version == freeze['python'] and c.file_hash(sys.executable) == freeze['python_executable_sha256'],
              'Python runtime changed')
    c.require(processor_topology() == freeze['topology'], 'Processor topology changed')
    for item in freeze['torch_runtime']:
        path = Path(item['absolute_path'])
        c.require(path.stat().st_size == item['bytes'] and c.file_hash(path) == item['sha256'], 'Pinned Torch runtime changed')
    from tools.splendorust_search_scale_20261024.compression import runtime_contract
    c.require(runtime_contract() == freeze['compression_runtime'], 'Compression runtime changed')


def prepare():
    from .fixtures import build_manifest, repeated_schedule
    from tools.splendorust_search_scale_20261024.compression import runtime_contract
    c.require(not (c.OUT/'freeze.json').exists(), 'Already frozen; no replacement')
    c.OUT.mkdir(parents=True, exist_ok=True)
    sources = verify_runtime_sources()
    c.require(c.file_hash(c.PRIOR/'freeze.json') == c.PRIOR_FREEZE_SHA, 'Prior source freeze changed')
    prior = c.read(c.PRIOR/'freeze.json')
    c.require(c.read(c.PRIOR/'completion.json')['status'] == 'completed_verified' and
              c.read(c.PRIOR/'execution.json')['status'] == 'completed', 'Prior study is not verified')
    for name in ('prelaunch_tests.json', 'prelaunch_review.json'):
        receipt = c.read(c.OUT/name)
        c.require(receipt['status'] == 'passed', 'Prelaunch tests/review missing')
        c.verify(receipt['source_inventory'])
    c.require(shutil.disk_usage(c.ROOT).free >= 7*c.GiB, 'Launch requires 7GiB free disk')
    manifest = build_manifest()
    c.save(c.OUT/'fixtures.json', manifest)
    schedule = repeated_schedule(manifest['development'])
    c.save(c.OUT/'diagnostic_schedule.json', schedule)
    c.save(c.OUT/'model_free_qualification.json', q.model_free_checks(32))
    c.save(c.OUT/'qualification_plan.json', q.build_plan(32))
    items = {item['path']: item for item in prior['inputs']}
    paths = set(path for path in c.PRIOR.rglob('*') if path.is_file())
    paths.update((c.ROOT/'tools/splendorust_timing_20261024').glob('*.py'))
    paths.update(c.ROOT.glob('tests/test_splendorust_timing*.py'))
    paths.add(c.PROTOCOL)
    paths.update(path for path in c.OUT.rglob('*') if path.is_file())
    for path in paths:
        if '__pycache__' in path.parts:
            continue
        item = c.artifact(path)
        if item['path'] in items:
            c.require(items[item['path']] == item, 'Prior frozen dependency changed')
        items[item['path']] = item
    compression = runtime_contract()
    c.require(compression == prior['compression_runtime'], 'Qualified codec runtime changed')
    freeze = dict(schema='splendorust_timing_diagnostic_v1', time=c.now(),
        inputs=sorted(items.values(), key=lambda item: item['path']),
        python=sys.version, python_executable_sha256=c.file_hash(sys.executable),
        torch_runtime=prior['torch_runtime'], compression_runtime=compression,
        topology=processor_topology(), native_artifacts=prior['native_artifacts'],
        source_guard=sources, job_seconds=c.JOB_SECONDS, stage_seconds=c.STAGES,
        collection_cutoff_seconds=c.COLLECTION_CUTOFF_SECONDS, automatic_retries=0,
        seconds_per_move=1., internal_seconds=.9, diagnostic_margin_seconds=.7,
        simulations=32, modes=list(MODES), profile=c.artifact(c.profile_path(32)),
        max_rss_bytes=2*c.GiB, max_evidence_bytes=512*c.MiB, final_reserve_bytes=64*c.MiB,
        diagnosis_only=True, larger_data_deferred=True, validation_positions_reserved=True,
        strength_games=0, optimization_selected=False,
        authorization='Approved timing diagnosis followed by a measured separate optimization; this freeze launches diagnosis only')
    verify_runtime(freeze)
    c.save(c.OUT/'freeze.json', freeze)
    print(c.canonical(dict(prepared=True, freeze_sha256=c.file_hash(c.OUT/'freeze.json'),
        inputs=len(items), requests=len(schedule), model_calls=0)), flush=True)


def clean_result(result, mode):
    c.require(mode in MODES, 'Unknown instrumentation mode')
    c.require(type(result) is dict, 'Response object required')
    clean = dict(result)
    if mode == 'profiled':
        c.require(type(clean.get('timing_profile')) is dict, 'Profiled response lacks timing profile')
        clean.pop('timing_profile')
    else:
        c.require('timing_profile' not in clean, 'Baseline instrumentation unexpectedly enabled')
    return clean


def make_worker(name, envelope, freeze, mode):
    from tools.splendorust_search_scale_20261024.adapters import validate_ready_metadata
    from tools.future_training_rules_20261021 import SOURCE_HASHES
    c.require(mode in MODES and 'torch' not in sys.modules, 'Referee must remain model-free; valid mode required')
    envelope.admit()
    folder = c.OUT/'workers'/name
    envelope.register_worker(folder)
    child = c.StudyWorker('tools.splendorust_timing_20261024.worker', folder,
        native_artifacts=freeze['native_artifacts'], profile_enabled=mode == 'profiled',
        aggregate_sample_guard=envelope.guard_worker_sample,
        affinity=[freeze['topology']['selected_logical_processor']], initialization_timeout=envelope.timeout(60),
        absolute_deadline=envelope.effective_deadline(), max_message_bytes=4*c.MiB, max_request_bytes=c.MiB,
        max_log_bytes=16*c.MiB, max_rss_bytes=2*c.GiB, resource_guard=envelope.guard,
        env={'PYTHONPATH': str(c.ROOT/'src')+os.pathsep+str(c.ROOT), 'STUDY_WORKER_DIR': str(folder),
             'STUDY_SEARCH_SIMULATIONS': '32', 'STUDY_TIMING_PROFILE': '1' if mode == 'profiled' else '0',
             'CUDA_VISIBLE_DEVICES': '', 'RAYON_NUM_THREADS': '1', 'TOKIO_WORKER_THREADS': '1'})
    try:
        envelope.own_worker(child)
        child.study_after_close = lambda: envelope.close_worker(folder, child)
        metadata = dict(child.initialization['result'])
        c.require(metadata.pop('timing_mode', None) == mode, 'Worker timing mode identity changed')
        telemetry = child._sample()
        validate_ready_metadata(metadata, expected_sources=SOURCE_HASHES, worker_pid=child.process.pid,
            process_tree=telemetry['process_tree'], expected_quota=32)
        child.timing_mode = mode
        c.require(not child._failure, 'Worker process supervision failed at initialization')
        envelope.guard()
        return child
    except BaseException:
        child.close()
        raise


def decision(child, fixture, spec, envelope, snapshot=None):
    envelope.admit()
    state = decode_training_snapshot(fixture['snapshot'] if snapshot is None else snapshot)
    observer, before = rules.observation_payload(state), rules.serialize_snapshot(state)
    c.require(observer == fixture['observer'] and c.digest(observer) == fixture['observer_sha256'], 'Fixture observer binding')
    ledger.check_observer(before, observer)
    legal = rules.legal_actions(state)
    actions = [serialize_action(action) for action in legal]
    request_id = f'request-{child._counter+1:08d}'
    c.save(child.output_dir/'inputs'/f'{request_id}.json', dict(player='search', observer=observer, before=before))
    def validator(result):
        clean = clean_result(result, child.timing_mode)
        c.require(clean.get('action') in actions, 'PLAYER_ILLEGAL_ACTION: response action')
        return audit.validate_result(clean, observer, 'search', expected_quota=32)
    receipt = child.request({'player': 'search', 'observer': observer}, envelope.timeout(1.), validate=validator)
    row = dict(spec, fixture_id=fixture['fixture_id'], before=before, observer=observer,
        completed=receipt['status'] == 'ok', clock=receipt, action=None, trace=None, after=None,
        worker_clock_file=(child.output_dir/f'{request_id}.json').relative_to(c.ROOT).as_posix())
    # The failed request is retained before any assertion, including resource
    # failures or a stage-clamped timeout. No failure is retried.
    if row['completed']:
        result = clean_result(receipt['result'], child.timing_mode)
        c.require(result['player'] == 'search' and result['observer_sha256'] == c.digest(observer) and
            result['legal_actions_sha256'] == legal_digest(legal) and result['legal_count'] == len(legal) and
            result['ruleset'] == rules.RULESET_ID and result['catalogue_sha256'] == rules.CATALOGUE_SHA256,
            'Worker binding differs')
        row.update(action=result['action'], trace=result['diagnostics'])
        row['after'] = rules.serialize_snapshot(rules.CorrectedEnv().step(state, deserialize_action(row['action'])))
    c.save(c.OUT/'decisions'/f"{spec['index']:04d}.json", row)
    envelope.guard()
    c.require(row['completed'], 'Diagnostic request failed; preserved without retries: '+str(receipt.get('detail')))
    audit.check_clock(receipt)
    audit.audit_decision(observer, row['action'], row['trace'], 'search', before=before,
                         after=row['after'], expected_quota=32)
    return row


def qualify(envelope, freeze):
    for mode in MODES:
        with make_worker('qualification_'+mode, envelope, freeze, mode) as child:
            receipt = child.request({'operation': 'entrypoint_checks'}, envelope.timeout(300))
            c.save(c.OUT/f'entrypoint_checks_{mode}.json', receipt)
            envelope.guard()
            c.require(receipt['status'] == 'ok' and receipt['result']['status'] == 'entrypoint_counterexamples_passed',
                      'Actual corrected-entrypoint qualification failed: '+str(receipt.get('detail')))
        check = verify_worker_logs(child.output_dir, boundary=envelope.guard, deadline=envelope.effective_deadline())
        c.save(c.OUT/f'qualification_log_integrity_{mode}.json', check)
        print(c.canonical(dict(stage='qualification', mode=mode, status='passed')), flush=True)


def fixture_lookup(manifest):
    fixtures = {fixture['fixture_id']: fixture for fixture in manifest['development']}
    c.require(len(fixtures) == len(manifest['development']), 'Duplicate fixture identity')
    return fixtures


def profile(envelope, freeze, manifest, schedule):
    from .fixtures import snapshot_for_request
    lookup, rows = fixture_lookup(manifest), []
    with ExitStack() as stack:
        children = {mode: stack.enter_context(make_worker('diagnostic_'+mode, envelope, freeze, mode)) for mode in MODES}
        for spec in schedule:
            row = decision(children[spec['mode']], lookup[spec['fixture_id']], spec, envelope,
                           snapshot=snapshot_for_request(manifest['development'], spec))
            rows.append(row)
            if len(rows) % 16 == 0:
                print(c.canonical(dict(stage='profiling', requests=len(rows), total=len(schedule))), flush=True)
    return rows


def distribution(values):
    c.require(values and all(type(value) in (int, float) and math.isfinite(value) for value in values), 'Finite timing sample required')
    ordered = sorted(values)
    index = .95*(len(ordered)-1)
    lower, upper = math.floor(index), math.ceil(index)
    return dict(count=len(values), mean=statistics.mean(values), median=statistics.median(values),
        p95=ordered[lower]+(ordered[upper]-ordered[lower])*(index-lower), maximum=max(values), minimum=min(values))


def validate_reproductions(rows, manifest, schedule):
    from .fixtures import snapshot_for_request
    c.require(len(rows) == len(schedule) and len({row['index'] for row in rows}) == len(schedule), 'Diagnostic inventory incomplete/duplicate')
    lookup = fixture_lookup(manifest)
    groups, historical = {}, 0
    for row, spec in zip(rows, schedule, strict=True):
        c.require(all(row.get(key) == value for key, value in spec.items()), 'Diagnostic schedule identity changed')
        fixture = lookup[row['fixture_id']]
        c.require(row['completed'] and row['before'] == snapshot_for_request(manifest['development'], spec)
                  and row['observer'] == fixture['observer'], 'Completed input binding changed')
        audit.check_clock(row['clock'])
        science = dict(action=row['action'], trace=row['trace'])
        previous = groups.setdefault(row['fixture_id'], science)
        c.require(c.canonical(science) == c.canonical(previous), 'Instrumentation/repeat changed exact decision or search evidence')
        prior = fixture.get('historical32')
        if prior:
            c.require(c.canonical(science) == c.canonical(dict(action=prior['action'], trace=prior['trace'])),
                      'Historical PUCT32 decision changed')
            historical += 1
    return dict(status='passed', decisions=len(rows), exact_reproduced_roots=len(groups), historical_comparisons=historical)


def summarize(rows):
    result = {'simulations': 32, 'interpretation': 'Instrumentation comparison only; no optimized player or strength conclusion',
              'modes': {}, 'margin_seconds': .7, 'deadline_seconds': 1., 'overlapping_spans_must_not_be_summed': True}
    by_pair = {}
    spans = {'worker': {}, 'parent': {}}
    for row in rows:
        key = (row['fixture_id'], row.get('variant', 'original'), row.get('pass_index', row.get('pass')))
        by_pair.setdefault(key, {})[row['mode']] = row['clock']['receipt_elapsed_seconds']
        if row['mode'] == 'profiled':
            for source, profile_value in (
                ('worker', row['clock']['result'].get('timing_profile', {})),
                ('parent', row['clock'].get('parent_timing', {})),
            ):
                totals = profile_value.get('totals', {}) if source == 'worker' else profile_value.get('spans', {})
                for label, span in totals.items():
                    if type(span) is dict and type(span.get('wall_ns')) in (int, float):
                        spans[source].setdefault(label, []).append(span['wall_ns']/1e9)
    for mode in MODES:
        selected = [row for row in rows if row['mode'] == mode]
        arrivals = [row['clock']['receipt_elapsed_seconds'] for row in selected]
        result['modes'][mode] = dict(arrival_seconds=distribution(arrivals),
            above_margin=sum(value > .7 for value in arrivals), above_deadline=sum(value > 1 for value in arrivals),
            first_request_arrival_seconds=selected[0]['clock']['receipt_elapsed_seconds'])
    c.require(all(set(pair) == set(MODES) for pair in by_pair.values()), 'Instrumentation paired sample incomplete')
    result['paired_profile_minus_baseline_seconds'] = distribution([pair['profiled']-pair['baseline'] for pair in by_pair.values()])
    result['observed_nested_spans_seconds'] = {source: {label: distribution(values) for label, values in groups.items()}
                                            for source, groups in spans.items()}
    result['scope_notes'] = [
        'Paired mode differences mix instrumentation cost with scheduling noise; no causal speedup claim.',
        'Parent process samples include post-arrival work, which is not charged to the move arrival deadline.',
        'Worker spans are nested wall durations; native exchange includes inference callbacks.',
        'Saved slow cases are diagnostic selection, not independent evidence of future tail latency.',
        'Independent reserved validation positions have not been evaluated by this attempt.',
    ]
    return result


def final_checks(envelope, freeze, manifest, schedule):
    paths = sorted((c.OUT/'decisions').glob('*.json'))
    rows = [c.read(path) for path in paths]
    c.require([path.name for path in paths] == [f"{spec['index']:04d}.json" for spec in schedule], 'Decision file inventory changed')
    reproductions = validate_reproductions(rows, manifest, schedule)
    for row in rows:
        envelope.guard()
        clock_path = c.ROOT/row['worker_clock_file']
        saved = c.read(clock_path)
        # Parent timing is appended after PersistentWorker has sealed its clock
        # receipt. It is auxiliary evidence, never a replacement clock.
        authoritative = {key: value for key, value in row['clock'].items() if key != 'parent_timing'}
        c.require(saved == authoritative, 'Trusted clock record changed')
        c.require(c.read(clock_path.parent/'inputs'/clock_path.name) ==
                  dict(player='search', observer=row['observer'], before=row['before']), 'Saved worker input changed')
        audit.audit_decision(row['observer'], row['action'], row['trace'], 'search',
                             before=row['before'], after=row['after'], expected_quota=32)
    c.save(c.OUT/'reproduction_checks.json', reproductions)
    logs = [verify_worker_logs(folder, boundary=envelope.guard, deadline=envelope.effective_deadline())
            for folder in sorted((c.OUT/'workers').iterdir())]
    c.require(len(logs) == 4, 'Expected two qualification and two diagnostic workers')
    c.save(c.OUT/'all_worker_log_integrity.json', dict(status='passed', workers=4, receipts=logs))
    summary = summarize(rows)
    summary['maximum_sampled_combined_worker_rss_bytes'] = envelope.peak_combined_rss
    c.save(c.OUT/'timing_summary.json', summary)
    verify_runtime(freeze, envelope)
    return summary


def run(expected_sha):
    c.require(c.file_hash(c.OUT/'freeze.json') == expected_sha, 'Freeze identity changed')
    claim = c.read(c.OUT/'watchdog_claim.json')
    c.require(claim['freeze_sha256'] == expected_sha, 'Watchdog identity')
    envelope, freeze = Envelope(claim), c.read(c.OUT/'freeze.json')
    envelope.enable_evidence_accounting()
    c.save(c.OUT/'launch.json', dict(time=c.now(), pid=os.getpid(), created=psutil.Process().create_time(), freeze_sha256=expected_sha))
    status, failure = 'incomplete', None
    try:
        envelope.begin('qualification')
        c.save(c.OUT/'supervisor_runtime_sources.json', dict(sources=verify_runtime_sources()))
        verify_runtime(freeze, envelope)
        envelope.admit()
        qualify(envelope, freeze)
        envelope.finish()
        manifest, schedule = c.read(c.OUT/'fixtures.json'), c.read(c.OUT/'diagnostic_schedule.json')
        envelope.begin('profiling')
        profile(envelope, freeze, manifest, schedule)
        envelope.finish()
        envelope.begin('final')
        final_checks(envelope, freeze, manifest, schedule)
        envelope.finish()
        status = 'completed_verified'
    except BaseException:
        failure = traceback.format_exc()
        c.save(c.OUT/'failure.json', dict(time=c.now(), stage=envelope.stage, error=failure))
    c.save(c.OUT/'completion.json', dict(status=status, time=c.now(), failure=failure, stage=envelope.stage,
        elapsed_seconds=time.monotonic()-envelope.start, wall_elapsed_seconds=time.time()-envelope.wall_start,
        freeze_sha256=expected_sha, larger_data_deferred=True, automatic_retries=0, strength_games=0,
        note='Acceptance also requires owning watchdog cleanup receipt with no errors; optimization and future runs need separate freeze.'))
    print(c.canonical(dict(status=status, stage=envelope.stage, failure=failure)), flush=True)
    return 0 if status == 'completed_verified' else 1


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('operation', choices=['prepare', 'run'])
    parser.add_argument('--freeze-sha256')
    args = parser.parse_args()
    return prepare() if args.operation == 'prepare' else run(args.freeze_sha256)


if __name__ == '__main__':
    raise SystemExit(main())
