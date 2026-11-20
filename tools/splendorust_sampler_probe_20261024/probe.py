"""Five-minute model-free microprofile of the unchanged Windows sampler."""
import argparse
from contextlib import contextmanager, ExitStack
from functools import wraps
import hashlib
import inspect
import io
import json
import math
import os
from pathlib import Path
import shutil
import statistics
import subprocess
import sys
import time
import traceback

import psutil
from tools.future_training_rules_20261021 import verify_runtime_sources
from tools.local_player_qualification_128.clock import _WindowsJob, processor_topology
from tools.splendorust_search_scale_20261024 import common as frozen
from tools.splendorust_search_scale_20261024.log_audit import verify_worker_logs

ROOT = frozen.ROOT
OUT = ROOT/'outputs/splendorust_sampler_probe_20261024_01'
PREVIOUS = ROOT/'outputs/splendorust_timing_20261024_01'
JOB_SECONDS = 300
GiB, MiB = 1024**3, 1024**2
require, canonical, digest = frozen.require, frozen.canonical, frozen.digest
read, now, file_hash, artifact, verify = frozen.read, frozen.now, frozen.file_hash, frozen.artifact, frozen.verify


def save(path, value):
    path = Path(path)
    require(path.resolve().is_relative_to(OUT.resolve()), 'Probe output boundary')
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open('x', encoding='utf-8') as stream:
        stream.write(canonical(value)+'\n')
        stream.flush()
        os.fsync(stream.fileno())
    return artifact(path)


def external_artifact(path):
    path = Path(path).resolve()
    return dict(absolute_path=str(path), bytes=path.stat().st_size, sha256=file_hash(path))


def numeric_summary(values):
    require(values and all(type(v) in (int, float) and math.isfinite(v) for v in values), 'Finite nonempty observations')
    ordered = sorted(values)
    offset = .95*(len(ordered)-1)
    low, high = math.floor(offset), math.ceil(offset)
    return dict(count=len(values), mean=statistics.mean(values), median=statistics.median(values),
        p95=ordered[low]+(ordered[high]-ordered[low])*(offset-low), minimum=min(values), maximum=max(values))


class Timings:
    """Nested inclusive/exclusive wall and current-parent CPU; no subtraction from clocks."""
    def __init__(self):
        self.stack, self.totals, self.records = [], {}, []

    @contextmanager
    def span(self, label):
        started, cpu = time.perf_counter_ns(), time.process_time_ns()
        frame = [0, 0]
        self.stack.append(frame)
        failed = False
        try:
            yield
        except BaseException:
            failed = True
            raise
        finally:
            elapsed_cpu, elapsed = time.process_time_ns()-cpu, time.perf_counter_ns()-started
            require(self.stack.pop() is frame, 'Sampler timing stack changed')
            if self.stack:
                self.stack[-1][0] += elapsed
                self.stack[-1][1] += elapsed_cpu
            row = dict(label=label, wall_ns=elapsed, cpu_ns=elapsed_cpu,
                exclusive_wall_ns=max(0, elapsed-frame[0]), exclusive_cpu_ns=max(0, elapsed_cpu-frame[1]),
                failed=failed)
            require(len(self.records) < 2048, 'Sampler interval ceiling')
            self.records.append(row)
            total = self.totals.setdefault(label, dict(calls=0, failures=0, wall_ns=0, cpu_ns=0,
                exclusive_wall_ns=0, exclusive_cpu_ns=0))
            total['calls'] += 1
            total['failures'] += int(failed)
            for key in ('wall_ns', 'cpu_ns', 'exclusive_wall_ns', 'exclusive_cpu_ns'):
                total[key] += row[key]

    def wrap(self, label, function):
        @wraps(function)
        def measured(*args, **kwargs):
            with self.span(label):
                return function(*args, **kwargs)
        return measured

    def snapshot(self):
        require(not self.stack, 'Unfinished timing spans')
        return dict(totals=self.totals, intervals=self.records, inclusive_labels_overlap=True,
            cpu_scope='current_sampler_parent_only', diagnostic_only=True)


@contextmanager
def instrument(timings):
    """Wrap exactly once, preserve result/exception and restore every binding."""
    patches = []
    def patch(owner, name, label):
        original = getattr(owner, name)
        patches.append((owner, name, original))
        setattr(owner, name, timings.wrap(label, original))
    try:
        for name in ('children', 'create_time', 'exe', 'ppid', 'memory_info', 'cpu_times',
                     'num_threads', 'cpu_affinity', 'is_running'):
            patch(psutil.Process, name, 'process.'+name)
        patch(Path, 'resolve', 'path.resolve')
        patch(Path, 'stat', 'path.stat')
        patch(frozen, 'file_hash', 'file_hash')
        yield
    finally:
        for owner, name, original in reversed(patches):
            setattr(owner, name, original)


def fixed_schedule():
    return [dict(index=3*cycle+offset, cycle=cycle, mode=mode)
            for cycle in range(10) for offset, mode in enumerate(('baseline', 'profiled', 'profiled'))]


def prepare():
    require(not (OUT/'freeze.json').exists(), 'Already frozen; no replacement')
    OUT.mkdir(parents=True, exist_ok=True)
    sources = verify_runtime_sources()
    require(read(PREVIOUS/'completion.json')['status'] == 'completed_verified' and
            read(PREVIOUS/'execution.json')['status'] == 'completed', 'Timing diagnosis must finish and clean up first')
    for name in ('prelaunch_tests.json', 'prelaunch_review.json'):
        receipt = read(OUT/name)
        require(receipt['status'] == 'passed', 'Prelaunch checks required')
        verify(receipt['source_inventory'])
    old = read(frozen.OUT/'freeze.json')
    paths = set((ROOT/'tools/splendorust_sampler_probe_20261024').glob('*.py'))
    paths.update(ROOT.glob('tests/test_splendorust_sampler_probe*.py'))
    # Bind the already-verified repository implementation dependency inventory,
    # without rehashing tens of thousands of unrelated historical game receipts.
    inputs = {item['path']: item for item in old['inputs'] if item['path'].startswith(('tools/', 'src/'))}
    for item in old['native_artifacts']:
        paths.add(ROOT/item['path'])
    paths.update((frozen.PROFILE, frozen.profile_path(32), PREVIOUS/'freeze.json', PREVIOUS/'completion.json',
                  PREVIOUS/'execution.json', PREVIOUS/'timing_summary.json', PREVIOUS/'reproduction_checks.json',
                  PREVIOUS/'all_worker_log_integrity.json', frozen.OUT/'freeze.json'))
    paths.update(path for path in OUT.rglob('*') if path.is_file())
    for path in paths:
        item = artifact(path)
        if item['path'] in inputs:
            require(item == inputs[item['path']], 'Preserved repository dependency changed')
        inputs[item['path']] = item
    psutil_root = Path(psutil.__file__).parent
    runtime_paths = {Path(sys.executable), Path(inspect.getfile(Path)), Path(os.__file__)}
    runtime_paths.update(path for path in psutil_root.rglob('*') if path.is_file() and path.suffix in ('.py', '.pyd'))
    import ntpath
    runtime_paths.add(Path(ntpath.__file__))
    import _pyio
    runtime_paths.add(Path(_pyio.__file__))
    runtime_paths.update(Path(sys.base_prefix).glob('python*.dll'))
    import _ctypes
    runtime_paths.add(Path(_ctypes.__file__))
    save(OUT/'schedule.json', fixed_schedule())
    inputs[artifact(OUT/'schedule.json')['path']] = artifact(OUT/'schedule.json')
    freeze = dict(schema='splendorust_model_free_sampler_probe_v1', time=now(), inputs=sorted(inputs.values(), key=lambda x:x['path']),
        runtime=[external_artifact(path) for path in sorted(runtime_paths)], psutil_version=psutil.__version__, python=sys.version,
        sources=sources, native_artifacts=old['native_artifacts'], topology=processor_topology(), job_seconds=300,
        scenarios=['one_idle_tree', 'two_idle_trees'], samples_per_scenario=30, profiled_per_scenario=20,
        baseline_per_scenario=10, model_calls=0, native_requests=0, model_loaded=False, automatic_retries=0,
        synthetic_echo_bytes=[1024, 16384, 65536, 65536, 16384, 1024], synthetic_echo_deadline_seconds=1.,
        max_rss_bytes=2*GiB, max_evidence_bytes=32*MiB, prior_diagnostic_freeze=artifact(PREVIOUS/'freeze.json'),
        scope='Unchanged sampler on idle authenticated executable trees; attribution only, not player timing qualification')
    verify_freeze(freeze)
    save(OUT/'freeze.json', freeze)
    print(canonical(dict(prepared=True, freeze_sha256=file_hash(OUT/'freeze.json'), model_calls=0)), flush=True)


def verify_freeze(freeze, boundary=lambda: None):
    require(freeze['python'] == sys.version and freeze['psutil_version'] == psutil.__version__, 'Runtime version changed')
    require(freeze['job_seconds'] == JOB_SECONDS and freeze['model_calls'] == 0 and freeze['native_requests'] == 0,
            'Model-free resource contract changed')
    for item in freeze['inputs']:
        boundary()
        verify([item])
    for item in freeze['runtime']:
        boundary()
        require(external_artifact(item['absolute_path']) == item, 'Installed runtime bytes changed')
    require(processor_topology() == freeze['topology'], 'Processor topology changed')


class Bound:
    def __init__(self, claim):
        self.start, self.wall = claim['monotonic_origin'], claim['unix_origin']
        self.workers, self.peak_rss = [], 0

    def remaining(self):
        return min(self.start+JOB_SECONDS-time.monotonic(), self.wall+JOB_SECONDS-time.time())

    def guard(self):
        require(self.remaining() > 0, 'Probe five-minute deadline')
        require(shutil.disk_usage(ROOT).free >= 5*GiB, 'Probe disk reserve')
        require(sum(path.stat().st_size for path in OUT.rglob('*') if path.is_file()) < 32*MiB, 'Probe evidence ceiling')

    def admission(self):
        self.guard()
        require(psutil.virtual_memory().available >= 2*GiB, 'Probe available RAM below 2GiB')

    def total_rss(self):
        seen, total = set(), 0
        for child in self.workers:
            if getattr(child, '_closed', False):
                continue
            for proc in [child._ps]+child._ps.children(recursive=True):
                identity = (proc.pid, proc.create_time())
                if identity not in seen:
                    total += proc.memory_info().rss
                    seen.add(identity)
        self.peak_rss = max(self.peak_rss, total)
        require(total <= 2*GiB, 'Aggregate probe RSS ceiling')
        return total


def new_worker(name, bound, freeze):
    bound.admission()
    child = frozen.StudyWorker('tools.splendorust_sampler_probe_20261024.idle_worker', OUT/'workers'/name,
        native_artifacts=freeze['native_artifacts'], affinity=[freeze['topology']['selected_logical_processor']],
        initialization_timeout=min(30., bound.remaining()), absolute_deadline=time.monotonic()+bound.remaining(),
        max_message_bytes=MiB, max_request_bytes=1024, max_log_bytes=MiB, max_rss_bytes=GiB,
        resource_guard=bound.guard, env={'PYTHONPATH': str(ROOT/'src')+os.pathsep+str(ROOT),
            'PROBE_SELECTED_CPU': str(freeze['topology']['selected_logical_processor']),
            'PROBE_NATIVE_ARTIFACTS': canonical(freeze['native_artifacts']),
            'PROBE_MONOTONIC_DEADLINE': str(time.monotonic()+bound.remaining()),
            'CUDA_VISIBLE_DEVICES': '', 'RAYON_NUM_THREADS': '1', 'TOKIO_WORKER_THREADS': '1'})
    try:
        result = child.initialization['result']
        require(result['probe_only'] is True and result['model_loaded'] is False and result['model_calls'] == 0 and
            result['native_requests'] == 0 and result['native_artifacts'] == freeze['native_artifacts'] and
            result['pid'] == child.process.pid and result['sources'] == freeze['sources'], 'Idle worker identity mismatch')
        sample = child._sample()
        require(not child._failure, 'Idle worker authentication failed')
        roles = {row['role']: row for row in sample['process_tree']}
        require({'python', 'search', 'reference'}.issubset(roles), 'Incomplete authenticated compute tree')
        require(set(result['native_pids']) == {roles['search']['pid'], roles['reference']['pid']}, 'Native ownership mismatch')
        bound.workers.append(child)
        bound.total_rss()
        return child
    except BaseException:
        child.close()
        raise


def identity(sample):
    return sorted((row['pid'], row['created'], row['parent_pid'], row['role'], row['executable'], tuple(row['affinity']))
                  for row in sample['process_tree'])


def observed_sample(active, other=None):
    sample = active._sample()
    require(not active._failure, 'Original sampler rejected process tree')
    other_rss = 0
    if other is not None:
        # Match the diagnostic's additional idle-tree aggregate-RSS work. This
        # is explicitly outside the original sampler's scientific provenance.
        for process in [other._ps]+other._ps.children(recursive=True):
            process.create_time()
            other_rss += process.memory_info().rss
    require(sample['rss_bytes']+other_rss <= 2*GiB, 'Aggregate observed RSS ceiling')
    return dict(active=sample, other_tree_rss_bytes=other_rss)


def raw_readline_mechanism():
    """Bounded executable proof on this runtime, without sampling a live worker."""
    class CountedRaw(io.FileIO):
        def __init__(self, *args, **kwargs):
            self.read_calls = 0
            super().__init__(*args, **kwargs)

        def read(self, size=-1):
            self.read_calls += 1
            return super().read(size)

    read_fd, write_fd = os.pipe()
    payload = b'x'*1024+b'\n'
    try:
        require(os.write(write_fd, payload) == len(payload), 'Synthetic pipe write incomplete')
    finally:
        os.close(write_fd)
    with CountedRaw(read_fd, 'rb', closefd=True) as stream:
        result = stream.readline(2048)
        calls = stream.read_calls
    require(result == payload, 'Raw readline synthetic content changed')
    return dict(payload_bytes=len(payload), raw_read_calls=calls, bytes_returned=len(result),
        implementation='io.FileIO subclass, inherited readline, overridden read delegates exactly once',
        scope='Mechanism illustration on this Python runtime; not a direct worker syscall count', model_calls=0)


def echo_checks(child, bound, freeze):
    rows = []
    for index, size in enumerate(freeze['synthetic_echo_bytes']):
        bound.admission()
        def validate(result):
            require(result == dict(operation='synthetic_echo', bytes=size, payload='x'*size,
                                   model_calls=0, native_requests=0), 'Synthetic response mismatch')
            return True
        receipt = child.request(dict(operation='synthetic_echo', bytes=size), min(1., bound.remaining()), validate=validate)
        row = dict(index=index, payload_bytes=size, clock=receipt,
                   stdout_type=type(child.process.stdout).__module__+'.'+type(child.process.stdout).__qualname__)
        save(OUT/'synthetic_echo'/f'{index:02d}.json', row)
        rows.append(row)
        bound.guard()
        require(receipt['status'] == 'ok', 'Synthetic echo deadline/protocol failure; no retry')
    save(OUT/'synthetic_echo_summary.json', dict(status='passed', rows=[dict(index=row['index'],
        payload_bytes=row['payload_bytes'], arrival_seconds=row['clock']['receipt_elapsed_seconds'],
        total_request_seconds=row['clock']['elapsed_seconds'], stdout_type=row['stdout_type']) for row in rows],
        mechanism='Unchanged raw stdout reader and unchanged trusted clock; no inference or native request'))


def collect(expected_sha):
    require(file_hash(OUT/'freeze.json') == expected_sha, 'Probe freeze changed')
    freeze, claim = read(OUT/'freeze.json'), read(OUT/'watchdog_claim.json')
    require(claim['freeze_sha256'] == expected_sha, 'Probe launch identity')
    bound = Bound(claim)
    failure, rows = None, []
    try:
        sources = verify_runtime_sources()
        require(sources == freeze['sources'], 'Probe source guard changed')
        require('torch' not in sys.modules, 'Probe imported Torch')
        verify_freeze(freeze, bound.guard)
        save(OUT/'raw_readline_mechanism.json', raw_readline_mechanism())
        with ExitStack() as stack:
            active = stack.enter_context(new_worker('active', bound, freeze))
            expected_identity = identity(active._sample())
            other = None
            expected_other_identity = None
            for scenario in freeze['scenarios']:
                if scenario == 'two_idle_trees':
                    other = stack.enter_context(new_worker('idle', bound, freeze))
                    other_initial = other._sample()
                    require(not other._failure, 'Idle initial authentication failed')
                    expected_other_identity = identity(other_initial)
                for spec in read(OUT/'schedule.json'):
                    bound.admission()
                    profiler = Timings() if spec['mode'] == 'profiled' else None
                    started, cpu = time.perf_counter_ns(), time.process_time_ns()
                    result = None
                    try:
                        if profiler:
                            with instrument(profiler):
                                with profiler.span('entire_sample'):
                                    result = observed_sample(active, other)
                        else:
                            result = observed_sample(active, other)
                        elapsed, elapsed_cpu = time.perf_counter_ns()-started, time.process_time_ns()-cpu
                        require(identity(result['active']) == expected_identity, 'Sampled ownership/affinity identity changed')
                    except BaseException:
                        save(OUT/'failed_sample.json', dict(**spec, scenario=scenario, error=traceback.format_exc(),
                            wall_ns=time.perf_counter_ns()-started, cpu_ns=time.process_time_ns()-cpu, result=result,
                            profile=None if profiler is None else profiler.snapshot()))
                        raise
                    row = dict(**spec, scenario=scenario, wall_ns=elapsed, cpu_ns=elapsed_cpu, result=result,
                               profile=None if profiler is None else profiler.snapshot())
                    save(OUT/'samples'/scenario/f"{spec['index']:03d}.json", row)
                    rows.append(row)
                    bound.guard()
                if other is not None:
                    other_final = other._sample()
                    require(not other._failure and identity(other_final) == expected_other_identity,
                            'Idle tree final authentication failed')
                bound.total_rss()
            echo_checks(active, bound, freeze)
        logs = [verify_worker_logs(child.output_dir, boundary=bound.guard, deadline=time.monotonic()+bound.remaining())
                for child in bound.workers]
        save(OUT/'log_integrity.json', dict(status='passed', workers=len(logs), receipts=logs))
        require(len(rows) == 60 and 'torch' not in sys.modules, 'Probe count or model-free gate')
        summary = summarize(rows)
        summary['maximum_sampled_combined_rss_bytes'] = bound.peak_rss
        save(OUT/'summary.json', summary)
        verify_freeze(freeze, bound.guard)
        bound.guard()
    except BaseException:
        failure = traceback.format_exc()
        save(OUT/'failure.json', dict(time=now(), error=failure, completed_samples=len(rows)))
    save(OUT/'completion.json', dict(status='completed_verified' if failure is None else 'failed', failure=failure,
        time=now(), samples=len(rows), model_calls=0, native_requests=0, model_loaded=False,
        elapsed_seconds=time.monotonic()-bound.start, freeze_sha256=expected_sha,
        acceptance_requires_outer_cleanup=True))
    return 0 if failure is None else 1


def summarize(rows):
    result = dict(model_calls=0, native_requests=0, model_loaded=False, scenarios={},
        caveats=['Idle process attribution is not actual-player timing qualification.',
                 'Two-tree extra RSS supervision is reported separately from the one-tree original sampler.',
                 'Inclusive spans overlap; use exclusive spans for attribution and retain timer overhead.',
                 'No optimization is selected or implemented by this probe.'])
    for scenario in ('one_idle_tree', 'two_idle_trees'):
        selected = [row for row in rows if row['scenario'] == scenario]
        require(len(selected) == 30 and [row['index'] for row in selected] == list(range(30)), 'Incomplete sampler schedule')
        out = dict(modes={}, profiled_labels={})
        for mode, expected in (('baseline', 10), ('profiled', 20)):
            mode_rows = [row for row in selected if row['mode'] == mode]
            require(len(mode_rows) == expected, 'Sampler mode count')
            out['modes'][mode] = numeric_summary([row['wall_ns']/1e9 for row in mode_rows])
        labels = sorted({label for row in selected if row['profile'] for label in row['profile']['totals']})
        for label in labels:
            items = [row['profile']['totals'].get(label, {}) for row in selected if row['profile']]
            out['profiled_labels'][label] = {metric: numeric_summary([item.get(metric, 0)/1e9 for item in items])
                for metric in ('wall_ns', 'exclusive_wall_ns', 'cpu_ns', 'exclusive_cpu_ns')}
            out['profiled_labels'][label]['calls'] = sum(item.get('calls', 0) for item in items)
        result['scenarios'][scenario] = out
    return result


def cleanup_owned(process, job, streams, *, monotonic_origin, unix_origin, root_identity=None):
    """Terminate ownership immediately and share one clock-clamped exit wait."""
    begun = time.monotonic()
    local_deadline = begun + 5.0
    def remaining():
        return max(0.0, min(local_deadline - time.monotonic(),
            monotonic_origin + JOB_SECONDS - time.monotonic(),
            unix_origin + JOB_SECONDS - time.time()))
    allowance = remaining()
    errors, owned, confirmed, unconfirmed = [], [], [], []
    if process is not None:
        try:
            parent = psutil.Process(process.pid)
            identity = {'pid': parent.pid, 'created': parent.create_time()}
            if root_identity is None:
                require(process.poll() is None, 'Missing ownership identity for exited runner')
            require(root_identity is None or identity == root_identity, 'Runner PID identity changed')
            owned = [(p, {'pid': p.pid, 'created': p.create_time()})
                     for p in [parent] + parent.children(recursive=True)]
        except psutil.NoSuchProcess:
            pass
        except BaseException as error:
            errors.append('Ownership enumeration: ' + repr(error))
    if job is not None:
        try:
            job.close()
        except BaseException as error:
            errors.append('Job closure: ' + repr(error))
    kill_errors = {}
    for child, identity in reversed(owned):
        try:
            # The original process object checks PID creation identity before
            # kill; never operate on a fresh PID lookup without that identity.
            child.kill()
        except psutil.NoSuchProcess:
            pass
        except psutil.AccessDenied as error:
            kill_errors[(identity['pid'], identity['created'])] = repr(error)
    if process is not None:
        try:
            if process.poll() is None:
                process.kill()
            process.wait(timeout=remaining())
        except BaseException as error:
            if process.poll() is None:
                errors.append('Runner handle wait: ' + repr(error))
    for child, identity in owned:
        try:
            status = child.wait(timeout=remaining())
            confirmed.append({**identity, 'exit_status': status})
        except psutil.NoSuchProcess:
            confirmed.append({**identity, 'state': 'absent'})
        except (psutil.TimeoutExpired, psutil.AccessDenied) as error:
            unconfirmed.append({**identity, 'error': repr(error),
                'kill_error': kill_errors.get((identity['pid'], identity['created']))})
    if unconfirmed:
        errors.append('Owned process exits unconfirmed before cleanup deadline')
    for stream in streams:
        if stream is not None:
            try:
                stream.close()
            except BaseException as error:
                errors.append('Log closure: ' + repr(error))
    return {'errors': errors, 'confirmed': confirmed, 'unconfirmed': unconfirmed,
            'observed_owned_identities': len(owned), 'maximum_seconds': 5.0,
            'effective_allowance_seconds': allowance, 'elapsed_seconds': time.monotonic()-begun,
            'method': 'single_exit_wait_budget_clamped_to_monotonic_and_utc_job_deadlines_v2'}


def run(expected_sha):
    require(not any((OUT/name).exists() for name in ('watchdog_claim.json', 'execution.json', 'collector_stdout.log',
        'collector_stderr.log')), 'No probe relaunch or clock reset')
    require(file_hash(OUT/'freeze.json') == expected_sha, 'Probe freeze changed')
    require(shutil.disk_usage(ROOT).free >= 7*GiB, 'Probe launch requires 7GiB free')
    started, wall = time.monotonic(), time.time()
    save(OUT/'watchdog_claim.json', dict(time=now(), monotonic_origin=started, unix_origin=wall,
        freeze_sha256=expected_sha, watchdog_pid=os.getpid(), watchdog_created=psutil.Process().create_time()))
    process = job = stdout = stderr = None
    root_identity = None
    status, errors = 'failed', []
    try:
        stdout, stderr = (OUT/'collector_stdout.log').open('xb'), (OUT/'collector_stderr.log').open('xb')
        environment = os.environ.copy()
        environment.update(PYTHONPATH=str(ROOT/'src')+os.pathsep+str(ROOT), PYTHONDONTWRITEBYTECODE='1',
            OMP_NUM_THREADS='1', MKL_NUM_THREADS='1', OPENBLAS_NUM_THREADS='1', CUDA_VISIBLE_DEVICES='')
        process = subprocess.Popen([sys.executable, '-B', '-u', '-m', 'tools.splendorust_sampler_probe_20261024.probe',
            'collect', '--freeze-sha256', expected_sha], cwd=ROOT, env=environment, stdout=stdout, stderr=stderr,
            creationflags=getattr(subprocess, 'CREATE_NO_WINDOW', 0))
        root_identity = dict(pid=process.pid, created=psutil.Process(process.pid).create_time())
        job = _WindowsJob(process.pid)
        save(OUT/'watchdog.json', dict(pid=process.pid, created=psutil.Process(process.pid).create_time(),
            watchdog_pid=os.getpid(), deadline_seconds=300, freeze_sha256=expected_sha))
        while process.poll() is None:
            if min(started+JOB_SECONDS-time.monotonic(), wall+JOB_SECONDS-time.time()) <= 0:
                status = 'watchdog_deadline'
                break
            time.sleep(.05)
        if process.poll() is not None:
            status = 'completed' if process.returncode == 0 else 'collector_failed'
    except BaseException:
        save(OUT/'watchdog_error.json', dict(error=traceback.format_exc()))
    finally:
        cleanup = cleanup_owned(process, job, (stdout, stderr), monotonic_origin=started,
                                unix_origin=wall, root_identity=root_identity)
        errors = cleanup['errors']
        elapsed = max(time.monotonic()-started, time.time()-wall)
        if elapsed >= JOB_SECONDS:
            status = 'watchdog_deadline'
        save(OUT/'execution.json', dict(status='cleanup_failed' if errors else status, time=now(),
            cleanup_errors=errors, cleanup_receipt=cleanup, elapsed_seconds=elapsed, exit_code=None if process is None else process.returncode,
            freeze_sha256=expected_sha, model_calls=0, native_requests=0))
    return 0 if status == 'completed' and not errors else 1


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('operation', choices=('prepare', 'run', 'collect'))
    parser.add_argument('--freeze-sha256')
    args = parser.parse_args()
    return prepare() if args.operation == 'prepare' else globals()[args.operation](args.freeze_sha256)


if __name__ == '__main__':
    raise SystemExit(main())
