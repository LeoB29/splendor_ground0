"""Study identities and two-native descendant supervision; no model import."""
from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path
import sys
import subprocess
import time

import psutil
from tools.local_player_qualification_128.clock import PersistentWorker
from .compression import StreamingGzipLog

ROOT = Path(__file__).resolve().parents[2]
OUT = ROOT/'outputs/splendorust_one_second_20261024_03'
PRIOR = ROOT/'outputs/splendorust_one_second_20261024_02'
PRIOR_FREEZE_SHA = '5b86734afec36093da0916cab4811c7abf076f273fd7c52ce42afc2dc4988f60'
PRIOR_ATTEMPTS = (
    (ROOT/'outputs/splendorust_one_second_20261024_01', '617b5c4852a8eddc18156b3df98b377f93c1aa57a6dad9291ad4ed478efc3f34'),
    (PRIOR, PRIOR_FREEZE_SHA),
)
PREP = ROOT/'outputs/splendorust_roadmap_20261024_preparation_01'
PROFILE = ROOT/'outputs/external_strength_screen_20261018_02/opponent_profile.json'
PROFILE_SHA = 'cd915908df1a7c6f39e55cfc9df59d88002a2b5ce167ce9512803e7c63d64424'
BUILD = ROOT/'outputs/move_evaluation_20261023_build_01/build_receipt.json'
PROTOCOL = ROOT/'docs/engineering/splendorust_one_second_comparison_20261024.md'
EXECUTION_PROTOCOL = ROOT/'docs/engineering/splendorust_one_second_execution_v3_20261024.md'
DOMAIN = 'splendorust_policy_puct16_1s_20261024_v1'
OPERATIONAL_DOMAIN = 'splendorust_policy_puct16_1s_20261024_v3_operational'
GiB, MiB = 1024**3, 1024**2
JOB_SECONDS = 21600
STAGES = {'qualification':3600, 'operational':1800, 'main':14400, 'final':1800}
WRITE_OBSERVER = None


def require(condition, message):
    if not condition:
        raise ValueError(message)


def canonical(value):
    return json.dumps(value, sort_keys=True, separators=(',', ':'), allow_nan=False)


def digest(value):
    return hashlib.sha256(canonical(value).encode()).hexdigest()


def read(path):
    return json.loads(Path(path).read_text(encoding='utf-8'))


def now():
    return datetime.now(timezone.utc).isoformat()


def file_hash(path):
    with Path(path).open('rb') as f:
        return hashlib.file_digest(f, 'sha256').hexdigest()


def artifact(path):
    p = Path(path).resolve()
    return {'path': p.relative_to(ROOT).as_posix(), 'sha256': file_hash(p), 'bytes': p.stat().st_size}


def verify(items):
    for item in items:
        require(artifact(ROOT/item['path']) == item, 'Changed frozen input: '+item['path'])


def save(path, value):
    p = Path(path)
    require(p.resolve().is_relative_to(OUT.resolve()), 'Output outside study directory')
    p.parent.mkdir(parents=True, exist_ok=True)
    with p.open('x', encoding='utf-8') as f:
        f.write(canonical(value)+'\n'); f.flush(); os.fsync(f.fileno())
    if WRITE_OBSERVER is not None: WRITE_OBSERVER(p,p.stat().st_size)
    return artifact(p)


def make_schedule():
    rows=[]
    for stage, pairs in [('operational',3), ('main',100)]:
        for pair in range(pairs):
            domain=OPERATIONAL_DOMAIN if stage=='operational' else DOMAIN
            seed=int.from_bytes(hashlib.sha256(canonical([domain,stage,pair]).encode()).digest(),'big')
            for game in range(2):
                rows.append(dict(stage=stage,pair_index=pair,game_index=2*pair+game,
                                 search_seat=(pair+game)%2,seed_decimal=str(seed)))
    return rows


class StudyWorker(PersistentWorker):
    """Unchanged trusted arrival clock, separately authenticated two-native tree."""
    def __init__(self, *args, native_artifacts, **kwargs):
        self._tree_cpu={}; self._roles={}; self._owners={}; self._verified={}
        self._initializing=True
        self._python=Path(sys.executable).resolve()
        self._natives={ (ROOT/a['path']).resolve(): ('search' if 'native_build_v2' in a['path'] else 'greedy')
                       for a in native_artifacts }
        verify(native_artifacts)
        self._expected={self._python:file_hash(self._python)}
        self._expected.update({(ROOT/a['path']).resolve():a['sha256'] for a in native_artifacts})
        self._console=(Path(os.environ['SystemRoot'])/'System32/conhost.exe').resolve() if os.name=='nt' else None
        if self._console: self._expected[self._console]=file_hash(self._console)
        try:
            super().__init__(*args, **kwargs)
        finally:
            self._initializing=False

    def _compressed_handle(self, name):
        """Replace only our newly created empty placeholder, under log lock."""
        handle = self._logs[name]
        if isinstance(handle, StreamingGzipLog):
            return handle
        original = self.output_dir / name
        require(original.parent.resolve() == self.output_dir.resolve() and
                original.stat().st_size == 0 and self._log_sizes[name] == 0,
                'Cannot replace nonempty/unowned transport log')
        handle.close()
        compressed = StreamingGzipLog(original.with_name(name + '.gz'), max_raw_bytes=self.max_log_bytes)
        self._logs[name] = compressed
        original.unlink()
        return compressed

    def _log(self, name, raw):
        # Inherited _stdout_reader timestamps before entering _log and enqueues
        # afterwards. Leave that trusted clock and its ordering unchanged.
        with self._log_lock:
            try:
                available = self.max_log_bytes - self._log_sizes[name]
                handle = self._compressed_handle(name)
                if not handle.closed and available > 0:
                    part = raw[:available]
                    handle.write(part)
                    self._log_sizes[name] += len(part)
                if len(raw) > available:
                    self._failure = 'log_limit'
            except BaseException as error:
                # The inherited stderr reader swallows I/O exceptions. Record
                # an infrastructure failure before propagating so no response
                # or timeout forfeit can silently accept lost evidence.
                self._failure = 'log_compression_error'
                self._failure_detail = repr(error)
                self._compression_error = repr(error)
                raise ValueError('Transport log compression failed: ' + repr(error)) from error

    def _sample(self):
        if self._ps is None: return dict(self._last_telemetry)
        try: processes=[self._ps]+self._ps.children(recursive=True)
        except psutil.NoSuchProcess: return dict(self._last_telemetry)
        except psutil.AccessDenied as error:
            self._failure,self._failure_detail='process_tree_unobservable',str(error)
            return dict(self._last_telemetry)
        live=[]; rss=0; threads=0
        for proc in processes:
            try:
                identity=(proc.pid,proc.create_time()); exe=Path(proc.exe()).resolve(); ppid=proc.ppid()
                if exe not in self._verified:
                    require(exe in self._expected and file_hash(exe)==self._expected[exe], 'Unexpected descendant executable')
                    self._verified[exe]=self._expected[exe]
                role=('python' if proc.pid==self._ps.pid and exe==self._python else
                      self._natives.get(exe) if ppid==self._ps.pid else None)
                if exe==self._console:
                    owners=[r for r,i in self._owners.items() if i[0]==ppid and r in ('python','search','greedy')]
                    role='console:'+owners[0] if len(owners)==1 else None
                # A helper can be observed before its parent; classify in a second pass.
                memory=proc.memory_info().rss; cpu=proc.cpu_times(); count=proc.num_threads()
                self._tree_cpu[identity]=max(self._tree_cpu.get(identity,0),cpu.user+cpu.system)
                live.append(dict(pid=proc.pid,created=identity[1],parent_pid=ppid,role=role,
                    executable=str(exe),rss_bytes=memory,cpu_seconds=self._tree_cpu[identity],
                    affinity=proc.cpu_affinity(),observed_threads=count))
                rss+=memory; threads+=count
                if role is not None:
                    require(role not in self._owners or self._owners[role]==identity,'Computation/helper replacement')
                    self._owners[role]=identity
            except psutil.NoSuchProcess: continue
            except (ValueError,psutil.AccessDenied) as error:
                self._failure,self._failure_detail='process_tree_limit',str(error)
        for row in live:
            identity=(row['pid'],row['created'])
            if row['role'] is None and Path(row['executable'])==self._console:
                owners=[r for r,i in self._owners.items() if i[0]==row['parent_pid'] and r in ('python','search','greedy')]
                if len(owners)==1:
                    row['role']='console:'+owners[0]
                    if row['role'] in self._owners and self._owners[row['role']]!=identity:
                        self._failure,self._failure_detail='process_tree_limit','Console replacement'
                    self._owners[row['role']]=identity
            if row['role'] is None:
                self._failure,self._failure_detail='process_tree_limit','Unrecognized descendant'
            if row['affinity']!=self.affinity:
                if self._initializing and str(row['role']).startswith('console:'):
                    try:
                        psutil.Process(row['pid']).cpu_affinity(self.affinity)
                        row['startup_affinity_before']=row['affinity']; row['affinity']=psutil.Process(row['pid']).cpu_affinity()
                    except (psutil.NoSuchProcess,psutil.AccessDenied): pass
                if row['affinity']!=self.affinity:
                    self._failure,self._failure_detail='affinity_changed','Worker descendant changed assigned CPU'
        if len(self._owners)>6:
            self._failure,self._failure_detail='process_tree_limit','More than three compute processes plus OS helpers'
        self._peak_rss=max(self._peak_rss,rss)
        self._last_telemetry=dict(cpu_seconds=sum(self._tree_cpu.values()),rss_bytes=rss,
            peak_sampled_rss_bytes=self._peak_rss,affinity=self.affinity,observed_threads=threads,
            process_tree=live,resource_scope='sampled_worker_and_all_descendants')
        if rss>self.max_rss_bytes:
            self._failure,self._failure_detail='rss_limit','Combined worker/helper RSS exceeded 2GiB'
        return dict(self._last_telemetry)

    def _receive(self, *args, **kwargs):
        receipt = super()._receive(*args, **kwargs)
        if receipt['status'] != 'ok':
            # close() is invoked inside the inherited request() path. Keep its
            # original receipt as primary even if cleanup also encounters trouble.
            self._primary_failure = {key: receipt.get(key) for key in
                                     ('status', 'request_id', 'detail')}
        return receipt

    def close(self):
        if getattr(self, '_closed', False):
            return getattr(self, 'cleanup_receipt', None)
        self._closed = True
        started = time.monotonic()
        deadline = started + 5.0
        if getattr(self, 'absolute_deadline', None) is not None:
            deadline = min(deadline, self.absolute_deadline)
        remaining = lambda: max(0.0, deadline - time.monotonic())
        primary = getattr(self, '_primary_failure', None)
        if primary is None and sys.exc_info()[1] is not None:
            primary = {'status': 'exception', 'detail': repr(sys.exc_info()[1])}
        identities = set(getattr(self, '_tree_cpu', {}))
        errors, confirmed, alive = [], [], []
        root = getattr(self, '_ps', None)
        if root is not None:
            try:
                identities.update((p.pid, p.create_time()) for p in [root] + root.children(recursive=True))
            except psutil.NoSuchProcess:
                pass
            except psutil.AccessDenied as error:
                errors.append('descendant enumeration: ' + repr(error))
        owned = []
        for pid, created in sorted(identities):
            try:
                process = psutil.Process(pid)
                if process.create_time() == created:
                    owned.append((process, {'pid': pid, 'created': created}))
                else:
                    confirmed.append({'pid': pid, 'created': created, 'state': 'original_identity_absent'})
            except psutil.NoSuchProcess:
                confirmed.append({'pid': pid, 'created': created, 'state': 'absent'})
            except psutil.AccessDenied as error:
                errors.append('identity access: ' + repr(error))
                alive.append({'pid': pid, 'created': created, 'state': 'unconfirmed'})
        job = getattr(self, '_job', None)
        if job is not None:
            try:
                job.close()
            except BaseException as error:
                errors.append('job close: ' + repr(error))
        kill_errors = {}
        for process, identity in reversed(owned):
            try:
                process.kill()
            except psutil.NoSuchProcess:
                pass
            except psutil.AccessDenied as error:
                # Windows may deny TerminateProcess for an already exited
                # object. Its exit wait below, rather than this race, decides.
                kill_errors[(identity['pid'], identity['created'])] = repr(error)
        # Wait on actual process completion. Windows may still enumerate an
        # exited process while another open handle keeps its object visible;
        # is_running() is therefore not a valid post-kill completion test.
        for process, identity in owned:
            try:
                status = process.wait(timeout=remaining())
                confirmed.append({**identity, 'state': 'exit_wait_confirmed', 'exit_status': status})
            except psutil.NoSuchProcess:
                confirmed.append({**identity, 'state': 'absent'})
            except (psutil.TimeoutExpired, psutil.AccessDenied) as error:
                alive.append({**identity, 'state': 'unconfirmed', 'error': repr(error),
                    'kill_error': kill_errors.get((identity['pid'], identity['created']))})
        popen = getattr(self, 'process', None)
        if popen is not None:
            try:
                if popen.poll() is None:
                    popen.kill()
                popen.wait(timeout=remaining())
            except (OSError, subprocess.TimeoutExpired) as error:
                if popen.poll() is None:
                    errors.append('root handle wait: ' + repr(error))
        threads = getattr(self, '_threads', [])
        for thread in threads:
            thread.join(timeout=min(.25, remaining()))
        live_threads = [thread.name for thread in threads if thread.is_alive()]
        if live_threads:
            errors.append('transport threads unconfirmed: ' + repr(live_threads))
        elif popen is not None:
            for pipe in (popen.stdin, popen.stdout, popen.stderr):
                if pipe is not None:
                    try:
                        pipe.close()
                    except (OSError, ValueError) as error:
                        errors.append('pipe close: ' + repr(error))
        log_compression = {}
        if getattr(self, '_compression_error', None) is not None:
            errors.append('earlier log compression: ' + self._compression_error)
        lock = getattr(self, '_log_lock', None)
        locked = lock is None or lock.acquire(timeout=remaining())
        if locked:
            try:
                for name in getattr(self, '_logs', {}):
                    try:
                        handle = self._compressed_handle(name)
                        log_compression[name] = handle.close_and_verify(deadline=deadline)
                    except BaseException as error:
                        errors.append('log close/integrity ' + name + ': ' + repr(error))
            finally:
                if lock is not None:
                    lock.release()
        else:
            errors.append('log lock not released before cleanup deadline')
        self.cleanup_receipt = dict(time=now(), status='failed' if alive or errors else 'passed',
            method='bounded_process_exit_wait_compressed_logs_v3', observed_owned_identities=len(identities),
            confirmed=confirmed, alive=alive, errors=errors, primary_failure=primary,
            log_compression=log_compression,
            elapsed_seconds=time.monotonic()-started, maximum_seconds=5.0,
            effective_allowance_seconds=max(0.0, deadline-started),
            deadline_clamped_to_stage_and_global=True)
        if hasattr(self, 'output_dir'):
            path = self.output_dir / 'cleanup.json'
            if not path.exists():
                with path.open('x', encoding='utf-8') as stream:
                    json.dump(self.cleanup_receipt, stream, sort_keys=True, allow_nan=False)
                    stream.write('\n'); stream.flush(); os.fsync(stream.fileno())
        callback = getattr(self, 'study_after_close', None)
        if callback is not None:
            try:
                callback()
            except BaseException as error:
                self.cleanup_callback_error = repr(error)
                if hasattr(self, 'output_dir'):
                    with (self.output_dir / 'cleanup_callback_error.json').open('x', encoding='utf-8') as stream:
                        json.dump({'error': repr(error), 'primary_failure': primary}, stream)
                        stream.write('\n'); stream.flush(); os.fsync(stream.fileno())
                if primary is None:
                    raise
        if (alive or errors) and primary is None:
            raise RuntimeError('Owned worker cleanup failed; see cleanup.json')
        return self.cleanup_receipt

    def __exit__(self, exc_type, exc, traceback):
        receipt = self.close()
        # request() must return and preserve an original player-failure receipt
        # even when its immediate cleanup also fails. A normal context exit
        # (including a proposed timeout forfeit) is a separate acceptance gate:
        # the runner cannot record a completed game or start another worker
        # until owned processes and bookkeeping have both been confirmed.
        # An exception already propagating from qualification/referee work
        # remains primary; cleanup.json retains the secondary failure.
        if exc_type is None and (receipt is None or receipt.get('status') != 'passed' or
                                 getattr(self, 'cleanup_callback_error', None) is not None):
            raise RuntimeError('Owned worker cleanup acceptance failed; see cleanup.json')
        return False
