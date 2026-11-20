"""Qualified resource accounting with immutable inherited phase clocks.

The base accounting class is a source copy of the accepted policy-head envelope.
Only the separate SegmentEnvelope binds a predetermined owner phase instead of
starting/resetting phases. No model execution is implemented here.
"""
import shutil
import time
from pathlib import Path
import psutil
from .. import common as c


def bytes_used():
    return sum(p.stat().st_size for p in c.OUT.rglob('*') if p.is_file())


class _AccountingEnvelope:
    """One non-resettable job clock, aggregate resident-worker resource bound."""
    def __init__(self, claim):
        self.start, self.wall_start = claim['monotonic_origin'], claim['unix_origin']
        self.end = self.start + c.JOB_SECONDS
        self.stage = None
        self.deadline, self.wall_deadline = self.end, self.wall_start + c.JOB_SECONDS
        self.waited, self.admitted, self.admissions = 0., False, 0
        self.worker_accounting = {}
        self.prior_worker_count = 0
        self.prior_qualified_source_workers = 0
        self.evidence = None
        self.resident = {}
        self.peak_combined_rss = 0
        self.last_resource_guard = -float("inf")
        self.continuation = None  # No old clock, evidence or worker accounting is inherited.

    def enable_evidence_accounting(self):
        self.evidence = dict(baseline=bytes_used(), parent_bytes=0, workers={})
        self.watchdog_control_baseline = self.watchdog_control_bytes()
        self.log_baseline = {name: (c.OUT/name).stat().st_size if (c.OUT/name).exists() else 0
                             for name in ('runner_stdout.log', 'runner_stderr.log')}
        c.WRITE_OBSERVER = self.note_write

    def watchdog_control_bytes(self):
        # One frozen placement exchange plus fixed owner publications. Reserve
        # their declared capacity rather than stat a pending publication live.
        return 0

    def note_write(self, path, size):
        resolved=path.resolve()
        if not any(resolved.is_relative_to(folder.resolve()) for folder in self.worker_accounting):
            self.evidence['parent_bytes'] += size

    def register_worker(self, folder):
        from .accounting import WorkerAccounting
        c.require(self.prior_worker_count+len(self.worker_accounting)<c.MAX_WORKER_LIFECYCLES,
                  'Fresh study worker lifecycle bound')
        self.worker_accounting[folder] = WorkerAccounting(folder)
        self.evidence['workers'][folder] = None

    def own_worker(self, child):
        proc = psutil.Process(child.process.pid)
        identity = (proc.pid, proc.create_time())
        c.require(identity not in self.resident, 'Duplicate resident worker identity')
        c.require(len(self.resident)<2,'At most two resident worker trees')
        self.resident[identity] = proc

    def close_worker(self, folder, child=None):
        if child is not None:
            identities = [identity for identity in self.resident if identity[0] == child.process.pid]
            for identity in identities:
                self.resident.pop(identity)
        if self.worker_accounting[folder] is None:
            c.require(child is None and self.evidence['workers'][folder] is not None,
                'Only a fully verified closed lifecycle may use its compact tombstone')
            return
        receipt = self.worker_accounting[folder].close(boundary=self.time_guard,
            lock_timeout=min(1.,max(0.,self.remaining())))
        self.evidence['workers'][folder] = receipt['accounted_bytes']

    def release_verified_worker(self,folder):
        c.require(self.evidence['workers'][folder] is not None and
            (folder.parent.parent/'closed_workers'/f'{folder.name}.json').is_file(),
            'Release accounting memory only after complete verified owned closure')
        c.require(not any(p.pid==self.worker_accounting[folder]._child.process.pid for p in self.resident.values()),
            'Cannot release a resident worker')
        self.worker_accounting[folder]=None

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
        c.require(total <= 2*c.GiB, 'Combined inference workers/helpers RSS exceeded 2GiB')
        return total

    def guard_worker_sample(self, child, telemetry):
        """Include a new tree before its constructor has returned readiness."""
        c.require(len([i for i in self.resident if i[0]!=child.process.pid])<2,'Constructing a third resident worker is forbidden')
        self.worker_accounting[child.output_dir].bind_child(child)
        self.aggregate_worker_rss(exclude_pid=child.process.pid, additional_rss=telemetry['rss_bytes'])

    def accounted_bytes(self):
        if self.evidence is None:
            return bytes_used()
        total = self.evidence['baseline'] + self.evidence['parent_bytes']
        for folder, closed in self.evidence['workers'].items():
            total += closed if closed is not None else self.worker_accounting[folder].snapshot(
                boundary=self.time_guard,lock_timeout=min(1.,max(0.,self.remaining())))['accounted_bytes']
        for name, initial in self.log_baseline.items():
            path = c.OUT/name
            total += max(0, (path.stat().st_size if path.exists() else 0)-initial)
        fit_folder=c.OUT/'fit'
        fit_bytes=sum(p.stat().st_size for p in fit_folder.rglob('*') if p.is_file()) if fit_folder.exists() else 0
        return total + fit_bytes + 32*c.MiB + 65536

    def time_guard(self):
        c.require(self.remaining()>0,'Shared clock while accounting evidence')

    def begin(self, stage):
        if stage=='final':c.require(not self.resident,'Saved-only final admission requires no resident model trees')
        mono, wall = time.monotonic(), time.time()
        self.stage = stage
        cap = c.STAGES[stage]
        cutoff = c.JOB_SECONDS if stage == 'final' else c.COLLECTION_CUTOFF_SECONDS
        self.deadline = min(mono+cap, self.start+cutoff, self.end)
        self.wall_deadline = min(wall+cap, self.wall_start+cutoff, self.wall_start+c.JOB_SECONDS)
        self.guard()
        c.atomic_save(c.OUT/'stages'/f'{stage}_start.json', dict(time=c.now(), monotonic=mono, unix_time=wall,
            monotonic_deadline=self.deadline, unix_deadline=self.wall_deadline, cap_seconds=cap))

    def remaining(self):
        return min(self.deadline-time.monotonic(), self.wall_deadline-time.time(),
                   self.end-time.monotonic(), self.wall_start+c.JOB_SECONDS-time.time())

    def effective_deadline(self):
        return time.monotonic()+self.remaining()

    def guard(self):
        c.require(self.remaining() > 0, 'Stage/global real-clock deadline')
        current=time.monotonic()
        if current-self.last_resource_guard<1.:return
        self.last_resource_guard=current
        c.require(shutil.disk_usage(c.ROOT).free >= 5*c.GiB, 'Free disk reserve')
        cap = c.MAX_EVIDENCE_BYTES if self.stage == 'final' else c.PREFINAL_EVIDENCE_BYTES
        c.require(self.accounted_bytes() < cap, 'Study evidence ceiling')
        self.aggregate_worker_rss()

    def timeout(self, maximum):
        self.guard()
        return min(maximum, self.remaining())

    def admit(self):
        self.guard()
        if not self.admitted:
            c.require(shutil.disk_usage(c.ROOT).free >= c.MINIMUM_FREE_DISK_BYTES, 'Actual free-disk floor')
            c.save(c.OUT/'memory_admissions'/'000001.json', dict(time=c.now(),
                observed_available_bytes=psutil.virtual_memory().available,
                ram_reserve_policy='advisory_only', wait_seconds=0., cumulative_wait_seconds=0., stable_seconds=0.))
        self.admitted=True
        self.admissions+=1

    def finish(self):
        self.last_resource_guard=-float("inf")
        self.guard()
        if self.evidence is not None:
            c.require(bytes_used() <= self.accounted_bytes(), 'Unaccounted study evidence')
        c.atomic_save(c.OUT/'stages'/f'{self.stage}_end.json', dict(time=c.now(), monotonic=time.monotonic(), unix_time=time.time()))


class SegmentEnvelope(_AccountingEnvelope):
    def __init__(self, claim_descriptor, phase_descriptor, segment_descriptor):
        c.verify([claim_descriptor, phase_descriptor, segment_descriptor])
        claim, phase, segment = (c.read(c.ROOT / x['path']) for x in
                                 (claim_descriptor, phase_descriptor, segment_descriptor))
        c.require(segment['claim'] == claim_descriptor and segment['phase_clock'] == phase_descriptor,
                  'Exact original claim and phase binding')
        super().__init__(claim)
        self.claim = claim_descriptor
        self.phase_receipt = phase_descriptor
        self.segment_claim = segment_descriptor
        self.segment_id = segment['segment_id']
        self.stage = phase['stage']
        c.require(self.stage in ('strength', 'strength_replay', 'final'), 'Fixed root phase')
        expected_phase = {'qualification':'strength', 'operational':'strength',
                          'main_block':'strength', 'replay':'strength_replay', 'saved':'final',
                          'independent':'final', 'aggregate':'final', 'reader':'final'}
        c.require(segment['mode'] in expected_phase and expected_phase[segment['mode']] == self.stage,
                  'Segment role binds its original phase and cutoff')
        c.require(phase['claim'] == claim_descriptor, 'Phase belongs to original owner')
        cap = c.STAGES[self.stage]
        cutoff = c.JOB_SECONDS if self.stage == 'final' else c.MODEL_CUTOFF_SECONDS
        c.require(type(phase['cap_seconds']) is int and phase['cap_seconds'] == cap and
                  phase['monotonic_deadline'] == min(phase['monotonic'] + cap, self.start + cutoff, self.end) and
                  phase['unix_deadline'] == min(phase['unix_time'] + cap, self.wall_start + cutoff,
                                                self.wall_start + c.JOB_SECONDS), 'Unreset original phase clocks')
        self.deadline, self.wall_deadline = phase['monotonic_deadline'], phase['unix_deadline']
        if segment.get('reader_clock') is not None:
            c.require(self.stage == 'final', 'Reader subclock only inside saved finalization')
            descriptor = segment['reader_clock']
            c.verify([descriptor])
            c.require(descriptor == c.artifact(c.OUT/'reader_clock.json'), 'Exact original reader clock')
            reader = c.read(c.OUT/'reader_clock.json')
            c.require(reader['claim'] == claim_descriptor and reader['original_phase'] == phase_descriptor and
                      type(reader['cap_seconds']) is int and reader['cap_seconds'] == c.INDEPENDENT_REVIEW_SECONDS and
                      phase['monotonic'] <= reader['monotonic'] and phase['unix_time'] <= reader['unix_time'] and
                      reader['monotonic_deadline'] == min(reader['monotonic']+c.INDEPENDENT_REVIEW_SECONDS,
                                                          phase['monotonic_deadline']) and
                      reader['unix_deadline'] == min(reader['unix_time']+c.INDEPENDENT_REVIEW_SECONDS,
                                                     phase['unix_deadline']), 'Unreset clipped independent reader clock')
            self.deadline = min(self.deadline, reader['monotonic_deadline'])
            self.wall_deadline = min(self.wall_deadline, reader['unix_deadline'])
        self.prior_worker_count = sum(1 for path in c.OUT.glob('strength/r*/workers/*') if path.is_dir())
        self.enable_evidence_accounting()
        self.guard()

    def begin(self, stage):
        raise ValueError('Only the original root owner creates phase clocks')

    def close_worker(self, folder, child=None):
        folder = Path(folder).resolve()
        if folder in self.worker_accounting:
            return super().close_worker(folder, child)
        c.require(self.stage == 'final' and child is None, 'Only saved-only readers inspect historical closures')
        c.require(folder.parent.name == 'workers' and folder.parent.parent.name in ('r0', 'r1') and
                  folder.parent.parent.parent == c.OUT / 'strength', 'Exact current-study worker path')
        saved = c.read(folder.parent.parent/'closed_workers'/(folder.name+'.json'))
        cleanup = c.read(folder/'cleanup.json')
        c.require(saved['status'] == 'passed' and saved['worker'] == folder.name and
                  not cleanup['alive'] and not cleanup['errors'] and
                  cleanup['deadline_clamped_to_stage_and_global'] is True, 'Saved closure must be complete')
        for item in cleanup['confirmed']:
            try: process = psutil.Process(item['pid'])
            except psutil.NoSuchProcess: continue
            c.require(process.create_time() != item['created'], 'Historical owned worker still alive')
        # Baseline bytes already include this tree. The caller independently
        # rechecks compressed raw bytes, responses, sidecars and the full closure.
        return None

    def admit(self):
        self.guard()
        if not self.admitted:
            c.save(c.OUT / 'memory_admissions' / (self.segment_id + '.json'),
                   dict(time=c.now(), observed_available_bytes=psutil.virtual_memory().available,
                        ram_reserve_policy='advisory_only', wait_seconds=0., cumulative_wait_seconds=0.,
                        stable_seconds=0., claim=self.claim, phase_clock=self.phase_receipt))
        self.admitted = True
        self.admissions += 1

    def finish(self):
        c.require(not self.resident, 'All segment workers closed before completion')
        self.last_resource_guard = -float('inf')
        self.guard()
        actual = bytes_used()
        c.require(actual <= self.accounted_bytes(), 'Unaccounted segment evidence')
        return dict(actual_evidence_bytes=actual, accounted_bytes=self.accounted_bytes(),
                    peak_combined_inference_rss=self.peak_combined_rss,
                    new_worker_lifecycles=len(self.worker_accounting),
                    prior_worker_lifecycles=self.prior_worker_count,
                    original_claim=self.claim, original_phase=self.phase_receipt)
