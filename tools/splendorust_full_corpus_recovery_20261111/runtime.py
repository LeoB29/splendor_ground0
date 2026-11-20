"""Owned-worker handshake and original clocks, with no resource forecast gates."""
import os
from pathlib import Path
import shutil
import time
import threading
import psutil
from . import common as c


def check_deadlines(claim, phase, owner, mono, wall, model=True):
    c.require(claim['monotonic_origin'] <= mono < claim['monotonic_origin']+c.TOTAL_SECONDS
              and claim['unix_origin'] <= wall < claim['unix_origin']+c.TOTAL_SECONDS,
              'Original overall dual clock')
    if model:
        c.require(mono < claim['monotonic_origin']+c.MODEL_SECONDS
                  and wall < claim['unix_origin']+c.MODEL_SECONDS, 'Original model cutoff')
    c.require(phase['monotonic'] <= mono < phase['monotonic_deadline']
              and phase['unix_time'] <= wall < phase['unix_deadline'], 'Original phase dual clock')
    c.require(owner['monotonic_origin'] <= mono < owner['monotonic_deadline']
              and owner['unix_origin'] <= wall < owner['unix_deadline'], 'Original worker dual clock')


class Boundary:
    def __init__(self, claim, phase, owner, folder, model):
        self.claim, self.phase, self.owner, self.folder, self.model = claim, phase, owner, folder, model
        self.calls, self.last_deep = 0, -float('inf')
        self._deep_lock = threading.Lock()

    def __call__(self):
        self.calls += 1
        mono, wall = time.monotonic(), time.time()
        check_deadlines(self.claim, self.phase, self.owner, mono, wall, self.model)
        if mono-self.last_deep >= 5 and self._deep_lock.acquire(blocking=False):
            try:
                if mono-self.last_deep < 5: return
                self.last_deep = mono
                parent = psutil.Process(self.claim['pid'])
                c.require(parent.create_time() == self.claim['created'] and parent.is_running(),
                          'Exact original owner remains alive')
                c.require(shutil.disk_usage(c.OUT).free >= c.MIN_DISK, 'Actual free disk floor')
                c.require(psutil.Process().memory_info().rss <= self.owner['maximum_rss_bytes'],
                          'Actual worker RSS cap')
                c.status(self.folder/'boundary/progress.json', dict(time=c.now(), mode=self.owner['mode'],
                    fit_id=self.owner.get('fit_id'), boundary_calls=self.calls,
                    elapsed_seconds=mono-self.owner['monotonic_origin'], model_work=self.model))
            finally:
                self._deep_lock.release()


def bootstrap(folder, expected, mode, model=True, fit_id=None):
    folder = Path(folder).resolve()
    c.require(folder.is_relative_to(c.OUT) and folder.name == 'worker', 'Fresh owned worker folder')
    c.require(c.sha(c.PLAN) == expected, 'Exact frozen study')
    plan, claim = c.read(c.PLAN), c.read(c.OUT/'claim.json')
    c.require(claim['freeze'] == c.artifact(c.PLAN), 'Original owner binds exact freeze')
    waited = time.monotonic()
    owner_path = folder.parent/'owner.json'
    while not owner_path.exists():
        c.require(time.monotonic()-waited < 20, 'Bounded owning-job handshake')
        time.sleep(.01)
    owner = c.read(owner_path)
    me = dict(pid=os.getpid(), created=psutil.Process().create_time())
    c.require(owner['identity'] == me and owner['windows_job_assigned'] is True
              and owner['mode'] == mode and owner.get('fit_id') == fit_id
              and owner['plan_sha256'] == expected
              and owner['global_claim'] == c.artifact(c.OUT/'claim.json'), 'Exact owned worker role')
    c.verify([owner['phase_clock']])
    phase = c.read(c.resolve(owner['phase_clock']))
    c.require(phase['claim'] == c.artifact(c.OUT/'claim.json'), 'Phase inherits original owner')
    phases={'qualification':'qualification','audit_qualification':'qualification','games':'collection',
            'audit_cell':'collection','corpus':'corpus','selection':'qualification','cache':'corpus','cache_audit':'corpus','fit':'fit','commit':'fit',
            'offline':'offline','audit':'final','analysis':'final','final':'final'}
    c.require(mode in phases and phase['stage']==phases[mode], 'Worker role in original fixed phase')
    cap=c.STAGES[phase['stage']]
    cutoff=c.TOTAL_SECONDS if phase['stage']=='final' else c.MODEL_SECONDS
    c.require(type(phase['cap_seconds']) is int and phase['cap_seconds']==cap
              and phase['monotonic_deadline']==min(phase['monotonic']+cap,claim['monotonic_origin']+cutoff)
              and phase['unix_deadline']==min(phase['unix_time']+cap,claim['unix_origin']+cutoff),
              'Unextended original phase deadlines')
    c.require(owner['monotonic_deadline']<=min(owner['monotonic_origin']+owner['maximum_seconds'],phase['monotonic_deadline'])
              and owner['unix_deadline']<=min(owner['unix_origin']+owner['maximum_seconds'],phase['unix_deadline']),
              'Worker allowance is clipped to original phase')
    boundary = Boundary(claim, phase, owner, folder, model)
    boundary()
    c.verify(plan['worker_inputs'], boundary=boundary)
    from tools.future_training_rules_20261021 import verify_runtime_sources
    sources = verify_runtime_sources()  # Exactly once in this worker's startup.
    c.require(sources == plan['corrected_source_receipt'], 'Frozen corrected runtime sources')
    c.save(folder/'runtime_sources.json', dict(sources=sources, mode=mode, fit_id=fit_id,
        original_claim=owner['global_claim'], original_phase=owner['phase_clock'], identity=me))
    return plan, boundary, sources
