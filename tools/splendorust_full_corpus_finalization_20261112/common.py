"""Immutable inputs, separate outputs and inherited clocks for finalization."""
from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path
import shutil
import sys
import time
import psutil
from tools.splendorust_full_corpus_recovery_20261111 import common as prior

ROOT = prior.ROOT
SOURCE = prior.OUT
ORIGINAL = prior.ORIGINAL_OUT
OUT = ROOT / 'outputs/splendorust_full_corpus_finalization_20261112_01'
PACKAGE = 'tools.splendorust_full_corpus_finalization_20261112'
PROTOCOL = ROOT / 'docs/engineering/splendorust_full_corpus_finalization_20261112.md'
PLAN = OUT / 'freeze.json'
DETACHED_ACCEPTANCE = OUT / 'prelaunch/detached_launcher_acceptance02.json'
SOURCE_SHA = 'fa7ce53819e73c57a3de2cdd8d1c0330c00e411bd9f6f0e159c28da2798659c5'
JOBS = ('independent_mixed_terminal_r1', 'reader_mixed_terminal_r0', 'final_acceptance')
JOB_CAPS = dict(zip(JOBS, (90 * 60, 90 * 60, 120 * 60)))
TOTAL_SECONDS = 4 * 3600
GiB = 1024 ** 3
MAX_RSS = 8 * GiB
MAX_NEW_BYTES = 2 * GiB
MAX_COMBINED_BYTES = prior.MAX_EVIDENCE_BYTES
MIN_FREE_BYTES = prior.MINIMUM_FREE_DISK_BYTES
SOURCE_IDENTITIES = ({'pid': 62252, 'created': 1794362749.6136715},
                     {'pid': 39812, 'created': 1794501812.5410624})
read, artifact, verify, require = prior.read, prior.artifact, prior.verify, prior.require
canonical, digest, file_hash, resolve = prior.canonical, prior.digest, prior.file_hash, prior.resolve
now = prior.now


def save(path, value):
    path = Path(path).resolve()
    require(path.is_relative_to(OUT), 'Finalization writes only its separate output')
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open('x', encoding='utf-8', newline='\n') as stream:
        stream.write(canonical(value) + '\n'); stream.flush(); os.fsync(stream.fileno())
    return artifact(path)


def status(value):
    path = OUT / 'stage.json'; temporary = OUT / 'stage.pending'
    with temporary.open('w', encoding='utf-8', newline='\n') as stream:
        stream.write(canonical(value) + '\n'); stream.flush(); os.fsync(stream.fileno())
    os.replace(temporary, path)


def identity():
    return dict(pid=os.getpid(), created=psutil.Process().create_time())


def assert_stopped():
    from tools.splendorust_gpu_probe_20261105.control import require_absent
    for item in SOURCE_IDENTITIES: require_absent(item)
    require(not any((SOURCE / name).exists() for name in
                    ('execution.json', 'completion.json', 'failure.json')),
            'Stopped source remains incomplete with unknown actual root exit')
    require(not (SOURCE / 'segments/independent_mixed_terminal_r1/result.json').exists()
            and not (SOURCE / 'segments/independent_mixed_terminal_r1/execution.json').exists(),
            'Only the exact unfinished independent check may be repeated')
    require(not (SOURCE / 'segments/reader_mixed_terminal_r0').exists()
            and not (SOURCE / 'final_audit').exists(), 'Remaining final stages were never started')


def original_limits():
    reader = read(SOURCE / 'reader_clock.json')
    phase = read(SOURCE / 'stages/final_start.json')
    claim = read(SOURCE / 'claim.json')
    return dict(monotonic_origin=max(reader['monotonic'], phase['monotonic'], claim['monotonic_origin']),
                unix_origin=max(reader['unix_time'], phase['unix_time'], claim['unix_origin']),
                monotonic_deadline=min(reader['monotonic_deadline'], phase['monotonic_deadline'],
                                      claim['monotonic_origin'] + claim['total_seconds']),
                unix_deadline=min(reader['unix_deadline'], phase['unix_deadline'],
                                  claim['unix_origin'] + claim['total_seconds']))


def check_deadlines(limits, *, mono=None, wall=None):
    mono = time.monotonic() if mono is None else mono
    wall = time.time() if wall is None else wall
    require(limits.get('monotonic_origin', 0) <= mono < limits['monotonic_deadline']
            and limits.get('unix_origin', 0) <= wall < limits['unix_deadline'],
            'Original reader/final/global dual deadlines remain in force')


def contract():
    return dict(schema='full_corpus_saved_finalization_v1', jobs=list(JOBS),
        job_caps=JOB_CAPS, total_seconds=TOTAL_SECONDS, max_rss_bytes=MAX_RSS,
        max_new_bytes=MAX_NEW_BYTES, max_combined_bytes=MAX_COMBINED_BYTES,
        minimum_free_disk_bytes=MIN_FREE_BYTES, new_games=0, model_calls=0,
        optimizer_calls=0, new_decision_retries=0, automatic_job_retries=0,
        source_actual_owner_exit='unknown', source_attempt_status='incomplete',
        original_attempt_status='incomplete', original_deadlines=original_limits(),
        source_freeze_sha256=SOURCE_SHA, available_ram_gate=False,
        forecast_refusal=False, original_model_remains_reference=True)


def validate_plan(plan):
    require(plan['contract'] == contract(), 'Exact saved-only finalization contract')
    require(plan['source_freeze'] == artifact(SOURCE / 'freeze.json')
            and plan['source_freeze']['sha256'] == SOURCE_SHA, 'Exact frozen interrupted recovery')
    require(plan['source_inventory']['path'] == (OUT / 'source_inventory.jsonl').relative_to(ROOT).as_posix(),
            'Fixed full source preservation inventory')
    check_deadlines(plan['contract']['original_deadlines'])


def load_plan(expected, *, verify_inputs=True):
    require(file_hash(PLAN) == expected, 'Exact separate finalization freeze')
    plan = read(PLAN); validate_plan(plan)
    if verify_inputs: verify(plan['inputs'], boundary=lambda: check_deadlines(original_limits()))
    return plan


def verify_source_inventory(plan, boundary=lambda: None):
    descriptor = plan['source_inventory']; verify([descriptor], boundary=boundary)
    seen = set(); total = 0
    with resolve(descriptor).open(encoding='utf-8') as stream:
        for line in stream:
            boundary(); item = json.loads(line); path = resolve(item)
            require(path.is_relative_to(SOURCE) and path not in seen,
                    'Unique source-only preservation paths')
            before = path.stat(); verify([item], boundary=boundary); after = path.stat()
            require((before.st_size, before.st_mtime_ns) == (after.st_size, after.st_mtime_ns),
                    'Source evidence stable while verified')
            seen.add(path); total += item['bytes']
    actual = set()
    for base, dirs, files in os.walk(SOURCE):
        boundary()
        for name in files: actual.add((Path(base) / name).resolve())
    require(actual == seen, 'Entire source inventory preserved, without additions or removal')
    return dict(status='passed', files=len(seen), bytes=total, inventory=descriptor)


def output_bytes(folder):
    return sum(p.stat().st_size for p in Path(folder).rglob('*') if p.is_file())


class Boundary:
    def __init__(self, claim, *, worker_owner=None):
        self.claim = claim; self.worker_owner = worker_owner; self.calls = 0
        self.limits = {key: min(claim[key], original_limits()[key]) for key in
                       ('monotonic_deadline', 'unix_deadline')}
        self.limits.update({key: max(claim[key], original_limits()[key]) for key in
                            ('monotonic_origin', 'unix_origin')})
        if worker_owner:
            self.limits = {key: (min(value, worker_owner[key]) if key.endswith('deadline')
                                else max(value, worker_owner[key])) for key, value in self.limits.items()}
        self._last_resource = -float('inf')

    def __call__(self):
        self.calls += 1; check_deadlines(self.limits)
        require('torch' not in sys.modules, 'No model runtime in saved-only finalization')
        if time.monotonic() - self._last_resource >= 5:
            self._last_resource = time.monotonic()
            require(shutil.disk_usage(ROOT).free >= MIN_FREE_BYTES, 'Actual unchanged disk floor')
            assert_stopped()
            if self.worker_owner:
                process = psutil.Process(self.claim['identity']['pid'])
                require(process.create_time() == self.claim['identity']['created'] and process.is_running(),
                        'Exact new audit owner alive')
    guard = __call__


def validate_job_name(name):
    require(name in JOBS, 'Only three fixed saved-evidence jobs are authorized')
    return name
