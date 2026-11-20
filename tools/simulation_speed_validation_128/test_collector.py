"""Small contract checks plus real hidden/parent-death subprocess fixtures."""
import copy
import json
import os
from pathlib import Path
import subprocess
import sys
import time

import psutil
import pytest

from . import collector as c


def sample_record():
    def outcome(turns):
        return {'reason': 'completed', 'winner': 0, 'turns': turns,
                'trace_sha256': str(turns), 'final_state_sha256': str(turns + 1)}
    return {'union': {'actions': [{}, {}, {}, {}]},
            'selection': {'worlds': [{'seed': i, 'state_sha256': str(i), 'outcomes': [outcome(j+1) for j in range(4)]} for i in range(128)]},
            'assessment': {'candidate_indices': [0, 3], 'worlds': [{'seed': 1000+i, 'state_sha256': str(i), 'outcomes': [outcome(10), outcome(20)]} for i in range(128)]}}


def test_fixed_schedule_and_sparse_assessment_wiring():
    record = sample_record()
    full = list(c.cases(record))
    checked = list(c.cases(record, True))
    assert len(full) == 128 * 4
    assert len(checked) == 8 * 4 + 2 * 2
    assessed = [row for row in checked if row[0] == 'assessment']
    assert [(row[1], row[2], row[4]['turns']) for row in assessed] == [(0,0,10),(0,3,20),(127,0,10),(127,3,20)]
    assert all(row[3]['seed'] >= 1000 for row in assessed)
    for workers in (1,2):
        rows = [r for r in c.SCHEDULE if r['workers'] == workers and r['backend'] != 'checked']
        assert len(rows) == 4
        assert [r['backend'] for r in rows[:2]] == list(reversed([r['backend'] for r in rows[2:]]))
    assert c.CONFIG['selection_continuations'] == 7040
    assert c.CONFIG['maximum_active_seconds'] == 14400


@pytest.mark.parametrize('change', ['short_worlds', 'short_outcomes', 'unknown_reason', 'bad_candidate', 'duplicate'])
def test_malformed_historical_record_fails(change):
    record = sample_record()
    if change == 'short_worlds': record['selection']['worlds'].pop()
    if change == 'short_outcomes': record['selection']['worlds'][0]['outcomes'].pop()
    if change == 'unknown_reason': record['selection']['worlds'][0]['outcomes'][0]['reason'] = 'unknown'
    if change == 'bad_candidate': record['assessment']['candidate_indices'] = [0, 4]
    if change == 'duplicate': record['assessment']['candidate_indices'] = [0, 0]
    with pytest.raises(ValueError): list(c.cases(record, True))


def test_resource_bounds_and_overlap():
    info = {'available_ram_bytes': 2*1024**3, 'commit_headroom_bytes': 8*1024**3,
            'process_rss_bytes': 25*1024**2, 'process_peak_rss_bytes': 30*1024**2}
    assert c.resource_ok(info, 8*1024**3)
    for key, value in [('available_ram_bytes',0), ('commit_headroom_bytes',0), ('process_peak_rss_bytes', c.CONFIG['maximum_parent_rss_bytes']+1)]:
        assert not c.resource_ok({**info, key:value}, 8*1024**3)
    assert not c.resource_ok(info, 8*1024**3, group_rss=c.CONFIG['maximum_group_rss_bytes']+1)
    assert not c.resource_ok(info, 0)
    assert c.interval_overlap([[1,5],[3,6],[7,10]]) == 2
    assert c.interval_overlap([[1,3],[3,5]]) == 0


def test_atomic_receipts_never_overwrite_and_incomplete_temps_are_not_completed(tmp_path, monkeypatch):
    path = tmp_path / 'result.json'
    c.save(path, {'a':1})
    with pytest.raises(FileExistsError): c.save(path, {'a':2})
    assert c.read(path) == {'a':1}
    assert not list(tmp_path.glob('*.pending_*'))
    attempt = tmp_path / 'attempts' / 'interrupted'
    attempt.mkdir(parents=True)
    (attempt / 'result.json.pending_partial').write_text('{')
    c.save(attempt / 'pulse.json', {'active_seconds':17})
    assert c.active_used(tmp_path) == 22
    assert c.completed_stages(tmp_path, {'binding':'test'}) == {}


def test_admission_heartbeats_five_healthy_samples_and_abort(tmp_path, monkeypatch):
    clock = [0.]
    monkeypatch.setattr(c.time, 'perf_counter', lambda: clock[0])
    monkeypatch.setattr(c.time, 'sleep', lambda seconds: clock.__setitem__(0,clock[0]+seconds))
    infos = [100, 800, 800, 700, 800, 800, 800, 800, 800]
    def guarded(*args, **kwargs):
        assert kwargs in ({'waiting':True}, {})
        return {'available_ram_bytes': (infos.pop(0) if infos else 800)*1024**2, 'commit_headroom_bytes':8*1024**3}
    monkeypatch.setattr(c, 'operational_guard', guarded)
    pulses, history = [], []
    c.admit(tmp_path,1,60,history,lambda:pulses.append(clock[0]))
    assert len(history) == len(pulses) == 9
    assert clock[0] == 8


def test_own_worker_detection_is_exact_and_resumption_rejects_live_receipt(tmp_path):
    own = [sys.executable,'-m','tools.simulation_speed_validation_128.collector','worker','--job',str(tmp_path/'attempts'/'x'/'root_1_job.json')]
    assert c.command_is_own_worker(own,tmp_path)
    assert not c.command_is_own_worker(own,tmp_path/'other')
    assert not c.command_is_own_worker([*own[:3], 'diagnostic', *own[4:]],tmp_path)
    path = tmp_path/'attempts'/'x'/'root_1_process.json'
    c.save(path,c.process_identity(os.getpid()))
    with pytest.raises(RuntimeError, match='still alive'): c.require_no_prior_workers(tmp_path)


def test_hidden_worker_actual_launch_and_thread_environment(tmp_path):
    job = tmp_path/'diagnostic.json'
    ready = tmp_path/'ready.json'
    c.save(job,{'ready':str(ready)})
    options = c.process_options()
    assert options['creationflags'] & subprocess.CREATE_NO_WINDOW
    assert options['creationflags'] & subprocess.BELOW_NORMAL_PRIORITY_CLASS
    result = subprocess.run([sys.executable,'-m','tools.simulation_speed_validation_128.collector','diagnostic','--job',str(job)],
                            capture_output=True,text=True,timeout=20,**options)
    assert result.returncode == 0, result.stderr
    observed = c.read(ready)
    assert observed['console_window'] == 0
    assert observed['priority'] == int(psutil.BELOW_NORMAL_PRIORITY_CLASS)
    assert set(observed['threads'].values()) == {'1'}
    assert not c.identity_alive(observed['identity'])


def test_worker_lease_expires_after_actual_abrupt_controller_exit(tmp_path):
    job, ready, final = [tmp_path/name for name in ('job.json','ready.json','lease_result.json')]
    script = '''
import os,sys,time,subprocess
from pathlib import Path
from tools.simulation_speed_validation_128 import collector as c
job,ready,final=map(Path,sys.argv[1:])
c.save(job,{'ready':str(ready),'result':str(final),'parent':c.process_identity(os.getpid())})
p=subprocess.Popen([sys.executable,'-m','tools.simulation_speed_validation_128.collector','diagnostic','--job',str(job)],**c.process_options())
start=time.monotonic()
while not ready.exists() and time.monotonic()-start<10: time.sleep(.05)
os._exit(0 if ready.exists() else 2)
'''
    result = subprocess.run([sys.executable,'-c',script,str(job),str(ready),str(final)],
                            capture_output=True,text=True,timeout=20,**c.process_options())
    assert result.returncode == 0, result.stderr
    start = time.monotonic()
    while not final.exists() and time.monotonic()-start < 10: time.sleep(.05)
    assert c.read(final)['lease_expired'] is True
    identity = c.read(ready)['identity']
    start = time.monotonic()
    while c.identity_alive(identity) and time.monotonic()-start < 5: time.sleep(.05)
    assert not c.identity_alive(identity)


def test_corrupt_complete_stage_cannot_be_reused(tmp_path):
    path = tmp_path/'attempts'/'test'/'result.json'
    c.save(path,{'binding':'test','exact':True,'stage':c.SCHEDULE[0], 'roots':[], 'receipt_sha256':'bad'})
    with pytest.raises(ValueError,match='receipt changed'): c.completed_stages(tmp_path,{'binding':'test'})
