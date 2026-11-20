"""Real saved-byte writes and Envelope accounting, with no learned execution."""
import json
import os
from pathlib import Path
import shutil
import subprocess
import time
from types import SimpleNamespace

import pytest

from tools.splendorust_minimal_logging_20261028 import common as c,runner,design,runtime_support as support
from tools.splendorust_minimal_logging_20261028 import logging_probe,analysis

ROOT=Path(__file__).resolve().parents[1]
PRIOR=ROOT/'outputs/splendorust_startup_logging_20261027_01'


@pytest.fixture(autouse=True)
def forbid_models(monkeypatch):
    def forbidden(*args,**kwargs):raise AssertionError('No process/model may run in accounting regression')
    monkeypatch.setattr(subprocess,'Popen',forbidden)
    monkeypatch.setattr(runner,'make_worker',forbidden)


@pytest.fixture
def scenario(tmp_path,monkeypatch):
    out=tmp_path/'trial';out.mkdir()
    monkeypatch.setattr(c,'ROOT',tmp_path);monkeypatch.setattr(c,'OUT',out);monkeypatch.setattr(c,'WRITE_OBSERVER',None)
    monkeypatch.setattr(runner.shutil,'disk_usage',lambda path:SimpleNamespace(free=100*c.GiB))
    baseline=tmp_path/'baseline';failure=tmp_path/'failure'
    monkeypatch.setattr(design,'BASELINE',baseline)
    # The explicit immutable CPU-failure source is separate from the newly
    # preserved logging attempt, which contains no workers.
    monkeypatch.setattr(design,'SCIENTIFIC_FAILURE',failure)
    events=[];positions=[]
    for index,case in enumerate(('private_reserves','gold_returns')):
        row=json.loads((PRIOR/'saved_replays'/f'{index:04d}.jsonl').read_bytes())
        event=row['event'];events.append(event);positions.append(dict(case_id=case,observer=event['observer']))
    for path in (baseline/'workers/qualification_frozen_cpu0/entrypoint_forensics.jsonl',
                 failure/'workers/qualification_frozen_cpu2/entrypoint_forensics.jsonl'):
        path.parent.mkdir(parents=True);path.write_text('\n'.join(json.dumps({'event':e}) for e in events)+'\n')
    envelope=runner.Envelope(dict(monotonic_origin=time.monotonic(),unix_origin=time.time()))
    envelope.enable_evidence_accounting();envelope.begin('saved_replay')
    initial_parent_bytes=envelope.evidence['parent_bytes']
    calls=[];recorders={};actual_factory=logging_probe.make_recorder
    def factory(path,enabled):
        recorder=actual_factory(path,enabled);recorders[path]=recorder;return recorder
    monkeypatch.setattr(logging_probe,'make_recorder',factory)
    note=envelope.note_write
    def observe(path,size):
        if path.suffix=='.jsonl':
            assert recorders[path].closed,'Raw bytes must be registered after close, outside the timer'
            assert size==path.stat().st_size
        calls.append((path,size));note(path,size)
    monkeypatch.setattr(c,'WRITE_OBSERVER',observe)
    # Only gate acceptance is deterministic here. Real Recorder and filesystem
    # writes, every byte counter, guard, stage finish and transition stay real.
    monkeypatch.setattr(analysis,'replay_overhead_gate',lambda rows:dict(passed=True,attempted=len(rows),test_only_gate=True))
    return SimpleNamespace(out=out,envelope=envelope,panel=dict(positions=positions),calls=calls,recorders=recorders,
                           initial_parent_bytes=initial_parent_bytes)


def test_full64_replay_finishes_and_enters_qualification_before_model_sentinel(scenario):
    s=scenario;gate=support.run_saved_replay(s.envelope,s.panel)
    raw=sorted((s.out/'saved_replays').glob('*.jsonl'))
    assert gate['passed'] is True and len(raw)==64
    assert sum(p.stat().st_size for p in raw)==1845824
    registered=[(p,n) for p,n in s.calls if p.suffix=='.jsonl']
    assert len(registered)==len({p for p,n in registered})==64
    assert {p for p,n in registered}==set(raw)
    assert s.envelope.evidence['parent_bytes']==s.initial_parent_bytes+sum(n for p,n in s.calls)
    assert runner.bytes_used()<=s.envelope.accounted_bytes()
    s.envelope.finish();s.envelope.begin('qualification')
    assert (s.out/'stages/saved_replay_end.json').is_file()
    assert (s.out/'stages/qualification_start.json').is_file()
    assert not s.envelope.resident and not (s.out/'workers').exists()


def test_exact_old_raw_gap_still_rejected_if_not_registered(scenario):
    s=scenario;folder=s.out/'saved_replays';folder.mkdir()
    paths=[]
    for source in sorted((PRIOR/'saved_replays').glob('*.jsonl')):
        path=folder/source.name;path.write_bytes(source.read_bytes());paths.append(path)
    assert len(paths)==64 and sum(p.stat().st_size for p in paths)==1845824
    with pytest.raises(ValueError,match='Unaccounted study evidence'):
        s.envelope.finish()
    assert not (s.out/'stages/saved_replay_end.json').exists()


def test_unregistered_extra_evidence_above_existing_cushion_still_rejected(scenario):
    s=scenario;support.run_saved_replay(s.envelope,s.panel)
    (s.out/'unregistered.bin').write_bytes(b'x'*65537)
    with pytest.raises(ValueError,match='Unaccounted study evidence'):
        s.envelope.finish()


def test_registered_bytes_still_subject_to_evidence_ceiling(scenario,monkeypatch):
    s=scenario;support.run_saved_replay(s.envelope,s.panel)
    monkeypatch.setattr(c,'PREFINAL_EVIDENCE_BYTES',s.envelope.accounted_bytes()-1)
    with pytest.raises(ValueError,match='Study evidence ceiling'):
        s.envelope.finish()


def test_partial_recorder_failure_is_counted_once_without_starting_qualification(scenario,monkeypatch):
    s=scenario
    class FailedRecorder:
        def __init__(self,path):self.path=path;self.stream=path.open('xb');self.closed=False;s.recorders[path]=self
        def record(self,event):
            self.stream.write(b'x'*70000);self.stream.flush();os.fsync(self.stream.fileno())
            raise RuntimeError('test-only original recorder write failure')
        def close(self):self.stream.close();self.closed=True
    monkeypatch.setattr(logging_probe,'make_recorder',lambda path,enabled:FailedRecorder(path))
    with pytest.raises(RuntimeError,match='original recorder write failure'):
        support.run_saved_replay(s.envelope,s.panel)
    raw=list((s.out/'saved_replays').glob('*.jsonl'))
    assert len(raw)==1 and raw[0].stat().st_size==70000
    assert [(p,n) for p,n in s.calls if p.suffix=='.jsonl']==[(raw[0],70000)]
    assert runner.bytes_used()<=s.envelope.accounted_bytes()
    assert not (s.out/'saved_replay_gate.json').exists()
    assert not (s.out/'stages/qualification_start.json').exists()


def test_missing_raw_file_is_not_silently_counted_as_zero(scenario,monkeypatch):
    s=scenario
    class MissingRecorder:
        def record(self,event):pass
        def close(self):pass
    monkeypatch.setattr(logging_probe,'make_recorder',lambda path,enabled:MissingRecorder())
    with pytest.raises(ValueError,match='Expected replay forensic file is missing'):
        support.run_saved_replay(s.envelope,s.panel)
    assert not (s.out/'saved_replay_gate.json').exists()


def test_close_error_retains_original_exception_and_counts_closed_bytes_once(scenario,monkeypatch):
    s=scenario
    class CloseFailedRecorder:
        def __init__(self,path):self.path=path;self.stream=path.open('xb');self.closed=False;s.recorders[path]=self
        def record(self,event):
            self.stream.write(b'x'*70000);self.stream.flush();os.fsync(self.stream.fileno())
        def close(self):
            self.stream.close();self.closed=True
            raise OSError('test-only original recorder close failure')
    monkeypatch.setattr(logging_probe,'make_recorder',lambda path,enabled:CloseFailedRecorder(path))
    with pytest.raises(OSError,match='original recorder close failure'):
        support.run_saved_replay(s.envelope,s.panel)
    raw=list((s.out/'saved_replays').glob('*.jsonl'))
    assert len(raw)==1 and raw[0].stat().st_size==70000
    assert [(p,n) for p,n in s.calls if p.suffix=='.jsonl']==[(raw[0],70000)]
    assert runner.bytes_used()<=s.envelope.accounted_bytes()
    assert not (s.out/'saved_replay_gate.json').exists()
    assert not (s.out/'stages/qualification_start.json').exists()
