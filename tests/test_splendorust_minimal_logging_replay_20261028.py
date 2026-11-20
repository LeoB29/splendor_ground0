"""Real saved bytes, matched deferred metadata and two-marker verification."""
import json
from pathlib import Path
import subprocess
import time
from types import SimpleNamespace

import pytest

from tools.splendorust_minimal_logging_20261028 import common as c, design, runner
from tools.splendorust_minimal_logging_20261028 import runtime_support as support, logging_probe

ROOT = Path(__file__).resolve().parents[1]
SAVED = ROOT/'outputs/splendorust_startup_logging_20261027_02/saved_replays'


@pytest.fixture(autouse=True)
def no_model_or_process(monkeypatch):
    def forbidden(*args, **kwargs):
        raise AssertionError('Saved-record verification may not create models or processes')
    monkeypatch.setattr(subprocess, 'Popen', forbidden)
    monkeypatch.setattr(runner, 'make_worker', forbidden)


@pytest.fixture
def replay(tmp_path, monkeypatch):
    out = tmp_path/'trial'; out.mkdir()
    monkeypatch.setattr(c, 'ROOT', tmp_path); monkeypatch.setattr(c, 'OUT', out)
    monkeypatch.setattr(c, 'WRITE_OBSERVER', None)
    monkeypatch.setattr(runner.shutil, 'disk_usage', lambda _: SimpleNamespace(free=100*c.GiB))
    baseline, failure = tmp_path/'baseline', tmp_path/'failure'
    monkeypatch.setattr(design, 'BASELINE', baseline)
    monkeypatch.setattr(design, 'SCIENTIFIC_FAILURE', failure)
    events, positions = [], []
    for index, case in enumerate(('private_reserves', 'gold_returns')):
        event = json.loads((SAVED/f'{index:04d}.jsonl').read_bytes())['event']
        events.append(event); positions.append(dict(case_id=case, observer=event['observer']))
    for path in (baseline/'workers/qualification_frozen_cpu0/entrypoint_forensics.jsonl',
                 failure/'workers/qualification_frozen_cpu2/entrypoint_forensics.jsonl'):
        path.parent.mkdir(parents=True)
        path.write_text('\n'.join(json.dumps(dict(event=e)) for e in events)+'\n')
    panel = dict(positions=positions)
    monkeypatch.setattr(design, 'build_panel', lambda: panel)
    (out/'saved_panel.json').write_text(json.dumps(panel))
    envelope = runner.Envelope(dict(monotonic_origin=time.monotonic(), unix_origin=time.time()))
    envelope.enable_evidence_accounting(); envelope.begin('saved_replay')
    metadata_modes = []; actual = logging_probe.make_recorder
    def make(path, enabled):
        recorder = actual(path, enabled); original = recorder.recorded_entries
        def deferred():
            assert recorder.closed, 'Replay metadata must follow close and measured callback'
            metadata_modes.append(enabled)
            return original()
        recorder.recorded_entries = deferred
        return recorder
    monkeypatch.setattr(logging_probe, 'make_recorder', make)
    gate = support.run_saved_replay(envelope, panel)
    return SimpleNamespace(out=out, envelope=envelope, gate=gate, metadata_modes=metadata_modes)


def mutate_row(case, index, mutation):
    results = json.loads((case.out/'saved_replay_results.json').read_text())
    row = results['rows'][index]; mutation(row)
    (case.out/'saved_replays'/f'{index:04d}.json').write_text(json.dumps(row))
    (case.out/'saved_replay_results.json').write_text(json.dumps(results))


def test_full_real_replay_matches_both_metadata_paths_and_preserves_accounting(replay):
    assert len(replay.metadata_modes) == 64
    assert replay.metadata_modes.count(False) == replay.metadata_modes.count(True) == 32
    result = json.loads((replay.out/'saved_replay_results.json').read_text())
    for row in result['rows']:
        assert len(row['timing_records']) == 1
        stamps = row['timing_records'][0]['timestamps']
        assert set(stamps) == {'start_ns', 'end_ns'}
        if row['logging_timing_enabled']:
            assert row['start_ns'] <= stamps['start_ns'] <= stamps['end_ns'] <= row['end_ns']
        else:
            assert stamps == {'start_ns': None, 'end_ns': None}
    assert support.verify_replay() == replay.gate
    replay.envelope.finish()
    assert (replay.out/'stages/saved_replay_end.json').is_file()
    assert not (replay.out/'workers').exists()


def test_off_metadata_may_not_claim_zero_time(replay):
    mutate_row(replay, 0, lambda row: row['timing_records'][0]['timestamps'].update(start_ns=0, end_ns=0))
    with pytest.raises(ValueError, match='OFF deferred'):
        support.verify_replay()


def test_legacy_component_markers_cannot_enter_minimal_replay(replay):
    mutate_row(replay, 2, lambda row: row['timing_records'][0]['timestamps'].update(synced_ns=0))
    with pytest.raises(ValueError, match='Exactly two'):
        support.verify_replay()


def test_on_span_must_fit_the_original_measured_call(replay):
    mutate_row(replay, 2, lambda row: row['timing_records'][0]['timestamps'].update(end_ns=row['end_ns']+1))
    with pytest.raises(ValueError, match='clock brackets'):
        support.verify_replay()


def test_missing_deferred_off_record_is_not_silently_accepted(replay):
    mutate_row(replay, 0, lambda row: row.update(timing_records=[]))
    with pytest.raises(ValueError, match='Exact deferred'):
        support.verify_replay()


def test_original_mode_order_and_event_bindings_remain_mandatory(replay):
    mutate_row(replay, 0, lambda row: row.update(logging_timing_enabled=True))
    with pytest.raises(ValueError, match='counterbalanced'):
        support.verify_replay()
