"""Deadline, privacy and supervision regressions; no models or game inference."""
import json
import os
from pathlib import Path
import time

import pytest

from tools.move_evaluation_20261023 import common as c, runner


def test_request_limit_never_extends_phase_deadline():
    assert runner.clamp_timeout(120, 101., clock=lambda: 100.) == 1.
    assert runner.clamp_timeout(120, 1000., clock=lambda: 100.) == 120
    with pytest.raises(ValueError, match='deadline'):
        runner.clamp_timeout(120, 100., clock=lambda: 100.)


def test_selector_strips_all_parent_secrets_and_outcomes():
    row = {k: k for k in ('observer', 'fit', 'candidates', 'selection_schedule')}
    row.update(assessment_world_seeds_parent_only=['secret'], control_means=[1], control_action='secret', snapshot='secret')
    assert set(c.selector_payload(row)) == {'observer', 'fit', 'candidates', 'selection_schedule'}


def test_extra_selector_seed_is_rejected_before_scoring():
    from tools.move_evaluation_20261023.evaluator import reproduce
    with pytest.raises(ValueError, match='allowlisted'):
        reproduce({'observer': {}, 'fit': 161001, 'candidates': {}, 'selection_schedule': {}, 'assessment_seed': 1})


def test_failed_receipt_is_never_an_action_or_automatic_retry():
    class Child:
        count = 0
        def request(self, payload, timeout):
            self.count += 1
            return {'status': 'timeout', 'accepted': False, 'detail': 'synthetic', 'receipt_elapsed_seconds': None}
    child = Child()
    with pytest.raises(ValueError, match='Request failed'):
        runner.request(child, {}, 120, time.monotonic()+120)
    assert child.count == 1


def test_native_and_python_supervisor_reuses_real_clock_and_kills_on_timeout(tmp_path):
    script = '''import json,sys,time
print(json.dumps({'type':'ready','metadata':{}}),flush=True)
for line in sys.stdin:
 r=json.loads(line)
 if r['payload'].get('hang'):time.sleep(10)
 print(json.dumps({'type':'response','request_id':r['request_id'],'result':{'ok':True}}),flush=True)
'''
    (tmp_path/'synthetic_move_worker.py').write_text(script)
    with c.StudyWorker('synthetic_move_worker', tmp_path/'run', native_artifact=None,
                      initialization_timeout=5, env={'PYTHONPATH': str(tmp_path)}) as child:
        good = child.request({}, 1)
        assert good['status'] == 'ok' and good['received_before_deadline']
        bad = child.request({'hang': True}, .1)
        assert bad['status'] == 'timeout' and not bad['accepted']
        assert child.process.poll() is not None
        assert json.loads((tmp_path/'run/request-00000002.json').read_text()) == bad


def test_memory_healthy_fast_path_does_not_wait(monkeypatch, tmp_path):
    monkeypatch.setattr(c, 'ROOT', tmp_path)
    monkeypatch.setattr(c, 'OUT', tmp_path)
    envelope = runner.Envelope({'monotonic': time.monotonic(), 'unix_time': time.time()})
    envelope.admitted = True
    monkeypatch.setattr(envelope, 'guard', lambda *a, **kw: None)
    monkeypatch.setattr(runner.psutil, 'virtual_memory', lambda: type('V', (), {'available': 3*c.GiB})())
    monkeypatch.setattr(runner.time, 'sleep', lambda _: pytest.fail('Healthy root must not sleep'))
    envelope.admit(envelope.end)
    assert envelope.waited < 1


def test_literal_primary_bootstrap_matches_independent_chunked_calculator():
    from tools.move_evaluation_20261023.primary_statistics import calculate
    from tools.move_evaluation_20261023.audit import bootstrap_totals
    import hashlib
    differences = [[(i+j)%5-2 for j in range(32)] for i in range(18)]
    rows = [{'fit': 161001+i//6, 'differences_halfpoints': differences[i]} for i in range(18)]
    result = calculate(rows)
    expected = bootstrap_totals(differences)
    assert result['bootstrap_integer_totals_sha256'] == hashlib.sha256(expected.astype('<i8').tobytes()).hexdigest()
