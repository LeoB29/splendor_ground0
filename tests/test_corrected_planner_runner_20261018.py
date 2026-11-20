"""Prospective schedule, binding and preservation guards, without production inference."""
from copy import deepcopy
import json
from pathlib import Path
import os
import sys

import pytest

from tools.corrected_planner_qualification_20261018 import common as c


def entries():
    fits = [161001] * 6 + [161002] * 6 + [161003] * 6
    return [{'fit': f, 'root_id': str(i)} for i, f in enumerate(fits + [161001] * 5)]


def test_fixed_schedule_covers_fits_and_never_eight_candidates():
    cases = c.build_cases(entries())
    assert len(cases) == 49
    assert [v['index'] for v in cases] == list(range(49))
    assert all(v['breadth'] is False for v in cases)
    assert all(v['allowance_seconds'] == 1 for v in cases[:23])
    assert all(v['allowance_seconds'] == 120 for v in cases[23:])
    assert [v['root_index'] for v in cases[46:]] == [0, 6, 12]
    assert all(v['hidden_variant'] and v['kind'] == 'repeat' for v in cases[46:])
    assert [v['fit'] for v in cases[46:]] == list(c.FITS)
    for fit in c.FITS:
        assert sum(v['fit'] == fit and v['kind'] == 'core' and v['allowance_seconds'] == 120 for v in cases) == 6


@pytest.mark.parametrize('count', [0, 18, 22, 24])
def test_schedule_rejects_replacement_panel(count):
    with pytest.raises(ValueError):
        c.build_cases((entries() * 2)[:count])


def test_schedule_rejects_best_fit_selection():
    rows = entries()
    for row in rows:
        row['fit'] = 161001
    with pytest.raises(ValueError):
        c.build_cases(rows)


def fixture(tmp_path):
    observer = {'lawful': 'fixture'}
    action = {'action_type': 'PASS'}
    trace = tmp_path / 'trace.gz'; trace.write_bytes(b'fixture bytes')
    value = {'decision': {'action': action, 'record': {'candidates': {'actions': [action]},
             'worlds': [{'outcomes': [{}]} for _ in range(128)], 'planner': 'reference'}},
             'trace': {'sha256': c.file_hash(trace), 'bytes': trace.stat().st_size, 'records': 128},
             'ruleset_id': c.RULESET_ID, 'catalogue_sha256': c.CATALOGUE_SHA256,
             'profile_id': c.PROFILE_ID, 'observer_sha256': c.digest(observer)}
    return observer, action, trace, value


def test_cheap_binding_accepts_complete_bounded_record(tmp_path):
    observer, action, trace, value = fixture(tmp_path)
    assert c.check_wrapper(value, observer, [action], trace)


@pytest.mark.parametrize('field', ['profile_id', 'ruleset_id', 'catalogue_sha256', 'observer_sha256'])
def test_cheap_binding_rejects_wrong_identity(tmp_path, field):
    observer, action, trace, value = fixture(tmp_path)
    value[field] = 'incorrect'
    with pytest.raises(ValueError):
        c.check_wrapper(value, observer, [action], trace)


def test_cheap_binding_rejects_truncated_worlds(tmp_path):
    observer, action, trace, value = fixture(tmp_path)
    value['decision']['record']['worlds'].pop()
    with pytest.raises(ValueError):
        c.check_wrapper(value, observer, [action], trace)


def test_cheap_binding_rejects_changed_trace(tmp_path):
    observer, action, trace, value = fixture(tmp_path)
    trace.write_bytes(b'changed bytes')
    with pytest.raises(ValueError):
        c.check_wrapper(value, observer, [action], trace)


def test_cheap_binding_rejects_illegal_action(tmp_path):
    observer, action, trace, value = fixture(tmp_path)
    with pytest.raises(ValueError):
        c.check_wrapper(value, observer, [], trace)


def test_preservation_checks_size_and_hash(tmp_path, monkeypatch):
    monkeypatch.setattr(c, 'ROOT', tmp_path)
    p = tmp_path / 'bound'; p.write_bytes(b'original')
    item = {'path': 'bound', 'bytes': 8, 'sha256': c.file_hash(p)}
    c.verify_artifacts([item])
    item['bytes'] = 7
    with pytest.raises(ValueError):
        c.verify_artifacts([item])


def test_python_only_tree_supervisor_receipt_and_timeout(tmp_path):
    from tools.corrected_planner_qualification_20261018.clock import PlannerWorker
    source = '''import sys,json,time
print(json.dumps({'type':'ready','metadata':{}}),flush=True)
for line in sys.stdin:
 r=json.loads(line)
 if r['payload'].get('hang'):time.sleep(30)
 print(json.dumps({'type':'response','request_id':r['request_id'],'result':{'ok':True}}),flush=True)
'''
    (tmp_path / 'planner_clock_fixture.py').write_text(source)
    env = {'PYTHONPATH': str(tmp_path)}
    with PlannerWorker('planner_clock_fixture', tmp_path / 'clock', env=env,
                       initialization_timeout=10) as worker:
        timely = worker.request({}, 2)
        assert timely['accepted'] and timely['received_before_deadline']
        assert timely['telemetry']['resource_scope'] == 'supervised_worker_and_all_descendants'
        assert all(p['role'] in ('python', 'console:python') for p in timely['telemetry']['process_tree'])
        late = worker.request({'hang': True}, 0.2)
        assert late['status'] == 'timeout' and not late['accepted']
        assert worker.process.poll() is not None


def test_python_only_tree_supervisor_rejects_computation_child(tmp_path):
    from tools.corrected_planner_qualification_20261018.clock import PlannerWorker
    source = '''import sys,json,time,subprocess,os
print(json.dumps({'type':'ready','metadata':{}}),flush=True)
for line in sys.stdin:
 child=subprocess.Popen([sys.executable,'-B','-c','import time;time.sleep(30)'],creationflags=subprocess.CREATE_NO_WINDOW if os.name=='nt' else 0)
 time.sleep(30)
'''
    (tmp_path / 'planner_child_fixture.py').write_text(source)
    with PlannerWorker('planner_child_fixture', tmp_path / 'child_clock',
                       env={'PYTHONPATH': str(tmp_path)}, initialization_timeout=10) as worker:
        receipt = worker.request({}, 3)
        assert receipt['status'] == 'process_tree_limit'
        assert not receipt['accepted'] and worker.process.poll() is not None


@pytest.mark.parametrize('proof', [{}, {'status': 'passed'}, {'status': 'failed'}])
def test_audit_cannot_pass_without_bound_full_coverage(tmp_path, proof):
    observer, _, _, result = fixture(tmp_path)
    with pytest.raises(ValueError):
        c.check_audit_proof(proof, observer, result)
