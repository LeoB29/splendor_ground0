"""Model-free acceptance and real hidden-process smoke checks."""
from copy import deepcopy
import sys

import psutil
import pytest

from tools.matched_strength_20261020 import completion as m


def good_report():
    return {'status': 'passed', 'final_acceptance': False, 'outcomes_opened': True,
            'freeze_sha256': 'f'*64, 'strength_analysis_games': 60,
            'completed_games': 60, 'completed_pairs': 30, 'statistics': {'score': .5}}


@pytest.mark.parametrize('key,value', [('status', 'failed'), ('final_acceptance', True),
    ('outcomes_opened', False), ('freeze_sha256', 'e'*64), ('strength_analysis_games', 59),
    ('completed_games', 61), ('completed_pairs', 29), ('statistics', None)])
def test_finalization_rejects_wrong_or_partial_scope(monkeypatch, key, value):
    monkeypatch.setattr(m.c, 'config', lambda: {'pair_count': 30})
    report = good_report(); report[key] = value
    with pytest.raises(ValueError, match='scope'):
        m.verify_report(report, 'f'*64)


def test_full_report_scope_is_eligible_but_not_self_accepting(monkeypatch):
    monkeypatch.setattr(m.c, 'config', lambda: {'pair_count': 30})
    report = good_report(); before = deepcopy(report)
    m.verify_report(report, 'f'*64)
    assert report == before and report['final_acceptance'] is False


def test_hidden_model_free_child_and_console_helper_close(tmp_path):
    argv = [sys.executable, '-B', '-c', 'import time; print("model-free smoke", flush=True); time.sleep(.3)']
    plan = m.contract()
    plan.update(analyzer_command=argv, runner_identity=m.identity(psutil.Process()))
    original = psutil.Process().cpu_affinity()
    result = m.supervise(argv, tmp_path/'child.log', plan, guard=lambda: None, maximum_seconds=10)
    assert result['exit_code'] == 0 and result['error'] is None
    assert not result['cleanup']['errors']
    assert result['cleanup']['parent_closed'] and result['cleanup']['descendants_closed']
    assert psutil.Process().cpu_affinity() == original
    assert all(x['allowed_console_helper'] for x in result['observed_descendants'])


def test_failed_guard_preserves_failure_and_closes_owned_process(tmp_path):
    argv = [sys.executable, '-B', '-c', 'import time; time.sleep(10)']
    plan = m.contract()
    plan.update(analyzer_command=argv, runner_identity=m.identity(psutil.Process()))
    def fail():
        raise RuntimeError('synthetic resource hold')
    result = m.supervise(argv, tmp_path/'held.log', plan, guard=fail, maximum_seconds=10)
    assert 'synthetic resource hold' in result['error']
    assert result['cleanup']['parent_closed'] and result['cleanup']['descendants_closed']
    assert result['automatic_retries'] == 0
