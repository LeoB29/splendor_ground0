"""Model-free archival prefix/repeat checks reject truncated and changed evidence."""
from copy import deepcopy
import gzip
import json
import pytest
from tools.move_evaluation_64_20261023 import checks as ck
from test_move_evaluation_64_audit_20261023 import full_fixture


def test_parent_checks_import_without_torch_or_model_module():
    import os
    import subprocess
    import sys
    env = dict(os.environ, PYTHONPATH='src;.')
    code = "import sys; from tools.move_evaluation_64_20261023 import checks; assert 'torch' not in sys.modules; assert 'tools.corrected_planner_qualification_20261018.local' not in sys.modules"
    result = subprocess.run([sys.executable, '-B', '-c', code], env=env, capture_output=True, text=True)
    assert result.returncode == 0, result.stderr


def fixture(tmp_path, monkeypatch, arm='B'):
    payload, result, path, rows, write = full_fixture(tmp_path, monkeypatch, arm)
    monkeypatch.setattr(ck.c, 'ROOT', tmp_path)
    result['trace']['path'] = path.name
    archive = deepcopy(result['decision'])
    rec = archive['record']; rec['schedule'] = deepcopy(rec['proposal']['schedule'])
    rec['worlds'] = rec['worlds']+deepcopy(rec['worlds'])
    old_path = tmp_path/'original.gz'
    with gzip.open(old_path, 'wt', encoding='utf-8') as f:
        for row in rows:
            f.write(json.dumps(row)+'\n')
    return payload, result, path, archive, old_path


@pytest.mark.parametrize('arm', ['A', 'B'])
def test_complete_result_and_historical_prefix(monkeypatch, tmp_path, arm):
    p, r, path, old, old_path = fixture(tmp_path, monkeypatch, arm)
    assert ck.check_selection(r, p, arm)['continuations'] == 64
    receipt = ck.check_archived_prefix(r, old, arm, trace_path=path, archived_trace_path=old_path)
    assert receipt['full_step_rows_compared'] == 64 and not receipt['historical_choices_used_for_selection']


def test_b_prefix_requires_fullstep_rows(monkeypatch, tmp_path):
    _, r, _, old, _ = fixture(tmp_path, monkeypatch)
    with pytest.raises(ValueError, match='full step-row'):
        ck.check_archived_prefix(r, old, 'B')


@pytest.mark.parametrize('damage', ['outcome', 'state', 'length', 'actionhash', 'schedule', 'scores'])
def test_archived_identity_mutations_rejected(monkeypatch, tmp_path, damage):
    _, r, path, old, old_path = fixture(tmp_path, monkeypatch)
    rec = old['record']
    if damage == 'outcome': rec['worlds'][0]['outcomes'][0]['winner'] = 1
    elif damage == 'state': rec['worlds'][0]['state_sha256'] = 'bad'
    elif damage == 'length': rec['worlds'].pop()
    elif damage == 'actionhash': rec['worlds'][0]['outcomes'][0]['trace_sha256'] = 'bad'
    elif damage == 'schedule': rec['schedule']['world_seeds'].reverse()
    else: rec['scores'] = [3.]
    with pytest.raises(ValueError): ck.check_archived_prefix(r, old, 'B', trace_path=path, archived_trace_path=old_path)


def test_repeat_ignores_only_output_path(monkeypatch, tmp_path):
    _, r, path, _, _ = fixture(tmp_path, monkeypatch)
    other = tmp_path/'second.gz'; other.write_bytes(path.read_bytes())
    repeat = deepcopy(r); repeat['trace']['path'] = other.name
    assert ck.compare_duplicate(r, repeat)['scientific_output_and_gzip_exact']
    repeat['decision']['record']['choice'] = True
    with pytest.raises(ValueError): ck.compare_duplicate(r, repeat)
