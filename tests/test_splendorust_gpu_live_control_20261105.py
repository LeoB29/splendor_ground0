"""Deadline supervision tests use no real model, game or process."""
import pytest
from tools.splendorust_gpu_live_20261105 import control

def registry():
    return dict(pid=123,active=[dict(client_id='test',started_unix=100.,deadline_unix=110.,started_perf=200.,deadline_perf=210.)])

def test_active_deadline_accepts_only_original_budget():
    control.check_active(registry(),{'pid':123},109.,209.)

@pytest.mark.parametrize('wall,perf',[(110.,209.),(109.,210.),(111.,211.)])
def test_either_clock_can_enforce_deadline(wall,perf):
    with pytest.raises(ValueError,match='Actual outer10s'):control.check_active(registry(),{'pid':123},wall,perf)

def test_rejects_deadline_extension():
    value=registry();value['active'][0]['deadline_unix']=111.
    with pytest.raises(ValueError,match='Unextended'):control.check_active(value,{'pid':123},109.,209.)

def test_rejects_other_process_registry():
    with pytest.raises(ValueError,match='Owned'):control.check_active(registry(),{'pid':124},109.,209.)
