"""Model-free safeguards: latency descriptions never trigger the old cutoff."""
import pytest
from tools.splendorust_gpu_batchfill_20261105 import common as c,control


def registry(kind='ordinary'):
    return dict(pid=123,active=[dict(kind=kind,started_unix=1000.,deadline_unix=1075.,
        started_perf=100.,deadline_perf=175.,execution_policy_id=c.EXECUTION_POLICY_ID)])


@pytest.mark.parametrize('elapsed',[1.01,9.01,10.01,15.,59.,74.99])
def test_descriptive_old_cutoffs_do_not_stop_an_active_move(elapsed):
    control.check_registry(registry(),{'pid':123},1000.+elapsed,100.+elapsed)


@pytest.mark.parametrize('wall,perf',[(1075.,174.),(1074.,175.)])
def test_either_original_outer_clock_expiry_stops_the_owned_decision(wall,perf):
    with pytest.raises(control.DecisionOuterDeadline) as saved:
        control.check_registry(registry(),{'pid':123},wall,perf)
    assert saved.value.row['started_perf']==100.
    assert saved.value.row['deadline_perf']==175.


def test_wrong_owner_and_restarted_interval_are_not_ordinary_timeouts():
    with pytest.raises(ValueError):control.check_registry(registry(),{'pid':124},1001.,101.)
    value=registry();value['active'][0]['deadline_perf']=180.
    with pytest.raises(ValueError):control.check_registry(value,{'pid':123},1001.,101.)


def test_old_policy_and_startup_timeout_cannot_admit_another_cell():
    value=registry();value['active'][0]['execution_policy_id']='old_9s'
    with pytest.raises(ValueError):control.check_registry(value,{'pid':123},1001.,101.)
    with pytest.raises(TimeoutError) as error:
        control.check_registry(registry('native_startup'),{'pid':123},1075.,175.)
    assert not isinstance(error.value,control.DecisionOuterDeadline)
