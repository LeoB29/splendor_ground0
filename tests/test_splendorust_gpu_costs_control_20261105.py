import pytest
from tools.splendorust_gpu_costs_20261105 import design, common as c, control


def test_fixed_scope_and_order():
    cells=design.cells()
    assert [x['observed'] for x in cells]==[False,True,True,False,True,True]
    assert all(x['concurrency']==16 and x['route']=='gpu' and x['scheduler']=='row_fifo32' for x in cells)
    assert all(sum(x['groups'],[])==list(range(32)) for x in cells)
    assert len(design.isolated())==4
    assert all(x['measured_cycles']==32 and x['warmup_cycles']==5 for x in design.isolated())
    assert c.INTERNAL_SECONDS==60 and c.OUTER_SECONDS==75


def test_original_complete_decision_registry():
    identity={'pid':123,'created':1.}
    row=dict(execution_policy_id=c.EXECUTION_POLICY_ID,started_unix=100.,deadline_unix=175.,
             started_perf=200.,deadline_perf=275.,kind='ordinary')
    value=dict(pid=123,active=[row])
    control.check_registry(value,identity,174.,274.)
    with pytest.raises(control.DecisionOuterDeadline): control.check_registry(value,identity,175.,274.)
    with pytest.raises(ValueError):
        control.check_registry(dict(pid=123,active=[dict(row,deadline_perf=276.)]),identity,174.,274.)


def test_evidence_outside_new_attempt_rejected(tmp_path):
    with pytest.raises(ValueError): c.save(tmp_path/'bad.json',{})
