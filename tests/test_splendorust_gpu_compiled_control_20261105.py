import pytest
from tools.splendorust_gpu_compiled_20261105 import design,common as c,control


def test_fixed_single_candidate_and_reversed_comparison():
    rows=design.cells()
    assert [r['implementation'] for r in rows]==['eager','compiled','compiled','eager']
    assert [r['block'] for r in rows]==[1,1,2,2]
    assert all(r['concurrency']==16 and r['scheduler']=='row_fifo32' for r in rows)
    assert all(sum(r['groups'],[])==list(range(32)) for r in rows)
    assert design.qualification_cell()['phase']=='decision_qualification'
    assert design.support()['fixtures']==36 and design.support()['repetitions_per_fixture']==2


def test_original_clock_cannot_be_extended():
    ident=dict(pid=1,created=1.)
    row=dict(execution_policy_id=c.EXECUTION_POLICY_ID,started_unix=100.,deadline_unix=175.,
             started_perf=200.,deadline_perf=275.,kind='ordinary')
    control.check_registry(dict(pid=1,active=[row]),ident,174.,274.)
    with pytest.raises(control.DecisionOuterDeadline):
        control.check_registry(dict(pid=1,active=[row]),ident,175.,274.)
    with pytest.raises(ValueError):
        control.check_registry(dict(pid=1,active=[dict(row,deadline_perf=276.)]),ident,174.,274.)


def test_new_output_ownership(tmp_path):
    with pytest.raises(ValueError):c.save(tmp_path/'old_evidence.json',{})
