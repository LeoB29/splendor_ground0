from types import SimpleNamespace
import time
import pytest
from tools.splendorust_gpu_collector_20261105 import common as c,control,design,audit


def test_matched_setups_and_counterbalanced_routes():
    games={'specifications':[{'game_index':i,'seed_decimal':str(i+100)} for i in range(32)]}
    cells=design.cells(games)
    assert [x['route'] for x in cells]==['cpu','gpu','gpu','cpu']
    assert [x['concurrency'] for x in cells]==[8,16,16,8]
    assert cells[0]['specifications']==cells[1]['specifications']
    assert cells[2]['specifications']==cells[3]['specifications']
    assert set(cells[0]['game_indices']).isdisjoint(cells[2]['game_indices'])


def test_original_deadline_cannot_be_reset_or_extended():
    ident={'pid':1,'created':1.}
    row=dict(execution_policy_id=c.EXECUTION_POLICY_ID,started_unix=100.,deadline_unix=175.,
             started_perf=200.,deadline_perf=275.,kind='ordinary')
    control.check_registry(dict(pid=1,active=[row]),ident,174.,274.)
    with pytest.raises(control.DecisionOuterDeadline):control.check_registry(dict(pid=1,active=[row]),ident,175.,274.)
    with pytest.raises(ValueError):control.check_registry(dict(pid=1,active=[dict(row,deadline_perf=276.)]),ident,174.,274.)
    with pytest.raises(ValueError):control.check_registry(dict(pid=2,active=[row]),ident,174.,274.)


def harness(tmp_path,monkeypatch,*,failed_stage=None,deadline_stage=None,gate=True):
    monkeypatch.setattr(c,'OUT',tmp_path);monkeypatch.setattr(c,'PLAN',tmp_path/'plan.json')
    monkeypatch.setattr(control.psutil,'Process',lambda *args:SimpleNamespace(cpu_affinity=lambda *args:None,create_time=lambda:123.))
    monkeypatch.setattr(control,'require_absent',lambda identity:None)
    cells=[dict(cell_id=x,route='gpu' if 'gpu' in x else 'cpu') for x in ('b1_cpu','b1_gpu','b2_gpu','b2_cpu')]
    plan=dict(inputs=[],supervisor_affinity=[19],total_seconds=c.TOTAL_SECONDS,model_seconds=c.MODEL_SECONDS,cells=cells)
    c.save(c.PLAN,plan);seen=[]
    def stage(plan,expected,mode,spec,folder,guard):
        name='qualification' if mode=='qualification' else spec['cell_id'];seen.append(name)
        assert not folder.exists();folder.mkdir(parents=True)
        mono,wall=time.monotonic(),time.time()
        status='fatal' if name==failed_stage else 'deadline_failed' if name==deadline_stage else 'completed'
        row=dict(stage_id=name,mode=mode,status=status,worker_exit_code=1 if status=='fatal' else 20 if status=='deadline_failed' else 0,
                 observed_owned_identities=[],monotonic_origin=mono,monotonic_end=mono,unix_origin=wall,unix_end=wall)
        c.save(folder/'execution.json',row);return row
    monkeypatch.setattr(control,'run_stage',stage)
    monkeypatch.setattr(audit,'audit_qualification',lambda *args:{'qualified':gate},raising=False)
    monkeypatch.setattr(audit,'qualification_gate',lambda proof:{'passed':proof['qualified']},raising=False)
    monkeypatch.setattr(audit,'audit_cell',lambda plan,spec,folder:{'cell_id':spec['cell_id']},raising=False)
    monkeypatch.setattr(audit,'comparison_summary',lambda proofs:{'count':len(proofs)},raising=False)
    return seen


def test_failed_collector_admission_never_starts_games(tmp_path,monkeypatch):
    seen=harness(tmp_path,monkeypatch,gate=False)
    assert control.run(c.sha(c.PLAN))==1
    assert seen==['qualification'] and (tmp_path/'failure.json').exists()
    assert not (tmp_path/'qualification_gate.json').exists() and not (tmp_path/'cells').exists()


def test_first_deadline_preserves_prefix_without_replacement_or_rate(tmp_path,monkeypatch):
    seen=harness(tmp_path,monkeypatch,deadline_stage='b1_gpu')
    assert control.run(c.sha(c.PLAN))==0
    assert seen==['qualification','b1_cpu','b1_gpu']
    assert not (tmp_path/'comparison.json').exists()
    completion=c.read(tmp_path/'completion.json')
    assert completion['incomplete_cell']=='b1_gpu' and completion['audited_stages']==3
    assert len(completion['cell_costs'])==2


def test_full_comparison_only_after_all_four_saved_audits(tmp_path,monkeypatch):
    seen=harness(tmp_path,monkeypatch)
    assert control.run(c.sha(c.PLAN))==0
    assert seen==['qualification','b1_cpu','b1_gpu','b2_gpu','b2_cpu']
    assert c.read(tmp_path/'comparison.json')=={'count':4}
    for name in seen[1:]:
        cost=c.read(tmp_path/'cells'/name/'cell_cost.json')
        assert cost['verified_seconds']>=cost['worker_seconds']+cost['saved_audit_seconds']
        c.verify([cost['saved_audit'],cost['execution'],cost['auditor']])
    preserved=(tmp_path/'completion.json').read_bytes()
    with pytest.raises(ValueError):control.run(c.sha(c.PLAN))
    assert (tmp_path/'completion.json').read_bytes()==preserved
