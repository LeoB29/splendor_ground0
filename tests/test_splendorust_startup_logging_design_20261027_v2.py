"""Workload, authorization gates and immutable clock regression checks."""
import ast
from collections import Counter
from copy import deepcopy
from pathlib import Path
import pytest
from tools.splendorust_startup_logging_20261027_02 import common as c,design,qualification,runner,preparation

def test_exact_prospective_schedule_and_player_work():
    panel=design.build_panel();rows=design.build_matrix(panel)['requests']
    assert [p['case_id'] for p in panel['positions']]==['private_reserves','gold_returns']
    assert len(rows)==96 and [r['sequence'] for r in rows]==list(range(96))
    assert Counter(r['stage'] for r in rows)=={'overhead':64,'diagnostic':32}
    for arm in design.ARMS:
        for case in design.FIXTURES:
            group=[r for r in rows if r['arm_id']==arm and r['case_id']==case]
            assert [r['logging_timing_enabled'] for r in group]==[False,True,True,False,True,False,False,True]+[True]*4
    assert all(r['quota']==64 and r['seed_label']=='A' and r['observer_enabled'] is False and r['telemetry'] is False for r in rows)
    assert all(r['world_pool']==(3 if r['route']=='frozen' else 6) for r in rows)

def test_exact_saved_science_and_repeated_science_reject_corruption():
    spec=design.build_matrix()['requests'][0];target=c.read(c.ROOT/spec['expected_result_file']['path'])['result']
    row=dict(spec=spec,completed=True,clock=dict(status='ok',accepted=True,result=target));cache={}
    assert qualification.validate_repeat_row(row,spec,cache)['historical_comparison']
    assert qualification.validate_repeat_row(row,spec,cache)['first_repeat_comparison']
    changed=deepcopy(row);changed['clock']['result']['legal_count']+=1
    with pytest.raises(ValueError,match='science'):qualification.validate_repeat_row(changed,spec,cache)
    changed=deepcopy(row);changed['clock']['accepted']=False
    with pytest.raises(ValueError,match='Completed'):qualification.validate_repeat_row(changed,spec,{})

def test_deadline_and_resource_contract_cannot_relax():
    contract=preparation.contract();assert runner.validate_freeze_contract(contract)==contract
    assert (contract['seconds_per_move'],contract['internal_seconds'])==(1.,.9)
    assert contract['job_seconds']==3529 and contract['scientific_decisions']==304 and contract['parent_requests']==104
    for key,value in [('seconds_per_move',1.01),('internal_seconds',1.),('automatic_retries',1),('max_workers',5),('total_requests',97)]:
        altered=deepcopy(contract);altered[key]=value
        with pytest.raises(ValueError,match='Exact frozen'):runner.validate_freeze_contract(altered)

def test_original_complete_response_transport_unchanged():
    from tools.splendorust_pool6_core_compare_20261027_02 import runner as old
    assert runner.request.__code__.co_code==old.request.__code__.co_code
    for name in ('request','_receive','_stdout_reader','_sample','close'):
        assert getattr(runner.PlacedProfiledStudyWorker,name) is getattr(old.PlacedProfiledStudyWorker,name)
    assert Path(runner.__file__).with_name('placement.py').read_bytes()==Path(old.__file__).with_name('placement.py').read_bytes()

def test_saved_replay_gate_precedes_actual_models_and_overhead_precedes_detail():
    tree=ast.parse(Path(runner.__file__).read_text());run=next(n for n in tree.body if isinstance(n,ast.FunctionDef) and n.name=='run')
    calls=[(n.lineno,n.func.id) for n in ast.walk(run) if isinstance(n,ast.Call) and isinstance(n.func,ast.Name)]
    line=lambda name:min(n for n,f in calls if f==name)
    assert line('run_saved_replay')<line('qualify')<line('start_host_observer')<line('overhead_gate')
    source=ast.unparse(run)
    assert "if not replay_gate['passed']" in source and "if gate['passed']" in source
    assert 'failed_startup_finalization' in source and "if qualified is None" in source

def test_saved_target_fixture_identity_is_original_failure():
    target=design.build_panel()['positions'][0]
    assert target['observer_sha256']=='6419c6415dbaaf04960d29ee8a5a6524430a161bd75a63d007adc0f65c2fab22'
    assert c.digest(target['before_snapshot'])=='d9bea1e70634d0d2a55b953ed06845ab27acc46dbdc057d781bd21f9b9133472'
