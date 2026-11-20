"""The new authorized dispatch attempt preserves both failures and other settings."""
from tools.splendorust_gpu_live_20261105_03 import common as c,control

def test_new_attempt_changes_only_gather_wait_and_binds_both_failures():
    before=c.read(c.ROOT/'outputs/splendorust_gpu_live_20261105_02/plan.json')
    after=control.build_plan()
    different={'time','authorization','protocol','technical_recovery','independent_attempt','inputs','tests','reviews','maximum_batch_wait_seconds'}
    assert {k:v for k,v in before.items() if k not in different}=={k:v for k,v in after.items() if k not in different}
    assert before['maximum_batch_wait_seconds']==.001 and after['maximum_batch_wait_seconds']==0.
    attempt=after['independent_attempt']
    assert attempt['fresh_clock_origins'] and attempt['predecessor_games']==0
    assert [p['attempt'] for p in attempt['predecessors']]==['01','02']
    for prior in attempt['predecessors']:
        assert prior['preserved_evidence'] and prior['preserved_reports'] and prior['frozen_inputs']
        c.verify(prior['preserved_evidence']+prior['preserved_reports']+prior['frozen_inputs'])
