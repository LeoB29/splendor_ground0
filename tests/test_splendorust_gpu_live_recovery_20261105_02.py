"""The technical recovery cannot change science or renew either study clock."""
from tools.splendorust_gpu_live_20261105_02 import common as c,control

def test_recovery_preserves_all_scientific_fields_and_original_clocks():
    before=c.read(c.ROOT/'outputs/splendorust_gpu_live_20261105_01/plan.json')
    after=control.build_plan()
    different={'time','authorization','protocol','technical_recovery','inputs','tests','reviews'}
    assert {k:v for k,v in before.items() if k not in different}=={k:v for k,v in after.items() if k not in different}
    claim=c.read(c.ROOT/'outputs/splendorust_gpu_live_20261105_01/claim.json')
    owner=c.read(c.ROOT/'outputs/splendorust_gpu_live_20261105_01/worker/owner.json')
    recovery=after['technical_recovery']
    assert recovery['original_monotonic_origin']==claim['monotonic_origin']
    assert recovery['original_unix_origin']==claim['unix_origin']
    assert recovery['original_worker_monotonic_origin']==owner['monotonic_origin']
    assert recovery['original_worker_unix_origin']==owner['unix_origin']
    assert recovery['predecessor_games']==0
