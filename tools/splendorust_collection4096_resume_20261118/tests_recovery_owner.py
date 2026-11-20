"""Model-free clock, recovery scope and worker-boundary counterexamples."""
from copy import deepcopy
import pytest
from . import clocks, common as c, owner, runtime


@pytest.fixture
def sample(monkeypatch):
    start=1_000_000.
    original=dict(schema='collection4096_original_deadlines_v1',original_unix_origin=start,
        global_unix_deadline=start+c.TOTAL_SECONDS,model_unix_deadline=start+c.MODEL_SECONDS,
        collection_unix_origin=start+600.,collection_unix_deadline=start+600.+120*3600,
        original_bindings=[])
    monkeypatch.setattr(clocks,'original_deadlines',lambda:deepcopy(original))
    mono,wall,boot=10_000.,start+100_000.,start+90_000.
    monkeypatch.setattr(clocks.psutil,'boot_time',lambda:boot)
    claim=clocks.create_limits(mono=mono,wall=wall,boot=boot)
    return dict(claim=claim,original=original,mono=mono,wall=wall,boot=boot)


def test_rebase_charges_downtime_and_retains_utc_bounds(sample):
    claim=sample['claim']
    assert claim['unix_origin']==sample['original']['original_unix_origin']
    assert claim['monotonic_origin']==-90_000.
    assert claim['monotonic_deadline']-sample['mono']==claim['unix_deadline']-sample['wall']
    assert claim['unix_deadline']==sample['original']['global_unix_deadline']
    assert claim['per_decision_clocks_rebased'] is False


@pytest.mark.parametrize('field,amount',[
    ('monotonic_origin',1.),('unix_origin',1.),('monotonic_deadline',1.),('unix_deadline',1.)])
def test_origin_or_allowance_refund_refused(sample,field,amount):
    bad=deepcopy(sample['claim']);bad[field]+=amount
    with pytest.raises(ValueError):clocks.validate(bad)


@pytest.mark.parametrize('field', ['global_unix_deadline','model_unix_deadline','collection_unix_deadline'])
def test_rehashed_original_deadline_extension_refused(sample,field):
    bad=deepcopy(sample['claim']);bad['original_deadlines'][field]+=1
    bad['original_deadlines_sha256']=c.digest(bad['original_deadlines'])
    with pytest.raises(ValueError):clocks.validate(bad)


def test_original_elapsed_global_budget_cannot_restart(sample):
    with pytest.raises(ValueError):
        clocks.create_limits(mono=sample['mono'],wall=sample['original']['global_unix_deadline'],boot=sample['boot'])


def test_restarted_monotonic_epoch_refused(sample):
    with pytest.raises(ValueError):
        clocks.check(sample['claim'],mono=1.,wall=sample['wall']+1,boot=sample['boot']+10)


@pytest.mark.parametrize('correction',[1.,5.])
def test_backward_wall_correction_tolerated_inside_fixed_bounds(sample,correction):
    before=deepcopy(sample['claim'])
    assert clocks.check(sample['claim'],mono=sample['mono']+10,wall=sample['wall']+10-correction,
                        boot=sample['boot']-correction)
    assert sample['claim']==before


def test_boot_estimate_change_is_diagnostic_only(sample):
    assert clocks.check(sample['claim'],mono=sample['mono']+10,wall=sample['wall']+10,boot=sample['boot']+10)


def test_wall_correction_just_after_anchor_does_not_reanchor_deadlines(sample):
    before=deepcopy(sample['claim'])
    assert clocks.check(sample['claim'],mono=sample['mono']+.1,wall=sample['wall']-5,boot=sample['boot']-5)
    assert sample['claim']==before


def test_small_clock_sampling_skew_tolerated_without_extending_deadline(sample):
    assert clocks.check(sample['claim'],mono=sample['mono']+.1,wall=sample['wall'],boot=sample['boot'])


def test_model_cutoff_precedes_overall_cutoff(sample):
    end=sample['original']['model_unix_deadline'];mono=clocks.mapped_mono(sample['claim'],end)
    assert clocks.check(sample['claim'],mono=mono,wall=end,boot=sample['boot'],model=False)
    with pytest.raises(ValueError):clocks.check(sample['claim'],mono=mono,wall=end,boot=sample['boot'],model=True)


def test_collection_phase_keeps_original_start_and_deadline(sample):
    phase=clocks.phase_limits(sample['claim'],'collection',mono=sample['mono'],wall=sample['wall'],boot=sample['boot'])
    assert phase['unix_time']==sample['original']['collection_unix_origin']
    assert phase['unix_deadline']==sample['original']['collection_unix_deadline']
    assert phase['monotonic']<sample['mono']
    assert phase['monotonic_deadline']-sample['mono']==phase['unix_deadline']-sample['wall']


def test_collection_phase_cannot_restart_after_shutdown(sample):
    phase=clocks.phase_limits(sample['claim'],'collection',mono=sample['mono'],wall=sample['wall'],boot=sample['boot'])
    phase.update(unix_time=sample['wall'],monotonic=sample['mono'])
    with pytest.raises(ValueError):clocks.validate_phase(sample['claim'],phase)


def test_expired_original_collection_stage_refused(sample):
    wall=sample['original']['collection_unix_deadline'];mono=clocks.mapped_mono(sample['claim'],wall)
    with pytest.raises(ValueError):
        clocks.phase_limits(sample['claim'],'collection',mono=mono,wall=wall,boot=sample['boot'])


@pytest.mark.parametrize('name,cap',[('supplemental',7200),('corpus',86400),('final',86400)])
def test_fresh_phase_cap_cannot_extend(sample,name,cap):
    phase=clocks.phase_limits(sample['claim'],name,mono=sample['mono'],wall=sample['wall'],boot=sample['boot'])
    assert phase['cap_seconds']==cap
    assert phase['unix_deadline']==sample['wall']+cap
    phase['cap_seconds']+=1
    with pytest.raises(ValueError):clocks.validate_phase(sample['claim'],phase)


def test_supplement_clipped_to_remaining_collection_time(sample):
    wall=sample['original']['collection_unix_deadline']-30;mono=clocks.mapped_mono(sample['claim'],wall)
    phase=clocks.phase_limits(sample['claim'],'supplemental',mono=mono,wall=wall,boot=sample['boot'])
    assert phase['unix_deadline']==wall+30
    assert phase['monotonic_deadline']==mono+30


@pytest.mark.parametrize('correction',[1.,5.])
def test_new_phase_cannot_refund_wall_clock_correction(sample,correction):
    mono,wall=sample['mono']+10,sample['wall']+10-correction
    phase=clocks.phase_limits(sample['claim'],'final',mono=mono,wall=wall,boot=sample['boot']-correction)
    assert phase['monotonic_deadline']<=mono+86400
    assert phase['unix_deadline']<=wall+86400
    assert phase['monotonic_deadline']<=sample['claim']['monotonic_deadline']
    assert phase['unix_deadline']<=sample['claim']['unix_deadline']


@pytest.mark.parametrize('correction',[1.,5.])
@pytest.mark.parametrize('scope',['overall','model','phase','worker'])
def test_backward_clock_correction_cannot_refund_any_cutoff(sample,correction,scope):
    claim=sample['claim']
    if scope in ('overall','model'):
        wall=sample['original']['global_unix_deadline' if scope=='overall' else 'model_unix_deadline']
        mono=clocks.mapped_mono(claim,wall)
        with pytest.raises(ValueError):
            clocks.check(claim,mono=mono,wall=wall-correction,boot=sample['boot']-correction,model=scope=='model')
    else:
        phase=clocks.phase_limits(claim,'supplemental',mono=sample['mono'],wall=sample['wall'],boot=sample['boot'])
        worker=dict(monotonic_origin=sample['mono'],unix_origin=sample['wall'],
            monotonic_deadline=sample['mono']+(7201 if scope=='phase' else 20),
            unix_deadline=sample['wall']+(7201 if scope=='phase' else 20))
        elapsed=7200 if scope=='phase' else 20
        with pytest.raises(ValueError):
            runtime.check_deadlines(claim,phase,worker,sample['mono']+elapsed,sample['wall']+elapsed-correction,True)


def test_worker_deadline_and_phase_enforced(sample):
    phase=clocks.phase_limits(sample['claim'],'supplemental',mono=sample['mono'],wall=sample['wall'],boot=sample['boot'])
    worker=dict(monotonic_origin=sample['mono'],unix_origin=sample['wall'],
        monotonic_deadline=sample['mono']+10,unix_deadline=sample['wall']+10)
    runtime.check_deadlines(sample['claim'],phase,worker,sample['mono']+9,sample['wall']+9,True)
    with pytest.raises(ValueError):
        runtime.check_deadlines(sample['claim'],phase,worker,sample['mono']+10,sample['wall']+10,True)


@pytest.mark.parametrize('correction',[1.,5.])
def test_running_worker_tolerates_backward_correction_inside_existing_bounds(sample,correction):
    phase=clocks.phase_limits(sample['claim'],'supplemental',mono=sample['mono'],wall=sample['wall'],boot=sample['boot'])
    worker=dict(monotonic_origin=sample['mono'],unix_origin=sample['wall'],
        monotonic_deadline=sample['mono']+20,unix_deadline=sample['wall']+20)
    before=deepcopy((sample['claim'],phase,worker))
    runtime.check_deadlines(sample['claim'],phase,worker,sample['mono']+10,sample['wall']+10-correction,True)
    assert (sample['claim'],phase,worker)==before


def test_recovery_scope_permits_only_eight_named_incomplete_decisions():
    contract=owner.recovery_contract()
    assert contract['specific_gameplay_retries']==[dict(game_id='g'+str(g),ply=p) for g,p in
        [(2168,27),(2169,23),(2170,22),(2171,22),(2172,22),(2173,21),(2174,20),(2175,17)]]
    assert contract['gameplay_retries']==8 and contract['new_verification_retries']==0
    assert contract['replacement_games']==contract['automatic_retries']==contract['completed_moves_retried']==0
    assert contract['remaining_cell_indices']==list(range(136,208))
    assert contract['maximum_selected_replay_attempts']==3328*3+1
    assert contract['new_qualification_fixture_decisions']==116
    assert contract['new_qualification_games']==contract['new_optimizer_updates']==0


def test_recovery_inherits_exact_scientific_contract():
    from tools.splendorust_collection4096_20261115.owner import contract as original
    assert owner.contract()==original()


def test_actual_original_deadlines_are_the_frozen_values():
    original=clocks.original_deadlines()
    assert original['collection_unix_deadline']==1795174250.1671505
    assert original['global_unix_deadline']-original['original_unix_origin']==168*3600
    assert original['model_unix_deadline']-original['original_unix_origin']==144*3600
