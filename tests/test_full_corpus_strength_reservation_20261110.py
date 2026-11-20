"""Fresh seed reservation and regression of the saved phase-binding failure."""
from copy import deepcopy
import os
from types import SimpleNamespace
import pytest
os.environ.setdefault('INDEPENDENT_DATA_REPETITION', '0')
os.environ.setdefault('FULL_CORPUS_MATCHUP','mixed_parent')
from tools.splendorust_full_corpus_20261110 import common as root, schedule
from tools.splendorust_full_corpus_20261110.strength import common as c, control
from tools.splendorust_independent_data_20261107 import schedule as prior_schedule

def phase_fixture(monkeypatch, tmp_path, mode):
    out = tmp_path / 'recovery'
    monkeypatch.setattr(root, 'ROOT', tmp_path); monkeypatch.setattr(root, 'OUT', out)
    monkeypatch.setattr(root, 'WRITE_OBSERVER', None)
    monkeypatch.setattr(c, 'ROOT', tmp_path); monkeypatch.setattr(c, 'OUT', out / 'strength/r0')
    monkeypatch.setattr(c, 'STUDY_OUT', out)
    claim = root.save(out / 'claim.json', {'monotonic_origin': 1000., 'unix_origin': 100000.})
    stage = c.STAGE_MAP['main' if mode == 'main_block' else 'final' if mode == 'saved' else mode]
    cap = root.STAGES[stage]
    cutoff = root.JOB_SECONDS if stage == 'final' else root.MODEL_SECONDS
    clock = {'stage': stage, 'claim': claim, 'monotonic': 2000., 'unix_time': 101000.,
             'cap_seconds': cap, 'monotonic_deadline': min(2000.+cap, 1000.+cutoff),
             'unix_deadline': min(101000.+cap, 100000.+cutoff), 'original_phase': {'saved': True}}
    descriptor = root.save(out / 'stages' / (stage + '_start.json'), clock)
    root.save(c.OUT / 'freeze.json', {'fixture': True})
    monkeypatch.setattr(control.time, 'monotonic', lambda: 2001.)
    monkeypatch.setattr(control.time, 'time', lambda: 101001.)
    env = SimpleNamespace(claim=claim, phase_receipt=descriptor, stage=stage, start=1000., wall_start=100000.,
        deadline=clock['monotonic_deadline'], wall_deadline=clock['unix_deadline'], segment_claim=None)
    return env, clock


@pytest.mark.parametrize('mode', ['qualification', 'operational', 'main_block', 'replay', 'saved'])
def test_actual_value_target_segment_accepts_shared_phase_without_reset(monkeypatch, tmp_path, mode):
    env, clock = phase_fixture(monkeypatch, tmp_path, mode)
    item = control._segment_start(env, {}, mode, 0 if mode == 'main_block' else None)
    value = root.read(root.ROOT/item['path'])
    assert value['phase'] == mode and value['root_phase_clock'] == env.phase_receipt
    assert value['monotonic_deadline'] == clock['monotonic_deadline']
    assert value['unix_deadline'] == clock['unix_deadline']
    assert value['original_monotonic_origin'] == 1000.
    assert value['deadline_reset'] is False and value['active_decision'] is False


@pytest.mark.parametrize('mutation', ['wrong_phase', 'wrong_claim', 'clock_extended', 'expired'])
def test_actual_value_target_start_rejects_phase_identity_and_clock_corruption(monkeypatch, tmp_path, mutation):
    env, clock = phase_fixture(monkeypatch, tmp_path, 'qualification')
    if mutation == 'wrong_phase': env.stage = 'qualification'
    elif mutation == 'wrong_claim': env.claim = {**env.claim, 'sha256': '0' * 64}
    elif mutation == 'clock_extended': env.deadline += 1
    else: monkeypatch.setattr(control.time, 'monotonic', lambda: clock['monotonic_deadline'])
    with pytest.raises(ValueError): control._segment_start(env, {}, 'qualification', None)
    assert not (c.OUT / 'segments').exists()



def test_fresh_strength_starts_and_intentionally_paired_fit_seeds():
    rows = schedule.strength_schedule()
    proof = schedule.validate(rows)
    assert proof['strength_games'] == 1236 and proof['strength_pairs'] == 618
    assert proof['new_collection_games'] == 0
    old = {row['seed_decimal'] for row in prior_schedule.strength_schedule()+prior_schedule.collection_schedule()+prior_schedule.qualification_schedule()}
    assert not old & {row['seed_decimal'] for row in rows}
    assert schedule.FIT_SEEDS == prior_schedule.FIT_SEEDS
    assert set(schedule.auxiliary_seeds()).isdisjoint(prior_schedule.BOOTSTRAP_SEEDS.values())

@pytest.mark.parametrize('field,value', [('repetition',False),('candidate_simulations',128.0),('seed_decimal','1'),('candidate_seat',True)])
def test_fresh_reservation_rejects_typed_or_seed_substitution(field,value):
    rows = deepcopy(schedule.strength_schedule()); rows[0][field]=value
    with pytest.raises(ValueError): schedule.validate(rows)

def test_corrected_fresh_pair_starts_roundtrip():
    from tools.future_training_rules_20261021 import decode_training_snapshot
    from tools.external_strength_screen_20261018 import rules
    for rep in (0,1):
        spec=schedule.strength_spec('operational',rep,0,0,'mixed_parent')
        initial=schedule._initial(spec['seed_decimal'])
        assert initial == rules.serialize_snapshot(decode_training_snapshot(initial))
        assert initial['state']['turn_index'] == 0 and initial['state']['terminal'] is False
