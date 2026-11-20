from collections import Counter
from copy import deepcopy
import pytest
from tools.splendorust_pool6_admission_20261027_02 import common as c,design,schedule,preparation
from tools.splendorust_pool6_live_20261027 import schedule as old_schedule
from tools.future_training_rules_20261021 import decode_training_snapshot
from tools.external_strength_screen_20261018 import audit as ledger


def test_fresh_positions_have_full_corrected_observers_and_fixed_strata():
    panel=c.read(design.PANEL)
    assert panel['domain']=='splendorust_pool6_admission_20261027_attempt02_reserved_v1'
    assert panel['model_calls']==0 and panel['no_result_selection'] and panel['no_replacement']
    assert Counter((p['target_ply'],p['seat']) for p in panel['positions'])==Counter(
        {(12,0):3,(13,1):3,(36,0):3,(37,1):3,(60,0):3,(61,1):3})
    old=c.read(preparation.NEGATIVE/'qualification_plan.json')['parity_requests']
    old_observers={c.digest(x['observer']) for x in old}
    for pos in panel['positions']:
        state=decode_training_snapshot(pos['before_snapshot'])
        assert state.turn_index==pos['target_ply'] and state.current_player==pos['seat'] and not state.terminal
        assert c.digest(pos['observer']) not in old_observers
        ledger.check_observer(pos['before_snapshot'],pos['observer'])


def test_fresh_namespaces_are_disjoint_from_all_direct_prior_worlds():
    reserved={p['seed_decimal'] for p in c.read(design.PANEL)['positions']}
    ops={r['seed_decimal'] for r in schedule.make_schedule()}
    old={p['seed_decimal'] for p in c.read(preparation.NEGATIVE/'reserved_panel.json')['positions']}
    for stage in ('operational','main'):old.update(r['seed_decimal'] for r in old_schedule.stage_schedule(stage))
    assert len(reserved)==18 and len(ops)==3 and reserved.isdisjoint(ops)
    assert (reserved|ops).isdisjoint(old)


def test_schedule_is_six_swapped_games_and_rejects_assessment():
    rows=schedule.make_schedule();assert schedule.validate_schedule(rows)==rows
    assert len(rows)==6
    for a,b in zip(rows[::2],rows[1::2]):
        assert a['seed_decimal']==b['seed_decimal'] and a['challenger_seat']!=b['challenger_seat']
        assert a['challenger_quota']==a['opponent_quota']==64
    with pytest.raises(ValueError):schedule.stage_schedule('main')
    with pytest.raises(ValueError):schedule.expected_seed('main',0)
    with pytest.raises(ValueError):schedule.validate_schedule(rows[::-1])
    mutated=deepcopy(rows);mutated[0]['seed_decimal']='1'
    with pytest.raises(ValueError):schedule.validate_schedule(mutated)


def test_fresh_observer_collision_stops_without_replacement(monkeypatch):
    panel=deepcopy(c.read(design.PANEL))
    old=c.read(preparation.NEGATIVE/'qualification_plan.json')['parity_requests'][0]
    panel['positions'][0]['observer']=old['observer']
    monkeypatch.setattr(design,'build_reserved_panel',lambda:panel)
    with pytest.raises(ValueError,match='duplicates'):design.prepare()


def test_seed_collision_stops_without_new_seed(monkeypatch):
    monkeypatch.setattr(schedule,'make_schedule',lambda:old_schedule.stage_schedule('operational'))
    with pytest.raises(ValueError,match='collide'):preparation.seed_inventory()


def test_contract_is_bounded_admission_without_benchmark_or_relaxed_margin():
    p=design.build_plan();limits=preparation.contract()
    assert p['games']==6 and p['pairs']==3 and p['assessment_games']==0
    assert not p['automatic_benchmark_launch'] and not p['automatic_promotion']
    assert p['admission_engineering_margin_seconds']==.7 and p['margin_operator']=='strictly less than'
    assert p['seconds_per_move']==1 and p['internal_seconds']==.9
    assert p['total_scientific_qualification_decisions']==194 and p['qualification_ordinary_requests']==90
    assert limits['job_seconds']==7200 and limits['collection_cutoff_seconds']==3600
    assert limits['main_games']==0 and limits['maximum_reproductions']==48
    assert limits['automatic_retries']==limits['failed_decision_retries']==0
    assert limits['max_workers']==limits['max_graph_builds']==62


def test_both_predecessors_stay_separately_labelled():
    p=c.read(c.OUT/'prelaunch/prior_attempt_preservation_before.json')
    assert p['prior_status']=='completed_verified' and not p['admission_passed']
    assert len(p['prior_attempt_artifacts'])==982
    negative=p['additional_prior_attempts'][0]
    assert negative['prior_status']=='not_admitted_verified' and not negative['admission_passed']
    assert not negative['benchmark_authorized_to_launch'] and len(negative['prior_attempt_artifacts'])==667
