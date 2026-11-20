"""Model-free schedule/ledger/reproduction counterexamples; no new player calls.

Historical full-game transition fixtures deliberately stub numerical validation
and seed initialization; the real corrected ledger and clocks remain active.
The separate actual saved256 reproduction fixture does not stub science.
"""
from copy import deepcopy
from itertools import product
import json
from pathlib import Path
from types import SimpleNamespace
import pytest
from tools.splendorust_network_live_20261031 import common as c,design as d,analysis as a

ROOT=Path(__file__).resolve().parents[1]
SAVED=ROOT/'outputs/splendorust_search128_strength_20261028_01/games/main/p016_g33'


def read(path):return json.loads(path.read_text())


def test_schedule_three_unique_setups_swaps_seats_and_preserves_science():
    rows=d.make_schedule();assert len(rows)==6 and d.validate_schedule(rows)==rows
    assert [s['game_index'] for s in rows]==list(range(6))
    assert len({s['seed_decimal'] for s in rows})==3
    assert [s['challenger_seat'] for s in rows]==[0,1,1,0,0,1]
    assert d.DOMAIN=='splendorust_network_live_20261031_v1'
    for pair in range(3):
        left,right=rows[pair*2:pair*2+2]
        assert left['seed_decimal']==right['seed_decimal'] and left['challenger_seat']!=right['challenger_seat']
        assert d.initial_snapshot(left)==d.initial_snapshot(right)
        for role in ('challenger','opponent'):assert left[role+'_quota']==256 and left[role+'_cpuct']==.4 and left[role+'_mode']=='research'


@pytest.mark.parametrize('field,value',[('seed_decimal','1'),('game_index',True),('challenger_quota',128),
    ('opponent_variant','candidate'),('challenger_seat',1),('stage','main'),('pair_index',3)])
def test_schedule_rejects_any_changed_assignment(field,value):
    row=d.make_schedule()[0];row[field]=value
    with pytest.raises(ValueError):d.validate_spec(row)


def test_schedule_rejects_partial_duplicate_or_extra_keys():
    rows=d.make_schedule()
    for wrong in (rows[:-1],rows[::-1],rows+[rows[0]]):
        with pytest.raises(ValueError):d.validate_schedule(wrong)
    rows[0]['extra']=1
    with pytest.raises(ValueError):d.validate_schedule(rows)


def test_lifecycle_cost_ceiling_and_identical_startup():
    values=d.lifecycle_inventory();assert set(values)==set(c.LIFECYCLES) and len(values)==4
    assert c.MAX_PLIES==2000 and c.MAX_LIVE_DECISIONS==12000
    for name,item in values.items():
        assert item['execution_variant']==name.split('_')[1] and item['route']=='research'
        assert item['startup_requests']==3 and item['ordinary_first_request_id']=='request-00000004'
        assert item['ordinary_requests']==(6000 if name.startswith('live_') else 48)


def selector_rows():
    rows=[]
    for spec in d.make_schedule():
        for variant,part in product(d.VARIANTS,range(3)):
            role='challenger' if variant=='candidate' else 'opponent'
            seat=spec['challenger_seat'] if role=='challenger' else 1-spec['challenger_seat']
            for private in range(4):
                players=[dict(reserves=[]),dict(reserves=[])]
                if private&1:players[seat]['reserves']=[dict(source='deck')]
                if private&2:players[1-seat]['reserves']=[dict(source='deck')]
                # Unique identity and fixed phase, without any outcome fields.
                ply=part*20+seat+private*2
                rows.append(dict(spec,ply=ply,variant=variant,seat=seat,player_role=role,quota=256,arm='search',
                    route='research',mode='research',completed=True,
                    observer=dict(observation=dict(current_player=seat,players=players))))
    return rows


def test_selector_covers_each_variant_game_phase_and_is_outcome_timing_blind():
    rows=selector_rows();sample=d.select_replay_sample(rows)
    assert sample['primary_selected']==36 and 36<=len(sample['decisions'])<=48
    assert not sample['empty_primary_cells'] and sample['secondary_selected']<=12
    assert len({(x['game_index'],x['variant'],x['phase']) for x in sample['decisions']})==36
    assert len({(x['game_index'],x['ply']) for x in sample['decisions']})==len(sample['decisions'])
    changed=deepcopy(rows)
    for row in changed:row.update(winner=1,clock={'receipt_elapsed_seconds':999},action='ignored',trace='ignored')
    assert d.select_replay_sample(reversed(changed))==sample
    assert sample['selection_uses_outcomes'] is sample['selection_uses_timing'] is False


def test_selector_fails_duplicates_wrong_variants_caps_and_preserves_failed_exclusion():
    rows=selector_rows()
    with pytest.raises(ValueError):d.select_replay_sample(rows+rows[:1])
    with pytest.raises(ValueError):d.select_replay_sample(rows,49)
    wrong=deepcopy(rows);wrong[0]['variant']='control' if wrong[0]['variant']=='candidate' else 'candidate'
    with pytest.raises(ValueError):d.select_replay_sample(wrong)
    failed=deepcopy(rows);failed[0]['completed']=False
    assert d.select_replay_sample(failed)['eligible_completed_decisions']==len(rows)-1


def test_seed_inventory_reports_scope_and_rejects_registered_reuse(tmp_path,monkeypatch):
    out=tmp_path/'outputs/new';out.mkdir(parents=True)
    prior=tmp_path/'outputs/prior/schedule.json';prior.parent.mkdir();prior.write_text('[123]')
    monkeypatch.setattr(c,'ROOT',tmp_path);monkeypatch.setattr(c,'OUT',out)
    monkeypatch.setattr(d.subprocess,'run',lambda *args,**kwargs:SimpleNamespace(stdout='outputs/prior/schedule.json\n'))
    value=d.seed_inventory();assert value['status']=='passed' and value['new_setup_count']==3 and not value['prior_seed_reuse']
    prior.write_text(json.dumps([int(d.expected_seed(0))]))
    with pytest.raises(ValueError,match='collision'):d.seed_inventory()


@pytest.fixture(scope='module')
def saved_game():
    return read(SAVED/'initial.json')['initial'],[read(x) for x in sorted((SAVED/'moves').glob('*.json'))],read(SAVED/'result.json')


def ledger_fixture(saved,monkeypatch,game_index=0):
    initial,rows,ending=deepcopy(saved);spec=d.make_schedule()[game_index]
    # Historical fixture keeps its seats while deliberately replacing study metadata.
    spec=next(s for s in d.make_schedule() if s['challenger_seat']==ending['challenger_seat']) if game_index==0 else spec
    monkeypatch.setattr(d,'initial_snapshot',lambda value:deepcopy(initial))
    monkeypatch.setattr(a.scientific,'validate_result',lambda *args,**kwargs:None)
    monkeypatch.setattr(a.scientific,'audit_decision',lambda *args,**kwargs:None)
    ending.update(spec)
    for row in rows:
        row.update(spec);seat=row['before']['state']['current_player'];role='challenger' if seat==spec['challenger_seat'] else 'opponent'
        row.update(variant=spec[role+'_variant'],player_role=role,arm='search',quota=256,route='research',mode='research',seat=seat)
    return initial,rows,ending,spec


def test_independent_dictionary_replays_saved_corrected_complete_game(saved_game,monkeypatch):
    initial,rows,ending,spec=ledger_fixture(saved_game,monkeypatch)
    result=a.audit_game(initial,rows,ending,spec)
    assert result['transitions']==54 and result['failed_moves']==0 and result['natural_terminal']


@pytest.mark.parametrize('fault',['transition','winner','variant','seat','ply','seed','omitted','deadline','late_as_forfeit','nonterminal','cap_draw'])
def test_game_audit_rejects_changed_or_incomplete_game(saved_game,monkeypatch,fault):
    initial,rows,ending,spec=ledger_fixture(saved_game,monkeypatch)
    if fault=='transition':rows[0]['after']['state']['bank_tokens']['gold']+=1
    elif fault=='winner':ending['winner']=1-ending['winner']
    elif fault=='variant':rows[0]['variant']='control' if rows[0]['variant']=='candidate' else 'candidate'
    elif fault=='seat':rows[0]['seat']=1-rows[0]['seat']
    elif fault=='ply':rows[1]['ply']=0
    elif fault=='seed':rows[0]['seed_decimal']='1'
    elif fault=='omitted':rows.pop(2)
    elif fault=='deadline':rows[0]['clock']['deadline_seconds']=9.9
    elif fault=='late_as_forfeit':ending.update(reason='forfeit',administrative_forfeit=True)
    elif fault in ('nonterminal','cap_draw'):
        rows=rows[:1];ending.update(plies=1,final=rows[0]['after'],winner=None)
        if fault=='cap_draw':monkeypatch.setattr(c,'MAX_PLIES',1)
    with pytest.raises(ValueError):a.audit_game(initial,rows,ending,spec)


def test_failure_only_exact_hard_guard_no_infrastructure_relabel(saved_game,monkeypatch):
    initial,rows,ending,spec=ledger_fixture(saved_game,monkeypatch);row=rows[0];seat=row['seat']
    row.update(completed=False,action=None,trace=None,after=None)
    row['clock'].update(status='timeout',result=None,accepted=False,received_before_deadline=False,
        deadline_seconds=10.,elapsed_seconds=10.01,receipt_elapsed_seconds=None)
    ending.update(plies=1,final=initial,reason='forfeit',winner=1-seat,loser=seat,failed_seat=seat,
        failed_role=row['player_role'],administrative_forfeit=True,failure='research_outer_hard_guard',
        failure_policy='research_hard_guard_9s_internal_10s_complete_response')
    assert a.audit_game(initial,[row],ending,spec)['failed_moves']==1
    row['clock'].update(status='worker_error',detail='source changed')
    with pytest.raises(ValueError):a.audit_game(initial,[row],ending,spec)


def actual_replay_fixture():
    row=read(c.PRACTICAL_OUT/'observations/0000.json');spec=d.make_schedule()[0]
    row.update(spec,ply=0,player_role='challenger' if row['variant']=='candidate' else 'opponent')
    selection={k:row[k] for k in ('game_index','pair_index','ply','variant','player_role','seat')}
    selection['observer_sha256']=c.digest(row['observer'])
    return row,deepcopy(row),selection


def test_actual_saved256_same_variant_complete_result_replay_no_stubs():
    original,replayed,selected=actual_replay_fixture()
    result=a.verify_replay(original,replayed,selected)
    assert result['full_scientific_output_exact'] and not result['cross_variant_equality_required']


@pytest.mark.parametrize('fault',['variant','action','trace','before','after','observer','selection','result','clock'])
def test_same_variant_replay_rejects_mutations(fault):
    original,replayed,selected=actual_replay_fixture()
    if fault=='variant':replayed['variant']='other'
    elif fault in ('action','trace','before','after','observer'):replayed[fault]={}
    elif fault=='selection':selected['game_index']=5
    elif fault=='result':replayed['clock']['result']['legal_count']+=1
    else:replayed['clock']['deadline_seconds']=9.
    with pytest.raises(ValueError):a.verify_replay(original,replayed,selected)


def test_distribution_is_descriptive_and_retains_late_completed_times():
    result=a.distribution([.5,.7,1.,1.5,2.])
    assert result['count']==5 and result['completed_over_one_second']==2 and result['median']==1.
    assert result['maximum']==2 and a.distribution([])['count']==0
    for wrong in ([float('nan')],[11.]):
        with pytest.raises(ValueError):a.distribution(wrong)
