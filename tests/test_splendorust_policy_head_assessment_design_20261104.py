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
from tools.splendorust_policy_head_20261104.assessment import common as c,design as d,analysis as a

ROOT=Path(__file__).resolve().parents[1]
SAVED=ROOT/'outputs/splendorust_search128_strength_20261028_01/games/main/p016_g33'


def read(path):return json.loads(path.read_text())


def test_exact_workload_counterbalance_and_original_science():
    ops,main=d.make_schedule('operational'),d.make_schedule('main')
    assert len(ops)==6 and len(main)==400 and d.validate_schedule(ops+main)==ops+main
    assert a.REPLICATES==50000 and a.SEED==d.BOOTSTRAP_SEED==20261104128
    assert len({s['seed_decimal'] for s in ops+main})==203
    assert not {s['seed_decimal'] for s in ops}&{s['seed_decimal'] for s in main}
    for stage,rows in [('operational',ops),('main',main)]:
        assert d.validate_schedule(rows,stage)==rows
        for pair in range(len(rows)//2):
            left,right=rows[2*pair:2*pair+2]
            assert left['seed_decimal']==right['seed_decimal']
            assert (left['challenger_seat'],right['challenger_seat'])==(pair%2,1-pair%2)
            for row in (left,right):
                assert (row['challenger_quota'],row['opponent_quota'])==(128,128)
                assert row['challenger_cpuct']==row['opponent_cpuct']==.4
                assert row['challenger_mode']==row['challenger_variant']=='candidate'
                assert row['opponent_mode']==row['opponent_variant']=='control'
        for pair in (0,len(rows)//2-1):assert d.initial_snapshot(rows[2*pair])==d.initial_snapshot(rows[2*pair+1])


def test_fresh_domain_disjoint_prior_main_and_operational_schedules():
    fresh={s['seed_decimal'] for s in d.make_schedule()}
    for name in ('splendorust_search256_strength_20261029_02','splendorust_network_live_20261031_01'):
        prior=read(ROOT/'outputs'/name/'schedule.json')
        if type(prior) is dict:prior=prior['operational']+prior['main']
        assert not fresh&{s['seed_decimal'] for s in prior}
    for attempt in ('01','02'):
        prior=read(ROOT/f'outputs/splendorust_exploration128_20261029_{attempt}/schedule.json')
        old={s['seed_decimal'] for stage in ('operational','development') for s in prior[stage]}
        old|={s['seed_decimal'] for branch in prior['confirmation'].values() for s in branch}
        assert len(old)==253 and not fresh&old


@pytest.mark.parametrize('field,value',[('seed_decimal','1'),('game_index',True),('challenger_quota',256),
    ('opponent_variant','candidate'),('challenger_seat',1),('stage','development'),('pair_index',201),
    ('challenger_cpuct',.8),('opponent_mode','research')])
def test_changed_schedule_identity_rejected(field,value):
    row=d.make_schedule('main')[0];row[field]=value
    with pytest.raises(ValueError):d.validate_spec(row)


def test_schedule_rejects_partial_reordered_duplicate_and_extra_fields():
    rows=d.make_schedule()
    for wrong in (rows[:-1],rows[::-1],rows+[rows[0]]):
        with pytest.raises(ValueError):d.validate_schedule(wrong)
    rows[0]['extra']=1
    with pytest.raises(ValueError):d.validate_schedule(rows)


def selector_rows():
    for spec in d.make_schedule():
        for variant,part,private in product(d.VARIANTS,range(3),range(4)):
            role='challenger' if variant=='candidate' else 'opponent'
            seat=spec['challenger_seat'] if role=='challenger' else 1-spec['challenger_seat']
            players=[dict(reserves=[]),dict(reserves=[])]
            if private&1:players[seat]['reserves']=[dict(source='deck')]
            if private&2:players[1-seat]['reserves']=[dict(source='deck')]
            yield dict(spec,ply=part*20+seat+private*2,variant=variant,seat=seat,player_role=role,
                quota=spec[role+'_quota'],arm='search',route=variant,mode=variant,completed=True,
                observer=dict(observation=dict(current_player=seat,players=players)))


def test_replay_preserves_old48_private_cells_adds_ops12_temporal36_and_is_blind():
    sample=d.select_replay_sample(selector_rows())
    assert {name:item['available_cells'] for name,item in sample['coverage'].items()}==dict(
        main_primary=48,operational=12,main_temporal=36)
    assert 48<=len(sample['decisions'])<=96
    chosen={(r['stage'],r['game_index'],r['ply']) for r in sample['decisions']}
    assert len(chosen)==len(sample['decisions'])
    for group in sample['coverage'].values():
        assert not group['empty_cells']
        assert all((r['stage'],r['game_index'],r['ply']) in chosen for r in group['cells'])
    changed=list(selector_rows())
    for row in changed:row.update(winner=1,clock={'receipt_elapsed_seconds':999},action='ignored',trace='ignored')
    assert d.select_replay_sample(reversed(changed))==sample
    assert sample['selection_uses_outcomes'] is sample['selection_uses_timing'] is False
    assert d.temporal_block(66)==0 and d.temporal_block(67)==1 and d.temporal_block(133)==1 and d.temporal_block(134)==2


def test_replay_empty_cells_failed_exclusion_and_identity_rejection():
    rows=list(selector_rows())[:2]
    rows[0]['completed']=False
    sample=d.select_replay_sample(rows)
    assert sample['excluded_failed_decisions']==1 and sample['eligible_completed_decisions']==1
    assert sample['coverage']['main_primary']['available_cells']==0
    assert len(sample['decisions'])==1 and sample['decisions'][0]['stage']=='operational'
    with pytest.raises(ValueError):d.select_replay_sample(rows+rows[:1])
    with pytest.raises(ValueError):d.select_replay_sample(rows,48)
    rows[1]['quota']=256
    if rows[1]['variant']=='control':rows[1]['quota']=512
    with pytest.raises(ValueError):d.select_replay_sample(rows)


def test_seed_inventory_explicit_scope_and_both_seed_kinds(tmp_path,monkeypatch):
    out=tmp_path/'outputs/new';out.mkdir(parents=True)
    prior=tmp_path/'outputs/prior/schedule.json';prior.parent.mkdir();prior.write_text('[123]')
    old=tmp_path/'outputs/old_pilot_schedule.json';old.write_text(json.dumps([{'seed_decimal':str(50000+i//2)} for i in range(406)]))
    monkeypatch.setattr(d,'PRIOR_ASSESSMENT_SCHEDULE',old)
    monkeypatch.setattr(c,'ROOT',tmp_path);monkeypatch.setattr(c.parent,'ROOT',tmp_path);monkeypatch.setattr(c,'OUT',out);monkeypatch.setattr(c.parent,'OUT',out)
    monkeypatch.setattr(d.subprocess,'run',lambda *args,**kwargs:SimpleNamespace(stdout='outputs/prior/schedule.json\n'))
    value=d.seed_inventory();assert value['status']=='passed' and value['new_setup_count']==203 and not value['prior_seed_reuse']
    for collision in (int(d.expected_seed('main',0)),d.BOOTSTRAP_SEED):
        prior.write_text(json.dumps([collision]))
        with pytest.raises(ValueError,match='collision'):d.seed_inventory()


@pytest.fixture(scope='module')
def saved_game():
    return read(SAVED/'initial.json')['initial'],[read(x) for x in sorted((SAVED/'moves').glob('*.json'))],read(SAVED/'result.json')


def ledger_fixture(saved,monkeypatch,game_index=0):
    initial,rows,ending=deepcopy(saved);spec=d.make_schedule('main')[game_index]
    # Historical fixture keeps its seats while deliberately replacing study metadata.
    spec=next(s for s in d.make_schedule('main') if s['challenger_seat']==ending['challenger_seat']) if game_index==0 else spec
    monkeypatch.setattr(d,'initial_snapshot',lambda value:deepcopy(initial))
    monkeypatch.setattr(a.scientific,'validate_result',lambda *args,**kwargs:None)
    monkeypatch.setattr(a.scientific,'audit_decision',lambda *args,**kwargs:None)
    ending.update(spec)
    for row in rows:
        row.update(spec);seat=row['before']['state']['current_player'];role='challenger' if seat==spec['challenger_seat'] else 'opponent'
        row.update(variant=spec[role+'_variant'],player_role=role,arm='search',quota=spec[role+'_quota'],route=spec[role+'_variant'],mode=spec[role+'_variant'],seat=seat)
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
    row=read(SAVED/'moves/0001.json');spec=d.make_schedule('main')[0]
    row.update(variant='control',route='control',mode='control',quota=128,arm='search')
    row.update(spec,ply=0,player_role='challenger' if row['variant']=='candidate' else 'opponent')
    selection={k:row[k] for k in ('stage','game_index','pair_index','ply','variant','player_role','seat')}
    selection['observer_sha256']=c.digest(row['observer'])
    return row,deepcopy(row),selection


def test_actual_saved128_same_variant_complete_result_replay_no_stubs():
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


def test_streaming_timing_keeps_operational_and_main_separate(saved_game,monkeypatch):
    _,rows,_,spec=ledger_fixture(saved_game,monkeypatch)
    observed=a.summarize_timing(iter(rows))
    assert observed['decisions']==54 and observed['failed_decisions']==0
    assert sum(observed['stages']['main'][v]['count'] for v in c.VARIANTS)==54
    assert sum(observed['stages']['operational'][v]['count'] for v in c.VARIANTS)==0
    assert observed['timing_used_for_strength_selection'] is False
    assert observed['one_second_is_descriptive'] and observed['seven_hundred_ms_is_descriptive']
    with pytest.raises(ValueError):a.summarize_timing(iter(rows+rows[:1]))
    rows[0]['quota']=256 if rows[0]['quota']==512 else 512
    with pytest.raises(ValueError):a.summarize_timing(iter(rows))


def test_operational_six_game_qualification_streamed_and_never_strength(saved_game,monkeypatch):
    games=[];all_rows=[]
    for spec in d.make_schedule('operational'):
        initial,rows,ending,_=ledger_fixture(saved_game,monkeypatch)
        ending.update(spec)
        for row in rows:
            row.update(spec);role='challenger' if row['seat']==spec['challenger_seat'] else 'opponent'
            variant=spec[role+'_variant']
            row.update(player_role=role,variant=variant,route=variant,mode=variant,quota=spec[role+'_quota'])
        games.append(ending);all_rows.extend(rows)
    result=a.summarize_operational(games,iter(all_rows),d.make_schedule('operational'))
    assert result['operational_games_passed'] and result['completed_games']==6 and result['decisions']==324
    assert not result['strength_test'] and not result['reference_promoted']
    assert sum(result['descriptive_wins'].values())==6
    with pytest.raises(ValueError):a.summarize_operational(games,iter(all_rows[54:]))
    with pytest.raises(ValueError):a.summarize_operational(games,iter(all_rows[::-1]))
    games[0]['reason']='forfeit'
    with pytest.raises(ValueError):a.summarize_operational(games,iter(all_rows))


@pytest.fixture(autouse=True)
def pilot_scope(monkeypatch):
    monkeypatch.setattr(c,'SCOPE','pilot');monkeypatch.setattr(c,'MAIN_GAMES',400);monkeypatch.setattr(c,'MAX_REPLAY_DECISIONS',96)
    monkeypatch.setattr(d,'PAIRS',{'operational':3,'main':200})
    domain='splendorust_policy_head_20261104_pilot_assessment_v1'
    monkeypatch.setattr(d,'DOMAIN',domain);monkeypatch.setattr(d,'REPLAY_DOMAIN',domain+'_replay_v1')




def test_explicit_previous_pilot_sample_is_fresh():
    prior=c.read(d.PRIOR_ASSESSMENT_SCHEDULE)
    assert len(prior)==406
    assert not {x['seed_decimal'] for x in prior}&{x['seed_decimal'] for x in d.make_schedule()}
    assert d.DOMAIN=='splendorust_policy_head_20261104_pilot_assessment_v1'


def test_only_pilot_scope_is_admitted():
    import subprocess,sys,os
    code='from tools.splendorust_policy_head_20261104 import common'
    result=subprocess.run([sys.executable,'-B','-c',code],cwd=ROOT,
        env={**os.environ,'SELFPLAY_JOB_SCOPE':'qualification'},capture_output=True,text=True,timeout=20)
    assert result.returncode!=0 and 'source collection is disabled' in result.stderr
