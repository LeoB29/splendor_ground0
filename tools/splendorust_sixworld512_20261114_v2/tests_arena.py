"""Model-free negative tests for schedule, custody, score and replay boundaries."""
from copy import deepcopy
import gzip
from pathlib import Path
import pytest
from . import common as c, schedule, arena, verifier, adapters
from tools.external_strength_screen_20261018 import rules


def clock(result=None,*,status='ok',request_id='request-00000001',detail=None):
    accepted=status=='ok'
    timeout=status=='timeout'
    return dict(clock='perf_counter_ns',clock_implementation='QueryPerformanceCounter()',request_id=request_id,
        deadline_seconds=120.,elapsed_seconds=120.01 if timeout else 1.1,
        receipt_elapsed_seconds=None if timeout else 1.,received_before_deadline=not timeout,
        accepted=accepted,status=status,result=result,detail=detail,telemetry={})


@pytest.fixture
def setup(tmp_path,monkeypatch):
    monkeypatch.setattr(c,'ROOT',tmp_path)
    monkeypatch.setattr(c,'OUT',tmp_path/'attempt')
    from . import qualification
    monkeypatch.setattr(qualification,'validate_live_result',lambda result,observer,pool,plan:{'status':'passed'})
    spec=schedule.expected_spec('main',0,0)
    initial=rules.serialize_snapshot(rules.CorrectedEnv(seed=int(spec['seed_decimal'])).initial_state())
    item=c.save_gzip(c.OUT/'initial.json.gz',initial)
    initials=dict(schema='sixworld512_initial_setup_inventory_v1',domain=schedule.DOMAIN,snapshots={
        'main:0':dict(artifact=item,snapshot_sha256=c.digest(initial),seed_decimal=spec['seed_decimal'])})
    plan={'initials':c.save(c.OUT/'initials.json',initials)}
    folder=c.OUT/'block/games/g0000';folder.mkdir(parents=True)
    controller=dict(identity={'pid':111,'created':22.},plan_sha256=c.digest(plan))
    c.save(folder.parent.parent/'controller.json',controller)
    return dict(spec=spec,initial=initial,initial_item=item,plan=plan,folder=folder,controller=controller)


def write_game(fixture,*,mutate_events=None,mutate_ending=None,bad_source=False):
    spec=fixture['spec'];folder=fixture['folder'];before=fixture['initial'];events=[];previous=None
    state=rules.deserialize_snapshot(before)
    action=rules.CorrectedEnv().legal_actions(state)[0]
    serialized=rules.serialize_action(action)
    after=rules.serialize_snapshot(rules.CorrectedEnv().step(state,action))
    packed=adapters.pack_result(dict(action=serialized,callback_bits=['original']))
    if bad_source:packed['raw_sha256']='0'*64
    source1=c.save(c.OUT/'transport/request-00000001.json',clock(packed))
    source2=c.save(c.OUT/'transport/request-00000002.json',clock(status='timeout',request_id='request-00000002'))
    def add(kind,data):
        nonlocal previous
        row=dict(sequence=len(events),previous_sha256=previous,kind=kind,data=data)
        events.append(row);previous=c.digest(row);return row
    for ply,current,source in ((0,before,source1),(1,after,source2)):
        seat=current['state']['current_player'];pool=6 if seat==spec['candidate_seat'] else 3
        d=add('dispatch',dict(spec=spec,ply=ply,seat=seat,pool=pool,before=current,
            observer=rules.observation_payload(rules.deserialize_snapshot(current)),deadline_seconds=120,
            attempt=1,time='fixed',dispatched_monotonic_ns=100+ply,identity=fixture['controller']['identity']))
        add('response',dict(ply=ply,source=source,dispatch_sha256=c.digest(d),time='fixed'))
        add('transition',dict(ply=ply,source=source,completed=ply==0,
            action=serialized if ply==0 else None,after=after if ply==0 else None,
            failure=None if ply==0 else 'external_120_second_guard'))
    if mutate_events:mutate_events(events)
    journal=folder/'events.jsonl.gz'
    with journal.open('xb') as f:
        for row in events:f.write(gzip.compress((c.canonical(row)+'\n').encode(),mtime=0))
    failed_seat=after['state']['current_player'];winner=1-failed_seat
    failure=dict(kind='external_120_second_guard',seat=failed_seat,
                 pool=6 if failed_seat==spec['candidate_seat'] else 3,source=source2)
    ending=dict(spec=spec,complete=True,initial=fixture['initial_item'],journal=c.artifact(journal),
        plies=2,reason='forfeit',winner=winner,failure=failure,final=after,
        candidate_points2=2 if winner==spec['candidate_seat'] else 0)
    if mutate_ending:mutate_ending(ending)
    c.save(folder/'result.json',ending)
    return events,ending


def rebuild_chain(events):
    previous=None
    for i,row in enumerate(events):
        row['sequence']=i;row['previous_sha256']=previous
        if row['kind']=='response':row['data']['dispatch_sha256']=c.digest(events[i-1])
        previous=c.digest(row)


def test_complete_schedule_fresh_pairs_and_seats():
    schedules=schedule.build_schedule()
    assert len(schedules['main'])==3600 and len(schedules['operational'])==6
    seeds={s['seed_decimal'] for rows in schedules.values() for s in rows}
    assert len(seeds)==1803
    for rows in schedules.values():
        for first,second in zip(rows[::2],rows[1::2],strict=True):
            assert first['seed_decimal']==second['seed_decimal']
            assert first['candidate_seat']==1-second['candidate_seat']


@pytest.mark.parametrize('change',[lambda s:s.update(candidate_pool=3),lambda s:s.update(total_simulations_each=256),
    lambda s:s.update(game_index=True),lambda s:s.update(seed_decimal='1')])
def test_changed_schedule_rejected(change):
    spec=schedule.expected_spec('main',0,0);change(spec)
    with pytest.raises(ValueError):schedule.validate_spec(spec)


def test_incomplete_or_reordered_interim_block_rejected():
    rows=schedule.stage_schedule('main')[:400]
    assert schedule.validate_block(rows)=='main'
    for bad in (rows[:399],rows[1:]+rows[:1],rows[:398]+[rows[398],rows[398]]):
        with pytest.raises(ValueError):schedule.validate_block(bad)


def test_valid_independent_transition_and_timeout_forfeit(setup):
    write_game(setup)
    result=verifier.verify_game(setup['folder'],setup['plan'],setup['spec'])
    assert result['plies']==2 and result['failed_decisions']==1 and len(result['replay_candidates'])==1


def test_original_compressed_clock_tampering_rejected(setup):
    write_game(setup,bad_source=True)
    with pytest.raises(ValueError):verifier.verify_game(setup['folder'],setup['plan'],setup['spec'])


def test_changed_hidden_prestate_rejected_even_with_rehashed_journal(setup):
    def change(events):
        events[3]['data']['before']['state']['bank_tokens']['white']+=1;rebuild_chain(events)
    write_game(setup,mutate_events=change)
    with pytest.raises(ValueError):verifier.verify_game(setup['folder'],setup['plan'],setup['spec'])


def test_extra_move_after_forfeit_rejected(setup):
    def change(events):events.append(deepcopy(events[0]));rebuild_chain(events)
    write_game(setup,mutate_events=change)
    with pytest.raises(ValueError):verifier.verify_game(setup['folder'],setup['plan'],setup['spec'])


def test_score_cannot_be_relabelled_after_timeout(setup):
    write_game(setup,mutate_ending=lambda e:e.update(candidate_points2=1))
    with pytest.raises(ValueError):verifier.verify_game(setup['folder'],setup['plan'],setup['spec'])


def test_wrong_dispatch_owner_rejected(setup):
    def change(events):events[0]['data']['identity']['created']=999;rebuild_chain(events)
    write_game(setup,mutate_events=change)
    with pytest.raises(ValueError):verifier.verify_game(setup['folder'],setup['plan'],setup['spec'])


@pytest.mark.parametrize('status,detail',[('worker_error','resource guard failed'),('worker_exit','exit124'),
    ('worker_error','catalogue source changed VARIABILITY_INTERNAL_DEADLINE')])
def test_technical_failure_never_becomes_game_loss(status,detail):
    with pytest.raises(ValueError):verifier.classify_deadline(clock(status=status,detail=detail))


def test_exact_internal_deadline_is_retained_as_forfeit():
    record=clock(status='worker_error',detail='VARIABILITY_INTERNAL_DEADLINE:119s')
    record['elapsed_seconds']=119.1;record['receipt_elapsed_seconds']=119.05
    assert verifier.classify_deadline(record)=='internal_119_second_guard'


def test_fake_early_internal_deadline_marker_is_not_forfeit():
    with pytest.raises(ValueError):verifier.classify_deadline(clock(status='worker_error',detail='VARIABILITY_INTERNAL_DEADLINE:119s'))


def test_durable_watchdog_requires_exact_request_service_and_freeze(setup,monkeypatch):
    monkeypatch.setattr(c,'PLAN',c.OUT/'freeze.json');c.save(c.PLAN,setup['plan'])
    folder=c.OUT/'watchdog/transport';folder.mkdir(parents=True)
    ident={'pid':987,'created':123.0}
    c.save(folder/'initialization.json',{'result':{'identity':ident,'freeze_sha256':c.sha(c.PLAN)}})
    record=clock(status='worker_exit');record['elapsed_seconds']=120.1
    source=c.save(folder/'request-00000001.json',record)
    proof=dict(request_id=record['request_id'],identity=ident,freeze_sha256=c.sha(c.PLAN),
        reason='sixworld512_decision_watchdog_120s',deadline_seconds=120.,elapsed_seconds=120.01)
    item=c.save(c.OUT/'watchdog/decision_timeout.json',proof)
    assert verifier.classify_deadline(record,timeout_evidence=item,source=source)=='service_120_second_watchdog'
    proof['request_id']='request-00000002';wrong=c.save(c.OUT/'wrong_watchdog.json',proof)
    with pytest.raises(ValueError):verifier.classify_deadline(record,timeout_evidence=wrong,source=source)


def test_stage_clamped_timeout_is_not_player_loss():
    record=clock(status='timeout');record['deadline_seconds']=20
    with pytest.raises(ValueError):verifier.classify_deadline(record)


def test_durable_journal_rejects_truncated_tail(setup):
    _,ending=write_game(setup)
    path=c.resolve(ending['journal']);raw=path.read_bytes();path.write_bytes(raw[:-6])
    newitem=c.artifact(path)
    with pytest.raises((EOFError,OSError,ValueError)):list(verifier.journal_rows(newitem))


def test_durable_dispatch_file_cannot_be_reopened(setup):
    path=setup['folder']/'exclusive.jsonl.gz';journal=arena.Journal(path)
    journal.append('dispatch',{'attempt':1});journal.close()
    with pytest.raises(FileExistsError):arena.Journal(path)


def test_replay_rejects_changed_callback_even_with_same_action(setup):
    write_game(setup);game=verifier.verify_game(setup['folder'],setup['plan'],setup['spec'])
    chosen=dict(game['replay_candidates'][0],replay_id='fixed')
    original=verifier.read_decision_source(chosen['source'])
    result=deepcopy(original['result']);result['callback_bits']=['changed']
    source=c.save(c.OUT/'replay/request-00000001.json',clock(adapters.pack_result(result)))
    with pytest.raises(ValueError):verifier.verify_replays([chosen],[{'replay_id':'fixed','source':source}],setup['plan'])


def test_replay_selection_fixed_pool_and_stage_coverage():
    rows=[]
    for pool in (3,6):
        for ply in (0,2,4,24,26,28):
            rows.append(dict(spec=schedule.expected_spec('main',0,0),pool=pool,ply=ply,source={},observer={}))
    selected=verifier.select_replays([{'replay_candidates':rows}],'main')
    assert len(selected)==8
    assert {(r['pool'],r['ply']>=24) for r in selected}=={(p,s) for p in (3,6) for s in (False,True)}
    assert selected==verifier.select_replays([{'replay_candidates':list(reversed(rows))}],'main')


def test_pair_score_draw_credit_and_missing_game_rejected(setup,monkeypatch):
    folder=c.OUT/'testblock';(folder/'games').mkdir(parents=True)
    specs=schedule.stage_schedule('main')[:400]
    for spec in specs:(folder/'games'/f"g{spec['game_index']:04d}").mkdir()
    points=[2,0,1,1,2,2,0,0]
    def fake(folder,plan,spec,**kwargs):
        index=spec['game_index']
        rows=[dict(spec=spec,ply=ply,pool=pool,source={},observer={}) for pool in (3,6) for ply in (0,2,24,26)]
        return dict(spec=spec,candidate_points2=points[index%8],game={},failed_decisions=0,
                    timings={'3':[1.],'6':[1.]},replay_candidates=rows,decision_sources=[f'game{index}/request1'])
    monkeypatch.setattr(verifier,'verify_game',fake)
    result=verifier.verify_block(folder,setup['plan'],specs)
    assert result['pair_scores'][:4]==[.5,.5,1.,0.]
    assert len(result['pair_scores'])==200
    def reused(folder,plan,spec,**kwargs):
        row=fake(folder,plan,spec,**kwargs);row['decision_sources']=['shared/request1'];return row
    monkeypatch.setattr(verifier,'verify_game',reused)
    with pytest.raises(ValueError,match='reused across games'):verifier.verify_block(folder,setup['plan'],specs)
    monkeypatch.setattr(verifier,'verify_game',fake)
    (folder/'games/g0399').rmdir()
    with pytest.raises(ValueError):verifier.verify_block(folder,setup['plan'],specs)


def test_journal_cannot_write_outside_attempt(setup):
    outside=c.OUT.parent/'outside.jsonl.gz'
    with pytest.raises(ValueError):arena.Journal(outside)
    assert not outside.exists()


def test_boolean_winner_rejected(setup):
    write_game(setup,mutate_ending=lambda e:e.update(winner=False))
    with pytest.raises(ValueError):verifier.verify_game(setup['folder'],setup['plan'],setup['spec'])


def test_live_runner_persists_dispatch_before_callback_and_source_before_failure(setup,monkeypatch):
    from . import qualification
    block=c.OUT/'integration';folder=block/'games/g0000'
    c.save(block/'controller.json',dict(identity=c.identity(),plan_sha256=c.digest(setup['plan'])))
    state=rules.deserialize_snapshot(setup['initial'])
    action=rules.serialize_action(rules.CorrectedEnv().legal_actions(state)[0])
    calls=[]
    def callback(observer,pool,*,context):
        rows=list(verifier.journal_rows(c.artifact(folder/'events.jsonl.gz')))
        assert rows[-1]['kind']=='dispatch' and rows[-1]['data']['before']==setup['initial']
        assert rows[-1]['data']['observer']==observer and context['ply']==0
        calls.append(context)
        result={'action':action,'callback_bits':['fixture']}
        source=c.save(c.OUT/'integration/transport/request-00000001.json',clock(adapters.pack_result(result)))
        return {'source':source,'clock':clock(result)}
    def rejection(*args,**kwargs):raise ValueError('saved failure fixture')
    monkeypatch.setattr(qualification,'validate_live_result',rejection)
    with pytest.raises(ValueError,match='saved failure fixture'):
        arena.play_game(folder,setup['plan'],setup['spec'],callback,lambda:None)
    rows=list(verifier.journal_rows(c.artifact(folder/'events.jsonl.gz')))
    assert [r['kind'] for r in rows]==['dispatch','response']
    assert len(calls)==1 and (folder/'failure.json').exists() and not (folder/'result.json').exists()
    assert verifier.read_decision_source(rows[-1]['data']['source'])['result']['action']==action
