from copy import deepcopy
from pathlib import Path
import ast,json,os,subprocess,sys
import pytest
from tools.splendorust_exploration128_20261029 import common as c,design,schedule,runner,preparation,independent_review,guardian

def test_schedule_common_paired_interleaved_and_disjoint():
    schedules={stage:schedule.make_schedule(stage,'low' if stage=='confirmation' else None) for stage in schedule.PAIRS}
    assert [len(schedules[x]) for x in schedule.PAIRS]==[12,200,400]
    for stage,rows in schedules.items():
        assert len({runner.game_folder(x) for x in rows})==len(rows)
        assert all(schedule.validate_spec(x)==x for x in rows)
        for pair in range(schedule.PAIRS[stage]):
            block=[r for r in rows if r['pair_index']==pair]
            assert len({r['seed_decimal'] for r in block})==1
            for arm in set(r['candidate'] for r in block):assert {r['challenger_seat'] for r in block if r['candidate']==arm}=={0,1}
    assert schedules['development'][0]['candidate']=='low'
    assert schedules['development'][4]['candidate']=='high'
    sets=[{r['seed_decimal'] for r in rows} for rows in schedules.values()]
    assert all(not sets[a]&sets[b] for a,b in [(0,1),(0,2),(1,2)])

@pytest.mark.parametrize('key,value',[('game_index',True),('challenger_quota',64),('challenger_cpuct',.4),('candidate','parity'),('seed_decimal','0')])
def test_schedule_tamper_rejected(key,value):
    spec=deepcopy(schedule.make_schedule('development')[0]);spec[key]=value
    with pytest.raises(ValueError):schedule.validate_spec(spec)

def test_only_one_confirmation_branch():
    for selected in ('low','high'):
        specs=schedule.make_schedule()+schedule.make_schedule('confirmation',selected)
        assert len(schedule.validate_schedule(specs,selected))==612
        with pytest.raises(ValueError):schedule.validate_schedule(specs,None)
        with pytest.raises(ValueError):schedule.validate_schedule(specs,('high' if selected=='low' else 'low'))

def test_all_prospective_lifecycles_route_and_quota():
    inventory=design.lifecycle_inventory()
    for route in ('frozen','parity','low','high'):
        item=inventory['qualification_'+route]
        assert item['ordinary_requests']==24 and item['entrypoint_request_id']=='request-00000002'
    for arm in ('low','high'):
        for spec in schedule.make_schedule()+schedule.make_schedule('confirmation',arm):
            for role in ('challenger','opponent'):
                item=inventory[design.game_worker_name(spec,role)]
                assert item['route']==spec[role+'_mode'] and item['ordinary_requests']==2000
    assert c.MAX_WORKERS==1372 and c.MAX_RSS_BYTES==2*c.GiB

def fixture_inputs():
    return dict(game_costs=[dict(seconds=50.,independent_seconds=1.,bytes=10*c.MiB,files=100) for _ in range(12)],
        worker_costs=[dict(initialization=2.,graph=1.,closure=.5) for _ in range(24)],hash_seconds=[.1]*12,
        runtime_seconds=2.,ram_wait_used=60.,current_bytes=200*c.MiB,current_files=2000,
        monotonic_origin=1000.,unix_origin=10000.,monotonic=2000.,unix_time=11000.)

def test_resource_forecast_full_workload_passes_and_keeps_descriptive_timing():
    result=runner.resource_projection(fixture_inputs())
    assert result['passed'] and result['remaining_games']==600 and result['replay_decisions_maximum']==144
    assert result['timing_used_for_admission'] is False
    assert result['projected_saved_final_seconds']>7200

@pytest.mark.parametrize('mutation',['phase','disk','global','nan','negative'])
def test_resource_forecast_rejects_invalid_or_insufficient(mutation):
    value=fixture_inputs()
    if mutation=='phase':
        for row in value['game_costs']:row['independent_seconds']=50.
    elif mutation=='disk':
        for row in value['game_costs']:row['bytes']=c.GiB
    elif mutation=='global':value['monotonic']=100000.
    elif mutation=='nan':value['runtime_seconds']=float('nan')
    elif mutation=='negative':value['ram_wait_used']=-1.
    if mutation in ('nan','negative'):
        with pytest.raises(ValueError):runner.resource_projection(value)
    else:assert runner.resource_projection(value)['passed'] is False

def test_all_model_stages_cutoff_and_final_no_live_workers(monkeypatch,tmp_path):
    monkeypatch.setattr(c,'OUT',tmp_path)
    monkeypatch.setattr(c,'ROOT',tmp_path)
    monkeypatch.setattr(runner.time,'monotonic',lambda:100.)
    monkeypatch.setattr(runner.time,'time',lambda:1000.)
    e=runner.Envelope(dict(monotonic_origin=50.,unix_origin=950.));e.guard=lambda:None
    for stage in c.STAGES:
        e.begin(stage)
        assert e.deadline<=e.end
        if stage!='final':assert e.deadline<=e.start+c.COLLECTION_CUTOFF_SECONDS
    e.resident={1:object()}
    with pytest.raises(ValueError,match='no resident'):e.begin('final')

def test_independent_reader_clock_is_not_reset():
    claim=dict(monotonic_origin=100.,unix_origin=1000.)
    final=dict(monotonic_deadline=100.+c.JOB_SECONDS-10,unix_deadline=1000.+c.JOB_SECONDS-20)
    result=independent_review.review_deadlines(claim,final,100.+c.JOB_SECONDS-100,1000.+c.JOB_SECONDS-100)
    assert result==(final['monotonic_deadline'],final['unix_deadline'])

def test_savedrows_streams_and_filters_candidate(tmp_path):
    paths=[]
    for arm in ('low','high'):
        p=tmp_path/arm/'p000_g0'/'moves'/'0000.json';p.parent.mkdir(parents=True);p.write_text(json.dumps(dict(candidate=arm)));paths.append(p)
    rows=runner.SavedRows(paths)
    assert len(rows)==2 and list(rows)==list(rows)
    assert list(rows.for_candidate('low'))==[dict(candidate='low')]

def test_guardian_environment_import_smoke():
    source=Path(guardian.__file__).read_text()
    assert 'SPLENDOR_SEARCH128_REPEAT_PHASE' not in source
    assert source.count('SPLENDOR_EXPLORATION128_PHASE=c.PHASE')==3
    env=os.environ.copy();env.update(PYTHONDONTWRITEBYTECODE='1',SPLENDOR_EXPLORATION128_PHASE='exploration',PYTHONPATH=str(c.ROOT/'src')+os.pathsep+str(c.ROOT))
    result=subprocess.run([sys.executable,'-B','-c',"import sys; from tools.splendorust_exploration128_20261029 import runner,worker,guardian; from tools.splendorust_search128_repeat_20261029 import common; assert 'torch' not in sys.modules"],cwd=c.ROOT,env=env,capture_output=True,text=True,timeout=30)
    assert result.returncode==0,result.stderr

def test_saved_reader_has_no_model_launch_calls():
    source=Path(independent_review.__file__).read_text();tree=ast.parse(source)
    calls=[n.func.attr for n in ast.walk(tree) if isinstance(n,ast.Call) and isinstance(n.func,ast.Attribute)]
    assert 'make_worker' not in calls and 'Popen' not in calls
    assert 'verify_runtime_sources' in source and 'writing=False' in source

def test_original_frozen_sources_and_profile_unchanged():
    c.verify([dict(path='outputs/splendorust_search128_strength_20261028_01/profiles/puct128.json',bytes=Path(c.ROOT/'outputs/splendorust_search128_strength_20261028_01/profiles/puct128.json').stat().st_size,sha256='048fea7e9d9377f5b3e477d4bafc96d3e96321661d2146e299a95cdc51b36edb')])


def test_conditional_projection_only_remaining400_48_and_real_remaining_clock():
    value=fixture_inputs();value['scope']='confirmation'
    result=runner.resource_projection(value)
    assert result['passed'] and result['remaining_games']==400 and result['replay_decisions_maximum']==48
    assert set(result['phase_projected_seconds'])=={'confirmation','confirmation_verification'}
    value['monotonic']=value['monotonic_origin']+c.COLLECTION_CUTOFF_SECONDS-1
    assert not runner.resource_projection(value)['passed']

def test_external_model_clock_stops_both_clocks_and_requires_savedonly_final():
    cutoff=c.COLLECTION_CUTOFF_SECONDS
    assert guardian.model_clock_open(0,10,cutoff-1,cutoff+9,final_started=False,owned_children=[])
    assert not guardian.model_clock_open(0,10,cutoff,cutoff+9,final_started=False,owned_children=[])
    assert not guardian.model_clock_open(0,10,cutoff-1,cutoff+10,final_started=False,owned_children=[])
    assert guardian.model_clock_open(0,10,cutoff+100,cutoff+110,final_started=True,owned_children=[])
    with pytest.raises(ValueError,match='model/helper'):guardian.model_clock_open(0,10,cutoff+100,cutoff+110,final_started=True,owned_children=[object()])

def test_live_run_orders_conditional_forecast_before_confirmation_collection():
    source=Path(runner.__file__).read_text();node=next(n for n in ast.parse(source).body if isinstance(n,ast.FunctionDef) and n.name=='run')
    text=ast.get_source_segment(source,node)
    assert text.index('confirmation_forecast(')<text.index("collect_games(schedules['confirmation']")
    readback=source[source.index('def verify_saved_attempt'):source.index('def final_checks')]
    assert "placement_admission.json" in readback and "permit['perf_counter_ns']" in readback


def test_confirmation_creation_precedes_ready_but_must_follow_selection():
    permit=dict(unix_time=100.,perf_counter_ns=1000)
    admission=dict(supervisor=dict(monotonic_ns=1200))
    assert runner.validate_confirmation_startup(permit,dict(identity=dict(created=101.)),admission)
    for created in (99.,100.):
        with pytest.raises(ValueError,match='process creation'):
            runner.validate_confirmation_startup(permit,dict(identity=dict(created=created)),admission)
    with pytest.raises(ValueError,match='placement'):
        runner.validate_confirmation_startup(permit,dict(identity=dict(created=101.)),dict(supervisor=dict(monotonic_ns=999)))
