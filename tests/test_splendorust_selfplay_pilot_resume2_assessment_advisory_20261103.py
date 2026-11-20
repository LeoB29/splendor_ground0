"""Forecasts remain exact recorded evidence but cannot stop the assessment."""
import ast
from copy import deepcopy
import os
from pathlib import Path
import tempfile
from types import SimpleNamespace
import pytest
from tools.splendorust_selfplay_pilot_resume2_20261103.assessment import common as c,control,transport


@pytest.fixture
def local(monkeypatch):
    base=c.ROOT/'outputs/splendorust_selfplay_pilot_resume2_20261103_01/prelaunch'
    base.mkdir(parents=True,exist_ok=True)
    with tempfile.TemporaryDirectory(prefix='advisory_test_',dir=base) as name:
        path=Path(name).resolve()
        monkeypatch.setattr(c.parent,'OUT',path)
        monkeypatch.setattr(c,'OUT',path/'assessment')
        monkeypatch.setattr(c,'MAIN_GAMES',400)
        monkeypatch.setattr(c,'MAX_GAMES',406)
        yield path


def overwrite(path,value):
    path.write_text(c.canonical(value)+'\n',encoding='utf-8')


def basis(local,monkeypatch,*,failure='main'):
    c.save(c.OUT/'freeze.json',{'test':True})
    owner=SimpleNamespace(start=100.,wall_start=1000.,deadline=300.,wall_deadline=1200.,waited=60.,
        remaining=lambda:100.,accounted_bytes=lambda:c.GiB,guard=lambda:None)
    monkeypatch.setattr(control.time,'monotonic',lambda:200.)
    monkeypatch.setattr(control.time,'time',lambda:1100.)
    monkeypatch.setattr(control.shutil,'disk_usage',lambda _:SimpleNamespace(free=60*c.GiB))
    stage=dict(monotonic=190.,unix_time=1090.,monotonic_deadline=300.,unix_deadline=1200.)
    c.save(c.OUT/'stages/main_start.json',stage)
    for spec in control.design.make_schedule('operational'):
        value=dict(seconds=31.,nonwait_seconds=30.,ram_wait_seconds=1.,accounted_growth_bytes=c.MiB,
            completed_decisions=50,plies=50,raw_stream_growth={r:{'stdout.log':c.MiB,'stderr.log':0,'events.jsonl':4096} for r in c.VARIANTS})
        if failure=='bytes':value['accounted_growth_bytes']=c.MAX_EVIDENCE_BYTES
        if failure=='raw':value['raw_stream_growth']['candidate']['stdout.log']=c.MAX_LOG_BYTES
        c.save(control.game_folder(spec)/'cost.json',value)
    for route in c.VARIANTS:
        c.save(c.OUT/'startup_costs'/f'operational_g000_{route}.json',dict(nonwait_seconds=10.,accounted_growth_bytes=c.MiB,
            raw_stream_bytes={'stdout.log':2*c.MiB,'stderr.log':0,'events.jsonl':8192}))
        c.save(c.OUT/'closed_worker_costs'/f'operational_g000_{route}.json',{'seconds':1.})
    return owner,stage


@pytest.mark.parametrize('failure',['main','bytes','raw'])
def test_actual_negative_forecast_is_saved_and_recomputed_without_stop(local,monkeypatch,failure):
    owner,_=basis(local,monkeypatch,failure=failure)
    result=control.assessment_gate(owner,writing=True)
    assert result['projection']['passed'] is False
    assert result['may_stop'] is False and result['forecast_policy']=='advisory_only'
    before=(c.OUT/'assessment_gate.json').read_bytes()
    assert control.assessment_gate(owner,writing=False)==result
    assert (c.OUT/'assessment_gate.json').read_bytes()==before
    estimate=result['projection']['main_seconds']
    assert estimate>owner.remaining()
    control.phase_admission(owner,'main',estimate,assessment_gate=c.artifact(c.OUT/'assessment_gate.json'))
    admission=control.verify_admission(owner,'main',estimate,assessment_gate=c.artifact(c.OUT/'assessment_gate.json'))
    assert admission['forecast_requirement_fits'] is False
    assert admission['effective_monotonic_deadline']==300.
    assert admission['effective_unix_deadline']==1200.


@pytest.mark.parametrize('tamper',['projection','inputs','snapshot','policy'])
def test_advisory_does_not_allow_wrong_saved_arithmetic_or_inputs(local,monkeypatch,tamper):
    owner,_=basis(local,monkeypatch)
    control.assessment_gate(owner,writing=True)
    path=c.OUT/'assessment_gate.json';value=c.read(path)
    if tamper=='projection':value['projection']['main_seconds']+=1
    elif tamper=='inputs':value['inputs']=[]
    elif tamper=='snapshot':value['snapshot']['model_remaining_seconds']+=1
    else:value['may_stop']=True
    overwrite(path,value)
    with pytest.raises(ValueError):control.assessment_gate(owner,writing=False)


@pytest.mark.parametrize('stage',['operational','main','replay'])
def test_all_phase_estimates_are_advisory_but_original_deadlines_bind(local,monkeypatch,stage):
    owner,record=basis(local,monkeypatch)
    if stage!='main':c.save(c.OUT/'stages'/f'{stage}_start.json',record)
    control.phase_admission(owner,stage,5000.)
    result=control.verify_admission(owner,stage,5000.)
    assert result['forecast_requirement_fits'] is False
    path=c.OUT/(stage+'_admission.json');value=c.read(path);value['forecast_requirement_fits']=True;overwrite(path,value)
    with pytest.raises(ValueError,match='arithmetic'):control.verify_admission(owner,stage,5000.)


@pytest.mark.parametrize('which',['monotonic','wall','resource_guard'])
def test_actual_deadline_or_resource_violation_still_stops(local,monkeypatch,which):
    owner,_=basis(local,monkeypatch)
    if which=='monotonic':owner.deadline=200.
    elif which=='wall':owner.wall_deadline=1100.
    else:
        def exhausted():raise ValueError('Actual worker RSS exceeded')
        owner.guard=exhausted
    with pytest.raises(ValueError):control.phase_admission(owner,'main',1e9)
    assert not (c.OUT/'main_admission.json').exists()


@pytest.mark.parametrize('tamper',['deadline','origin','time','estimate'])
def test_saved_phase_does_not_accept_clock_or_estimate_rewriting(local,monkeypatch,tamper):
    owner,_=basis(local,monkeypatch)
    control.phase_admission(owner,'main',5000.)
    path=c.OUT/'main_admission.json';v=c.read(path)
    if tamper=='deadline':v['effective_monotonic_deadline']+=1
    elif tamper=='origin':v['original_unix_origin']+=1
    elif tamper=='time':v['monotonic']=300.
    else:v['required_seconds']-=1
    overwrite(path,v)
    with pytest.raises(ValueError):control.verify_admission(owner,'main',5000.)


def test_production_saved_reader_advances_past_negative_forecast(local,monkeypatch):
    owner,_=basis(local,monkeypatch)
    gate=control.assessment_gate(owner,writing=True)
    control.phase_admission(owner,'main',gate['projection']['main_seconds'],assessment_gate=c.artifact(c.OUT/'assessment_gate.json'))
    c.save(c.OUT/'qualification_summary.json',{})
    c.save(c.OUT/'stages/operational_start.json',c.read(c.OUT/'stages/main_start.json'))
    control.phase_admission(owner,'operational',300.,qualification_summary=c.artifact(c.OUT/'qualification_summary.json'))
    specs=control.design.make_schedule()
    games=[{**s,'reason':'natural','plies':2} for s in specs]
    for s in specs:(control.game_folder(s)).mkdir(exist_ok=True,parents=True)
    summary={'passed':True};c.save(c.OUT/'operational_analysis.json',summary)
    monkeypatch.setattr(control,'verify_runtime',lambda *a:None)
    monkeypatch.setattr(control,'load_inputs',lambda *a:(specs,{}))
    monkeypatch.setattr(control,'read_games',lambda *a:(games,[],{}))
    monkeypatch.setattr(control.analysis,'summarize_operational',lambda *a,**k:summary)
    class LaterVerification(Exception):pass
    def reached(*a):raise LaterVerification('Real saved reader reached subsequent block verification')
    monkeypatch.setattr(control,'verify_blocks',reached)
    with pytest.raises(LaterVerification):control._verify_saved(owner,{},writing=False)


@pytest.mark.parametrize('operational_failure',[False,True])
def test_actual_controller_continues_false_forecast_but_not_failed_science(local,monkeypatch,operational_failure):
    events=[]
    class Owner:
        start=100.;wall_start=1000.;waited=60.;resident={};worker_accounting={};evidence={'workers':{}}
        def guard(self):pass
        def remaining(self):return 100.
        def begin(self,name):
            self.stage=name;events.append(('begin',name));self.deadline=300.;self.wall_deadline=1200.
            c.parent.save(local/'stages'/f'{name}_start.json',dict(monotonic=190.,unix_time=1090.,monotonic_deadline=300.,unix_deadline=1200.))
        def finish(self):c.parent.save(local/'stages'/f'{self.stage}_end.json',{})
    owner=Owner()
    monkeypatch.setattr(control.time,'monotonic',lambda:200.)
    monkeypatch.setattr(control.time,'time',lambda:1100.)
    c.parent.save(local/'launch.json',dict(pid=os.getpid(),created=control.psutil.Process().create_time()))
    def binding(*a):c.save(c.OUT/'freeze.json',{});return {}
    monkeypatch.setattr(control,'bind_checkpoint',binding)
    monkeypatch.setattr(control,'verify_runtime',lambda *a:None)
    monkeypatch.setattr(control,'load_inputs',lambda *a:(control.design.make_schedule(),{}))
    def qualify(name,workers,*a):
        workers[name]=SimpleNamespace(_closed=False)
        c.save(c.OUT/'qualification'/f'{name}.json',{'qualified':True})
        c.save(c.OUT/'startup_costs'/f'{name}.json',{'nonwait_seconds':10.})
    monkeypatch.setattr(control,'qualify',qualify)
    monkeypatch.setattr(control,'advance_idle_worker',lambda *a:None)
    def play(spec,*a):return {**spec,'reason':'forfeit' if operational_failure else 'natural','plies':2}
    monkeypatch.setattr(control,'play_game',play)
    monkeypatch.setattr(control,'close_workers',lambda workers,*a:workers.clear())
    monkeypatch.setattr(control.transport,'close_and_audit',lambda *a:None)
    monkeypatch.setattr(control.analysis,'summarize_operational',lambda *a,**k:{'passed':True})
    def forecast(*a,**k):
        value={'projection':{'passed':False,'main_seconds':1e8}}
        c.save(c.OUT/'assessment_gate.json',value);return value
    monkeypatch.setattr(control,'assessment_gate',forecast)
    def main(specs,*a):events.append(('main',len(specs)));return [{**s,'reason':'natural','plies':2} for s in specs]
    monkeypatch.setattr(control,'collect_main',main)
    monkeypatch.setattr(control.design,'select_replay_sample',lambda *a,**k:{'decisions':[{}]*96})
    monkeypatch.setattr(control,'replay_selected',lambda *a:events.append(('replays',96)))
    monkeypatch.setattr(control,'prior_pair_closures',lambda *a:[])
    if operational_failure:
        with pytest.raises(ValueError,match='operational failure'):control.run(owner,{}, {})
        assert not any(k=='main' for k,v in events)
    else:
        result=control.run(owner,{}, {})
        assert result['games']==406 and result['replays']==96
        assert ('main',400) in events and ('replays',96) in events
        assert c.read(c.OUT/'main_admission.json')['forecast_requirement_fits'] is False
        assert c.read(c.OUT/'replay_admission.json')['forecast_requirement_fits'] is False


def test_pricing_science_transport_and_hardguards_are_unchanged():
    root=c.ROOT/'tools';old=root/'splendorust_selfplay_pilot_budget_20261102/assessment';new=root/'splendorust_selfplay_pilot_resume2_20261103/assessment'
    assert (new/'transport.py').read_text()==(old/'transport.py').read_text().replace('splendorust_selfplay_pilot_budget_20261102','splendorust_selfplay_pilot_resume2_20261103')
    oldtree=ast.parse((old/'control.py').read_text());newtree=ast.parse((new/'control.py').read_text())
    oldf={n.name:n for n in oldtree.body if isinstance(n,ast.FunctionDef)}
    newf={n.name:n for n in newtree.body if isinstance(n,ast.FunctionDef)}
    for name in ('forecast_projection','qualify','play_game','validate_row','read_games','replay_selected','verify_replays',
                 'verify_blocks','collect_main','verify_binding','advance_idle_worker'):
        assert ast.dump(oldf[name])==ast.dump(newf[name]),name
    assert c.INTERNAL_SECONDS==9. and c.SECONDS_PER_MOVE==10.
    assert c.FORECAST_POLICY=='advisory_only'
    assert control.contract()['resource_forecasts_may_stop'] is False
