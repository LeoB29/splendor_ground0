"""Scope, inherited clocks and preserved numerical qualification contracts."""
import ast
from pathlib import Path
import pytest
from . import common as c,owner,runtime,fit,sampling


def test_collection_scope_and_workload():
    row=owner.contract()
    assert row['scope']=='collection_only'
    assert row['new_games']==3328 and row['collection_cells']*row['games_per_cell']==3328
    assert row['partition_counts']=={'train':3072,'validation':128,'offline':128}
    assert row['combined_training_games']==4096 and row['fresh_assessment_games']==256
    assert row['scientific_optimizer_calls']==row['main_strength_games']==0
    assert row['maximum_model_replays']==9984
    assert row['worlds']==3 and row['teacher_simulations']==512
    assert not row['automatic_followup'] and not row['available_ram_gate']


def test_no_scientific_fit_schedule():
    assert c.fits()==sampling.fits()==[]
    with pytest.raises(ValueError,match='outside'):sampling.fit_identity(0)


def test_actual_cli_cannot_fit(monkeypatch):
    monkeypatch.setattr('sys.argv',['fit','--mode','fit','--plan-sha256','x','--output','x'])
    with pytest.raises(SystemExit) as exc:fit.main()
    assert exc.value.code==2


def test_retained_numeric_math_is_accepted():
    proof=fit.source_equivalence()
    assert proof['models']==proof['optimizer_calls']==0
    assert len(proof['functions'])==8


@pytest.mark.parametrize('stage',['fit','offline','analysis','strength','unknown'])
def test_unapproved_phases_rejected(stage):
    with pytest.raises(ValueError,match='Fixed study phase'):owner.begin_phase({},stage)


@pytest.mark.parametrize('name',['qualification','collection','corpus','final'])
def test_phase_clock_clips_to_original_origin(name,monkeypatch):
    origin=100.;cut=c.TOTAL_SECONDS if name=='final' else c.MODEL_SECONDS
    now=origin+cut-100
    claim={'monotonic_origin':origin,'unix_origin':origin+1000}
    monkeypatch.setattr(owner.time,'monotonic',lambda:now)
    monkeypatch.setattr(owner.time,'time',lambda:now+1000)
    monkeypatch.setattr(owner,'guard',lambda *_:None)
    monkeypatch.setattr(c,'artifact',lambda _:dict(path='claim',bytes=1,sha256='a'*64))
    monkeypatch.setattr(c,'save',lambda p,v:v)
    actual=owner.begin_phase(claim,name)
    assert actual['monotonic_deadline']==origin+cut
    assert actual['unix_deadline']==origin+1000+cut


@pytest.mark.parametrize('expired',['overall','model','phase','worker','wall'])
def test_deadlines_are_real_and_independent(expired):
    claim={'monotonic_origin':10,'unix_origin':1000}
    phase={'monotonic':10,'unix_time':1000,'monotonic_deadline':1e9,'unix_deadline':1e9}
    own={'monotonic_origin':10,'unix_origin':1000,'monotonic_deadline':1e9,'unix_deadline':1e9}
    mono,wall=20,1010
    if expired=='overall':mono=10+c.TOTAL_SECONDS
    if expired=='model':mono=10+c.MODEL_SECONDS
    if expired=='phase':phase['monotonic_deadline']=mono
    if expired=='worker':own['monotonic_deadline']=mono
    if expired=='wall':wall=1000+c.TOTAL_SECONDS
    with pytest.raises(ValueError):runtime.check_deadlines(claim,phase,own,mono,wall)


def test_source_guard_exactly_once_in_bootstrap():
    tree=ast.parse(Path(runtime.__file__).read_text())
    function=next(x for x in tree.body if isinstance(x,ast.FunctionDef) and x.name=='bootstrap')
    calls=[x for x in ast.walk(function) if isinstance(x,ast.Call)
           and isinstance(x.func,ast.Name) and x.func.id=='verify_runtime_sources']
    assert len(calls)==1


def test_original_scope_not_expanded_by_training_cli():
    tree=ast.parse(Path(fit.__file__).read_text())
    main=next(x for x in tree.body if isinstance(x,ast.FunctionDef) and x.name=='main')
    calls={x.func.id for x in ast.walk(main) if isinstance(x,ast.Call) and isinstance(x.func,ast.Name)}
    assert 'run_qualification' in calls and not {'run_fit','create_commitment','verify_fit'}&calls
