"""Independent model-free qualification, retained-clock and owner checks."""
from copy import deepcopy
from pathlib import Path
from types import SimpleNamespace

import pytest

from tools.splendorust_search256_recovery_20261030_02 import common as c
from tools.splendorust_search256_recovery_20261030_02 import guardian, runner, qualification


@pytest.fixture
def retained_setup(monkeypatch):
    saved = {}
    suites = {route: dict(status='ok', result=dict(execution_route=route,
        complete_fixture=['full scientific result', route], counts=dict(complete_decisions=67)))
        for route in ('research', 'frozen')}
    children = {}
    def make(name, envelope, freeze, route):
        child = SimpleNamespace(route=route, output_dir=c.OUT/'workers'/name, _counter=2,
            _closed=False, recovery_accounting=SimpleNamespace(state='idle'),
            absolute_deadline=123., process=SimpleNamespace(pid=100+len(children)))
        children[route] = child
        return child
    def save(path, value):
        saved[str(path)] = deepcopy(value)
        return dict(path=str(path), sha256='a'*64, bytes=1)
    monkeypatch.setattr(runner, 'make_worker', make)
    monkeypatch.setattr(runner.adapters, 'actual_entrypoint',
        lambda child, *a, **k: deepcopy(suites[child.route]))
    monkeypatch.setattr(qualification, 'verify_entrypoint', lambda *a, **k: dict(status='passed'))
    monkeypatch.setattr(c, 'read', lambda path: deepcopy(suites[Path(path).stem]))
    monkeypatch.setattr(c, 'save', save)
    monkeypatch.setattr(c, 'artifact', lambda path: dict(path=str(path), sha256='b'*64, bytes=1))
    monkeypatch.setattr(runner, 'ordinary', lambda *a, **k: pytest.fail('No ordinary model requests during replacement qualification'))
    spec=dict(stage='main', candidate='research', game_index=294,
        challenger_mode='research', opponent_mode='frozen')
    envelope=SimpleNamespace(guard=lambda:None, stage='main',
        effective_deadline=lambda:c.ORIGINAL_MAIN_MONOTONIC_DEADLINE)
    return spec, envelope, suites, children, saved


def test_actual_qualification_retains_exact_two_trees_and_counts134(retained_setup):
    spec, envelope, suites, children, saved = retained_setup
    workers={}
    result=runner.qualify_retained_workers(spec,envelope,{}, {},workers)
    assert set(workers)=={'challenger','opponent'}
    assert workers['challenger'] is children['research']
    assert workers['opponent'] is children['frozen']
    assert (result['scientific_decisions'],result['parent_requests'],result['graph_cases'])==(134,4,22)
    assert result['retained_for_game294'] and result['original_qualification_reused']
    assert all(child._counter==2 and not child._closed for child in children.values())


def test_qualification_compares_entire_original_science(retained_setup, monkeypatch):
    spec, envelope, suites, children, saved = retained_setup
    def changed(child,*a,**k):
        value=deepcopy(suites[child.route]); value['result']['complete_fixture'][0]='changed'
        return value
    monkeypatch.setattr(runner.adapters,'actual_entrypoint',changed)
    with pytest.raises(ValueError,match='unchanged scientific suite outputs'):
        runner.qualify_retained_workers(spec,envelope,{}, {},{})
    assert not any(path.endswith('recovery_qualification_summary.json') for path in saved)


def test_retained_workers_move_phase_only_while_idle(retained_setup):
    spec, envelope, suites, children, saved = retained_setup
    workers={};runner.qualify_retained_workers(spec,envelope,{}, {},workers)
    runner.transition_retained_workers(workers,envelope)
    assert all(child._counter==2 and child.absolute_deadline==c.ORIGINAL_MAIN_MONOTONIC_DEADLINE
        for child in children.values())
    proof=saved[str(c.OUT/'retained_worker_phase_transition.json')]
    assert all(row['active_request'] is False and row['request_counter']==2 for row in proof['workers'])


@pytest.mark.parametrize('field,value',[('_counter',3),('_closed',True),('state','waiting')])
def test_active_or_consumed_worker_cannot_rebind_deadline(retained_setup,field,value):
    spec, envelope, suites, children, saved = retained_setup
    workers={};runner.qualify_retained_workers(spec,envelope,{}, {},workers)
    target=children['research']
    setattr(target.recovery_accounting if field=='state' else target,field,value)
    with pytest.raises(ValueError,match='No active request'):
        runner.transition_retained_workers(workers,envelope)
    assert target.absolute_deadline==123.


@pytest.mark.parametrize('stage',['recovery_qualification','main'])
def test_runner_clamps_new_stage_to_original_main_on_both_clocks(monkeypatch,stage):
    envelope=runner.Envelope.__new__(runner.Envelope)
    envelope.start=c.PRIOR_ORIGIN;envelope.wall_start=c.PRIOR_WALL_START
    envelope.recovery_owner_start=c.ORIGINAL_MAIN_MONOTONIC_DEADLINE-20
    envelope.recovery_owner_wall_start=c.ORIGINAL_MAIN_UNIX_DEADLINE-20
    envelope.end=envelope.start+c.JOB_SECONDS
    envelope.guard=lambda:None
    monkeypatch.setattr(runner.time,'monotonic',lambda:c.ORIGINAL_MAIN_MONOTONIC_DEADLINE-10)
    monkeypatch.setattr(runner.time,'time',lambda:c.ORIGINAL_MAIN_UNIX_DEADLINE-10)
    monkeypatch.setattr(c,'save',lambda *args:None)
    envelope.begin(stage)
    assert envelope.deadline==c.ORIGINAL_MAIN_MONOTONIC_DEADLINE
    assert envelope.wall_deadline==c.ORIGINAL_MAIN_UNIX_DEADLINE


def test_existing_guardian_claim_prevents_any_new_owner_spawn(monkeypatch,tmp_path):
    (tmp_path/'guardian_claim.json').write_text('{}',encoding='utf-8')
    monkeypatch.setattr(c,'OUT',tmp_path)
    monkeypatch.setattr(guardian.subprocess,'Popen',lambda *a,**k:pytest.fail('Existing claim cannot spawn'))
    with pytest.raises(ValueError,match='No relaunch'):
        guardian.guardian_main(SimpleNamespace(freeze_sha256='0'*64))
