"""Independent continuation integration checks: no inference or game simulation."""
from copy import deepcopy
import ast
import inspect
import json
import os
from pathlib import Path
import subprocess
import sys
from types import SimpleNamespace

import pytest

from tools.splendorust_selfplay_pilot_resume_20261103 import common as c, design, qualification_control as q


def test_resume_default_scope_can_never_write_to_accepted_qualification():
    env = dict(os.environ); env.pop('SELFPLAY_JOB_SCOPE', None)
    command = [sys.executable, '-c', 'from tools.splendorust_selfplay_pilot_resume_20261103 import common as c; print(c.SCOPE); print(c.OUT.name)']
    result = subprocess.run(command, env=env, capture_output=True, text=True, timeout=20)
    assert result.returncode == 0 and result.stdout.splitlines() == ['pilot', 'splendorust_selfplay_pilot_resume_20261103_01']
    env['SELFPLAY_JOB_SCOPE'] = 'qualification'
    rejected = subprocess.run(command, env=env, capture_output=True, text=True, timeout=20)
    assert rejected.returncode != 0


def test_output_writer_rejects_old_prefix_and_accepted_qualification(tmp_path, monkeypatch):
    new = tmp_path / 'new'; monkeypatch.setattr(c, 'ROOT', tmp_path); monkeypatch.setattr(c, 'OUT', new)
    monkeypatch.setattr(c, 'WRITE_OBSERVER', None)
    for old in [tmp_path/'old_pilot', tmp_path/'accepted_qualification']:
        with pytest.raises(ValueError): c.save(old/'invented.json', {})
        assert not old.exists()
    c.save(new/'okay.json', {})
    assert (new/'okay.json').exists()


def test_exact_prefix_paths_and_remaining_collection_blocks():
    assert q.game_folder({'game_index': 0}) == q.continuation.OLD_OUT/'games/g000'
    assert q.game_folder({'game_index': 31}) == q.continuation.OLD_OUT/'games/g031'
    assert q.game_folder({'game_index': 32}) == c.OUT/'games/g032'
    assert q.game_folder({'game_index': 255}) == c.OUT/'games/g255'
    blocks = design.remaining_collection_blocks()
    assert len(blocks) == 28 and [b['block_index'] for b in blocks] == list(range(4,32))
    assert [i for b in blocks for i in b['game_indices']] == list(range(32,256))
    assert blocks[0]['worker'] == 'collect_004' and blocks[-1]['worker'] == 'collect_031'
    names = set(q.lifecycle_inventory())
    assert len(names) == 34 and len(names-set(q.continuation.PREFIX_NAMES)) == 29
    assert set(q.continuation.PREFIX_NAMES) == {'capture_off','capture_on','collect_001','collect_002','collect_003'}
    old = c.read(q.continuation.OLD_OUT/'game_schedule.json')
    assert design.schedule() == old


@pytest.mark.parametrize('index', [0, 31])
def test_preserved_game_is_rejected_before_any_live_call(index):
    with pytest.raises(ValueError, match='Never regenerate'):
        q.play_game(None, {'game_index':index}, None, None, None)


def inherited():
    return {'original_monotonic_origin':100.,'original_unix_origin':1000.,'prior_source_lifecycles':5,
            'prior_ram_wait_seconds':12.,'prior_evidence_bytes':1234,
            'original_collection':{'monotonic_deadline':700.,'unix_deadline':1600.}}


def test_prefix_is_inherited_before_accounting_and_charged_once(monkeypatch):
    value = inherited()
    monkeypatch.setattr(q.continuation, 'verify_continuation', lambda item, boundary: value)
    monkeypatch.setattr(q.continuation, 'verify_prefix_review', lambda item: {'passed':True})
    monkeypatch.setattr(q, 'bytes_used', lambda: 100)
    env = q.Envelope({'monotonic_origin':100.,'unix_origin':1000.})
    env.inherit_prefix({'continuation':{},'saved_prefix_review':{}})
    assert env.prior_worker_count == 5 and env.waited == 12
    monkeypatch.setattr(c, 'WRITE_OBSERVER', None)
    env.enable_evidence_accounting()
    assert env.evidence['baseline'] == 1334
    c.WRITE_OBSERVER = None


def test_prefix_rejects_reset_owner_origins(monkeypatch):
    monkeypatch.setattr(q.continuation, 'verify_continuation', lambda item,boundary: inherited())
    monkeypatch.setattr(q.continuation, 'verify_prefix_review', lambda item: None)
    env = q.Envelope({'monotonic_origin':101.,'unix_origin':1000.})
    with pytest.raises(ValueError, match='inherit both'): env.inherit_prefix({'continuation':{},'saved_prefix_review':{}})


def test_collection_keeps_original_absolute_deadlines(monkeypatch):
    env=q.Envelope({'monotonic_origin':100.,'unix_origin':1000.});env.continuation=inherited()
    monkeypatch.setattr(q.time, 'monotonic', lambda:500.)
    monkeypatch.setattr(q.time, 'time', lambda:1400.)
    monkeypatch.setattr(env, 'guard', lambda:None)
    records=[];monkeypatch.setattr(c, 'atomic_save', lambda path,value:records.append(value))
    env.begin('collection')
    assert (env.deadline,env.wall_deadline)==(700.,1600.) and env.remaining()==200.
    assert records[0]['monotonic_deadline']==700. and records[0]['unix_deadline']==1600.


def test_actual_model_and_global_deadlines_still_stop(monkeypatch):
    env=q.Envelope({'monotonic_origin':100.,'unix_origin':1000.})
    env.deadline=700.;env.wall_deadline=1600.
    monkeypatch.setattr(q.time, 'monotonic', lambda:701.)
    monkeypatch.setattr(q.time, 'time', lambda:1601.)
    with pytest.raises(ValueError,match='real-clock'):env.guard()


@pytest.mark.parametrize('changed', [False, True])
def test_actual_original_qualification_interval_is_rechecked_in_final_reader(monkeypatch,changed):
    # Execute the production final verifier's interval block with its real saved
    # original qualification receipts, without repeating full models/game audit.
    tree=ast.parse(inspect.getsource(q.verify_saved))
    assignment=next(node for node in ast.walk(tree) if isinstance(node,ast.Assign) and
                    any(isinstance(target,ast.Name) and target.id=='expected_stages' for target in node.targets))
    assert 'qualification' in ast.literal_eval(assignment.value)
    loop=next(node for node in ast.walk(tree) if isinstance(node,ast.For) and
              isinstance(node.target,ast.Name) and node.target.id=='stage')
    claim=c.read(q.continuation.OLD_OUT/'guardian_claim.json')
    env=SimpleNamespace(start=claim['monotonic_origin'],wall_start=claim['unix_origin'])
    original_read=c.read
    def read(path):
        value=original_read(path)
        if changed and path==q.continuation.OLD_OUT/'stages/qualification_end.json':
            value=deepcopy(value)
            start=original_read(q.continuation.OLD_OUT/'stages/qualification_start.json')
            value['monotonic']=start['monotonic_deadline']
        return value
    monkeypatch.setattr(c,'read',read)
    code=compile(ast.fix_missing_locations(ast.Module(body=[loop],type_ignores=[])), '<production-stage-check>', 'exec')
    scope=dict(expected_stages=['qualification'],continuation=q.continuation,c=c,envelope=env,preserved={})
    if changed:
        with pytest.raises(ValueError,match='Original stage/model clocks'):exec(code,scope)
    else:exec(code,scope)


def test_block004_advisory_false_is_saved_and_verified_without_old_writes(monkeypatch):
    block=design.remaining_collection_blocks()[0]
    snapshot={'original_monotonic_origin':100.,'original_unix_origin':1000.,'stage_monotonic_deadline':700.,
              'stage_unix_deadline':1600.,'completed_games':40,'qualified_source_workers':6}
    env=SimpleNamespace(start=100.,wall_start=1000.)
    monkeypatch.setattr(q, 'forecast_snapshot', lambda *args,**kwargs:deepcopy(snapshot))
    projection={'affordable':False,'checks':{'collection':False},'snapshot':deepcopy(snapshot)}
    seen=[]
    monkeypatch.setattr(design, 'source_forecast', lambda freeze,snapshot,source_costs:deepcopy(projection))
    monkeypatch.setattr(c, 'artifact', lambda path:{'path':str(path),'bytes':1,'sha256':'0'*64})
    def read(path):
        if path.name=='collection_start.json':return {'monotonic_deadline':700.,'unix_deadline':1600.}
        if path.name=='cost.json':return {'seconds':1.,'accounted_growth_bytes':1}
        if path.name=='004.json':return seen[-1][1]
        raise AssertionError(path)
    monkeypatch.setattr(c,'read',read)
    monkeypatch.setattr(c,'save',lambda path,value:seen.append((path,deepcopy(value))))
    result=q.verify_block_forecast(block,env,{},writing=True)
    assert result['affordable'] is False and seen[0][0]==c.OUT/'block_forecasts/004.json'
    assert len(result['cost_inputs'])==40 and str(q.continuation.OLD_OUT) in result['cost_inputs'][0]['path']
    assert str(c.OUT) in result['cost_inputs'][32]['path']
    assert q.verify_block_forecast(block,env,{},writing=False)==result


def test_fresh_owner_orchestration_retains32_and_generates_only224(tmp_path,monkeypatch):
    out=tmp_path/'new';out.mkdir()
    monkeypatch.setattr(c,'ROOT',tmp_path);monkeypatch.setattr(c,'OUT',out);monkeypatch.setattr(c,'WRITE_OBSERVER',None)
    specs=design.schedule()
    prefix=[{'path':f'old/g{i:03d}.json','bytes':1,'sha256':'0'*64} for i in range(32)]
    freeze={'source_continuation':{'games':prefix,'original_freeze':{}},'continuation':{},'saved_prefix_review':{}}
    c.save(out/'freeze.json',freeze);expected=c.file_hash(out/'freeze.json')
    c.save(out/'watchdog_claim.json',{'freeze_sha256':expected})
    class Envelope:
        def __init__(self,claim):self.start=0.;self.wall_start=0.;self.stage=None
        def inherit_prefix(self,freeze):events.append('inherit')
        def enable_evidence_accounting(self):events.append('accounting')
        def begin(self,stage):self.stage=stage;events.append(stage)
        def admit(self):pass
        def finish(self):pass
    events=[];played=[];qualified=[];blocks=[]
    monkeypatch.setattr(q,'Envelope',Envelope)
    monkeypatch.setattr(q.placement,'configure_execution',lambda *args:None)
    monkeypatch.setattr(q,'verify_runtime_sources',lambda:[])
    monkeypatch.setattr(q,'verify_runtime',lambda *args:None)
    monkeypatch.setattr(q,'load_inputs',lambda *args:(specs,{}))
    monkeypatch.setattr(q,'switch_placement',lambda *args:None)
    monkeypatch.setattr(design,'initial_states',lambda:[{'spec':s,'initial':{}} for s in specs])
    def qualify(name,workers,*args):qualified.append(name);workers[name]=SimpleNamespace(_closed=False)
    monkeypatch.setattr(q,'qualify',qualify)
    def play(child,spec,initial,*args):
        i=spec['game_index'];assert i>=32;played.append(i)
        c.save(out/'progress'/f'game_{i:03d}.json',{})
        return {'path':f'new/games/g{i:03d}/game.json','bytes':1,'sha256':'0'*64}
    monkeypatch.setattr(q,'play_game',play)
    monkeypatch.setattr(q,'close_workers',lambda workers,*args:workers.clear())
    monkeypatch.setattr(q,'verify_block_forecast',lambda block,*args,**kwargs:blocks.append(block['block_index']) or {'affordable':False})
    monkeypatch.setattr(q,'select_replays',lambda games,env:[])
    monkeypatch.setattr(q,'reproduce',lambda *args:None)
    monkeypatch.setattr(q,'run_fit',lambda *args:c.save(out/'fit/fit_result.json',{'checkpoint_manifest':{}}))
    monkeypatch.setattr(q,'verify_fit',lambda *args:{})
    from tools.splendorust_selfplay_pilot_resume_20261103.assessment import control as assessment
    monkeypatch.setattr(assessment,'run',lambda *args:events.append('assessment'))
    monkeypatch.setattr(q,'final_checks',lambda *args:{'passed':True})
    assert q.run(expected)==0
    assert played==list(range(32,256)) and blocks==list(range(4,32))
    assert qualified==[f'collect_{i:03d}' for i in range(4,32)]+['replay_on']
    manifest=c.read(out/'corpus_manifest.json')
    assert manifest['games'][:32]==prefix and len(manifest['games'])==256
    assert events[:3]==['inherit','accounting','collection']
    assert c.read(out/'completion.json')['completed_games']==256


@pytest.mark.parametrize('new_bytes,accepted', [(29,True),(30,False),(31,False)])
def test_independent_reader_combines_prior_physical_bytes_and_reports_only_new_games(tmp_path,monkeypatch,new_bytes,accepted):
    from tools.splendorust_selfplay_pilot_resume_20261103 import independent_review as reader
    out=tmp_path/'new';out.mkdir()
    monkeypatch.setattr(c,'ROOT',tmp_path);monkeypatch.setattr(c,'OUT',out);monkeypatch.setattr(c,'WRITE_OBSERVER',None)
    monkeypatch.setattr(c,'MAX_EVIDENCE_BYTES',100)
    prior=c.save(out/'continuation_manifest.json',{'prior_evidence_bytes':70})
    c.save(out/'freeze.json',{'continuation':prior,'inputs':[]})
    sha=c.file_hash(out/'freeze.json');coverage={'new_source_games':224,'preserved_source_games':32}
    c.save(out/'guardian_claim.json',{'freeze_sha256':sha})
    c.save(out/'stages/final_start.json',{'monotonic':1.,'unix_time':10.,'monotonic_deadline':3.,'unix_deadline':30.})
    c.save(out/'stages/final_end.json',{'monotonic':2.,'unix_time':20.})
    c.save(out/'completion.json',{'status':'completed_verified','failure':None,'freeze_sha256':sha,'coverage':coverage})
    c.save(out/'completed_evidence_seal.json',{'artifacts':[]})
    monkeypatch.setattr(reader,'Guard',lambda *args:SimpleNamespace(guard=lambda:None,peak_rss=0))
    monkeypatch.setattr(reader,'verify_runtime_sources',lambda:[])
    monkeypatch.setattr(reader.placement,'configure_execution',lambda *args:None)
    monkeypatch.setattr(reader,'check_successful_owners',lambda *args:[])
    monkeypatch.setattr(q,'verify_runtime',lambda *args:None)
    monkeypatch.setattr(q,'verify_saved',lambda *args,**kwargs:coverage)
    monkeypatch.setattr(q,'bytes_used',lambda:new_bytes)
    # The empty miniature fixture isolates this final physical-accounting gate;
    # complete real source/row/closure verification is exercised independently.
    monkeypatch.setattr(reader,'DIRECTORIES',())
    if not accepted:
        with pytest.raises(ValueError,match='Combined original'):reader.review(sha)
        assert not (out/'independent_review.json').exists()
    else:
        reader.review(sha)
        result=c.read(out/'independent_review.json')
        assert result['new_games']==224 and result['preserved_source_games']==32 and result['original_attempt_remains_incomplete']
