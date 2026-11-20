"""Counterexamples for source-aware saved acceptance; no model execution."""
import ast
from copy import deepcopy
import json
import os
from pathlib import Path
import subprocess
import sys
import pytest

os.environ.setdefault('INDEPENDENT_DATA_REPETITION','0')
from tools.splendorust_independent_data_disk_recovery_20261109_02.strength import recovered_audit as a, report

ROOT=Path(__file__).resolve().parents[1]


def test_statistical_and_chart_functions_unchanged():
    old=ROOT/'tools/splendorust_independent_data_recovery_20261109/strength'
    new=ROOT/'tools/splendorust_independent_data_disk_recovery_20261109_02/strength'
    assert (old/'analysis.py').read_bytes()==(new/'analysis.py').read_bytes()
    def functions(path):
        return {n.name:ast.dump(n,include_attributes=False) for n in ast.parse(path.read_text()).body if isinstance(n,(ast.FunctionDef,ast.ClassDef))}
    before,after=functions(old/'report.py'),functions(new/'report.py')
    for name in ('chart','aggregate','main'):assert before[name]==after[name]


@pytest.mark.parametrize('rep',[0,1])
def test_expected_worker_boundary_is_only_partial_game(monkeypatch,rep):
    monkeypatch.setattr(a.c,'REPETITION',rep)
    spec=dict(stage='main',game_index=159)
    start,names,prefix=a.expected_pair(spec,dict(reason='natural'),150)
    assert start==(159 if rep else 150)
    assert prefix['control']=='main_g150_control'
    assert names['control']==('main_g159_control' if rep else 'main_g150_control')
    start,names,_=a.expected_pair(dict(stage='main',game_index=160),dict(reason='natural'),start)
    assert start==160 and names['candidate']=='main_g160_candidate'


def test_classified_loss_uses_next_game_not_retry(monkeypatch):
    monkeypatch.setattr(a.c,'REPETITION',0)
    start,names,_=a.expected_pair(dict(stage='main',game_index=163),dict(reason='forfeit'),160)
    assert start==163 and names['candidate']=='main_g163_candidate'
    start,_,_=a.expected_pair(dict(stage='main',game_index=164),dict(reason='natural'),163)
    assert start==163


@pytest.fixture
def interrupted_files(tmp_path,monkeypatch):
    monkeypatch.setattr(a.c,'REPETITION',1)
    monkeypatch.setattr(a.c,'LIVE_SOURCE_OUT',tmp_path/'old')
    monkeypatch.setattr(a.c,'OUT',tmp_path/'new')
    spec=dict(stage='main',game_index=159)
    old=a.lineage.old_game_folder(spec);new=a.lineage.game_folder(spec)
    (old/'moves').mkdir(parents=True);(new/'moves').mkdir(parents=True)
    for i in range(26):(old/'moves'/f'{i:06d}.json').write_text('{}')
    (new/'moves/000026.json').write_text('{}')
    return spec,old,new


def test_exact_prefix_and_suffix_inventory(interrupted_files):
    spec,old,new=interrupted_files
    a.game_inventory(spec,dict(plies=27))


@pytest.mark.parametrize('mutation',['copied_prefix','missing_prefix','missing_suffix','extra_suffix','old_result','old_replay','short_game','bool_plies'])
def test_prefix_replacement_or_missing_evidence_rejected(interrupted_files,mutation):
    spec,old,new=interrupted_files;ending=dict(plies=27)
    if mutation=='copied_prefix':(new/'moves/000025.json').write_text('{}')
    elif mutation=='missing_prefix':(old/'moves/000025.json').unlink()
    elif mutation=='missing_suffix':(new/'moves/000026.json').unlink()
    elif mutation=='extra_suffix':(new/'moves/000027.json').write_text('{}')
    elif mutation=='old_result':(old/'result.json').write_text('{}')
    elif mutation=='old_replay':(old/'online_replay.json').write_text('{}')
    elif mutation=='short_game':ending['plies']=26
    elif mutation=='bool_plies':ending['plies']=True
    with pytest.raises(ValueError):a.game_inventory(spec,ending)


def test_worker_segment_cannot_invent_another_partial_game(monkeypatch):
    monkeypatch.setattr(a.c,'REPETITION',1)
    assert a.worker_segment('main_g159_candidate','new')==('resume_partial','resume_partial',None)
    assert a.worker_segment('main_g159_candidate','old')==('main_150','main_block',150)
    monkeypatch.setattr(a.c,'REPETITION',0)
    with pytest.raises(ValueError):a.worker_segment('main_g159_candidate','new')


def test_report_requires_exact_original_and_continued_sources(tmp_path,monkeypatch):
    monkeypatch.setattr(report.c,'LIVE_SOURCE_OUT',tmp_path/'old')
    monkeypatch.setattr(report.c,'OUT',tmp_path/'new')
    for rep in (0,1):
        for index in range(200):
            base=report.c.LIVE_SOURCE_OUT if index<160 and (rep,index)!=(1,159) else report.c.OUT
            path=base/'strength'/f'r{rep}'/'games/main'/f'g{index:03d}'/'result.json'
            path.parent.mkdir(parents=True);path.write_text(json.dumps(dict(repetition=rep,game_index=index,source=base.name)))
    rows=report.main_games()
    assert len(rows[0])==len(rows[1])==200
    assert rows[0][159]['source']=='old' and rows[1][159]['source']=='new'
    path=report.c.OUT/'strength/r1/games/main/g159/result.json';path.unlink()
    wrong=report.c.LIVE_SOURCE_OUT/'strength/r1/games/main/g159/result.json'
    wrong.parent.mkdir(parents=True,exist_ok=True);wrong.write_text('{}')
    with pytest.raises(ValueError):report.main_games()


def test_saved_failure_actual_prefix_and_raw_boundary():
    code=r'''
import os,sys,time,json
from pathlib import Path
os.environ['INDEPENDENT_DATA_REPETITION']='1'
sys.path.insert(0,str(Path.cwd()/'src'))
from tools.future_training_rules_20261021 import verify_runtime_sources
verify_runtime_sources()
from tools.splendorust_independent_data_recovery_20261109.strength import common as c,control,transport
from tools.splendorust_independent_data_disk_recovery_20261109_02.strength import recovered_audit as audit
from tools.splendorust_gpu_probe_20261105.control import require_absent
from copy import deepcopy
from types import SimpleNamespace
end=time.monotonic()+90
class Boundary:
 def guard(self):assert time.monotonic()<end
 def effective_deadline(self):return end
env=Boundary();freeze=c.read(c.OUT/'freeze.json');folder=c.OUT/'games/main/g159';spec=c.read(folder/'spec.json')
rows=[c.read(folder/'moves'/f'{i:06d}.json') for i in range(26)];previous=c.read(folder/'initial.json')['initial']
names=c.read(folder/'workers.json')
for ply,row in enumerate(rows):
 assert row['ply']==ply and row['before']==previous and row['completed'] is True
 assert control.validate_row(row,spec,freeze,env,worker_name=names[row['route']]) is None
 previous=row['after']
assert previous['state']['terminal'] is False
bad=deepcopy(rows[-1]);bad['after']=bad['before']
try:control.validate_row(bad,spec,freeze,env,worker_name=names[bad['route']])
except ValueError:pass
else:raise AssertionError('Mutated saved afterstate accepted')
for name in names.values():
 worker=c.OUT/'workers'/name;closed=c.read(c.OUT/'closed_workers'/f'{name}.json')
 raw=transport.verify_raw_transport(worker,freeze,env)
 assert raw==closed['raw'] and raw['requests']==269 and raw['failed_requests']==0
 for ident in c.read(worker/'cleanup.json')['confirmed']:require_absent(ident)
 expected=[f'request-{n:08d}.json' for n in range(1,270)]
 assert [p.name for p in sorted(worker.glob('request-????????.json'))]==expected
 assert [p.name for p in sorted((worker/'inputs').glob('request-????????.json'))]==expected[2:]
origin=c.read(c.STUDY_OUT/'claim.json');env.start=origin['monotonic_origin'];env.wall_start=origin['unix_origin']
start,ended,failed=audit.segment_bounds('main_g150_candidate','old',env)
assert failed is True and start['phase']=='main_block'
assert 'torch' not in sys.modules
print(json.dumps(dict(saved_moves=26,raw_requests=538,failed_requests=0,in_flight=False,models=0,games=0)))
'''
    env=os.environ.copy();env.update(INDEPENDENT_DATA_REPETITION='1',PYTHONDONTWRITEBYTECODE='1',PYTHONPATH=str(ROOT/'src')+os.pathsep+str(ROOT))
    done=subprocess.run([sys.executable,'-B','-c',code],cwd=ROOT,env=env,text=True,capture_output=True,timeout=100)
    assert done.returncode==0,done.stdout+'\n'+done.stderr
    assert json.loads(done.stdout.splitlines()[-1])==dict(saved_moves=26,raw_requests=538,failed_requests=0,in_flight=False,models=0,games=0)


def test_saved_verifiers_do_not_construct_models():
    assert 'torch' not in sys.modules
    functions={node.name:node for node in ast.parse(Path(a.__file__).read_text()).body if isinstance(node,ast.FunctionDef)}
    for name in ('read_games','verify_saved','segment_bounds','verify_admissions'):
        calls=[node.func.id for node in ast.walk(functions[name]) if isinstance(node,ast.Call) and isinstance(node.func,ast.Name)]
        assert not set(calls)&{'load_model','train','forward','optimizer','decision','qualify'}
