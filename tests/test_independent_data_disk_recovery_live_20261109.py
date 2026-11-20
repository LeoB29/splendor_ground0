"""Source routing and no-retry live continuation; all workers are inert fixtures."""
from copy import deepcopy
import ast
import os
from pathlib import Path
from types import SimpleNamespace
import pytest

os.environ.setdefault('INDEPENDENT_DATA_REPETITION','0')
from tools.splendorust_independent_data_disk_recovery_20261109 import common as root
from tools.splendorust_independent_data_disk_recovery_20261109.strength import common as c,control,lineage,transport


def put(path,value):
    path=Path(path)
    assert path.is_relative_to(root.ROOT) and 'pytest-' in str(root.ROOT)
    path.parent.mkdir(parents=True,exist_ok=True)
    with path.open('x',encoding='utf-8') as stream:stream.write(root.canonical(value))
    return root.artifact(path)


@pytest.fixture
def isolated(tmp_path,monkeypatch):
    monkeypatch.setattr(root,'ROOT',tmp_path)
    monkeypatch.setattr(root,'OUT',tmp_path/'new')
    monkeypatch.setattr(root,'LIVE_SOURCE_OUT',tmp_path/'old')
    monkeypatch.setattr(root,'WRITE_OBSERVER',None)
    monkeypatch.setattr(c,'ROOT',tmp_path)
    monkeypatch.setattr(c,'STUDY_OUT',root.OUT)
    monkeypatch.setattr(c,'OUT',root.OUT/'strength/r1')
    monkeypatch.setattr(c,'LIVE_SOURCE_OUT',root.LIVE_SOURCE_OUT/'strength/r1')
    monkeypatch.setattr(c,'REPETITION',1)
    return tmp_path


@pytest.mark.parametrize('rep,stage,index,expected',[(0,'operational',0,'old'),(1,'operational',5,'old'),
    (0,'main',159,'old'),(1,'main',158,'old'),(1,'main',159,'new'),(0,'main',160,'new'),(1,'main',199,'new')])
def test_fixed_game_source_matrix(isolated,monkeypatch,rep,stage,index,expected):
    monkeypatch.setattr(c,'REPETITION',rep)
    spec=dict(stage=stage,game_index=index)
    assert lineage.game_source(spec)==expected
    assert lineage.game_folder(spec).is_relative_to(c.LIVE_SOURCE_OUT if expected=='old' else c.OUT)


@pytest.mark.parametrize('stage,index',[('main',-1),('main',200),('operational',6),('other',0),('main',True)])
def test_source_rejects_invalid_game(isolated,stage,index):
    with pytest.raises(ValueError):lineage.game_folder(dict(stage=stage,game_index=index))


def test_partial_reads_exact_split_and_no_copy(isolated):
    game=dict(stage='main',game_index=159,plies=29)
    for ply in range(29):put(lineage.move_path(game,ply),dict(ply=ply,source='old' if ply<26 else 'new'))
    rows=lineage.read_rows(game)
    assert [r['ply'] for r in rows]==list(range(29))
    assert [r['source'] for r in rows]==['old']*26+['new']*3
    assert not (lineage.game_folder(game)/'moves/000025.json').exists()
    assert not (lineage.old_game_folder(game)/'moves/000026.json').exists()
    assert lineage.source_move_path(dict(**game,ply=25))==lineage.old_game_folder(game)/'moves/000025.json'


@pytest.mark.parametrize('source',['old','new'])
def test_row_source_is_exact_worker_parent(isolated,source):
    folder=lineage.worker_folder('main_g159_control',source)
    assert lineage.source_of(dict(worker_folder=folder.relative_to(c.ROOT).as_posix()))==source
    with pytest.raises(ValueError):lineage.source_of(dict(worker_folder=(folder/'nested').relative_to(c.ROOT).as_posix()))


def test_unpinned_old_local_freeze_rejected(isolated):
    descriptor=put(c.LIVE_SOURCE_OUT/'freeze.json',dict(test='old'))
    put(c.STUDY_OUT/'freeze.json',dict(inputs=[]))
    with pytest.raises(ValueError):lineage.old_freeze()
    (c.STUDY_OUT/'freeze.json').write_text(root.canonical(dict(inputs=[descriptor])))
    assert lineage.old_freeze()==dict(test='old')


def test_readonly_old_boundary_rejects_live_or_unknown_worker(isolated):
    calls=[];guard=SimpleNamespace(guard=lambda:calls.append('guard'))
    boundary=lineage.old_boundary(guard)
    boundary.guard();assert calls==['guard']
    boundary.close_worker(lineage.worker_folder('main_g150_control','old'))
    with pytest.raises(ValueError):boundary.close_worker(lineage.worker_folder('main_g150_control','old'),object())
    with pytest.raises(ValueError):boundary.close_worker(lineage.worker_folder('main_g159_control','new'))


def test_failed_source_segment_never_masquerades_as_complete(isolated,monkeypatch):
    assert lineage.prior_segment('main_150')==c.OUT/'segments/resume_partial/complete.json'
    monkeypatch.setattr(c,'REPETITION',0)
    assert lineage.prior_segment('main_150')==c.LIVE_SOURCE_OUT/'segments/main_150/complete.json'
    assert lineage.prior_segment('main_160')==c.OUT/'segments/main_160/complete.json'
    assert lineage.prior_segment('operational')==c.LIVE_SOURCE_OUT/'segments/operational/complete.json'


def test_only_resume_lifecycle_maps_to_new_owner(isolated,monkeypatch):
    for route in c.VARIANTS:assert control._worker_segment('main_g159_'+route)=='resume_partial'
    assert control._worker_segment('main_g160_control')=='main_160'
    monkeypatch.setattr(c,'REPETITION',0)
    assert control._worker_segment('main_g159_control')=='main_150'


@pytest.mark.parametrize('mode',['prepare','resume_partial','replay','saved'])
def test_new_segment_identity(isolated,mode):assert control.segment_id(mode)==mode


def test_resume_only_fixed_repetition_and_no_block_offset(isolated,monkeypatch):
    with pytest.raises(ValueError):control.segment_id('resume_partial',159)
    monkeypatch.setattr(c,'REPETITION',0)
    with pytest.raises(ValueError):control.segment_id('resume_partial')


def prefix_fixture(monkeypatch):
    spec=dict(stage='main',game_index=159,challenger_seat=0,challenger_mode='candidate',opponent_mode='control')
    folder=lineage.old_game_folder(spec)
    initial=dict(state=dict(terminal=False,current_player=0),counter=0)
    put(folder/'spec.json',spec);put(folder/'initial.json',dict(**spec,initial=initial))
    names=dict(control='main_g150_control',candidate='main_g150_candidate')
    put(folder/'workers.json',names)
    for index in range(26):
        before=dict(state=dict(terminal=False,current_player=index%2),counter=index)
        after=dict(state=dict(terminal=False,current_player=(index+1)%2),counter=index+1)
        put(folder/'moves'/f'{index:06d}.json',dict(ply=index,before=before,after=after,completed=True,
            variant='candidate' if index%2==0 else 'control'))
    for name in names.values():put(lineage.worker_folder(name,'old')/'cleanup.json',dict(alive=False,errors=[]))
    put(c.LIVE_SOURCE_OUT/'freeze.json',dict(fixture=True))
    put(c.OUT/'segments/resume_partial/start.json',dict(fixture='newsegment'))
    monkeypatch.setattr(control.design,'make_schedule',lambda stage:[None]*159+[spec])
    monkeypatch.setattr(control.design,'initial_snapshot',lambda supplied:deepcopy(initial))
    monkeypatch.setattr(control,'validate_row',lambda *a,**kw:None)
    monkeypatch.setattr(control,'verify_closed_before',lambda *a:None)
    monkeypatch.setattr(control,'_both_operational_passed',lambda:[])
    monkeypatch.setattr(control,'decode_training_snapshot',lambda value:SimpleNamespace(terminal=False,current_player=value['state']['current_player'],counter=value['counter']))
    monkeypatch.setattr(control.rules,'serialize_snapshot',lambda value:dict(state=dict(terminal=value.terminal,current_player=value.current_player),counter=value.counter))
    return spec,folder


def test_resume_first_request_is_new_ply26_and_preserves_old_prefix(isolated,monkeypatch):
    spec,source=prefix_fixture(monkeypatch);before={p:root.file_hash(p) for p in source.rglob('*') if p.is_file()}
    events=[];workers={}
    def qualify(stage,index,destination,*unused):
        assert stage=='main' and index==159
        events.append('qualified')
        for route in c.VARIANTS:destination[f'main_g159_{route}']=SimpleNamespace(route=route,_log_sizes={})
    monkeypatch.setattr(control,'qualify_pair',qualify)
    class FirstDecision(Exception):pass
    def decision(child,spec,before,envelope,freeze,*,ply):
        assert events==['qualified'] and ply==26 and before['counter']==26
        events.append('new26');raise FirstDecision()
    monkeypatch.setattr(control,'decision',decision)
    env=SimpleNamespace(start=1.,wall_start=2.,deadline=300.,wall_deadline=400.,waited=0.,accounted_bytes=lambda:0)
    with pytest.raises(FirstDecision):control.resume_partial(workers,env,{},None)
    assert events==['qualified','new26']
    assert before=={p:root.file_hash(p) for p in source.rglob('*') if p.is_file()}
    admission=root.read(c.OUT/'resume_admission.json');continuation=root.read(lineage.game_folder(spec)/'continuation.json')
    assert admission['start_ply']==26 and len(admission['prefix'])==26 and admission['active_decision'] is False
    assert continuation['original_decisions_retried']==0 and continuation['game_replacements']==0


@pytest.mark.parametrize('fault',['missing','extra','incomplete','wrong_before','terminal'])
def test_bad_prefix_rejects_before_any_fresh_worker(isolated,monkeypatch,fault):
    spec,folder=prefix_fixture(monkeypatch)
    path=folder/'moves/000025.json';row=root.read(path)
    if fault=='missing':path.unlink()
    elif fault=='extra':put(folder/'moves/000026.json',row)
    else:
        if fault=='incomplete':row['completed']=False
        elif fault=='wrong_before':row['before']['counter']=-1
        else:row['after']['state']['terminal']=True
        path.write_text(root.canonical(row))
    called=[];monkeypatch.setattr(control,'qualify_pair',lambda *a:called.append(True))
    with pytest.raises(ValueError):control.resume_partial({},SimpleNamespace(),{},None)
    assert called==[]


def test_unchanged_scientific_algorithm_sources():
    original=Path(__file__).resolve().parents[1]/'tools/splendorust_independent_data_recovery_20261109/strength'
    current=Path(control.__file__).parent
    for name in ('worker.py','scientific.py','graph.py','adapters.py','analysis.py','qualification.py',
                 'corpus.py','checkpoint.py','design.py','boundary_adapter.py','parent_profile.py','accounting.py'):
        assert (original/name).read_bytes()==(current/name).read_bytes(),name
    assert transport.OBSERVER_MODULE=='tools.splendorust_independent_data_disk_recovery_20261109.strength.worker'
