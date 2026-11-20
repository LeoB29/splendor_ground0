from copy import deepcopy
from pathlib import Path
import pytest
from tools.splendorust_training4096_20261120 import common as c,corpus,selection,seal


def test_metadata_only_actual_inventory():
    result=selection.metadata()
    assert len(result['training_games'])==4096
    assert len(result['training_inventory'])==227203
    assert len(result['assessment_games'])==256
    assert result['assessment_protected_records_read']==0
    assert len(selection.validate_namespace(result['training_inventory']))==227203


def test_same_game_index_two_sources_remains_distinct():
    row=dict(source=selection.SOURCES[0],game_index=0,ply=0,split='train',row_sha256='a'*64,root_input_sha256='b'*64)
    assert len(selection.validate_namespace([row,dict(row,source=selection.SOURCES[1])]))==2
    with pytest.raises(ValueError):selection.validate_namespace([row,row])


@pytest.mark.parametrize('change',[{'source':'unknown'},{'split':'validation'},{'game_index':True},{'ply':-1},{'row_sha256':'z'*64}])
def test_namespace_counterexamples(change):
    row=dict(source=selection.SOURCES[0],game_index=0,ply=0,split='train',row_sha256='a'*64,root_input_sha256='b'*64)
    row.update(change)
    with pytest.raises(ValueError):selection.validate_namespace([row])


def test_training_rejects_assessment_before_any_protected_read(monkeypatch):
    training=dict(source=selection.SOURCES[0],spec=dict(game_index=0,split='train'),artifact={},source_game={},roots=[])
    fresh=dict(source=selection.SOURCES[1],spec=dict(game_index=3072,split='validation'),artifact={},source_game={},roots=[])
    monkeypatch.setattr(selection,'load',lambda boundary=lambda:None:dict(training_games=[training],assessment_games=[fresh]))
    opened=[];monkeypatch.setattr(corpus.data,'read_artifact',lambda item:opened.append(item))
    with pytest.raises(ValueError):corpus.decode_game(fresh,'training')
    with pytest.raises(ValueError):corpus.decode_game(fresh,'assessment')
    assert not opened


def test_descriptor_substitution_rejected_before_target_io(monkeypatch):
    game=dict(source=selection.SOURCES[1],spec=dict(game_index=0,split='train'),artifact={'sha256':'a'},source_game={},roots=[])
    monkeypatch.setattr(selection,'load',lambda boundary=lambda:None:dict(training_games=[game],assessment_games=[]))
    bad=deepcopy(game);bad['artifact']['sha256']='b';opened=[]
    monkeypatch.setattr(corpus.data,'read_artifact',lambda item:opened.append(item))
    with pytest.raises(ValueError):corpus.decode_game(bad,'training')
    assert not opened


def test_actual_corrected_fixture_cache_collator(tmp_path):
    # Keep the preflight coordinator free of Torch; this child loads no model.
    import subprocess,sys,json
    code="""import json,sys
from pathlib import Path
from tools.splendorust_training4096_20261120 import common as c,corpus
c.OUT=Path(sys.argv[1])
from tools.future_training_rules_20261021 import verify_runtime_sources
verify_runtime_sources()
r=corpus.qualify_training_inputs(c.OUT/'qualification')
assert len(r['qualification_examples'])==32 and r['qualification_structural_examples']
assert all(x['source']=='qualification_excluded' for x in r['qualification_examples'])
print(json.dumps(r['proof']))
"""
    result=subprocess.run([sys.executable,'-B','-c',code,str(tmp_path)],capture_output=True,text=True,timeout=90)
    assert result.returncode==0,result.stderr
    proof=json.loads(result.stdout)
    assert proof['actual_mmap_training_path_qualified']
    assert proof['actual_selected_loader_and_collator_exercised']
    assert proof['fresh_assessment_records_read']==0


def test_dataset_rejects_negative_and_missing_shards():
    data=corpus.Dataset([],[],None)
    with pytest.raises(ValueError):data[-1]
    with pytest.raises(ValueError):data[0]


def test_protected_initial_files_rejected_without_open(monkeypatch,tmp_path):
    game=dict(artifact={'path':str(tmp_path/'targets/g3000.json')},source_game={'path':str(tmp_path/'games/g3000/target_game.json')})
    monkeypatch.setattr(selection,'load',lambda:dict(assessment_games=[game]))
    for path in (tmp_path/'targets/g3000.json',tmp_path/'games/g3000/target_game.json',tmp_path/'games/g3000/moves/000001.json',c.ASSESS_CACHE/'index.jsonl'):
        with pytest.raises(ValueError):selection.validate_initial_inputs([{'path':str(path)}])
    assert selection.validate_initial_inputs([{'path':str(tmp_path/'catalogue.json')}])['protected_file_reads']==0


def test_actual_engineering_game_through_production_decoder(monkeypatch):
    from tools.splendorust_independent_data_20261107.collection import common as cc,data
    from tools.splendorust_independent_data_20261107 import schedule
    gate=cc.read(cc.OUT/'qualification_gate.json');proof=cc.read(c.resolve(gate['audit']));spec=schedule.qualification_schedule()[0]
    target=proof['engineering_targets'][0];raw=data.read_artifact(target);examples=corpus.source_corpus._parse_target(raw,spec)['examples']
    roots=[dict(source='qualification_excluded',game_index=spec['game_index'],ply=x['ply'],split=spec['split'],row_sha256=x['row_sha256'],root_input_sha256=x['root_input_sha256']) for x in examples]
    game=dict(source='qualification_excluded',spec=spec,artifact=target,source_game=proof['engineering_game_proofs'][0]['document'],roots=roots)
    # Only the prior accepted excluded game enters this test's explicit allowlist.
    monkeypatch.setattr(selection,'allowed_game',lambda actual,*a,**k:c.require(actual==game,'Fixture-only exact game'))
    decoded,index=corpus.decode_game(game,'training')
    independent,independent_index=corpus.decode_game(game,'training',independent=True)
    assert decoded==independent and index==independent_index and len(index)==len(roots)
    assert all(x['source']=='qualification_excluded' for x in index)


def test_actual_prior_cache_projects_only_accepted_training_rows():
    dataset=corpus.prior_training_dataset()
    assert len(dataset)==57022
    assert all(x['source']==selection.SOURCES[0] and x['split']=='train' for x in dataset.index)
    assert corpus._inventory(dataset.index)==selection.load()['training_inventory'][:57022]
    assert dataset[0]['source']==selection.SOURCES[0] and len(dataset[57021]['tokens'])==1488


def test_final_source_preservation_is_sealed_first(tmp_path,monkeypatch):
    def reject(*a,**k):raise ValueError('still sealed')
    monkeypatch.setattr(seal,'acquire_release',reject)
    monkeypatch.setattr(selection,'load',lambda *a,**k:pytest.fail('no catalogue before gate'))
    with pytest.raises(ValueError):corpus.verify_consumed_sources(tmp_path)


def test_consumed_source_hash_tamper_rejected(tmp_path,monkeypatch):
    monkeypatch.setattr(c,'OUT',tmp_path)
    move=c.save(tmp_path/'move.json',{'saved':1});target=c.save(tmp_path/'target.json',{'target':1})
    spec={'game_index':0,'split':'train'}
    document=dict(spec=spec,reason='completed',rows=[move],**c.IDENTITY)
    doc=c.save(tmp_path/'game.json',document)
    game=dict(source=selection.SOURCES[1],spec=spec,artifact=target,source_game=doc,roots=[{'ply':0}])
    monkeypatch.setattr(selection,'allowed_game',lambda actual,*a,**k:c.require(actual==game,'Exact saved fixture'))
    monkeypatch.setattr(corpus,'_read_source',lambda item,*a:c.read(c.resolve(item)))
    assert corpus._verify_game_sources(game,'training',None,lambda:None)==[target,doc,move]
    c.resolve(move).write_text('{"saved":2}')
    with pytest.raises(ValueError):corpus._verify_game_sources(game,'training',None,lambda:None)
