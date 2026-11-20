"""Model-free provenance/split controls for the new collection and old/new union."""
from copy import deepcopy
import ast
from collections import Counter
from pathlib import Path
import pytest
from . import common as c,corpus
from .collection import common as cc,data


@pytest.fixture
def workspace(tmp_path,monkeypatch):
    monkeypatch.setattr(c,'ROOT',tmp_path);monkeypatch.setattr(c,'OUT',tmp_path/'new')
    monkeypatch.setattr(c,'PLAN',c.OUT/'freeze.json')
    monkeypatch.setattr(cc,'ROOT',tmp_path);monkeypatch.setattr(cc,'OUT',c.OUT/'collection')
    monkeypatch.setattr(cc,'PLAN',cc.OUT/'plan.json')
    corpus._CATALOGUES.clear()
    return tmp_path


def source_fixture(folder,*,fresh):
    counts=c.NEW_GAME_COUNTS if fresh else {'train':1024,'validation':64,'offline':64}
    games=[];index=0
    for split,count in counts.items():
        for _ in range(count):
            spec=dict(game_id=f'g{index:04d}',game_index=index,split=split,
                seed_decimal=str(index+(100000 if fresh else 200000)),quota=512,world_pool=3,cpuct=.4)
            games.append(dict(spec=spec,roots=[dict(game_index=index,ply=0,split=split,
                row_sha256='a'*64,root_input_sha256='b'*64)]))
            index+=1
    cells=[dict(specifications=[g['spec'] for g in games[start:start+16]]) for start in range(0,index,16)]
    plan=c.save(folder/'plan.json',dict(cells=cells))
    cat=c.save(folder/'root_catalogue.json',dict(schema='collection4096_new_root_catalogue_v1' if fresh else
        'independent_data_root_catalogue_v1',teacher_simulations=512,complete_game_count=index,
        targets_in_catalogue=False,source_plan=plan,games=games))
    accepted=dict(schema='collection4096_new_corpus_acceptance_v1' if fresh else
        'independent_corrected_corpus_acceptance_v1',status='accepted',passed=True,
        game_count=index,independent_setup_count=index,partition_counts=counts,catalogue=cat,
        teacher_simulations=512,raw_semantics_and_all_transitions_verified=True,
        selected_model_replays_verified=True,source_plan=plan,**data.IDENTITY)
    item=c.save(folder/'corpus_acceptance.json',accepted)
    return dict(acceptance=item,catalogue=cat),games


@pytest.fixture
def union_plan(workspace):
    prior,_=source_fixture(c.OUT/'prior',fresh=False)
    fresh,_=source_fixture(cc.OUT,fresh=True)
    return dict(existing_training_corpus=prior)


def test_union_has_4096_train_and_256_only_fresh_assessment(union_plan):
    result=corpus.build_union_inventory(union_plan)
    proof=corpus.validate_union_inventory(result)
    assert proof['training_games']==4096 and proof['fresh_assessment_games']==256
    assert Counter(r['source'] for r in result['training_members'])=={'prior_20261107':1024,'fresh_20261115':3072}
    assert all(r['source']=='fresh_20261115' for r in result['fresh_assessment_members'])
    assert len(result['reused_development_members_excluded_from_fresh_assessment'])==128
    assert not result['future_scientific_training_authorized']


def test_old_and_new_same_local_ids_keep_distinct_provenance(union_plan):
    result=corpus.build_union_inventory(union_plan)
    roots=[r for r in result['training_members'] if r['game_id']=='g0000']
    assert len(roots)==2 and {r['source'] for r in roots}=={'prior_20261107','fresh_20261115'}


def test_union_saved_inventory_is_rebuilt_independently(union_plan):
    receipt=corpus.union_inventory(union_plan)
    assert corpus.verify_union_inventory(receipt['inventory'],union_plan)==receipt['verification']
    path=c.resolve(receipt['inventory']);value=c.read(path);value['new_training_games']=1
    path.write_text(c.canonical(value))
    with pytest.raises(ValueError):corpus.verify_union_inventory(c.artifact(path),union_plan)


@pytest.mark.parametrize('mutate',[
    lambda v:v['fresh_assessment_members'].__setitem__(0,deepcopy(v['reused_development_members_excluded_from_fresh_assessment'][0])),
    lambda v:v['training_members'].__setitem__(0,deepcopy(v['reused_development_members_excluded_from_fresh_assessment'][0])),
    lambda v:v['training_members'].__setitem__(0,deepcopy(v['training_members'][1])),
    lambda v:v['fresh_assessment_members'][0].update(source='prior_20261107'),
    lambda v:v['training_members'][0].update(split='validation'),
    lambda v:v['training_members'][0].update(game_index=1024),
    lambda v:v['training_members'][0].update(game_id='renumbered'),
    lambda v:v.update(future_scientific_training_authorized=True),
    lambda v:v.update(original_targets_modified=True),
    lambda v:v.update(fresh_assessment_sealed_until_future_checkpoints_committed=False),
])
def test_union_rejects_leakage_relabelling_and_extra_authority(union_plan,mutate):
    value=corpus.build_union_inventory(union_plan);mutate(value)
    with pytest.raises(ValueError):corpus.validate_union_inventory(value)


def test_changed_prior_catalogue_cannot_enter_union(union_plan):
    path=c.resolve(union_plan['existing_training_corpus']['catalogue'])
    with path.open('a') as stream:stream.write('\n')
    with pytest.raises(ValueError):corpus.build_union_inventory(union_plan)


@pytest.mark.parametrize('field,value',[
    ('schema','independent_corrected_corpus_acceptance_v1'),('game_count',1152),
    ('independent_setup_count',1152),('partition_counts',{'train':1024,'validation':64,'offline':64}),
    ('status','incomplete'),('ruleset_id','historical')])
def test_new_corpus_receipt_rejects_old_schema_counts_and_rules(union_plan,field,value):
    path=cc.OUT/'corpus_acceptance.json';receipt=cc.read(path);receipt[field]=value
    path.write_text(c.canonical(receipt))
    with pytest.raises(ValueError):corpus.selection_catalogue(c.artifact(path))


def test_overlapping_fresh_seed_rejected_even_with_matching_catalogue_hash(union_plan):
    path=cc.OUT/'root_catalogue.json';cat=cc.read(path);cat['games'][0]['spec']['seed_decimal']='200000'
    path.write_text(c.canonical(cat));planpath=cc.PLAN;plan=cc.read(planpath)
    plan['cells'][0]['specifications'][0]['seed_decimal']='200000';planpath.write_text(c.canonical(plan))
    cat['source_plan']=c.artifact(planpath);path.write_text(c.canonical(cat))
    acceptancepath=cc.OUT/'corpus_acceptance.json';acceptance=cc.read(acceptancepath)
    acceptance.update(source_plan=c.artifact(planpath),catalogue=c.artifact(path));acceptancepath.write_text(c.canonical(acceptance))
    with pytest.raises(ValueError,match='disjoint'):corpus.build_union_inventory(union_plan)


def test_actual_loader_qualification_function_preserved_exactly():
    base=Path(__file__).resolve().parents[1]/'splendorust_independent_data_20261107/corpus.py'
    def function(path,name):
        tree=ast.parse(path.read_text(encoding='utf-8'))
        return ast.dump(next(node for node in tree.body if isinstance(node,ast.FunctionDef) and node.name==name),include_attributes=False)
    assert function(base,'qualify_training_inputs')==function(Path(corpus.__file__),'qualify_training_inputs')


@pytest.mark.parametrize('missing_last_cell',[False,True])
def test_accept_all_requires_exact_208_cells_and_new_partition(workspace,monkeypatch,missing_last_cell):
    from . import schedule
    specs=schedule.collection_schedule()
    assert len(specs)==3328
    cells=[dict(cell_id=f'c{index:03d}',specifications=specs[index*16:(index+1)*16]) for index in range(208)]
    c.save(cc.PLAN,dict(cells=cells))
    dummy=c.save(c.OUT/'synthetic_metadata_fixture.json',{'no_models':True})
    calls=[]
    def cell_audit(plan,cell,container):
        calls.append(cell['cell_id'])
        return dict(status='passed',games=16,full_cohort_verified=True,
            game_proofs=[dict(target=dummy,document=dummy) for _ in cell['specifications']])
    monkeypatch.setattr(corpus.audit,'set_boundary',lambda boundary:None)
    monkeypatch.setattr(corpus.audit,'audit_cell',cell_audit)
    monkeypatch.setattr(corpus,'_target',lambda item,spec:dict(examples=[dict(ply=0,row_sha256='a'*64,root_input_sha256='b'*64)]))
    if missing_last_cell:
        with pytest.raises(ValueError):corpus.accept_all(dict(cells=cells[:-1]))
        assert len(calls)==207 and not (cc.OUT/'corpus_acceptance.json').exists()
    else:
        result=corpus.accept_all(dict(cells=cells))
        assert result['game_count']==3328 and result['partition_counts']=={'train':3072,'validation':128,'offline':128}
        assert len(calls)==208
        catalogue,_=corpus.selection_catalogue(c.artifact(cc.OUT/'corpus_acceptance.json'))
        assert len(catalogue)==3328
