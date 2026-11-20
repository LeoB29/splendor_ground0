"""Synthetic evaluator boundary/provenance tests; no models or source targets."""
from copy import deepcopy
import sys
from types import ModuleType, SimpleNamespace

import pytest

from tools.splendorust_training4096_20261120 import common as c, offline as o


def example(source='fresh_20261115', game=0, split='train', ply=0):
    return dict(source=source, game_index=game, ply=ply, split=split,
        row_sha256='1'*64, root_input_sha256='2'*64,
        mask=[True, True]+[False]*79, policy=[.75, .25]+[0.]*79,
        terminal_value=[1., -1.], root_q_win_credit=.25,
        value=[.3162765378334241, -.3162765378334241], tokens=[])


def outputs():
    return [0., 1.]+[0.]*79+[.25, -.25]


def test_production_prediction_retains_official_not_mixed_value(monkeypatch):
    monkeypatch.setattr(o, 'predict_outputs', lambda model, rows: [outputs() for _ in rows])
    rows=o.predict_batch(None, [example(), example('prior_20261107')])
    assert rows[0]['value_target2']==[1., -1.]
    assert rows[0]['root_q_win_credit']==.25
    assert o.identity(rows[0])!=o.identity(rows[1])
    assert 'tokens' not in rows[0]


@pytest.mark.parametrize('mutate', [
    lambda r:r.update(source='legacy'),
    lambda r:r.update(game_index=True),
    lambda r:r.update(value_target2=[.5,-.5]),
    lambda r:r.update(root_q_win_credit=True),
    lambda r:r['policy_target81'].__setitem__(2,.1),
    lambda r:r['legal_mask81'].__setitem__(0,1),
    lambda r:r['output83'].__setitem__(0,.1),
    lambda r:r['output83'].__setitem__(81,2.),
    lambda r:r.update(row_sha256='changed'),
    lambda r:r.update(extra='not schema'),
])
def test_reject_malformed_predictions(mutate):
    row=o.prediction_row(example(),outputs());mutate(row)
    with pytest.raises(ValueError):o.strict_prediction(row)


def test_official_label_required():
    x=example();del x['terminal_value']
    with pytest.raises(ValueError,match='official'):o.prediction_row(x,outputs())


def test_gate_runs_before_any_assessment_or_cache_read(monkeypatch):
    import tools.splendorust_training4096_20261120 as package
    events=[]
    def forbidden(*args,**kwargs):
        events.append('protected');raise AssertionError('Protected read before commitment')
    def refuse(*args,**kwargs):
        events.append('gate');raise ValueError('Uncommitted')
    for name, attrs in [('seal',{'validate_commitment':refuse}),
                        ('corpus',{'training_dataset':forbidden,'assessment_dataset':forbidden}),
                        ('selection',{'panel_indices':forbidden})]:
        mod=ModuleType(package.__name__+'.'+name)
        for key,value in attrs.items():setattr(mod,key,value)
        monkeypatch.setitem(sys.modules,mod.__name__,mod)
        monkeypatch.setattr(package,name,mod,raising=False)
    monkeypatch.setattr(c,'read',forbidden);monkeypatch.setattr(c,'artifact',forbidden)
    with pytest.raises(ValueError,match='Uncommitted'):o._evaluation_context()
    assert events==['gate']


def test_evaluation_cannot_silently_invoke_a_second_source_guard(tmp_path,monkeypatch):
    monkeypatch.setattr(c,'OUT',tmp_path)
    monkeypatch.setattr(o,'_evaluation_context',lambda *args:pytest.fail('No cache access without supplied startup'))
    with pytest.raises(ValueError,match='single verified startup'):
        o.run_evaluation(tmp_path/'jobs/offline/worker')


def test_complete_namespaced_panel_and_assessment_inventory():
    train=[]
    for source,n in [('prior_20261107',1024),('fresh_20261115',3072)]:
        for game in range(n):
            for ply in range(4):train.append(example(source,game,'train',ply))
    assessment=[example(game=g,split='validation') for g in range(3072,3200)]
    assessment += [example(game=g,split='offline') for g in range(3200,3328)]
    train.sort(key=o.identity)
    groups={'train':o._View(SimpleNamespace(index=train),range(len(train))),
        'validation':o._View(SimpleNamespace(index=assessment),range(128)),
        'offline':o._View(SimpleNamespace(index=assessment),range(128,256))}
    o._validate_group_inventory(groups)
    assessment[128]['game_index']=3072
    with pytest.raises(ValueError):o._validate_group_inventory(groups)


@pytest.fixture
def saved_predictions(tmp_path,monkeypatch):
    monkeypatch.setattr(c,'OUT',tmp_path)
    folder=tmp_path/'jobs/offline/worker'
    groups={'train':[example('prior_20261107')],
            'validation':[example(game=3072,split='validation')],
            'offline':[example(game=3200,split='offline')]}
    registry={mid:dict(id=mid,kind='parent' if mid=='parent' else
        ('historical' if mid.startswith('historical') else 'new'),
        checkpoint={'path':'synthetic','bytes':0,'sha256':'3'*64},weights_sha256='4'*64)
        for mid in o.MODEL_IDS}
    provenance={'test':'synthetic_authenticated_context'}
    monkeypatch.setattr(o,'_evaluation_context',lambda boundary:(registry,groups,provenance))
    startup=c.save(folder/'startup.json',{'runtime_sources':{'synthetic':True}})
    monkeypatch.setattr(o,'_startup_proof',lambda item,boundary:c.read(c.resolve(item)))
    entries=[]
    for mid in o.MODEL_IDS:
        batches=[]
        for split,examples in groups.items():
            rows=[o.prediction_row(x,outputs()) for x in examples]
            batches.append(c.save(folder/'predictions'/mid/split/'000000.json',dict(
                schema='training4096_prediction_batch_v1',model_id=mid,split=split,
                batch_index=0,predictor=registry[mid],provenance=provenance,rows=rows)))
        entries.append(dict(model_id=mid,predictor=registry[mid],batches=batches,
                            root_counts={s:1 for s in o.SPLITS},forward_calls=3))
    doc=dict(schema='training4096_offline_evaluation_v1',status='all_seven_fixed_predictions_complete',
        provenance=provenance,models=entries,forward_calls=21,optimizer_calls=0,
        historical_predictions_reused=False,all_seven_predictions_fresh=True,
        fresh_assessment_games=256,reused_development_games_used=0,training_panel_roots=16384,
        all_endpoints_committed_before_any_assessment_read=True,checkpoint_selection=False,
        reference_promoted=False,runtime_sources={'synthetic':True},startup=startup,**c.IDENTITY)
    c.save(folder/'evaluation_result.json',doc)
    return folder,doc


def replace_json(path,value):
    path.write_text(c.canonical(value)+'\n',encoding='utf-8')


def test_saved_prediction_inventory_is_model_free(saved_predictions):
    folder,_=saved_predictions
    was_loaded='torch' in sys.modules
    bundles,document=o.load_predictions(folder)
    assert tuple(bundles)==o.MODEL_IDS and document['forward_calls']==21
    assert o.verify_evaluation(folder)['model_calls']==0
    assert ('torch' in sys.modules)==was_loaded


def test_saved_labels_cannot_change_even_with_rehashed_batch(saved_predictions):
    folder,doc=saved_predictions
    item=doc['models'][0]['batches'][1];path=c.resolve(item);batch=c.read(path)
    batch['rows'][0]['root_q_win_credit']=.5
    replace_json(path,batch);doc['models'][0]['batches'][1]=c.artifact(path)
    replace_json(folder/'evaluation_result.json',doc)
    with pytest.raises(ValueError,match='original source'):o.load_predictions(folder)


def test_no_selective_model_omission(saved_predictions):
    folder,doc=saved_predictions;doc['models'].pop()
    replace_json(folder/'evaluation_result.json',doc)
    with pytest.raises(ValueError,match='seven'):o.load_predictions(folder)


def test_extra_prediction_file_rejected(saved_predictions):
    folder,_=saved_predictions
    (folder/'predictions/unregistered.json').write_text('{}')
    with pytest.raises(ValueError,match='extra prediction'):o.load_predictions(folder)


def test_checkpoint_binding_cannot_be_swapped(saved_predictions):
    folder,doc=saved_predictions
    doc['models'][0]['predictor']=deepcopy(doc['models'][1]['predictor'])
    replace_json(folder/'evaluation_result.json',doc)
    with pytest.raises(ValueError,match='endpoint'):o.load_predictions(folder)


def test_evaluator_source_receipt_cannot_be_substituted(saved_predictions):
    folder,doc=saved_predictions;doc['runtime_sources']={'changed':True}
    replace_json(folder/'evaluation_result.json',doc)
    with pytest.raises(ValueError,match='startup guard'):o.load_predictions(folder)
