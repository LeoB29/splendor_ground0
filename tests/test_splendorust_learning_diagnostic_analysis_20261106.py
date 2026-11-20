"""Model-free scientific/statistical checks of the new saved-row analysis."""
from copy import deepcopy
import hashlib
import json
import math
import struct

import numpy as np
import pytest

from tools.splendorust_learning_diagnostic_20261106 import analysis as a


def f32(x):
    return struct.unpack('<f',struct.pack('<f',x))[0]


def row(game=192,ply=0,split='validation',logit=0.,value=.25):
    return dict(game_index=game,ply=ply,split=split,
        root_input_sha256=hashlib.sha256(f'input:{game}:{ply}'.encode()).hexdigest(),
        row_sha256=hashlib.sha256(f'row:{game}:{ply}'.encode()).hexdigest(),
        legal_mask81=[True,True]+[False]*79,policy_target81=[1.,0.]+[0.]*79,
        value_target2=[1.,-1.],output83=[f32(logit),0.]+[0.]*79+[f32(value),f32(-value)])


def small_bundles():
    bundles=[]
    for spec in a.c.fits():
        evaluations={}
        for step in a.STEPS:
            amount=(step/668)*(.2 if spec['breadth']=='broad' else .1)*(1 if spec['scope']=='joint' else .5)
            values=.25+(step/668)*.05 if spec['scope']=='joint' else .25
            evaluations[str(step)]={split:[row(g,ply,split,amount,values) for g in ids for ply in (0,1)]
                for split,ids in [('train',[0,1]),('validation',[192,193]),('offline',[224,225])]}
        bundles.append(dict(spec=spec,evaluations=evaluations))
    return bundles


def small_analysis(bundles=None):
    return a.analyze_bundle(small_bundles() if bundles is None else bundles,draws=100,
        expected_heldout={'validation':[192,193],'offline':[224,225]},strict_counts=False)


def test_independent_masked_losses_and_large_common_logits():
    item=row();item['output83'][2]=f32(1e30)
    result=a.prediction_metrics(item)
    assert result['policy']==pytest.approx(math.log(2))
    assert result['value']==pytest.approx(.75**2)
    item['output83'][:2]=[f32(3e38),f32(3e38)]
    assert a.prediction_metrics(item)['policy']==pytest.approx(math.log(2))


@pytest.mark.parametrize('field,value',[
    ('policy_target81',[.5,0.]+[0.]*79),
    ('value_target2',[1.,1.]),
    ('legal_mask81',[False]*81),
    ('root_input_sha256','bad'),
    ('split','test'),
])
def test_invalid_targets_and_identity_rejected(field,value):
    item=row();item[field]=value
    with pytest.raises(ValueError):a.prediction_metrics(item)


def test_illegal_target_and_non_float32_output_rejected():
    item=row();item['policy_target81']=[0.,0.,1.]+[0.]*78
    with pytest.raises(ValueError,match='legal visit'):a.prediction_metrics(item)
    item=row();item['output83'][0]=.1
    with pytest.raises(ValueError,match='float32'):a.prediction_metrics(item)


def test_equal_game_weighting_not_position_pseudoreplication():
    rows=[row(192,0,logit=2.)]+[row(193,p,logit=-2.) for p in range(3)]
    summary=a.summarize_rows(rows,expected_games=[192,193])
    losses=[a.prediction_metrics(x)['policy'] for x in rows]
    assert summary['equal_game_means']['policy']==pytest.approx((losses[0]+losses[1])/2)
    assert summary['equal_game_means']['policy']!=pytest.approx(sum(losses)/4)
    assert summary['game_count']==2 and summary['root_count']==4


def test_pairing_duplicate_root_and_signed_zero_guards():
    parent=[row()];changed=deepcopy(parent);changed[0]['row_sha256']='f'*64
    with pytest.raises(ValueError,match='same source'):a.summarize_rows(changed,parent)
    with pytest.raises(ValueError,match='unique'):a.summarize_rows(parent+parent)
    parent=[row(value=0.)];changed=deepcopy(parent);changed[0]['output83'][81]=-0.
    with pytest.raises(ValueError,match='exact parent'):a.summarize_rows(changed,parent,policy_only=True)


def test_bootstrap_uses_whole_games_and_has_reproducible_shared_indices():
    indices=a.bootstrap_indices(2,np.random.Generator(np.random.PCG64(20261106)),20000)
    proof=a.describe_difference([-1.,1.],[1.,1.],indices)
    assert proof['absolute_difference']==0 and proof['nominal_ci95']==[-1.,1.]
    assert proof['game_count']==2 and proof['games_lower']==proof['games_higher']==1
    again=a.bootstrap_indices(2,np.random.Generator(np.random.PCG64(20261106)),20000)
    assert np.array_equal(indices,again)
    with pytest.raises(ValueError):a.describe_difference([-1.,1.],[1.,1.],indices[:,:1])


def test_relative_change_is_ratio_of_means_and_zero_reference_is_null():
    indices=np.array([[0,1],[1,0]],dtype=np.int64)
    result=a.describe_difference([1.,1.],[1.,9.],indices)
    assert result['relative_percent']==20.
    assert a.describe_difference([0.,0.],[0.,0.],indices)['relative_percent'] is None
    assert a.describe_difference([0.,0.],None,indices)['relative_percent'] is None


def test_fixed_matrix_primary_secondary_and_repetition_mean():
    result=small_analysis()
    assert len(result['contrasts'])==180
    selected=[x for x in result['contrasts'] if x['kind']=='breadth' and x['scope']=='joint'
        and x['metric']=='policy' and x['checkpoint_step']==668 and x['split']=='pooled']
    assert [x['repetition'] for x in selected]==[0,1,'mean']
    assert all(x['absolute_difference']<0 and x['game_count']==4 for x in selected)
    assert selected[2]['absolute_difference']==pytest.approx(sum(x['absolute_difference'] for x in selected[:2])/2)
    assert result['bootstrap']==small_analysis()['bootstrap']
    assert result['stronger_player_established'] is False and result['checkpoint_selection'] is False
    assert all(row['endpoint']=='diagnostic' for row in result['contrasts'] if row['checkpoint_step']==334)
    assert all(row['endpoint']=='primary' for row in selected)


def test_plot_falls_back_to_absolute_change_for_zero_parent_loss():
    result=small_analysis()
    for fit in result['summaries'].values():
        fit['checkpoints']['0']['pooled']['equal_game_means']['value']=0.
    series,label=a.trajectory_series(result,'pooled','value')
    assert 'Absolute' in label and 'zero baseline' in label
    assert all(values[0]==0 and all(math.isfinite(x) for x in values) for values in series.values())


def test_missing_fixed_fit_or_shifted_parent_baseline_rejected():
    bundles=small_bundles()
    with pytest.raises(ValueError,match='eight-cell'):small_analysis(bundles[:-1])
    bundles[2]['evaluations']['0']['offline'][0]['output83'][0]=1.
    with pytest.raises(ValueError,match='pristine-parent'):small_analysis(bundles)


def fixture_evaluator(tmp_path,monkeypatch):
    monkeypatch.setattr(a.c,'ROOT',tmp_path);monkeypatch.setattr(a.c,'OUT',tmp_path/'out')
    selection={'schema':'fixture_selection'};selection_item=a.c.save(a.c.OUT/'selection.json',selection)
    spec=a.c.fits()[0];folder=a.c.OUT/'diagnostics'/spec['fit_id']/'worker';checkpoints=[]
    for step in a.STEPS:
        weight=f'{step:064x}';metadata=dict(spec,checkpoint_step=step,weights_sha256=weight,selection_sha256=a.c.digest(selection))
        manifest=a.c.save(a.c.OUT/'fits'/spec['fit_id']/f'step_{step:03d}'/'checkpoint_manifest.json',
            dict(schema='learning_diagnostic_checkpoint_manifest_v1',metadata=metadata))
        batches=[]
        for split,game in [('train',0),('validation',192),('offline',224)]:
            batches.append(a.c.save(folder/'evaluation'/f'step_{step:03d}'/split/'000000.json',
                dict(spec,checkpoint_step=step,split=split,batch_index=0,selection_sha256=a.c.digest(selection),
                     weights_sha256=weight,checkpoint_manifest=manifest,rows=[row(game,0,split)])))
        probes=[]
        for index in range(4):
            probes.append(a.c.save(folder/'probes'/f'step_{step:03d}'/f'{index:02d}.json',dict(spec,
                checkpoint_step=step,probe_index=index,selection_sha256=a.c.digest(selection),
                checkpoint_manifest=manifest,checkpoint_weights_sha256=weight,optimizer_calls=0,forward_calls=1,
                backward_probes=2,weights_unchanged=True,rng_unchanged=True,zero_optimizer_calls=True,
                policy={'norm':1.,'shared_norm':1.,'finite':True},value={'norm':2.,'shared_norm':2.,'finite':True},
                shared_dot=.5,shared_cosine=.25,undefined_cosine=False)))
        checkpoints.append(dict(checkpoint_step=step,checkpoint_manifest=manifest,batches=batches,
            gradient_probes=probes,weights_sha256=weight,group_displacement={},forward_calls=7,backward_probes=8))
    result=dict(spec,schema='learning_diagnostic_evaluation_v1',status='diagnostics_complete',
        selection_sha256=a.c.digest(selection),optimizer_calls=0,opened_heldout_exploratory_only=True,
        checkpoint_selection=False,strength_claim=False,checkpoints=checkpoints,backward_probes=24,forward_calls=21)
    a.c.save(folder/'diagnostics_result.json',result)
    return {'selection':selection_item},spec,folder


def test_actual_saved_batch_schema_canonical_digest_and_probe_inventory(tmp_path,monkeypatch):
    plan,spec,folder=fixture_evaluator(tmp_path,monkeypatch)
    assert plan['selection']['sha256']!=a.c.digest(a.c.read(a.c.resolve(plan['selection'])))
    bundle,provenance=a.load_bundle(plan,spec,folder)
    assert len(bundle['evaluations'])==3
    assert len(provenance['parameter_and_gradient_diagnostics'])==3
    assert sum(len(x['gradient_probes']) for x in provenance['parameter_and_gradient_diagnostics'])==12


def test_omitted_extra_or_changed_prediction_artifact_rejected(tmp_path,monkeypatch):
    plan,spec,folder=fixture_evaluator(tmp_path,monkeypatch)
    extra=folder/'evaluation/step_000/train/000001.json';extra.write_text('{}')
    with pytest.raises(ValueError,match='extra prediction'):a.load_bundle(plan,spec,folder)
    extra.unlink();path=folder/'evaluation/step_000/train/000000.json';path.write_text('{}')
    with pytest.raises(ValueError,match='artifact changed'):a.load_bundle(plan,spec,folder)


def test_tampered_zero_norm_gradient_cosine_rejected(tmp_path,monkeypatch):
    plan,spec,folder=fixture_evaluator(tmp_path,monkeypatch)
    path=folder/'probes/step_000/00.json';probe=json.loads(path.read_text());probe['policy']['shared_norm']=0.
    probe['undefined_cosine']=True;probe['shared_cosine']=0.;path.write_text(json.dumps(probe))
    result_path=folder/'diagnostics_result.json';result=json.loads(result_path.read_text())
    result['checkpoints'][0]['gradient_probes'][0]=a.c.artifact(path);result_path.write_text(json.dumps(result))
    with pytest.raises(ValueError,match='invented zero-norm'):a.load_bundle(plan,spec,folder)
