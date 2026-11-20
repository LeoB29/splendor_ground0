"""Five fixed predictors and paired saved-game value-target diagnostics.

Pooled mixed-minus-terminal outcome MSE is diagnostic primary. These positions
were previously examined; only the fresh live games establish playing strength.
No diagnostic loss selects a checkpoint or admission to strength games.
"""
from __future__ import annotations
from collections import defaultdict
from copy import deepcopy
import hashlib
import math
from pathlib import Path
import struct
import sys

from tools.splendorust_policy_head_20261104 import training as accepted
from . import common as c
from tools.splendorust_full_corpus_20261110 import sampling, fit, targets
from tools.splendorust_independent_data_20261107 import fit as original_fit
from tools.splendorust_independent_data_20261107.collection import data as source_data

MODEL_IDS = ("parent", "r0_mixed_joint", "r1_mixed_joint", "r0_terminal_joint", "r1_terminal_joint")
SPLITS = ("validation", "offline")
DRAW_COUNT = 20000
require = c.require


def identity(row):
    require(type(row.get("game_index")) is int and type(row.get("ply")) is int and
            type(row.get("split")) is str, "strict prediction root identity")
    return row["game_index"], row["ply"], row["split"]


def target_identity(row):
    return {k: row[k] for k in ("game_index", "ply", "split", "root_input_sha256", "row_sha256",
                                "legal_mask81", "policy_target81", "value_target2")}


def strict_prediction(row):
    require(set(row) == {"game_index","ply","split","root_input_sha256","row_sha256",
                         "legal_mask81","policy_target81","value_target2","output83"}, "complete exact prediction schema")
    identity(row)
    for key in ("root_input_sha256", "row_sha256"):
        sampling._sha(row[key])
    require(type(row["output83"]) is list and len(row["output83"]) == 83 and
            all(type(x) is float and math.isfinite(x) and struct.unpack("<f",struct.pack("<f",x))[0] == x
                for x in row["output83"]), "exact finite saved float32 outputs")
    require(all(-1. <= x <= 1. for x in row["output83"][81:]), "bounded value predictions")
    require(type(row["value_target2"]) is list and len(row["value_target2"]) == 2 and
            all(type(x) in (int,float) and x in (-1.,0.,1.) for x in row["value_target2"]), "genuine signed values, no booleans")
    return accepted.prediction_losses(row)


def _rows_for_model(selection, heldout, train_union, model_id):
    train=[train_union[tuple(k)] for k in selection['diagnostic_train_root_ids']]
    return {'train':train,**{split:[x for x in heldout if x['split']==split] for split in SPLITS}}


def prediction_bindings(committed, *, boundary=lambda: None):
    """Use the independently checked pristine starts and fixed final manifests."""
    bindings={}; parent_weights=set()
    for row in committed["fits"]:
        result=fit.read_json(row["fit_result"],boundary=boundary)
        zero=fit.read_json(result["checkpoints"]["0"],boundary=boundary)
        final=fit.read_json(row["final_checkpoint"],boundary=boundary)
        fit.validate_checkpoint_metadata(zero["metadata"]);fit.validate_checkpoint_metadata(final["metadata"])
        require(zero["metadata"]["checkpoint_step"]==0 and final["metadata"]["checkpoint_step"]==1782,
                "verified pristine parent and prospective fixed final")
        parent_weights.add(zero["metadata"]["weights_sha256"])
        bindings[row["spec"]["fit_id"]]=dict(checkpoint=row["final_checkpoint"],weights_sha256=final["metadata"]["weights_sha256"])
    require(len(parent_weights)==1,"all four matched full-corpus fits begin at the exact same pristine parent")
    require(c.sha(accepted.PARENT_CHECKPOINT)==accepted.PARENT_SHA256,"original reference archive")
    bindings["parent"]=dict(checkpoint=c.artifact(accepted.PARENT_CHECKPOINT),weights_sha256=next(iter(parent_weights)))
    return bindings


def search_targets(selection,heldout,*,boundary=lambda:None):
    """Labels from the separately independently source-verified immutable cache."""
    result=[]
    for offset in heldout.indices:
        boundary();row=heldout.cache['index'][offset]
        require(row['split'] in SPLITS,'Only reused diagnostic source roots')
        result.append(dict(game_index=row['game_index'],ply=row['ply'],split=row['split'],
            row_sha256=row['row_sha256'],root_input_sha256=row['root_input_sha256'],actor=row['actor'],
            root_q_signed=row['root_q_signed'],root_q_win_credit=row['root_q_win_credit'],source_move=row['root_q_source']))
    return result


def run_evaluation(selection, commitment, folder, *, boundary=lambda: None, runtime_sources=None):
    """Commit all four matched full-corpus fits before loading held-out examples or constructing models."""
    folder=Path(folder)
    committed=fit.validate_commitment(selection,commitment,boundary=boundary)
    expected_models=prediction_bindings(committed,boundary=boundary)
    fit.engine(folder,runtime_sources=runtime_sources)
    import torch
    heldout=sampling.load_heldout(selection,commitment,boundary=boundary)
    labels=search_targets(selection,heldout,boundary=boundary)
    search_item=c.save(folder/'search_value_targets.json',dict(schema='full_corpus_saved_search_labels_v1',
        selection_sha256=c.digest(selection),rows=labels,unit='signed_actor_win_credit',teacher_simulations=512,
        reused_diagnostic_only=True,not_official_outcome=True))
    train_union={(x['game_index'],x['ply']):x for x in sampling.load_diagnostic_train(selection,boundary=boundary)}
    results=[]
    for model_id in MODEL_IDS:
        boundary()
        if model_id == "parent":
            checkpoint=c.artifact(accepted.PARENT_CHECKPOINT); model=accepted.load_parent()
        else:
            row=next(x for x in committed["fits"] if x["spec"]["fit_id"]==model_id)
            checkpoint=row["final_checkpoint"]; model,_,manifest=fit.load_checkpoint(checkpoint,boundary=boundary)
            require(manifest["metadata"]["checkpoint_step"]==1782, "only final fixed endpoint evaluated")
        model.eval(); weights=accepted.state_digest(model); batches=[]; counts={}
        require(expected_models[model_id]==dict(checkpoint=checkpoint,weights_sha256=weights),"actual predictor is exact fixed committed endpoint")
        groups=_rows_for_model(selection,heldout,train_union,model_id)
        for split,examples in groups.items():
            counts[split]=len(examples)
            for index,offset in enumerate(range(0,len(examples),32)):
                boundary(); subset=examples[offset:offset+32]
                tokens=sampling.collate_inputs(subset,[1.]*len(subset))["tokens"]
                with torch.no_grad(): outputs=accepted.tensor_forward(model,tokens).tolist()
                rows=[fit.prediction_row(x,y) for x,y in zip(subset,outputs,strict=True)]
                for row in rows: strict_prediction(row)
                batches.append(c.save(folder/"predictions"/model_id/split/f"{index:06d}.json",
                    dict(schema="full_corpus_prediction_batch_v1",model_id=model_id,split=split,batch_index=index,
                         checkpoint=checkpoint,weights_sha256=weights,selection_sha256=c.digest(selection),
                         commitment=commitment,rows=rows)))
        results.append(dict(model_id=model_id,checkpoint=checkpoint,weights_sha256=weights,
                            batches=batches,root_counts=counts,forward_calls=len(batches)))
        del model
    value=dict(schema="full_corpus_offline_evaluation_v1",status="all_fixed_predictions_complete",
               selection_sha256=c.digest(selection),commitment=commitment,models=results,
               optimizer_calls=0,forward_calls=sum(x["forward_calls"] for x in results),
               all_fits_committed_before_heldout=True,checkpoint_selection=False,heldout_games=128,
               reference_promoted=False,search_value_targets=search_item,**sampling.IDENTITY)
    item=c.save(folder/"evaluation_result.json",value);boundary();return item


def _mean_game_rows(rows):
    by_game=defaultdict(list)
    for row in rows: by_game[row["game_index"]].append(strict_prediction(row))
    return [{"game_index":game,"roots":len(values),
             **{metric:math.fsum(x[metric] for x in values)/len(values) for metric in ("policy","value")}}
            for game,values in sorted(by_game.items())]


def prediction_changes(bundles):
    """Descriptive equal-game changes; raw policy agreement is not strength."""
    results={}
    for model in MODEL_IDS[1:]:
        games=defaultdict(list)
        for split in SPLITS:
            for row,parent in zip(bundles[model][split],bundles['parent'][split],strict=True):
                require(target_identity(row)==target_identity(parent),'Same-input prediction change evidence')
                legal=[i for i,ok in enumerate(row['legal_mask81']) if ok]
                best=lambda output:max(legal,key=lambda i:(output[i],-i))
                chosen=best(row['output83']);parent_chosen=best(parent['output83'])
                teacher=max(legal,key=lambda i:(row['policy_target81'][i],-i))
                games[row['game_index']].append(dict(
                    mean_absolute_value_drift=math.fsum(abs(a-b) for a,b in zip(row['output83'][81:],parent['output83'][81:]))/2.,
                    raw_policy_choice_changed=float(chosen!=parent_chosen),
                    raw_policy_teacher_agreement=float(chosen==teacher),
                    parent_policy_teacher_agreement=float(parent_chosen==teacher)))
        metrics=('mean_absolute_value_drift','raw_policy_choice_changed','raw_policy_teacher_agreement','parent_policy_teacher_agreement')
        per_game=[dict(game_index=game,roots=len(rows),**{key:math.fsum(x[key] for x in rows)/len(rows) for key in metrics})
            for game,rows in sorted(games.items())]
        results[model]=dict(games=per_game,mean_game={key:math.fsum(x[key] for x in per_game)/len(per_game) for key in metrics})
    return dict(models=results,descriptive_only=True,ties='lowest_legal_channel_index',
        raw_policy_is_not_searched_action=True,strength_claim=False)


def search_value_errors(bundles,labels):
    labelmap={identity(x):x for x in labels};results={}
    require(len(labelmap)==len(labels),'Unique authenticated search-label inventory')
    for model in MODEL_IDS:
        by_game=defaultdict(list)
        for split in SPLITS:
            for row in bundles[model][split]:
                label=labelmap[identity(row)]
                require(label['row_sha256']==row['row_sha256'] and label['root_input_sha256']==row['root_input_sha256'],
                    'Search-value diagnostics bind exact prediction input')
                q=label['root_q_signed']
                by_game[row['game_index']].append(math.fsum((x-y)**2 for x,y in zip(row['output83'][81:],[q,-q]))/2.)
        per_game=[dict(game_index=g,roots=len(values),search_value_mse=math.fsum(values)/len(values))
            for g,values in sorted(by_game.items())]
        results[model]=dict(games=per_game,mean_game_search_value_mse=math.fsum(x['search_value_mse'] for x in per_game)/len(per_game))
    return dict(models=results,descriptive_only=True,target='saved_teacher512_signed_actor_root_Q',
        objective_related_metric_not_independent_outcome_validation=True,strength_claim=False)


def analyze_predictions(bundles, *, draws=DRAW_COUNT, seed=sampling.BOOTSTRAP_SEED,
                        expected_games=None, strict=True, boundary=lambda: None):
    """Model-free statistics; small explicit counts are supported only for tests."""
    import numpy as np
    require(type(draws) is int and draws>0 and type(seed) is int, "strict bootstrap settings")
    if strict: require(draws==DRAW_COUNT and seed==sampling.BOOTSTRAP_SEED, "fixed bootstrap plan")
    require(set(bundles)==set(MODEL_IDS), "all five fixed models, no selective omission")
    if expected_games is None:
        expected_games={split:sorted({x["game_index"] for x in bundles["parent"][split]}) for split in SPLITS}
    require(set(expected_games)==set(SPLITS) and not set(expected_games[SPLITS[0]]) & set(expected_games[SPLITS[1]]), "disjoint held-out partitions")
    if strict: require(all(len(expected_games[s])==64 for s in SPLITS), "exact64+64 held-out games")
    summaries={}; losses={}
    for model_id in MODEL_IDS:
        require(set(bundles[model_id])=={"train",*SPLITS}, "all fixed evaluation splits")
        summaries[model_id]={};losses[model_id]={}
        for split in SPLITS:
            rows=bundles[model_id][split];parent=bundles["parent"][split]
            ids=[identity(x) for x in rows]
            require(ids==sorted(set(ids)) and all(x[2]==split for x in ids), "unique complete prediction order")
            require([target_identity(x) for x in rows]==[target_identity(x) for x in parent], "paired same-input target evidence")
            summary=_mean_game_rows(rows)
            require([x["game_index"] for x in summary]==expected_games[split], "every held-out game contributes")
            summaries[model_id][split]=summary
            losses[model_id][split]={m:np.array([x[m] for x in summary],dtype=np.float64) for m in ("policy","value")}
    rng=np.random.Generator(np.random.PCG64(seed));indices={};bootstrap={}
    for split in SPLITS:
        n=len(expected_games[split]);require(n>1,"nondegenerate partition sample")
        indices[split]=rng.integers(0,n,size=(draws,n),dtype=np.int64)
        bootstrap[split]=dict(game_ids=expected_games[split],draws=draws,index_sha256=hashlib.sha256(indices[split].astype('<i8',copy=False).tobytes()).hexdigest())
    contrasts=[]
    definitions=[("mixed_minus_terminal",{f"r{r}_mixed_joint":.5 for r in (0,1)}|{f"r{r}_terminal_joint":-.5 for r in (0,1)})]
    definitions += [(model+"_minus_parent",{model:1.,"parent":-1.}) for model in MODEL_IDS if model!="parent"]
    definitions += [("mixed_mean_minus_parent",{f"r{r}_mixed_joint":.5 for r in (0,1)}|{"parent":-1.}), ("terminal_mean_minus_parent",{f"r{r}_terminal_joint":.5 for r in (0,1)}|{"parent":-1.})]
    for name,coeff in definitions:
        for metric in ("policy","value"):
            boundary();delta={s:sum(weight*losses[model][s][metric] for model,weight in coeff.items()) for s in SPLITS}
            samples={s:delta[s][indices[s]].mean(axis=1) for s in SPLITS}
            for split in (*SPLITS,"pooled"):
                if split=="pooled":
                    n=sum(len(delta[s]) for s in SPLITS)
                    point=sum(float(delta[s].sum()) for s in SPLITS)/n
                    values=sum(samples[s]*len(delta[s]) for s in SPLITS)/n
                else: point=float(delta[split].mean());values=samples[split]
                lo,hi=np.quantile(values,[.025,.975],method="linear")
                contrasts.append(dict(contrast=name,metric=metric,split=split,difference=point,ci95=[float(lo),float(hi)],
                    primary=name=="mixed_minus_terminal" and metric=="value" and split=="pooled",
                    lower_is_better=True,degenerate_interval=bool(lo==hi)))
    # Training loss is descriptive and uses each fit's exact selected roots.
    parent_train={identity(x):x for x in bundles["parent"]["train"]}
    require(len(parent_train)==len(bundles["parent"]["train"]), "unique parent training diagnostic roots")
    training={}
    for spec in sampling.fits():
        model=spec["fit_id"]; rows=bundles[model]["train"]
        require(len(rows)==len({identity(x) for x in rows}), "unique candidate training diagnostic roots")
        parent=[parent_train[identity(x)] for x in rows]
        require(all(target_identity(a)==target_identity(b) for a,b in zip(rows,parent,strict=True)), "paired selected training targets")
        if strict:
            count=defaultdict(int)
            for row in rows: require(row["split"]=="train","training diagnostic split");count[row["game_index"]]+=1
            require(len(rows)==3072, "fixed common prospective training diagnostic panel")
        training[model]=dict(candidate=_mean_game_rows(rows),parent=_mean_game_rows(parent))
    primary=next(x for x in contrasts if x["primary"])
    return dict(schema="full_corpus_offline_analysis_v1",status="fixed_prediction_contrasts_analyzed",
        contrasts=contrasts,primary=primary,game_summaries=summaries,training_diagnostics=training,
        bootstrap=dict(seed=seed,draws=draws,generator="numpy.PCG64",unit="whole_game",
                       strata=bootstrap,pooled="partition-size-weighted_same_draws",same_draws_all_models=True),
        independent_heldout_games=sum(len(expected_games[s]) for s in SPLITS),fixed_fits=4,
        checkpoint_selection=False,offline_strength_admission_gate=False,strength_improvement_established=False,
        limitations=["Pooled mixed-minus-terminal outcome MSE is the diagnostic primary; other intervals are descriptive nominal95%.",
                     "Both repetitions share the same source games; they do not increase independent game count.",
                     "Both target arms train on all57022 examples in1782 updates; both heads and trunk are jointly updated.",
                     "Intervals condition on these fixed fits; they do not estimate arbitrary training-seed variability.",
                     "Held-out positions were previously examined and reused diagnostically. The fresh live mixed-versus-terminal matchup supplies the primary strength contrast; the two EF8A matchups are secondary parent comparisons."])


def load_predictions(selection, commitment, evaluation, *, boundary=lambda: None):
    """Saved-only row/target/inventory verifier; no model construction."""
    committed=fit.validate_commitment(selection,commitment,boundary=boundary)
    expected_models=prediction_bindings(committed,boundary=boundary)
    result=fit.read_json(evaluation,boundary=boundary)
    require(result["schema"]=="full_corpus_offline_evaluation_v1" and result["status"]=="all_fixed_predictions_complete" and
            result["selection_sha256"]==c.digest(selection) and result["commitment"]==commitment and
            result["optimizer_calls"]==0 and result["all_fits_committed_before_heldout"] is True and
            [x["model_id"] for x in result["models"]]==list(MODEL_IDS), "complete fixed evaluation commitment")
    heldout=sampling.load_heldout(selection,commitment,boundary=boundary);train_union={}
    search_document=fit.read_json(result['search_value_targets'],boundary=boundary)
    require(search_document['schema']=='full_corpus_saved_search_labels_v1' and
        search_document['selection_sha256']==c.digest(selection) and search_document['teacher_simulations']==512 and
        search_document['reused_diagnostic_only'] is True and search_document['not_official_outcome'] is True,
        'Explicit previously examined search-target diagnostic')
    expected_search=search_targets(selection,heldout,boundary=boundary)
    require(c.canonical(search_document['rows'])==c.canonical(expected_search),'Independent raw saved-search label rederivation')
    train_union={(x['game_index'],x['ply']):x for x in sampling.load_diagnostic_train(selection,boundary=boundary)}
    bundles={};allpaths=[];base=c.resolve(evaluation).parent
    for model in result["models"]:
        model_id=model["model_id"];expected=_rows_for_model(selection,heldout,train_union,model_id)
        require(expected_models[model_id]=={key:model[key] for key in ("checkpoint","weights_sha256")},"saved predictor binds accepted fixed model")
        bundles[model_id]={split:[] for split in ("train",*SPLITS)};paths=[]
        for item in model["batches"]:
            boundary();batch=fit.read_json(item,boundary=boundary);split=batch["split"]
            require(split in expected and batch["model_id"]==model_id and batch["commitment"]==commitment and
                    batch["selection_sha256"]==c.digest(selection) and batch["weights_sha256"]==model["weights_sha256"] and
                    batch["checkpoint"]==model["checkpoint"], "batch exact model/source binding")
            index=len(bundles[model_id][split])//32
            require(type(batch["batch_index"]) is int and batch["batch_index"]==index and
                    c.resolve(item)==base/"predictions"/model_id/split/f"{index:06d}.json", "complete canonical batch order")
            require(0<len(batch["rows"])<=32 and len(batch["rows"])==min(32,len(expected[split])-index*32), "complete fixed evaluation batch")
            bundles[model_id][split].extend(batch["rows"]);paths.append(c.resolve(item));allpaths.append(c.resolve(item))
        require(len(paths)==len(set(paths))==model["forward_calls"], "exact forward and batch inventory")
        for split,examples in expected.items():
            rows=bundles[model_id][split];require(len(rows)==len(examples)==model["root_counts"][split],"all selected roots predicted")
            for example,row in zip(examples,rows,strict=True):
                strict_prediction(row)
                require(target_identity(row)==target_identity(fit.prediction_row(example,row["output83"])), "actual corrected example and saved target identity")
    require(set(base.glob("predictions/*/*/*.json"))==set(allpaths) and result["forward_calls"]==len(allpaths), "no missing or extra evaluation batches")
    return bundles,result


def verify_arithmetic(bundles, reported, *, boundary=lambda: None):
    """Independent saved-row loss/aggregation/percentile implementation; no models."""
    import numpy as np
    counts=reported["bootstrap"];draws=counts["draws"];seed=counts["seed"]
    require(type(draws) is int and draws>0 and type(seed) is int,"strict independent bootstrap counts")
    matrices={};game_ids={}
    for split in SPLITS:
        game_ids[split]=sorted({r["game_index"] for r in bundles["parent"][split]})
        rows=[]
        for model_id in MODEL_IDS:
            by_game=defaultdict(list)
            for row in bundles[model_id][split]:
                # Independent float64 array calculation on authenticated saved outputs.
                out=np.asarray(row["output83"],dtype=np.float64)
                mask=np.asarray(row["legal_mask81"],dtype=np.bool_);target=np.asarray(row["policy_target81"],dtype=np.float64)
                maximum=float(out[:81][mask].max())
                normalizer=maximum+float(np.log(np.exp(out[:81][mask]-maximum).sum(dtype=np.float64)))
                policy=float(np.sum(target*(normalizer-out[:81]),dtype=np.float64))
                value=float(np.mean(np.square(out[81:]-np.asarray(row["value_target2"],dtype=np.float64))))
                by_game[row["game_index"]].append([max(0.,policy),value])
            require(sorted(by_game)==game_ids[split],"independent every-game inventory")
            means=np.asarray([np.asarray(by_game[g],dtype=np.float64).mean(axis=0) for g in game_ids[split]])
            saved=reported["game_summaries"][model_id][split]
            require([r["game_index"] for r in saved]==game_ids[split] and
                    np.allclose(means,np.asarray([[r["policy"],r["value"]] for r in saved]),rtol=1e-12,atol=1e-12),
                    "independent position-to-game loss arithmetic")
            rows.append(means)
        matrices[split]=np.stack(rows,axis=0)
    rng=np.random.Generator(np.random.PCG64(seed));sampled={};index_hashes={}
    for split in SPLITS:
        n=len(game_ids[split]);indices=np.empty((draws,n),dtype=np.int64)
        # Draw-by-draw generation independently checks the saved batched stream.
        for draw in range(draws):
            if draw%500==0:boundary()
            indices[draw]=rng.integers(0,n,size=n,dtype=np.int64)
        index_hashes[split]=hashlib.sha256(indices.astype('<i8',copy=False).tobytes()).hexdigest()
        require(index_hashes[split]==counts["strata"][split]["index_sha256"],"independent stratified draw stream")
        sampled[split]=matrices[split][:,indices,:].mean(axis=2)
    definitions={"mixed_minus_terminal":np.asarray([0.,.5,.5,-.5,-.5])}
    for index,model_id in enumerate(MODEL_IDS[1:],start=1):
        weights=np.zeros(5);weights[0]=-1.;weights[index]=1.;definitions[model_id+"_minus_parent"]=weights
    definitions["terminal_mean_minus_parent"]=np.asarray([-1.,0.,0.,.5,.5])
    definitions["mixed_mean_minus_parent"]=np.asarray([-1.,.5,.5,0.,0.])
    expected={(name,metric,split) for name in definitions for metric in ("policy","value") for split in (*SPLITS,"pooled")}
    observed={(r["contrast"],r["metric"],r["split"]) for r in reported["contrasts"]}
    require(len(reported["contrasts"])==len(expected) and observed==expected,"complete prospective contrast inventory")
    for row in reported["contrasts"]:
        boundary();weights=definitions[row["contrast"]];column=0 if row["metric"]=="policy" else 1
        splits=SPLITS if row["split"]=="pooled" else (row["split"],)
        total=sum(len(game_ids[s]) for s in splits)
        point=sum(float(np.dot(weights,matrices[s][:,:,column].sum(axis=1))) for s in splits)/total
        samples=sum(np.dot(weights,sampled[s][:,:,column])*len(game_ids[s]) for s in splits)/total
        ordered=np.sort(samples)
        def quantile(p):
            position=(draws-1)*p;left=int(position);fraction=position-left
            return float(ordered[left]*(1.-fraction)+ordered[min(left+1,draws-1)]*fraction)
        interval=[quantile(.025),quantile(.975)]
        require(math.isclose(row["difference"],point,rel_tol=1e-12,abs_tol=1e-12) and
                all(math.isclose(a,b,rel_tol=1e-12,abs_tol=1e-12) for a,b in zip(row["ci95"],interval,strict=True)),
                "independent fixed contrast and two-sided percentile bounds")
        require(row["primary"] is (row["contrast"]=="mixed_minus_terminal" and row["metric"]=="value" and row["split"]=="pooled"),
                "sole prospective primary contrast")
    require(reported["primary"]==next(x for x in reported["contrasts"] if x["primary"]),"reported primary matches verified contrast")
    return dict(status="independent_saved_arithmetic_verified",contrasts=len(expected),draws=draws,seed=seed,
                bootstrap_index_sha256=index_hashes,independent_loss_formula=True,independent_game_aggregation=True,
                independent_draw_by_draw_stream=True,independent_sorted_linear_percentiles=True,
                model_calls=0,optimizer_calls=0)


def run_analysis(selection, commitment, evaluation, output, *, boundary=lambda: None):
    require(not any(x=="torch" or x.startswith("torch.") for x in sys.modules), "fresh no-Torch analysis worker")
    bundles,source=load_predictions(selection,commitment,evaluation,boundary=boundary)
    result=analyze_predictions(bundles,boundary=boundary)
    result['prediction_changes']=prediction_changes(bundles)
    search_document=fit.read_json(source['search_value_targets'],boundary=boundary)
    result['search_value_errors']=search_value_errors(bundles,search_document['rows'])
    arithmetic=verify_arithmetic(bundles,result,boundary=boundary)
    arithmetic_item=c.save(Path(output)/"arithmetic_verification.json",arithmetic)
    result.update(selection_sha256=c.digest(selection),commitment=commitment,evaluation=evaluation,
                  verified_forward_calls=source["forward_calls"],analysis_model_calls=0,analysis_optimizer_calls=0,
                  independent_arithmetic=arithmetic_item)
    item=c.save(Path(output)/"analysis.json",result)
    chart=render_chart(result,output,boundary=boundary)
    require(not any(x=="torch" or x.startswith("torch.") for x in sys.modules), "analysis remained model-free")
    return dict(status="offline_saved_verified",analysis=item,arithmetic=arithmetic_item,chart=chart,checkpoint_selection=False,models=0,optimizer_calls=0)


def render_chart(result, output, *, boundary=lambda: None):
    require(not any(x=="torch" or x.startswith("torch.") for x in sys.modules), "fresh no-Torch renderer")
    import matplotlib
    matplotlib.use("Agg")
    from matplotlib import pyplot as plt
    boundary();path=Path(output)/"prediction_transfer.png"
    require(path.resolve().is_relative_to(c.OUT) and not path.exists(), "exclusive new study chart")
    rows=[x for x in result["contrasts"] if x["contrast"]=="mixed_minus_terminal"]
    fig,axes=plt.subplots(1,2,figsize=(11,4.8),sharey=True)
    labels={"validation":"Validation: 64 games","offline":"Assessment: 64 games","pooled":"Pooled: 128 games"}
    for ax,metric,title in zip(axes,("policy","value"),("Policy: predicting search choices","Value: predicting outcomes")):
        for y,split in enumerate((*SPLITS,"pooled")):
            row=next(x for x in rows if x["split"]==split and x["metric"]==metric)
            lo,hi=row["ci95"];color="#0f766e" if row["primary"] else "#64748b"
            ax.plot([lo,hi],[y,y],color=color,lw=2.4)
            ax.scatter([row["difference"]],[y],color=color,s=65 if row["primary"] else 35,zorder=3)
        ax.axvline(0,color="#334155",ls="--",lw=1);ax.set_title(title)
        ax.set_xlabel("Mixed minus terminal-target loss\nNegative favors mixed value targets")
        ax.grid(axis="x",alpha=.2);ax.spines[["top","right"]].set_visible(False)
        ax.ticklabel_format(axis="x",style="sci",scilimits=(-3,3))
    axes[0].set_yticks(range(3),[labels[s] for s in (*SPLITS,"pooled")]);axes[0].invert_yaxis()
    fig.suptitle("Do mixed outcome/search value targets help learning?",fontsize=15)
    fig.text(.5,.035,"Same 57,022 roots from 1,024 games / 1,782 updates / two fixed repetitions per recipe\n"
        "Bars: paired whole-game 95% intervals. Pooled outcome MSE is diagnostic primary; other intervals are descriptive.",ha="center",fontsize=9)
    fig.tight_layout(rect=(0,.14,1,.92));fig.savefig(path,dpi=160);plt.close(fig);boundary()
    return c.artifact(path)
