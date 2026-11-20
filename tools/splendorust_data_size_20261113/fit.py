"""Separate-core fixed-update nested-game training; original numeric recipe.

No model runs on import. Every worker supplies its single corrected-source guard
receipt. No legacy engine or CPU0 placement function is called.
"""
from __future__ import annotations
from copy import deepcopy
from pathlib import Path
import hashlib
import math
import os
import random
import time
from . import common as c
from . import selection as select
from tools import future_training_rules_20261021 as guard
from tools.splendorust_full_corpus_20261110 import fit as old
from tools.splendorust_full_corpus_20261110 import sampling, corpus, targets
from tools.splendorust_policy_head_20261104 import data as accepted_data
from tools.splendorust_independent_data_20261107.collection import common as cc, data, storage
from tools.splendorust_independent_data_20261107 import schedule

accepted=old.accepted
IDENTITY=sampling.IDENTITY
STEPS,BATCH=1782,32
UPDATE_KEYS={'batch_examples','batch_denominator','loss','policy_mean_unweighted','value_mean_unweighted',
    'gradient_norm_before_clip','clipped','finite_gradients','forward_backward_calls','optimizer_calls',
    'before_group_sha256','after_group_sha256','outputs83','scope','seconds','gradient_sha256','gradient_evidence'}
_STARTUP=_ENGINE=None
require,canonical,digest=c.require,c.canonical,c.digest
tensor_tree,rng_state,restore_rng=old.tensor_tree,old.rng_state,old.restore_rng
selected_batch,scoped_parameters=old.selected_batch,old.scoped_parameters
make_optimizer,verify_optimizer_payload=old.make_optimizer,old.verify_optimizer_payload
loss_components=old.loss_components
prediction_row=old.prediction_row
_validate_features,_target,_parse_target=corpus._validate_features,corpus._target,corpus._parse_target
feature_arrays,index_row,attach_game_targets=corpus.feature_arrays,corpus.index_row,corpus.attach_game_targets

def startup(preverified=None):
    global _STARTUP
    if _STARTUP is None:
        receipt = deepcopy(preverified) if preverified is not None else guard.verify_runtime_sources()
        require(type(receipt) is dict and receipt == guard.SOURCE_HASHES, 'Actual corrected runtime receipt')
        _STARTUP = receipt
    elif preverified is not None:
        require(_STARTUP == preverified, 'One unchanged startup receipt')
    return deepcopy(_STARTUP)

def engine(folder, *, runtime_sources=None):
    global _ENGINE
    sources = startup(runtime_sources)
    if _ENGINE is None:
        import psutil
        process = psutil.Process()
        process.cpu_affinity([4]); process.nice(psutil.NORMAL_PRIORITY_CLASS)
        require(process.cpu_affinity() == [4], 'Separate physical CPU4 placement')
        import torch
        torch.set_num_threads(1); torch.set_num_interop_threads(1)
        _ENGINE = dict(pid=process.pid, created=process.create_time(), cpu_affinity=[4],
                       priority='NORMAL_PRIORITY_CLASS', torch_threads=1,
                       torch_interop_threads=1, device='cpu', precision='float32')
    c.save(Path(folder)/'startup.json', dict(runtime_sources=sources, engine=_ENGINE, **IDENTITY))
    return deepcopy(_ENGINE)

def seed_rng(repetition):
    import numpy as np
    import torch
    require(type(repetition) is int and repetition in (0, 1), "strict fixed RNG repetition")
    seed = sampling.SEEDS[repetition]
    random.seed(seed); np.random.seed(seed); torch.manual_seed(seed)
    return seed


def verify_startup(item,*,boundary=lambda:None):
    c.verify([item],boundary=boundary);receipt=c.read(c.resolve(item));info=receipt['engine']
    accepted_data.verify_identity(receipt)
    require(receipt['runtime_sources']==guard.SOURCE_HASHES and
        info['cpu_affinity']==[4] and info['torch_threads']==info['torch_interop_threads']==1 and
        info['device']=='cpu' and info['precision']=='float32' and info['priority']=='NORMAL_PRIORITY_CLASS',
        'Actual corrected-source startup on separate single-thread CPU4')
    return receipt

def step(model, optimizer, batch, scope, *, boundary=lambda: None):
    """Accepted forward/loss and AdamW recipe; add complete saved update evidence."""
    import torch
    boundary()
    started = time.perf_counter()
    require(1 <= len(batch["tokens"]) <= BATCH, "Genuine full or partial batch of1..32")
    model.train()
    params = scoped_parameters(model, scope)
    before = accepted.group_digests(model)
    optimizer.zero_grad(set_to_none=True)
    outputs = accepted.tensor_forward(model, batch["tokens"])
    total, policy, value = loss_components(outputs, batch)
    total.backward()
    boundary()
    require(all(p.grad is not None and bool(torch.isfinite(p.grad).all()) for p in params),
            "finite gradients for every trainable tensor")
    scoped_parameters(model, scope)
    norm = torch.nn.utils.clip_grad_norm_(params, 1., error_if_nonfinite=True, foreach=False)
    optimizer.step()
    boundary()
    require(all(bool(torch.isfinite(p).all()) for p in model.parameters()), "finite optimizer state")
    after = accepted.group_digests(model)
    require(before != after, "actual parameter update")
    if scope == "policy":
        require(all(before[k] == after[k] for k in ("shared", "value")) and
                before["policy"] != after["policy"], "policy-only exact frozen groups")
    scoped_parameters(model, scope)
    return {"batch_examples": len(batch["tokens"]), "batch_denominator": len(batch["tokens"]),
            "loss": float(total.detach()), "policy_mean_unweighted": float(policy.detach().mean()),
            "value_mean_unweighted": float(value.detach().mean()),
            "gradient_norm_before_clip": float(norm), "clipped": float(norm) > 1.,
            "finite_gradients": True, "forward_backward_calls": 1, "optimizer_calls": 1,
            "before_group_sha256": before, "after_group_sha256": after,
            "outputs83": outputs.detach().tolist(), "scope": scope,
            "seconds": time.perf_counter() - started}

def recorded_step(model, optimizer, batch, scope, *, boundary=lambda: None):
    """Preserve exact accepted update math; additionally retain every clipped gradient hash."""
    result = step(model, optimizer, batch, scope, boundary=boundary)
    result["gradient_sha256"] = {name: hashlib.sha256(p.grad.detach().contiguous().numpy().tobytes()).hexdigest()
        for name,p in model.named_parameters() if p.requires_grad}
    result["gradient_evidence"] = "all_trainable_clipped_float32_gradient_bytes"
    return result

def live_trees(model,optimizer):
    return {k:digest(tensor_tree(v)) for k,v in dict(state_dict=model.state_dict(),
        optimizer=optimizer.state_dict(),rng=rng_state()).items()}

def compare_update(actual,expected):
    require(UPDATE_KEYS<=set(actual) and UPDATE_KEYS<=set(expected) and
            all(actual[k]==expected[k] for k in actual if k!='seconds'),
            'Exact update outputs, loss, gradients and model state')


def saved_batch_statistics(examples,outputs):
    require(len(examples)==len(outputs) and 1 <= len(examples) <= BATCH,'Genuine full or partial saved batch of1..32')
    losses=[targets.prediction_losses(prediction_row(x,y)) for x,y in zip(examples,outputs,strict=True)]
    n=len(examples)
    return {'loss':math.fsum(x['combined'] for x in losses)/n,
            'policy_mean_unweighted':math.fsum(x['policy'] for x in losses)/n,
            'value_mean_unweighted':math.fsum(x['value'] for x in losses)/n}

def fixture_cache_roundtrip(examples,folder):
    """Exercise the production float32 writer and mmap reader on rule fixtures."""
    import numpy as np
    folder=Path(folder);c.require(folder.resolve().is_relative_to(c.OUT),'New qualification cache only')
    folder.mkdir(parents=True,exist_ok=False)
    arrays=feature_arrays(examples);loaded={};proof=[]
    for key,array in arrays.items():
        path=folder/(key+'.npy')
        with path.open('xb') as stream:np.save(stream,array,allow_pickle=False)
        descriptor=c.artifact(path);c.verify([descriptor]);proof.append(descriptor)
        loaded[key]=np.load(path,allow_pickle=False,mmap_mode='r')
        c.require(loaded[key].dtype==array.dtype and loaded[key].shape==array.shape and
            loaded[key].tobytes()==array.tobytes(),'Actual cached fixture feature bits '+key)
    return loaded,proof

def qualify_training_inputs(folder,*,boundary=lambda:None):
    from tools.splendorust_independent_data_20261107.collection import qualification
    folder=Path(folder);gate=cc.read(cc.OUT/'qualification_gate.json')
    c.require(gate['passed'] is True and gate['collection_plan']==c.artifact(cc.PLAN),'Actual new collector qualification before trainer')
    c.verify([gate['audit']],boundary=boundary);proof=cc.read(c.resolve(gate['audit']))
    c.require(proof['status']=='passed' and proof['complete_decisions']==116 and proof['engineering_games']==2,'All corrected fixture and natural-game checks')
    child=cc.read(cc.PLAN);fixture_plan=child['qualification']['loader'];fixture_cases=fixture_plan['cases']+[fixture_plan['private_variant']]
    fixture_rows=[];eligible=[];source_rows=[]
    for index,case in enumerate(fixture_cases):
        boundary();path=cc.OUT/'qualification/worker/cpu/loader'/f'{index:02d}'/'moves/000000.json'
        item=c.artifact(path);row=data.read_artifact(item);spec=qualification.fixture_spec(case,index)
        decoded=data.validate_row(row,spec,case['snapshot'],0)
        fixture_rows.append(row);source_rows.append(item)
        if decoded['eligible']:eligible.append(decoded)
    structural=data.qualify_captured_rows(fixture_plan,fixture_rows,boundary=boundary)
    c.require(structural==proof['loader_row_checks'],'Same16 actual corrected source fixtures in trainer worker')
    arrays=_validate_features(eligible);cache_artifacts=[]
    for start,expected_arrays in zip(range(0,len(eligible),32),arrays,strict=True):
        part=eligible[start:start+32];batch=sampling.collate_inputs(part,[1.]*len(part))
        c.require(all(batch[key].tolist()==expected_arrays[key].tolist() for key in expected_arrays),
            'Actual selected-training input/policy collator preserves all structural arrays')
        cached,cache_proof=fixture_cache_roundtrip(part,folder/'structural_cache'/str(start))
        cache_artifacts.extend(cache_proof)
        c.require(all(cached[key].tobytes()==expected_arrays[key].tobytes() for key in cached),
            'All corrected structural counterexamples survive actual cache path')
    examples=[];specs=schedule.qualification_schedule()
    for spec,item in zip(specs,proof['engineering_targets'],strict=True):
        boundary();examples.extend(_target(item,spec)['examples'])
    selected=[x for spec in specs for x in sorted((x for x in examples if x['game_index']==spec['game_index']),key=lambda x:x['ply'])[:16]]
    c.require(len(selected)==32 and all(x['split']=='engineering' for x in selected)
        and {x['value'][0] for x in selected}=={-1.,0.,1.},'Exactly16 genuine roots from each excluded game cover win/loss/draw')
    # Exercise the same corrected collator used to verify collected targets.
    arrays=data.collate_arrays(selected,[1.]*32)
    c.require(arrays['tokens'].reshape(32,-1).tolist()==[x['tokens'] for x in selected],'Actual corrected target tensor conversion')
    terminal_selected=deepcopy(selected)
    mixed=[]
    for spec,gameproof in zip(specs,proof['engineering_game_proofs'],strict=True):
        rows=[x for x in selected if x['game_index']==spec['game_index']]
        if rows:mixed.extend(attach_game_targets(dict(spec=spec,source_game=gameproof['document']),rows,boundary=boundary))
    selected=sorted(mixed,key=lambda x:(x['game_index'],x['ply']))
    cached,cache_proof=fixture_cache_roundtrip(selected,folder/'genuine_cache')
    cache_artifacts.extend(cache_proof)
    index=[]
    for example in selected:
        label=dict(root_q_win_credit=example['root_q_win_credit'],root_q_signed=example['root_q_signed'],source_move=example['root_q_source'])
        index.append(index_row({**example,'value':example['terminal_value']},label))
    from tools.splendorust_full_corpus_20261110.sampling import CachedExamples
    cached_examples=CachedExamples(dict(index=index,arrays=cached),list(range(32)),'mixed')
    cached_batch=sampling.collate([cached_examples[i] for i in range(32)],[1.]*32)
    genuine=sampling.collate(selected,[1.]*32)
    c.require(genuine['value'].tolist()==[[targets.f32(v) for v in x['value']] for x in selected] and
        all(genuine[key].tolist()==arrays[key].tolist() for key in arrays),'Actual selected-training collator preserves fixed genuine outcome/search mixtures')
    c.require(all(genuine[key].tolist()==cached_batch[key].tolist() for key in genuine),
        'Actual mmap-backed training batches match genuine corrected decoded targets exactly')
    original=data.read_artifact(proof['engineering_targets'][0]);spec=specs[0];negative=[]
    def reject(name,operation):
        try:operation()
        except (ValueError,KeyError,TypeError,AssertionError):negative.append(dict(case=name,rejected=True))
        else:raise ValueError('Actual production target loader accepted '+name)
    for key in data.IDENTITY:
        bad=deepcopy(original);bad.pop(key)
        reject('missing_'+key,lambda bad=bad:_parse_target(bad,spec))
        bad=deepcopy(original);bad[key]='historical_or_wrong'
        reject('wrong_'+key,lambda bad=bad:_parse_target(bad,spec))
    bad=deepcopy(original);bad['spec']['seed_decimal']='0'
    reject('substituted_seed',lambda:_parse_target(bad,spec))
    bad=deepcopy(original);bad['examples'][0]['mask'][0]=1
    reject('nonbool_mask',lambda:_parse_target(bad,spec))
    bad=deepcopy(original);bad['examples'][0]['value']=[1,1]
    reject('invalid_official_value',lambda:_parse_target(bad,spec))
    bad=deepcopy(proof['engineering_targets'][0]);bad['sha256']='0'*64
    reject('physical_artifact_hash',lambda:data.read_artifact(bad))
    for key in ('value_target_contract','terminal_value','root_q_win_credit','root_q_signed'):
        bad=deepcopy(selected[0]);bad.pop(key)
        reject('mixed_missing_'+key,lambda bad=bad:sampling.collate([bad],[1.]))
    bad=deepcopy(selected[0]);bad['value']=list(reversed(bad['value']))
    reject('mixed_reversed_actor_targets',lambda:sampling.collate([bad],[1.]))
    bad=deepcopy(selected[0]);bad['root_q_win_credit']=1.01
    reject('mixed_invalid_win_credit',lambda:sampling.collate([bad],[1.]))
    example=terminal_selected[0];item=selected[0]['root_q_source'];raw=data.read_artifact(item)
    bad=deepcopy(raw);bad['training_capture']['actor_seat']=1-example['actor']
    reject('mixed_wrong_canonical_actor',lambda:targets.attach(example,bad,item,specs[0]))
    bad=deepcopy(raw);bad['training_capture']['native_capture']['root_value']=0. if raw['training_capture']['native_capture']['root_value'] else 1.
    reject('mixed_changed_saved_root_q',lambda:targets.attach(example,bad,item,specs[0]))
    bad=deepcopy(item);bad['sha256']='0'*64
    reject('mixed_changed_source_hash',lambda:data.read_artifact(bad))
    result=dict(status='actual_production_loader_qualified',collector_audit=gate['audit'],targets=proof['engineering_targets'],
        structural_loader_checks=structural,structural_source_rows=source_rows,full_loader_checks=proof['full_loader_checks'],
        actual_selected_loader_and_collator_exercised=True,structural_terminal_labels_fabricated=False,
        actual_target_parser_rejections=negative,cache_feature_artifacts=cache_artifacts,
        actual_mmap_training_path_qualified=True,
        engineering_examples_available=len(examples),qualification_examples=32,source_plan=c.artifact(cc.PLAN),
        zero_scientific_source_games=True,synthetic_value_targets=False,value_target_contract=targets.contract(),value_target_sources_sha256=targets.source_digest(selected),storage=storage.contract(),**data.IDENTITY)
    c.save(folder/'loader_qualification.json',result)
    return dict(proof=result,qualification_examples=selected,qualification_weights=[1.]*32,qualification_structural_examples=eligible)


def source_checkpoint(repetition, step=1782):
    require(type(repetition) is int and repetition in (0,1),'Fixed reference seed')
    require(step in (0,1781,1782),'Accepted reference checkpoint')
    result=c.read(c.ORIGINAL_OUT/'fits'/f'r{repetition}_mixed_joint'/'worker/fit_result.json')
    require(result['status']=='fit_complete' and result['repetition']==repetition and
        result['breadth']=='mixed' and result['optimizer_calls']==1782,'Accepted unchanged reference fit')
    item=result['checkpoints'][str(step)];c.verify([item]);return item


def load_training(game_count,repetition,*,boundary=lambda:None):
    selection=c.read(c.SELECTION)
    loaded=select.load_selected(selection,repetition,game_count,boundary=boundary)
    schedule=select.build_schedules(selection,repetition,game_count)
    require(loaded['batches']==schedule['batches'] and len(loaded['batches'])==STEPS,'Exact fixed update schedule')
    require(loaded['train_weights']==[1.]*len(loaded['train_examples']) and
        len(loaded['train_game_ids'])==game_count and loaded['cache_validated'] is True,'Actual equal-weight selected cache')
    require(loaded['corpus_cache']==c.artifact(c.CACHE_ACCEPTANCE),'Accepted source cache')
    examples=loaded['train_examples']
    require(all(examples.cache['index'][i]['split']=='train' for i in examples.indices),'Only scientific training games')
    return selection,loaded,schedule


def metadata(game_count,repetition,loaded,schedule,*,qualification=False):
    return dict(study_id=c.STUDY,fit_id=f'g{game_count}_r{repetition}_mixed',
        game_count=game_count,repetition=repetition,seed=sampling.SEEDS[repetition],breadth='mixed',scope='joint',
        qualification_only=qualification,historical_data_relabelled=False,
        historical_warm_start_sha256=accepted.PARENT_SHA256,parameter_count=accepted.PARAMETERS,
        model_source_sha256=accepted.MODEL_SOURCE_SHA256,source_training_games=game_count,
        training_records=len(loaded['train_examples']),one_pass_step=schedule['one_pass_step'],
        fixed_final_step=STEPS,batch_size=BATCH,last_batch_size=len(schedule['batches'][schedule['one_pass_step']-1]),
        retained_checkpoint_steps=[0,*schedule['checkpoint_steps']],
        optimizer_recipe={k:v for k,v in accepted.RECIPE.items() if k not in ('epochs','shuffle','example_weight','batch_reduction')},
        batch_reduction='mean_actual_batch_size_equal_record_weights',value_target_contract=targets.contract('mixed'),
        value_target_sources_sha256=loaded['value_target_sources_sha256'],corpus_cache=loaded['corpus_cache'],
        schedule_sha256=digest(schedule),selection_sha256=digest(c.read(c.SELECTION)),
        checkpoint_selection=False,best_seed_selection=False,model_cpu=4,original_reference_cpu=0,
        cpu0_training_executed=False,training_initializer='original_EF8A_and_fresh_AdamW_same_seed',**IDENTITY)


def validate_checkpoint_metadata(meta):
    require(type(meta) is dict and meta.get('study_id')==c.STUDY and type(meta.get('qualification_only')) is bool,
        'Strict separate study checkpoint')
    accepted_data.verify_identity(meta)
    size,rep,index=meta.get('game_count'),meta.get('repetition'),meta.get('checkpoint_step')
    require(type(size) is int and size in (256,512) and type(rep) is int and rep in (0,1) and
        type(index) is int,'Exact game count, seed and optimizer index')
    require(meta.get('fit_id')==f'g{size}_r{rep}_mixed' and meta.get('seed')==sampling.SEEDS[rep] and
        meta.get('breadth')=='mixed' and meta.get('scope')=='joint','Exact mixed joint fit identity')
    for key,value in dict(source_training_games=size,fixed_final_step=STEPS,batch_size=BATCH,
        parameter_count=accepted.PARAMETERS,model_cpu=4,original_reference_cpu=0).items():
        require(type(meta.get(key)) is int and meta[key]==value,'Fixed checkpoint '+key)
    one=meta.get('one_pass_step');roots=meta.get('training_records');last=meta.get('last_batch_size')
    require(type(roots) is int and roots>=size and type(one) is int and 1<one<STEPS and
        type(last) is int and 1<=last<=32 and (one-1)*32+last==roots,'Exact genuine one-pass accounting')
    allowed=[0,one-1,one,STEPS-1,STEPS]
    require(meta.get('retained_checkpoint_steps')==allowed and index in ((0,1) if meta['qualification_only'] else allowed),
        'Only prospective qualification/boundary checkpoints')
    expected_endpoint=('qualification' if meta['qualification_only'] else
        ('onepass' if index==one else 'matched' if index==STEPS else 'boundary'))
    require(meta.get('endpoint')==expected_endpoint,'Checkpoint endpoint identity')
    require(meta.get('historical_data_relabelled') is False and meta.get('historical_warm_start_sha256')==accepted.PARENT_SHA256 and
        meta.get('checkpoint_selection') is False and meta.get('best_seed_selection') is False and
        meta.get('cpu0_training_executed') is False and meta.get('value_target_contract')==targets.contract('mixed'),
        'No historical relabelling, model selection, CPU0 training or target drift')
    require(meta.get('batch_reduction')=='mean_actual_batch_size_equal_record_weights','Actual batch denominator')
    require(meta.get('model_source_sha256')==accepted.MODEL_SOURCE_SHA256 and
        meta.get('optimizer_recipe')=={k:v for k,v in accepted.RECIPE.items() if k not in ('epochs','shuffle','example_weight','batch_reduction')} and
        meta.get('training_initializer')=='original_EF8A_and_fresh_AdamW_same_seed','Exact unchanged model and optimizer recipe')
    if meta['qualification_only']:
        require(type(meta.get('qualification_batch_size')) is int and
            meta['qualification_batch_size']==last,'Actual subset partial qualification batch')
    require(type(meta.get('record_presentations')) is int and 0<=meta['record_presentations']<=index*32,
        'Bounded actual record presentations')
    for key in ('selection_sha256','schedule_sha256','value_target_sources_sha256'):sampling._sha(meta.get(key))
    c.verify([meta['corpus_cache']]);return meta


def save_checkpoint(model,optimizer,index,meta,schedule,folder,*,boundary=lambda:None):
    import torch
    boundary();accepted.validate_architecture(model)
    actual=dict(meta,checkpoint_step=index,record_presentations=(sum(len(x) for x in schedule['batches'][:index])
        if not meta['qualification_only'] else (0 if index==0 else meta['qualification_batch_size'])),
        endpoint=('qualification' if meta['qualification_only'] else
            'onepass' if index==meta['one_pass_step'] else 'matched' if index==STEPS else 'boundary'),
        weights_sha256=accepted.state_digest(model),group_sha256=accepted.group_digests(model))
    validate_checkpoint_metadata(actual)
    payload=dict(schema='data_size_checkpoint_v1',kind='entity',metadata=actual,
        state_dict=model.state_dict(),optimizer=optimizer.state_dict(),rng=rng_state())
    verify_optimizer_payload(payload)
    folder=Path(folder);require(folder.resolve().is_relative_to(c.OUT),'New output checkpoint only')
    folder.mkdir(parents=True,exist_ok=True);path=folder/'checkpoint.pt'
    with path.open('xb') as stream:torch.save(payload,stream);stream.flush();os.fsync(stream.fileno())
    trees={k:tensor_tree(payload[k]) for k in ('state_dict','optimizer','rng')}
    item=c.save(folder/'tensor_manifest.json',trees)
    result=c.save(folder/'checkpoint_manifest.json',dict(schema='data_size_checkpoint_manifest_v1',
        checkpoint=c.artifact(path),tensor_manifest=item,metadata=actual,tree_sha256={k:digest(v) for k,v in trees.items()}))
    boundary();return result


def checkpoint_payload(item,*,boundary=lambda:None):
    import torch
    c.verify([item],boundary=boundary);manifest=c.read(c.resolve(item))
    require(manifest['schema']=='data_size_checkpoint_manifest_v1','New checkpoint envelope')
    validate_checkpoint_metadata(manifest['metadata']);descriptor=manifest['checkpoint']
    require(0<descriptor['bytes']<256*1024**2,'Bounded checkpoint archive')
    c.verify([descriptor,manifest['tensor_manifest']],boundary=boundary)
    path=c.resolve(descriptor);before=path.stat()
    with path.open('rb') as stream:payload=torch.load(stream,map_location='cpu',weights_only=True)
    after=path.stat();require((before.st_size,before.st_mtime_ns)==(after.st_size,after.st_mtime_ns),'Stable checkpoint read')
    c.verify([descriptor],boundary=boundary)
    require(payload['schema']=='data_size_checkpoint_v1' and payload['kind']=='entity' and
        payload['metadata']==manifest['metadata'],'Exact payload metadata binding')
    hashes=old.state_mapping_hashes(payload['state_dict'])
    require(hashes.pop('all')==manifest['metadata']['weights_sha256'] and hashes==manifest['metadata']['group_sha256'],
        'Every saved model tensor bit')
    verify_optimizer_payload(payload)
    trees={k:tensor_tree(payload[k]) for k in ('state_dict','optimizer','rng')}
    require(trees==c.read(c.resolve(manifest['tensor_manifest'])) and
        {k:digest(v) for k,v in trees.items()}==manifest['tree_sha256'],'Exact optimizer and RNG byte manifests')
    return payload,manifest


def load_checkpoint(item,*,optimizer=False,boundary=lambda:None):
    payload,manifest=checkpoint_payload(item,boundary=boundary)
    model=accepted.load_parent();model.load_state_dict(payload['state_dict'],strict=True);model.eval()
    accepted.validate_architecture(model)
    require(accepted.state_digest(model)==manifest['metadata']['weights_sha256'],'Exact restored model')
    opt=make_optimizer(model,'joint');opt.load_state_dict(payload['optimizer'])
    require(tensor_tree(opt.state_dict())==tensor_tree(payload['optimizer']),'Exact restored AdamW')
    restore_rng(payload['rng']);require(tensor_tree(rng_state())==tensor_tree(payload['rng']),'Exact restored RNG')
    return model,opt if optimizer else None,manifest


def source_equivalence():
    import ast
    def functions(path):
        text=Path(path).read_text(encoding='utf8')
        return {x.name:x for x in ast.parse(text).body if isinstance(x,ast.FunctionDef)}
    original,actual=functions(old.__file__),functions(__file__)
    class PartialAdmission(ast.NodeTransformer):
        def visit_Compare(self,node):
            if ast.unparse(node)=="len(batch['tokens']) in (30, 32)":
                return ast.parse("1 <= len(batch['tokens']) <= BATCH",mode='eval').body
            if ast.unparse(node)=="len(examples) in (30, 32)":
                return ast.parse("1 <= len(examples) <= BATCH",mode='eval').body
            return self.generic_visit(node)
        def visit_Constant(self,node):
            replacement={'Complete32 or final partial30 batch':'Genuine full or partial batch of1..32',
                'Full or final partial saved batch':'Genuine full or partial saved batch of1..32'}
            return ast.Constant(value=replacement[node.value]) if isinstance(node.value,str) and node.value in replacement else node
    require(ast.dump(PartialAdmission().visit(original['step']),include_attributes=False)==
        ast.dump(actual['step'],include_attributes=False),'Only numerical step change is accepted actual batch-size range')
    require(ast.dump(original['recorded_step'],include_attributes=False)==ast.dump(actual['recorded_step'],include_attributes=False),
        'Unchanged complete gradient evidence')
    require(ast.dump(PartialAdmission().visit(original['saved_batch_statistics']),include_attributes=False)==
        ast.dump(actual['saved_batch_statistics'],include_attributes=False),'Only saved loss change is actual batch-size admission')
    from tools.splendorust_training_duration_20261113 import fit as duration
    before=functions(duration.__file__)
    for name in ('fixture_cache_roundtrip','qualify_training_inputs'):
        require(ast.dump(before[name],include_attributes=False)==ast.dump(actual[name],include_attributes=False),
            'Same actual corrected production fixture path: '+name)
    require(loss_components is old.loss_components and make_optimizer is old.make_optimizer and
        selected_batch is old.selected_batch and restore_rng is old.restore_rng,'Direct original optimizer/loss/collator/RNG functions')
    return dict(status='same_numeric_recipe_generalized_partial_admission',old_math=old.source_equivalence(),
        actual_partial_range=[1,32],models=0,optimizer_calls=0,original_sources=[c.artifact(old.__file__),c.artifact(duration.__file__)])


def run_fit(game_count,repetition,folder,*,boundary=lambda:None,runtime_sources=None):
    folder=Path(folder);engine(folder,runtime_sources=runtime_sources)
    selection,loaded,schedule=load_training(game_count,repetition,boundary=boundary)
    meta=metadata(game_count,repetition,loaded,schedule)
    schedule_digest,selection_digest=digest(schedule),digest(selection)
    schedule_item=c.save(folder/'schedules.json',schedule)
    seed_rng(repetition);model=accepted.load_parent();opt=make_optimizer(model,'joint')
    initial=accepted.group_digests(model)
    checkpoints={'0':save_checkpoint(model,opt,0,meta,schedule,folder/'checkpoints/0',boundary=boundary)};updates=[]
    for index,indices in enumerate(schedule['batches'],start=1):
        boundary();update=recorded_step(model,opt,selected_batch(loaded,indices),'joint',boundary=boundary)
        update.update(game_count=game_count,repetition=repetition,step=index,
            example_indices=indices,schedule_sha256=schedule_digest,selection_sha256=selection_digest,
            batch_source_sha256=digest([[loaded['train_examples'][i]['game_index'],loaded['train_examples'][i]['ply'],
                loaded['train_examples'][i]['row_sha256']] for i in indices]))
        updates.append(c.save(folder/'updates'/f'{index:04d}.json',update))
        if index in schedule['checkpoint_steps']:
            checkpoints[str(index)]=save_checkpoint(model,opt,index,meta,schedule,folder/'checkpoints'/str(index),boundary=boundary)
        if index%32==0 or index in (schedule['one_pass_step'],STEPS):
            c.status(folder/'progress.json',dict(game_count=game_count,repetition=repetition,completed_updates=index,
                total_updates=STEPS,one_pass_step=schedule['one_pass_step'],time=c.now()))
    final=accepted.group_digests(model);del model,opt
    replays=[]
    for endpoint,index in [('onepass',schedule['one_pass_step']),('matched',STEPS)]:
        model,opt,_=load_checkpoint(checkpoints[str(index-1)],optimizer=True,boundary=boundary)
        replay=recorded_step(model,opt,selected_batch(loaded,schedule['batches'][index-1]),'joint',boundary=boundary)
        item=save_checkpoint(model,opt,index,meta,schedule,folder/'replay'/endpoint/'checkpoint',boundary=boundary)
        c.save(folder/'replay'/endpoint/'actual_update.json',replay)
        expected=c.read(c.resolve(checkpoints[str(index)]));actual=c.read(c.resolve(item))
        require(expected['tree_sha256']==actual['tree_sha256'],'Exact endpoint model/AdamW/RNG replay')
        compare_update(replay,c.read(c.resolve(updates[index-1])))
        replays.append(c.save(folder/'replay'/endpoint/'result.json',dict(status='exact_final_update_reproduced',
            endpoint=endpoint,step=index,from_checkpoint=checkpoints[str(index-1)],expected_checkpoint=checkpoints[str(index)],
            actual_checkpoint=item,expected_update=updates[index-1],actual_update=replay,tree_sha256=expected['tree_sha256'],optimizer_calls=1)))
        del model,opt
    result=dict(schema='data_size_fit_v1',status='fit_complete',study_id=c.STUDY,game_count=game_count,repetition=repetition,
        fit_id=meta['fit_id'],startup=c.artifact(folder/'startup.json'),schedules=schedule_item,checkpoints=checkpoints,
        endpoint_steps=dict(onepass=schedule['one_pass_step'],matched=STEPS),
        endpoints={label:checkpoints[str(index)] for label,index in [('onepass',schedule['one_pass_step']),('matched',STEPS)]},
        updates=updates,replays=replays,optimizer_calls=STEPS,replay_optimizer_calls=2,
        record_presentations=sum(map(len,schedule['batches'])),training_records=len(loaded['train_examples']),
        one_pass_step=schedule['one_pass_step'],before_group_sha256=initial,after_group_sha256=final,
        checkpoint_selection=False,optimizer_restart=False,epoch_rng_reseed=False,cpu0_training_executed=False,**IDENTITY)
    c.save(folder/'fit_result.json',result);boundary();return result


def verify_fit(folder,*,boundary=lambda:None):
    folder=Path(folder);result_item=c.artifact(folder/'fit_result.json');result=c.read(c.resolve(result_item))
    require(result['startup']==c.artifact(folder/'startup.json'),'Exact startup receipt binding')
    verify_startup(result['startup'],boundary=boundary)
    size,rep=result['game_count'],result['repetition'];selection,loaded,schedule=load_training(size,rep,boundary=boundary)
    schedule_digest,selection_digest=digest(schedule),digest(selection)
    require(result['schema']=='data_size_fit_v1' and result['study_id']==c.STUDY and result['status']=='fit_complete' and
        result['optimizer_calls']==STEPS and result['replay_optimizer_calls']==2 and
        result['record_presentations']==sum(map(len,schedule['batches'])) and result['training_records']==len(loaded['train_examples']) and
        result['one_pass_step']==schedule['one_pass_step'] and result['checkpoint_selection'] is False and
        result['optimizer_restart'] is False and result['epoch_rng_reseed'] is False and result['cpu0_training_executed'] is False,
        'Complete prospective fixed work and source scope')
    c.verify([result['schedules']],boundary=boundary);require(c.read(c.resolve(result['schedules']))==schedule,'Exact prospective batches')
    expected_meta=metadata(size,rep,loaded,schedule);expected_steps=[0,*schedule['checkpoint_steps']]
    require(set(result['checkpoints'])==set(map(str,expected_steps)),'Exact checkpoint inventory')
    manifests={};gradient_names=None
    for index in expected_steps:
        payload,manifest=checkpoint_payload(result['checkpoints'][str(index)],boundary=boundary)
        require(manifest['metadata']['checkpoint_step']==index and
            all(manifest['metadata'][k]==v for k,v in expected_meta.items()),'Exact frozen fit checkpoint metadata')
        require(manifest['metadata']['record_presentations']==sum(map(len,schedule['batches'][:index])),'Exact cumulative examples')
        manifests[index]=manifest;gradient_names=set(payload['state_dict']);del payload
    require(manifests[0]['metadata']['group_sha256']==result['before_group_sha256'],'Exact initial checkpoint state')
    # The initial snapshot must equal the independently authenticated EF8A reference.
    oldzero=c.read(c.resolve(source_checkpoint(rep,0)))
    require(manifests[0]['tree_sha256']==oldzero['tree_sha256'],'Same original model, fresh optimizer and seeded RNG as1024 reference')
    expected_updates=[c.artifact(folder/'updates'/f'{index:04d}.json') for index in range(1,STEPS+1)]
    require(result['updates']==expected_updates and len(list((folder/'updates').glob('*.json')))==STEPS,'Every update exactly once')
    groups=result['before_group_sha256'];clipped=0
    for index,item in enumerate(expected_updates,start=1):
        boundary();update=c.read(c.resolve(item));indices=schedule['batches'][index-1]
        examples=[loaded['train_examples'][i] for i in indices]
        require(type(update['step']) is int and update['step']==index and update['game_count']==size and update['repetition']==rep and
            update['example_indices']==indices and update['schedule_sha256']==schedule_digest and
            update['selection_sha256']==selection_digest and update['batch_source_sha256']==digest(
                [[x['game_index'],x['ply'],x['row_sha256']] for x in examples]),'Exact prospective source update')
        require(update['batch_examples']==update['batch_denominator']==len(indices) and
            update['forward_backward_calls']==update['optimizer_calls']==1 and update['finite_gradients'] is True and
            math.isfinite(update['gradient_norm_before_clip']) and update['gradient_norm_before_clip']>=0 and
            update['clipped']==(update['gradient_norm_before_clip']>1.) and update['before_group_sha256']==groups and
            update['after_group_sha256']!=groups,'Finite continuous update chain')
        require(update['gradient_evidence']=='all_trainable_clipped_float32_gradient_bytes' and
            set(update['gradient_sha256'])==gradient_names,'All actual gradient bytes recorded')
        for value in update['gradient_sha256'].values():sampling._sha(value)
        expected=saved_batch_statistics(examples,update['outputs83'])
        require(all(math.isclose(update[k],v,rel_tol=1e-6,abs_tol=1e-5) for k,v in expected.items()),'Independent saved loss arithmetic')
        groups=update['after_group_sha256'];clipped+=int(update['clipped'])
        if index in manifests:require(groups==manifests[index]['metadata']['group_sha256'],'Checkpoint on exact update chain')
    require(groups==result['after_group_sha256'] and len(result['replays'])==2,'Exact final state and two endpoint replays')
    for (label,index),item in zip([('onepass',schedule['one_pass_step']),('matched',STEPS)],result['replays'],strict=True):
        c.verify([item],boundary=boundary);replay=c.read(c.resolve(item))
        payload,manifest=checkpoint_payload(replay['actual_checkpoint'],boundary=boundary);del payload
        require(replay['status']=='exact_final_update_reproduced' and replay['endpoint']==label and replay['step']==index and
            replay['optimizer_calls']==1 and replay['from_checkpoint']==result['checkpoints'][str(index-1)] and
            replay['expected_checkpoint']==result['checkpoints'][str(index)] and replay['expected_update']==expected_updates[index-1] and
            manifest['tree_sha256']==manifests[index]['tree_sha256']==replay['tree_sha256'],'Exact saved endpoint replay chain')
        compare_update(replay['actual_update'],c.read(c.resolve(replay['expected_update'])))
    require(result['endpoints']=={label:result['checkpoints'][str(index)] for label,index in
        [('onepass',schedule['one_pass_step']),('matched',STEPS)]},'Exact assessment endpoints')
    require(result['endpoint_steps']==dict(onepass=schedule['one_pass_step'],matched=STEPS),'Exact assessment endpoint steps')
    return dict(status='data_size_fit_saved_verified',result=result_item,game_count=size,repetition=rep,optimizer_calls=STEPS,
        replay_optimizer_calls=2,clipped_steps=clipped,checkpoints=result['checkpoints'],endpoints=result['endpoints'],
        endpoint_steps=result['endpoint_steps'],**IDENTITY)


def _saved_prediction(model_id):
    result=c.read(c.ORIGINAL_OUT/'offline/worker/evaluation_result.json')
    require(result['status']=='all_fixed_predictions_complete','Accepted saved evaluator')
    row=next(x for x in result['models'] if x['model_id']==model_id)
    path=c.ORIGINAL_OUT/'offline/worker/predictions'/model_id/'train/000000.json'
    item=c.artifact(path);require(item in row['batches'],'Saved CPU0 first train batch bound in accepted evaluation')
    c.verify([item]);return item,c.read(path)


def run_qualification(folder,*,boundary=lambda:None,runtime_sources=None):
    folder=Path(folder);engine(folder,runtime_sources=runtime_sources)
    equivalence=source_equivalence();inputs=qualify_training_inputs(folder/'inputs',boundary=boundary)
    from . import offline
    import torch
    original_selection=c.read(c.SOURCE_SELECTION)
    original_training=sampling.load_diagnostic_train(original_selection,boundary=boundary)
    cpu_proofs=[];evaluations=[]
    duration_path=c.ROOT/'outputs/splendorust_training_duration_20261113_01/jobs/qualification/worker/qualification_result.json'
    duration_item=c.artifact(duration_path);duration=c.read(duration_path)
    require(duration['status']=='qualification_passed' and duration['study_id']=='splendorust_training_duration_20261113_v1',
        'Accepted saved CPU0 evaluator fixture evidence')
    for rep in (0,1):
        loaded=old.validate_loaded(sampling.load_selected(original_selection,rep,'mixed',boundary=boundary))
        _,expected=old.checkpoint_payload(source_checkpoint(rep),boundary=boundary)
        model,opt,_=old.load_checkpoint(source_checkpoint(rep,1781),optimizer=True,boundary=boundary)
        update=recorded_step(model,opt,selected_batch(loaded,loaded['batches'][-1]),'joint',boundary=boundary)
        actual=live_trees(model,opt)
        c.save(folder/'observations'/f'cpu4_r{rep}_update.json',dict(actual_update=update,actual_tree_sha256=actual))
        require(actual==expected['tree_sha256'],'CPU4 exact saved CPU0 model/AdamW/RNG afterlast30')
        source_update=c.artifact(c.ORIGINAL_OUT/'fits'/f'r{rep}_mixed_joint'/'worker/updates/1782.json')
        compare_update(update,c.read(c.resolve(source_update)))
        cpu_proofs.append(dict(repetition=rep,source_before=source_checkpoint(rep,1781),source_after=source_checkpoint(rep),
            source_update=source_update,actual_update=update,actual_tree_sha256=actual,optimizer_calls=1,
            cpu0_executed=False,discarded=True))
        del model,opt
        model,opt,_=old.load_checkpoint(source_checkpoint(rep),optimizer=True,boundary=boundary)
        before_rng=digest(tensor_tree(rng_state()))
        if rep==0:
            saved=duration['evaluator_qualification']
            structural=offline.predict_outputs(model,inputs['qualification_structural_examples'])
            c.save(folder/'observations'/'r0_structural.json',dict(outputs83=structural))
            require(accepted.full_output_bytes(structural)==accepted.full_output_bytes(saved['structural_outputs83']),
                'CPU4 exact CPU0 structural evaluator output bits')
            genuine=[targets.for_arm(x,'terminal') for x in inputs['qualification_examples']]
            rows=offline.predict_batch(model,genuine)
            c.save(folder/'observations'/'r0_genuine.json',dict(rows=rows))
            require(rows==saved['prediction_rows'],'CPU4 exact CPU0 genuine corrected evaluator records')
            evaluations.append(dict(model_id='r0_mixed_joint',saved_cpu0=duration_item,forward_calls=2,
                structural_outputs83=structural,genuine_rows=rows,structural_rows=len(structural),genuine_count=len(rows),
                structural_outcomes_fabricated=False))
        else:
            source,packet=_saved_prediction('r1_mixed_joint')
            rows=offline.predict_batch(model,original_training[:32])
            c.save(folder/'observations'/'r1_evaluator.json',dict(rows=rows))
            require(rows==packet['rows'],'CPU4 exact saved CPU0 r1 evaluator batch')
            evaluations.append(dict(model_id='r1_mixed_joint',saved_cpu0=source,forward_calls=1,rows=rows))
        require(before_rng==digest(tensor_tree(rng_state())),'Evaluator leaves restored training RNG unchanged')
        del model,opt
    model=accepted.load_parent();model.eval();source,packet=_saved_prediction('parent')
    rows=offline.predict_batch(model,original_training[:32])
    c.save(folder/'observations'/'parent_evaluator.json',dict(rows=rows))
    require(rows==packet['rows'],'CPU4 exact saved CPU0 parent evaluator batch')
    evaluations.append(dict(model_id='parent',saved_cpu0=source,forward_calls=1,rows=rows));del model
    partials=[]
    for size in (256,512):
        _,loaded,schedule=load_training(size,0,boundary=boundary)
        indices=schedule['batches'][schedule['one_pass_step']-1];batch=selected_batch(loaded,indices)
        seed_rng(0);model=accepted.load_parent();opt=make_optimizer(model,'joint')
        meta=metadata(size,0,loaded,schedule,qualification=True);meta['qualification_batch_size']=len(indices)
        before=save_checkpoint(model,opt,0,meta,schedule,folder/f'partial{size}/before',boundary=boundary)
        direct=recorded_step(model,opt,batch,'joint',boundary=boundary)
        c.save(folder/'observations'/f'partial{size}_direct.json',direct)
        after=save_checkpoint(model,opt,1,meta,schedule,folder/f'partial{size}/after',boundary=boundary)
        expected=live_trees(model,opt);del model,opt
        model,opt,_=load_checkpoint(before,optimizer=True,boundary=boundary)
        repeated=recorded_step(model,opt,batch,'joint',boundary=boundary)
        repeated_tree=live_trees(model,opt)
        c.save(folder/'observations'/f'partial{size}_repeated.json',dict(actual_update=repeated,actual_tree_sha256=repeated_tree))
        compare_update(repeated,direct);require(repeated_tree==expected,'Exact actual subset partial update afterreload')
        partials.append(dict(game_count=size,repetition=0,batch_size=len(indices),example_indices=indices,
            before=before,after=after,direct_update=direct,repeated_update=repeated,after_tree_sha256=expected,
            optimizer_calls=2,discarded=True));del model,opt
    result=dict(schema='data_size_qualification_v1',status='qualification_passed',study_id=c.STUDY,
        startup=c.artifact(folder/'startup.json'),
        data_qualification=inputs['proof'],numerical_equivalence=equivalence,cpu4_exact_cpu0_updates=cpu_proofs,
        evaluator_qualification=evaluations,actual_partial_batches=partials,optimizer_calls=6,
        evaluator_forward_calls=4,scientific_optimizer_calls=0,cpu0_executed=False,**IDENTITY)
    c.save(folder/'qualification_result.json',result);boundary();return result


def verify_qualification(folder,*,boundary=lambda:None):
    item=c.artifact(Path(folder)/'qualification_result.json');result=c.read(c.resolve(item))
    require(result['startup']==c.artifact(Path(folder)/'startup.json'),'Exact qualification startup receipt binding')
    verify_startup(result['startup'],boundary=boundary)
    require(result['status']=='qualification_passed' and result['study_id']==c.STUDY and result['optimizer_calls']==6 and
        result['evaluator_forward_calls']==4 and result['scientific_optimizer_calls']==0 and result['cpu0_executed'] is False,
        'Complete bounded CPU4 qualification')
    proof=result['data_qualification'];require(proof['actual_selected_loader_and_collator_exercised'] is True and
        proof['actual_mmap_training_path_qualified'] is True,'Actual corrected loader/collator/cache qualified')
    c.verify(proof['cache_feature_artifacts'],boundary=boundary)
    require([x['repetition'] for x in result['cpu4_exact_cpu0_updates']]==[0,1],'Both original training seed comparisons')
    for row in result['cpu4_exact_cpu0_updates']:
        c.verify([row['source_before'],row['source_after'],row['source_update']],boundary=boundary)
        require(row['optimizer_calls']==1 and row['cpu0_executed'] is False and row['discarded'] is True,'Only discarded CPU4 replay')
        compare_update(row['actual_update'],c.read(c.resolve(row['source_update'])))
        require(row['actual_tree_sha256']==c.read(c.resolve(row['source_after']))['tree_sha256'],
            'Exact CPU0 reference model/optimizer/RNG receipt')
    require([x['model_id'] for x in result['evaluator_qualification']]==['r0_mixed_joint','r1_mixed_joint','parent'],'Fixed three model evaluator comparisons')
    for row in result['evaluator_qualification']:
        c.verify([row['saved_cpu0']],boundary=boundary);saved=c.read(c.resolve(row['saved_cpu0']))
        if row['model_id']=='r0_mixed_joint':
            saved=saved['evaluator_qualification']
            require(row['forward_calls']==2 and row['structural_outcomes_fabricated'] is False and
                row['genuine_rows']==saved['prediction_rows'] and row['genuine_count']==32 and
                accepted.full_output_bytes(row['structural_outputs83'])==accepted.full_output_bytes(saved['structural_outputs83']),
                'Saved corrected structural/genuine CPU4 evaluator exact CPU0 proof')
        else:require(row['forward_calls']==1 and row['rows']==saved['rows'],'Saved CPU4 reference evaluation exact CPU0 proof')
    require([x['game_count'] for x in result['actual_partial_batches']]==[256,512],'Both actual partial batch sizes')
    for row in result['actual_partial_batches']:
        _,loaded,schedule=load_training(row['game_count'],0,boundary=boundary)
        indices=schedule['batches'][schedule['one_pass_step']-1]
        require(row['batch_size']==len(indices) and row['example_indices']==indices and row['optimizer_calls']==2 and row['discarded'] is True,
            'Actual planned subset partial replay')
        before_payload,before=checkpoint_payload(row['before'],boundary=boundary)
        after_payload,after=checkpoint_payload(row['after'],boundary=boundary)
        require(before['metadata']['checkpoint_step']==0 and after['metadata']['checkpoint_step']==1 and
            before['metadata']['qualification_only'] is True and after['metadata']['qualification_only'] is True and
            after['tree_sha256']==row['after_tree_sha256'],'Exact qualification checkpoint pair')
        compare_update(row['repeated_update'],row['direct_update'])
        require(row['direct_update']['batch_examples']==row['direct_update']['batch_denominator']==len(indices) and
            row['direct_update']['before_group_sha256']==before['metadata']['group_sha256'] and
            row['direct_update']['after_group_sha256']==after['metadata']['group_sha256'],'Actual partial update boundary')
        expected=saved_batch_statistics([loaded['train_examples'][i] for i in indices],row['direct_update']['outputs83'])
        require(all(math.isclose(row['direct_update'][k],v,rel_tol=1e-6,abs_tol=1e-5) for k,v in expected.items()),
            'Independent partial-batch mean loss')
        del before_payload,after_payload
    return dict(status='qualification_saved_verified',result=item,optimizer_calls=6,evaluator_forward_calls=4,
        cpu4_exact_cpu0_math=True,cpu0_executed=False,**IDENTITY)
