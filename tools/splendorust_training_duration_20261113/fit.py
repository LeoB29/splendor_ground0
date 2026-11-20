"""Authenticated continuation of the two accepted mixed-target one-pass fits.

Numerical updates are direct calls to frozen qualified functions. Only schedules,
continuation metadata, checkpoint envelopes and new output paths change.
"""
from __future__ import annotations
from copy import deepcopy
from pathlib import Path
import hashlib
import math
import os
import time
from . import common as c
from tools import future_training_rules_20261021 as guard
from tools.splendorust_full_corpus_20261110 import fit as old
from tools.splendorust_full_corpus_20261110 import sampling, corpus, targets
from tools.splendorust_policy_head_20261104 import data as accepted_data
from tools.splendorust_independent_data_20261107.collection import common as cc, data, storage
from tools.splendorust_independent_data_20261107 import schedule

accepted = old.accepted
IDENTITY = sampling.IDENTITY
TRAIN_ROOTS, STEPS, BATCH = 57022, 1782, 32
PASSES = (2, 3, 4)
CHECKPOINTS = (3563, 3564, 5345, 5346, 7127, 7128)
QUALIFICATION_CHECKPOINTS = (1782, 1783)
_STARTUP = _ENGINE = None
require, canonical, digest = c.require, c.canonical, c.digest
tensor_tree, rng_state, restore_rng = old.tensor_tree, old.rng_state, old.restore_rng
recorded_step, selected_batch = old.recorded_step, old.selected_batch
make_optimizer, verify_optimizer_payload = old.make_optimizer, old.verify_optimizer_payload
_validate_features, _target, _parse_target = corpus._validate_features, corpus._target, corpus._parse_target
feature_arrays, index_row, attach_game_targets = corpus.feature_arrays, corpus.index_row, corpus.attach_game_targets


def startup(preverified=None):
    global _STARTUP
    if _STARTUP is None:
        _STARTUP = deepcopy(preverified) if preverified is not None else guard.verify_runtime_sources()
        require(type(_STARTUP) is dict, 'Actual corrected runtime receipt')
    elif preverified is not None:
        require(_STARTUP == preverified, 'One unchanged startup receipt')
    return deepcopy(_STARTUP)


def engine(folder, *, runtime_sources=None):
    global _ENGINE
    sources = startup(runtime_sources)
    if _ENGINE is None:
        import psutil
        process = psutil.Process()
        process.cpu_affinity([0]); process.nice(psutil.NORMAL_PRIORITY_CLASS)
        require(process.cpu_affinity() == [0], 'Actual unchanged CPU0 placement')
        import torch
        torch.set_num_threads(1); torch.set_num_interop_threads(1)
        _ENGINE = dict(pid=process.pid, created=process.create_time(), cpu_affinity=[0],
                       priority='NORMAL_PRIORITY_CLASS', torch_threads=1,
                       torch_interop_threads=1, device='cpu', precision='float32')
    c.save(Path(folder)/'startup.json', dict(runtime_sources=sources, engine=_ENGINE, **IDENTITY))
    return deepcopy(_ENGINE)


def source_checkpoint(repetition, step=1782):
    require(type(repetition) is int and repetition in (0, 1), 'Fixed training seed')
    require(type(step) is int and step in (1781, 1782), 'Accepted source checkpoint')
    path = c.ORIGINAL_OUT/'fits'/f'r{repetition}_mixed_joint'/'worker/fit_result.json'
    result = c.read(path)
    require(result['status'] == 'fit_complete' and result['repetition'] == repetition and
            result['breadth'] == 'mixed' and result['scope'] == 'joint' and
            result['optimizer_calls'] == STEPS, 'Accepted mixed one-pass source fit')
    item = result['checkpoints'][str(step)]
    c.verify([item])
    return item


def source_inputs(repetition):
    folder = c.ORIGINAL_OUT/'fits'/f'r{repetition}_mixed_joint'
    return dict(selection=c.artifact(c.SELECTION), cache=c.artifact(c.CACHE_ACCEPTANCE),
                source_fit=c.artifact(folder/'worker/fit_result.json'),
                source_execution=c.artifact(folder/'execution.json'),
                source_checkpoint=source_checkpoint(repetition))


def build_schedules(selection, repetition):
    """Separate deterministic shuffles never read outcomes or consume model RNG."""
    require(type(repetition) is int and repetition in (0, 1), 'Fixed repetition')
    cell = next(x for x in selection['cells'] if x['fit_id'] == f'r{repetition}_mixed_joint')
    refs = cell['root_ids']
    require(len(refs) == TRAIN_ROOTS and len(set(map(tuple, refs))) == TRAIN_ROOTS,
            'All unique accepted training roots')
    schedules = {}
    for number in PASSES:
        order = sorted(range(TRAIN_ROOTS), key=lambda i: (
            digest([c.STUDY, 'full_pass', sampling.SEEDS[repetition], number, refs[i]]), i))
        schedules[str(number)] = [order[i:i+BATCH] for i in range(0, TRAIN_ROOTS, BATCH)]
    validate_schedules(schedules)
    return schedules


def validate_schedules(schedules):
    require(set(schedules) == {'2', '3', '4'}, 'Only prospective passes2/3/4')
    for batches in schedules.values():
        require(len(batches) == STEPS and all(type(b) is list and len(b) ==
                (30 if i == STEPS-1 else 32) and all(type(x) is int for x in b)
                for i, b in enumerate(batches)), '1781 full32 plus genuine final30')
        require(sorted(x for b in batches for x in b) == list(range(TRAIN_ROOTS)),
                'Complete pass without duplication or padding')
    require(len({digest(batches) for batches in schedules.values()}) == 3,
            'Distinct predetermined pass orders')
    return schedules


def load_training(repetition, *, boundary=lambda: None):
    selection = c.read(c.SELECTION)
    loaded = old.validate_loaded(sampling.load_selected(selection, repetition, 'mixed', boundary=boundary))
    schedules = build_schedules(selection, repetition)
    return selection, loaded, schedules


def metadata(repetition, loaded, schedules, *, qualification=False):
    require(type(repetition) is int and repetition in (0, 1), 'Fixed repetition')
    return dict(study_id=c.STUDY, fit_id=f'r{repetition}_mixed_duration', repetition=repetition,
        seed=sampling.SEEDS[repetition], breadth='mixed', scope='joint', **IDENTITY,
        qualification_only=qualification, historical_data_relabelled=False,
        historical_warm_start_sha256=accepted.PARENT_SHA256,
        parameter_count=accepted.PARAMETERS, model_source_sha256=accepted.MODEL_SOURCE_SHA256,
        source_one_pass_checkpoint=source_checkpoint(repetition), source_inputs=source_inputs(repetition),
        continuation_starts_at_step=STEPS, original_optimizer_and_rng_restored=True,
        optimizer_recipe={k:v for k,v in accepted.RECIPE.items() if k not in
                          ('epochs', 'shuffle', 'example_weight', 'batch_reduction')},
        training_records=TRAIN_ROOTS, source_training_games=1024, batch_size=32,
        last_batch_size=30, fixed_final_step=7128, diagnostic_passes=[1,2,4],
        checkpoint_selection=False, best_seed_selection=False,
        batch_reduction='mean_actual_batch_size_equal_record_weights',
        value_target_contract=targets.contract('mixed'),
        value_target_sources_sha256=loaded['value_target_sources_sha256'],
        corpus_cache=loaded['corpus_cache'], schedule_sha256=digest(schedules),
        selection_sha256=digest(c.read(c.SELECTION)))


def validate_checkpoint_metadata(meta):
    require(type(meta) is dict and meta.get('study_id') == c.STUDY and
            type(meta.get('qualification_only')) is bool and
            type(meta.get('checkpoint_step')) is int, 'Strict continuation checkpoint metadata')
    accepted_data.verify_identity(meta)
    rep = meta.get('repetition'); require(type(rep) is int and rep in (0,1), 'Fixed seed identity')
    require(meta.get('fit_id') == f'r{rep}_mixed_duration' and meta.get('seed') == sampling.SEEDS[rep]
            and meta.get('breadth') == 'mixed' and meta.get('scope') == 'joint', 'Mixed joint trajectory')
    allowed = QUALIFICATION_CHECKPOINTS if meta['qualification_only'] else CHECKPOINTS
    index = meta['checkpoint_step']
    require(index in allowed, 'Only declared continuation checkpoints')
    constants = dict(training_records=TRAIN_ROOTS, source_training_games=1024, batch_size=32,
                     last_batch_size=30, fixed_final_step=7128, continuation_starts_at_step=1782,
                     parameter_count=accepted.PARAMETERS)
    for key, value in constants.items():
        require(type(meta.get(key)) is int and meta[key] == value, 'Fixed checkpoint field '+key)
    require(meta.get('complete_passes') == index//STEPS and
            type(meta.get('complete_passes')) is int and
            meta.get('record_presentations') == (index//STEPS)*TRAIN_ROOTS+(index%STEPS)*32,
            'Exact cumulative update and example accounting')
    require(meta.get('checkpoint_selection') is False and meta.get('best_seed_selection') is False and
            meta.get('diagnostic_passes') == [1,2,4] and meta.get('original_optimizer_and_rng_restored') is True,
            'No selection, optimizer restart or extra diagnostic checkpoint')
    require(meta.get('historical_data_relabelled') is False and
            meta.get('historical_warm_start_sha256') == accepted.PARENT_SHA256 and
            meta.get('value_target_contract') == targets.contract('mixed'), 'Exact historical warmstart and mixed target')
    require(meta.get('batch_reduction') == 'mean_actual_batch_size_equal_record_weights', 'Actual batch denominator')
    for key in ('selection_sha256','schedule_sha256','value_target_sources_sha256'):
        sampling._sha(meta.get(key))
    c.verify([meta['source_one_pass_checkpoint'], meta['corpus_cache']])
    return meta


def save_checkpoint(model, optimizer, step_index, meta, folder, *, boundary=lambda: None):
    import torch
    boundary(); accepted.validate_architecture(model)
    metadata_ = {**meta, 'checkpoint_step':step_index, 'complete_passes':step_index//STEPS,
        'record_presentations':(step_index//STEPS)*TRAIN_ROOTS+(step_index%STEPS)*32,
        'weights_sha256':accepted.state_digest(model), 'group_sha256':accepted.group_digests(model)}
    validate_checkpoint_metadata(metadata_)
    payload = dict(schema='duration_checkpoint_v1', kind='entity', metadata=metadata_,
                   state_dict=model.state_dict(), optimizer=optimizer.state_dict(), rng=rng_state())
    verify_optimizer_payload(payload)
    folder=Path(folder); require(folder.resolve().is_relative_to(c.OUT), 'New checkpoint output only')
    folder.mkdir(parents=True, exist_ok=True)
    path=folder/'checkpoint.pt'
    with path.open('xb') as stream:
        torch.save(payload, stream); stream.flush(); os.fsync(stream.fileno())
    trees={k:tensor_tree(payload[k]) for k in ('state_dict','optimizer','rng')}
    tensor_item=c.save(folder/'tensor_manifest.json',trees)
    manifest=dict(schema='duration_checkpoint_manifest_v1', checkpoint=c.artifact(path),
        tensor_manifest=tensor_item, metadata=metadata_, tree_sha256={k:digest(v) for k,v in trees.items()})
    result=c.save(folder/'checkpoint_manifest.json',manifest);boundary();return result


def checkpoint_payload(item, *, boundary=lambda: None):
    import torch
    c.verify([item],boundary=boundary); manifest=c.read(c.resolve(item))
    require(manifest['schema']=='duration_checkpoint_manifest_v1','New checkpoint envelope')
    validate_checkpoint_metadata(manifest['metadata']); descriptor=manifest['checkpoint']
    require(0<descriptor['bytes']<256*1024**2,'Bounded checkpoint')
    c.verify([descriptor,manifest['tensor_manifest']],boundary=boundary)
    path=c.resolve(descriptor);before=path.stat()
    with path.open('rb') as stream:payload=torch.load(stream,map_location='cpu',weights_only=True)
    after=path.stat()
    require((before.st_size,before.st_mtime_ns)==(after.st_size,after.st_mtime_ns),'Archive stable while loading')
    c.verify([descriptor],boundary=boundary)
    require(payload['schema']=='duration_checkpoint_v1' and payload['kind']=='entity' and
            payload['metadata']==manifest['metadata'],'Exact payload metadata')
    hashes=old.state_mapping_hashes(payload['state_dict'])
    require(hashes.pop('all')==manifest['metadata']['weights_sha256'] and
            hashes==manifest['metadata']['group_sha256'],'Every saved model tensor authenticated')
    verify_optimizer_payload(payload)
    trees={k:tensor_tree(payload[k]) for k in ('state_dict','optimizer','rng')}
    require(trees==c.read(c.resolve(manifest['tensor_manifest'])) and
            {k:digest(v) for k,v in trees.items()}==manifest['tree_sha256'],'Exact model/optimizer/RNG bytes')
    boundary();return payload,manifest


def load_checkpoint(item, *, optimizer=False, boundary=lambda: None):
    payload,manifest=checkpoint_payload(item,boundary=boundary)
    model=accepted.load_parent();model.load_state_dict(payload['state_dict'],strict=True);model.eval()
    accepted.validate_architecture(model)
    require(accepted.state_digest(model)==manifest['metadata']['weights_sha256'],'Exact restored model')
    opt=make_optimizer(model,'joint');opt.load_state_dict(payload['optimizer'])
    require(tensor_tree(opt.state_dict())==tensor_tree(payload['optimizer']),'Restored full AdamW state')
    restore_rng(payload['rng'])
    require(tensor_tree(rng_state())==tensor_tree(payload['rng']),'Restored all RNG state')
    return model,opt if optimizer else None,manifest


def live_trees(model,optimizer):
    return {k:digest(tensor_tree(v)) for k,v in dict(state_dict=model.state_dict(),
        optimizer=optimizer.state_dict(),rng=rng_state()).items()}


def compare_update(actual,expected):
    require(all(actual[k]==expected[k] for k in actual if k!='seconds'),
            'Exact update outputs, loss, gradients and model state')


def source_equivalence():
    """Model-free proof that only fixture output ownership changed."""
    import ast
    original=Path(corpus.__file__).read_text(encoding='utf8')
    current=Path(__file__).read_text(encoding='utf8')
    def functions(source):
        return {x.name:x for x in ast.parse(source).body if isinstance(x,ast.FunctionDef)}
    before,after=functions(original),functions(current)
    require(ast.dump(before['fixture_cache_roundtrip'],include_attributes=False)==
            ast.dump(after['fixture_cache_roundtrip'],include_attributes=False),'Same fixture cache path operations')
    # The sampler module is the same accepted module, imported at module scope.
    class Imports(ast.NodeTransformer):
        def visit_ImportFrom(self,node):
            if node.module is None and node.level==1 and any(x.name=='sampling' for x in node.names):return None
            if node.module=='sampling' and node.level==1:
                node.module='tools.splendorust_full_corpus_20261110.sampling';node.level=0
            return node
        def visit_Return(self,node):
            if (isinstance(node.value,ast.Call) and isinstance(node.value.func,ast.Name) and node.value.func.id=='dict' and
                    any(k.arg=='qualification_examples' for k in node.value.keywords)):
                node.value.keywords.append(ast.keyword(arg='qualification_structural_examples',value=ast.Name(id='eligible',ctx=ast.Load())))
            return self.generic_visit(node)
    require(ast.dump(Imports().visit(before['qualify_training_inputs']),include_attributes=False)==
            ast.dump(after['qualify_training_inputs'],include_attributes=False),'Same actual corrected counterexamples')
    require(recorded_step is old.recorded_step and selected_batch is old.selected_batch and
            make_optimizer is old.make_optimizer and restore_rng is old.restore_rng,
            'Direct unchanged numerical implementations')
    return dict(status='unchanged_continuation_math_and_actual_fixture_route',
                original=old.source_equivalence(),models=0,optimizer_calls=0)


def run_qualification(folder, *, boundary=lambda: None, runtime_sources=None):
    folder=Path(folder);engine(folder,runtime_sources=runtime_sources)
    equivalence=source_equivalence()
    inputs=qualify_training_inputs(folder/'inputs',boundary=boundary)
    proofs=[];evaluator_proof=None
    for rep in (0,1):
        selection,loaded,schedules=load_training(rep,boundary=boundary)
        expected_payload,expected_manifest=old.checkpoint_payload(source_checkpoint(rep),boundary=boundary)
        model,opt,_=old.load_checkpoint(source_checkpoint(rep,1781),optimizer=True,boundary=boundary)
        replay=recorded_step(model,opt,selected_batch(loaded,loaded['batches'][-1]),'joint',boundary=boundary)
        last30_trees=live_trees(model,opt)
        require(last30_trees==expected_manifest['tree_sha256'],'Exact accepted original last30 replay')
        expected_update=c.read(c.ORIGINAL_OUT/'fits'/f'r{rep}_mixed_joint'/'worker/updates/1782.json')
        compare_update(replay,expected_update)
        del expected_payload,model,opt
        model,opt,_=old.load_checkpoint(source_checkpoint(rep),optimizer=True,boundary=boundary)
        if rep==0:
            import torch
            from . import offline
            examples=[targets.for_arm(row,'terminal') for row in inputs['qualification_examples']]
            before_rng=digest(tensor_tree(rng_state()))
            rows=offline.predict_batch(model,examples)
            batch_tokens=sampling.collate_inputs(examples,[1.]*len(examples))['tokens']
            with torch.no_grad():direct_outputs=accepted.tensor_forward(model,batch_tokens).tolist()
            require(accepted.full_output_bytes([r['output83'] for r in rows])==
                    accepted.full_output_bytes(direct_outputs),'Actual evaluator exact corrected fixture outputs')
            structural=inputs['qualification_structural_examples']
            structural_outputs=offline.predict_outputs(model,structural)
            structural_tokens=sampling.collate_inputs(structural,[1.]*len(structural))['tokens']
            with torch.no_grad():structural_reference=accepted.tensor_forward(model,structural_tokens).tolist()
            require(accepted.full_output_bytes(structural_outputs)==accepted.full_output_bytes(structural_reference),
                    'Actual evaluator handles corrected structural counterexamples exactly')
            require(before_rng==digest(tensor_tree(rng_state())),'Qualification inference does not perturb restored RNG')
            evaluator_proof=dict(status='actual_evaluator_route_qualified',records=len(rows),
                source='genuine_excluded_engineering_targets',prediction_rows=rows,
                direct_outputs83=direct_outputs,forward_calls=4,rng_unchanged=True,
                structural_records=len(structural),structural_outputs83=structural_outputs,
                structural_reference_outputs83=structural_reference,structural_official_targets_created=False)
        meta=metadata(rep,loaded,schedules,qualification=True)
        before=save_checkpoint(model,opt,1782,meta,folder/f'r{rep}/before',boundary=boundary)
        indices=schedules['2'][0];batch=selected_batch(loaded,indices)
        direct=recorded_step(model,opt,batch,'joint',boundary=boundary)
        after=save_checkpoint(model,opt,1783,meta,folder/f'r{rep}/direct',boundary=boundary)
        expected_trees=live_trees(model,opt);del model,opt
        model,opt,_=load_checkpoint(before,optimizer=True,boundary=boundary)
        resumed=recorded_step(model,opt,batch,'joint',boundary=boundary)
        compare_update(resumed,direct)
        require(live_trees(model,opt)==expected_trees,'Next actual32 continuation exact after save/reload')
        del model,opt
        proofs.append(dict(repetition=rep,source_last30=source_checkpoint(rep),
            source_update=c.artifact(c.ORIGINAL_OUT/'fits'/f'r{rep}_mixed_joint'/'worker/updates/1782.json'),
            last30_replay=replay,last30_tree_sha256=last30_trees,checkpoint_before=before,checkpoint_after=after,
            next32_indices=indices,direct_update=direct,resumed_update=resumed,
            after_tree_sha256=expected_trees,optimizer_calls=3,discarded=True))
    result=dict(schema='duration_qualification_v1',status='qualification_passed',study_id=c.STUDY,
        data_qualification=inputs['proof'],numerical_equivalence=equivalence,seeds=proofs,
        evaluator_qualification=evaluator_proof,extra_evaluator_forward_calls=4,
        optimizer_calls=6,scientific_optimizer_calls=0,**IDENTITY)
    c.save(folder/'qualification_result.json',result);boundary();return result


def verify_qualification(folder, *, boundary=lambda: None):
    item=c.artifact(Path(folder)/'qualification_result.json');result=c.read(c.resolve(item))
    require(result['status']=='qualification_passed' and result['study_id']==c.STUDY and
            result['optimizer_calls']==6 and result['scientific_optimizer_calls']==0 and
            [x['repetition'] for x in result['seeds']]==[0,1],'Complete two-seed continuation qualification')
    proof=result['data_qualification']
    require(proof['actual_selected_loader_and_collator_exercised'] is True and
            proof['actual_mmap_training_path_qualified'] is True,'Actual corrected entry point qualified')
    c.verify(proof['cache_feature_artifacts'],boundary=boundary)
    ev=result['evaluator_qualification']
    require(ev['status']=='actual_evaluator_route_qualified' and ev['records']==32 and ev['forward_calls']==4 and
            ev['rng_unchanged'] is True and result['extra_evaluator_forward_calls']==4 and
            accepted.full_output_bytes([r['output83'] for r in ev['prediction_rows']])==
            accepted.full_output_bytes(ev['direct_outputs83']),'Saved evaluator qualification parity')
    require(ev['structural_records']>0 and ev['structural_official_targets_created'] is False and
            accepted.full_output_bytes(ev['structural_outputs83'])==accepted.full_output_bytes(ev['structural_reference_outputs83']),
            'Saved structural evaluator qualification parity without fabricated outcome labels')
    for record in result['seeds']:
        require(record['optimizer_calls']==3 and record['discarded'] is True,'Only discarded qualification updates')
        compare_update(record['last30_replay'],c.read(c.resolve(record['source_update'])))
        c.verify([record['source_update'],record['source_last30']],boundary=boundary)
        source=c.read(c.resolve(record['source_last30']))
        require(record['last30_replay']['after_group_sha256']==source['metadata']['group_sha256'] and
                record['last30_tree_sha256']==source['tree_sha256'],'Old final30 replay restored final model, optimizer and RNG')
        before_payload,before=checkpoint_payload(record['checkpoint_before'],boundary=boundary)
        after_payload,after=checkpoint_payload(record['checkpoint_after'],boundary=boundary)
        require(before['metadata']['checkpoint_step']==1782 and after['metadata']['checkpoint_step']==1783 and
                before['metadata']['qualification_only'] is True and after['metadata']['qualification_only'] is True and
                before['tree_sha256']==source['tree_sha256'] and
                after['tree_sha256']==record['after_tree_sha256'],'Exact old state and continued state binding')
        compare_update(record['resumed_update'],record['direct_update'])
        require(record['direct_update']['batch_examples']==32 and
                record['direct_update']['before_group_sha256']==before['metadata']['group_sha256'] and
                record['direct_update']['after_group_sha256']==after['metadata']['group_sha256'],'Correct genuine next32')
        del before_payload,after_payload
    return dict(status='qualification_saved_verified',result=item,optimizer_calls=6,**IDENTITY)


def run_fit(repetition, folder, *, boundary=lambda: None, runtime_sources=None):
    folder=Path(folder);engine(folder,runtime_sources=runtime_sources)
    selection,loaded,schedules=load_training(repetition,boundary=boundary)
    meta=metadata(repetition,loaded,schedules)
    selection_sha, schedule_sha = digest(selection), digest(schedules)
    schedule_item=c.save(folder/'schedules.json',schedules)
    model,opt,source_manifest=old.load_checkpoint(source_checkpoint(repetition),optimizer=True,boundary=boundary)
    initial=accepted.group_digests(model);checkpoints={};updates=[]
    for number in PASSES:
        for local_index,indices in enumerate(schedules[str(number)],start=1):
            index=(number-1)*STEPS+local_index;boundary()
            update=recorded_step(model,opt,selected_batch(loaded,indices),'joint',boundary=boundary)
            update.update(repetition=repetition,pass_number=number,step=index,local_step=local_index,
                example_indices=indices,selection_sha256=selection_sha,schedule_sha256=schedule_sha,
                batch_source_sha256=digest([[loaded['train_examples'][i]['game_index'],
                    loaded['train_examples'][i]['ply'],loaded['train_examples'][i]['row_sha256']] for i in indices]))
            updates.append(c.save(folder/'updates'/f'{index:04d}.json',update))
            if index in CHECKPOINTS:
                checkpoints[str(index)]=save_checkpoint(model,opt,index,meta,
                    folder/'checkpoints'/str(index),boundary=boundary)
            if local_index%32==0 or local_index==STEPS:
                c.status(folder/'progress.json',dict(repetition=repetition,pass_number=number,
                    completed_new_updates=len(updates),absolute_step=index,total_new_updates=5346,time=c.now()))
    final=accepted.group_digests(model);del model,opt
    replays=[]
    for number in PASSES:
        index=number*STEPS
        model,opt,_=load_checkpoint(checkpoints[str(index-1)],optimizer=True,boundary=boundary)
        replay=recorded_step(model,opt,selected_batch(loaded,schedules[str(number)][-1]),'joint',boundary=boundary)
        item=save_checkpoint(model,opt,index,meta,folder/'replay'/str(index)/'checkpoint',boundary=boundary)
        expected=c.read(c.resolve(checkpoints[str(index)]));actual=c.read(c.resolve(item))
        require(expected['tree_sha256']==actual['tree_sha256'],'Exact final30 boundary replay')
        original_update=updates[index-STEPS-1];compare_update(replay,c.read(c.resolve(original_update)))
        replays.append(c.save(folder/'replay'/str(index)/'result.json',dict(status='exact_final_update_reproduced',
            pass_number=number,from_checkpoint=checkpoints[str(index-1)],expected_checkpoint=checkpoints[str(index)],
            actual_checkpoint=item,expected_update=original_update,actual_update=replay,
            tree_sha256=expected['tree_sha256'],optimizer_calls=1)))
        del model,opt
    result=dict(schema='duration_fit_v1',status='fit_complete',study_id=c.STUDY,repetition=repetition,
        source_inputs=source_inputs(repetition),source_checkpoint=source_checkpoint(repetition),
        source_tree_sha256=source_manifest['tree_sha256'],schedules=schedule_item,
        checkpoints=checkpoints,updates=updates,replays=replays,optimizer_calls=5346,
        disposable_optimizer_calls=3,new_record_presentations=TRAIN_ROOTS*3,total_record_presentations=TRAIN_ROOTS*4,
        before_group_sha256=initial,after_group_sha256=final,checkpoint_selection=False,
        optimizer_restart=False,epoch_rng_reseed=False,**IDENTITY)
    c.save(folder/'fit_result.json',result);boundary();return result


def verify_fit(folder, *, boundary=lambda: None):
    folder=Path(folder);result_item=c.artifact(folder/'fit_result.json');result=c.read(c.resolve(result_item))
    rep=result['repetition'];selection,loaded,schedules=load_training(rep,boundary=boundary)
    selection_sha, schedule_sha = digest(selection), digest(schedules)
    require(result['schema']=='duration_fit_v1' and result['study_id']==c.STUDY and result['status']=='fit_complete' and
            result['optimizer_calls']==5346 and result['disposable_optimizer_calls']==3 and
            result['new_record_presentations']==TRAIN_ROOTS*3 and result['total_record_presentations']==TRAIN_ROOTS*4 and
            result['checkpoint_selection'] is False and result['optimizer_restart'] is False and
            result['epoch_rng_reseed'] is False,'Complete fixed continuation work')
    require(result['source_inputs']==source_inputs(rep) and result['source_checkpoint']==source_checkpoint(rep),'Same accepted source')
    c.verify([result['schedules']],boundary=boundary)
    require(c.read(c.resolve(result['schedules']))==schedules,'Exact prospectively fixed pass orders')
    source_payload,source_manifest=old.checkpoint_payload(result['source_checkpoint'],boundary=boundary)
    gradient_names=set(source_payload['state_dict']);del source_payload
    require(result['source_tree_sha256']==source_manifest['tree_sha256'] and
            result['before_group_sha256']==source_manifest['metadata']['group_sha256'],'Continuation starts at accepted one-pass state')
    require(set(result['checkpoints'])==set(map(str,CHECKPOINTS)),'Exact boundary checkpoint inventory')
    expected_meta=metadata(rep,loaded,schedules);manifests={}
    for index in CHECKPOINTS:
        payload,manifest=checkpoint_payload(result['checkpoints'][str(index)],boundary=boundary)
        require(type(manifest['metadata']['checkpoint_step']) is int and manifest['metadata']['checkpoint_step']==index,
                'Checkpoint endpoint matches its declared inventory slot')
        require(all(manifest['metadata'][k]==v for k,v in expected_meta.items()),'Exact frozen continuation metadata')
        manifests[index]=manifest;del payload
    expected_updates=[c.artifact(folder/'updates'/f'{index:04d}.json') for index in range(1783,7129)]
    require(result['updates']==expected_updates and len(list((folder/'updates').glob('*.json')))==5346,'Every new update exactly once')
    groups=result['before_group_sha256'];clipped=0
    for index,descriptor in enumerate(expected_updates,start=1783):
        boundary();update=c.read(c.resolve(descriptor))
        require(type(update['step']) is int and update['step']==index,'Exact ordered absolute update index')
        number=(index-1)//STEPS+1;local=(index-1)%STEPS+1;indices=schedules[str(number)][local-1]
        examples=[loaded['train_examples'][i] for i in indices]
        require(update['repetition']==rep and update['pass_number']==number and update['local_step']==local and
                update['example_indices']==indices and update['schedule_sha256']==schedule_sha and
                update['selection_sha256']==selection_sha and update['batch_source_sha256']==digest(
                    [[x['game_index'],x['ply'],x['row_sha256']] for x in examples]),'Every update matches prospective source batch')
        require(update['batch_examples']==update['batch_denominator']==len(indices) and
                update['forward_backward_calls']==update['optimizer_calls']==1 and update['finite_gradients'] is True and
                math.isfinite(update['gradient_norm_before_clip']) and update['gradient_norm_before_clip']>=0 and
                update['clipped']==(update['gradient_norm_before_clip']>1.) and update['before_group_sha256']==groups and
                update['after_group_sha256']!=groups,'Finite complete continuous update chain')
        require(update['gradient_evidence']=='all_trainable_clipped_float32_gradient_bytes' and
                set(update['gradient_sha256'])==gradient_names,'All actual clipped gradient evidence')
        for value in update['gradient_sha256'].values():sampling._sha(value)
        expected=old.saved_batch_statistics(examples,update['outputs83'])
        require(all(math.isclose(update[k],v,rel_tol=1e-6,abs_tol=1e-5) for k,v in expected.items()),'Independent saved loss arithmetic')
        groups=update['after_group_sha256'];clipped+=int(update['clipped'])
        if index in manifests:require(groups==manifests[index]['metadata']['group_sha256'],'Checkpoint on exact update chain')
    require(groups==result['after_group_sha256'] and len(result['replays'])==3,'Complete final state and three replays')
    for number,item in zip(PASSES,result['replays'],strict=True):
        c.verify([item],boundary=boundary);replay=c.read(c.resolve(item));index=number*STEPS
        payload,manifest=checkpoint_payload(replay['actual_checkpoint'],boundary=boundary);del payload
        require(replay['status']=='exact_final_update_reproduced' and replay['pass_number']==number and
                replay['optimizer_calls']==1 and replay['from_checkpoint']==result['checkpoints'][str(index-1)] and
                replay['expected_checkpoint']==result['checkpoints'][str(index)] and
                replay['expected_update']==expected_updates[index-STEPS-1] and
                manifest['tree_sha256']==manifests[index]['tree_sha256']==replay['tree_sha256'],'Saved boundary replay binds exact expected update')
        compare_update(replay['actual_update'],c.read(c.resolve(replay['expected_update'])))
    return dict(status='duration_fit_saved_verified',result=result_item,repetition=rep,optimizer_calls=5346,
                replay_optimizer_calls=3,clipped_steps=clipped,checkpoints=result['checkpoints'],**IDENTITY)


# Same actual corrected loader/collator fixtures, with new output ownership only.
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
