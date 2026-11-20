"""Fixed four-fit comparison; unchanged numerical updates and explicit provenance.

Importing this module loads no tensor runtime, model or protected assessment.
The owner supplies the single corrected-source startup receipt to model workers.
"""
from __future__ import annotations
from copy import deepcopy
from pathlib import Path
import math
import os
import hashlib
import struct

from . import common as c
from tools import future_training_rules_20261021 as guard
from tools.splendorust_data_size_20261113 import fit as numeric
from tools.splendorust_full_corpus_20261110 import fit as old, sampling, targets
from tools.splendorust_policy_head_20261104 import data as accepted_data

accepted = old.accepted
IDENTITY = c.IDENTITY
STEPS, BATCH, ROOTS = 7101, 32, 227203
require, digest = c.require, c.digest
recorded_step = numeric.recorded_step
saved_batch_statistics = numeric.saved_batch_statistics
compare_update = numeric.compare_update
live_trees = numeric.live_trees
tensor_tree, rng_state, restore_rng = old.tensor_tree, old.rng_state, old.restore_rng
make_optimizer, verify_optimizer_payload = old.make_optimizer, old.verify_optimizer_payload
prediction_row, loss_components = old.prediction_row, old.loss_components
_STARTUP = _ENGINE = None


def startup(preverified=None):
    global _STARTUP
    if _STARTUP is None:
        value = guard.verify_runtime_sources() if preverified is None else deepcopy(preverified)
        require(value == guard.SOURCE_HASHES, 'Actual corrected runtime receipt')
        _STARTUP = value
    elif preverified is not None:
        require(preverified == _STARTUP, 'Exactly one unchanged corrected startup receipt')
    return deepcopy(_STARTUP)


def engine(folder, *, runtime_sources=None):
    global _ENGINE
    sources = startup(runtime_sources)
    if _ENGINE is None:
        import psutil
        process = psutil.Process()
        process.cpu_affinity([c.TRAIN_CPU]); process.nice(psutil.NORMAL_PRIORITY_CLASS)
        require(process.cpu_affinity() == [0], 'Qualified CPU0 numerical route')
        import torch
        torch.set_num_threads(1); torch.set_num_interop_threads(1)
        _ENGINE = dict(pid=process.pid, created=process.create_time(), cpu_affinity=[0],
            priority='NORMAL_PRIORITY_CLASS', torch_threads=1, torch_interop_threads=1,
            device='cpu', precision='float32')
    c.save(Path(folder)/'startup.json', dict(runtime_sources=sources, engine=_ENGINE, **IDENTITY))
    return deepcopy(_ENGINE)


def verify_startup(item, *, boundary=lambda: None):
    c.verify([item], boundary=boundary); receipt = c.read(c.resolve(item))
    accepted_data.verify_identity(receipt); info = receipt['engine']
    require(receipt['runtime_sources'] == guard.SOURCE_HASHES and info['cpu_affinity'] == [0]
        and info['torch_threads'] == info['torch_interop_threads'] == 1
        and info['device'] == 'cpu' and info['precision'] == 'float32'
        and info['priority'] == 'NORMAL_PRIORITY_CLASS', 'Exact accepted single-thread CPU startup')
    return receipt


def seed_rng(repetition):
    require(type(repetition) is int and repetition in (0, 1), 'Fixed optimization seed')
    return numeric.seed_rng(repetition)


def source_checkpoint(repetition, step=1782):
    require(type(repetition) is int and repetition in (0, 1) and step in (0, 1781, 1782),
        'Fixed historical checkpoint')
    value = c.read(c.ORIGINAL_OUT/'fits'/f'r{repetition}_mixed_joint'/'worker/fit_result.json')
    require(value['status'] == 'fit_complete' and value['repetition'] == repetition
        and value['breadth'] == 'mixed' and value['optimizer_calls'] == 1782, 'Accepted historical fit')
    item = value['checkpoints'][str(step)]; c.verify([item]); return item


def consumed_inputs():
    """Exact historical evidence actually consumed; no catalogue flattening."""
    paths=[c.PROPOSAL,c.BINDINGS,c.SOURCE_SELECTION,
        c.ORIGINAL_OUT/'offline/worker/evaluation_result.json',
        c.ROOT/'outputs/splendorust_training_duration_20261113_01/jobs/qualification/worker/qualification_result.json']
    items=[c.artifact(path) for path in paths]
    for repetition in (0,1):
        folder=c.ORIGINAL_OUT/'fits'/f'r{repetition}_mixed_joint'/'worker'
        items.extend(c.artifact(folder/name) for name in ('fit_result.json','updates/1782.json'))
        for step in (0,1781,1782):
            descriptor=source_checkpoint(repetition,step);manifest=c.read(c.resolve(descriptor))
            items.extend((descriptor,manifest['checkpoint'],manifest['tensor_manifest']))
    for model in ('r1_mixed_joint','parent'):
        items.append(c.artifact(c.ORIGINAL_OUT/'offline/worker/predictions'/model/'train/000000.json'))
    unique={}
    for item in items:require(unique.setdefault(item['path'],item)==item,'Consistent consumed historical input')
    return sorted(unique.values(),key=lambda x:x['path'])


def fit_spec(fit_id):
    require(fit_id in c.FIT_IDS, 'One of four fixed fits')
    proposal = c.read(c.PROPOSAL)
    rows = [x for x in proposal['schedules'] if x['fit_id'] == fit_id]
    require(len(rows) == 1, 'Unique prepared schedule')
    spec = rows[0]
    require(spec['seed'] == c.SEEDS[spec['repetition']] and spec['presentations'] == ROOTS
        and spec['updates'] == STEPS and spec['full_batches'] == 7100
        and spec['final_batch_size'] == 3 and spec['retained_checkpoint_steps'] == [0, 7100, 7101],
        'Exact prepared exposure')
    return spec


def load_order(fit_id, *, boundary=lambda: None):
    """Only metadata/order bytes are opened, never assessment or training targets."""
    import struct
    spec = fit_spec(fit_id); c.verify([spec['order']], boundary=boundary)
    raw = c.resolve(spec['order']).read_bytes()
    require(len(raw) == ROOTS*4, 'Exact little-endian schedule length')
    order = [x[0] for x in struct.iter_unpack('<I', raw)]
    limit = ROOTS if spec['arm'] == 'larger' else 57022
    require(all(0 <= i < limit for i in order), 'Training-only canonical indices')
    require(len(set(order)) == limit, 'Exact distinct training records')
    if spec['arm'] == 'larger': require(sorted(order) == list(range(ROOTS)), 'One complete pass')
    else:
        from collections import Counter
        require(Counter(Counter(order).values()) == {3: 885, 4: 56137}, 'Three complete plus truncated fourth pass')
    return spec, order


def batch_indices(order, step):
    require(type(step) is int and 1 <= step <= STEPS, 'Fixed update index')
    require(len(order) == ROOTS, 'Full prepared order')
    values = [int(x) for x in order[(step-1)*BATCH:step*BATCH]]
    require(len(values) == (3 if step == STEPS else BATCH), 'Actual full32 or final3 batch')
    return values


def batch_source(examples):
    return digest([[x['source'], x['game_index'], x['ply'], x['row_sha256']] for x in examples])


def metadata(fit_id, cache, target_digest, *, qualification_case=None, qualification_batch_size=None):
    spec = fit_spec(fit_id)
    result = dict(study_id=c.STUDY, fit_id=fit_id, arm=spec['arm'], repetition=spec['repetition'],
        seed=spec['seed'], breadth='mixed', scope='joint', qualification_only=qualification_case is not None,
        qualification_case=qualification_case, qualification_batch_size=qualification_batch_size,
        historical_data_relabelled=False, historical_warm_start_sha256=accepted.PARENT_SHA256,
        parameter_count=accepted.PARAMETERS, model_source_sha256=accepted.MODEL_SOURCE_SHA256,
        training_initializer='original_EF8A_and_fresh_AdamW_same_seed',
        source_training_games=4096 if spec['arm'] == 'larger' else 1024,
        unique_training_records=ROOTS if spec['arm'] == 'larger' else 57022,
        training_record_presentations=ROOTS, fixed_final_step=STEPS, batch_size=BATCH, last_batch_size=3,
        retained_checkpoint_steps=[0, 7100, 7101], schedule=spec['order'],
        training_inventory_sha256=c.read(c.BINDINGS)['canonical_training_inventory']['sha256'],
        corpus_cache=cache, value_target_sources_sha256=target_digest,
        optimizer_recipe={k:v for k,v in accepted.RECIPE.items()
            if k not in ('epochs','shuffle','example_weight','batch_reduction')},
        batch_reduction='mean_actual_batch_size_equal_record_weights', value_target_contract=targets.contract('mixed'),
        model_cpu=0, checkpoint_selection=False, best_seed_selection=False, **IDENTITY)
    return result


def validate_checkpoint_metadata(meta):
    accepted_data.verify_identity(meta); spec = fit_spec(meta.get('fit_id'))
    for key, value in dict(study_id=c.STUDY, arm=spec['arm'], repetition=spec['repetition'], seed=spec['seed'],
        breadth='mixed', scope='joint', parameter_count=accepted.PARAMETERS,
        model_source_sha256=accepted.MODEL_SOURCE_SHA256, historical_warm_start_sha256=accepted.PARENT_SHA256,
        training_initializer='original_EF8A_and_fresh_AdamW_same_seed', source_training_games=4096 if spec['arm']=='larger' else 1024,
        unique_training_records=ROOTS if spec['arm']=='larger' else 57022, training_record_presentations=ROOTS,
        fixed_final_step=STEPS, batch_size=BATCH, last_batch_size=3, model_cpu=0,
        batch_reduction='mean_actual_batch_size_equal_record_weights', value_target_contract=targets.contract('mixed'),
        retained_checkpoint_steps=[0,7100,7101], schedule=spec['order'],
        optimizer_recipe={k:v for k,v in accepted.RECIPE.items() if k not in ('epochs','shuffle','example_weight','batch_reduction')}).items():
        require(meta.get(key) == value, 'Fixed checkpoint '+key)
        if type(value) is int:
            require(type(meta.get(key)) is int, 'Strict checkpoint integer '+key)
    for key in ('qualification_only','historical_data_relabelled','checkpoint_selection','best_seed_selection'):
        require(type(meta.get(key)) is bool, 'Strict checkpoint boolean '+key)
    require(not any(meta[k] for k in ('historical_data_relabelled','checkpoint_selection','best_seed_selection')),
        'No relabelling or endpoint selection')
    step = meta.get('checkpoint_step'); require(type(step) is int, 'Strict completed step')
    if meta['qualification_only']:
        require(meta['qualification_case'] in ('full32','final3','cross1','cross2','cross3')
            and step in (0,1), 'Disposable qualification checkpoint')
        expected_size = 3 if meta['qualification_case'] == 'final3' else 32
        require(meta['qualification_batch_size'] == expected_size, 'Exact qualification batch size')
        presentations = step*expected_size
    else:
        require(step in (0,7100,7101) and meta['qualification_case'] is None
            and meta['qualification_batch_size'] is None, 'Only fixed scientific checkpoints')
        presentations = min(step*32, ROOTS)
    require(type(meta.get('record_presentations')) is int and meta['record_presentations'] == presentations,
        'Exact cumulative presentations')
    require(meta['training_inventory_sha256'] == c.read(c.BINDINGS)['canonical_training_inventory']['sha256'],
        'Exact namespaced training inventory')
    sampling._sha(meta['value_target_sources_sha256']); c.verify([meta['corpus_cache'], meta['schedule']])
    return meta


def save_checkpoint(model, optimizer, step, meta, folder, *, boundary=lambda: None):
    import torch
    boundary(); accepted.validate_architecture(model)
    presentations = step*meta['qualification_batch_size'] if meta['qualification_only'] else min(step*32,ROOTS)
    actual = dict(meta, checkpoint_step=step, record_presentations=presentations,
        weights_sha256=accepted.state_digest(model), group_sha256=accepted.group_digests(model))
    validate_checkpoint_metadata(actual)
    payload = dict(schema='training4096_checkpoint_v1', kind='entity', metadata=actual,
        state_dict=model.state_dict(), optimizer=optimizer.state_dict(), rng=rng_state())
    verify_optimizer_payload(payload)
    folder=Path(folder); require(folder.resolve().is_relative_to(c.OUT), 'New checkpoints only')
    folder.mkdir(parents=True,exist_ok=True); path=folder/'checkpoint.pt'
    with path.open('xb') as stream:
        torch.save(payload,stream); stream.flush(); os.fsync(stream.fileno())
    trees={k:tensor_tree(payload[k]) for k in ('state_dict','optimizer','rng')}
    tensor_manifest=c.save(folder/'tensor_manifest.json',trees)
    result=c.save(folder/'checkpoint_manifest.json',dict(schema='training4096_checkpoint_manifest_v1',
        checkpoint=c.artifact(path),tensor_manifest=tensor_manifest,metadata=actual,
        tree_sha256={k:digest(v) for k,v in trees.items()}))
    boundary(); return result


def checkpoint_payload(item, *, boundary=lambda: None):
    import torch
    c.verify([item],boundary=boundary); manifest=c.read(c.resolve(item))
    require(manifest['schema']=='training4096_checkpoint_manifest_v1','Separate checkpoint schema')
    validate_checkpoint_metadata(manifest['metadata']); descriptor=manifest['checkpoint']
    require(0 < descriptor['bytes'] < 256*1024**2,'Bounded checkpoint archive')
    c.verify([descriptor,manifest['tensor_manifest']],boundary=boundary)
    path=c.resolve(descriptor); before=path.stat()
    with path.open('rb') as stream: payload=torch.load(stream,map_location='cpu',weights_only=True)
    after=path.stat(); require((before.st_size,before.st_mtime_ns)==(after.st_size,after.st_mtime_ns),'Stable checkpoint bytes')
    c.verify([descriptor],boundary=boundary)
    require(payload['schema']=='training4096_checkpoint_v1' and payload['kind']=='entity'
        and payload['metadata']==manifest['metadata'],'Exact payload metadata')
    hashes=old.state_mapping_hashes(payload['state_dict'])
    require(hashes.pop('all')==manifest['metadata']['weights_sha256']
        and hashes==manifest['metadata']['group_sha256'],'All saved model tensor bits')
    verify_optimizer_payload(payload)
    trees={k:tensor_tree(payload[k]) for k in ('state_dict','optimizer','rng')}
    require(trees==c.read(c.resolve(manifest['tensor_manifest']))
        and {k:digest(v) for k,v in trees.items()}==manifest['tree_sha256'],'All AdamW and RNG bits')
    return payload,manifest


def _decode_tensor_tree(node):
    """Parse exact JSON evidence without importing a tensor runtime."""
    require(type(node) is dict,'Strict tensor evidence node')
    if node.get('tensor') is True:
        require(set(node)=={'tensor','dtype','shape','bytes','sha256'}
            and node['dtype'] in ('torch.float32','torch.uint8','torch.int64')
            and type(node['shape']) is list and all(type(x) is int and x>=0 for x in node['shape']),
            'Known finite-size tensor evidence')
        width={'torch.float32':4,'torch.uint8':1,'torch.int64':8}[node['dtype']]
        require(type(node['bytes']) is int and node['bytes']==math.prod(node['shape'])*width,'Exact tensor byte count')
        sampling._sha(node['sha256']);return node
    if set(node)=={'dict'}:
        require(type(node['dict']) is list,'Ordered dictionary evidence')
        result={}
        for pair in node['dict']:
            require(type(pair) is list and len(pair)==2,'Exact dictionary pair')
            key=_decode_tensor_tree(pair[0]);require(type(key) in (str,int) and key not in result,'Unique typed dictionary key')
            result[key]=_decode_tensor_tree(pair[1])
        return result
    for kind in ('list','tuple'):
        if set(node)=={kind}:
            require(type(node[kind]) is list,'Sequence evidence')
            values=[_decode_tensor_tree(x) for x in node[kind]]
            return values if kind=='list' else tuple(values)
    if set(node)=={'float64'}:
        require(type(node['float64']) is str and len(node['float64'])==16,'Exact double evidence')
        value=struct.unpack('<d',bytes.fromhex(node['float64']))[0]
        require(math.isfinite(value),'Finite scalar evidence');return value
    require(set(node)=={'type','value'},'Strict scalar evidence')
    types={'str':str,'int':int,'bool':bool,'NoneType':type(None)}
    require(node['type'] in types and type(node['value']) is types[node['type']],'Scalar evidence type')
    return node['value']


def _validate_saved_tensor_inventory(trees, meta, reference_trees):
    require(set(trees)=={'state_dict','optimizer','rng'},'All saved state components')
    decoded={k:_decode_tensor_tree(v) for k,v in trees.items()}
    reference=_decode_tensor_tree(reference_trees['state_dict'])
    state=decoded['state_dict'];require(list(state)==list(reference),'Exact ordered model parameter inventory')
    for name,value in state.items():
        require(type(value) is dict and value.get('tensor') is True
            and value['dtype']=='torch.float32'
            and {k:value[k] for k in ('shape','dtype','bytes')}=={k:reference[name][k] for k in ('shape','dtype','bytes')},
            'Exact architecture tensor layout '+name)
    require(sum(v['bytes']//4 for v in state.values())==accepted.PARAMETERS,'Actual saved parameter count')
    optimizer=decoded['optimizer'];require(set(optimizer)=={'state','param_groups'}
        and len(optimizer['param_groups'])==1,'Exactly one saved optimizer group')
    group=optimizer['param_groups'][0]
    expected=dict(lr=1e-5,betas=(.9,.999),eps=1e-8,weight_decay=1e-4,amsgrad=False,foreach=False,
        fused=False,maximize=False,capturable=False,differentiable=False,params=list(range(len(state))))
    require(group==expected,'Exact saved AdamW options and named parameter order')
    step=meta['checkpoint_step'];slots=optimizer['state']
    require(set(slots)==(set(range(len(state))) if step else set()),'Complete per-parameter AdamW state')
    for i,name in enumerate(state):
        if not step:break
        value=slots[i];require(set(value)=={'step','exp_avg','exp_avg_sq'},'Exact AdamW slots')
        expected_step=dict(tensor=True,dtype='torch.float32',shape=[],bytes=4,
            sha256=hashlib.sha256(struct.pack('<f',float(step))).hexdigest())
        require(value['step']==expected_step,'Exact saved optimizer step bytes')
        for key in ('exp_avg','exp_avg_sq'):
            require({k:value[key][k] for k in ('shape','dtype','bytes')}==
                {k:state[name][k] for k in ('shape','dtype','bytes')},'Moment layout matches actual parameter')
    rng=decoded['rng'];ref_rng=_decode_tensor_tree(reference_trees['rng'])
    require(set(rng)=={'torch','python','numpy'}
        and {k:rng['torch'][k] for k in ('shape','dtype','bytes')}==
            {k:ref_rng['torch'][k] for k in ('shape','dtype','bytes')},'Complete saved CPU RNG inventory')
    require(type(rng['python']) is tuple and len(rng['python'])==3 and rng['python'][0]==3
        and type(rng['python'][1]) is tuple and len(rng['python'][1])==625
        and all(type(x) is int and 0<=x<2**32 for x in rng['python'][1][:-1])
        and type(rng['python'][1][-1]) is int and 0<=rng['python'][1][-1]<=624
        and (rng['python'][2] is None or type(rng['python'][2]) is float),'Complete Python RNG state')
    nr=rng['numpy'];require(type(nr) is list and len(nr)==5 and nr[0]=='MT19937'
        and type(nr[1]) is list and len(nr[1])==624
        and all(type(x) is int and 0<=x<2**32 for x in nr[1]) and type(nr[2]) is int and 0<=nr[2]<=624
        and type(nr[3]) is int and nr[3] in (0,1) and type(nr[4]) is float,'Complete NumPy RNG state')
    return decoded


def checkpoint_manifest(item, *, boundary=lambda: None):
    """Saved-only audit: authenticate archive bytes and strict JSON state evidence."""
    c.verify([item],boundary=boundary);manifest=c.read(c.resolve(item))
    require(manifest['schema']=='training4096_checkpoint_manifest_v1','Separate saved checkpoint schema')
    meta=validate_checkpoint_metadata(manifest['metadata'])
    require(0<manifest['checkpoint']['bytes']<256*1024**2,'Bounded immutable checkpoint archive')
    c.verify([manifest['checkpoint'],manifest['tensor_manifest']],boundary=boundary)
    trees=c.read(c.resolve(manifest['tensor_manifest']))
    require({k:digest(v) for k,v in trees.items()}==manifest['tree_sha256'],'Exact JSON tensor/RNG tree hashes')
    reference=c.read(c.resolve(source_checkpoint(meta['repetition'],0)))
    c.verify([reference['tensor_manifest']],boundary=boundary)
    decoded=_validate_saved_tensor_inventory(trees,meta,c.read(c.resolve(reference['tensor_manifest'])))
    for value in (meta['weights_sha256'],*meta['group_sha256'].values()):sampling._sha(value)
    require(set(meta['group_sha256'])=={'shared','policy','value'},'Exact parameter-group hashes')
    return decoded,manifest


def qualify_checkpoint_payloads(items, folder, *, boundary=lambda: None):
    """Actual model worker independently parses every archive before clean exit."""
    rows=[]
    for item in items:
        payload,manifest=checkpoint_payload(item,boundary=boundary);del payload
        rows.append(dict(manifest=item,checkpoint=manifest['checkpoint'],tensor_manifest=manifest['tensor_manifest'],
            tree_sha256=manifest['tree_sha256'],actual_payload_deserialized=True))
    return c.save(Path(folder)/'checkpoint_payload_qualification.json',dict(status='all_checkpoint_payloads_verified',
        checkpoints=rows,optimizer_calls=0,model_forward_calls=0))


def verify_checkpoint_payload_qualification(item, expected, *, boundary=lambda: None):
    c.verify([item],boundary=boundary);proof=c.read(c.resolve(item))
    require(proof['status']=='all_checkpoint_payloads_verified' and proof['optimizer_calls']==0
        and proof['model_forward_calls']==0 and [r['manifest'] for r in proof['checkpoints']]==expected,
        'Actual worker qualified every retained archive, with no extra optimizer call')
    for row in proof['checkpoints']:
        _,manifest=checkpoint_manifest(row['manifest'],boundary=boundary)
        require(row['actual_payload_deserialized'] is True
            and all(row[k]==manifest[k] for k in ('checkpoint','tensor_manifest','tree_sha256')),
            'Archive payload qualification bound to exact saved evidence')
    return proof


def load_checkpoint(item, *, optimizer=False, boundary=lambda: None):
    payload,manifest=checkpoint_payload(item,boundary=boundary)
    model=accepted.load_parent(); model.load_state_dict(payload['state_dict'],strict=True); model.eval()
    accepted.validate_architecture(model)
    require(accepted.state_digest(model)==manifest['metadata']['weights_sha256'],'Exact restored network')
    opt=make_optimizer(model,'joint'); opt.load_state_dict(payload['optimizer'])
    require(tensor_tree(opt.state_dict())==tensor_tree(payload['optimizer']),'Exact restored AdamW')
    restore_rng(payload['rng']); require(tensor_tree(rng_state())==tensor_tree(payload['rng']),'Exact restored RNG')
    return model,opt if optimizer else None,manifest


def source_equivalence():
    require(recorded_step is numeric.recorded_step and saved_batch_statistics is numeric.saved_batch_statistics
        and make_optimizer is old.make_optimizer and loss_components is old.loss_components
        and rng_state is old.rng_state and restore_rng is old.restore_rng,
        'Direct unchanged accepted optimizer/loss/RNG/gradient evidence')
    return dict(status='unchanged_accepted_numeric_functions',accepted_equivalence=numeric.source_equivalence(),
        original_sources=[c.artifact(numeric.__file__),c.artifact(old.__file__)],models=0,optimizer_calls=0)


def require_assessment_unopened():
    require(not (c.OUT/'assessment_opened.json').exists(),
        'No fit, qualification update or checkpoint replacement after assessment opens')


def _training_boundary(callback):
    def check():
        require_assessment_unopened(); callback()
    return check


def run_fit(fit_id, folder, *, boundary=lambda: None, runtime_sources=None):
    require_assessment_unopened()
    boundary=_training_boundary(boundary)
    from . import corpus
    folder=Path(folder); engine(folder,runtime_sources=runtime_sources)
    spec,order=load_order(fit_id,boundary=boundary); dataset=corpus.training_dataset(boundary=boundary)
    require(len(dataset)==ROOTS,'Complete accepted training-only dataset')
    meta=metadata(fit_id,dataset.cache_acceptance,dataset.value_target_sources_sha256)
    boundary();seed_rng(spec['repetition']); model=accepted.load_parent(); opt=make_optimizer(model,'joint')
    initial=accepted.group_digests(model)
    checkpoints={'0':save_checkpoint(model,opt,0,meta,folder/'checkpoints/0',boundary=boundary)}
    zero=c.read(c.resolve(checkpoints['0'])); oldzero=c.read(c.resolve(source_checkpoint(spec['repetition'],0)))
    require(zero['tree_sha256']==oldzero['tree_sha256'],'Pristine EF8A, fresh AdamW and original seeded RNG')
    updates=[]
    for step in range(1,STEPS+1):
        require_assessment_unopened();boundary(); indices=batch_indices(order,step); examples=[dataset[i] for i in indices]
        update=recorded_step(model,opt,dataset.collate(indices),'joint',boundary=boundary)
        update.update(fit_id=fit_id,step=step,repetition=spec['repetition'],example_indices=indices,
            schedule=spec['order'],batch_source_sha256=batch_source(examples))
        updates.append(c.save(folder/'updates'/f'{step:04d}.json',update))
        if step in (7100,7101):
            checkpoints[str(step)]=save_checkpoint(model,opt,step,meta,folder/'checkpoints'/str(step),boundary=boundary)
        if step%32==0 or step==STEPS:
            c.status(folder/'progress.json',dict(fit_id=fit_id,completed_updates=step,total_updates=STEPS,time=c.now()))
    final=accepted.group_digests(model); del model,opt
    require_assessment_unopened()
    model,opt,_=load_checkpoint(checkpoints['7100'],optimizer=True,boundary=boundary)
    replay=recorded_step(model,opt,dataset.collate(batch_indices(order,STEPS)),'joint',boundary=boundary)
    actual_item=save_checkpoint(model,opt,STEPS,meta,folder/'replay/checkpoint',boundary=boundary)
    c.save(folder/'replay/actual_update.json',replay)
    expected=c.read(c.resolve(checkpoints['7101'])); actual=c.read(c.resolve(actual_item))
    require(actual['tree_sha256']==expected['tree_sha256'],'Exact final3 model/AdamW/RNG reproduction')
    compare_update(replay,c.read(c.resolve(updates[-1])))
    replay_item=c.save(folder/'replay/result.json',dict(status='exact_final_update_reproduced',step=STEPS,
        from_checkpoint=checkpoints['7100'],expected_checkpoint=checkpoints['7101'],actual_checkpoint=actual_item,
        expected_update=updates[-1],actual_update=replay,tree_sha256=expected['tree_sha256'],optimizer_calls=1))
    del model,opt
    payload_qualification=qualify_checkpoint_payloads([checkpoints[str(i)] for i in (0,7100,7101)]+[actual_item],folder,boundary=boundary)
    result=dict(schema='training4096_fit_v1',status='fit_complete',study_id=c.STUDY,fit_id=fit_id,
        repetition=spec['repetition'],arm=spec['arm'],startup=c.artifact(folder/'startup.json'),schedule=spec['order'],
        checkpoints=checkpoints,endpoint=checkpoints['7101'],updates=updates,replay=replay_item,
        checkpoint_payload_qualification=payload_qualification,
        optimizer_calls=STEPS,replay_optimizer_calls=1,record_presentations=ROOTS,
        before_group_sha256=initial,after_group_sha256=final,checkpoint_selection=False,
        optimizer_restart=False,epoch_rng_reseed=False,**IDENTITY)
    c.save(folder/'fit_result.json',result); boundary(); return result


def verify_fit(folder, *, boundary=lambda: None):
    from . import corpus
    folder=Path(folder); item=c.artifact(folder/'fit_result.json'); result=c.read(c.resolve(item))
    require(result['startup']==c.artifact(folder/'startup.json'),'Exact actual startup receipt')
    verify_startup(result['startup'],boundary=boundary)
    spec,order=load_order(result['fit_id'],boundary=boundary); dataset=corpus.training_dataset(boundary=boundary)
    require(result['schema']=='training4096_fit_v1' and result['status']=='fit_complete'
        and result['study_id']==c.STUDY and result['arm']==spec['arm'] and result['repetition']==spec['repetition']
        and result['schedule']==spec['order'] and result['optimizer_calls']==STEPS
        and result['replay_optimizer_calls']==1 and result['record_presentations']==ROOTS
        and result['checkpoint_selection'] is result['optimizer_restart'] is result['epoch_rng_reseed'] is False,
        'Complete fixed fit and no selection/restart')
    require(set(result['checkpoints'])=={'0','7100','7101'} and result['endpoint']==result['checkpoints']['7101'],
        'Only prospective endpoints')
    expected_meta=metadata(result['fit_id'],dataset.cache_acceptance,dataset.value_target_sources_sha256)
    manifests={}; gradient_names=None
    for step in (0,7100,7101):
        payload,manifest=checkpoint_manifest(result['checkpoints'][str(step)],boundary=boundary)
        require(manifest['metadata']['checkpoint_step']==step
            and all(manifest['metadata'][k]==v for k,v in expected_meta.items()),'Exact source/recipe checkpoint')
        manifests[step]=manifest; gradient_names=set(payload['state_dict']); del payload
    require(manifests[0]['tree_sha256']==c.read(c.resolve(source_checkpoint(spec['repetition'],0)))['tree_sha256'],
        'Same pristine model/fresh optimizer/seeded RNG as accepted reference')
    require(manifests[0]['metadata']['group_sha256']==result['before_group_sha256'],'Actual initial model state')
    expected=[c.artifact(folder/'updates'/f'{i:04d}.json') for i in range(1,STEPS+1)]
    require(result['updates']==expected and len(list((folder/'updates').glob('*.json')))==STEPS,
        'Every scientific update exactly once')
    groups=result['before_group_sha256']; clipped=0
    for step,descriptor in enumerate(expected,1):
        boundary(); update=c.read(c.resolve(descriptor)); indices=batch_indices(order,step)
        examples=[dataset[i] for i in indices]
        require(update['fit_id']==result['fit_id'] and type(update['step']) is int and update['step']==step
            and update['repetition']==spec['repetition'] and update['schedule']==spec['order']
            and update['example_indices']==indices and update['batch_source_sha256']==batch_source(examples),
            'Exact ordered namespaced source batch')
        require(update['batch_examples']==update['batch_denominator']==len(indices)
            and update['forward_backward_calls']==update['optimizer_calls']==1 and update['finite_gradients'] is True
            and math.isfinite(update['gradient_norm_before_clip']) and update['gradient_norm_before_clip']>=0
            and update['clipped']==(update['gradient_norm_before_clip']>1.)
            and update['before_group_sha256']==groups and update['after_group_sha256']!=groups,
            'Finite continuous optimizer chain')
        require(update['gradient_evidence']=='all_trainable_clipped_float32_gradient_bytes'
            and set(update['gradient_sha256'])==gradient_names,'All actual clipped gradient bytes')
        for value in update['gradient_sha256'].values(): sampling._sha(value)
        arithmetic=saved_batch_statistics(examples,update['outputs83'])
        require(all(math.isclose(update[k],v,rel_tol=1e-6,abs_tol=1e-5) for k,v in arithmetic.items()),
            'Independent actual-batch loss arithmetic')
        groups=update['after_group_sha256']; clipped+=int(update['clipped'])
        if step in manifests: require(groups==manifests[step]['metadata']['group_sha256'],'Checkpoint/update chain')
    require(groups==result['after_group_sha256'],'Final model/update chain')
    c.verify([result['replay']],boundary=boundary); replay=c.read(c.resolve(result['replay']))
    payload,actual=checkpoint_manifest(replay['actual_checkpoint'],boundary=boundary); del payload
    require(replay['status']=='exact_final_update_reproduced' and replay['step']==STEPS
        and replay['optimizer_calls']==1 and replay['from_checkpoint']==result['checkpoints']['7100']
        and replay['expected_checkpoint']==result['endpoint'] and replay['expected_update']==expected[-1]
        and replay['tree_sha256']==actual['tree_sha256']==manifests[7101]['tree_sha256'],
        'Exactly one full-state final3 replay')
    compare_update(replay['actual_update'],c.read(c.resolve(expected[-1])))
    require(result['checkpoint_payload_qualification']==c.artifact(folder/'checkpoint_payload_qualification.json'),
        'Exact actual fit payload qualification receipt')
    verify_checkpoint_payload_qualification(result['checkpoint_payload_qualification'],
        [result['checkpoints'][str(i)] for i in (0,7100,7101)]+[replay['actual_checkpoint']],boundary=boundary)
    return dict(status='training4096_fit_saved_verified',result=item,fit_id=result['fit_id'],
        endpoint=result['endpoint'],checkpoints=result['checkpoints'],optimizer_calls=STEPS,
        replay_optimizer_calls=1,record_presentations=ROOTS,clipped_steps=clipped,**IDENTITY)


QUALIFICATION_BATCHES = (('full32',1),('final3',7101),('cross1',1782),('cross2',3564),('cross3',5346))


def _saved_prediction(model_id):
    result=c.read(c.ORIGINAL_OUT/'offline/worker/evaluation_result.json')
    require(result['status']=='all_fixed_predictions_complete','Accepted historical evaluator')
    model=next(x for x in result['models'] if x['model_id']==model_id)
    path=c.ORIGINAL_OUT/'offline/worker/predictions'/model_id/'train/000000.json'
    item=c.artifact(path); require(item in model['batches'],'Authenticated old training prediction batch')
    return item,c.read(path)


def run_qualification(folder, *, boundary=lambda: None, runtime_sources=None):
    """Twelve discarded updates, all from engineering or old training evidence.

    This runs before the larger training-cache build. Every numerical scheduled
    batch is from the control, which can address only the old canonical57022.
    """
    require_assessment_unopened()
    boundary=_training_boundary(boundary)
    from . import corpus, offline
    folder=Path(folder); engine(folder,runtime_sources=runtime_sources)
    equivalence=source_equivalence()
    inputs=corpus.qualify_training_inputs(folder/'inputs',boundary=boundary)
    historical_selection=c.read(c.SOURCE_SELECTION)
    historical_panel=sampling.load_diagnostic_train(historical_selection,boundary=boundary)
    duration_path=c.ROOT/'outputs/splendorust_training_duration_20261113_01/jobs/qualification/worker/qualification_result.json'
    duration_item=c.artifact(duration_path); duration=c.read(duration_path)
    require(duration['status']=='qualification_passed' and duration['study_id']=='splendorust_training_duration_20261113_v1',
        'Accepted historical corrected evaluator fixture proof')
    references=[];evaluations=[]
    for repetition in (0,1):
        require_assessment_unopened()
        loaded=old.validate_loaded(sampling.load_selected(historical_selection,repetition,'mixed',boundary=boundary))
        _,expected=old.checkpoint_payload(source_checkpoint(repetition),boundary=boundary)
        model,opt,_=old.load_checkpoint(source_checkpoint(repetition,1781),optimizer=True,boundary=boundary)
        actual=recorded_step(model,opt,old.selected_batch(loaded,loaded['batches'][-1]),'joint',boundary=boundary)
        trees=live_trees(model,opt)
        c.save(folder/'observations'/f'old_r{repetition}_update.json',dict(actual_update=actual,actual_tree_sha256=trees))
        source_update=c.artifact(c.ORIGINAL_OUT/'fits'/f'r{repetition}_mixed_joint'/'worker/updates/1782.json')
        compare_update(actual,c.read(c.resolve(source_update)))
        require(trees==expected['tree_sha256'],'Actual original final30 network/AdamW/RNG exact reproduction')
        references.append(dict(repetition=repetition,source_before=source_checkpoint(repetition,1781),
            source_after=source_checkpoint(repetition),source_update=source_update,actual_update=actual,
            actual_tree_sha256=trees,optimizer_calls=1,discarded=True))
        del model,opt
        model,opt,_=old.load_checkpoint(source_checkpoint(repetition),optimizer=True,boundary=boundary)
        before_rng=digest(tensor_tree(rng_state()))
        if repetition==0:
            structural=offline.predict_outputs(model,inputs['qualification_structural_examples'])
            genuine=[targets.for_arm(x,'terminal') for x in inputs['qualification_examples']]
            # These excluded engineering examples have the historical schema;
            # the r1 comparison below exercises the new namespaced production schema.
            for example in genuine: example.pop('source',None)
            predicted=offline.predict_batch(model,genuine)
            saved=duration['evaluator_qualification']
            c.save(folder/'observations/structural.json',dict(outputs83=structural))
            c.save(folder/'observations/genuine.json',dict(rows=predicted))
            require(accepted.full_output_bytes(structural)==accepted.full_output_bytes(saved['structural_outputs83'])
                and predicted==saved['prediction_rows'],'Actual new evaluator exact structural and genuine saved outputs')
            evaluations.append(dict(model_id='historical_r0',saved=duration_item,forward_calls=2,
                structural_outputs83=structural,prediction_rows=predicted,structural_outcomes_fabricated=False))
        else:
            item,packet=_saved_prediction('r1_mixed_joint')
            examples=[dict(x,source='prior_20261107') for x in historical_panel[:32]]
            predicted=offline.predict_batch(model,examples)
            c.save(folder/'observations/historical_r1.json',dict(rows=predicted))
            expected_rows=[dict(row,source='prior_20261107',root_q_win_credit=x['root_q_win_credit'])
                for row,x in zip(packet['rows'],examples,strict=True)]
            require(predicted==expected_rows,'Production namespaced evaluator exact historical r1 training panel')
            evaluations.append(dict(model_id='historical_r1',saved=item,forward_calls=1,prediction_rows=predicted,
                production_namespace_exercised=True,search_win_credit=[x['root_q_win_credit'] for x in examples]))
        require(before_rng==digest(tensor_tree(rng_state())),'Evaluation does not consume training RNG')
        del model,opt
    model=accepted.load_parent();model.eval();item,packet=_saved_prediction('parent')
    predicted=offline.predict_batch(model,historical_panel[:32])
    c.save(folder/'observations/parent.json',dict(rows=predicted))
    require(predicted==packet['rows'],'New evaluator exact EF8A training panel')
    evaluations.append(dict(model_id='parent',saved=item,forward_calls=1,prediction_rows=predicted));del model
    dataset=corpus.prior_training_dataset(boundary=boundary)
    require(len(dataset)==57022,'Only accepted prior training rows for scheduled numerical qualification')
    _,order=load_order('control_r0',boundary=boundary); pairs=[]
    for name,scheduled_step in QUALIFICATION_BATCHES:
        require_assessment_unopened()
        indices=batch_indices(order,scheduled_step);examples=[dataset[i] for i in indices]
        require(all(x['source']=='prior_20261107' and x['split']=='train' for x in examples),
            'Qualification never reads fresh assessment or future larger-cache rows')
        boundary();seed_rng(0);model=accepted.load_parent();opt=make_optimizer(model,'joint')
        meta=metadata('control_r0',dataset.cache_acceptance,dataset.value_target_sources_sha256,
            qualification_case=name,qualification_batch_size=len(indices))
        before=save_checkpoint(model,opt,0,meta,folder/'batches'/name/'before',boundary=boundary)
        direct=recorded_step(model,opt,dataset.collate(indices),'joint',boundary=boundary)
        c.save(folder/'observations'/f'{name}_direct.json',direct)
        after=save_checkpoint(model,opt,1,meta,folder/'batches'/name/'after',boundary=boundary)
        trees=live_trees(model,opt);del model,opt
        model,opt,_=load_checkpoint(before,optimizer=True,boundary=boundary)
        repeated=recorded_step(model,opt,dataset.collate(indices),'joint',boundary=boundary)
        repeated_trees=live_trees(model,opt)
        c.save(folder/'observations'/f'{name}_repeated.json',dict(actual_update=repeated,actual_tree_sha256=repeated_trees))
        compare_update(repeated,direct);require(repeated_trees==trees,'Direct versus reloaded scheduled batch exact state')
        pairs.append(dict(case=name,scheduled_step=scheduled_step,batch_size=len(indices),example_indices=indices,
            batch_source_sha256=batch_source(examples),before=before,after=after,direct_update=direct,
            repeated_update=repeated,after_tree_sha256=trees,optimizer_calls=2,discarded=True))
        del model,opt
    payload_qualification=qualify_checkpoint_payloads([item for row in pairs for item in (row['before'],row['after'])],
        folder,boundary=boundary)
    result=dict(schema='training4096_qualification_v1',status='qualification_passed',study_id=c.STUDY,
        startup=c.artifact(folder/'startup.json'),numerical_equivalence=equivalence,data_qualification=inputs['proof'],
        old_exact_updates=references,evaluator_qualification=evaluations,scheduled_batch_replays=pairs,
        checkpoint_payload_qualification=payload_qualification,
        optimizer_calls=12,evaluator_forward_calls=4,scientific_optimizer_calls=0,
        fresh_assessment_reads=0,all_qualification_updates_discarded=True,**IDENTITY)
    c.save(folder/'qualification_result.json',result);boundary();return result


def verify_qualification(folder, *, boundary=lambda: None):
    from . import corpus
    folder=Path(folder);item=c.artifact(folder/'qualification_result.json');result=c.read(c.resolve(item))
    require(result['startup']==c.artifact(folder/'startup.json'),'Exact qualification startup binding')
    verify_startup(result['startup'],boundary=boundary)
    require(result['schema']=='training4096_qualification_v1' and result['study_id']==c.STUDY
        and result['status']=='qualification_passed' and result['optimizer_calls']==12
        and result['evaluator_forward_calls']==4 and result['scientific_optimizer_calls']==0
        and result['fresh_assessment_reads']==0 and result['all_qualification_updates_discarded'] is True,
        'Complete twelve-update qualification, not scientific fitting')
    require(result['numerical_equivalence']==source_equivalence(),'Exact unchanged numerical source binding')
    proof=result['data_qualification']
    require(proof['actual_selected_loader_and_collator_exercised'] is True
        and proof['actual_mmap_training_path_qualified'] is True,'Actual production decoder/cache/collator exercised')
    c.verify(proof['cache_feature_artifacts'],boundary=boundary)
    require([r['repetition'] for r in result['old_exact_updates']]==[0,1],'Both historical final30 updates')
    for row in result['old_exact_updates']:
        c.verify([row[k] for k in ('source_before','source_after','source_update')],boundary=boundary)
        require(row['source_before']==source_checkpoint(row['repetition'],1781)
            and row['source_after']==source_checkpoint(row['repetition']) and row['optimizer_calls']==1
            and row['discarded'] is True,'Exact discarded original-seed replay')
        compare_update(row['actual_update'],c.read(c.resolve(row['source_update'])))
        require(row['actual_tree_sha256']==c.read(c.resolve(row['source_after']))['tree_sha256'],
            'Saved actual complete original network/AdamW/RNG parity')
    require([r['model_id'] for r in result['evaluator_qualification']]==['historical_r0','historical_r1','parent'],
        'All fixed evaluator qualification comparisons')
    for row in result['evaluator_qualification']:
        c.verify([row['saved']],boundary=boundary);saved=c.read(c.resolve(row['saved']))
        if row['model_id']=='historical_r0':
            saved=saved['evaluator_qualification']
            require(row['forward_calls']==2 and row['structural_outcomes_fabricated'] is False
                and row['prediction_rows']==saved['prediction_rows']
                and accepted.full_output_bytes(row['structural_outputs83'])==accepted.full_output_bytes(saved['structural_outputs83']),
                'Exact structural/genuine evaluator arithmetic')
        elif row['model_id']=='historical_r1':
            expected,_=_saved_prediction('r1_mixed_joint')
            panel=sampling.load_diagnostic_train(c.read(c.SOURCE_SELECTION),boundary=boundary)
            credits=[x['root_q_win_credit'] for x in panel[:32]]
            expected_rows=[dict(value,source='prior_20261107',root_q_win_credit=q)
                for value,q in zip(saved['rows'],credits,strict=True)]
            require(row['saved']==expected and row['forward_calls']==1
                and row['production_namespace_exercised'] is True and row['search_win_credit']==credits
                and row['prediction_rows']==expected_rows,'Exact production namespaced evaluator parity')
        else:
            expected,_=_saved_prediction('r1_mixed_joint' if row['model_id']=='historical_r1' else 'parent')
            require(row['saved']==expected and row['forward_calls']==1 and row['prediction_rows']==saved['rows'],
                'Exact accepted training-panel evaluator arithmetic')
    dataset=corpus.prior_training_dataset(boundary=boundary);_,order=load_order('control_r0',boundary=boundary)
    require([(r['case'],r['scheduled_step']) for r in result['scheduled_batch_replays']]==list(QUALIFICATION_BATCHES),
        'Full32, final3 and all three control pass-boundary batches')
    for row in result['scheduled_batch_replays']:
        indices=batch_indices(order,row['scheduled_step']);examples=[dataset[i] for i in indices]
        require(row['example_indices']==indices and row['batch_size']==len(indices)
            and row['batch_source_sha256']==batch_source(examples) and row['optimizer_calls']==2
            and row['discarded'] is True,'Actual namespaced scheduled qualification rows')
        before_payload,before=checkpoint_manifest(row['before'],boundary=boundary)
        after_payload,after=checkpoint_manifest(row['after'],boundary=boundary)
        require(before['metadata']['checkpoint_step']==0 and after['metadata']['checkpoint_step']==1
            and before['metadata']['qualification_only'] is after['metadata']['qualification_only'] is True
            and before['metadata']['qualification_case']==after['metadata']['qualification_case']==row['case']
            and after['tree_sha256']==row['after_tree_sha256'],'Exact disposable checkpoint pair')
        require(before['tree_sha256']==c.read(c.resolve(source_checkpoint(0,0)))['tree_sha256'],
            'Every numerical fixture starts at pristine EF8A/freshAdamW/seed0')
        compare_update(row['repeated_update'],row['direct_update'])
        require(row['direct_update']['batch_examples']==row['direct_update']['batch_denominator']==len(indices)
            and row['direct_update']['before_group_sha256']==before['metadata']['group_sha256']
            and row['direct_update']['after_group_sha256']==after['metadata']['group_sha256'],
            'Exact genuine-batch model-state chain')
        arithmetic=saved_batch_statistics(examples,row['direct_update']['outputs83'])
        require(all(math.isclose(row['direct_update'][k],v,rel_tol=1e-6,abs_tol=1e-5) for k,v in arithmetic.items()),
            'Independent actual full/partial/control-boundary mean losses')
        del before_payload,after_payload
    require(result['checkpoint_payload_qualification']==c.artifact(folder/'checkpoint_payload_qualification.json'),
        'Exact disposable checkpoint payload qualification receipt')
    verify_checkpoint_payload_qualification(result['checkpoint_payload_qualification'],
        [item for row in result['scheduled_batch_replays'] for item in (row['before'],row['after'])],boundary=boundary)
    return dict(status='qualification_saved_verified',result=item,optimizer_calls=12,
        evaluator_forward_calls=4,scientific_optimizer_calls=0,fresh_assessment_reads=0,**IDENTITY)
