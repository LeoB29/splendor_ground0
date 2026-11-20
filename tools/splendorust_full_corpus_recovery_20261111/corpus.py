"""Corrected saved-row loader plus mixed-target qualification in the new output."""
from pathlib import Path
from copy import deepcopy
from . import common as c, targets
import json
import os
import hashlib
from tools.splendorust_independent_data_20261107 import corpus as original, schedule
from tools.splendorust_independent_data_20261107.collection import common as cc, data, storage
_validate_features=original._validate_features
_target=original._target
_parse_target=original._parse_target
_CACHES={}

def feature_arrays(examples):
    """The same corrected production collation used for training input tensors."""
    import numpy as np
    arrays=data.collate_arrays(examples,[1.]*len(examples))
    return {key:np.ascontiguousarray(arrays[key]) for key in ('tokens','policy','mask')}

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

def index_row(example,label):
    keys=('game_id','game_index','ply','split','actor','canonical_players','observer_sha256',
          'row_sha256','root_input_sha256','route')
    row={key:deepcopy(example[key]) for key in keys}
    row.update(terminal_value=deepcopy(example['value']),root_q_win_credit=label['root_q_win_credit'],
        root_q_signed=label['root_q_signed'],root_q_source=deepcopy(label['source_move']))
    targets.for_arm(row,'terminal');targets.for_arm(row,'mixed')
    return row

def _write_jsonl(path,records):
    c.require(path.resolve().is_relative_to(c.OUT),'New cache only')
    with path.open('xb') as stream:
        for record in records:stream.write((c.canonical(record)+'\n').encode())
        stream.flush();os.fsync(stream.fileno())
    return c.artifact(path)

def build_cache(selection,folder,*,boundary=lambda:None):
    """One bounded source decode, with every corrected capture authenticated."""
    import numpy as np
    from . import sampling
    sampling.validate_selection(selection,boundary=boundary)
    base=c.OUT/'corpus';base.mkdir(parents=True,exist_ok=False)
    n=sampling.TOTAL_ROOTS
    layout={'tokens':((n,31,48),np.float32),'policy':((n,81),np.float32),'mask':((n,81),np.bool_)}
    arrays={k:np.lib.format.open_memmap(base/(k+'.npy'),mode='w+',dtype=d,shape=shape) for k,(shape,d) in layout.items()}
    index=[];ranges=[];offset=0
    for game in selection['games']:
        boundary();loaded=original.load_selected_game(game,selection['source_receipt'],boundary=boundary)
        examples=loaded['examples'];document=data.read_artifact(game['source_game'])
        data._legacy.accepted.verify_identity(document)
        c.require(document['spec']==game['spec'] and document['reason']=='completed','Natural authenticated source game')
        start=offset
        for j in range(0,len(examples),32):
            subset=examples[j:j+32];features=feature_arrays(subset)
            for key,array in features.items():arrays[key][offset:offset+len(subset)]=array
            for example in subset:
                boundary();item=document['rows'][example['ply']];row=data.read_artifact(item)
                label=targets.search_label(example,row,item,game['spec'])
                index.append(index_row(example,label));offset+=1
        ranges.append(dict(game_index=game['spec']['game_index'],split=game['spec']['split'],start=start,stop=offset,
            target=game['artifact'],source_game=game['source_game']))
    c.require(offset==n and len(index)==n,'All and only complete accepted source roots cached')
    for array in arrays.values():array.flush()
    del arrays
    descriptors={key:c.artifact(base/(key+'.npy')) for key in layout}
    idx=_write_jsonl(base/'index.jsonl',index)
    digests={arm:targets.source_digest(targets.for_arm(row,arm) for row in index if row['split']=='train') for arm in sampling.ARMS}
    manifest=dict(schema='full_corpus_encoded_cache_v1',**data.IDENTITY,selection=c.artifact(c.SELECTION),
        source_receipt=selection['source_receipt'],rows=n,root_counts=selection['root_counts'],arrays=descriptors,
        index=idx,game_ranges=ranges,training_target_digests=digests,precision='actual_training_float32_features',
        outcome_labels='official_terminal_only',search_labels='authenticated_actor_visit_weighted_root_value',
        source_moves_authenticated=n,model_calls=0,optimizer_calls=0)
    item=c.save(c.CACHE_MANIFEST,manifest)
    return dict(status='cache_built_pending_independent_audit',manifest=item,rows=n,models=0)

def _cache(selection,*,boundary=lambda:None,require_acceptance=True):
    import numpy as np
    from . import sampling
    descriptor=c.artifact(c.CACHE_MANIFEST);c.verify([descriptor],boundary=boundary)
    manifest=c.read(c.CACHE_MANIFEST)
    c.require(manifest['schema']=='full_corpus_encoded_cache_v1' and manifest['selection']==c.artifact(c.SELECTION)
        and manifest['source_receipt']==selection['source_receipt'] and manifest['root_counts']==selection['root_counts']
        and manifest['rows']==sampling.TOTAL_ROOTS and manifest['source_moves_authenticated']==sampling.TOTAL_ROOTS,
        'Exact full corrected cache recipe and source binding')
    data._legacy.accepted.verify_identity(manifest)
    built=c.read(c.OUT/'corpus_build/worker/result.json')
    c.require(built['status']=='passed' and built['mode']=='cache' and built['result']['manifest']==descriptor,
        'Manifest is the actual successful builder result')
    if require_acceptance:
        accepted=c.read(c.CACHE_ACCEPTANCE)
        c.require(accepted['status']=='independently_verified' and accepted['manifest']==descriptor
            and accepted['rows']==sampling.TOTAL_ROOTS and accepted['all_source_rows_checked'] is True,'Complete independent cache acceptance')
        audit_result=c.read(c.OUT/'corpus_audit/worker/result.json')
        c.require(audit_result['status']=='passed' and audit_result['mode']=='cache_audit'
            and audit_result['result']['acceptance']==c.artifact(c.CACHE_ACCEPTANCE),
            'Acceptance is the actual successful auditor result')
        for name in ('corpus_build','corpus_audit'):
            result=c.read(c.OUT/name/'execution.json')
            c.require(result['status']=='passed' and result['worker_exit_code']==0 and result['cleanup_errors']==[]
                and result['error'] is None,'Actual clean cache builder/auditor exit before training')
    inputs=[manifest['index'],*manifest['arrays'].values()];c.verify(inputs,boundary=boundary)
    index=[]
    with c.resolve(manifest['index']).open('r',encoding='utf-8') as stream:
        for line in stream:
            if len(index)%256==0:boundary()
            row=json.loads(line,object_pairs_hook=c._unique)
            targets.for_arm(row,'terminal');targets.for_arm(row,'mixed');index.append(row)
    c.require(len(index)==sampling.TOTAL_ROOTS,'Exact compact row inventory')
    arrays={key:np.load(c.resolve(item),allow_pickle=False,mmap_mode='r') for key,item in manifest['arrays'].items()}
    expected={'tokens':((sampling.TOTAL_ROOTS,31,48),np.dtype('float32')),
              'policy':((sampling.TOTAL_ROOTS,81),np.dtype('float32')),'mask':((sampling.TOTAL_ROOTS,81),np.dtype('bool'))}
    c.require(set(arrays)==set(expected),'All and only training input arrays')
    for key,(shape,dtype) in expected.items():
        c.require(arrays[key].shape==shape and arrays[key].dtype==dtype,'Exact mmap array shape and precision')
    for arm in sampling.ARMS:
        c.require(manifest['training_target_digests'][arm]==targets.source_digest(targets.for_arm(row,arm) for row in index if row['split']=='train'),
            'Exact complete value-target stream digest')
    return dict(manifest=manifest,index=index,arrays=arrays,descriptor=descriptor)

def load_cache(selection,*,boundary=lambda:None):
    """Authenticate immutable cache once per worker; no repeated raw decoder work."""
    item=c.artifact(c.CACHE_MANIFEST);key=c.digest(item)
    if key not in _CACHES:_CACHES[key]=_cache(selection,boundary=boundary)
    return _CACHES[key]

def audit_cache(selection,folder,*,boundary=lambda:None):
    """Independent raw-target/capture arithmetic comparison against every cache row."""
    import numpy as np
    import math
    from . import sampling
    result=c.read(c.OUT/'corpus_build/execution.json')
    c.require(result['status']=='passed' and result['worker_exit_code']==0 and not result['cleanup_errors'],'Cache builder actually exited')
    cache=_cache(selection,boundary=boundary,require_acceptance=False);offset=0
    for game,span in zip(selection['games'],cache['manifest']['game_ranges'],strict=True):
        boundary();examples=original.load_selected_game(game,selection['source_receipt'],boundary=boundary)['examples']
        c.require(span==dict(game_index=game['spec']['game_index'],split=game['spec']['split'],start=offset,
            stop=offset+len(examples),target=game['artifact'],source_game=game['source_game']),'Exact independent game spans')
        document=data.read_artifact(game['source_game']);data._legacy.accepted.verify_identity(document)
        c.require(document['spec']==game['spec'] and document['reason']=='completed','Exact natural accepted game')
        for example in examples:
            boundary();item=document['rows'][example['ply']];raw=data.read_artifact(item)
            # Independently recompute Q and tensor rows; source validator authenticates legal/rule/capture chain.
            decoded=data.validate_row(raw,game['spec'],raw['before'],example['ply'])
            native=raw['training_capture']['native_capture'];visits=native['visits67'];qs=native['q67']
            credit=math.fsum(float(v)*int(n) for n,v in zip(visits,qs) if v is not None)/sum(visits)
            c.require(sum(visits)==512 and abs(credit-native['root_value'])<=2e-6
                and raw['training_capture']['actor_seat']==example['actor'] and c.digest(raw)==example['row_sha256'],
                'Independent actor/Q/row authentication')
            for key in ('tokens','policy','mask','canonical_players','observer_sha256','root_input_sha256'):
                c.require(c.canonical(decoded[key])==c.canonical(example[key]),'Independent corrected encoded target '+key)
            label=dict(root_q_win_credit=native['root_value'],root_q_signed=2*native['root_value']-1,source_move=item)
            c.require(cache['index'][offset]==index_row(example,label),'Independent exact index/official outcome/Q')
            expected={'tokens':np.asarray(example['tokens'],dtype=np.float32).reshape(31,48),
                'policy':np.asarray(example['policy'],dtype=np.float32),'mask':np.asarray(example['mask'],dtype=np.bool_)}
            for key,array in expected.items():
                c.require(cache['arrays'][key][offset].tobytes()==array.tobytes(),'Independent exact encoded cache bytes '+key)
            offset+=1
    c.require(offset==sampling.TOTAL_ROOTS,'Every original root independently audited')
    proof=dict(status='independently_verified',manifest=cache['descriptor'],rows=offset,all_source_rows_checked=True,
        all_feature_bytes_checked=True,all_q_actor_and_terminal_targets_checked=True,models=0,optimizer_calls=0,
        builder_execution=c.artifact(c.OUT/'corpus_build/execution.json'),**data.IDENTITY)
    return dict(acceptance=c.save(c.CACHE_ACCEPTANCE,proof),**proof)

def attach_game_targets(game,examples,*,boundary=lambda:None):
    document=data.read_artifact(game['source_game'])
    data._legacy.accepted.verify_identity(document)
    c.require(document['schema']==data.GAME_SCHEMA and document['spec']==game['spec'] and document['reason']=='completed',
        'Accepted natural corrected source game')
    out=[]
    for example in examples:
        boundary();ply=example['ply']
        c.require(type(ply) is int and 0<=ply<len(document['rows']) and example['game_index']==game['spec']['game_index'],
            'Exact selected source-game move')
        item=document['rows'][ply];row=data.read_artifact(item)
        out.append(targets.attach(example,row,item,game['spec']))
    return out

def qualify_training_inputs(folder,*,boundary=lambda:None):
    from . import sampling
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
    from .sampling import CachedExamples
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
    return dict(proof=result,qualification_examples=selected,qualification_weights=[1.]*32)
