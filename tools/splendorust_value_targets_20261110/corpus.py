"""Corrected saved-row loader plus mixed-target qualification in the new output."""
from pathlib import Path
from copy import deepcopy
from . import common as c, targets
from tools.splendorust_independent_data_20261107 import corpus as original, schedule
from tools.splendorust_independent_data_20261107.collection import common as cc, data, storage
_validate_features=original._validate_features
_target=original._target
_parse_target=original._parse_target

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
    arrays=_validate_features(eligible)
    for start,expected_arrays in zip(range(0,len(eligible),32),arrays,strict=True):
        part=eligible[start:start+32];batch=sampling.collate_inputs(part,[1.]*len(part))
        c.require(all(batch[key].tolist()==expected_arrays[key].tolist() for key in expected_arrays),
            'Actual selected-training input/policy collator preserves all structural arrays')
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
    genuine=sampling.collate(selected,[1.]*32)
    c.require(genuine['value'].tolist()==[[targets.f32(v) for v in x['value']] for x in selected] and
        all(genuine[key].tolist()==arrays[key].tolist() for key in arrays),'Actual selected-training collator preserves fixed genuine outcome/search mixtures')
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
        actual_target_parser_rejections=negative,
        engineering_examples_available=len(examples),qualification_examples=32,source_plan=c.artifact(cc.PLAN),
        zero_scientific_source_games=True,synthetic_value_targets=False,value_target_contract=targets.contract(),value_target_sources_sha256=targets.source_digest(selected),storage=storage.contract(),**data.IDENTITY)
    c.save(folder/'loader_qualification.json',result)
    return dict(proof=result,qualification_examples=selected,qualification_weights=[1.]*32)
