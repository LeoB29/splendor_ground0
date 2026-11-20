"""Independently accepted production targets and outcome-blind root catalogue."""
from pathlib import Path
from copy import deepcopy
from collections import Counter
from . import common as c, schedule
from .collection import common as cc, data, audit, storage
_CATALOGUES={}


def _catalogue(receipt,*,force=False,boundary=lambda:None):
    """One authenticated immutable catalogue per worker, never target-value cache."""
    item=receipt['catalogue'];path=c.resolve(item);stat=path.stat()
    stamp=(stat.st_size,stat.st_mtime_ns,stat.st_ino);key=c.digest(item)
    if force or key not in _CATALOGUES:
        c.verify([item],boundary=boundary)
        value=cc.read(path);after=path.stat()
        c.require(stamp==(after.st_size,after.st_mtime_ns,after.st_ino),'Stable catalogue parse')
        _CATALOGUES[key]=(stamp,value)
    c.require(_CATALOGUES[key][0]==stamp,'Authenticated catalogue remains immutable')
    return _CATALOGUES[key][1]


def _roots(examples,spec):
    return [dict(game_index=spec['game_index'],ply=x['ply'],split=spec['split'],
                 row_sha256=x['row_sha256'],root_input_sha256=x['root_input_sha256']) for x in examples]


def _validate_features(examples):
    """Actual target-loader validation, also exercised by structural qualification."""
    c.require(type(examples) is list and examples,'Nonempty actual target features')
    batches=[]
    for start in range(0,len(examples),32):
        selected=examples[start:start+32]
        for row in selected:
            c.require(type(row['ply']) is int and type(row['actor']) is int and row['actor'] in (0,1), 'Strict decoded actor/ply')
            for key in ('row_sha256','root_input_sha256'):
                c.require(type(row[key]) is str and len(row[key])==64 and set(row[key])<=set('0123456789abcdef'),'Complete source-root identity')
            c.require(type(row['mask']) is list and all(type(x) is bool for x in row['mask']),'Strict legal mask flags')
        arrays=data.collate_arrays(selected,[1.]*len(selected))
        c.require(arrays['tokens'].reshape(len(selected),-1).tolist()==[x['tokens'] for x in selected], 'Input conversion preserves actual public tensors')
        batches.append(arrays)
    return batches


def _parse_target(value,spec):
    data._legacy.accepted.verify_identity(value)
    c.require(value['schema']=='gpu_collector_examples_v1' and c.canonical(value['spec'])==c.canonical(spec)
        and type(value['teacher_simulations']) is int and value['teacher_simulations']==512
        and value['official_terminal_only'] is True and value['training_updates']==0,'Authentic complete official targets')
    examples=deepcopy(value['examples'])
    c.require(examples and [x['ply'] for x in examples]==sorted({x['ply'] for x in examples}),'Unique ordered actual usable roots')
    _validate_features(examples)
    for x in examples:
        c.require(x['game_id']==spec['game_id'] and x['game_index']==spec['game_index'] and x['route']=='cpu','Exact target game identity')
        c.require(type(x['value']) is list and len(x['value'])==2 and all(type(v) in (int,float) and v in (-1,0,1) for v in x['value'])
            and sum(x['value'])==0,'Official signed actor/opponent value shape')
        x['split']=spec['split']
    return dict(examples=examples)


def _target(item,spec):
    return _parse_target(data.read_artifact(item),spec)


def accept_all(plan,*,boundary=lambda:None):
    """Re-audit every full cell, then publish only a complete1152-game corpus."""
    audit.set_boundary(boundary);games=[];proofs=[];specs=schedule.collection_schedule()
    for cell in plan['cells']:
        boundary();container=cc.OUT/'cells'/cell['cell_id']
        proof=audit.audit_cell(plan,cell,container)
        c.require(proof['status']=='passed' and proof['games']==16 and proof['full_cohort_verified'] is True,'Complete independent cell audit')
        proof_path=cc.OUT/'corpus_cell_audits'/(cell['cell_id']+'.json')
        proofs.append(cc.save(proof_path,proof))
        for spec,gameproof in zip(cell['specifications'],proof['game_proofs'],strict=True):
            boundary();loaded=_target(gameproof['target'],spec)
            games.append(dict(spec=spec,artifact=gameproof['target'],roots=_roots(loaded['examples'],spec),
                source_game=gameproof['document'],independent_cell_audit=proofs[-1]))
    c.require(len(games)==1152 and [g['spec'] for g in games]==specs,'Exact complete production sample before fitting')
    c.require(Counter(g['spec']['split'] for g in games)=={'train':1024,'validation':64,'offline':64},'Independent partitions complete')
    catalogue=dict(schema='independent_data_root_catalogue_v1',games=games,teacher_simulations=512,source_plan=c.artifact(cc.PLAN),
        complete_game_count=1152,targets_in_catalogue=False,storage=storage.contract())
    cat=cc.save(cc.OUT/'root_catalogue.json',catalogue)
    result=dict(schema='independent_corrected_corpus_acceptance_v1',status='accepted',passed=True,
        source_plan=c.artifact(cc.PLAN),catalogue=cat,independent_cell_audits=proofs,game_count=1152,
        independent_setup_count=1152,partition_counts={'train':1024,'validation':64,'offline':64},
        engineering_games_excluded=2,teacher_simulations=512,raw_semantics_and_all_transitions_verified=True,
        all_original_callbacks_attributed=True,selected_model_replays_verified=True,models_run=0,games_run=0,
        **data.IDENTITY)
    cc.save(cc.OUT/'corpus_acceptance.json',result)
    return result


def _receipt(acceptance,boundary):
    path=cc.OUT/'corpus_acceptance.json';descriptor=c.artifact(path)
    if type(acceptance) is dict and set(acceptance)=={'path','bytes','sha256'}:
        c.require(acceptance==descriptor,'Exact complete corpus acceptance')
    elif type(acceptance) is dict:
        c.require(c.canonical(acceptance)==c.canonical(cc.read(path)),'Exact accepted corpus document')
    else:c.require(Path(acceptance).resolve()==path.resolve(),'Only fixed corpus acceptance')
    boundary();value=cc.read(path)
    c.require(value['status']=='accepted' and value['passed'] is True and value['game_count']==1152
        and value['source_plan']==c.artifact(cc.PLAN),'Independent production acceptance and frozen source')
    return dict(acceptance=descriptor,catalogue=value['catalogue'],source_plan=value['source_plan'])


def selection_catalogue(acceptance,*,boundary=lambda:None):
    receipt=_receipt(acceptance,boundary)
    catalogue=_catalogue(receipt,force=True,boundary=boundary)
    c.require(catalogue['schema']=='independent_data_root_catalogue_v1' and catalogue['teacher_simulations']==512
        and catalogue['targets_in_catalogue'] is False,'Minimal outcome-blind root catalogue')
    return deepcopy(catalogue['games']),receipt


def authenticate_selection_source(selection,*,boundary=lambda:None):
    receipt=selection['source_receipt'];expected=_receipt(receipt['acceptance'],boundary)
    c.require(c.canonical(receipt)==c.canonical(expected),'Exact source receipt')
    catalogue=_catalogue(receipt,force=True,boundary=boundary)
    c.require(c.canonical(selection['games'])==c.canonical(catalogue['games']),'Selection preserves every authenticated game and root identity')
    return receipt


def load_selected_game(game,receipt,*,boundary=lambda:None):
    boundary();expected=_receipt(receipt['acceptance'],boundary)
    c.require(c.canonical(receipt)==c.canonical(expected),'Accepted source identity before target release')
    catalogue=_catalogue(receipt,boundary=boundary)
    index=game['spec']['game_index']
    c.require(type(index) is int and 0<=index<1152 and c.canonical(catalogue['games'][index])==c.canonical(game),'Only accepted scheduled target descriptor')
    value=_target(game['artifact'],game['spec'])
    c.require(_roots(value['examples'],game['spec'])==game['roots'],'All released roots bind independently accepted original evidence')
    return value


def qualify_training_inputs(folder,*,boundary=lambda:None):
    from . import sampling
    from .collection import qualification
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
    selected=sorted(examples,key=lambda x:(x['game_index'],x['ply']))[:32]
    c.require(len(selected)==32 and all(x['split']=='engineering' for x in selected),'Exactly32 genuine excluded qualification examples')
    # Exercise the same corrected collator used to verify collected targets.
    arrays=data.collate_arrays(selected,[1.]*32)
    c.require(arrays['tokens'].reshape(32,-1).tolist()==[x['tokens'] for x in selected],'Actual corrected target tensor conversion')
    genuine=sampling.collate(selected,[1.]*32)
    c.require(genuine['value'].tolist()==[x['value'] for x in selected] and
        all(genuine[key].tolist()==arrays[key].tolist() for key in arrays),'Actual selected-training collator preserves genuine official outcomes')
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
    result=dict(status='actual_production_loader_qualified',collector_audit=gate['audit'],targets=proof['engineering_targets'],
        structural_loader_checks=structural,structural_source_rows=source_rows,full_loader_checks=proof['full_loader_checks'],
        actual_selected_loader_and_collator_exercised=True,structural_terminal_labels_fabricated=False,
        actual_target_parser_rejections=negative,
        engineering_examples_available=len(examples),qualification_examples=32,source_plan=c.artifact(cc.PLAN),
        zero_scientific_source_games=True,synthetic_value_targets=False,storage=storage.contract(),**data.IDENTITY)
    c.save(folder/'loader_qualification.json',result)
    return dict(proof=result,qualification_examples=selected,qualification_weights=[1.]*32)
