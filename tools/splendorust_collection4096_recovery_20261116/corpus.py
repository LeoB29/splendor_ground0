"""Independently accepted production targets and outcome-blind root catalogue."""
from pathlib import Path
from copy import deepcopy
from collections import Counter
from . import common as c, schedule
from .collection import common as cc, data, audit, storage
_CATALOGUES={}


def cell_source(cell_id):
    """Frozen segment mapping; existing games are never copied or renamed."""
    c.require(type(cell_id) is str and len(cell_id)==4 and cell_id[0]=='c'
        and cell_id[1:].isdigit(), 'Canonical cell identifier')
    index=int(cell_id[1:])
    c.require(0<=index<c.CELLS and cell_id==f'c{index:03d}', 'Scheduled cell only')
    if index<41:return 'original_passed',c.ORIGINAL_OUT/'collection/cells'/cell_id
    if index==41:return 'original_recovered',c.ORIGINAL_OUT/'collection/cells'/cell_id
    return 'recovery_new',cc.OUT/'cells'/cell_id


def authenticate_original_cell(plan,cell,*,boundary=lambda:None):
    """Reuse already-passed semantic/replay checks after authenticating all bytes.

    The original stage checker only inspects saved execution clocks and actual
    process identities. It neither consults the rebooted current monotonic clock
    nor writes anything. Full move records bind the raw inputs, outputs, capture,
    complete action, official transition and original clock.
    """
    from tools.splendorust_collection4096_20261115.collection import audit as oldaudit
    original_plan=cc.read(c.ORIGINAL_OUT/'collection/plan.json')
    c.require(plan==original_plan, 'Recovery keeps exact original scientific collection plan')
    for mode,kind in (('games','cells'),('audit_cell','cell_audits')):
        boundary();oldaudit.stage_execution(original_plan,c.ORIGINAL_OUT/'collection'/kind/cell['cell_id'],
            mode=mode,cell=cell)
    path=first_audit_path(cell['cell_id']);proof=cc.read(path)
    c.require(proof['status']=='passed' and proof['full_cohort_verified'] is True
        and proof['cell_id']==cell['cell_id'] and proof['games']==16
        and proof['model_calls']==proof['training_updates']==0
        and proof['seed_decimals']==[s['seed_decimal'] for s in cell['specifications']],
        'Original complete independently accepted cohort')
    lossless=proof['journal']['lossless_storage'];raw=lossless['raw'];stored=lossless['stored']
    c.require(lossless['exact_byte_roundtrip'] is True and proof['journal']['journal']==stored
        and raw['path']==stored['path'] and raw['sha256']!=stored['sha256'],
        'Original raw journal is a logical digest; stored journal is physical evidence')
    seen=set()
    for item in c.nested_artifacts(proof):
        boundary()
        if item==raw:continue
        key=(item['path'],item['sha256'])
        if key not in seen:c.verify([item],boundary=boundary);seen.add(key)
    for game in proof['game_proofs']:
        document=cc.read(c.resolve(game['document']))
        c.verify(document['rows'],boundary=boundary)
    return proof


def first_audit_path(cell_id):
    segment,_=cell_source(cell_id)
    if segment=='original_passed':
        return c.ORIGINAL_OUT/'collection/cell_audits'/cell_id/'worker/saved_audit.json'
    if segment=='original_recovered':
        return cc.OUT/'recovery_audit/worker/saved_audit.json'
    return cc.OUT/'cell_audits'/cell_id/'worker/saved_audit.json'


def audit_source_cell(plan,cell,*,boundary=lambda:None):
    segment,container=cell_source(cell['cell_id'])
    if segment=='original_passed':
        proof=authenticate_original_cell(plan,cell,boundary=boundary)
    elif segment=='original_recovered':
        from . import partial
        proof=partial.audit_supplement(plan,boundary=boundary)
    else:
        audit.set_boundary(boundary);proof=audit.audit_cell(plan,cell,container)
    first=first_audit_path(cell['cell_id'])
    c.require(c.canonical(proof)==c.canonical(cc.read(first)),
        'Repeated full cell audit matches its first independent serialized receipt')
    returned=cc.read(first.parent/'result.json')
    c.require(returned['status']=='passed' and returned['result']==c.artifact(first),
        'Actual independent auditor returned this exact saved proof')
    c.require([p['game_id'] for p in proof['game_proofs']]==[s['game_id'] for s in cell['specifications']],
        'Every source game retains its original position and identity')
    return proof


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
    """Re-audit every full cell, then publish only a complete3,328-game fresh corpus."""
    audit.set_boundary(boundary);games=[];proofs=[];specs=schedule.collection_schedule()
    for cell in plan['cells']:
        boundary();proof=audit_source_cell(plan,cell,boundary=boundary)
        c.require(proof['status']=='passed' and proof['games']==16 and proof['full_cohort_verified'] is True,'Complete independent cell audit')
        proof_path=cc.OUT/'corpus_cell_audits'/(cell['cell_id']+'.json')
        proofs.append(cc.save(proof_path,proof))
        for spec,gameproof in zip(cell['specifications'],proof['game_proofs'],strict=True):
            boundary();loaded=_target(gameproof['target'],spec)
            games.append(dict(spec=spec,artifact=gameproof['target'],roots=_roots(loaded['examples'],spec),
                source_game=gameproof['document'],independent_cell_audit=proofs[-1]))
    c.require(len(games)==c.NEW_GAMES and [g['spec'] for g in games]==specs,'Exact complete production sample before fitting')
    c.require(Counter(g['spec']['split'] for g in games)==c.NEW_GAME_COUNTS,'Independent partitions complete')
    catalogue=dict(schema='collection4096_new_root_catalogue_v1',games=games,teacher_simulations=512,source_plan=c.artifact(cc.PLAN),
        complete_game_count=c.NEW_GAMES,targets_in_catalogue=False,storage=storage.contract())
    cat=cc.save(cc.OUT/'root_catalogue.json',catalogue)
    result=dict(schema='collection4096_new_corpus_acceptance_v1',status='accepted',passed=True,
        source_plan=c.artifact(cc.PLAN),catalogue=cat,independent_cell_audits=proofs,game_count=c.NEW_GAMES,
        independent_setup_count=c.NEW_GAMES,partition_counts=c.NEW_GAME_COUNTS,
        engineering_games_excluded=2,teacher_simulations=512,raw_semantics_and_all_transitions_verified=True,
        all_original_callbacks_attributed=True,selected_model_replays_verified=True,models_run=0,games_run=0,
        original_attempt_status='incomplete',amended_recovered_verification=True,
        original_collection_freeze=c.artifact(c.ORIGINAL_OUT/'freeze.json'),
        recovery_freeze=c.artifact(c.PLAN),original_completed_games=672,newly_collected_games=2656,
        gameplay_retries=0,verification_retries=1,
        interrupted_verification=dict(game_id='g0666',ply=40),
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
    c.require(value['schema']=='collection4096_new_corpus_acceptance_v1' and
        value['status']=='accepted' and value['passed'] is True and value['game_count']==c.NEW_GAMES
        and value['independent_setup_count']==c.NEW_GAMES and value['partition_counts']==c.NEW_GAME_COUNTS
        and value['source_plan']==c.artifact(cc.PLAN),'Independent production acceptance and frozen source')
    c.require(value['original_attempt_status']=='incomplete' and value['amended_recovered_verification'] is True
        and value['original_collection_freeze']==c.artifact(c.ORIGINAL_OUT/'freeze.json')
        and value['recovery_freeze']==c.artifact(c.PLAN)
        and value['original_completed_games']==672 and value['newly_collected_games']==2656
        and value['gameplay_retries']==0 and value['verification_retries']==1
        and value['interrupted_verification']==dict(game_id='g0666',ply=40),
        'Recovered acceptance explicitly preserves original failure and one verification-only retry')
    data._legacy.accepted.verify_identity(value)
    return dict(acceptance=descriptor,catalogue=value['catalogue'],source_plan=value['source_plan'])


def selection_catalogue(acceptance,*,boundary=lambda:None):
    receipt=_receipt(acceptance,boundary)
    catalogue=_catalogue(receipt,force=True,boundary=boundary)
    c.require(catalogue['schema']=='collection4096_new_root_catalogue_v1' and catalogue['teacher_simulations']==512
        and catalogue['targets_in_catalogue'] is False and catalogue['complete_game_count']==c.NEW_GAMES
        and catalogue['source_plan']==receipt['source_plan'] and len(catalogue['games'])==c.NEW_GAMES
        and Counter(g['spec']['split'] for g in catalogue['games'])==c.NEW_GAME_COUNTS,
        'Complete minimal outcome-blind root catalogue')
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
    c.require(type(index) is int and 0<=index<c.NEW_GAMES and c.canonical(catalogue['games'][index])==c.canonical(game),'Only accepted scheduled target descriptor')
    value=_target(game['artifact'],game['spec'])
    c.require(_roots(value['examples'],game['spec'])==game['roots'],'All released roots bind independently accepted original evidence')
    return value


def _union_source(acceptance_item,catalogue_item,*,fresh,boundary):
    """Authenticate compact accepted metadata; never relabel or recalculate targets."""
    c.verify([acceptance_item,catalogue_item],boundary=boundary)
    accepted=cc.read(c.resolve(acceptance_item));data._legacy.accepted.verify_identity(accepted)
    counts=c.NEW_GAME_COUNTS if fresh else {'train':1024,'validation':64,'offline':64}
    total=sum(counts.values())
    c.require(accepted['schema']==('collection4096_new_corpus_acceptance_v1' if fresh else
        'independent_corrected_corpus_acceptance_v1') and accepted['status']=='accepted' and accepted['passed'] is True
        and accepted['game_count']==accepted['independent_setup_count']==total
        and accepted['partition_counts']==counts and accepted['catalogue']==catalogue_item
        and accepted['teacher_simulations']==512 and accepted['raw_semantics_and_all_transitions_verified'] is True
        and accepted['selected_model_replays_verified'] is True,'Exact independently accepted source corpus')
    c.verify([accepted['source_plan']],boundary=boundary)
    source_plan=cc.read(c.resolve(accepted['source_plan']))
    planned=[spec for cell in source_plan['cells'] for spec in cell['specifications']]
    catalogue=cc.read(c.resolve(catalogue_item))
    c.require(catalogue['schema']==('collection4096_new_root_catalogue_v1' if fresh else
        'independent_data_root_catalogue_v1') and catalogue['teacher_simulations']==512 and
        catalogue['complete_game_count']==total and catalogue['targets_in_catalogue'] is False and
        catalogue['source_plan']==accepted['source_plan'],'Source-bound minimal accepted catalogue')
    games=catalogue['games']
    c.require(len(games)==total and [g['spec'] for g in games]==planned and
        [g['spec']['game_index'] for g in games]==list(range(total)) and
        Counter(g['spec']['split'] for g in games)==counts,'Exact full source partitions and schedule')
    seeds={g['spec']['seed_decimal'] for g in games}
    c.require(len(seeds)==total,'Unique independent setups inside source corpus')
    for game in games:
        boundary();spec=game['spec']
        c.require(spec['quota']==512 and spec['world_pool']==3 and spec['cpuct']==.4,
            'Three-world512 source game metadata unchanged')
        c.require(game['roots'] and all(r['game_index']==spec['game_index'] and r['split']==spec['split']
            for r in game['roots']),'Accepted roots retain actual source partition')
    return dict(acceptance=acceptance_item,catalogue=catalogue_item,source_plan=accepted['source_plan'],
        games=games,seeds=seeds,partition_counts=counts)


def build_union_inventory(plan,*,boundary=lambda:None):
    """Pure compact union: old training only, new training and new sealed assessment."""
    old=plan['existing_training_corpus']
    c.require(type(old) is dict and set(old)=={'acceptance','catalogue'},'Frozen prior source bindings')
    fresh_item=c.artifact(cc.OUT/'corpus_acceptance.json')
    fresh_accepted=cc.read(c.resolve(fresh_item));_receipt(fresh_item,boundary)
    prior=_union_source(old['acceptance'],old['catalogue'],fresh=False,boundary=boundary)
    fresh=_union_source(fresh_item,fresh_accepted['catalogue'],fresh=True,boundary=boundary)
    c.require(not prior['seeds'].intersection(fresh['seeds']),'Fresh setup seeds disjoint from every prior source game')
    sources={name:{k:value[k] for k in ('acceptance','catalogue','source_plan','partition_counts')}
        for name,value in (('prior_20261107',prior),('fresh_20261115',fresh))}
    def members(source,split):
        data_source=prior if source=='prior_20261107' else fresh
        return [dict(source=source,game_index=g['spec']['game_index'],game_id=g['spec']['game_id'],split=split)
            for g in data_source['games'] if g['spec']['split']==split]
    training=members('prior_20261107','train')+members('fresh_20261115','train')
    validation=members('fresh_20261115','validation');offline=members('fresh_20261115','offline')
    reused=members('prior_20261107','validation')+members('prior_20261107','offline')
    result=dict(schema='collection4096_accepted_training_union_v1',status='accepted',sources=sources,
        training_members=training,fresh_assessment_members=validation+offline,
        reused_development_members_excluded_from_fresh_assessment=reused,
        training_games=4096,fresh_assessment_games=256,fresh_assessment_partitions={'validation':128,'offline':128},
        old_training_games=1024,new_training_games=3072,old_reused_development_games=128,
        source_game_ids_are_namespaced_not_renumbered=True,original_targets_modified=False,
        fresh_assessment_sealed_until_future_checkpoints_committed=True,
        future_loader_must_reverify_original_target_artifacts=True,
        future_scientific_training_authorized=False,models_run=0,games_run=0,**data.IDENTITY)
    validate_union_inventory(result)
    return result


def validate_union_inventory(value):
    """Check the disjoint split arithmetic without consulting outcomes or targets."""
    data._legacy.accepted.verify_identity(value)
    c.require(value['schema']=='collection4096_accepted_training_union_v1' and value['status']=='accepted',
        'Accepted union schema')
    training=value['training_members'];fresh=value['fresh_assessment_members']
    reused=value['reused_development_members_excluded_from_fresh_assessment']
    expected=((training,4096),(fresh,256),(reused,128))
    for rows,count in expected:
        c.require(type(rows) is list and len(rows)==count,'Exact complete union cohort')
        c.require(all(type(r) is dict and set(r)=={'source','game_index','game_id','split'} and
            type(r['game_index']) is int and type(r['game_id']) is str for r in rows),'Strict namespaced membership')
    keys=[(r['source'],r['game_index']) for rows,_ in expected for r in rows]
    c.require(len(keys)==len(set(keys)),'No duplicate or cross-partition source member')
    c.require(all(r['split']=='train' for r in training) and
        Counter(r['source'] for r in training)=={'prior_20261107':1024,'fresh_20261115':3072},
        'Only1024oldtraining plus3072newtraining')
    c.require(all(r['source']=='fresh_20261115' and r['split'] in ('validation','offline') for r in fresh)
        and Counter(r['split'] for r in fresh)=={'validation':128,'offline':128},'Only new256 games are fresh assessment')
    c.require(all(r['source']=='prior_20261107' and r['split'] in ('validation','offline') for r in reused)
        and Counter(r['split'] for r in reused)=={'validation':64,'offline':64},'Old128 remain reused development only')
    for rows,_ in expected:
        for row in rows:
            n=row['game_index'];split=row['split'];source=row['source']
            allowed=(0<=n<1024 if split=='train' else 1024<=n<1088 if split=='validation' else 1088<=n<1152) if source=='prior_20261107' else (
                0<=n<3072 if split=='train' else 3072<=n<3200 if split=='validation' else 3200<=n<3328)
            c.require(source in ('prior_20261107','fresh_20261115') and allowed and row['game_id']==f'g{n:04d}',
                'Exact original index and split retained, no relabelling')
    c.require(value['training_games']==4096 and value['fresh_assessment_games']==256 and
        value['old_training_games']==1024 and value['new_training_games']==3072 and value['old_reused_development_games']==128
        and value['fresh_assessment_partitions']=={'validation':128,'offline':128},'Exact union count claims')
    c.require(value['source_game_ids_are_namespaced_not_renumbered'] is True and value['original_targets_modified'] is False
        and value['fresh_assessment_sealed_until_future_checkpoints_committed'] is True
        and value['future_loader_must_reverify_original_target_artifacts'] is True
        and value['future_scientific_training_authorized'] is False and value['models_run']==value['games_run']==0,
        'Union is metadata preparation, not new models or scientific training')
    return dict(status='passed',training_games=4096,fresh_assessment_games=256,reused_development_games=128,
        namespaced_source_members=4480,models_run=0,games_run=0)


def union_inventory(plan=None,*,boundary=lambda:None):
    result=build_union_inventory(c.read(c.PLAN) if plan is None else plan,boundary=boundary)
    item=c.save(c.OUT/'training_union.json',result)
    return dict(status='accepted',inventory=item,verification=validate_union_inventory(result))


def verify_union_inventory(item,plan=None,*,boundary=lambda:None):
    c.verify([item],boundary=boundary)
    c.require(c.resolve(item)==c.OUT/'training_union.json','Only this attempt union inventory')
    saved=c.read(c.resolve(item))
    expected=build_union_inventory(c.read(c.PLAN) if plan is None else plan,boundary=boundary)
    c.require(c.canonical(saved)==c.canonical(expected),'Independent union reconstruction from accepted source catalogues')
    return validate_union_inventory(saved)


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
