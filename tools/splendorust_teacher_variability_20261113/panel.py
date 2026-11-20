"""Prospective, outcome-blind subset of the accepted 512-root development panel."""
from copy import deepcopy
from collections import Counter
from . import common as c
from tools.splendorust_proxy_calibration_20261113 import panel as source_panel

require, digest, canonical = c.require, c.digest, c.canonical
SOURCE = c.ROOT/'outputs/splendorust_proxy_calibration_20261113_01'
RECOVERY = c.ROOT/'outputs/splendorust_proxy_cleanup_recovery_20261113_01'
SOURCE_FREEZE_SHA='51340bd0cae293d84fbc181b8985195f76f1db61d4a30cc09fbaa0a012a18940'
SOURCE_PANEL_SHA='eeb8954f681513cd9a93ccbe7851eedb99c82fce29492b07579ef497a72c2c15'
CLOSEOUT=c.ROOT/'outputs/splendorust_proxy_recovery_closeout_20261113_01/closeout_complete.json'
CLOSEOUT_SHA='72d3d1aa013851f69c468513e97e306367760abc000536f0a316296cf258df15'
CONDITIONS=tuple((pool,'legacy',s,s) for pool in (3,6) for s in 'ABCD')+tuple(
    (pool,'split',w,e) for pool in (3,6) for w in 'AB' for e in 'AB')

def selections(panel):
    meta=panel['host_metadata']; chosen=[]; deep=[]
    for split in ('validation','offline'):
        games=sorted({m['game_index'] for m in meta.values() if m['split']==split},
            key=lambda g:(digest([c.STUDY,'primary_game_order',g]),g))
        require(len(games)==64,'64 source games per partition')
        groups={i:[] for i in range(4)}
        for i,g in enumerate(games):
            band=i%4
            candidates=[key for key,m in meta.items() if m['game_index']==g and m['quantile_band']==band]
            require(len(candidates)==1,'Unique existing position in assigned chronological band')
            groups[band].append(candidates[0]);chosen.append(candidates[0])
        for band,keys in groups.items():
            deep+=sorted(keys,key=lambda k:(digest([c.STUDY,'deep_subset',k]),k))[:8]
    chosen.sort(key=lambda k:(digest([c.STUDY,'request_root_order',k]),k))
    deep.sort(key=lambda k:(digest([c.STUDY,'request_root_order',k]),k))
    return chosen,deep

def build_panel(runtime_sources, boundary=lambda:None):
    from tools.external_strength_screen_20261018_02 import bridge
    from tools.splendorust_search_scale_20261024 import audit
    require(c.sha(SOURCE/'freeze.json')==SOURCE_FREEZE_SHA,'Accepted source freeze')
    require(c.sha(SOURCE/'panel.json')==SOURCE_PANEL_SHA,'Accepted source panel')
    require(c.sha(CLOSEOUT)==CLOSEOUT_SHA,'Recovered independent closeout')
    saved=c.read(SOURCE/'panel.json');source_panel.validate_panel(saved,authenticate=True,boundary=boundary)
    require(saved['runtime_sources']==runtime_sources,'Corrected source receipt')
    primary,deep=selections(saved);roots={r['root_id']:r for r in saved['roots']}
    acceptance_item=c.read(CLOSEOUT)['independent_acceptance'];c.verify([acceptance_item],boundary)
    accepted=c.read(c.resolve(acceptance_item))
    require(accepted['approved'] and accepted['status']=='recovered_independent_acceptance_passed','Accepted recovered source evidence')
    proof_item=accepted['saved_evidence'];c.verify([proof_item],boundary)
    prior_proof=c.read(c.resolve(proof_item))
    index_item=prior_proof['ordinary']['index'];c.verify([index_item],boundary)
    require(index_item==c.artifact(SOURCE/'jobs/diagnostic/worker/index.json'),'Accepted original search inventory')
    index=c.read(c.resolve(index_item))
    by={(r['spec']['root_id'],r['spec']['quota'],r['spec']['seed_label']):r for r in index['requests']}
    probes={};bindings=[c.artifact(SOURCE/'freeze.json'),c.artifact(SOURCE/'panel.json'),c.artifact(CLOSEOUT),
        index_item,acceptance_item,proof_item]
    for rid in primary:
        boundary();summed=[0]*67
        for seed in ('R','S'):
            item=by[rid,2048,seed]['receipt'];c.verify([item],boundary)
            receipt=c.read(c.resolve(item));require(receipt['status']=='ok' and receipt['accepted'],'Passed saved reference')
            visits=receipt['result']['research']['phases'][0]['telemetry']['visits']
            require(len(visits)==67 and sum(visits)==2048,'Original reference visits')
            summed=[a+b for a,b in zip(summed,visits)];bindings.append(item)
        observer=roots[rid]['observer'];native=bridge.encode_observation(observer)
        safe=audit.safe_choices(native,bridge.main_actions(observer))
        indices=sorted({audit.canonical_index(a) for a in safe},key=lambda i:(-summed[i],i))
        require(len(indices)>=2,'Two distinct safe probe actions')
        probes[rid]=indices[:2]
    result=deepcopy(saved)
    result.update(teacher_variability_study=c.STUDY,primary_root_ids=primary,deep_root_ids=deep,
        probe_actions67=probes,selection_uses_fresh_outputs=False,selection_uses_outcomes=False,
        reference_probe_selection='pooled prior2048R/S visit counts; not fresh results',
        prior_bindings=bindings)
    validate_panel(result)
    return result

def validate_panel(panel,authenticate=False,boundary=lambda:None):
    source_panel.validate_panel(panel,authenticate=authenticate,boundary=boundary)
    require(panel['teacher_variability_study']==c.STUDY,'Diagnostic panel identity')
    primary,deep=selections(panel)
    require(panel['primary_root_ids']==primary and panel['deep_root_ids']==deep,'Fixed outcome-blind selection')
    require(len(primary)==128 and len(deep)==64 and set(deep)<=set(primary),'Exact128/64 selection')
    require(set(panel['probe_actions67'])==set(primary),'Complete probe inventory')
    for pair in panel['probe_actions67'].values():
        require(len(pair)==2 and len(set(pair))==2 and all(type(i) is int and 0<=i<67 for i in pair),'Distinct Main probes')
    if authenticate:c.verify(panel['prior_bindings'],boundary)
    return panel

def request_matrix(panel):
    result=[]
    for ordinal,rid in enumerate(panel['primary_root_ids']):
        quotas=(512,2048) if rid in panel['deep_root_ids'] else (512,)
        for quota in quotas:
            rotated=CONDITIONS[ordinal%16:]+CONDITIONS[:ordinal%16]
            for pool,mode,w,e in rotated:
                result.append(dict(request_id=f'{rid}_q{quota}_p{pool}_{mode}_{w}_{e}',root_id=rid,
                    quota=quota,pool=pool,mode=mode,world_seed_label=w,encoding_seed_label=e,replicate=f'{w}_{e}'))
    require(len(result)==3072,'Fixed ordinary matrix')
    return result

def reproduction_matrix(panel):
    roots=sorted(panel['deep_root_ids'],key=lambda k:(digest([c.STUDY,'repeat',k]),k))[:16]
    by={(s['root_id'],s['quota'],s['pool'],s['mode'],s['world_seed_label'],s['encoding_seed_label']):s for s in request_matrix(panel)}
    result=[]
    for i,rid in enumerate(roots):
        for quota in (512,2048):
            pool,mode,w,e=CONDITIONS[i]
            original=by[rid,quota,pool,mode,w,e]
            result.append(dict(original,request_id='repeat_'+original['request_id'],original_request_id=original['request_id']))
    return result
