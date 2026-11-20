"""Outcome-blind whole-game subsets and fixed-update repeated-pass schedules.

Selection reads accepted identities and ranges, never outcomes or target values.
The qualified corrected cache is reused read-only; no new corpus is generated.
"""
from __future__ import annotations
from collections import Counter
from copy import deepcopy
import math
from . import common as c

SCHEMA = 'splendorust_data_size_selection_v1'
SCHEDULE_SCHEMA = 'splendorust_data_size_schedule_v1'
DOMAIN = 'splendorust_data_size_20261113_v1'
SEEDS = (202611071, 202611072)
SIZES = (256, 512)
TRAIN_ROOTS, TOTAL_ROOTS, BATCH, STEPS = 57022, 64108, 32, 1782
SOURCE_SELECTION = dict(path='outputs/splendorust_full_corpus_20261110_01/selection.json',
    bytes=19335311, sha256='07f2b18cc99c0e51393d9e1cc5f670c2f0eae9e4389f51ab133bc40939593527')
SOURCE_MANIFEST = dict(path='outputs/splendorust_full_corpus_20261110_01/corpus/cache_manifest.json',
    bytes=579498, sha256='04754a3722b95cd09cc9b73292d17f9fda812df8747dbd42239f1100fce9cb26')
SOURCE_ACCEPTANCE = dict(path='outputs/splendorust_full_corpus_20261110_01/corpus/cache_acceptance.json',
    bytes=805, sha256='447163bcb1ff900bf4a77d53a433faf9e2ecdd7c9105123858f7371818142e03')
require, digest, canonical = c.require, c.digest, c.canonical
_VALIDATED = set()


def game_order(game_ids):
    require(type(game_ids) is list and len(game_ids)==1024 and
        all(type(x) is int for x in game_ids) and set(game_ids)==set(range(1024)),
        'All and only corrected training game identities')
    return sorted(game_ids,key=lambda gid:(digest([DOMAIN,'nested_whole_game_subset',gid]),gid))


def subset_definitions(refs,cache_indices):
    """Pure selection receives only identities, never labels or game outcomes."""
    require(type(refs) is list and type(cache_indices) is list and len(refs)==len(cache_indices),
        'Root identities and cache offsets align')
    require(all(type(r) is list and len(r)==2 and all(type(x) is int for x in r)
        and 0<=r[0]<1024 and 0<=r[1]<124 for r in refs),'Only eligible training root identities')
    require(len(set(map(tuple,refs)))==len(refs) and len(set(cache_indices))==len(refs)
        and all(type(i) is int and i>=0 for i in cache_indices),'Unique roots and cache offsets')
    order=game_order(sorted({r[0] for r in refs}));subsets={}
    for size in SIZES:
        games=set(order[:size]);local=[i for i,r in enumerate(refs) if r[0] in games]
        selected=[deepcopy(refs[i]) for i in local];n=len(selected)
        require(len({r[0] for r in selected})==size and n>0,'Every selected game contributes its complete root list')
        subsets[str(size)]=dict(game_count=size,game_ids=sorted(games),root_ids=selected,
            cache_indices=[cache_indices[i] for i in local],roots=n,one_pass_steps=math.ceil(n/BATCH),
            last_batch_size=n%BATCH or BATCH,all_eligible_roots_per_selected_game=True)
    require(set(subsets['256']['game_ids'])<set(subsets['512']['game_ids']), 'Strict nested whole-game subsets')
    return order,subsets


def _source_definition(source,manifest,acceptance,bindings,runtime_sources):
    from tools import future_training_rules_20261021 as guard
    require(runtime_sources==guard.SOURCE_HASHES,'Actual startup source receipt required')
    for item in (source,manifest,acceptance):
        require(all(item.get(k)==v for k,v in c.IDENTITY.items()),'Corrected source identity, no relabelling')
    require(source['schema']=='full_corpus_selection_v1' and source['historical_data_relabelled'] is False
        and source['selection_uses_outcomes'] is False and source['selection_uses_targets'] is False,
        'Accepted source selection semantics')
    require(source['root_counts']==dict(train=57022,validation=3533,offline=3553)
        and source['game_counts']==dict(train=1024,validation=64,offline=64),'Fixed accepted source population')
    require(acceptance['status']=='independently_verified' and acceptance['manifest']==SOURCE_MANIFEST
        and acceptance['rows']==TOTAL_ROOTS and all(acceptance[k] is True for k in
        ('all_feature_bytes_checked','all_q_actor_and_terminal_targets_checked','all_source_rows_checked')),
        'Independent corrected cache acceptance')
    require(manifest['schema']=='full_corpus_encoded_cache_v1' and manifest['selection']==SOURCE_SELECTION
        and manifest['source_receipt']==source['source_receipt'] and manifest['rows']==TOTAL_ROOTS
        and manifest['source_moves_authenticated']==TOTAL_ROOTS,'Accepted cache binds every source move')
    mixed=[x for x in source['cells'] if x['breadth']=='mixed']
    require(len(mixed)==2 and sorted(x['repetition'] for x in mixed)==[0,1], 'Two accepted mixed reference seeds')
    refs=mixed[0]['root_ids']
    require(len(refs)==TRAIN_ROOTS and all(x['root_ids']==refs and x['roots']==TRAIN_ROOTS
        and x['rng_seed']==SEEDS[x['repetition']] and x['optimizer_steps']==STEPS for x in mixed),
        'Identical full reference population and accepted training seeds')
    games=[g for g in source['games'] if g['spec']['split']=='train']
    require([g['spec']['game_index'] for g in games]==list(range(1024)), 'Complete ordered source training games')
    flattened=[[r['game_index'],r['ply']] for g in games for r in g['roots']]
    require(flattened==refs and all(r['split']=='train' for g in games for r in g['roots']),
        'Source cell contains every eligible training root and no held-out root')
    ranges={x['game_index']:x for x in manifest['game_ranges'] if x['split']=='train'}
    indices=[]
    for game in games:
        gid=game['spec']['game_index'];span=ranges[gid]
        require(span['source_game']==game['source_game'] and span['target']==game['artifact']
            and span['stop']-span['start']==len(game['roots']),'Accepted cache range and source identities')
        indices.extend(range(span['start'],span['stop']))
    order,subsets=subset_definitions(refs,indices)
    for subset in subsets.values():
        selected_lookup={tuple(ref):i for i,ref in enumerate(subset['root_ids'])}
        subset['first_pass_orders']={}
        for cell in mixed:
            rep=cell['repetition'];permutation=cell['permutations'][0]
            require(sorted(permutation)==list(range(TRAIN_ROOTS)) and
                [i for batch in cell['batches'] for i in batch]==permutation,
                'Accepted reference complete one-pass order')
            subset['first_pass_orders'][str(rep)]=[selected_lookup[tuple(refs[i])] for i in permutation
                if tuple(refs[i]) in selected_lookup]
    require(all(x['one_pass_steps']<STEPS for x in subsets.values()),'Both one-pass checkpoints precede common update budget')
    return dict(schema=SCHEMA,study_id=DOMAIN,**c.IDENTITY,runtime_sources=deepcopy(runtime_sources),
        source_selection=deepcopy(SOURCE_SELECTION),source_cache_manifest=deepcopy(SOURCE_MANIFEST),
        source_cache_acceptance=deepcopy(SOURCE_ACCEPTANCE),source_receipt=deepcopy(source['source_receipt']),
        source_bindings=deepcopy(bindings),source_training_games=1024,source_training_roots=TRAIN_ROOTS,
        nested_game_hash_order=order,subsets=subsets,seeds=list(SEEDS),batch_size=BATCH,matched_updates=STEPS,
        common_reference='accepted two mixed-target1024-game one-pass checkpoints',
        selected_subset_replicates=1,optimization_replicates=2,subsets_shared_across_seeds=True,
        selection_uses_outcomes=False,selection_uses_actions=False,selection_uses_values=False,
        selection_uses_timing=False,heldout_training_games=0,historical_data_relabelled=False,
        first_pass_order='accepted1024-reference permutation filtered to selected roots, separately per seed',
        later_pass_order='fresh prospective study/size/seed/pass/root identity hashes',
        no_new_games=True,cache_copied=False,source_game_rescan=False,
        limitation='Two optimization seeds do not measure uncertainty over alternative training-game subsets.')


def build_selection(*,runtime_sources=None,boundary=lambda:None):
    from tools import future_training_rules_20261021 as guard
    sources=guard.verify_runtime_sources() if runtime_sources is None else runtime_sources
    pins=[SOURCE_SELECTION,SOURCE_MANIFEST,SOURCE_ACCEPTANCE]
    c.verify(pins,boundary=boundary)
    source,manifest,acceptance=[c.read(c.resolve(item)) for item in pins]
    bindings=[deepcopy(x) for x in pins]+[manifest['index'],*manifest['arrays'].values()]
    for role,mode,result_key,expected in (
        ('corpus_build','cache','manifest',SOURCE_MANIFEST),
        ('corpus_audit','cache_audit','acceptance',SOURCE_ACCEPTANCE)):
        path=c.ORIGINAL_OUT/role/'worker/result.json';result=c.read(path);bindings.append(c.artifact(path))
        require(result['status']=='passed' and result['mode']==mode and result['error'] is None
            and result['result'][result_key]==expected,'Actual accepted cache worker result')
        path=c.ORIGINAL_OUT/role/'execution.json';execution=c.read(path);bindings.append(c.artifact(path))
        require(execution['status']=='passed' and execution['worker_exit_code']==0 and execution['error'] is None
            and execution['cleanup_errors']==[],'Observed accepted cache worker exit')
    bindings=sorted({x['path']:x for x in bindings}.values(),key=lambda x:x['path'])
    c.verify(bindings,boundary=boundary)
    return _source_definition(source,manifest,acceptance,bindings,sources)


def validate_selection(selection,*,authenticate=True,boundary=lambda:None):
    key=digest(selection)
    if key not in _VALIDATED:
        expected=build_selection(runtime_sources=selection.get('runtime_sources'),boundary=boundary)
        require(canonical(selection)==canonical(expected),'Exact prospective nested selection and source identities')
        _VALIDATED.add(key)
    if authenticate:c.verify(selection['source_bindings'],boundary=boundary)
    return selection


def _make_schedule(selection,repetition,game_count):
    require(type(repetition) is int and repetition in (0,1),'Fixed optimization repetition')
    require(type(game_count) is int and game_count in SIZES,'Fixed whole-game size')
    subset=selection['subsets'][str(game_count)];refs=subset['root_ids'];n=len(refs)
    require(n==subset['roots'] and math.ceil(n/BATCH)==subset['one_pass_steps']<STEPS,
        'Fixed subset root count and one-pass update count')
    batches=[];batch_passes=[];passes=[];presentations=0;per_game=Counter();number=0
    while len(batches)<STEPS:
        number+=1;start=len(batches)
        order=deepcopy(subset['first_pass_orders'][str(repetition)]) if number==1 else sorted(range(n),
            key=lambda i:(digest([DOMAIN,'subset_pass',game_count,SEEDS[repetition],number,refs[i]]),i))
        require(sorted(order)==list(range(n)),'Every complete-pass order contains all subset roots exactly once')
        full=[order[i:i+BATCH] for i in range(0,n,BATCH)]
        admitted=full[:STEPS-len(batches)]
        for batch in admitted:
            batches.append(batch);batch_passes.append(number);presentations+=len(batch)
            per_game.update(refs[i][0] for i in batch)
        passes.append(dict(pass_number=number,start_step=start+1,stop_step=len(batches),
            complete=len(admitted)==len(full),root_presentations=sum(map(len,admitted)),
            complete_pass_order_sha256=digest(order),complete_pass_batches=len(full)))
    endpoint=subset['one_pass_steps'];checkpoints=sorted({endpoint-1,endpoint,STEPS-1,STEPS})
    return dict(schema=SCHEDULE_SCHEMA,study_id=DOMAIN,game_count=game_count,repetition=repetition,
        seed=SEEDS[repetition],roots=n,batch_size=BATCH,batches=batches,batch_passes=batch_passes,
        pass_boundaries=passes,one_pass_step=endpoint,final_step=STEPS,checkpoint_steps=checkpoints,
        assessment_steps=dict(onepass=endpoint,matched=STEPS),presentations=presentations,
        presentation_count_by_game={str(k):v for k,v in sorted(per_game.items())},
        partial_batches=[dict(step=i+1,batch_size=len(b),pass_number=batch_passes[i]) for i,b in enumerate(batches) if len(b)<BATCH],
        complete_passes=sum(x['complete'] for x in passes),final_pass_complete=passes[-1]['complete'],
        one_pass_presentations=n,matched_reference_presentations=TRAIN_ROOTS,
        presentation_difference_from_reference=presentations-TRAIN_ROOTS,
        optimizer_updates_equal_reference=True,exact_equal_presentations=False,
        first_pass_order='filtered accepted1024-reference order',later_pass_order='fresh prospective identity hashes',
        shuffle_consumes_model_rng=False,
        padding=False,dropped_end_of_pass_rows=False,adaptive_stopping=False)


def build_schedules(selection,repetition,game_count):
    return _make_schedule(selection,repetition,game_count)


def validate_schedules(schedule,selection,repetition,game_count):
    require(canonical(schedule)==canonical(_make_schedule(selection,repetition,game_count)),
        'Exact fixed complete/partial pass schedule and presentation accounting')
    return schedule


def load_selected(selection,repetition,game_count,*,boundary=lambda:None):
    """New strict subset entry point; original corrected mmap reader is read-only."""
    from tools.splendorust_full_corpus_20261110 import corpus,sampling,targets
    validate_selection(selection,boundary=boundary)
    source=c.read(c.resolve(SOURCE_SELECTION));cache=corpus.load_cache(source,boundary=boundary)
    subset=selection['subsets'][str(game_count)];indices=subset['cache_indices']
    require([[cache['index'][i]['game_index'],cache['index'][i]['ply']] for i in indices]==subset['root_ids']
        and all(cache['index'][i]['split']=='train' for i in indices),'Actual corrected cache subset/root identity and no held-out fitting')
    schedule=build_schedules(selection,repetition,game_count)
    source_digest=targets.source_digest(targets.for_arm(cache['index'][i],'mixed') for i in indices)
    cell=dict(fit_id=f'g{game_count}_r{repetition}',game_count=game_count,repetition=repetition,
        rng_seed=SEEDS[repetition],game_ids=subset['game_ids'],root_ids=subset['root_ids'],
        roots=subset['roots'],optimizer_steps=STEPS,one_pass_step=schedule['one_pass_step'],
        scope='joint',breadth='mixed',batches=schedule['batches'],probe_batches=[])
    return dict(train_examples=sampling.CachedExamples(cache,indices,'mixed'),train_weights=[1.]*len(indices),
        train_game_ids=subset['game_ids'],batches=schedule['batches'],probe_batches=[],selection_cell=cell,
        source_receipt=selection['source_receipt'],value_target_sources_sha256=source_digest,
        corpus_cache=deepcopy(SOURCE_ACCEPTANCE),cache_validated=True,schedule=schedule,
        full_corpus_value_target_sources_sha256=cache['manifest']['training_target_digests']['mixed'],
        subset_game_count=game_count,subset_root_count=len(indices))
