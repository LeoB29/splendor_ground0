"""Fixed, model-free development-panel sampling study; no data-dependent choice."""
from copy import deepcopy
import hashlib
from . import common as c

PANEL = c.ROOT/'outputs/decision_quality_preparation_20261027_01/panel.json'
OLD_MATRIX = c.ROOT/'outputs/decision_quality_design_20261027_01/prepared_request_matrix.json'
PREVIOUS = c.ROOT/'outputs/splendorust_decision_quality_20261027_02'
PREVIOUS_FREEZE_SHA = '2382d92fb7bdfea2835abfc2a3e8dcd0da769f4940921de8032e766bbebab65c'
PREVIOUS_REVIEW_SHA = '92a84da454acc395e653268548f653b56a5285faa6fa082875120a360d18a7e2'
LABELS = ('A','B','C','D','E','F')
POOLS = (3,6)
DOMAIN = 'sampling_stability_20261027_v1'

def seed(observer,label):
    c.require(label in LABELS,'Fixed seed label')
    if label=='A':return int(c.digest(observer)[:16],16)
    domain='decision_quality_saved_panel_20261027_v1' if label in ('B','C') else DOMAIN
    return int(hashlib.sha256((domain+'|'+label+'|'+c.canonical(observer)).encode()).hexdigest()[:16],16)

def build_matrix():
    panel,old=c.read(PANEL),c.read(OLD_MATRIX)
    c.require(panel['positions']==24 and len(panel['roots'])==24,'Fixed24 development roots')
    roots={r['root_id']:r for r in panel['roots']}
    pilot=list(old['pilot_roots'])
    c.require(len(pilot)==6 and all(isinstance(r,str) for r in pilot),'Six immutable pilot roots')
    ordered=[]
    for row in old['requests']:
        if row['stage'] in ('research_pilot','research_main') and row['root_id'] not in ordered:
            ordered.append(row['root_id'])
    c.require(len(ordered)==24 and ordered[:6]==pilot,'Previously prepared round-robin order')
    result=[]
    def add(rid,stage,pool,label,route='research',repeat_of=None):
        root=roots[rid]
        item=dict(sequence=len(result),root_id=rid,stage=stage,route=route,quota=64,
            world_pool=pool,seed_label=label,seed_family=label,seed=seed(root['observer'],label),
            observer_sha256=root['observer_sha256'],telemetry=route=='research',
            condition=f'pool{pool}-{label}',seconds_per_move=1.,internal_seconds=.9,
            complete_response_seconds=1.,seed_decimal=str(seed(root['observer'],label)))
        if repeat_of is not None:item['repeat_of']=repeat_of
        result.append(item)
    for root in panel['roots']:add(root['root_id'],'frozen_reference_reproduction',3,'A','frozen')
    conditions=[(pool,label) for label in LABELS for pool in POOLS]
    for index,rid in enumerate(ordered):
        offset=index%len(conditions);rotated=conditions[offset:]+conditions[:offset]
        for pool,label in rotated:add(rid,'research_pilot' if rid in pilot else 'research_main',pool,label)
    for rid in pilot:
        for pool in POOLS:
            parent=next(r['sequence'] for r in result if r['root_id']==rid and r['world_pool']==pool
                and r['seed_label']=='B' and r['stage']=='research_pilot')
            add(rid,'research_reproduction',pool,'B',repeat_of=parent)
    counts={stage:sum(r['stage']==stage for r in result) for stage in
            ('frozen_reference_reproduction','research_pilot','research_main','research_reproduction')}
    c.require(counts==dict(frozen_reference_reproduction=24,research_pilot=72,research_main=216,
                         research_reproduction=12) and len(result)==324,'Exact324 request matrix')
    return dict(schema='sampling_stability_fixed_matrix_v1',pilot_roots=pilot,root_order=ordered,
        seed_labels=list(LABELS),world_pools=list(POOLS),requests=result,request_counts=counts,
        condition_order='Within each root rotate the twelve seed/pool conditions by root-order index.',
        repeats='First fixed root per phase/seat cell, both pools, seedB, exact science.',
        no_retries=True,no_replacement_roots=True)

def build_plan():
    return dict(schema='sampling_stability_bounded_plan_v1',
        hypothesis='Root-pool6 may reduce whole-decision sensitivity to hidden-sampling seeds at the same64 simulations.',
        design_role='development diagnostic; not independent strength confirmation',
        panel=c.artifact(PANEL),previous_review=c.artifact(PREVIOUS/'postrun/independent_completion_review.json'),
        controls=dict(frozen_optimized64_unchanged=True,new_bridge_pool3_requires_exact_science=True,
            baseline_world_pool=3,candidate_world_pool=6,simulations=64,depth=64,batch=1,threads=1,
            weights_unchanged=True,precision='float32',chance_universes_setting=3,
            native_core_unchanged=True,reference16_world_pool=3),
        seeds=dict(labels=list(LABELS),A='First16 hex characters of canonical lawful-observer SHA256',
            BC='Exact previous decision_quality_saved_panel_20261027_v1 domain',DEF=DOMAIN,
            inputs='Lawful observer and fixed label only; no future draws, winners or timing'),
        random_coupling='Root-world draws and later transfer::encode hidden completions share RNG; '
            'pool6 shifts later neural-input completions. Canonical transitions ignore RNG; '
            'PUCT/ties are deterministic on this route. No separate-stream causal claim.',
        measurement=dict(primary='For each root: unequal Main actions among15 unordered seed pairs/15; '
                'mean paired pool6-minus-pool3 difference, lower is more stable.',
            secondary=['complete-action pair disagreement','roots better/equal/worse',
                'all-six-seed unanimity','fixed phase/seat tables','matched-seed pool3/pool6 action changes',
                'receipt timing and inference callback counts'],
            prohibited=['Q or stability as factual accuracy','playing-strength or Elo claims',
                'seed/root/profile selection after results','population significance claim','automatic promotion']),
        resources=dict(job_seconds=c.JOB_SECONDS,stage_seconds=c.STAGES,
            collection_cutoff_seconds=c.COLLECTION_CUTOFF_SECONDS,ordinary_external_seconds=1.,
            ordinary_internal_seconds=.9,initial_ram_stability_seconds=60,minimum_available_ram_bytes=2*c.GiB,
            maximum_ram_wait_seconds=300,max_rss_bytes=2*c.GiB,initial_free_disk_bytes=7*c.GiB,
            minimum_free_disk_bytes=5*c.GiB,max_evidence_bytes=c.MAX_EVIDENCE_BYTES,
            final_reserve_bytes=c.FINAL_RESERVE_BYTES,one_serial_worker=True),
        qualification=dict(actual_entrypoint_decisions=104,parity_decisions=25,total_decisions=129,
            graph_cases_total=44,workers=4,candidate_pool6_actual_rules_suite=True),
        diagnostic_requests=324,scientific_decisions_total=453,parent_requests_total=355,
        first_failure_stops=True,automatic_retries=0,new_games=0,training=False,
        external_candidate_testing='paused',larger_data_deferred=True,
        next_step='Recommend whether a separately qualified candidate merits fresh paired one-second games; '
            'this saved panel cannot establish stronger play.')

def prepare():
    c.require(c.file_hash(PREVIOUS/'freeze.json')==PREVIOUS_FREEZE_SHA,'Prior execution freeze changed')
    p=PREVIOUS/'postrun/independent_completion_review.json'
    c.require(c.file_hash(p)==PREVIOUS_REVIEW_SHA and c.read(p)['accepted'],'Prior acceptance changed')
    for name,value in (('study_plan.json',build_plan()),('prepared_request_matrix.json',build_matrix())):
        path=c.OUT/name
        if path.exists():c.require(c.read(path)==value,'Never change prepared study inputs')
        else:c.save(path,value)
    return dict(plan=c.artifact(c.OUT/'study_plan.json'),matrix=c.artifact(c.OUT/'prepared_request_matrix.json'))

if __name__=='__main__':print(c.canonical(prepare()))
