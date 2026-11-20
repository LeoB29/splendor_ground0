"""Immutable evidence, accepted inputs and prospective resource bounds."""
from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path
import time

ROOT = Path(__file__).resolve().parents[2]
PACKAGE = 'tools.splendorust_teacher_variability_20261113'
STUDY = 'splendorust_teacher_variability_20261113_v1'
OUT = ROOT / 'outputs/splendorust_teacher_variability_20261113_01'
ORIGINAL_OUT = ROOT / 'outputs/splendorust_full_corpus_20261110_01'
SELECTION = SOURCE_SELECTION = ORIGINAL_OUT / 'selection.json'
ORIGINAL_COMMITMENT = ORIGINAL_OUT / 'all_fits_committed.json'
ORIGINAL_EVALUATION = ORIGINAL_OUT / 'offline/worker/evaluation_result.json'
CACHE_ACCEPTANCE = ORIGINAL_OUT / 'corpus/cache_acceptance.json'
CACHE_MANIFEST = ORIGINAL_OUT / 'corpus/cache_manifest.json'
PARENT_CHECKPOINT = ROOT / 'outputs/external_opponent_search_20261018_02/splendorust_model.pt'
PARENT_CHECKPOINT_SHA = 'ef8a4521cd6c03c7075f15efee23f4cde6ec94d1b5398765cf0c24a09ad745cb'
PROTOCOL = ROOT / 'docs/engineering/splendorust_teacher_variability_20261113.md'
PLAN = OUT / 'freeze.json'
COMMITMENT = OUT / 'all_checkpoints_committed.json'
IDENTITY = dict(ruleset_id='splendor_base_qualified_rules_20261018_v1',
    catalogue_sha256='db4776918e612eee5a6f2617719c85ec5f413d592bbab0a64058babc83a70312',
    turn_contract='atomic_prereveal_completed_turn_v1',origin='generated_under_declared_rules')
GiB = 1024**3
TOTAL_SECONDS = 24*3600
MODEL_SECONDS = 18*3600
MIN_DISK = 5*GiB
MAX_OUTPUT = 16*GiB
MODEL_RSS = 4*GiB
AUDIT_RSS = 8*GiB
JOBS = ('qualification', 'diagnostic', 'reproduction', 'analysis', 'acceptance')
CAPS = dict(qualification=2*3600,diagnostic=12*3600,reproduction=2*3600,analysis=4*3600,acceptance=4*3600)
MODEL_JOBS = ('qualification','diagnostic','reproduction')
PANEL = OUT/'panel.json'
MATRIX = OUT/'requests.json'
REPEATS = OUT/'reproductions.json'
QUALIFICATION = OUT/'qualification_plan.json'
GRAPH_PLAN = ROOT/'outputs/splendorust_inference_opt_20261025_01/graph_parity_plan.json'

def require(condition,message):
    if not condition: raise ValueError('PROXY_CALIBRATION: '+message)

def canonical(value):
    return json.dumps(value,sort_keys=True,separators=(',',':'),allow_nan=False)

def digest(value): return hashlib.sha256(canonical(value).encode()).hexdigest()

def _unique(pairs):
    result={}
    for key,value in pairs:
        require(key not in result,'Duplicate JSON key');result[key]=value
    return result

def read(path):
    return json.loads(Path(path).read_text(encoding='utf-8'),object_pairs_hook=_unique,
        parse_constant=lambda value: (_ for _ in ()).throw(ValueError('Nonfinite JSON constant')))

def now(): return datetime.now(timezone.utc).isoformat()

def file_hash(path):
    with Path(path).open('rb') as stream: return hashlib.file_digest(stream,'sha256').hexdigest()
sha = file_hash

def artifact(path):
    path=Path(path).resolve()
    return dict(path=path.relative_to(ROOT).as_posix() if path.is_relative_to(ROOT) else str(path),
                bytes=path.stat().st_size,sha256=sha(path))

def resolve(item): return (ROOT/item['path']).resolve()

def verify(items,boundary=lambda:None):
    for item in items:
        boundary()
        require(set(item)=={'path','bytes','sha256'},'Exact artifact descriptor')
        require(type(item['bytes']) is int and item['bytes']>=0,'Strict artifact size')
        require(artifact(resolve(item))==item,'Changed input: '+item['path'])

def save(path,value):
    path=Path(path).resolve();require(path.is_relative_to(OUT),'New output boundary')
    path.parent.mkdir(parents=True,exist_ok=True)
    with path.open('x',encoding='utf-8',newline='\n') as stream:
        stream.write(canonical(value)+'\n');stream.flush();os.fsync(stream.fileno())
    return artifact(path)

def atomic_save(path,value):
    path=Path(path);require(not path.exists(),'Exclusive receipt')
    temporary=path.with_suffix(path.suffix+'.pending');save(temporary,value)
    os.rename(temporary,path);return artifact(path)

def status(path,value):
    path=Path(path).resolve()
    require(path.is_relative_to(OUT) and path.name in ('stage.json','progress.json'),'Mutable status allowlist')
    path.parent.mkdir(parents=True,exist_ok=True);temporary=path.with_suffix('.tmp')
    with temporary.open('w',encoding='utf-8',newline='\n') as stream:
        stream.write(canonical(value)+'\n');stream.flush();os.fsync(stream.fileno())
    from tools.splendorust_gpu_live_20261105_03.status_io import atomic_replace_status
    atomic_replace_status(temporary,path)

def nested_artifacts(value):
    if isinstance(value,dict):
        if {'path','bytes','sha256'}<=set(value):
            yield {k:value[k] for k in ('path','bytes','sha256')}
        else:
            for v in value.values(): yield from nested_artifacts(v)
    elif isinstance(value,list):
        for v in value: yield from nested_artifacts(v)

def identity():
    import psutil
    p=psutil.Process();return dict(pid=p.pid,created=p.create_time())

def exact_alive(ident):
    import psutil
    try:
        p=psutil.Process(ident['pid']);return p.create_time()==ident['created'] and p.is_running()
    except psutil.NoSuchProcess:return False

def check_deadlines(limits):
    require(limits['monotonic_origin']<=time.monotonic()<limits['monotonic_deadline'] and
            limits['unix_origin']<=time.time()<limits['unix_deadline'],'Original dual clock expired')

def design():
    return dict(study_id=STUDY, reference_checkpoint_sha256=PARENT_CHECKPOINT_SHA,
        source_roots=512, heldout_games=128, primary_roots=128, deep_roots=64,
        primary_quota=512, deep_quota=2048, pools=[3,6],
        legacy_seeds=['A','B','C','D'], split_world_seeds=['A','B'], split_encoding_seeds=['A','B'],
        ordinary_decisions=3072, exact_reproduction_decisions=32,
        maximum_model_services=3, startup_graph_cases_per_service=11,
        research_deadline_seconds=120, internal_deadline_seconds=119,
        precision='float32',cpuct=.4,depth=64,weight_updates=0,new_games=0,
        root_selection='one per game; balanced fixed chronological bands and partitions; outcome-blind hash order',
        split_rng='separate root-world generator; encoding/residual stream retains original use',
        nested_worlds=True, mode_is_disclosed_intervention=True, production_players_unchanged=True,
        primary_metric='legacy_mean_pairwise_JSD_pool6_minus_pool3_at512',
        primary_ci_confidence=.95, bootstrap_draws=20000,bootstrap_seed=202611133072,
        bootstrap_unit='whole_source_game_stratified_validation_offline',
        secondary_metrics='TV, actual-action disagreement, root-value variability, fixed-probe Q-contrast variability, timing',
        secondary_intervals_exploratory=True, stream_contrasts_not_additive_variance=True,
        budget_contrast='2048_minus512_on_same64_deep_roots_only; joint_Q_support_across_budgets',
        q_probe_selection='top two safe actions by pooled existing2048R/S visits; fixed before fresh requests',
        q_missing_policy='complete paired support; report coverage; no imputation',
        no_irreducible_noise_floor_claim=True,no_strength_claim=True,
        reused_development_positions=True,one_second_descriptive_only=True,margin700ms_descriptive_only=True,
        automatic_promotion=False,automatic_followup=False,automatic_retries=0,
        available_ram_gate=False,forecast_gate=False,
        total_seconds=TOTAL_SECONDS,model_seconds=MODEL_SECONDS,job_caps=CAPS,
        model_rss=MODEL_RSS,audit_rss=AUDIT_RSS,output_cap=MAX_OUTPUT,actual_disk_floor=MIN_DISK,
        **IDENTITY)
