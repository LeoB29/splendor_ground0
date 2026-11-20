"""Immutable evidence, accepted inputs and prospective resource bounds."""
from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path
import time

ROOT = Path(__file__).resolve().parents[2]
PACKAGE = 'tools.splendorust_data_size_20261113'
STUDY = 'splendorust_data_size_20261113_v1'
OUT = ROOT / 'outputs/splendorust_data_size_20261113_01'
ORIGINAL_OUT = ROOT / 'outputs/splendorust_full_corpus_20261110_01'
SOURCE_SELECTION = ORIGINAL_SELECTION = ORIGINAL_OUT / 'selection.json'
SELECTION = OUT / 'selection.json'
ORIGINAL_COMMITMENT = ORIGINAL_OUT / 'all_fits_committed.json'
ORIGINAL_EVALUATION = ORIGINAL_OUT / 'offline/worker/evaluation_result.json'
CACHE_ACCEPTANCE = ORIGINAL_OUT / 'corpus/cache_acceptance.json'
CACHE_MANIFEST = ORIGINAL_OUT / 'corpus/cache_manifest.json'
PARENT_CHECKPOINT = ROOT / 'outputs/external_opponent_search_20261018_02/splendorust_model.pt'
PARENT_CHECKPOINT_SHA = 'ef8a4521cd6c03c7075f15efee23f4cde6ec94d1b5398765cf0c24a09ad745cb'
PROTOCOL = ROOT / 'docs/engineering/splendorust_data_size_20261113.md'
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
TRAIN_CPU, SUPERVISOR_CPU = 4, 6
STEPS, BATCH_SIZE = 1782, 32
FIT_IDS = ('g256_r0','g256_r1','g512_r0','g512_r1')
JOBS = ('qualification',*FIT_IDS,'commit','offline','analysis','acceptance')
CAPS = dict(qualification=2*3600,**{x:2*3600 for x in FIT_IDS},commit=3600,
            offline=3*3600,analysis=3*3600,acceptance=4*3600)
MODEL_JOBS = ('qualification',*FIT_IDS,'offline')
ACTIVE_PROXY = ROOT/'outputs/splendorust_proxy_calibration_20261113_01'
ACTIVE_PROXY_SHA = '51340bd0cae293d84fbc181b8985195f76f1db61d4a30cc09fbaa0a012a18940'

def require(condition,message):
    if not condition: raise ValueError('DATA_SIZE: '+message)

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
    return dict(study_id=STUDY,game_counts=[256,512,1024],fresh_game_counts=[256,512],seeds=[0,1],target='mixed',
        endpoints=['onepass','matched'],matched_optimizer_updates=1782,batch_size=32,
        subsets='fixed_outcome_blind_nested_whole_games_shared_across_training_seeds',
        complete_subset_passes=True,genuine_partial_batches=True,final_pass_may_be_partial=True,
        fixed_train_updates_per_fit=1782,fresh_fits=4,scientific_new_updates=7128,
        maximum_qualification_updates=6,replay_updates=8,maximum_new_updates=7142,
        qualification_extra_evaluator_forwards=4,qualification_saved_cpu0_reproduction_on_cpu4=True,
        reused_1024_onepass=True,historical_parent_checkpoint_sha256=PARENT_CHECKPOINT_SHA,
        heldout_games=128,reused_heldout_diagnostic_only=True,train_panel_roots=3072,
        train_panel_membership_stratification=True,best_seed_selection=False,
        learning_rate=1e-5,value_loss_weight=1.,policy_loss_weight=1.,precision='float32',
        train_cpu=TRAIN_CPU,supervisor_cpu=SUPERVISOR_CPU,maximum_simultaneous_training_workers=1,
        active_proxy_unchanged=True,active_proxy_freeze_sha256=ACTIVE_PROXY_SHA,
        new_games=0,automatic_promotion=False,automatic_followup=False,
        available_ram_gate=False,forecast_gate=False,total_seconds=TOTAL_SECONDS,model_seconds=MODEL_SECONDS,
        job_caps=CAPS,model_rss=MODEL_RSS,audit_rss=AUDIT_RSS,output_cap=MAX_OUTPUT,actual_disk_floor=MIN_DISK,
        automatic_retries=0,**IDENTITY)
