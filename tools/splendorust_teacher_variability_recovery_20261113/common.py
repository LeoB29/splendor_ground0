"""Immutable evidence, accepted inputs and prospective resource bounds."""
from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path
import time

ROOT = Path(__file__).resolve().parents[2]
PACKAGE = 'tools.splendorust_teacher_variability_recovery_20261113'
STUDY = 'splendorust_teacher_variability_recovery_20261113_v1'
OUT = ROOT / 'outputs/splendorust_teacher_variability_recovery_20261113_01'
ORIGINAL_OUT = ROOT / 'outputs/splendorust_full_corpus_20261110_01'
SELECTION = SOURCE_SELECTION = ORIGINAL_OUT / 'selection.json'
ORIGINAL_COMMITMENT = ORIGINAL_OUT / 'all_fits_committed.json'
ORIGINAL_EVALUATION = ORIGINAL_OUT / 'offline/worker/evaluation_result.json'
CACHE_ACCEPTANCE = ORIGINAL_OUT / 'corpus/cache_acceptance.json'
CACHE_MANIFEST = ORIGINAL_OUT / 'corpus/cache_manifest.json'
PARENT_CHECKPOINT = ROOT / 'outputs/external_opponent_search_20261018_02/splendorust_model.pt'
PARENT_CHECKPOINT_SHA = 'ef8a4521cd6c03c7075f15efee23f4cde6ec94d1b5398765cf0c24a09ad745cb'
PROTOCOL = ROOT / 'docs/engineering/splendorust_teacher_variability_recovery_20261113.md'
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
    if not condition: raise ValueError('TEACHER_VARIABILITY_RECOVERY: '+message)

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
    from .clocks import check_deadlines as validate
    return validate(limits)

ORIGINAL_OUT=ROOT/'outputs/splendorust_teacher_variability_20261113_01'
ORIGINAL_FREEZE=ORIGINAL_OUT/'freeze.json'
ORIGINAL_FREEZE_SHA='60239d1f7221aa8f25557a6ebae5b18218d24c521fd086dceb0f796ed6aada54'
BOUNDARY=ROOT/'outputs/splendorust_teacher_variability_crash_review_20261113_01/crash_boundary.json'
PRESERVATION=BOUNDARY.parent/'preservation_inventory.json'
PREFIX_COUNT=1992
PANEL=ORIGINAL_OUT/'panel.json'
MATRIX=ORIGINAL_OUT/'requests.json'
REPEATS=ORIGINAL_OUT/'reproductions.json'
QUALIFICATION=ORIGINAL_OUT/'qualification_plan.json'
ADMISSION=OUT/'admission_requests.json'
SUFFIX=OUT/'suffix_requests.json'
JOBS=('admission','diagnostic','reproduction','analysis','acceptance')
MODEL_JOBS=('admission','diagnostic','reproduction')
CAPS=dict(admission=2*3600,diagnostic=12*3600,reproduction=2*3600,analysis=4*3600,acceptance=4*3600)

def require_original_absence():
    rows=read(BOUNDARY)['process_identity_checks']
    require(len(rows)==24 and all(not exact_alive(r['identity']) for r in rows),
        'All exact original identities must remain absent; reused PIDs are not original owners')

def require_original_inventory():
    expected={resolve(i) for i in read(PRESERVATION)['artifacts']}
    actual={p.resolve() for p in ORIGINAL_OUT.rglob('*') if p.is_file()}
    require(actual==expected,'Original evidence file inventory stays exact; no additions or removals')

def design():
    return dict(study_id=STUDY,science=read(ORIGINAL_FREEZE)['design'],
        original_freeze_sha256=ORIGINAL_FREEZE_SHA,original_attempt_remains_incomplete=True,
        result_label='amended_recovered_verification_after_host_reboot',
        preserved_prefix_decisions=PREFIX_COUNT,remaining_decisions=3072-PREFIX_COUNT,
        exact_reproduction_decisions=32,admission_exact_probes=2,
        original_qualification_decisions_reused=253,new_games=0,weight_updates=0,
        disclosed_ambiguous_retry_request_index=PREFIX_COUNT,maximum_ambiguous_retries=1,
        completed_decisions_replaced=0,record_before_dispatch=True,
        new_model_services=3,total_original_plus_recovery_services=5,
        original_absolute_deadlines_unchanged=True,downtime_charged=True,
        fresh_monotonic_epoch_mapped_from_remaining_wall_time=True,
        aggregate_original_plus_recovery_output_cap=MAX_OUTPUT,
        actual_disk_floor=MIN_DISK,model_rss=MODEL_RSS,audit_rss=AUDIT_RSS,
        no_forecast_gate=True,no_available_ram_gate=True,automatic_followup=False,
        original_scientific_thresholds_unchanged=True,**IDENTITY)
