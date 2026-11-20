"""Separate receipts and original-clock bounds for saved-evidence recovery."""
from datetime import datetime, timezone
import hashlib,json,os,time
from pathlib import Path
ROOT=Path(__file__).resolve().parents[2]
PACKAGE='tools.splendorust_proxy_cleanup_recovery_20261113'
STUDY='splendorust_proxy_cleanup_recovery_20261113_v1'
OUT=ROOT/'outputs/splendorust_proxy_cleanup_recovery_20261113_01'
PLAN=OUT/'freeze.json'
PROTOCOL=ROOT/'docs/engineering/splendorust_proxy_cleanup_recovery_20261113.md'
OLD_OUT=ROOT/'outputs/splendorust_proxy_calibration_20261113_01'
OLD_PLAN=OLD_OUT/'freeze.json'
OLD_SHA='51340bd0cae293d84fbc181b8985195f76f1db61d4a30cc09fbaa0a012a18940'
GiB=1024**3
AUDIT_RSS=8*GiB
MODEL_RSS=AUDIT_RSS
MAX_OUTPUT=2*GiB
MIN_DISK=5*GiB
TOTAL_SECONDS=2*3600
MODEL_SECONDS=0
JOBS=('analysis','acceptance')
MODEL_JOBS=()
CAPS=dict(analysis=TOTAL_SECONDS,acceptance=TOTAL_SECONDS)
TRAIN_CPU=SUPERVISOR_CPU=6
DISCOVERED_UNIX=datetime.fromisoformat('2026-11-13T17:49:58+00:00').timestamp()
IDENTITY=dict(ruleset_id='splendor_base_qualified_rules_20261018_v1',catalogue_sha256='db4776918e612eee5a6f2617719c85ec5f413d592bbab0a64058babc83a70312',turn_contract='atomic_prereveal_completed_turn_v1',origin='generated_under_declared_rules')

def require(condition,message):
    if not condition: raise ValueError('PROXY_CLEANUP_RECOVERY: '+message)

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


def recovery_limits():
    original=read(OLD_OUT/'launch/owner.json')['limits']
    until=min(original['unix_deadline'],DISCOVERED_UNIX+TOTAL_SECONDS)
    return dict(original,unix_deadline=until,monotonic_deadline=min(original['monotonic_deadline'],original['monotonic_origin']+until-original['unix_origin']))

def design():
    return dict(study_id=STUDY,original_freeze_sha256=OLD_SHA,scope='saved_evidence_only_cleanup_proof_recovery',
        recovery_discovered_unix=DISCOVERED_UNIX,recovery_max_seconds=TOTAL_SECONDS,
        original_global_clock_preserved=True,original_analysis_deadline_preserved=True,
        original_attempt_remains_incomplete=True,replace_failed_receipt=False,models=0,search_calls=0,
        new_games=0,replay_decisions=0,rerun_saved_reproduction_checks=32,ordinary_saved_records=5120,
        unchanged_scientific_analysis=True,stages=list(JOBS),actual_rss=AUDIT_RSS,output_cap=MAX_OUTPUT,
        disk_floor=MIN_DISK,available_ram_gate=False,forecast_gate=False,automatic_retries=0,
        automatic_followup=False,reference_promoted=False,**IDENTITY)
