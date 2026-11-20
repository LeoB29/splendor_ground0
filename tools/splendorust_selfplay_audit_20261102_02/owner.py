"""Own one primary and one independent saved reader, with observed OS exits."""
import argparse,os,sys,time,subprocess,traceback,shutil
from . import common as a,verifier as v
from tools.future_training_rules_20261021 import verify_runtime_sources
from tools.splendorust_search256_recovery_20261030_02 import placement
from tools.local_player_qualification_128.clock import _WindowsJob
from tools.splendorust_selfplay_20261101.qualification_watchdog import cleanup_owned
c=a.c

def worker(mode,sha):
    c.require(c.file_hash(a.OUT/'freeze.json')==sha,'Exact separate audit freeze')
    freeze=c.read(a.OUT/'freeze.json');claim=c.read(a.OUT/'claim.json');old,absent=a.check_original()
    c.require(claim['freeze_sha256']==sha,'Audit owner binding')
    guard=a.Guard(claim,old,a.PRIMARY_SECONDS if mode=='primary' else a.READER_SECONDS)
    original=c.read(c.OUT/'freeze.json');placement.configure_execution(original,'isolated')
    guard.guard();sources=verify_runtime_sources()
    c.verify(freeze['inputs']);c.verify([freeze['original_inventory']])
    snapshot=c.read(c.ROOT/freeze['original_inventory']['path'])['artifacts']
    c.require(a.inventory(guard.guard)==snapshot,'Every original evidence file preserved before readback')
    original_state=v.original.verify_runtime(original,guard)
    coverage=v.verify_saved(guard,original)
    c.require(a.inventory(guard.guard)==snapshot,'Every original evidence file preserved after readback')
    c.verify(freeze['inputs']);guard.guard()
    if mode=='primary':
        v.publish_verification(coverage,original_inventory=freeze['original_inventory'],audit_freeze=c.artifact(a.OUT/'freeze.json'))
    else:
        primary=c.read(a.OUT/'primary_exit.json')
        c.require(primary['status']=='completed' and primary['exit_code']==0 and not primary['cleanup']['errors'],
                  'Independent read follows actual primary exit')
        c.require(a.absent(primary['identity']),'Primary verifier absent')
        c.require(c.read(a.OUT/'recovered_verification.json')['coverage']==coverage,'Independent full coverage equality')
    value=dict(approved=True,status='recovered_saved_verification_only',mode=mode,freeze_sha256=sha,
        identity=dict(pid=os.getpid(),created=a.psutil.Process().create_time()),coverage=coverage,
        original_attempt_status='incomplete',original_final_stage_passed=False,source_guard=sources,
        original_owner_identities_absent=absent,original_inventory=freeze['original_inventory'],
        peak_rss_bytes=guard.peak_rss,new_models=0,new_games=0,new_optimizer_updates=0,
        monotonic_end=time.monotonic(),unix_end=time.time())
    a.save(mode+'_review.json',value);return 0

def run_child(mode,sha,claim,old):
    limit=a.PRIMARY_SECONDS if mode=='primary' else a.READER_SECONDS
    end,wall_end=a.deadlines(claim,old,time.monotonic(),time.time(),limit)
    def remaining():return min(end-time.monotonic(),wall_end-time.time())
    child=job=out=err=None;identity=None;failure=None;status='failed'
    try:
        c.require(remaining()>5,'Remaining explicit saved-reader allowance')
        out=(a.OUT/(mode+'_stdout.log')).open('xb');err=(a.OUT/(mode+'_stderr.log')).open('xb')
        env=os.environ.copy();env.update(PYTHONDONTWRITEBYTECODE='1',
            PYTHONPATH=str(c.ROOT/'src')+os.pathsep+str(c.ROOT),
            OMP_NUM_THREADS='1',MKL_NUM_THREADS='1',OPENBLAS_NUM_THREADS='1')
        child=subprocess.Popen([sys.executable,'-B','-u','-m','tools.splendorust_selfplay_audit_20261102_02.owner',
            '--mode',mode,'--freeze-sha256',sha],cwd=c.ROOT,env=env,stdout=out,stderr=err,
            creationflags=getattr(subprocess,'DETACHED_PROCESS',0))
        identity=dict(pid=child.pid,created=a.psutil.Process(child.pid).create_time());job=_WindowsJob(child.pid)
        a.save(mode+'_owner.json',dict(identity=identity,monotonic_deadline=end,unix_deadline=wall_end))
        while child.poll() is None:
            c.require(remaining()>0,'Owning saved-audit real clock')
            p=a.psutil.Process(child.pid)
            c.require(p.create_time()==identity['created'] and not p.children(recursive=True),'Exact child, no model descendants')
            c.require(p.memory_info().rss<=2*c.GiB,'Saved reader RSS2GiB')
            c.require(shutil.disk_usage(c.ROOT).free>=5*c.GiB,'Preserved disk reserve')
            c.require(sum(p.stat().st_size for p in a.OUT.rglob('*') if p.is_file())<a.MAX_BYTES,'Saved audit evidence cap')
            time.sleep(.1)
        c.require(child.returncode==0 and remaining()>0,'Saved reader actual successful exit within allowance')
        proof=c.read(a.OUT/(mode+'_review.json'))
        c.require(proof['approved'] and proof['identity']==identity and proof['freeze_sha256']==sha,'Actual child acceptance identity')
        status='completed'
    except BaseException:failure=traceback.format_exc()
    finally:
        cleanup=cleanup_owned(child,job,(out,err),monotonic_origin=old['monotonic_origin'],unix_origin=old['unix_origin'],
            root_identity=identity,extra_deadlines=(end,wall_end))
        if cleanup['errors'] or remaining()<=0:status='failed'
        a.save(mode+'_exit.json',dict(status=status,exit_code=None if child is None else child.returncode,
            identity=identity,error=failure,cleanup=cleanup,monotonic_end=time.monotonic(),unix_end=time.time()))
    return status=='completed'

def owner(sha):
    c.require(c.file_hash(a.OUT/'freeze.json')==sha,'Exact separate audit freeze')
    c.require(not(a.OUT/'claim.json').exists(),'Exclusive audit; no retry')
    freeze=c.read(a.OUT/'freeze.json');c.verify(freeze['inputs']);old,absent=a.check_original()
    review=c.read(a.OUT/'prelaunch_review.json')
    c.require(review['approved'] and review['freeze_sha256']==sha,'Independent exact audit freeze review')
    c.require(a.psutil.virtual_memory().available>=2*c.GiB and shutil.disk_usage(c.ROOT).free>=5*c.GiB,'Saved-only initial resources')
    mono,wall=time.monotonic(),time.time()
    prior=c.read(c.ROOT/'outputs/splendorust_selfplay_audit_20261102_01/claim.json')
    end=min(mono+a.SECONDS,old['monotonic_origin']+c.JOB_SECONDS,prior['monotonic_deadline'])
    wall_end=min(wall+a.SECONDS,old['unix_origin']+c.JOB_SECONDS,prior['unix_deadline'])
    c.require(min(end-mono,wall_end-wall)>=300,'At least five minutes of original global allowance remain')
    claim=dict(freeze_sha256=sha,monotonic_origin=mono,unix_origin=wall,monotonic_deadline=end,unix_deadline=wall_end,
        identity=dict(pid=os.getpid(),created=a.psutil.Process().create_time()),original_attempt_remains_incomplete=True)
    a.save('claim.json',claim);placement.configure_execution(c.read(c.OUT/'freeze.json'),'isolated',role='watchdog')
    success=run_child('primary',sha,claim,old)
    if success:success=run_child('independent',sha,claim,old)
    c.require(time.monotonic()<end and time.time()<wall_end,'Final audit owner acceptance within both original and audit clocks')
    a.save('completion.json',dict(status='recovered_verified_requires_actual_owner_exit' if success else 'incomplete',
        freeze_sha256=sha,original_attempt_status='incomplete',original_final_stage_passed=False,
        new_models=0,new_games=0,new_optimizer_updates=0,monotonic_end=time.monotonic(),unix_end=time.time()))
    c.require(time.monotonic()<end and time.time()<wall_end,'Final audit receipt flush within both clocks')
    return 0 if success else 1

def main():
    p=argparse.ArgumentParser();p.add_argument('--mode',choices=['owner','primary','independent'],default='owner');p.add_argument('--freeze-sha256',required=True)
    args=p.parse_args()
    try:return owner(args.freeze_sha256) if args.mode=='owner' else worker(args.mode,args.freeze_sha256)
    except BaseException:
        a.save(args.mode+'_failure.json',dict(time=c.now(),error=traceback.format_exc()));return 1

if __name__=='__main__':raise SystemExit(main())
