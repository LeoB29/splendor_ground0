"""Owning watchdog for the fresh isolated qualification; clocks never reset."""
import argparse
import os
import shutil
import subprocess
import sys
import time
import traceback
import psutil
from . import common as c, placement
from tools.local_player_qualification_128.clock import _WindowsJob


def require_isolated_request(request):
    c.require(request['placement_mode']=='isolated','Only isolated supervisory placement is authorized')


def cleanup_owned(process, job, streams, *, monotonic_origin, unix_origin, root_identity=None):
    """Terminate ownership immediately and share one clock-clamped exit wait."""
    begun = time.monotonic()
    local_deadline = begun + 5.0
    def remaining():
        return max(0.0, min(local_deadline - time.monotonic(),
            monotonic_origin + c.JOB_SECONDS - time.monotonic(),
            unix_origin + c.JOB_SECONDS - time.time()))
    allowance = remaining()
    errors, owned, confirmed, unconfirmed = [], [], [], []
    if process is not None:
        try:
            parent = psutil.Process(process.pid)
            identity = {'pid': parent.pid, 'created': parent.create_time()}
            if root_identity is None:
                c.require(process.poll() is None, 'Missing ownership identity for exited runner')
            c.require(root_identity is None or identity == root_identity, 'Runner PID identity changed')
            owned = [(p, {'pid': p.pid, 'created': p.create_time()})
                     for p in [parent] + parent.children(recursive=True)]
        except psutil.NoSuchProcess:
            pass
        except BaseException as error:
            errors.append('Ownership enumeration: ' + repr(error))
    if job is not None:
        try:
            job.close()
        except BaseException as error:
            errors.append('Job closure: ' + repr(error))
    kill_errors = {}
    for child, identity in reversed(owned):
        try:
            # The original process object checks PID creation identity before
            # kill; never operate on a fresh PID lookup without that identity.
            child.kill()
        except psutil.NoSuchProcess:
            pass
        except psutil.AccessDenied as error:
            kill_errors[(identity['pid'], identity['created'])] = repr(error)
    if process is not None:
        try:
            if process.poll() is None:
                process.kill()
            process.wait(timeout=remaining())
        except BaseException as error:
            if process.poll() is None:
                errors.append('Runner handle wait: ' + repr(error))
    for child, identity in owned:
        try:
            status = child.wait(timeout=remaining())
            confirmed.append({**identity, 'exit_status': status})
        except psutil.NoSuchProcess:
            confirmed.append({**identity, 'state': 'absent'})
        except (psutil.TimeoutExpired, psutil.AccessDenied) as error:
            unconfirmed.append({**identity, 'error': repr(error),
                'kill_error': kill_errors.get((identity['pid'], identity['created']))})
    if unconfirmed:
        errors.append('Owned process exits unconfirmed before cleanup deadline')
    for stream in streams:
        if stream is not None:
            try:
                stream.close()
            except BaseException as error:
                errors.append('Log closure: ' + repr(error))
    return {'errors': errors, 'confirmed': confirmed, 'unconfirmed': unconfirmed,
            'observed_owned_identities': len(owned), 'maximum_seconds': 5.0,
            'effective_allowance_seconds': allowance, 'elapsed_seconds': time.monotonic()-begun,
            'method': 'single_exit_wait_budget_clamped_to_monotonic_and_utc_job_deadlines_v2'}


def inner_main(args):
    claim=c.read(c.OUT/'guardian_claim.json')
    c.require(claim['freeze_sha256']==args.freeze_sha256,'Guardian freeze identity')
    started,wall_started=claim['monotonic_origin'],claim['unix_origin']
    owner_path=c.OUT/'guardian_owner.json'
    handshake_end=time.monotonic()+5.
    while not owner_path.exists():
        c.require(time.monotonic()<min(handshake_end,started+c.JOB_SECONDS) and
                  time.time()<wall_started+c.JOB_SECONDS,'Bounded guardian owner handshake')
        time.sleep(.01)
    owner=c.read(owner_path)
    self_identity=dict(pid=os.getpid(),created=psutil.Process().create_time())
    c.require(owner['guardian']==claim and owner['watchdog']==self_identity,'Guardian owning process binding')
    guardian=psutil.Process(claim['guardian_pid'])
    c.require(guardian.create_time()==claim['guardian_created'] and guardian.is_running(),
              'Original owning guardian must be live')
    c.require(not any((c.OUT/n).exists() for n in ('launch.json','watchdog.json','watchdog_claim.json',
        'execution.json','runner_stdout.log','runner_stderr.log')),'No automatic relaunch or clock reset')
    c.save(c.OUT/'watchdog_claim.json',dict(time=c.now(),monotonic_origin=started,unix_origin=wall_started,
        freeze_sha256=args.freeze_sha256,watchdog_pid=os.getpid(),watchdog_created=psutil.Process().create_time(),
        guardian_owner_sha256=c.file_hash(owner_path),guardian_pid=claim['guardian_pid'],guardian_created=claim['guardian_created']))
    stdout=stderr=process=job=None; status='failed'; root_identity=None
    try:
        c.require(c.file_hash(c.OUT/'freeze.json')==args.freeze_sha256,'Execution freeze changed')
        freeze=c.read(c.OUT/'freeze.json')
        c.require(freeze.get('phase')==c.PHASE,'Frozen phase differs from owning supervisor process')
        c.require(freeze['job_seconds']==c.JOB_SECONDS and freeze['stage_seconds']==c.STAGES and
            freeze['collection_cutoff_seconds']==c.COLLECTION_CUTOFF_SECONDS and freeze['automatic_retries']==0,'Fixed resource windows changed')
        c.require(shutil.disk_usage(c.ROOT).free>=c.INITIAL_FREE_DISK_BYTES,'Launch requires frozen initial free disk')
        initial_placement=placement.configure_execution(freeze,'isolated',role='watchdog')
        c.save(c.OUT/'watchdog_initial_placement.json',initial_placement)
        stdout=(c.OUT/'runner_stdout.log').open('xb'); stderr=(c.OUT/'runner_stderr.log').open('xb')
        environment=os.environ.copy()
        environment.update(PYTHONPATH=str(c.ROOT/'src')+os.pathsep+str(c.ROOT),PYTHONDONTWRITEBYTECODE='1',
            OMP_NUM_THREADS='1',MKL_NUM_THREADS='1',OPENBLAS_NUM_THREADS='1',CUDA_VISIBLE_DEVICES='')
        c.require(time.monotonic()<started+c.JOB_SECONDS and time.time()<wall_started+c.JOB_SECONDS,
                  'Shared job deadline before runner spawn')
        process=subprocess.Popen([sys.executable,'-B','-u','-m','tools.splendorust_pool6_hostdiag_20261027_02.runner',
            'run','--freeze-sha256',args.freeze_sha256],cwd=c.ROOT,env=environment,stdout=stdout,stderr=stderr,
            creationflags=getattr(subprocess,'DETACHED_PROCESS',0))
        root_identity={'pid':process.pid,'created':psutil.Process(process.pid).create_time()}
        job=_WindowsJob(process.pid)
        c.save(c.OUT/'watchdog.json',dict(time=c.now(),watchdog_pid=os.getpid(),watchdog_created=psutil.Process().create_time(),
            runner_pid=process.pid,runner_created=psutil.Process(process.pid).create_time(),freeze_sha256=args.freeze_sha256,
            scope='owned_runner_and_descendants',job_seconds=c.JOB_SECONDS,monotonic_origin=started,unix_origin=wall_started))
        mode='isolated';sequence=0
        while process.poll() is None:
            if time.monotonic()>=started+c.JOB_SECONDS or time.time()>=wall_started+c.JOB_SECONDS:
                status='watchdog_deadline'; break
            request_path=c.OUT/'placement_requests'/f'{sequence+1:04d}.json'
            if request_path.exists():
                request=c.read(request_path)
                require_isolated_request(request)
                c.require(request['sequence']==sequence+1 and request['freeze_sha256']==args.freeze_sha256 and
                    request['runner']==root_identity and request['watchdog']==dict(pid=os.getpid(),created=psutil.Process().create_time()),
                    'Watchdog placement request identity differs')
                runner=psutil.Process(process.pid)
                c.require(runner.create_time()==root_identity['created'] and
                    runner.cpu_affinity()==freeze['placement']['role_masks'][request['placement_mode']],
                    'Runner current placement differs from request')
                placement.verify_saved_runtime(request['runner_receipt'],freeze,request['placement_mode'],identity=root_identity)
                receipt=placement.configure_execution(freeze,request['placement_mode'],role='watchdog')
                ack=dict(sequence=request['sequence'],request_sha256=c.digest(request),freeze_sha256=args.freeze_sha256,
                    placement_mode=request['placement_mode'],watchdog_receipt=receipt)
                c.validate_exchange(request,ack,freeze,c.read(c.OUT/'watchdog.json'),root_identity)
                c.atomic_save(c.OUT/'placement_acks'/f'{sequence+1:04d}.json',ack)
                mode=request['placement_mode'];sequence+=1
            placement.verify_runtime(freeze,mode,role='watchdog',full=False)
            time.sleep(.1)
        if process.poll() is not None: status='completed' if process.returncode==0 else 'runner_failed'
    except BaseException:
        c.save(c.OUT/'watchdog_error.json',dict(time=c.now(),error=traceback.format_exc()))
        raise
    finally:
        cleanup=cleanup_owned(process,job,(stdout,stderr),monotonic_origin=started,
                              unix_origin=wall_started,root_identity=root_identity)
        errors=cleanup['errors']
        mono_end,wall_end=time.monotonic(),time.time()
        charged=max(mono_end-started,wall_end-wall_started)
        if charged>=c.JOB_SECONDS: status='watchdog_deadline'
        c.save(c.OUT/'execution.json',dict(status='cleanup_failed' if errors else status,
            exit_code=None if process is None else process.returncode,time=c.now(),freeze_sha256=args.freeze_sha256,
            monotonic_origin=started,unix_origin=wall_started,monotonic_end=mono_end,unix_end=wall_end,
            outer_elapsed_seconds=mono_end-started,wall_elapsed_seconds=wall_end-wall_started,charged_job_seconds=charged,
            cleanup_errors=errors,cleanup_receipt=cleanup,
            cleanup='Windows job closed; owned descendants terminated' if not errors else 'Incomplete cleanup'))




    return 0 if status=='completed' and not errors else 1


def guardian_main(args):
    """Own the actual watchdog handle; preserve both OS exit codes."""
    c.require(not (c.OUT/'guardian_claim.json').exists(),'No relaunch or clock reset')
    mono,wall=time.monotonic(),time.time()
    c.require(c.file_hash(c.OUT/'freeze.json')==args.freeze_sha256,'Guardian execution freeze identity')
    review=c.read(c.OUT/'prelaunch_review.json')
    c.require(review.get('approved') is True and review.get('freeze_sha256')==args.freeze_sha256,
              'Frozen attempt requires independent prelaunch approval')
    claim=dict(time=c.now(),monotonic_origin=mono,unix_origin=wall,
        freeze_sha256=args.freeze_sha256,guardian_pid=os.getpid(),guardian_created=psutil.Process().create_time(),
        prelaunch_review_sha256=c.file_hash(c.OUT/'prelaunch_review.json'))
    c.save(c.OUT/'guardian_claim.json',claim)
    process=job=stdout=stderr=None;root_identity=None;status='guardian_failed';failure=None
    try:
        freeze=c.read(c.OUT/'freeze.json')
        c.require(freeze.get('phase')==c.PHASE,'Frozen phase differs from owning supervisor process')
        placement.configure_execution(freeze,'isolated',role='watchdog')
        stdout=(c.OUT/'watchdog_stdout.log').open('xb');stderr=(c.OUT/'watchdog_stderr.log').open('xb')
        env=os.environ.copy();env.update(PYTHONPATH=str(c.ROOT/'src')+os.pathsep+str(c.ROOT),PYTHONDONTWRITEBYTECODE='1',
            OMP_NUM_THREADS='1',MKL_NUM_THREADS='1',OPENBLAS_NUM_THREADS='1',CUDA_VISIBLE_DEVICES='')
        c.require(time.monotonic()<mono+c.JOB_SECONDS and time.time()<wall+c.JOB_SECONDS,
                  'Shared job deadline before watchdog spawn')
        process=subprocess.Popen([sys.executable,'-B','-u','-m','tools.splendorust_pool6_hostdiag_20261027_02.watchdog',
            '--freeze-sha256',args.freeze_sha256],cwd=c.ROOT,env=env,stdout=stdout,stderr=stderr,
            creationflags=getattr(subprocess,'DETACHED_PROCESS',0))
        root_identity=dict(pid=process.pid,created=psutil.Process(process.pid).create_time())
        job=_WindowsJob(process.pid)
        c.save(c.OUT/'guardian_owner.json',dict(guardian=claim,watchdog=root_identity))
        while process.poll() is None:
            if time.monotonic()>=mono+c.JOB_SECONDS or time.time()>=wall+c.JOB_SECONDS:
                status='guardian_deadline';break
            placement.verify_runtime(freeze,'isolated',role='watchdog',full=False)
            time.sleep(.25)
        if process.poll() is not None:
            status='completed' if process.returncode==0 else 'watchdog_failed'
    except BaseException as error:
        failure=repr(error)
        c.save(c.OUT/'guardian_error.json',dict(time=c.now(),error=failure,traceback=traceback.format_exc()))
    finally:
        cleanup=cleanup_owned(process,job,(stdout,stderr),monotonic_origin=mono,unix_origin=wall,root_identity=root_identity)
        ended,wall_end=time.monotonic(),time.time()
        charged=max(ended-mono,wall_end-wall)
        if charged>=c.JOB_SECONDS:status='guardian_deadline'
        if cleanup['errors']:status='cleanup_failed'
        execution=dict(status=status,exit_code=None if process is None else process.returncode,
            observed_process='owning_watchdog',exit_code_observed_by='guardian Popen handle',
            freeze_sha256=args.freeze_sha256,time=c.now(),monotonic_origin=mono,unix_origin=wall,
            monotonic_end=ended,unix_end=wall_end,charged_job_seconds=charged,
            cleanup_errors=cleanup['errors'],cleanup_receipt=cleanup,error=failure)
        c.save(c.OUT/'watchdog_execution.json',execution)
    return 0 if status=='completed' else 1


def main():
    parser=argparse.ArgumentParser()
    parser.add_argument('--freeze-sha256',required=True)
    parser.add_argument('--inner',action='store_true')
    args=parser.parse_args()
    return inner_main(args) if args.inner else guardian_main(args)


if __name__=='__main__':
    raise SystemExit(main())
