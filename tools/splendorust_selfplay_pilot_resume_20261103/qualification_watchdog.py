"""Owning watchdog for the single-clock fixed512 versus unchanged256 study; clocks never reset."""
import argparse
import os
import shutil
import subprocess
import sys
import time
import traceback
import psutil
from . import common as c
from tools.splendorust_search256_recovery_20261030_02 import placement
from tools.splendorust_search256_efficiency_20261031.common import validate_exchange
from tools.local_player_qualification_128.clock import _WindowsJob


def inherited_origins(freeze, *, model_required=True):
    """Authenticate original clocks without starting another job allowance."""
    from . import continuation
    c.require(c.SCOPE=='pilot' and c.JOB_SECONDS==72*3600 and c.COLLECTION_CUTOFF_SECONDS==60*3600,
              'Continuation retains original pilot total/model windows')
    c.verify([freeze['continuation']])
    manifest=c.read(c.ROOT/freeze['continuation']['path'])
    c.require(freeze['continuation']==c.artifact(continuation.MANIFEST) and
              manifest['original_job_seconds']==c.JOB_SECONDS and manifest['original_model_seconds']==c.COLLECTION_CUTOFF_SECONDS and
              manifest['next_game_index']==32 and manifest['failed_decision_retries']==0 and
              manifest['original_status_remains_incomplete'] is True and manifest['future_forecasts_advisory'] is True,
              'Exact clean-prefix continuation clock contract')
    path=continuation.OLD_OUT/'guardian_claim.json'
    descriptor=c.artifact(path)
    c.require(descriptor in manifest['inputs'],'Original guardian clock receipt remains preserved')
    original=c.read(path)
    mono,wall=manifest['original_monotonic_origin'],manifest['original_unix_origin']
    c.require(mono==original['monotonic_origin']==904900.656 and wall==original['unix_origin']==1793649048.764625,
              'Original pilot clock origins never reset')
    now,utc=time.monotonic(),time.time()
    cap=c.COLLECTION_CUTOFF_SECONDS if model_required else c.JOB_SECONDS
    c.require(mono<=now<mono+cap and wall<=utc<wall+cap,'Actual original continuation clocks remain open')
    return mono,wall


def require_isolated_request(request):
    c.require(request['placement_mode']=='isolated','Only isolated supervisory placement is authorized')


def cleanup_owned(process, job, streams, *, monotonic_origin, unix_origin, root_identity=None, extra_deadlines=None):
    """Terminate ownership immediately and share one clock-clamped exit wait."""
    begun = time.monotonic()
    local_deadline = begun + 5.0
    def remaining():
        available = min(local_deadline - time.monotonic(),
            monotonic_origin + c.JOB_SECONDS - time.monotonic(),
            unix_origin + c.JOB_SECONDS - time.time())
        if extra_deadlines is not None:
            available=min(available,extra_deadlines[0]-time.monotonic(),extra_deadlines[1]-time.time())
        return max(0.0,available)
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



def model_clock_open(started,wall_started,now,wall,*,final_started,owned_children):
    if final_started:
        c.require(not owned_children,'Saved-only final stage cannot own any model/helper process')
        return True
    return now<started+c.COLLECTION_CUTOFF_SECONDS and wall<wall_started+c.COLLECTION_CUTOFF_SECONDS


def stage_clock_open(start,now,wall):
    return now<start['monotonic_deadline'] and wall<start['unix_deadline']


def inner_main(args):
    claim=c.read(c.OUT/'guardian_claim.json')
    c.require(claim['freeze_sha256']==args.freeze_sha256,'Guardian freeze identity')
    started,wall_started=claim['monotonic_origin'],claim['unix_origin']
    c.require(c.file_hash(c.OUT/'freeze.json')==args.freeze_sha256,'Watchdog frozen continuation identity')
    freeze=c.read(c.OUT/'freeze.json')
    c.require((started,wall_started)==inherited_origins(freeze) and claim['inherited_original_clocks'] is True and
              started<=claim['owner_started_monotonic']<=time.monotonic() and
              wall_started<=claim['owner_started_unix_time']<=time.time(),
              'New watchdog inherits old clocks and distinct current owner starts')
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
            OMP_NUM_THREADS='1',MKL_NUM_THREADS='1',OPENBLAS_NUM_THREADS='1',CUDA_VISIBLE_DEVICES='',SPLENDOR_SEARCH256_STRENGTH_PHASE='strength',SPLENDOR_SEARCH512_STRENGTH_PHASE='strength')
        c.require(time.monotonic()<started+c.JOB_SECONDS and time.time()<wall_started+c.JOB_SECONDS,
                  'Shared job deadline before runner spawn')
        process=subprocess.Popen([sys.executable,'-B','-u','-m','tools.splendorust_selfplay_pilot_resume_20261103.qualification_control',
            'run','--freeze-sha256',args.freeze_sha256],cwd=c.ROOT,env=environment,stdout=stdout,stderr=stderr,
            creationflags=getattr(subprocess,'DETACHED_PROCESS',0))
        root_identity={'pid':process.pid,'created':psutil.Process(process.pid).create_time()}
        job=_WindowsJob(process.pid)
        c.save(c.OUT/'watchdog.json',dict(time=c.now(),watchdog_pid=os.getpid(),watchdog_created=psutil.Process().create_time(),
            runner_pid=process.pid,runner_created=psutil.Process(process.pid).create_time(),freeze_sha256=args.freeze_sha256,
            scope='owned_runner_and_descendants',job_seconds=c.JOB_SECONDS,monotonic_origin=started,unix_origin=wall_started))
        mode='isolated';sequence=0;final_admitted=False
        while process.poll() is None:
            if time.monotonic()>=started+c.JOB_SECONDS or time.time()>=wall_started+c.JOB_SECONDS:
                status='watchdog_deadline'; break
            active=[]
            for stage in c.STAGES:
                path=c.OUT/'stages'/f'{stage}_start.json'
                if path.exists() and not(c.OUT/'stages'/f'{stage}_end.json').exists():
                    active.append(c.read(path))
            c.require(len(active)<=1,'No simultaneous diagnostic stages')
            if active and not stage_clock_open(active[0],time.monotonic(),time.time()):
                status='stage_deadline';break
            final_path=c.OUT/'stages/final_start.json'
            if final_path.exists():
                owner=psutil.Process(process.pid)
                c.require(owner.create_time()==root_identity['created'],'Final-stage runner identity')
                c.require(not owner.children(recursive=True),'Saved-only final stage cannot own any model/helper process')
                final_admitted=True
            if not model_clock_open(started,wall_started,time.monotonic(),time.time(),final_started=final_admitted,owned_children=[]):
                status='collection_deadline';break
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
                validate_exchange(request,ack,freeze,c.read(c.OUT/'watchdog.json'),root_identity)
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


def independent_main(args,claim,freeze):
    """Own one saved-only reader after both experimental process exits."""
    from .independent_review import review_deadlines
    final=c.read(c.OUT/'stages/final_start.json')
    deadlines=review_deadlines(claim,final,time.monotonic(),time.time())
    def remaining():return min(deadlines[0]-time.monotonic(),deadlines[1]-time.time())
    process=job=stdout=stderr=None;identity=None;status='failed';error=None;review_artifact=None
    try:
        c.require(remaining()>0,'Original final allowance before independent readback')
        stdout=(c.OUT/'independent_review_stdout.log').open('xb')
        stderr=(c.OUT/'independent_review_stderr.log').open('xb')
        env=os.environ.copy();env.update(PYTHONPATH=str(c.ROOT/'src')+os.pathsep+str(c.ROOT),
            PYTHONDONTWRITEBYTECODE='1',SPLENDOR_SEARCH256_STRENGTH_PHASE='strength',SPLENDOR_SEARCH512_STRENGTH_PHASE='strength')
        process=subprocess.Popen([sys.executable,'-B','-u','-m',
            'tools.splendorust_selfplay_pilot_resume_20261103.independent_review','--freeze-sha256',args.freeze_sha256],
            cwd=c.ROOT,env=env,stdout=stdout,stderr=stderr,creationflags=getattr(subprocess,'DETACHED_PROCESS',0))
        identity=dict(pid=process.pid,created=psutil.Process(process.pid).create_time())
        job=_WindowsJob(process.pid)
        while process.poll() is None and remaining()>0:
            placement.verify_runtime(freeze,'isolated',role='watchdog',full=False)
            time.sleep(min(.1,max(0.,remaining())))
        if process.poll()==0 and remaining()>0:
            value=c.read(c.OUT/'independent_review.json')
            c.require(value['approved'] is True and value['freeze_sha256']==args.freeze_sha256 and
                value['reviewer']==identity,'Actual independent reader output identity')
            review_artifact=c.artifact(c.OUT/'independent_review.json')
            status='completed'
    except BaseException:error=traceback.format_exc()
    finally:
        cleanup=cleanup_owned(process,job,(stdout,stderr),monotonic_origin=claim['monotonic_origin'],
            unix_origin=claim['unix_origin'],root_identity=identity,extra_deadlines=deadlines)
        if cleanup['errors'] or remaining()<=0:status='failed'
        c.save(c.OUT/'independent_review_os_exit.json',dict(status=status,
            exit_code=None if process is None else process.returncode,reviewer=identity,
            observer='guardian owning Python Popen handle',freeze_sha256=args.freeze_sha256,
            original_monotonic_origin=claim['monotonic_origin'],original_unix_origin=claim['unix_origin'],
            monotonic_end=time.monotonic(),unix_end=time.time(),effective_monotonic_deadline=deadlines[0],
            effective_unix_deadline=deadlines[1],cleanup_errors=cleanup['errors'],cleanup_receipt=cleanup,
            review=review_artifact,error=error))
    return 0 if status=='completed' else 1


def guardian_main(args):
    """Own the actual watchdog handle; preserve both OS exit codes."""
    c.require(not (c.OUT/'guardian_claim.json').exists(),'No relaunch or clock reset')
    c.require(c.file_hash(c.OUT/'freeze.json')==args.freeze_sha256,'Guardian execution freeze identity')
    freeze=c.read(c.OUT/'freeze.json');mono,wall=inherited_origins(freeze)
    review=c.read(c.OUT/'prelaunch_review.json')
    c.require(review.get('approved') is True and review.get('freeze_sha256')==args.freeze_sha256,
              'Frozen attempt requires independent prelaunch approval')
    claim=dict(time=c.now(),monotonic_origin=mono,unix_origin=wall,
        freeze_sha256=args.freeze_sha256,guardian_pid=os.getpid(),guardian_created=psutil.Process().create_time(),
        prelaunch_review_sha256=c.file_hash(c.OUT/'prelaunch_review.json'),planned_games=c.MAX_GAMES,
        inherited_original_clocks=True,continuation=freeze['continuation'],
        owner_started_monotonic=time.monotonic(),owner_started_unix_time=time.time())
    c.atomic_save(c.OUT/'guardian_claim.json',claim)
    process=job=stdout=stderr=None;root_identity=None;status='guardian_failed';failure=None
    try:
        freeze=c.read(c.OUT/'freeze.json')
        c.require(freeze.get('phase')==c.PHASE,'Frozen phase differs from owning supervisor process')
        placement.configure_execution(freeze,'isolated',role='watchdog')
        stdout=(c.OUT/'watchdog_stdout.log').open('xb');stderr=(c.OUT/'watchdog_stderr.log').open('xb')
        env=os.environ.copy();env.update(PYTHONPATH=str(c.ROOT/'src')+os.pathsep+str(c.ROOT),PYTHONDONTWRITEBYTECODE='1',
            OMP_NUM_THREADS='1',MKL_NUM_THREADS='1',OPENBLAS_NUM_THREADS='1',CUDA_VISIBLE_DEVICES='',SPLENDOR_SEARCH256_STRENGTH_PHASE='strength',SPLENDOR_SEARCH512_STRENGTH_PHASE='strength')
        c.require(time.monotonic()<mono+c.JOB_SECONDS and time.time()<wall+c.JOB_SECONDS,
                  'Shared job deadline before watchdog spawn')
        process=subprocess.Popen([sys.executable,'-B','-u','-m','tools.splendorust_selfplay_pilot_resume_20261103.qualification_watchdog',
            '--inner','--freeze-sha256',args.freeze_sha256],cwd=c.ROOT,env=env,stdout=stdout,stderr=stderr,
            creationflags=getattr(subprocess,'DETACHED_PROCESS',0))
        root_identity=dict(pid=process.pid,created=psutil.Process(process.pid).create_time())
        job=_WindowsJob(process.pid)
        c.atomic_save(c.OUT/'guardian_owner.json',dict(guardian=claim,watchdog=root_identity))
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
    return independent_main(args,claim,freeze) if status=='completed' else 1


def main():
    parser=argparse.ArgumentParser()
    parser.add_argument('--freeze-sha256',required=True)
    parser.add_argument('--inner',action='store_true')
    args=parser.parse_args()
    return inner_main(args) if args.inner else guardian_main(args)


if __name__=='__main__':
    raise SystemExit(main())
