"""Owning watchdog: one audit child and one non-resettable30-minute clock."""
import argparse
import os
import shutil
import subprocess
import sys
import time
import traceback
import psutil
from . import common as c
AUDITOR_CREATION_FLAGS=subprocess.DETACHED_PROCESS

def remaining(start,wall):return min(start+c.JOB_SECONDS-time.monotonic(),wall+c.JOB_SECONDS-time.time())

def cleanup(process,job,streams,start,wall,identity):
    begun=time.monotonic();errors=[];unconfirmed=[]
    def allowance():return max(0.,min(5-(time.monotonic()-begun),remaining(start,wall)))
    if job is not None:
        try:job.close()
        except BaseException as error:errors.append('job close: '+repr(error))
    if process is not None:
        try:
            if process.poll() is None:process.kill()
            process.wait(timeout=allowance())
        except BaseException as error:
            if process.poll() is None:unconfirmed.append(dict(identity=identity,error=repr(error)))
    for stream in streams:
        if stream is not None:
            try:stream.close()
            except BaseException as error:errors.append('log close: '+repr(error))
    if unconfirmed:errors.append('Owned audit exit unconfirmed within shared deadline')
    return dict(errors=errors,unconfirmed=unconfirmed,elapsed_seconds=time.monotonic()-begun,
        maximum_seconds=5,clamped_to_both_global_clocks=True)

def resource_guard(process,parent_bytes=0):
    c.require(psutil.virtual_memory().available>=c.MIN_AVAILABLE,'Watchdog available RAM reserve')
    c.require(shutil.disk_usage(c.ROOT).free>=c.MIN_DISK,'Watchdog free disk reserve')
    c.require(parent_bytes+sum(path.stat().st_size for path in c.OUT.rglob('*') if path.is_file())<c.MAX_OUTPUT,'Watchdog combined output ceiling')
    if process is not None:
        try:
            child=psutil.Process(process.pid)
            c.require(not child.children(recursive=True),'Auditor launched an unexpected child')
            c.require(child.memory_info().rss<=c.MAX_RSS,'Watchdog auditor RSS ceiling')
            c.require(child.cpu_affinity()==list(range(2,20)),'Auditor placement changed')
        except psutil.NoSuchProcess:
            c.require(process.poll() is not None,'Auditor process identity disappeared')

def main():
    parser=argparse.ArgumentParser();parser.add_argument('--freeze-sha256',required=True);args=parser.parse_args()
    start,wall=c.SHARED_MONOTONIC_ORIGIN,c.SHARED_UNIX_ORIGIN
    c.require(remaining(start,wall)>0,'Original shared recovery deadline already expired')
    c.require(not any((c.OUT/name).exists() for name in ('watchdog_claim.json','execution.json','auditor_stdout.log','auditor_stderr.log')),'No relaunch or clock reset')
    me=psutil.Process();me.cpu_affinity(list(range(2,20)));me.nice(psutil.NORMAL_PRIORITY_CLASS)
    parent_bytes=sum(path.stat().st_size for path in c.PARENT.rglob('*') if path.is_file())
    c.save(c.OUT/'watchdog_claim.json',dict(time=c.now(),monotonic_origin=start,unix_origin=wall,
        freeze_sha256=args.freeze_sha256,watchdog_pid=me.pid,watchdog_created=me.create_time(),parent_output_bytes=parent_bytes))
    process=job=stdout=stderr=None;identity=None;status='failed'
    try:
        c.require(c.artifact(c.OUT/'freeze.json')['sha256']==args.freeze_sha256,'Recovery freeze changed')
        freeze=c.read(c.OUT/'freeze.json')
        c.require(freeze['job_seconds']==c.JOB_SECONDS and freeze['automatic_retries']==0,'Fixed clock differs')
        c.require(freeze['shared_monotonic_origin']==start and freeze['shared_unix_origin']==wall,'Shared recovery clock changed')
        c.require(parent_bytes==sum(row['bytes'] for row in freeze['first_recovery_preservation']),'Parent output baseline differs')
        c.verify(freeze['first_recovery_preservation'],lambda:c.require(remaining(start,wall)>0,'Shared startup deadline'))
        c.verify(freeze['inputs'],lambda:c.require(remaining(start,wall)>0,'Watchdog startup deadline'))
        original=c.read(c.ORIGINAL/'freeze.json')
        c.require(c.artifact(c.ORIGINAL/'freeze.json')['sha256']==c.ORIGINAL_SHA,'Original freeze changed')
        clock_item=next(row for row in original['inputs'] if row['path']=='tools/local_player_qualification_128/clock.py')
        c.verify([clock_item])
        from tools.local_player_qualification_128.clock import _WindowsJob
        resource_guard(None,parent_bytes)
        stdout=(c.OUT/'auditor_stdout.log').open('xb');stderr=(c.OUT/'auditor_stderr.log').open('xb')
        environment=os.environ.copy();environment.update(PYTHONPATH=str(c.ROOT/'src')+os.pathsep+str(c.ROOT),
            PYTHONDONTWRITEBYTECODE='1',OMP_NUM_THREADS='1',MKL_NUM_THREADS='1',OPENBLAS_NUM_THREADS='1',CUDA_VISIBLE_DEVICES='')
        c.require(remaining(start,wall)>0,'Watchdog deadline before launch')
        process=subprocess.Popen([sys.executable,'-B','-u','-m','tools.splendorust_reliability_recovery_v2_20261026.runner',
            '--freeze-sha256',args.freeze_sha256],cwd=c.ROOT,env=environment,stdout=stdout,stderr=stderr,
            creationflags=AUDITOR_CREATION_FLAGS)
        identity=dict(pid=process.pid,created=psutil.Process(process.pid).create_time());job=_WindowsJob(process.pid)
        c.save(c.OUT/'watchdog.json',dict(time=c.now(),auditor=identity,watchdog=dict(pid=me.pid,created=me.create_time()),
            cpus=me.cpu_affinity(),freeze_sha256=args.freeze_sha256,job_seconds=c.JOB_SECONDS))
        last=-1.
        while process.poll() is None:
            if remaining(start,wall)<=0:status='watchdog_deadline';break
            stamp=time.monotonic()
            if stamp-last>=.8:resource_guard(process,parent_bytes);last=stamp
            time.sleep(.1)
        if process.poll() is not None:status='completed' if process.returncode==0 else 'auditor_failed'
    except BaseException:
        c.save(c.OUT/'watchdog_error.json',dict(time=c.now(),error=traceback.format_exc()))
        raise
    finally:
        receipt=cleanup(process,job,(stdout,stderr),start,wall,identity)
        charged=max(time.monotonic()-start,time.time()-wall)
        if charged>=c.JOB_SECONDS:status='watchdog_deadline'
        c.save(c.OUT/'execution.json',dict(time=c.now(),status='cleanup_failed' if receipt['errors'] else status,
            exit_code=None if process is None else process.returncode,freeze_sha256=args.freeze_sha256,
            charged_job_seconds=charged,monotonic_origin=start,unix_origin=wall,monotonic_end=time.monotonic(),unix_end=time.time(),
            cleanup_errors=receipt['errors'],cleanup_receipt=receipt,original_status='incomplete',automatic_retries=0,bootstrap_relaunches=1,model_or_decision_retries=0))

if __name__=='__main__':main()
