"""Retain the launcher's original process handle and record its actual exit."""
import argparse,subprocess,sys,time,traceback
import psutil
from . import common as c,control,placement

def main():
    placement.set_own(c.SUPERVISOR_CPU)
    p=argparse.ArgumentParser();g=p.add_mutually_exclusive_group(required=True)
    g.add_argument('--probe-label');g.add_argument('--freeze-sha256');a=p.parse_args()
    mode=['--probe-label',a.probe_label] if a.probe_label else ['--freeze-sha256',a.freeze_sha256]
    folder=c.OUT/'prelaunch'/('observer_'+a.probe_label) if a.probe_label else c.OUT/'observer'
    c.save(folder/'claim.json',dict(identity=c.identity(),time=c.now(),mode=mode))
    proc=None;identity=None;error=None;code=None
    with (folder/'stdout.log').open('xb') as out,(folder/'stderr.log').open('xb') as err:
        try:
            proc=subprocess.Popen([sys.executable,'-B','-u','-m',c.PACKAGE+'.launcher',*mode],cwd=c.ROOT,
                env=control.environment(),stdout=out,stderr=err,creationflags=subprocess.CREATE_NO_WINDOW)
            identity=dict(pid=proc.pid,created=psutil.Process(proc.pid).create_time())
            c.save(folder/'launcher.json',dict(identity=identity,time=c.now()))
            code=proc.wait(timeout=120 if a.probe_label else c.TOTAL_SECONDS+120)
        except BaseException:
            error=traceback.format_exc()
            if proc is not None and proc.poll() is None:
                from tools.splendorust_collection4096_resume_20261118_v2.ownership import finish_exact_process
                finish_exact_process(proc,identity)
    absent=identity is not None and not c.exact_alive(identity)
    if a.probe_label and code==0 and error is None:
        probe=c.read(c.OUT/'prelaunch'/('probe_'+a.probe_label)/'result.json')
        c.save(c.OUT/'prelaunch'/('probe_observed_'+a.probe_label+'.json'),dict(**probe,
            probe_owner_exit=code,probe_owner_identity=identity,actual_exit_observed=True,
            exact_absence=absent,observer_identity=c.identity(),time=c.now()))
    elif a.probe_label:
        c.save(folder/'failure.json',dict(status='failed',launcher_identity=identity,exit_code=code,error=error,
            exact_absence=absent,observer_identity=c.identity(),time=c.now()))
    else:
        observed=c.OUT/'observed_owner_exit.json'
        passed=code==0 and error is None and absent and observed.is_file()
        c.save(c.OUT/'observed_launcher_exit.json',dict(status='passed' if passed else 'failed',
            launcher_identity=identity,actual_exit_observed=code is not None,exit_code=code,
            exact_absence=absent,error=error,observer_identity=c.identity(),time=c.now(),
            observed_owner_exit=c.artifact(observed) if observed.is_file() else None))
    return 0 if code==0 and error is None and absent else 1

if __name__=='__main__':raise SystemExit(main())
