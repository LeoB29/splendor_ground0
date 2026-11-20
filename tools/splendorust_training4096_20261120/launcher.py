"""Detached owner with an actual positive/negative native ownership qualification."""
import argparse,sys,time,traceback
from . import common as c,control,placement

def probe(label):
    c.require(label.isalnum(),'Exclusive probe label')
    folder=c.OUT/'prelaunch'/('probe_'+label)
    c.save(folder/'claim.json',dict(identity=c.identity(),time=c.now()))
    for name,code in (('positive',0),('negative',7)):
        mono,wall=time.monotonic(),time.time()
        limits=dict(monotonic_origin=mono,unix_origin=wall,monotonic_deadline=mono+30,unix_deadline=wall+30)
        try:
            control.launch([sys.executable,'-B','-c',f'import time; time.sleep(2); raise SystemExit({code})'],
                folder/name,limits,dict(role='model_free_fixture',models=0,games=0),rss=256*1024**2)
        except ValueError:
            if name!='negative':raise
        receipt=c.read(folder/name/'actual_exit.json')
        from tools.splendorust_collection4096_resume_20261118_v2.ownership import verify_observations
        verify_observations(receipt)
        c.require(receipt['identity'] in receipt['observed_owned_identities'],'Both fixtures have exact native ownership proof')
        c.require(receipt['exit_code']==code and receipt['actual_exit_observed'] and receipt['exact_owned_absence']
            and not receipt['cleanup_errors'] and receipt['error'] is None,'Actual expected probe exit and cleanup')
        c.require(receipt['status']==('passed' if code==0 else 'failed'),'Negative exits cannot pass')
    c.save(folder/'result.json',dict(status='passed',identity=c.identity(),positive_exit=0,negative_exit=7,
        cleanup_passed=True,models=0,games=0,source_inputs=[c.artifact(__file__),c.artifact(control.__file__),c.artifact(c.__file__)]))
    return 0

def main():
    placement.set_own(c.SUPERVISOR_CPU)
    p=argparse.ArgumentParser();group=p.add_mutually_exclusive_group(required=True)
    group.add_argument('--freeze-sha256');group.add_argument('--probe-label');a=p.parse_args()
    if a.probe_label:return probe(a.probe_label)
    c.require(c.sha(c.PLAN)==a.freeze_sha256,'Exact plan before launch')
    mono,wall=time.monotonic(),time.time()
    limits=dict(monotonic_origin=mono,unix_origin=wall,monotonic_deadline=mono+c.TOTAL_SECONDS,unix_deadline=wall+c.TOTAL_SECONDS)
    c.save(c.OUT/'launcher_claim.json',dict(identity=c.identity(),limits=limits,freeze=c.artifact(c.PLAN),time=c.now()))
    try:
        execution=control.launch([sys.executable,'-B','-u','-m',c.PACKAGE+'.owner','run','--freeze-sha256',a.freeze_sha256],
            c.OUT/'launch',limits,dict(role='root',freeze_sha256=a.freeze_sha256),rss=c.AUDIT_RSS+512*1024**2)
        c.save(c.OUT/'observed_owner_exit.json',dict(status='accepted_completed_verified',actual_exit=execution,
            completion=c.artifact(c.OUT/'completion.json'),time=c.now()))
        return 0
    except BaseException:
        c.save(c.OUT/'launcher_failure.json',dict(status='incomplete',error=traceback.format_exc(),time=c.now()));raise

if __name__=='__main__':raise SystemExit(main())
