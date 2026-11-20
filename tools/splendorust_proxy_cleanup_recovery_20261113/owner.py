"""Two automatic saved-evidence stages under unchanged original outer clocks."""
import argparse
import sys
import traceback
from . import common as c
from . import control


def run(expected):
    control.set_own_cpu()
    owner=control.handshake(c.OUT/'launch',expected,'root');bounds=owner['limits']
    control.bounds(bounds);c.require(c.sha(c.PLAN)==expected,'Exact frozen recovery plan')
    c.require(tuple(c.JOBS)==('analysis','acceptance') and not c.MODEL_JOBS,'No model or decision retry route')
    maximum=c.recovery_limits()
    for key in ('monotonic_deadline','unix_deadline'):
        c.require(bounds[key]<=maximum[key],'Recovery cannot extend original discovery/global deadlines')
    plan=c.read(c.PLAN);c.require(plan['design']==c.design(),'Unchanged recovery design')
    c.verify(plan['worker_inputs']);c.verify(plan['data_inputs'])
    c.save(c.OUT/'claim.json',dict(identity=c.identity(),freeze=c.artifact(c.PLAN),limits=bounds,
        original_attempt_preserved=True,new_models=0,new_games=0,time=c.now()))
    receipts=[];error=None
    try:
        for role in c.JOBS:
            c.status(c.OUT/'stage.json',dict(time=c.now(),stage=role,status='running',completed_jobs=len(receipts)))
            limits=control.stage_limits(bounds,role)
            folder=c.OUT/'jobs'/role
            command=[sys.executable,'-B','-u','-m',c.PACKAGE+'.worker','--role',role,'--freeze-sha256',expected]
            item=control.launch(command,folder,limits,dict(role=role,freeze_sha256=expected),rss=c.AUDIT_RSS)
            actual=c.read(c.resolve(item));result_path=folder/'worker/result.json';result=c.read(result_path)
            c.require(actual['status']=='passed' and actual['exit_code']==0 and actual['result_artifact']==c.artifact(result_path)
                and actual['exact_owned_absence'] and not actual['cleanup_errors'],'Exact clean saved-evidence worker exit')
            c.require(result['status']=='passed' and result['role']==role and result['error'] is None
                and result['freeze_sha256']==expected and result['identity']==actual['identity'],'Exact passed worker result')
            receipts.append(item)
        c.verify(plan['worker_inputs']);c.verify(plan['data_inputs']);control.bounds(bounds)
    except BaseException:error=traceback.format_exc()
    execution=c.save(c.OUT/'execution.json',dict(status='passed' if error is None else 'failed',
        freeze_sha256=expected,receipts=receipts,error=error,time=c.now(),new_models=0,new_games=0))
    if error:
        c.save(c.OUT/'failure.json',dict(status='incomplete_recovery',execution=execution,error=error))
        c.status(c.OUT/'stage.json',dict(status='failed',stage='failed',time=c.now()));return 1
    c.save(c.OUT/'completion.json',dict(status='recovered_verification_pending_actual_owner_exit',execution=execution,
        acceptance=c.artifact(c.OUT/'jobs/acceptance/worker/result.json'),original_attempt_preserved=True,
        new_models=0,new_games=0,automatic_followup=False))
    c.status(c.OUT/'stage.json',dict(status='recovered_verification_pending_actual_owner_exit',stage='complete',time=c.now()))
    return 0


def main():
    control.set_own_cpu()
    p=argparse.ArgumentParser();p.add_argument('mode',choices=('run',));p.add_argument('--freeze-sha256',required=True);a=p.parse_args()
    try:return run(a.freeze_sha256)
    except BaseException:
        error=traceback.format_exc()
        if not(c.OUT/'failure.json').exists():
            c.save(c.OUT/'failure.json',dict(status='incomplete_recovery',stage='root_bootstrap',error=error,time=c.now()))
        print(error,flush=True);return 1


if __name__=='__main__':raise SystemExit(main())
