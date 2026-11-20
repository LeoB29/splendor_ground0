"""Detached saved-evidence recovery; no model worker can be launched."""
import argparse
import sys
import time
from . import common as c
from . import control


def main():
    control.set_own_cpu()
    p=argparse.ArgumentParser();group=p.add_mutually_exclusive_group(required=True)
    group.add_argument('--freeze-sha256');group.add_argument('--fixture-label');a=p.parse_args()
    if a.fixture_label:
        c.require(a.fixture_label.isalnum(),'Simple exclusive fixture label')
        mono,wall=time.monotonic(),time.time()
        limits=dict(monotonic_origin=mono,unix_origin=wall,monotonic_deadline=mono+30,unix_deadline=wall+30)
        c.save(c.OUT/'prelaunch'/('fixture_launcher_'+a.fixture_label+'.json'),dict(identity=c.identity(),time=c.now()))
        control.launch([sys.executable,'-B','-c','import time; time.sleep(2)'],
            c.OUT/'prelaunch'/('detached_fixture_'+a.fixture_label),limits,
            dict(role='model_free_fixture',models=0,new_games=0,
                source_inputs=[c.artifact(__file__),c.artifact(control.__file__),c.artifact(c.__file__)]),rss=256*1024**2)
        return 0
    c.require(c.sha(c.PLAN)==a.freeze_sha256,'Exact recovery plan before launch')
    c.require(tuple(c.JOBS)==('analysis','acceptance') and not c.MODEL_JOBS,'No model recovery route')
    limits=c.recovery_limits();control.bounds(limits)
    c.save(c.OUT/'launcher_claim.json',dict(identity=c.identity(),limits=limits,freeze=c.artifact(c.PLAN),
        original_attempt_preserved=True,new_models=0,new_games=0,time=c.now()))
    control.launch([sys.executable,'-B','-u','-m',c.PACKAGE+'.owner','run','--freeze-sha256',a.freeze_sha256],
        c.OUT/'launch',limits,dict(role='root',freeze_sha256=a.freeze_sha256),rss=c.AUDIT_RSS)
    return 0


if __name__=='__main__':raise SystemExit(main())
