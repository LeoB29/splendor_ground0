"""Durable detached root owner, independent of the interactive tool session."""
import argparse
import sys
import time
from . import common as c
from . import control,clocks

def main():
    p=argparse.ArgumentParser();group=p.add_mutually_exclusive_group(required=True)
    group.add_argument('--freeze-sha256');group.add_argument('--fixture-label');a=p.parse_args()
    if a.fixture_label:
        c.require(a.fixture_label.isalnum(),'Simple exclusive fixture label')
        limits=clocks.fixture_limits(30)
        c.save(c.OUT/'prelaunch'/('fixture_launcher_'+a.fixture_label+'.json'),dict(identity=c.identity(),time=c.now()))
        control.launch([sys.executable,'-B','-c','import time; time.sleep(2)'],
            c.OUT/'prelaunch'/('detached_fixture_'+a.fixture_label),limits,
            dict(role='model_free_fixture',models=0,new_games=0,
                source_inputs=[c.artifact(__file__),c.artifact(control.__file__),c.artifact(c.__file__)]),rss=256*1024**2)
        return 0
    c.require(c.sha(c.PLAN)==a.freeze_sha256,'Exact plan before launch')
    c.require_original_absence()
    c.require(not (c.OUT/'ambiguous_retry_consumed.json').exists(),'A consumed interrupted request cannot be retried again')
    limits=clocks.fresh_global_limits()
    c.save(c.OUT/'launcher_claim.json',dict(identity=c.identity(),limits=limits,freeze=c.artifact(c.PLAN),time=c.now()))
    control.launch([sys.executable,'-B','-u','-m',c.PACKAGE+'.owner','run','--freeze-sha256',a.freeze_sha256],
        c.OUT/'launch',limits,dict(role='root',freeze_sha256=a.freeze_sha256),rss=c.AUDIT_RSS+512*1024**2)
    return 0

if __name__=='__main__':raise SystemExit(main())
