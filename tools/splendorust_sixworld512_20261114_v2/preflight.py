"""Complete source-bound model-free tests, with preserved failing receipts."""
import argparse
import subprocess
import sys
from . import common as c

def suite():
    import pytest
    class Counts:
        collected=passed=failed=skipped=0
        def pytest_collection_finish(self,session):self.collected=len(session.items)
        def pytest_runtest_logreport(self,report):
            if report.failed:self.failed+=1
            if report.skipped:self.skipped+=1
            if report.when=='call' and report.passed:self.passed+=1
    counts=Counts()
    paths=list((c.ROOT/c.PACKAGE.replace('.','/')).glob('tests_*.py'))
    code=pytest.main(['-q',*[str(p) for p in sorted(paths)]],plugins=[counts])
    summary=dict(collected=counts.collected,passed=counts.passed,failed=counts.failed,skipped=counts.skipped,
        tensor_runtime_loaded='torch' in sys.modules,models=0,new_games=0)
    print(c.canonical(summary),flush=True)
    return 0 if code==0 and counts.passed==counts.collected>=20 and not counts.skipped and 'torch' not in sys.modules else 1

def record(label):
    from .owner import source_inputs
    from .control import environment
    c.require(label.isalnum(),'Simple exclusive test label');inputs=source_inputs()
    folder=c.OUT/'prelaunch';folder.mkdir(parents=True,exist_ok=True);log=folder/(label+'.log')
    error=None;code=None;summary=None
    try:
        with log.open('xb') as stream:
            process=subprocess.run([sys.executable,'-B','-u','-m',c.PACKAGE+'.preflight','suite'],
                cwd=c.ROOT,env=environment(),stdout=stream,stderr=subprocess.STDOUT,timeout=600)
        code=process.returncode;c.verify(inputs)
        summary=__import__('json').loads(log.read_text(encoding='utf-8').splitlines()[-1])
    except BaseException:
        error=__import__('traceback').format_exc()
    item=c.save(folder/(label+'.json'),dict(status='passed' if code==0 and error is None else 'failed',
        exit_code=code,summary=summary,error=error,inputs=inputs,log=c.artifact(log),time=c.now()))
    print(c.canonical(item),flush=True);return 0 if code==0 and error is None else 1

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('mode',choices=('suite','record'));p.add_argument('--label',default='tests01');a=p.parse_args()
    raise SystemExit(suite() if a.mode=='suite' else record(a.label))
