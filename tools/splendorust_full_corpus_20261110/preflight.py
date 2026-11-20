"""Source-bound model-free test runner; never launches scientific work."""
import argparse
import subprocess
import sys
from . import common as c


def suite():
    import pytest
    class Counts:
        collected=passed=failed=skipped=0
        def pytest_collection_finish(self,session): self.collected=len(session.items)
        def pytest_runtest_logreport(self,report):
            if report.failed:self.failed+=1
            if report.skipped:self.skipped+=1
            if report.when=='call' and report.passed:self.passed+=1
    counts=Counts()
    paths=sorted((c.ROOT/'tests').glob('*full_corpus*20261110*.py'))
    result=pytest.main(['-q',*[str(p) for p in paths]],plugins=[counts])
    no_tensor='torch' not in sys.modules
    print(c.canonical(dict(tests_collected=counts.collected,tests_passed=counts.passed,
        failures=counts.failed,skipped=counts.skipped,
        tensor_runtime_loaded=not no_tensor,models=0,games=0)),flush=True)
    return 0 if result==0 and no_tensor and counts.passed==counts.collected>=100 and not counts.skipped else 1


def record(name):
    from .owner import child_environment,source_inputs
    c.require(name.isalnum(),'Simple exclusive preflight identity')
    folder=c.OUT/'prelaunch';folder.mkdir(parents=True,exist_ok=True)
    inputs=source_inputs();log=folder/(name+'.log')
    env=child_environment(0);env['PYTEST_DISABLE_PLUGIN_AUTOLOAD']='1'
    with log.open('xb') as stream:
        process=subprocess.run([sys.executable,'-B','-u','-m',c.PACKAGE+'.preflight','suite'],
            cwd=c.ROOT,env=env,stdout=stream,stderr=subprocess.STDOUT,timeout=600)
    c.verify(inputs)
    import json
    summary=json.loads(log.read_text(encoding='utf-8').splitlines()[-1])
    proof=c.save(folder/(name+'.json'),dict(schema='full_corpus_complete_pytest_v1',
        exit_code=process.returncode,inputs=inputs,runner=c.artifact(__file__),full_suite=True,
        suite_summary=summary,log=c.artifact(log),models=0,games=0,time=c.now(),
        model_free_assertion='Torch absent after test suite'))
    print(c.canonical(proof),flush=True)
    return process.returncode


if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('mode',choices=('suite','record'))
    parser.add_argument('--name',default='tests01');args=parser.parse_args()
    raise SystemExit(suite() if args.mode=='suite' else record(args.name))
