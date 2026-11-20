"""Source-bound model-free tests, without loading a tensor runtime."""
import argparse,sys
from . import common as c,owner,placement

class Receipt:
    def __init__(self):self.collected=0;self.passed=0;self.failed=0;self.skipped=0
    def pytest_collection_finish(self,session):self.collected=len(session.items)
    def pytest_runtest_logreport(self,report):
        if report.failed:self.failed+=1
        if report.skipped:self.skipped+=1
        if report.when=='call' and report.passed:self.passed+=1

def main():
    placement.set_own(c.SUPERVISOR_CPU)
    p=argparse.ArgumentParser();p.add_argument('--label',required=True);a=p.parse_args()
    c.require(a.label.isalnum(),'Simple exclusive test label')
    inputs=owner.source_inputs()
    import pytest
    receipt=Receipt();code=int(pytest.main(['-q','-p','no:cacheprovider',str(c.ROOT/c.PACKAGE.replace('.','/')/'tests')],plugins=[receipt]))
    summary=dict(collected=receipt.collected,passed=receipt.passed,failed=receipt.failed,skipped=receipt.skipped,
        tensor_runtime_loaded='torch' in sys.modules,models=0,games=0)
    c.verify(inputs)
    c.save(c.OUT/'prelaunch'/('tests_'+a.label+'.json'),dict(status='passed' if code==0 else 'failed',
        exit_code=code,summary=summary,inputs=inputs,time=c.now()))
    return code

if __name__=='__main__':raise SystemExit(main())
