"""Source-bound model-free cancellation regression receipt."""
import argparse,subprocess,sys,json,traceback,os
from . import cancel as c

def suite():
 import pytest
 class Counts:
  collected=passed=failed=skipped=0
  def pytest_collection_finish(self,session):self.collected=len(session.items)
  def pytest_runtest_logreport(self,report):
   if report.failed:self.failed+=1
   if report.skipped:self.skipped+=1
   if report.when=='call' and report.passed:self.passed+=1
 counts=Counts();code=pytest.main(['-q',str(__file__).replace('preflight.py','tests_cancel.py')],plugins=[counts])
 print(c.canonical(dict(collected=counts.collected,passed=counts.passed,failed=counts.failed,skipped=counts.skipped,
  models=0,new_games=0,tensor_runtime_loaded='torch' in sys.modules)),flush=True)
 return 0 if code==0 and counts.passed==counts.collected>=12 and not counts.skipped and 'torch' not in sys.modules else 1

def record(label):
 c.require(label.isalnum(),'Exclusive test label');inputs=c.sources();folder=c.OUT/'prelaunch';folder.mkdir(parents=True,exist_ok=True)
 log=folder/(label+'.log');error=None;code=None;summary={}
 try:
  env=os.environ.copy();env.update(PYTHONDONTWRITEBYTECODE='1',PYTEST_DISABLE_PLUGIN_AUTOLOAD='1')
  with log.open('xb') as stream:
   result=subprocess.run([sys.executable,'-B','-u','-m',c.PACKAGE+'.preflight','suite'],cwd=c.ROOT,
    stdout=stream,stderr=subprocess.STDOUT,env=env,timeout=120)
  code=result.returncode;summary=json.loads(log.read_text(encoding='utf-8').splitlines()[-1]);c.verify(inputs)
 except BaseException:error=traceback.format_exc()
 item=c.save(folder/(label+'.json'),dict(status='passed' if code==0 and error is None else 'failed',exit_code=code,
  inputs=inputs,log=c.artifact(log),error=error,time=c.now(),**summary))
 print(c.canonical(item));return 0 if code==0 and error is None else 1

if __name__=='__main__':
 p=argparse.ArgumentParser();p.add_argument('mode',choices=('suite','record'));p.add_argument('--label',default='tests01');a=p.parse_args()
 raise SystemExit(suite() if a.mode=='suite' else record(a.label))
