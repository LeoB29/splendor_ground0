"""Model-free finalizer qualification, with exclusive source-bound receipts."""
import argparse,json,os,subprocess,sys
from . import audit as a


def suite():
 import pytest
 class Counts:
  collected=passed=failed=skipped=0
  def pytest_collection_finish(self,session):self.collected=len(session.items)
  def pytest_runtest_logreport(self,report):
   if report.failed:self.failed+=1
   if report.skipped:self.skipped+=1
   if report.when=='call' and report.passed:self.passed+=1
 counts=Counts();code=pytest.main(['-q',*[str(p) for p in sorted(a.Path(__file__).parent.glob('tests_*.py'))]],plugins=[counts])
 print(a.stop.canonical(dict(tests_collected=counts.collected,tests_passed=counts.passed,failed=counts.failed,
  skipped=counts.skipped,tensor_runtime_loaded='torch' in sys.modules,models=0,new_games=0)),flush=True)
 return 0 if code==0 and counts.passed==counts.collected>=12 and not counts.skipped and 'torch' not in sys.modules else 1


def record(label):
 a.require(label.isalnum(),'Simple exclusive test label')
 sources=a.sources();folder=a.OUT/'prelaunch';folder.mkdir(parents=True,exist_ok=True)
 log=folder/(label+'.log');env=os.environ.copy();env['PYTEST_DISABLE_PLUGIN_AUTOLOAD']='1'
 with log.open('xb') as stream:
  result=subprocess.run([sys.executable,'-B','-u','-m',a.PACKAGE+'.preflight','suite'],cwd=a.ROOT,
   env=env,stdout=stream,stderr=subprocess.STDOUT,timeout=120)
 a.stop.verify(sources);summary=json.loads(log.read_text().splitlines()[-1])
 a.save(folder/(label+'.json'),dict(status='passed' if result.returncode==0 else 'failed',exit_code=result.returncode,
  inputs=sources,log=a.artifact(log),time=a.stop.now(),**summary))
 print(a.stop.canonical(dict(exit_code=result.returncode,**summary)),flush=True)
 return result.returncode


if __name__=='__main__':
 parser=argparse.ArgumentParser();parser.add_argument('mode',choices=('suite','record'));parser.add_argument('--label',default='tests01');args=parser.parse_args()
 raise SystemExit(suite() if args.mode=='suite' else record(args.label))
