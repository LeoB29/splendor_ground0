"""Source-bound model-free qualification only; never runs a learned player."""
import argparse,contextlib,sys,time
import xml.etree.ElementTree as ET
from . import common as c
from .preparation import source_inputs

def run(label):
 import pytest
 c.require(label.isalnum(),'Exclusive safe qualification label')
 folder=c.PREP/('qualification_'+label);folder.mkdir(parents=True,exist_ok=False)
 inputs=source_inputs();tests=sorted((c.ROOT/c.PACKAGE.replace('.','/')).rglob('tests_*.py'))
 c.require(tests and 'torch' not in sys.modules,'Complete model-free tests only')
 started=time.monotonic();log=folder/'tests.log';xml=folder/'tests.xml'
 with log.open('x',encoding='utf-8') as stream,contextlib.redirect_stdout(stream),contextlib.redirect_stderr(stream):
  code=pytest.main([*[str(x) for x in tests],'--import-mode=importlib','-q','-p','no:cacheprovider','--junitxml='+str(xml)])
 tree=ET.parse(xml);cases=tree.findall('.//testcase');failed=sum(case.find('failure') is not None or case.find('error') is not None for case in cases)
 skipped=sum(case.find('skipped') is not None for case in cases)
 unchanged=inputs==source_inputs();tensor='torch' in sys.modules
 result=dict(status='passed' if code==0 and failed==skipped==0 and not tensor and unchanged else 'failed',
  exit_code=int(code),collected=len(cases),passed=len(cases)-failed-skipped,failed=failed,skipped=skipped,
  tensor_runtime_loaded=tensor,models=0,games=0,inputs=inputs,unchanged=unchanged,
  log=c.artifact(log),junit=c.artifact(xml),elapsed_seconds=time.monotonic()-started)
 item=c.save(folder/'result.json',result);print(c.canonical(dict(receipt=item,status=result['status'],passed=result['passed'])))
 return 0 if result['status']=='passed' else 1

if __name__=='__main__':
 parser=argparse.ArgumentParser();parser.add_argument('--label',required=True);args=parser.parse_args();raise SystemExit(run(args.label))
