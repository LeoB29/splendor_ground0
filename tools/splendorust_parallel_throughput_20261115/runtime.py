"""Single-use owned workers; original real clocks and no implicit model launch."""
import argparse, importlib, os, shutil, sys, time, traceback, threading
from pathlib import Path
import psutil
from . import common as c

MODULES={c.PACKAGE+'.live.worker',c.PACKAGE+'.collection.worker',c.PACKAGE+'.fixture',c.PACKAGE+'.acceptance'}

class Boundary:
 def __init__(self,plan,claim,owner,folder):
  self.plan,self.claim,self.owner,self.folder=plan,claim,owner,folder
  self.last=-float('inf')
  self._deep_lock=threading.Lock()
 def __call__(self):
  c.check_deadlines(self.owner)
  c.require(c.exact_alive(self.claim['identity']),'Original pilot owner alive')
  c.require(not (c.OUT/'stop.json').exists(),'Pilot fail-stop recorded')
  if time.monotonic()-self.last>=5 and self._deep_lock.acquire(blocking=False):
   try:
    if time.monotonic()-self.last<5:return
    self.last=time.monotonic()
    c.require(shutil.disk_usage(self.folder).free>=self.plan['limits']['disk_floor'],'Actual disk floor')
    c.require(psutil.Process().memory_info().rss<=self.plan['limits']['process_rss'],'Actual individual RSS')
    c.status(self.folder/'boundary/progress.json',dict(time=c.now(),mode=self.owner['request']['mode'],identity=c.identity()))
   finally:self._deep_lock.release()

def bootstrap(folder,plan_path,expected,module):
 folder=Path(folder).resolve();plan_path=Path(plan_path).resolve()
 c._destination(folder/'result.json');c.require(c.sha(plan_path)==expected,'Exact frozen worker plan')
 plan=c.read(plan_path);started=time.monotonic();owner_path=folder.parent/'owner.json'
 while not owner_path.exists():
  c.require(time.monotonic()-started<20,'Bounded ownership handshake');time.sleep(.01)
 owner=c.read(owner_path);claim=c.read(c.resolve(owner['claim']))
 c.require(owner['identity']==c.identity() and owner['module']==module and owner['windows_job_assigned']
  and owner['plan']==c.artifact(plan_path),'Exact worker role and ownership')
 limit=c.MODEL_SECONDS if owner['model_work'] else c.TOTAL_SECONDS
 c.require(owner['monotonic_deadline']==claim['monotonic_origin']+limit
  and owner['unix_deadline']==claim['unix_origin']+limit,'Original unextended dual clock')
 boundary=Boundary(plan,claim,owner,folder);boundary();c.verify(plan['worker_inputs'],boundary)
 return plan,owner,boundary

def main():
 parser=argparse.ArgumentParser();parser.add_argument('--module',required=True,choices=sorted(MODULES))
 parser.add_argument('--output',required=True);parser.add_argument('--plan',required=True)
 parser.add_argument('--plan-sha256',required=True);args=parser.parse_args();folder=Path(args.output)
 error=None;payload=None;status='failed';owner=None
 try:
  plan,owner,boundary=bootstrap(folder,args.plan,args.plan_sha256,args.module)
  payload=importlib.import_module(args.module).run(folder,plan,owner,boundary,owner['request'])
  boundary();status='passed'
 except BaseException:error=traceback.format_exc()
 c.save(folder/'result.json',dict(status=status,identity=c.identity(),module=args.module,
  request=None if owner is None else owner['request'],payload=payload,error=error,time=c.now()))
 if status!='passed':raise SystemExit(1)

if __name__=='__main__':main()
