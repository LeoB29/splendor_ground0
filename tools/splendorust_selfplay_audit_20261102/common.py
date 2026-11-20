"""Separate saved-only audit bounds; original qualification stays incomplete."""
from pathlib import Path
import os,sys,time,shutil
import psutil
from . import verifier as v
c=v.c
OUT=v.AUDIT_OUT
PROTOCOL=c.ROOT/'docs/engineering/splendorust_selfplay_audit_20261102.md'
SECONDS=900
PRIMARY_SECONDS=540
READER_SECONDS=300
MAX_BYTES=128*c.MiB

def save(name,value):return v.publish_new(name,value)

def inventory(boundary=lambda:None):
    result=[]
    for p in sorted(c.OUT.rglob('*')):
        if p.is_file():boundary();result.append(c.artifact(p))
    return result

def absent(identity):
    try:p=psutil.Process(identity['pid'])
    except psutil.NoSuchProcess:return True
    return p.create_time()!=identity['created']

def check_original():
    c.require(c.file_hash(c.OUT/'freeze.json')==v.ORIGINAL_FREEZE_SHA,'Original freeze unchanged')
    failure=c.read(c.OUT/'failure.json');completion=c.read(c.OUT/'completion.json')
    c.require(failure['stage']=='final' and 'FileExistsError' in failure['error'] and
              'capture_qualification.json' in failure['error'] and completion['status']=='incomplete',
              'Only the preserved final receipt-recreation failure is recoverable')
    c.require(not(c.OUT/'stages/final_end.json').exists() and not(c.OUT/'independent_review.json').exists(),
              'Never relabel original final stage or acceptance')
    old=c.read(c.OUT/'guardian_claim.json');owner=c.read(c.OUT/'guardian_owner.json');launch=c.read(c.OUT/'launch.json')
    identities=[dict(pid=old['guardian_pid'],created=old['guardian_created']),owner['watchdog'],
                dict(pid=launch['pid'],created=launch['created'])]
    c.require(all(absent(x) for x in identities),'Original exact owners absent')
    for name in ('execution.json','watchdog_execution.json'):
        item=c.read(c.OUT/name)
        c.require(item['exit_code']==1 and not item['cleanup_errors'] and not item['cleanup_receipt']['unconfirmed'],
                  'Actual old failed exit and clean ownership closure')
    return old,identities

def deadlines(claim,old,mono,wall,limit):
    return (min(claim['monotonic_deadline'],old['monotonic_origin']+c.JOB_SECONDS,mono+limit),
            min(claim['unix_deadline'],old['unix_origin']+c.JOB_SECONDS,wall+limit))

class Guard:
    def __init__(self,claim,old,limit):
        self.start=old['monotonic_origin'];self.wall_start=old['unix_origin'];self.end=self.start+c.JOB_SECONDS
        self.deadline,self.wall_deadline=deadlines(claim,old,time.monotonic(),time.time(),limit)
        self.last=-float('inf');self.peak_rss=0;self.placement_mode='isolated'
        self.placement_sequence=len(list((c.OUT/'placement_requests').glob('*.json')))
    def remaining(self):return min(self.deadline-time.monotonic(),self.wall_deadline-time.time())
    def effective_deadline(self):return time.monotonic()+self.remaining()
    def guard(self):
        c.require(self.remaining()>0,'New saved-only audit/original global deadline')
        c.require('torch' not in sys.modules,'No model runtime in saved-only audit')
        if time.monotonic()-self.last<1:return
        self.last=time.monotonic();p=psutil.Process();self.peak_rss=max(self.peak_rss,p.memory_info().rss)
        c.require(not p.children(recursive=True),'Saved-only verifier cannot own child/model processes')
        c.require(self.peak_rss<=2*c.GiB and shutil.disk_usage(c.ROOT).free>=5*c.GiB,'Saved reader resource caps')
        c.require(sum(p.stat().st_size for p in OUT.rglob('*') if p.is_file())<MAX_BYTES,'Separate audit evidence cap')
    def close_worker(self,folder,child=None):
        c.require(child is None,'No live model closure in saved audit');self.guard()

