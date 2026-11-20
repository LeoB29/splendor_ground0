"""Model-free root safety, clock, accounting and actual sequence counterexamples."""
from pathlib import Path
import pytest
from . import common as c,owner,runtime,preparation,execution

@pytest.fixture
def workspace(tmp_path,monkeypatch):
 monkeypatch.setattr(c,'OUT',tmp_path/'out');monkeypatch.setattr(c,'PREP',tmp_path/'prep')
 monkeypatch.setattr(c,'PLAN',c.OUT/'freeze.json')
 return tmp_path

def test_preparation_is_not_launch_authorization(workspace):
 plan=dict(design=preparation.design(),**c.IDENTITY)
 with pytest.raises(ValueError,match='Later explicit'):owner.launch_admission(plan,{})

@pytest.mark.parametrize('key,value',[('worlds',6),('simulations',256),('precision','float16')])
def test_changed_science_rejected_before_any_launch(workspace,key,value):
 plan=dict(design=preparation.design(),**c.IDENTITY);plan['design'][key]=value
 with pytest.raises(ValueError,match='Unchanged engineering'):owner.launch_admission(plan,{})

def test_output_cannot_escape_new_package(workspace):
 with pytest.raises(ValueError,match='Exclusive new pilot output'):c.save(workspace/'outside.json',{})
 with pytest.raises(ValueError,match='Exclusive new pilot output'):c.save(c.OUT/'..'/'outside.json',{})

def test_exclusive_failures_cannot_be_overwritten(workspace):
 c.save(c.OUT/'failure.json',{'failure':'preserve'})
 with pytest.raises(FileExistsError):c.save(c.OUT/'failure.json',{'failure':'replace'})
 assert c.read(c.OUT/'failure.json')['failure']=='preserve'

@pytest.mark.parametrize('arm,path',[('A1','live/phase0/lane0/worker'),('A4','live/phase2/lane3/worker'),
 ('B-control','collection/cells/control_X/players/p0/worker'),('B-candidate','collection/cells/candidate_Y/replays/worker')])
def test_resource_sampling_groups_overlap_under_same_arm(workspace,arm,path):
 assert owner.metric_arm(c.OUT/path)==arm

def phases(slow=1.):
 return [dict(result=dict(arm=arm,elapsed_seconds=duration,games=[{}]*8,
  audits=[dict(timings=[slow]*8,decisions=8)])) for arm,duration in [('A1',100),('A4',40),('A4',40),('A1',100)]]

def test_live_bar_counts_checks_and_exact_maximum():
 result=execution.live_metrics(phases(),20)
 assert result['A1']['verified_seconds']==210 and result['A4']['verified_seconds']==90
 assert result['adopt'] is True
 assert execution.live_metrics(phases(60.),0)['adopt'] is False

def test_live_bar_not_met_is_valid_negative_not_exception():
 result=execution.live_metrics(phases(),400)
 assert result['adopt'] is False

@pytest.mark.parametrize('expired',('monotonic','wall'))
def test_original_both_clocks_enforced(monkeypatch,expired):
 monkeypatch.setattr(c.time,'monotonic',lambda:21 if expired=='monotonic' else 11)
 monkeypatch.setattr(c.time,'time',lambda:121 if expired=='wall' else 111)
 with pytest.raises(ValueError):c.check_deadlines(dict(monotonic_origin=10,monotonic_deadline=20,unix_origin=110,unix_deadline=120))

def test_no_unapproved_worker_module():
 assert runtime.MODULES=={c.PACKAGE+'.live.worker',c.PACKAGE+'.collection.worker',c.PACKAGE+'.fixture',c.PACKAGE+'.acceptance'}

def test_limits_and_deadlines_remain_brief_exact():
 assert (c.TOTAL_SECONDS,c.MODEL_SECONDS,c.PROCESS_RSS,c.AGGREGATE_RSS,c.MAX_OUTPUT,c.MIN_DISK)==(21600,16200,2*c.GiB,12*c.GiB,10*c.GiB,5*c.GiB)
 assert preparation.design()['live_guard']==[119.,120.]
 assert preparation.design()['collection_guard']==[60.,75.]

def test_root_inherits_original_launcher_clock_without_reset(monkeypatch):
 monkeypatch.setattr(c,'identity',lambda:dict(pid=123,created=456))
 monkeypatch.setattr(c,'exact_alive',lambda i:True)
 release=dict(identity=c.identity(),parent=dict(pid=111,created=222),windows_job_assigned=True,
  plan={'frozen':True},monotonic_origin=1.,unix_origin=2.)
 value=owner.claim_from_release(release,{'frozen':True})
 assert value['monotonic_origin']==1. and value['unix_origin']==2.
 release['plan']={'wrong':True}
 with pytest.raises(ValueError):owner.claim_from_release(release,{'frozen':True})

def test_normal_exit_during_owned_snapshot_is_not_failure(monkeypatch):
 def exited(pid):raise owner.psutil.NoSuchProcess(pid)
 monkeypatch.setattr(owner.psutil,'Process',exited)
 assert owner.owned_tree(dict(pid=123,created=456))==[]

@pytest.mark.parametrize('encoded',[b'123456789',b'"000123456789"',b'"\\u0031\\u00323456789"',
 '"١٢٣٤٥٦٧٨٩"'.encode('utf-8'),b'"\\u0661\\u0662\\u0663\\u0664\\u0665\\u0666\\u0667\\u0668\\u0669"'])
def test_streamed_reservation_scan_matches_semantic_seed_check(workspace,monkeypatch,encoded):
 from . import schedule
 monkeypatch.setattr(c,'ROOT',workspace)
 path=workspace/'reservation.json';path.write_bytes(b'{"seeds":['+encoded+b']}')
 found,item,fallback=schedule.scan_registered(path,{123456789},block_bytes=3)
 assert found and item['sha256']==c.sha(path)
 assert 123456789 in set(schedule.seeds_in(c.read(path)))

def test_streamed_scan_rejects_changed_registered_evidence(workspace,monkeypatch):
 from . import schedule
 monkeypatch.setattr(c,'ROOT',workspace);path=workspace/'reservation.json';path.write_text('{"seed":123}',encoding='utf-8')
 calls=0
 def mutate():
  nonlocal calls
  calls+=1
  if calls==2:path.write_text('{"seed":1234}',encoding='utf-8')
 with pytest.raises(ValueError,match='changed while scanning'):schedule.scan_registered(path,{999},mutate,block_bytes=3)

@pytest.mark.parametrize('encoding',['utf-16-le','utf-32-le'])
def test_streamed_scan_decodes_wide_json_even_without_bom(workspace,monkeypatch,encoding):
 from . import schedule
 monkeypatch.setattr(c,'ROOT',workspace);path=workspace/'reservation.json'
 path.write_bytes('{"seed":"123456789"}'.encode(encoding))
 found,item,fallback=schedule.scan_registered(path,{123456789},block_bytes=3)
 assert found and fallback and item['sha256']==c.sha(path)

def test_concurrent_boundary_keeps_one_status_writer(workspace,monkeypatch):
 from concurrent.futures import ThreadPoolExecutor
 from types import SimpleNamespace
 import threading
 gate=threading.Barrier(8);writes=[]
 monkeypatch.setattr(c,'check_deadlines',lambda x:None)
 monkeypatch.setattr(c,'exact_alive',lambda i:gate.wait() is not None)
 monkeypatch.setattr(c,'identity',lambda:dict(pid=1,created=2))
 monkeypatch.setattr(runtime.shutil,'disk_usage',lambda p:SimpleNamespace(free=10))
 monkeypatch.setattr(runtime.psutil,'Process',lambda:SimpleNamespace(memory_info=lambda:SimpleNamespace(rss=1)))
 monkeypatch.setattr(c,'status',lambda *a:writes.append(a))
 b=runtime.Boundary(dict(limits=dict(disk_floor=1,process_rss=10)),dict(identity={}),dict(request=dict(mode='fixture')),c.OUT)
 with ThreadPoolExecutor(max_workers=8) as pool:list(pool.map(lambda _:b(),range(8)))
 assert len(writes)==1

def test_first_matched_identity_failure_stops_before_later_games(workspace,monkeypatch):
 from .live import scheduler,audit
 from .collection import launch
 events=[];c.save(c.OUT/'claim.json',{'identity':{}});c.save(c.PLAN,{'prepared':True})
 class Jobs:
  receipts=[]
  def guard(self,model):pass
  def launch_job(self,*args,**kwargs):events.append('qualification');return {}
 def stage(folder,plan,root,boundary,specs,arm,phase,job):
  events.append(('phase',phase))
  return dict(result=dict(arm=arm,games=[dict(game={}) for _ in specs]))
 def mismatch(*args):events.append('identity');raise ValueError('mismatch')
 monkeypatch.setattr(scheduler,'run_stage',stage);monkeypatch.setattr(audit,'compare_subset',mismatch)
 monkeypatch.setattr(launch,'qualify',lambda *a: {})
 monkeypatch.setattr(launch,'run',lambda *a: events.append('must_not_collect'))
 with pytest.raises(ValueError,match='mismatch'):
  execution.execute(dict(placement=dict(model_cpus=[0,2,4,6],controller_cpus=[12],original_allowed=list(range(20)))),Jobs())
 assert ('phase',2) not in events and 'must_not_collect' not in events
 assert events[-1]=='identity'
