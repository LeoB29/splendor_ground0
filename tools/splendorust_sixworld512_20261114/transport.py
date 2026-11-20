"""Original trusted clock; lossless response envelopes and explicit owned lives."""
from copy import deepcopy
from pathlib import Path
import gzip,json,os,sys,time,traceback
import psutil
from . import common as c,control

class Services:
 def __init__(self,folder,plan,owner,boundary,role,max_services):
  self.folder=Path(folder);self.plan=plan;self.owner=owner;self.boundary=boundary;self.role=role
  self.max_services=max_services;self.child=None;self.lives=[];self.active=None
  allowed=psutil.Process().cpu_affinity()
  if 2 in allowed:psutil.Process().cpu_affinity([2])
 def start(self):
  from tools.local_player_qualification_128.clock import PersistentWorker
  from tools.splendorust_teacher_variability_20261113 import ownership,qualification
  self.boundary();c.require(self.child is None,'No duplicate resident model service')
  c.require(len(self.lives)<self.max_services,'Prospective service lifetime bound')
  folder=self.folder/'services'/f'life_{len(self.lives):04d}'
  env=control.environment();env.update(SIXWORLD_SERVICE_FOLDER=str(folder/'service'),
   SIXWORLD_FREEZE_SHA256=c.sha(c.PLAN),SIXWORLD_ROLE=self.role)
  child=PersistentWorker(c.PACKAGE+'.service',folder/'transport',affinity=[0],initialization_timeout=240,
   env=env,max_message_bytes=16*c.MiB,max_log_bytes=8*c.GiB,max_rss_bytes=c.MODEL_RSS,
   absolute_deadline=self.owner['limits']['monotonic_deadline'],resource_guard=self.boundary)
  self.child=child;self.active=dict(folder=folder,child=child,requests=[],initialization=c.artifact(folder/'transport/initialization.json'))
  self.lives.append(self.active)
  initial=child.initialization['result']
  c.require(initial['identity']==child.identity and initial['role']==self.role and initial['freeze_sha256']==c.sha(c.PLAN),
   'Exact frozen service readiness')
  runtime=c.read(folder/'service/variability_runtime_identity.json')
  expected={}
  for name in runtime['native_pids']:
   item=self.plan['control_profiles']['legacy_p3_q16'] if name=='control' else self.plan['research_profiles']['legacy_p3_q16' if name=='reference' else 'legacy_p3_q512']
   doc=c.read(c.resolve(item));expected[name]=c.artifact(c.ROOT/doc['native_binary'])
   c.require(expected[name]['sha256']==doc['native_binary_sha256'],'Exact original native role')
  startup=ownership.inspect_service_tree(child.identity,runtime['native_pids'],expected,self.plan['console_executable'])
  self.active.update(runtime=runtime,expected=expected,startup=startup)
  c.save(folder/'native_startup.json',dict(identity=child.identity,rows=startup,runtime=c.artifact(folder/'service/variability_runtime_identity.json')))
  fixtures=c.read(c.resolve(self.plan['graph_parity_plan']))['fixtures']
  result=self.request(dict(operation='graph_parity',fixtures=fixtures),300,start=False)
  c.require(result['clock']['status']=='ok' and result['clock']['result']['passed'],'Fresh exact graph admission')
  qualification.verify_graph_evidence(folder/'service/graph_parity.json',fixtures)
  qualified=c.read(c.PRIOR_OUT/'jobs/qualification/worker/service/variability_runtime_identity.json')['graph_identity']
  for key in ('checkpoint_sha256','weights_before_sha256','weights_after_sha256','torch_version','numpy_version','backend','options'):
   c.require(runtime['graph_identity'][key]==qualified[key],'Exact original graph identity '+key)
  return folder/'service'
 @property
 def service_folder(self):
  c.require(self.active is not None,'An admitted service is required')
  return self.active['folder']/'service'
 def request(self,payload,timeout=120,*,start=True):
  from .adapters import unpack_result
  self.boundary()
  if self.child is None:
   c.require(start,'No nested service initialization');self.start()
  child=self.child;folder=self.active['folder']
  c.require(self.owner['limits']['monotonic_deadline']-time.monotonic()>timeout+1,'Full original request allowance remains')
  request_id=f'request-{child._counter+1:08d}'
  # This durable input/identity receipt precedes the trusted clock's dispatch.
  input_item=c.save_gzip(folder/'inputs'/f'{request_id}.json.gz',dict(payload=payload,request_id=request_id,
   identity=child.identity,freeze_sha256=c.sha(c.PLAN),time=c.now()))
  raw=child.request(payload,timeout)
  source=c.artifact(folder/'transport'/f'{request_id}.json')
  c.require(raw['request_id']==request_id,'Contiguous original request sequence')
  decoded=deepcopy(raw)
  if raw['status']=='ok':decoded['result']=unpack_result(raw['result'])
  item=c.save(folder/'links'/f'{request_id}.json',dict(source=source,input=input_item,
   decoded_sha256=c.digest(decoded),result_sha256=None if decoded['result'] is None else c.digest(decoded['result']),
   identity=child.identity,request_id=request_id,timeout=timeout))
  self.active['requests'].append(item)
  result=dict(clock=decoded,source=source,input=input_item,link=item)
  timeout_path=folder/'service/decision_timeout.json'
  if timeout_path.exists():result['timeout_evidence']=c.artifact(timeout_path)
  if raw['status']!='ok':
   # Closure never retries the request. A subsequent game may get a fresh life.
   self.close(failed_clock=source)
  return result
 def decision(self,observer,pool,*,context):
  return self.request(dict(player='search',observer=observer,quota=512,pool=pool,mode='legacy',
   world_seed_label='A',encoding_seed_label='A',telemetry=True),120)
 def close(self,failed_clock=None):
  if self.child is None:return None
  from tools.splendorust_teacher_variability_20261113 import ownership
  child=self.child;live=self.active;folder=live['folder'];error=None;book=None;code=None
  owned=[dict(pid=r['pid'],created=r['created']) for r in live.get('startup',[])]
  try:
   if failed_clock is None:
    end=ownership.inspect_service_tree(child.identity,live['runtime']['native_pids'],live['expected'],self.plan['console_executable'])
    book=ownership.make_book(child.identity,live['runtime']['native_pids'],live['startup'],end,live['expected'],self.plan['console_executable'])
    child.process.stdin.close();code=child.process.wait(timeout=20)
    exit_receipt=c.read(folder/'service/service_exit.json')
    c.require(code==0 and exit_receipt['status']=='passed' and exit_receipt['error'] is None,'Clean service exit0')
   else:code=child.process.poll()
  except BaseException:error=traceback.format_exc()
  finally:
   child.close()
   if code is None:code=child.process.poll()
  grace=control.observe_absence([child.identity,*owned],self.owner['limits'])
  item=c.save(folder/'actual_exit.json',dict(status='passed' if error is None and failed_clock is None else 'closed_after_failed_request',
   identity=child.identity,exit_code=code,exact_owned_absence=grace['exact_owned_absence'],cleanup_grace=grace,
   error=error,failed_clock=failed_clock,book=book,owned_identities=owned,requests=live['requests'],
   initialization=live['initialization'],runtime=c.artifact(folder/'service/variability_runtime_identity.json'),role=self.role,time=c.now()))
  live['actual_exit']=item;self.child=None;self.active=None
  c.require(error is None and grace['exact_owned_absence'],'Service exit/cleanup failed; preserved')
  return item
 def finish(self):
  self.close()
  return [life['actual_exit'] for life in self.lives]

def verify_services(receipts,plan,boundary=lambda:None):
 """Rebind every raw packed response to decoded evidence and native ownership."""
 from .adapters import unpack_result,LIVE_SCHEMA,PACK_SCHEMA
 from .qualification import verify_service_forensics
 from .verifier import classify_deadline
 from tools.splendorust_teacher_variability_20261113 import ownership,qualification
 proofs=[]
 for item in receipts:
  boundary();c.verify([item],boundary);exit=c.read(c.resolve(item));folder=c.resolve(item).parent
  c.require(exit['error'] is None and exit['exact_owned_absence'],'Actual service cleanup')
  c.require(all(not c.exact_alive(i) for i in [exit['identity'],*exit['owned_identities']]),'All exact owned identities absent')
  c.verify([exit['runtime'],exit['initialization']],boundary)
  c.require(c.resolve(exit['runtime'])==folder/'service/variability_runtime_identity.json' and
   c.resolve(exit['initialization'])==folder/'transport/initialization.json','Canonical service lifecycle artifacts')
  readiness=c.read(c.resolve(exit['initialization']))['result']
  c.require(readiness==dict(identity=exit['identity'],role=exit['role'],freeze_sha256=c.sha(c.PLAN),
   live_schema=LIVE_SCHEMA,result_encoding=PACK_SCHEMA),'Exact ready owner, frozen role and encoding')
  runtime=c.read(c.resolve(exit['runtime']));expected={}
  for name in ('control','reference','search'):
   profile=plan['control_profiles']['legacy_p3_q16'] if name=='control' else plan['research_profiles']['legacy_p3_q16' if name=='reference' else 'legacy_p3_q512']
   doc=c.read(c.resolve(profile));expected[name]=c.artifact(c.ROOT/doc['native_binary'])
   c.require(expected[name]['sha256']==doc['native_binary_sha256'],'Frozen native executable')
  startup=c.read(folder/'native_startup.json')
  ownership.validate_snapshot(exit['identity'],runtime['native_pids'],startup['rows'],expected,plan['console_executable'])
  c.require({(i['pid'],i['created']) for i in exit['owned_identities']}==
   {(i['pid'],i['created']) for i in startup['rows']},'Complete owned absence inventory')
  if exit['failed_clock'] is None:
   c.require(exit['status']=='passed' and exit['exit_code']==0,'Actual service exit0')
   book=exit['book'];ownership.validate_book(book,expected,plan['console_executable'])
   c.require(book['service']==exit['identity'] and book['startup']==startup['rows'],'Bound actual service ownership')
  else:
   c.verify([exit['failed_clock']],boundary)
   failed=c.read(c.resolve(exit['failed_clock']))
   timeout=folder/'service/decision_timeout.json'
   classify_deadline(failed,source=exit['failed_clock'],timeout_evidence=c.artifact(timeout) if timeout.exists() else None)
  sources=c.read(folder/'service/runtime_sources.json')
  c.require(sources['sources']==plan['corrected_source_receipt'],'Actual corrected service guard')
  c.verify([exit['runtime'],exit['initialization']],boundary)
  fixtures=c.read(c.resolve(plan['graph_parity_plan']))['fixtures']
  qualification.verify_graph_evidence(folder/'service/graph_parity.json',fixtures)
  indexed={};accepted={};failed_request=None
  for ordinal,link_item in enumerate(exit['requests'],1):
   boundary();c.verify([link_item],boundary);link=c.read(c.resolve(link_item));c.verify([link['source'],link['input']],boundary)
   raw=c.read(c.resolve(link['source']));inp=c.read(c.resolve(link['input']));request=f'request-{ordinal:08d}'
   c.require(c.resolve(link_item)==folder/'links'/f'{request}.json' and
    c.resolve(link['source'])==folder/'transport'/f'{request}.json' and
    c.resolve(link['input'])==folder/'inputs'/f'{request}.json.gz','Canonical life-local raw/input/link custody')
   expected_timeout=300 if ordinal==1 else 6000 if inp['payload'].get('operation')=='entrypoint_checks' else 120
   c.require(link['timeout']==expected_timeout,'Only prospective complete clock allowance')
   qualification.check_clock(raw,limit=expected_timeout)
   c.require(raw['request_id']==link['request_id']==inp['request_id']==request and inp['identity']==exit['identity']
    and inp['freeze_sha256']==c.sha(c.PLAN),'Every input and raw clock binds exact request/owner/freeze')
   decoded=deepcopy(raw)
   if raw['status']=='ok':decoded['result']=unpack_result(raw['result'])
   c.require(c.digest(decoded)==link['decoded_sha256'],'Complete clock/result lossless binding')
   c.require(link['identity']==exit['identity'],'Link owner matches actual process')
   indexed[request]=raw['status']
   if raw['status']=='ok':accepted[request]=c.digest(raw['result'])
   else:
    c.require(ordinal==len(exit['requests']) and link['source']==exit['failed_clock'],'Only final request can fail')
    failed_request=dict(request_id=request,payload=inp['payload'])
  c.require(len(list((folder/'links').glob('*.json')))==len(indexed) and
   len(list((folder/'inputs').glob('*.json.gz')))==len(indexed),'No unaccounted request input/link')
  # The response wire must agree with every accepted packed clock packet.
  packets=set();ready=0;partial_tail=None
  with (folder/'transport/stdout.log').open('rb') as stream:
   while True:
    line=stream.readline(16*c.MiB+1)
    if not line:break
    boundary();c.require(len(line)<=16*c.MiB,'Bounded protocol frame')
    if not line.endswith(b'\n'):
     c.require(failed_request is not None and not stream.read(1),'Only unaccepted failed final wire tail may be partial')
     partial_tail=dict(bytes=len(line),sha256=__import__('hashlib').sha256(line).hexdigest(),
      failed_request_id=failed_request['request_id'],accepted=False);break
    p=json.loads(line,object_pairs_hook=c._unique)
    if p.get('type')=='response':
     request=p['request_id'];c.require(request in indexed and request not in packets,'Only one solicited response')
     packets.add(request)
     if request in accepted:c.require(c.digest(p['result'])==accepted[request],'Exact original packed wire result')
    elif p.get('type')=='ready':
     ready+=1;c.require(p['metadata']==c.read(c.resolve(exit['initialization']))['result'],'Exact saved readiness')
    else:c.require(p.get('type')=='error' and failed_request is not None,'Only explicit failed service emits error')
  c.require(ready==1 and set(accepted)<=packets,'Every accepted packet and exact readiness retained')
  def forensic_expected():
   for link_item in exit['requests']:
    boundary();link=c.read(c.resolve(link_item));raw=c.read(c.resolve(link['source']))
    if raw['status']=='ok':
     inp=c.read(c.resolve(link['input']))
     yield dict(request_id=raw['request_id'],payload=inp['payload'],result=unpack_result(raw['result']))
  forensic=verify_service_forensics(folder/'service',forensic_expected(),boundary=boundary,failed_request=failed_request)
  artifacts=[c.artifact(folder/p) for p in ('native_startup.json','transport/stdout.log','service/runtime_sources.json','service/graph_parity.json')]
  artifacts += [c.artifact(folder/'service'/name) for name in ('forensic_summary.json','service_exit.json','failure.json','decision_timeout.json') if (folder/'service'/name).exists()]
  proofs.append(dict(actual_exit=item,requests=len(indexed),status='passed',forensics=forensic,
   partial_unaccepted_wire_tail=partial_tail,evidence=artifacts))
 return proofs
