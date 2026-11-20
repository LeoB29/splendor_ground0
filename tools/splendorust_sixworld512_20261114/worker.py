"""Owned qualification, paired play, fresh reproduction and independent audits."""
import argparse,sys,traceback
from pathlib import Path
from . import common as c,control

def passed_job(role,expected):
 folder=c.OUT/'jobs'/role;exit_item=c.artifact(folder/'actual_exit.json');exit=c.read(c.resolve(exit_item))
 c.require(exit['status']=='passed' and exit['exit_code']==0 and exit['actual_exit_observed'] and
  exit['exact_owned_absence'] and exit['freeze_sha256']==expected and not exit['error'] and not exit['cleanup_errors'],
  'Actual passed owned stage '+role)
 c.require(all(not c.exact_alive(i) for i in [exit['identity'],*exit['observed_owned_identities']]),'Exact previous owners absent')
 item=exit['result_artifact'];c.verify([item]);result=c.read(c.resolve(item))
 c.require(result['status']=='passed' and result['error'] is None and result['identity']==exit['identity'] and
  result['role']==role and result['freeze_sha256']==expected,'Exact passed worker result')
 return result['result'],exit_item

def stage_specs(plan,role):
 from . import schedule
 schedules=c.read(c.resolve(plan['schedule']))
 for stage in ('operational','main'):schedule.validate_stage_schedule(schedules[stage],stage)
 if role=='operational':return schedules['operational']
 c.require(role in c.BLOCKS,'Only prospective main blocks')
 offset=c.BLOCKS.index(role)*400;return schedules['main'][offset:offset+400]

def model_stage(folder,plan,owner,boundary,role):
 from . import transport,qualification,arena,verifier
 purpose='qualification' if role=='qualification' else 'operational' if role=='operational' else 'main'
 manager=transport.Services(folder/'live',plan,owner,boundary,purpose,402 if role in c.BLOCKS else 8)
 services=[];replay_services=[]
 try:
  manager.start()
  if role=='qualification':
   result=qualification.run_live_qualification(manager.request,plan,service_folder=manager.service_folder,boundary=boundary)
   services=manager.finish()
   proof=transport.verify_services(services,plan,boundary)
   return dict(qualification=c.save(folder/'qualification.json',result),services=services,
    service_audit=c.save(folder/'services_audit.json',proof))
  specs=stage_specs(plan,role)
  block=arena.run_block(folder/'arena',plan,specs,manager.decision,boundary)
  services=manager.finish()
 finally:
  if manager.child is not None:manager.close()
 selection=block['result']['replay_selection'];replayed=[]
 replay=transport.Services(folder/'replay',plan,owner,boundary,'replay',1)
 try:
  replay.start()
  for row in selection:
   boundary();value=replay.decision(row['observer'],row['pool'],context=dict(replay_id=row['replay_id']))
   replayed.append(dict(replay_id=row['replay_id'],source=value['source']))
   c.save(folder/'replay_rows'/f"{row['replay_id']}.json",replayed[-1])
   c.require(value['clock']['status']=='ok','Exact replay must finish successfully; no retry')
  replay_services=replay.finish()
 finally:
  if replay.child is not None:replay.close()
 proof=verifier.verify_replays(selection,replayed,plan,boundary=boundary)
 service_proof=transport.verify_services(services+replay_services,plan,boundary)
 return dict(block=block['artifact'],replayed=replayed,replay_proof=c.save(folder/'replay_proof.json',proof),
  services=services,replay_services=replay_services,service_audit=c.save(folder/'services_audit.json',service_proof))

def audit_saved(plan,expected,boundary):
 from . import transport,qualification,verifier,sequential,statistical_qualification
 calibration=c.read(c.resolve(plan['calibration']));statistical_qualification.verify_qualification(calibration)
 q,qexit=passed_job('qualification',expected);qualification_data=verifier.read_bound(q['qualification'])
 qfolder=c.resolve(q['services'][0]).parent/'service'
 proof=qualification.verify_live_qualification(qualification_data['requests'],plan,service_folder=qfolder,boundary=boundary)
 c.require(proof==qualification_data['verification'],'Independent complete live qualification')
 c.require(transport.verify_services(q['services'],plan,boundary)==verifier.read_bound(q['service_audit']),
  'Complete qualification service evidence')
 pairs=[];timings={'3':[],'6':[]};games=[];interims=[];owners=[qexit];stop=False;completed=[];replays=0
 for role in ('operational',*c.BLOCKS):
  path=c.OUT/'jobs'/role
  if role in c.BLOCKS and stop:
   c.require(not path.exists(),'No games after prospectively valid stopping decision');continue
  c.require(path.exists(),'All blocks through a valid stop must exist')
  result,exit=passed_job(role,expected);owners.append(exit)
  specs=stage_specs(plan,role);block=verifier.read_bound(result['block'])
  audit=verifier.verify_block(path/'worker/arena',plan,specs,boundary=boundary)
  c.require(audit==verifier.read_bound(block['verified_game_audit']) and block['pair_scores']==audit['pair_scores'] and
   block['replay_selection']==audit['replay_selection'],'Independent full saved game audit unchanged')
  replay=verifier.verify_replays(audit['replay_selection'],result['replayed'],plan,boundary=boundary)
  c.require(replay==verifier.read_bound(result['replay_proof']),'Independent exact replay proof')
  c.require(transport.verify_services(result['services']+result['replay_services'],plan,boundary)==
   verifier.read_bound(result['service_audit']),'Independent complete wire, forensics and actual cleanup')
  replays+=replay['repetitions']
  if role=='operational':
   c.require(audit['official_games']==6 and audit['failed_decisions']==0,'Six clean excluded operational games');continue
  completed.append(role);pairs+=audit['pair_scores'];games+=audit['games']
  for pool in timings:timings[pool]+=audit['timings'][pool]
  item=c.artifact(c.OUT/'interims'/f'look_{2*len(pairs):06d}.json');look=c.read(c.resolve(item))
  sequential.verify_analysis(pairs,calibration,look);interims.append(item);stop=look['stop']
 c.require(stop and 400<=2*len(pairs)<=3600,'One prospectively valid terminal look')
 final=c.read(c.resolve(interims[-1]));counts={str(s):dict(games=0,points2=0) for s in (0,1)}
 failures=[]
 for game in games:
  seat=str(game['spec']['candidate_seat']);counts[seat]['games']+=1;counts[seat]['points2']+=game['candidate_points2']
  if game['failed_decisions']:failures.append(verifier.read_bound(game['game'])['failure'])
 return dict(schema='sixworld512_complete_saved_evidence_v1',status='passed',completed_blocks=completed,
  games=len(games),pairs=len(pairs),pair_scores=pairs,candidate_score=sum(pairs)/len(pairs),
  terminal_look=final,interims=interims,seat_counts=counts,timings=timings,failed_decisions=failures,
  exact_reproductions=replays,operational_games=6,owner_exits=owners,
  models=0,new_games=0,original_attempt=True,qualification=proof)

def main():
 p=argparse.ArgumentParser();p.add_argument('--role',choices=c.JOBS,required=True);p.add_argument('--freeze-sha256',required=True);args=p.parse_args()
 role=args.role;folder=c.OUT/'jobs'/role/'worker';error=None;result=None
 try:
  owner=control.handshake(folder.parent,args.freeze_sha256,role);folder.mkdir(parents=True,exist_ok=False)
  boundary=control.Boundary(folder,owner);boundary()
  from tools.future_training_rules_20261021 import verify_runtime_sources
  sources=verify_runtime_sources()
  c.require(c.sha(c.PLAN)==args.freeze_sha256,'Exact frozen execution plan');plan=c.read(c.PLAN)
  c.require(sources==plan['corrected_source_receipt'],'Corrected actual worker entrypoint guard')
  c.save(folder/'runtime_sources.json',dict(sources=sources,identity=c.identity(),**c.IDENTITY))
  c.verify(plan['worker_inputs'],boundary);c.verify(plan['data_inputs'],boundary)
  if role!='qualification':passed_job('qualification',args.freeze_sha256)
  if role in c.BLOCKS:passed_job('operational',args.freeze_sha256)
  if role in c.MODEL_JOBS:result=model_stage(folder,plan,owner,boundary,role)
  else:
   result=audit_saved(plan,args.freeze_sha256,boundary)
   report=c.save_gzip(folder/'saved_audit.json.gz',result)
   if role=='analysis':
    from .charts import render
    result=dict(report=report,charts=render(result,folder/'charts'))
   else:
    from .verifier import read_bound
    analysis,_=passed_job('analysis',args.freeze_sha256)
    c.require(read_bound(analysis['report'])==result,'Independent final acceptance matches all analysis evidence')
    c.verify(analysis['charts'],boundary)
    result=dict(approved=True,report=report,analysis=analysis,
     player_eligible=result['terminal_look']['player_eligible'],teacher_eligible=result['terminal_look']['teacher_eligible'],
     automatic_promotion=False,automatic_followup=False)
  c.verify(plan['worker_inputs'],boundary);c.verify(plan['data_inputs'],boundary);boundary()
 except BaseException:error=traceback.format_exc()
 body=dict(status='passed' if error is None else 'failed',role=role,identity=c.identity(),freeze_sha256=args.freeze_sha256,
  result=result,error=error,time=c.now())
 c.save(folder/'result.json',body)
 if error:print(error,flush=True)
 return 0 if error is None else 1

if __name__=='__main__':raise SystemExit(main())
