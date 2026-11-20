"""Exclusive freeze and bounded serial qualification/comparison ownership."""
import argparse,sys,time,traceback,os
from pathlib import Path
from . import common as c,control

def qualified_calibration(path):
 from . import statistical_qualification,sequential
 calibration=c.read(path)
 c.require(calibration.get('status')=='passed','Only passed statistical qualification may launch models')
 proof=statistical_qualification.verify_qualification(calibration)
 c.require(proof['qualification_passed'] is True,'Independently recomputed qualification must pass')
 sequential.validate_calibration(calibration)
 return calibration

def source_inputs():
 from tools.splendorust_search512_20261101.preparation import python_dependencies
 import importlib.util
 c.require(c.sha(c.PRIOR_FREEZE)==c.PRIOR_FREEZE_SHA,'Preserved original freeze')
 prior=c.read(c.PRIOR_FREEZE)
 paths=list((c.ROOT/c.PACKAGE.replace('.','/')).glob('*.py'))
 items=prior['worker_inputs']+c.read(c.FAILED_OUT/'freeze.json')['worker_inputs']+[c.artifact(p) for p in python_dependencies(paths)]
 items += [c.artifact(c.PROTOCOL),c.artifact(c.PREPARATION_PROTOCOL),c.artifact(c.CORRECTION_PROTOCOL)]
 # Bind added statistical and plotting implementation, including native modules.
 for name in ('scipy','matplotlib'):
  package=Path(importlib.util.find_spec(name).origin).parent
  items += [c.artifact(p) for p in package.rglob('*') if p.is_file() and p.suffix in ('.py','.pyd','.dll')]
 unique={}
 for item in items:
  if item['path'] in unique:c.require(unique[item['path']]==item,'Consistent preserved dependency')
  unique[item['path']]=item
 return sorted(unique.values(),key=lambda i:i['path'])

def prepare(calibration_path):
 from tools.future_training_rules_20261021 import verify_runtime_sources
 c.OUT.mkdir(parents=True,exist_ok=True)
 c.require(not c.PLAN.exists() and not (c.OUT/'launcher_claim.json').exists(),'Separate unfrozen startup correction')
 c.require(c.sha(c.FAILED_OUT/'freeze.json')==c.FAILED_FREEZE_SHA,'Exact preserved first attempt freeze')
 original=c.read(c.FAILED_OUT/'freeze.json');sources=verify_runtime_sources()
 c.require(sources==original['corrected_source_receipt'] and c.design()==original['design'],'Unchanged corrected science')
 qualified_calibration(c.resolve(original['calibration']))
 c.verify(original['worker_inputs']);c.verify(original['data_inputs'])
 identities=[]
 for relative in ('launch/actual_exit.json','jobs/qualification/actual_exit.json'):
  receipt=c.read(c.FAILED_OUT/relative)
  c.require(receipt['status']=='failed' and receipt['exit_code']==1 and receipt['actual_exit_observed'] and
   receipt['exact_owned_absence'] and not receipt['cleanup_errors'],'Preserved failed owner actually exited')
  identities += [receipt['identity'],*receipt['observed_owned_identities']]
 launch=c.read(c.FAILED_OUT/'launcher_claim.json');identities.append(launch['identity'])
 c.require(all(not c.exact_alive(i) for i in identities),'No original exact owner remains alive')
 worker=c.read(c.FAILED_OUT/'jobs/qualification/worker/result.json')
 c.require(worker['result'] is None and 'Worker must use one allowed logical processor' in worker['error'],
  'Only the diagnosed pre-spawn CPU admission failure')
 transport=c.FAILED_OUT/'jobs/qualification/worker/live/services/life_0000/transport'
 c.require(not (transport/'initialization.json').exists() and (transport/'stdout.log').stat().st_size==0 and
  not (transport.parent/'service').exists() and not (c.FAILED_OUT/'jobs/operational').exists(),
  'No original model service, decision or game was started')
 inventory=[c.artifact(p) for p in sorted(c.FAILED_OUT.rglob('*')) if p.is_file()]
 preservation=c.save(c.OUT/'original_preservation.json',dict(inputs=inventory,models=0,new_games=0,
  original_limits=launch['limits'],original_identities=identities,reason='CPU admission before service spawn',time=c.now()))
 keys=('design','data_inputs','schedule','initials','qualification_plan','calibration','research_profiles',
  'control_profiles','graph_parity_plan','console_executable','corrected_source_receipt')
 prepared={key:original[key] for key in keys}
 prepared['data_inputs']=original['data_inputs']+inventory+[preservation]
 prepared.update(worker_inputs=source_inputs(),original_limits=launch['limits'],startup_correction=preservation,time=c.now())
 return c.save(c.OUT/'prepared.json',prepared)

def refresh(label):
 c.require(label.isalnum() and not c.PLAN.exists() and not (c.OUT/'launcher_claim.json').exists(),'Only unfrozen preparation refresh')
 prepared=c.read(c.OUT/'prepared.json');c.verify(prepared['data_inputs'])
 c.require(prepared['design']==c.design(),'No prospective scientific design change during source-only refresh')
 prepared.update(worker_inputs=source_inputs(),time=c.now(),prior_preparation=c.artifact(c.OUT/'prepared.json'))
 return c.save(c.OUT/('prepared_'+label+'.json'),prepared)

def freeze(prepared_path,tests,reviews,fixture_label):
 from . import statistical_qualification
 prepared=c.read(prepared_path)
 c.require(prepared['design']==c.design() and prepared['worker_inputs']==source_inputs(),'Current source/design preparation')
 qualified_calibration(c.resolve(prepared['calibration']))
 proof=c.read(tests);summary=proof['summary']
 c.require(proof['status']=='passed' and proof['exit_code']==0 and proof['inputs']==prepared['worker_inputs'] and
  summary['failed']==summary['skipped']==0 and summary['passed']==summary['collected']>=100 and
  not summary['tensor_runtime_loaded'] and summary['models']==summary['new_games']==0,'Complete source-bound tests')
 c.require(len(reviews)>=3,'Independent implementation reviews')
 covered=set();reviewers=set()
 for path in reviews:
  review=c.read(path);c.require(review['approved'] is True and not review['blocking_findings'],'Independent approval')
  c.verify(review['inputs']);covered.update(i['path'] for i in review['inputs']);reviewers.add(review['reviewer'])
 c.require(len(reviewers)>=3,'Three independent reviewers')
 required={'adapters.py','service.py','qualification.py','arena.py','verifier.py','schedule.py','sequential.py',
  'statistical_qualification.py','owner.py','worker.py','transport.py','control.py','launcher.py','common.py'}
 prefix=c.PACKAGE.replace('.','/')+'/'
 c.require({prefix+n for n in required}<=covered,'Every production component independently reviewed')
 fixture=c.OUT/'prelaunch'/('detached_fixture_'+fixture_label)/'actual_exit.json';f=c.read(fixture)
 c.require(f['status']=='passed' and f['exit_code']==0 and f['exact_owned_absence'] and f['actual_exit_observed'] and
  f['models']==f['new_games']==0 and not f['cleanup_errors'] and not f['error'],'Detached lifecycle fixture passed')
 c.verify(f['source_inputs'])
 launcher=c.read(c.OUT/'prelaunch'/('fixture_launcher_'+fixture_label+'.json'))['identity']
 c.require(not c.exact_alive(launcher) and not c.exact_alive(f['identity']),'Fixture actually exited')
 c.verify(prepared['worker_inputs']);c.verify(prepared['data_inputs'])
 return c.save(c.PLAN,dict(**prepared,prepared_input=c.artifact(prepared_path),tests=c.artifact(tests),
  reviews=[c.artifact(p) for p in reviews],detached_fixture=c.artifact(fixture),frozen_at=c.now()))

def run(expected):
 from .worker import passed_job
 from . import sequential
 from tools.future_training_rules_20261021 import verify_runtime_sources
 owner=control.handshake(c.OUT/'launch',expected,'root');bounds=owner['limits']
 c.require(c.sha(c.PLAN)==expected,'Exact frozen plan');plan=c.read(c.PLAN)
 sources=verify_runtime_sources();c.require(sources==plan['corrected_source_receipt'],'Corrected root evaluator source guard')
 c.save(c.OUT/'root_runtime_sources.json',dict(sources=sources,identity=c.identity(),**c.IDENTITY))
 c.verify(plan['worker_inputs']);c.verify(plan['data_inputs']);control.bounds(bounds)
 c.save(c.OUT/'claim.json',dict(identity=c.identity(),freeze=c.artifact(c.PLAN),limits=bounds,time=c.now()))
 receipts=[];error=None;pairs=[];stop=False;calibration=qualified_calibration(c.resolve(plan['calibration']))
 try:
  for role in c.JOBS:
   if role in c.BLOCKS and stop:continue
   c.status(c.OUT/'stage.json',dict(time=c.now(),stage=role,status='running',completed_jobs=len(receipts),verified_main_games=2*len(pairs)))
   limits=control.clipped_limits(bounds,c.CAPS[role],model=role in c.MODEL_JOBS)
   command=[sys.executable,'-B','-u','-m',c.PACKAGE+'.worker','--role',role,'--freeze-sha256',expected]
   receipt=control.launch(command,c.OUT/'jobs'/role,limits,dict(role=role,freeze_sha256=expected),
    rss=c.MODEL_RSS if role in c.MODEL_JOBS else c.AUDIT_RSS)
   result,_=passed_job(role,expected);receipts.append(receipt)
   if role in c.BLOCKS:
    block=c.read(c.resolve(result['block']));pairs+=block['pair_scores']
    analysis=sequential.analyze_pairs(pairs,calibration);sequential.verify_analysis(pairs,calibration,analysis)
    c.save(c.OUT/'interims'/f'look_{2*len(pairs):06d}.json',analysis);stop=analysis['stop']
  c.require(stop,'Comparison ends only under a frozen stopping rule')
  c.verify(plan['worker_inputs']);c.verify(plan['data_inputs']);control.bounds(bounds)
 except BaseException:error=traceback.format_exc()
 execution=c.save(c.OUT/'execution.json',dict(status='passed' if error is None else 'failed',
  freeze_sha256=expected,receipts=receipts,error=error,verified_main_games=2*len(pairs),time=c.now()))
 if error:
  c.save(c.OUT/'failure.json',dict(status='incomplete',execution=execution,error=error,time=c.now()))
  c.status(c.OUT/'stage.json',dict(status='failed',stage='failed',time=c.now()));return 1
 c.save(c.OUT/'completion.json',dict(status='verified_pending_actual_owner_exit',execution=execution,
  acceptance=c.artifact(c.OUT/'jobs/acceptance/worker/result.json'),automatic_followup=False))
 c.status(c.OUT/'stage.json',dict(status='verified_pending_actual_owner_exit',stage='complete',time=c.now()))
 return 0

def main():
 p=argparse.ArgumentParser();p.add_argument('mode',choices=('prepare','refresh','freeze','run'))
 p.add_argument('--calibration');p.add_argument('--label');p.add_argument('--prepared');p.add_argument('--tests')
 p.add_argument('--reviews',nargs='*');p.add_argument('--fixture-label',default='final');p.add_argument('--freeze-sha256');a=p.parse_args()
 if a.mode=='prepare':print(c.canonical(prepare(a.calibration)));return 0
 if a.mode=='refresh':print(c.canonical(refresh(a.label)));return 0
 if a.mode=='freeze':print(c.canonical(freeze(a.prepared,a.tests,a.reviews,a.fixture_label)));return 0
 try:return run(a.freeze_sha256)
 except BaseException:
  error=traceback.format_exc()
  if not (c.OUT/'failure.json').exists():c.save(c.OUT/'failure.json',dict(status='incomplete',stage='root_bootstrap',error=error,time=c.now()))
  print(error,flush=True);return 1

if __name__=='__main__':raise SystemExit(main())
