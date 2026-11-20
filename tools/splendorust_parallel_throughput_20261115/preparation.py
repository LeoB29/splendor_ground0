"""Model-free preparation and freeze; no run is authorized by these commands."""
import argparse, sys
from copy import deepcopy
from pathlib import Path
from . import common as c

def design():
 return dict(study=c.STUDY,scope='engineering_orchestration_only',checkpoint=c.PARENT_CHECKPOINT_SHA,
  simulations=512,worlds=3,cpuct=.4,depth=64,dynamic_fpu=True,precision='float32',inference_batch=1,
  live_setups=16,live_games=32,collection_setups=32,collection_games=64,
  live_order=['A1_0_7','A4_0_7','A4_8_15','A1_8_15'],collection_order=['control_X','candidate_X','candidate_Y','control_Y'],
  collection_overlap='candidate_X_checks_with_candidate_Y_play',live_guard=[119.,120.],collection_guard=[60.,75.],
  live_adoption_speedup=2.,live_adoption_max_response_seconds=60.,collection_adoption_speedup=1.3,
  identity='exact_every_scientific_field_and_float_bytes',all_engineering_excluded_from_corpora=True,
  automatic_retries=0,replacement_games=0,scientific_training_updates=0,strength_claims=False,**c.IDENTITY)

def source_inputs():
 from tools.splendorust_search512_20261101.preparation import python_dependencies
 files=list((c.ROOT/c.PACKAGE.replace('.','/')).rglob('*.py'))
 paths=python_dependencies(files)
 return [c.artifact(p) for p in sorted(set(paths+[c.BRIEF,c.PROTOCOL]))]

def unique(items):
 out={}
 for item in items:c.require(out.setdefault(item['path'],item)==item,'Consistent bound source '+item['path'])
 return sorted(out.values(),key=lambda x:x['path'])

def prepare():
 from tools.future_training_rules_20261021 import verify_runtime_sources
 from . import schedule,topology
 from .live.prepare import prepare as live_prepare
 from .collection.preparation import build_plan
 c.require(not (c.OUT/'prepared.json').exists() and not c.PLAN.exists(),'One separate preparation')
 c.require(c.sha(c.ACTIVE_OUT/'freeze.json')==c.ACTIVE_FREEZE_SHA,'Current collection freeze unchanged')
 prior=c.read(c.ROOT/'outputs/splendorust_sixworld512_20261114_02/freeze.json')
 sources=verify_runtime_sources();c.require(sources==prior['corrected_source_receipt'],'Unchanged corrected sources')
 plan={key:deepcopy(prior[key]) for key in ('research_profiles','control_profiles','graph_parity_plan','console_executable','corrected_source_receipt')}
 reservation=schedule.prepare();plan.update(live_prepare(c.OUT/'live_inputs',plan))
 top=topology.read_topology();plan.update(schema=c.STUDY,design=design(),topology=top,placement=topology.choose_placement(top),
  collection=build_plan(schedule.collection_specs(),schedule.qualification_specs()),reservation=reservation,
  limits=dict(total_seconds=c.TOTAL_SECONDS,model_seconds=c.MODEL_SECONDS,process_rss=c.PROCESS_RSS,
   aggregate_rss=c.AGGREGATE_RSS,evidence_bytes=c.MAX_OUTPUT,disk_floor=c.MIN_DISK),
  evidence_review=c.artifact(c.PREP/'evidence_review.json'),protocol=c.artifact(c.PROTOCOL),brief=c.artifact(c.BRIEF),
  admission=dict(current_collection_freeze=c.artifact(c.ACTIVE_OUT/'freeze.json'),required_receipts=[
   dict(path='outputs/splendorust_collection4096_20261115_01/observed_owner_exit.json',equals=dict(status='accepted_completed_verified',actual_root_exit=0,exact_root_identity_absent=True,owned_cleanup_verified=True)),
   dict(path='outputs/splendorust_collection4096_20261115_01/final_audit/worker/result.json',equals=dict(status='passed',error=None)),
   dict(path='outputs/splendorust_collection4096_launch_20261115_01/observed_launcher_exit.json',equals=dict(status='passed',actual_launcher_exit_code=0,actual_launcher_exit_observed=True,exact_launcher_identity_absent=True,error=None))]),
  launch_authorized=False,preparation_only=True,models_run=0,games_run=0,**c.IDENTITY)
 # Preserve the complete accepted runtime bindings, add the new dependencies.
 collection_prior=c.read(c.ACTIVE_OUT/'freeze.json')
 plan['inherited_worker_inputs']=unique(prior['worker_inputs']+collection_prior['worker_inputs'])
 plan['worker_inputs']=unique(plan['inherited_worker_inputs']+source_inputs())
 c.require('torch' not in sys.modules,'Preparation loaded no tensor runtime')
 return c.save(c.OUT/'prepared.json',plan)

def freeze(tests,review,fixture):
 c.require(not c.PLAN.exists(),'Exclusive future pilot freeze')
 plan=c.read(c.OUT/'prepared.json');code=source_inputs();proof=c.read(tests);audit=c.read(review);probe=c.read(fixture)
 reservation_check=c.PREP/'reservation_final_validation.json';reservation=c.read(reservation_check)
 c.require(reservation['status']=='passed' and reservation['seed_inventory']==plan['reservation']['seed_inventory']
  and reservation['collisions']==[] and reservation['target_count']==50,'Authenticated complete fresh reservation')
 c.require(proof['status']=='passed' and proof['failed']==proof['skipped']==0 and proof['models']==proof['games']==0
  and proof['collected']==proof['passed'] and proof['passed']>0,
  'Passed complete model-free tests')
 c.require(proof['inputs']==code,'Tests bind final source inventory')
 c.require(audit['approved'] and not audit['blocking_findings'],'One independent review passed')
 reviewed={x['path']:x for x in audit['inputs']}
 for item in code:
  if item['path'].startswith(c.PACKAGE.replace('.','/')+'/'):
   c.require(reviewed.get(item['path'])==item,'Final reviewed new source: '+item['path'])
 c.require(probe['status']=='passed' and probe['detached'] and probe['models']==probe['games']==0,'Actual detached fixture passed')
 for key in ('owner.py','runtime.py','fixture.py'):
  item=c.artifact(c.ROOT/c.PACKAGE.replace('.','/')/key);c.require(item in probe['inputs'],'Fixture covers final lifecycle source')
 c.verify(proof['inputs']);c.verify(audit['inputs']);c.verify(probe['inputs'])
 plan['worker_inputs']=unique(plan['inherited_worker_inputs']+code)
 plan.update(tests=c.artifact(tests),independent_review=c.artifact(review),detached_fixture=c.artifact(fixture),
  reservation_final_validation=c.artifact(reservation_check),frozen_at=c.now())
 plan['inputs']=unique(plan['worker_inputs']+list(c.nested_artifacts(plan)))
 c.verify(plan['inputs']);c.require('torch' not in sys.modules,'No models in freeze')
 return c.save(c.PLAN,plan)

def main():
 parser=argparse.ArgumentParser();parser.add_argument('mode',choices=('prepare','freeze'));parser.add_argument('--tests');parser.add_argument('--review');parser.add_argument('--fixture')
 args=parser.parse_args();print(c.canonical(prepare() if args.mode=='prepare' else freeze(args.tests,args.review,args.fixture)))

if __name__=='__main__':main()
