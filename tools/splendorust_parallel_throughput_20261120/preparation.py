"""Model-free preparation and freeze; no run is authorized by these commands."""
import argparse, os, sys
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
 return [c.artifact(p) for p in sorted(set(paths+[c.BRIEF,c.PROTOCOL,c.CORRECTION]))]

def admission_receipts():
 """Completion receipts that actually exist for finished predecessors.

 The 15 Nov version bound the original collection attempt, which ended
 incomplete and never wrote these files. Bind the accepted amended recovery
 and the later completed training study instead; nothing else changes.
 """
 rel=lambda path:c.normalized(path).relative_to(c.ROOT).as_posix()
 return [
  dict(path=rel(c.ACCEPTED_COLLECTION/'observed_owner_exit.json'),equals=dict(status='accepted_completed_verified',
   actual_root_exit=0,exact_root_identity_absent=True,owned_cleanup_verified=True)),
  dict(path=rel(c.ACCEPTED_COLLECTION/'final_audit/worker/result.json'),equals=dict(status='passed',error=None,
   plan_sha256=c.ACCEPTED_COLLECTION_FREEZE_SHA)),
  dict(path=rel(c.ACCEPTED_COLLECTION_LAUNCH/'observed_launcher_exit.json'),equals=dict(status='passed',
   actual_launcher_exit_code=0,actual_launcher_exit_observed=True,exact_launcher_identity_absent=True,error=None)),
  dict(path=rel(c.TRAINING_OUT/'observed_owner_exit.json'),equals=dict(status='accepted_completed_verified')),
  dict(path=rel(c.TRAINING_OUT/'launch/actual_exit.json'),equals=dict(status='passed',exit_code=0,error=None,
   actual_exit_observed=True,exact_owned_absence=True,freeze_sha256=c.TRAINING_FREEZE_SHA)),
  dict(path=rel(c.TRAINING_OUT/'observed_launcher_exit.json'),equals=dict(status='passed',exit_code=0,error=None,
   actual_exit_observed=True,exact_absence=True))]

def rebind_console(plan):
 """Pin the CURRENT System32 console host; same path only, never another file.

 Windows servicing replaced conhost.exe after the inherited 14 Nov six-world
 freeze. Runtime ownership requires every console child's hash to equal this
 pin, so a stale pin would fail the first live service. The superseded hash is
 recorded as plain fields, not as an artifact descriptor that inputs re-verify.
 """
 inherited=plan['console_executable']
 path=(Path(os.environ['SystemRoot'])/'System32/conhost.exe').resolve()
 c.require(set(inherited)=={'path','bytes','sha256'} and c.normalized(inherited['path'])==c.normalized(path),
  'Same pinned System32 console host path')
 current=c.artifact(path)
 if current!=inherited:
  plan['console_rebinding']=dict(path=current['path'],inherited_sha256=inherited['sha256'],inherited_bytes=inherited['bytes'],
   current_sha256=current['sha256'],current_bytes=current['bytes'],
   reason='Windows servicing replaced System32 conhost.exe after the inherited six-world freeze')
 plan['console_executable']=current
 # The same host is also inherited among the worker inputs that each worker re-verifies.
 for key in ('inherited_worker_inputs','worker_inputs'):
  if key in plan:
   plan[key]=[current if c.normalized(item['path'])==c.normalized(path) else item for item in plan[key]]
 return plan

def rebind_note(plan):
 """Bind the correction note's final bytes; it gained defect 3 after preparation.

 The note is documentation, also bound through source_inputs(). The prepared
 hash is kept as plain fields, not as an artifact descriptor inputs re-verify.
 """
 prepared=plan['correction'];current=c.artifact(c.CORRECTION)
 c.require(set(prepared)=={'path','bytes','sha256'} and prepared['path']==current['path'],'Same correction note path')
 if current!=prepared:
  plan['correction_at_preparation']=dict(path=prepared['path'],prepared_sha256=prepared['sha256'],prepared_bytes=prepared['bytes'])
 plan['correction']=current
 return plan

def original_unlaunched():
 """The superseded 15 Nov version stays frozen, unchanged and never launched."""
 c.require(c.sha(c.ORIGINAL_OUT/'freeze.json')==c.ORIGINAL_FREEZE_SHA,'Superseded original pilot freeze unchanged')
 for name in ('launcher_claim.json','claim.json','launch','observed_owner_exit.json','complete.json','failure.json'):
  c.require(not (c.ORIGINAL_OUT/name).exists(),'Superseded original pilot was never launched: '+name)
 return dict(package=c.ORIGINAL_PACKAGE,freeze=c.artifact(c.ORIGINAL_OUT/'freeze.json'),launched=False,
  reason='launch gate bound the incomplete original collection attempt; registry discovery required rg on PATH')

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
 c.require(c.sha(c.ACCEPTED_COLLECTION/'freeze.json')==c.ACCEPTED_COLLECTION_FREEZE_SHA,'Accepted collection recovery freeze')
 c.require(c.sha(c.TRAINING_OUT/'freeze.json')==c.TRAINING_FREEZE_SHA,'Completed training study freeze')
 superseded=original_unlaunched()
 prior=c.read(c.ROOT/'outputs/splendorust_sixworld512_20261114_02/freeze.json')
 sources=verify_runtime_sources();c.require(sources==prior['corrected_source_receipt'],'Unchanged corrected sources')
 plan={key:deepcopy(prior[key]) for key in ('research_profiles','control_profiles','graph_parity_plan','console_executable','corrected_source_receipt')}
 reservation=schedule.prepare();plan.update(live_prepare(c.OUT/'live_inputs',plan))
 top=topology.read_topology();plan.update(schema=c.STUDY,design=design(),topology=top,placement=topology.choose_placement(top),
  collection=build_plan(schedule.collection_specs(),schedule.qualification_specs()),reservation=reservation,
  limits=dict(total_seconds=c.TOTAL_SECONDS,model_seconds=c.MODEL_SECONDS,process_rss=c.PROCESS_RSS,
   aggregate_rss=c.AGGREGATE_RSS,evidence_bytes=c.MAX_OUTPUT,disk_floor=c.MIN_DISK),
  evidence_review=c.artifact(c.ORIGINAL_PREP/'evidence_review.json'),protocol=c.artifact(c.PROTOCOL),brief=c.artifact(c.BRIEF),
  correction=c.artifact(c.CORRECTION),supersedes=superseded,
  admission=dict(current_collection_freeze=c.artifact(c.ACTIVE_OUT/'freeze.json'),
   accepted_collection_freeze=c.artifact(c.ACCEPTED_COLLECTION/'freeze.json'),
   training_freeze=c.artifact(c.TRAINING_OUT/'freeze.json'),required_receipts=admission_receipts()),
  launch_authorized=False,preparation_only=True,models_run=0,games_run=0,**c.IDENTITY)
 # Preserve the complete accepted runtime bindings, add the new dependencies.
 collection_prior=c.read(c.ACTIVE_OUT/'freeze.json')
 plan['inherited_worker_inputs']=unique(prior['worker_inputs']+collection_prior['worker_inputs'])
 plan['worker_inputs']=unique(plan['inherited_worker_inputs']+source_inputs())
 rebind_console(plan)
 c.require('torch' not in sys.modules,'Preparation loaded no tensor runtime')
 return c.save(c.OUT/'prepared.json',plan)

def freeze(tests,review,fixture):
 c.require(not c.PLAN.exists(),'Exclusive future pilot freeze')
 plan=c.read(c.OUT/'prepared.json');code=source_inputs();proof=c.read(tests);audit=c.read(review);probe=c.read(fixture)
 # prepared.json (written before these helpers existed) keeps the inherited
 # console pin and the earlier note; the freeze rebinds both and records it.
 rebind_console(plan);rebind_note(plan)
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
