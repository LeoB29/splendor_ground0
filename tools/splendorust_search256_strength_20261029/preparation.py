"""Finite additive source/profile freeze; no models or games are imported."""
import argparse
import ast
from copy import deepcopy
from pathlib import Path
import shutil
import sys
from . import common as c,design,adapters,qualification,transport
from tools.future_training_rules_20261021 import verify_runtime_sources,RULESET_ID,CATALOGUE_SHA256,TURN_CONTRACT
from tools.splendorust_pool6_live_20261027.preparation import RUNTIME_KEYS,nested_artifacts
ORIGINAL=c.ROOT/'outputs/splendorust_search128_strength_20261028_01/freeze.json'
ORIGINAL_SHA='9026f33a3a6dcc449bb28f025493825230f04d91298f21c2c5bb2382b69587c9'
PREDECESSOR=c.ROOT/'outputs/splendorust_exploration128_20261029_02'
PREDECESSOR_SHA='3d899628cf0ba1a50688d53abe9a4b329e364131e269729eeac3b1f03bf87cbd'

def contract():
    return dict(phase=c.PHASE,job_seconds=c.JOB_SECONDS,collection_cutoff_seconds=c.COLLECTION_CUTOFF_SECONDS,
      stage_seconds=c.STAGES,independent_review_maximum_seconds=c.INDEPENDENT_REVIEW_SECONDS,
      max_evidence_bytes=c.MAX_EVIDENCE_BYTES,final_reserve_bytes=c.FINAL_RESERVE_BYTES,prefinal_evidence_bytes=c.PREFINAL_EVIDENCE_BYTES,
      max_rss_bytes=c.MAX_RSS_BYTES,initial_available_ram_bytes=c.INITIAL_AVAILABLE_RAM_BYTES,minimum_available_ram_bytes=c.MINIMUM_AVAILABLE_RAM_BYTES,
      initial_ram_stability_seconds=60,maximum_ram_wait_seconds=c.RAM_WAIT_SECONDS,startup_minimum_available_ram_bytes=c.STARTUP_MINIMUM_AVAILABLE_RAM_BYTES,
      initial_free_disk_bytes=c.INITIAL_FREE_DISK_BYTES,minimum_free_disk_bytes=5*c.GiB,
      seconds_per_move=10.,internal_seconds=9.,research_seconds_per_move=10.,research_internal_seconds=9.,
      research_log_max_bytes=128*c.MiB,frozen_log_max_bytes=128*c.MiB,research_response_max_bytes=4*c.MiB,frozen_response_max_bytes=4*c.MiB,request_max_bytes=c.MiB,
      placement_mode='isolated',max_resident_workers=2,max_workers=c.MAX_WORKERS,max_graph_builds=c.MAX_GRAPH_BUILDS,
      qualification_decisions=c.QUALIFICATION_DECISIONS,qualification_ordinary_requests=c.QUALIFICATION_ORDINARY_REQUESTS,qualification_graph_cases=22,
      operational_games=6,main_games=400,maximum_reproductions=48,
      automatic_retries=0,failed_decision_retries=0,training=False,seven_hundred_ms_gate=False,one_second_gate=False,
      declared_worker_lifecycles=design.lifecycle_inventory())

def python_dependencies(seeds):
    """Conservative static local import closure, including function-local imports."""
    pending=list(seeds);seen=set()
    def resolve(module):
        roots=(c.ROOT,c.ROOT/'src')
        for root in roots:
            p=root.joinpath(*module.split('.'))
            for file in (p.with_suffix('.py'),p/'__init__.py'):
                if file.is_file():return file.resolve()
    while pending:
        path=Path(pending.pop()).resolve()
        if path in seen:continue
        seen.add(path)
        relative=path.relative_to(c.ROOT).with_suffix('').parts
        if relative[0]=='src':relative=relative[1:]
        package=list(relative[:-1] if relative[-1]!='__init__' else relative[:-1])
        for node in ast.walk(ast.parse(path.read_text(encoding='utf-8-sig'))):
            modules=[]
            if isinstance(node,ast.Import):modules=[x.name for x in node.names]
            elif isinstance(node,ast.ImportFrom):
                prefix=package[:len(package)-node.level+1] if node.level else []
                base='.'.join(prefix+([node.module] if node.module else []))
                modules=[base]+[base+'.'+x.name for x in node.names if x.name!='*']
            for module in modules:
                file=resolve(module)
                if file and file not in seen:pending.append(file)
    return sorted(seen)


def prepare():
    prepared=design.prepare()
    prepared.update(adapters.prepare_profiles())
    prepared['qualification_plan']=design.fixed(c.OUT/'qualification_plan.json',qualification.build_plan())
    return prepared


def freeze():
    from . import runner,analysis
    c.require(not(c.OUT/'freeze.json').exists(),'No replacement freeze')
    c.require(c.file_hash(ORIGINAL)==ORIGINAL_SHA,'Original scientific freeze identity')
    original=c.read(ORIGINAL);prepared=prepare();source_guard=verify_runtime_sources()
    seeds=design.seed_inventory();design.fixed(c.OUT/'seed_inventory.json',seeds)
    test=c.read(c.OUT/'prelaunch/model_free_tests.json');review=c.read(c.OUT/'prelaunch/implementation_review.json')
    c.require(test.get('passed') is True and test.get('exit_code')==0,'Final model-free checks')
    c.verify(test['inputs'])
    c.require(review.get('approved') is True,'Independent implementation review')
    c.verify(review['inputs'])
    pins={}
    def add(item):
        c.verify([item]);previous=pins.get(item['path'])
        c.require(previous is None or previous==item,'Conflicting frozen source pin');pins[item['path']]=item
    frozen={key:deepcopy(original[key]) for key in RUNTIME_KEYS}
    frozen.update(contract());frozen.update({key:prepared[key] for key in ('profiles','native_artifacts_by_route','native_roles_by_route')})
    for item in nested_artifacts(frozen):add(item)
    for item in nested_artifacts(c.read(c.ROOT/prepared['qualification_plan']['path'])):add(item)
    for route in adapters.ROUTES:
        for profile in frozen['profiles'][route].values():
            value=c.read(c.ROOT/profile['path'])
            for item in nested_artifacts(value):add(item)
            for key in ('checkpoint','model_definition','native_binary'):add(c.artifact(c.ROOT/value[key]))
    for item in test['inputs']+review['inputs']+seeds['checked']:add(item)
    seeds=list(Path(__file__).parent.glob('*.py'))+[c.ROOT/'tools/future_training_rules_20261021.py']
    for path in python_dependencies(seeds):add(c.artifact(path))
    tests=set((c.ROOT/'tests').glob('test_splendorust_search256*20261029.py'))
    for path in sorted(tests):add(c.artifact(path))
    prepared_paths=list(c.OUT.glob('*.json'))+list((c.OUT/'prelaunch').rglob('*'))+list((c.OUT/'profiles').glob('*.json'))+list((c.OUT/'native_build').glob('*'))
    for path in sorted(prepared_paths):
        if path.is_file() and '__pycache__' not in path.parts:add(c.artifact(path))
    c.require(c.file_hash(PREDECESSOR/'freeze.json')==PREDECESSOR_SHA,'Accepted predecessor freeze preserved')
    prior=c.read(PREDECESSOR/'freeze.json')
    for item in prior['inputs']:add(item)
    acceptance=c.ROOT/'outputs/splendorust_exploration128_20261029_02_postrun/acceptance.json'
    c.require(c.file_hash(acceptance)=='8113aeee97cfa9f05f5bb44124eeaad497fd2ecc46c833d0ea472e5aec994a40','Accepted predecessor receipt unchanged')
    c.require(c.read(acceptance)['approved'] is True and c.read(acceptance)['freeze_sha256']==PREDECESSOR_SHA,'Accepted predecessor, no new retry')
    preserve=[PREDECESSOR/name for name in ('freeze.json','completion.json','independent_review.json','independent_review_os_exit.json',
        'execution.json','watchdog_execution.json','completed_evidence_seal.json','selection.json','selection_verification.json')]
    preserve += [ORIGINAL,acceptance,
      c.ROOT/'outputs/splendorust_search128_repeat_20261029_01/acceptance.json',
      c.ROOT/'outputs/splendorust_search128_saved_audit_20261029_continuity/recovered_verification_accepted.json']
    for path in preserve+[c.PROTOCOL,c.PREPARED_PROTOCOL,c.ROOT/'docs/engineering/future_training_rules_20261021.md']:add(c.artifact(path))
    frozen.update(schema='splendorust_search256_strength_v1',time=c.now(),protocol=c.artifact(c.PROTOCOL),
      schedule=prepared['schedule'],qualification_plan=prepared['qualification_plan'],profiles=prepared['profiles'],source_guard=source_guard,
      ruleset_id=RULESET_ID,catalogue_sha256=CATALOGUE_SHA256,turn_contract=TURN_CONTRACT,
      statistics_runtime=analysis.runtime_contract(),prior_scientific_freeze=c.artifact(ORIGINAL),
      source_preservation_scope='Finite imported sources/runtime/profile/build artifacts and accepted historical receipts; no unrelated historical full-tree audit',
      accepted_predecessor=c.artifact(PREDECESSOR/'freeze.json'),accepted_predecessor_receipt=c.artifact(acceptance),fresh_full_qualification_and_operational_run=True,
      only_scientific_change='simulations_128_to_256',weights_unchanged=True,precision_unchanged=True,reference_profile_unchanged=True,
      baseline_profiles=deepcopy(prepared['profiles']['frozen']),research_profiles=deepcopy(prepared['profiles']['research']),
      inputs=sorted(pins.values(),key=lambda x:x['path']))
    c.require('torch' not in sys.modules,'No learned model import in preparation')
    c.require(shutil.disk_usage(c.ROOT).free>=c.INITIAL_FREE_DISK_BYTES,'Initial disk reserve')
    runner.validate_freeze_contract(frozen);c.verify(frozen['inputs'])
    return dict(status='frozen_not_launched',freeze=c.save(c.OUT/'freeze.json',frozen),inputs=len(pins))

if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('action',choices=['prepare','freeze']);args=parser.parse_args()
    print(c.canonical(prepare() if args.action=='prepare' else freeze()))
