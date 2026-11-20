"""Finite source/profile/panel freeze; never imports a learned model."""
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


def contract():
    return dict(phase=c.PHASE,job_seconds=c.JOB_SECONDS,collection_cutoff_seconds=c.COLLECTION_CUTOFF_SECONDS,
        stage_seconds=c.STAGES,independent_review_maximum_seconds=c.INDEPENDENT_REVIEW_SECONDS,
        max_evidence_bytes=c.MAX_EVIDENCE_BYTES,final_reserve_bytes=c.FINAL_RESERVE_BYTES,
        prefinal_evidence_bytes=c.PREFINAL_EVIDENCE_BYTES,max_rss_bytes=c.MAX_RSS_BYTES,
        initial_available_ram_bytes=c.INITIAL_AVAILABLE_RAM_BYTES,
        minimum_available_ram_bytes=c.MINIMUM_AVAILABLE_RAM_BYTES,initial_ram_stability_seconds=60,
        maximum_ram_wait_seconds=c.RAM_WAIT_SECONDS,startup_minimum_available_ram_bytes=c.STARTUP_MINIMUM_AVAILABLE_RAM_BYTES,
        initial_free_disk_bytes=c.INITIAL_FREE_DISK_BYTES,minimum_free_disk_bytes=5*c.GiB,
        seconds_per_move=10.,internal_seconds=9.,research_seconds_per_move=10.,research_internal_seconds=9.,
        research_log_max_bytes=128*c.MiB,frozen_log_max_bytes=128*c.MiB,
        research_response_max_bytes=4*c.MiB,frozen_response_max_bytes=4*c.MiB,request_max_bytes=c.MiB,
        placement_mode='isolated',max_workers=2,max_graph_builds=2,ordinary_requests=120,
        qualification_decisions=119,qualification_ordinary_requests=0,graph_cases=22,parent_requests=124,
        automatic_retries=0,failed_decision_retries=0,new_games=0,training=False,
        seven_hundred_ms_gate=False,one_second_gate=False,no_player_admission=True,
        declared_worker_lifecycles=transport.lifecycle_inventory())


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


def freeze():
    from . import runner
    c.require(not(c.OUT/'freeze.json').exists(),'No replacement freeze')
    c.require(c.file_hash(ORIGINAL)==ORIGINAL_SHA,'Original scientific freeze identity')
    original=c.read(ORIGINAL);prepared=design.prepare();source_guard=verify_runtime_sources()
    plan=qualification.build_plan();qpin=design.fixed(c.OUT/'qualification_plan.json',plan)
    test=c.read(c.OUT/'prelaunch/model_free_tests_02.json')
    review=c.read(c.OUT/'prelaunch/implementation_review_02.json')
    c.require(test.get('passed') is True and test.get('exit_code')==0,'Final model-free checks')
    c.require(review.get('approved') is True,'Independent implementation review')
    c.verify(review['inputs'])
    pins={}
    def add(item):
        c.verify([item]);previous=pins.get(item['path'])
        c.require(previous is None or previous==item,'Conflicting frozen source pin')
        pins[item['path']]=item
    panel=c.read(c.ROOT/prepared['panel']['path'])
    for item in panel['inputs']+review['inputs']:add(item)
    frozen={key:deepcopy(original[key]) for key in RUNTIME_KEYS}
    frozen.update(contract())
    frozen.update(baseline_profiles=deepcopy(original['baseline_profiles']),
                  research_profiles=deepcopy(original['research_profiles']))
    for item in nested_artifacts(frozen):add(item)
    for profile in list(frozen['baseline_profiles'].values())+list(frozen['research_profiles'].values()):
        value=c.read(c.ROOT/profile['path'])
        for item in nested_artifacts(value):add(item)
        for key in ('checkpoint','model_definition','native_binary'):add(c.artifact(c.ROOT/value[key]))
    seeds=list(Path(__file__).parent.glob('*.py'))+[c.ROOT/'tools/future_training_rules_20261021.py']
    for path in python_dependencies(seeds):add(c.artifact(path))
    for path in sorted((c.ROOT/'tests').glob('test_splendorust_search128_repeat*20261029*.py')):add(c.artifact(path))
    for path in sorted(c.OUT.rglob('*')):
        if path.is_file() and '__pycache__' not in path.parts:add(c.artifact(path))
    for path in (ORIGINAL,c.PROTOCOL,c.ROOT/'docs/engineering/future_training_rules_20261021.md'):add(c.artifact(path))
    frozen.update(schema='splendorust_search128_repeat_v1',time=c.now(),
        protocol=c.artifact(c.PROTOCOL),panel=prepared['panel'],request_matrix=prepared['request_matrix'],
        qualification_plan=qpin,source_guard=source_guard,
        ruleset_id=RULESET_ID,catalogue_sha256=CATALOGUE_SHA256,turn_contract=TURN_CONTRACT,
        prior_scientific_freeze=c.artifact(ORIGINAL),prior_saved_diagnostic_freeze_sha256=design.FREEZE_SHA,
        source_preservation_scope='Finite imported sources/runtime/profile artifacts plus20 consumed saved roots; no historical full-tree audit',
        numerical_player_change=False,weights_unchanged=True,profiles_unchanged=True,
        same_workers_retained_for_repeats=True,inputs=sorted(pins.values(),key=lambda x:x['path']))
    c.require('torch' not in sys.modules,'No learned model import in preparation')
    c.require(shutil.disk_usage(c.ROOT).free>=c.INITIAL_FREE_DISK_BYTES,'Initial disk reserve')
    runner.validate_freeze_contract(frozen);c.verify(frozen['inputs'])
    return dict(status='frozen_not_launched',freeze=c.save(c.OUT/'freeze.json',frozen),inputs=len(pins))


if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('action',choices=['prepare','freeze'])
    args=parser.parse_args();print(c.canonical(design.prepare() if args.action=='prepare' else freeze()))
