"""Model-free preparation and exclusive prospective freeze of a single study."""
import argparse,ast,shutil,sys
from copy import deepcopy
from pathlib import Path
from . import common as c
from tools.future_training_rules_20261021 import verify_runtime_sources,RULESET_ID,CATALOGUE_SHA256,TURN_CONTRACT
from tools.splendorust_pool6_live_20261027.preparation import RUNTIME_KEYS,nested_artifacts

REVIEWS=('independent_review_worker.json','independent_review_controller.json','independent_review_design.json')


def fixed(path,value):
    if path.exists():
        c.require(c.read(path)==value,'Prepared immutable receipt differs: '+str(path))
        return c.artifact(path)
    return c.save(path,value)


def accepted_predecessor():
    c.require(c.file_hash(c.ACCEPTED_OUT/'freeze.json')==c.ACCEPTED_FREEZE_SHA and
        c.file_hash(c.ACCEPTED_RECEIPT)==c.ACCEPTED_RECEIPT_SHA,'Exact accepted256 study identity')
    prior=c.read(c.ACCEPTED_OUT/'freeze.json');receipt=c.read(c.ACCEPTED_RECEIPT)
    c.require(receipt['approved'] is True and not receipt['unresolved_blocking_findings'] and
        receipt['freeze_sha256']==c.ACCEPTED_FREEZE_SHA and receipt['guardian_actual_exit']['exit_code']==0 and
        receipt['actual_runner_watchdog_reader_exit_codes']==[0,0,0] and not receipt['cleanup_errors'] and
        receipt['coverage']['main_games']==400 and receipt['coverage']['complete'] is True,
        'Independently accepted256 predecessor with actual exits')
    c.require(c.file_hash(c.LIVE_OUT/'freeze.json')==c.LIVE_FREEZE_SHA and
        c.file_hash(c.LIVE_CLOSEOUT)==c.LIVE_CLOSEOUT_SHA,'Previous live optimization evidence preserved')
    return prior,receipt


def python_dependencies(seeds):
    """Conservative local import closure, including function-local imports."""
    pending=list(seeds);seen=set()
    def resolve(module):
        for root in (c.ROOT,c.ROOT/'src'):
            p=root.joinpath(*module.split('.'))
            for file in (p.with_suffix('.py'),p/'__init__.py'):
                if file.is_file():return file.resolve()
    while pending:
        path=Path(pending.pop()).resolve()
        if path in seen:continue
        seen.add(path)
        relative=path.relative_to(c.ROOT).with_suffix('').parts
        if relative[0]=='src':relative=relative[1:]
        package=list(relative[:-1])
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


def verify_seed_receipt(saved):
    from . import design
    c.require(saved['status']=='passed' and not saved['collisions'] and
        saved['domain']==design.DOMAIN and saved['new_setup_count']==203 and
        saved['bootstrap_seed']==design.BOOTSTRAP_SEED and saved['prior_seed_reuse'] is False,
        'Seed freshness must bind this exact fresh schedule and statistics seed')
    c.verify(saved['checked'])


def prepare():
    from . import design,adapters,qualification,control
    source_guard=verify_runtime_sources()
    prior,receipt=accepted_predecessor()
    prepared=design.prepare();prepared.update(adapters.prepare_profiles())
    prepared['qualification_plan']=fixed(c.OUT/'qualification_plan.json',qualification.build_plan())
    prepared['lifecycles']=fixed(c.OUT/'declared_worker_lifecycles.json',control.lifecycle_inventory())
    seeds=c.OUT/'seed_freshness.json'
    if seeds.exists():
        verify_seed_receipt(c.read(seeds))
    else:c.save(seeds,design.seed_inventory())
    prepared['seed_freshness']=c.artifact(seeds)
    prepared['source_guard']=source_guard
    prepared['preparation']=fixed(c.OUT/'preparation.json',dict(schema='search512_preparation_v1',
        models=0,games=0,source_guard=source_guard,documents=deepcopy(prepared),
        accepted_predecessor=c.artifact(c.ACCEPTED_OUT/'freeze.json'),
        accepted_predecessor_receipt=c.artifact(c.ACCEPTED_RECEIPT),
        preceding_optimization_closeout=c.artifact(c.LIVE_CLOSEOUT),
        authorization='User: ok lets do the 512 comparison you suggest',
        only_scientific_change='simulations_256_to_512',training=False,scheduled_monitor=False,
        original_network_both_routes=True,optimization_candidate_used=False))
    c.require('torch' not in sys.modules,'Preparation must not load learned models')
    return prepared


def require_tests(path):
    value=c.read(path)
    c.require(value['exit_code']==0 and value['models']==value['games']==0 and value['inputs'],
        'Passed actual model-free tests bound to current source')
    c.verify(value['inputs']+[value['log']]);return value


def require_review(path):
    value=c.read(path)
    c.require(value['approved'] is True and not value.get('blocking_findings',[]) and value['inputs'],
        'Independent current-source review must approve')
    c.verify(value['inputs']);return value


def freeze():
    from . import control,analysis
    c.require(not(c.OUT/'freeze.json').exists(),'Exclusive prospectively frozen512 attempt')
    prior,accepted=accepted_predecessor();prepared=prepare()
    test_path=c.OUT/'prelaunch/model_free_tests.json';tests=require_tests(test_path)
    reviews=[c.OUT/'prelaunch'/name for name in REVIEWS]
    for path in reviews:require_review(path)
    value={key:deepcopy(prior[key]) for key in RUNTIME_KEYS}
    value.update(control.contract())
    value.update(prepared)
    value.update(schema='splendorust_search512_strength_v1',time=c.now(),protocol=c.artifact(c.PROTOCOL),
        ruleset_id=RULESET_ID,catalogue_sha256=CATALOGUE_SHA256,turn_contract=TURN_CONTRACT,
        statistics_runtime=analysis.runtime_contract(),
        model_free_tests=c.artifact(test_path),implementation_reviews=[c.artifact(p) for p in reviews],
        accepted_predecessor=c.artifact(c.ACCEPTED_OUT/'freeze.json'),
        accepted_predecessor_receipt=c.artifact(c.ACCEPTED_RECEIPT),
        preceding_optimization_closeout=c.artifact(c.LIVE_CLOSEOUT),
        only_scientific_change='simulations_256_to_512',weights_unchanged=True,precision_unchanged=True,
        original_network_both_routes=True,optimization_candidate_used=False,reference_profile_unchanged=True,
        fresh_full_qualification_and_operational_run=True,original_failed_attempts_preserved=True,
        historical_retries_in_this_study=0,training=False,reference_promoted=False,
        source_preservation_scope='Imported local source closure, runtime/profile/checkpoint/native artifacts, qualification references and accepted receipts. Prior full experiment trees are not re-audited or modified.',
        baseline_profiles=deepcopy(prepared['profiles']['control']),
        research_profiles=deepcopy(prepared['profiles']['candidate']))
    pins={}
    def add(item):
        c.require(pins.setdefault(item['path'],item)==item,'Conflicting frozen input pin')
    for item in nested_artifacts(value):add(item)
    for item in nested_artifacts(c.read(c.OUT/'qualification_plan.json')):add(item)
    for route in ('control','candidate'):
        for item in value['profiles'][route].values():
            profile=c.read(c.ROOT/item['path'])
            for nested in nested_artifacts(profile):add(nested)
            for key in ('checkpoint','model_definition','native_binary'):add(c.artifact(c.ROOT/profile[key]))
    for item in tests['inputs']+c.read(c.OUT/'seed_freshness.json')['checked']:add(item)
    for path in reviews:
        for item in c.read(path)['inputs']:add(item)
    paths=list(Path(__file__).parent.glob('*.py'))
    paths+=list((c.ROOT/'tests').glob('test_splendorust_search512*_20261101.py'))
    for path in python_dependencies(paths):add(c.artifact(path))
    # Profiles/imports cover the actual inherited implementation; accepted receipts
    # preserve provenance without repeatedly scanning unrelated historical GBs.
    for path in [c.PROTOCOL,c.ROOT/'docs/engineering/future_training_rules_20261021.md',
        c.ACCEPTED_OUT/'freeze.json',c.ACCEPTED_RECEIPT,c.LIVE_OUT/'freeze.json',c.LIVE_CLOSEOUT]:add(c.artifact(path))
    for path in sorted(c.OUT.rglob('*')):
        if path.is_file() and '__pycache__' not in path.parts:add(c.artifact(path))
    value['inputs']=sorted(pins.values(),key=lambda item:item['path'])
    c.require(shutil.disk_usage(c.ROOT).free>=c.INITIAL_FREE_DISK_BYTES,'Initial37GiB disk reserve')
    c.require('torch' not in sys.modules,'No learned model import while freezing')
    c.verify(value['inputs']);control.validate_freeze_contract(value)
    return dict(status='frozen_not_launched',freeze=c.save(c.OUT/'freeze.json',value),inputs=len(pins),models=0,games=0)


def main():
    parser=argparse.ArgumentParser();parser.add_argument('action',choices=['prepare','freeze']);args=parser.parse_args()
    print(c.canonical({'prepare':prepare,'freeze':freeze}[args.action]()),flush=True)


if __name__=='__main__':main()
