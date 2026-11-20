"""Freeze only a separately reviewed continuation and its immutable source tree."""
import argparse
from copy import deepcopy
from pathlib import Path
import shutil
import time
from . import common as c
from tools.future_training_rules_20261021 import verify_runtime_sources
from tools.splendorust_search128_strength_20261028.review_runtime import ReviewRuntime


def old_inventory(*, boundary=lambda: None):
    c.require(c.file_hash(c.OLD_OUT/'freeze.json') == c.OLD_FREEZE_SHA, 'Original freeze changed')
    c.require(c.read(c.OLD_OUT/'completion.json')['status'] == 'incomplete', 'Original status changed')
    c.require(not (c.OLD_OUT/'postrun/independent_completion_review.json').exists(),
              'Original incomplete attempt cannot become accepted')
    values=[]
    for path in sorted(c.OLD_OUT.rglob('*')):
        boundary()
        if path.is_file(): values.append(c.artifact(path))
    return values


def preserve(bridge_path):
    clock=ReviewRuntime(c.ROOT,c.OUT,default_seconds=900.)
    clock.shared_deadlines=(c.ORIGINAL_MONOTONIC+c.COLLECTION_CUTOFF_SECONDS,
                            c.ORIGINAL_UNIX+c.COLLECTION_CUTOFF_SECONDS)
    original=c.read(c.OLD_OUT/'freeze.json')
    clock.verify_many([(c.ROOT/x['path'],x) for x in original['inputs']])
    artifacts=old_inventory(boundary=clock.guard)
    for name,key in [('owner_launch.json',None),('guardian_claim.json','guardian'),
                     ('watchdog_claim.json','watchdog'),('launch.json',None)]:
        value=c.read(c.OLD_OUT/name)
        identity=dict(pid=value['pid'] if key is None else value[key+'_pid'],
                      created=value['created'] if key is None else value[key+'_created'])
        clock.absent(identity)
    claim=c.read(c.OLD_OUT/'guardian_claim.json')
    c.require((claim['monotonic_origin'],claim['unix_origin']) ==
              (c.ORIGINAL_MONOTONIC,c.ORIGINAL_UNIX), 'Original real clock identity')
    bridge=c.read(bridge_path)
    value=dict(schema='search128_disclosed_recovery_manifest_v1',time=c.now(),
        old_freeze=c.artifact(c.OLD_OUT/'freeze.json'), original_claim=c.artifact(c.OLD_OUT/'guardian_claim.json'),
        original_tree=artifacts,original_tree_bytes=sum(x['bytes'] for x in artifacts),
        original_tree_files=len(artifacts),bridge=c.artifact(bridge_path),
        completed_main_games=38,partial_game_index=38,completed_prefix_plies=39,
        recovered_ply=39,continuation_ply=40,original_attempt_status='incomplete',
        original_failed_receipt_preserved=True,move_clock_reset=False,new_call_for_failed_move=False,
        original_monotonic_origin=c.ORIGINAL_MONOTONIC,original_unix_origin=c.ORIGINAL_UNIX,
        prior_final_seconds=c.PRIOR_FINAL_SECONDS,prior_ram_wait_seconds=c.PRIOR_RAM_WAIT_SECONDS,
        prior_ram_wait_is_conservative_allocation=True,maximum_combined_worker_lifecycles=c.MAX_WORKERS,
        new_entrypoint_decisions=119,qualified_prior_decisions=203,automatic_retries=0)
    clock.guard()
    return c.save(c.OUT/'recovery_manifest.json',value)


def freeze():
    c.require(not (c.OUT/'freeze.json').exists(),'Never replace a recovery freeze')
    c.require(shutil.disk_usage(c.ROOT).free>=c.INITIAL_FREE_DISK_BYTES,'Original launch disk reserve')
    old=c.read(c.OLD_OUT/'freeze.json')
    c.require(c.file_hash(c.OLD_OUT/'freeze.json')==c.OLD_FREEZE_SHA,'Original freeze SHA')
    manifest=c.read(c.OUT/'recovery_manifest.json')
    tests=c.read(c.OUT/'preparation/model_free_tests.json')
    review=c.read(c.OUT/'preparation/implementation_review.json')
    c.require(tests['passed'] is True and tests['exit_code']==0,'Qualified model-free tests required')
    c.require(review['approved'] is True,'Independent implementation approval required')
    c.verify(review['inputs'])
    inputs={x['path']:x for x in old['inputs']}
    def add(item):
        c.require(item['path'] not in inputs or inputs[item['path']]==item,'Frozen input collision')
        inputs[item['path']]=item
    for item in manifest['original_tree']:add(item)
    for path in sorted(Path(__file__).parent.glob('*.py')):add(c.artifact(path))
    for path in sorted((c.ROOT/'tests').glob('test_splendorust_search128_recovery*20261028*.py')):add(c.artifact(path))
    for path in sorted(c.OUT.rglob('*')):
        if path.is_file():add(c.artifact(path))
    add(c.artifact(c.PROTOCOL))
    source_guard=verify_runtime_sources()
    runtime=c.save(c.OUT/'runtime_sources.json',source_guard);add(runtime)
    frozen=deepcopy(old)
    frozen.update(schema='splendorust_search128_disclosed_recovery_v1',time=c.now(),attempt=1,
        protocol=c.artifact(c.PROTOCOL),stage_seconds=c.STAGES,max_workers=c.MAX_WORKERS,
        max_graph_builds=c.MAX_GRAPH_BUILDS,recovery_manifest=c.artifact(c.OUT/'recovery_manifest.json'),
        original_freeze=c.artifact(c.OLD_OUT/'freeze.json'),
        original_monotonic_origin=c.ORIGINAL_MONOTONIC,original_unix_origin=c.ORIGINAL_UNIX,
        prior_final_seconds=c.PRIOR_FINAL_SECONDS,prior_ram_wait_seconds=c.PRIOR_RAM_WAIT_SECONDS,
        recovery_entrypoint_decisions=119,recovery_entrypoint_parent_requests=2,
        source_attempt_remains_incomplete=True,recovered_response_policy='adopt_single_authenticated_already_emitted_response',
        partial_game_resume_ply=40,original_failed_receipt_overwritten=False,
        inputs=sorted(inputs.values(),key=lambda x:x['path']))
    from .runner import validate_freeze_contract
    validate_freeze_contract(frozen)
    c.verify(frozen['inputs'])
    c.require(time.monotonic()<c.ORIGINAL_MONOTONIC+c.COLLECTION_CUTOFF_SECONDS and
              time.time()<c.ORIGINAL_UNIX+c.COLLECTION_CUTOFF_SECONDS,'Original collection window exhausted')
    return c.save(c.OUT/'freeze.json',frozen)


if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('action',choices=['preserve','freeze'])
    parser.add_argument('--bridge');args=parser.parse_args()
    print(c.canonical(preserve(Path(args.bridge)) if args.action=='preserve' else freeze()))
