"""Recover verification from saved evidence without executing any player."""
import argparse
import os
from pathlib import Path
import time
import traceback
import psutil
from . import common as c
from .path_runtime import AuthorizedRuntimeReview
from tools.splendorust_search128_resume_20261029 import independent_review as frozen
from tools.future_training_rules_20261021 import verify_runtime_sources


def initialize(sha):
    clock=c.read(c.OUT/'owner_clock.json');c.guard_clock(clock)
    c.require(c.file_hash(c.OUT/'freeze.json')==sha and clock['freeze_sha256']==sha,'Exact new audit freeze')
    freeze=c.read(c.OUT/'freeze.json')
    c.verify_pins(freeze['implementation_inputs'])
    r=AuthorizedRuntimeReview(c.ROOT,c.SOURCE,default_seconds=c.LIMIT)
    r.shared_deadlines=(clock['deadline_monotonic'],clock['deadline_unix']);r.guard()
    r.verify_many([(r.physical(x['path']),x) for x in freeze['implementation_inputs']])
    c.require(r.pin(c.SOURCE/'freeze.json')['sha256']==c.SOURCE_SHA,'Exact immutable source experiment')
    historical=c.historical_failure(r)
    final=historical['original_final_clock']
    c.require(c.deadline_pair(clock['monotonic_origin'],clock['unix_origin'],final)==r.shared_deadlines,
              'New explicit recovery clock clipped to unchanged final/global limits')
    r.verify(freeze['preservation_manifest'])
    manifest=r.read_path(c.ROOT/freeze['preservation_manifest']['path'])
    return r,freeze,clock,manifest,historical


def preserve(r,manifest,*,hashes):
    for tree in manifest['trees']:
        path=c.ROOT/tree['root'];items=tree['artifacts']
        r.exact_tree(path,[x['path'] for x in items])
        if hashes:r.verify_many([(r.physical(x['path']),x) for x in items])
    r.guard()
    return dict(status='passed',files=sum(len(x['artifacts']) for x in manifest['trees']),
        bytes=manifest['total_bytes'],all_four_original_trees_unchanged=True)


def publish(r,path,value,manifest):
    r.guard()
    c.require(manifest['total_bytes']+c.output_size()+len(c.canonical(value).encode())+1024**2<16*c.GiB,
              'Combined old evidence and recovery audit cap')
    pin=c.save(path,value);r.guard();return pin


def component(sha):
    r,freeze,clock,manifest,historical=initialize(sha)
    runtime=verify_runtime_sources()
    publish(r,c.OUT/'runtime_sources.json',runtime,manifest)
    preserved=preserve(r,manifest,hashes=True)
    result=frozen.review_all(r,c.SOURCE_SHA)
    preserve(r,manifest,hashes=False)
    checked=r.revalidate_identities()
    identities=publish(r,c.OUT/'identity_inventory.json',frozen.identity_receipt(r),manifest)
    value=dict(status='passed_saved_recovery_component',accepted=False,audit_freeze_sha256=sha,
        source_freeze_sha256=c.SOURCE_SHA,reader=dict(pid=os.getpid(),created=psutil.Process().create_time()),
        reader_source=c.artifact(Path(__file__)),path_runtime=c.artifact(Path(__file__).with_name('path_runtime.py')),
        clock=clock,historical_failure=historical,preservation=preserved,result=result,
        identity_inventory=identities,final_identity_validation=checked,authentication_metrics=dict(r.metrics),
        peak_reader_rss_bytes=r._peak_rss,elapsed_seconds=time.monotonic()-r.start,
        model_calls=0,new_games=0,search_calls=0,scientific_replays=0,bootstrap_executions=0,
        original_attempts_accepted=False,source_attempt_accepted=False,
        acceptance_requires_actual_reader_exit0=True)
    publish(r,c.OUT/'component.json',value,manifest)
    print(c.canonical(dict(status=value['status'],counts=result['counts'],elapsed_seconds=value['elapsed_seconds'])),flush=True)
    return 0


def check_reader_terminal(r,value,pin,terminal,sha):
    c.require(value['status']=='passed_saved_recovery_component' and value['accepted'] is False and
        value['audit_freeze_sha256']==terminal['freeze_sha256']==sha and
        value['source_freeze_sha256']==c.SOURCE_SHA and terminal['exit_code']==0 and
        terminal['cleanup_reason'] is None and terminal['observed_by']=='owned Process handle' and
        terminal['pid']==value['reader']['pid'] and abs(terminal['created']-value['reader']['created'])<.0001 and
        terminal['component_sha256']==pin['sha256'] and terminal['script_sha256']==value['reader_source']['sha256']==
        r.pin(Path(__file__))['sha256'],'Exact successful new reader actual exit and component')
    c.require(value['original_attempts_accepted'] is False and value['source_attempt_accepted'] is False and
        all(value[x]==0 for x in ('model_calls','new_games','search_calls','scientific_replays','bootstrap_executions')),
        'Saved-only scope and failed historical disposition')


def accept(sha):
    r,freeze,clock,manifest,historical=initialize(sha)
    runtime=verify_runtime_sources()
    c.require(not(c.OUT/'failure.json').exists() and not(c.OUT/'acceptance_invalidated.json').exists(),
              'No failed new audit may be accepted')
    value=r.read_path(c.OUT/'component.json');pin=r.pin(c.OUT/'component.json')
    terminal=r.read_path(c.OUT/'reader_os_exit.json')
    check_reader_terminal(r,value,pin,terminal,sha)
    stable=lambda v:{k:x for k,x in v.items() if k!='owned_identities_absent'}
    c.require(value['clock']==clock and stable(value['historical_failure'])==stable(historical),
              'Same recovery clock and immutable old failure; original process absence rechecked')
    r.absent(value['reader'])
    r.verify(value['identity_inventory']);identities=r.read_path(c.ROOT/value['identity_inventory']['path'])
    checked=frozen.verify_identity_receipt(r,identities)
    preserved=preserve(r,manifest,hashes=False)
    r.verify_many([(r.physical(x['path']),x) for x in freeze['implementation_inputs']])
    r.revalidate_identities()
    receipt=dict(status='recovered_verification_pending_actual_acceptance_and_owner_exits',
        recovered_verification=True,original_acceptance=False,source_attempt_accepted=False,
        audit_freeze_sha256=sha,source_freeze_sha256=c.SOURCE_SHA,component=pin,
        reader_exit=r.pin(c.OUT/'reader_os_exit.json'),clock=clock,historical_failure=historical,
        preservation=preserved,final_identity_check=checked,runtime_sources=runtime,
        model_calls=0,new_games=0,search_calls=0,scientific_replays=0,bootstrap_executions=0,
        replaces_only_failed_publication_condition=True,
        publication_condition='Separate new reader, acceptance and owner must actually exit0; original failed attempts remain unaccepted.',
        result=value['result'],time=c.now())
    publish(r,c.OUT/'acceptance.json',receipt,manifest)
    print(c.canonical(dict(status=receipt['status'],recovered_verification=True)),flush=True)
    return 0


def main():
    p=argparse.ArgumentParser();p.add_argument('mode',choices=['component','accept']);p.add_argument('--freeze-sha256',required=True)
    args=p.parse_args()
    try:return component(args.freeze_sha256) if args.mode=='component' else accept(args.freeze_sha256)
    except BaseException as exc:
        name='failure.json' if args.mode=='component' else 'acceptance_invalidated.json'
        if not(c.OUT/name).exists():c.save(c.OUT/name,dict(status='failed_no_recovered_acceptance',error=repr(exc),
            traceback=traceback.format_exc(),time=c.now(),model_calls=0,new_games=0,source_attempt_unchanged=True))
        print(c.canonical(dict(status='failed',error=repr(exc))),flush=True);return 1


if __name__=='__main__':raise SystemExit(main())
