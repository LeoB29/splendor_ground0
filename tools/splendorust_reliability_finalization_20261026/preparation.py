"""Freeze only metadata and new source; no scientific validation or full scan."""
import time
import psutil
from . import common as c, preservation as p

def sources():
    paths=list((c.ROOT/'tools/splendorust_reliability_finalization_20261026').glob('*.py'))
    paths+=list((c.ROOT/'tests').glob('test_splendorust_reliability_finalization*.py'))
    return sorted(paths+[c.PROTOCOL],key=str)

def merge_artifacts(groups):
    result={};folded={}
    for group in groups:
        for item in group:
            path=item['path'];p.safe_relative(path)
            c.require(path.casefold() not in folded or folded[path.casefold()]==path,'Conflicting case-aliased frozen input')
            c.require(path not in result or result[path]==item,'Conflicting frozen input identity')
            result[path]=item;folded[path.casefold()]=path
    return sorted(result.values(),key=lambda row:row['path'])

def require_owners_absent():
    owners=[]
    old=c.read(c.ORIGINAL/'watchdog.json')
    owners.extend(dict(pid=old[role+'_pid'],created=old[role+'_created']) for role in ('runner','watchdog'))
    for root in (c.V1,c.V2):
        saved=c.read(root/'watchdog.json');owners.extend(saved[role] for role in ('auditor','watchdog'))
    for identity in owners:
        try:c.require(psutil.Process(identity['pid']).create_time()!=identity['created'],'Historical owner still active')
        except psutil.NoSuchProcess:pass
    return [{**identity,'same_identity_absent':True} for identity in owners]

def prepare():
    from .receipts import REQUIRED_RECEIPTS
    start,wall=time.monotonic(),time.time()
    def guard():c.require(time.monotonic()-start<300 and time.time()-wall<300,'Finalization metadata preparation deadline')
    c.require(not any((c.OUT/name).exists() for name in ('freeze.json','watchdog_claim.json','execution.json')),'Finalization already frozen or started')
    current=[c.artifact(path,guard) for path in sources()]
    for name in ('prelaunch_tests.json','prelaunch_review.json'):
        receipt=c.read(c.OUT/name)
        c.require(receipt['status'] in ('passed','approved') and receipt['sources']==current,'Prelaunch check/source binding differs')
    for root,sha in ((c.ORIGINAL,c.ORIGINAL_SHA),(c.V1,c.V1_SHA),(c.V2,c.V2_SHA)):
        c.require(c.artifact(root/'freeze.json',guard)['sha256']==sha,'Historical freeze changed')
    c.require(c.artifact(c.V2/'independent_receipt_review.json',guard)['sha256']==c.INDEPENDENT_SHA,'Pinned partial review changed')
    old=c.read(c.ORIGINAL/'freeze.json');v1=c.read(c.V1/'freeze.json');v2=c.read(c.V2/'freeze.json')
    original=v2['original_snapshot'];c.require(len(original)==74579,'Original snapshot length differs')
    original_root=c.ORIGINAL.relative_to(c.ROOT).as_posix();p.expected_index(original_root,original)
    first=v2['first_recovery_preservation'];first_root=c.V1.relative_to(c.ROOT).as_posix()
    current_first,_=p.inventory(first_root,guard)
    c.require(set(current_first)=={row['path'] for row in first},'First recovery inventory changed')
    c.verify(first,guard)
    second_root=c.V2.relative_to(c.ROOT).as_posix();second_names,_=p.inventory(second_root,guard)
    second=[c.artifact(c.ROOT/name,guard) for name in sorted(second_names)]
    trees={original_root:original,first_root:first,second_root:second}
    protected=merge_artifacts(trees.values())
    protected_index={row['path']:row for row in protected}
    source_items=merge_artifacts((old['inputs'],v1['inputs'],v2['inputs']))
    for item in source_items:
        if item['path'] in protected_index:c.require(item==protected_index[item['path']],'Protected source overlap differs')
    source_items=[item for item in source_items if item['path'] not in protected_index]
    pins=[c.artifact(c.ROOT/name,guard) for name in REQUIRED_RECEIPTS]
    inputs=current+[c.artifact(c.OUT/name,guard) for name in ('prelaunch_tests.json','prelaunch_review.json')]
    if (c.OUT/'prelaunch_boundary_initial_failure.json').exists():
        inputs.append(c.artifact(c.OUT/'prelaunch_boundary_initial_failure.json',guard))
    c.verify(current,guard);owners=require_owners_absent();guard()
    value=dict(schema='splendorust_finalization_freeze_v1',time=c.now(),job_seconds=900,automatic_retries=0,
        original_snapshot=original,preservation_trees=trees,source_preservation=source_items,
        receipt_pins=pins,inputs=inputs,historical_owners=owners,
        original_freeze_sha256=c.ORIGINAL_SHA,v1_freeze_sha256=c.V1_SHA,v2_freeze_sha256=c.V2_SHA,
        independent_partial_review_sha256=c.INDEPENDENT_SHA,
        reused_scientific_receipts=True,scientific_replays_to_run=0,models_to_run=0,games_to_run=0,
        ruleset='splendor_base_qualified_rules_20261018_v1',catalogue_sha256='db4776918e612eee5a6f2617719c85ec5f413d592bbab0a64058babc83a70312',
        turn_interface='atomic_prereveal_completed_turn_v1',supervisor_cpus=list(range(2,20)),
        limits=dict(rss_bytes=c.MAX_RSS,output_bytes=c.MAX_OUTPUT,available_ram_bytes=c.MIN_AVAILABLE,free_disk_bytes=c.MIN_DISK),
        original_status='incomplete',v1_status='failed',v2_status='incomplete',
        preparation_seconds=max(time.monotonic()-start,time.time()-wall),no_scientific_acceptance_during_preparation=True,
        new_separately_authorized_clock=True,larger_data_deferred=True)
    receipt=c.save(c.OUT/'freeze.json',value);print(c.canonical(receipt),flush=True);return receipt

if __name__=='__main__':prepare()
