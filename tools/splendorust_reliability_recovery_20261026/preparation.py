"""Metadata-only preparation of the one-shot saved-evidence recovery."""
import argparse
import time
import psutil
from . import common as c

PINS={
    'completion.json':'256c47139e0b5e7123edcc829252591847d3026c8bde6b5993a7f157567f9724',
    'failure.json':'08c0b04cbe13d5beafbeac80fdf550b29bdd9461422fc3c5544e830485affd2e',
    'execution.json':'634fa38887af747f666993b2d4f867b61495246aaa8cf36b7224ce0899dd8cc5'}

def inventory(root,boundary=lambda:None):
    result=set()
    for path in root.rglob('*'):
        boundary()
        c.require(not path.is_symlink(),'Symlink in original evidence')
        if path.is_file():
            resolved=path.resolve();c.require(resolved.is_relative_to(root.resolve()),'Original inventory path escape')
            result.add(resolved.relative_to(c.ROOT).as_posix())
    return result

def validate_snapshot(items,actual_paths):
    expected={row['path'] for row in items}
    c.require(len(expected)==len(items),'Duplicate original snapshot entry')
    for row in items:
        path=(c.ROOT/row['path']).resolve()
        c.require(path.is_relative_to(c.ORIGINAL) and path.relative_to(c.ROOT).as_posix()==row['path'],'Snapshot path escape/noncanonical path')
        c.require(type(row['bytes']) is int and row['bytes']>=0 and len(row['sha256'])==64,'Invalid snapshot artifact')
    c.require(expected==actual_paths,'Original snapshot inventory changed')

def sources():
    paths=list((c.ROOT/'tools/splendorust_reliability_recovery_20261026').glob('*.py'))
    paths+=list((c.ROOT/'tests').glob('test_splendorust_reliability_recovery*.py'))
    return sorted(paths+[c.PROTOCOL],key=str)

def prepare():
    start,wall=time.monotonic(),time.time()
    def guard():c.require(time.monotonic()-start<300 and time.time()-wall<300,'Metadata preparation deadline')
    c.require(not (c.OUT/'freeze.json').exists(),'Recovery already frozen')
    for name in ('watchdog_claim.json','execution.json','completion.json'):
        c.require(not (c.OUT/name).exists(),'Recovery attempt already started')
    tests=c.read(c.OUT/'prelaunch_tests_final.json');review=c.read(c.OUT/'prelaunch_review.json')
    c.require(tests['status']=='passed' and review['status']=='approved','Prelaunch tests/review not approved')
    current=[c.artifact(path,guard) for path in sources()]
    c.require(tests['sources']==current and review['sources']==current,'Tests/review do not bind current recovery sources')
    c.require(c.artifact(c.ORIGINAL/'freeze.json',guard)['sha256']==c.ORIGINAL_SHA,'Original freeze changed')
    for name,sha in PINS.items():c.require(c.artifact(c.ORIGINAL/name,guard)['sha256']==sha,'Original interruption changed: '+name)
    owner=c.read(c.ORIGINAL/'watchdog.json')
    for role in ('watchdog','runner'):
        try:
            p=psutil.Process(owner[role+'_pid'])
            c.require(p.create_time()!=owner[role+'_created'],'Original process still active')
        except psutil.NoSuchProcess:pass
    for name in ('coverage.json','verification_summary.json','stages/final_end.json','collection_stop.json'):
        c.require(not (c.ORIGINAL/name).exists(),'Original success marker appeared')
    seal_doc=c.read(c.ORIGINAL/'completed_evidence_seal.json')
    seal=seal_doc['artifacts']
    c.require(len(seal)==74528 and sum(x['bytes'] for x in seal)==700262089,'Original seal inventory differs')
    known={item['path']:item for item in seal};c.require(len(known)==len(seal),'Duplicate sealed path')
    actual=inventory(c.ORIGINAL,guard)
    c.require(set(known)<=actual,'Original sealed file missing')
    snapshot=[]
    for relative in sorted(actual):
        guard();path=c.ROOT/relative
        if relative in known:
            c.require(path.stat().st_size==known[relative]['bytes'],'Sealed size differs at preparation')
            snapshot.append(known[relative])
        else:snapshot.append(c.artifact(path,guard))
    validate_snapshot(snapshot,actual)
    c.verify(current,guard)
    inputs=current+[c.artifact(c.OUT/name,guard) for name in ('prelaunch_tests_final.json','prelaunch_review.json')]
    result=dict(schema='saved_reliability_recovery_freeze_v1',time=c.now(),job_seconds=c.JOB_SECONDS,
        preparation_seconds=max(time.monotonic()-start,time.time()-wall),automatic_retries=0,
        original_freeze_sha256=c.ORIGINAL_SHA,original_status='incomplete',original_exit_code=1,
        inputs=inputs,original_snapshot=snapshot,original_pins=PINS,
        sealed_hashes_reused_as_expectations_only=True,no_saved_scientific_acceptance_at_preparation=True,
        ruleset='splendor_base_qualified_rules_20261018_v1',catalogue_sha256='db4776918e612eee5a6f2617719c85ec5f413d592bbab0a64058babc83a70312',
        turn_interface='atomic_prereveal_completed_turn_v1',supervisor_cpus=list(range(2,20)),
        limits=dict(rss_bytes=c.MAX_RSS,new_output_bytes=c.MAX_OUTPUT,available_ram_bytes=c.MIN_AVAILABLE,free_disk_bytes=c.MIN_DISK),
        reused_decision_reproductions=48,fresh_games=0,fresh_models=0,fresh_decisions=0,larger_data_deferred=True)
    guard();receipt=c.save(c.OUT/'freeze.json',result)
    print(c.canonical(receipt),flush=True)
    return receipt

if __name__=='__main__':
    argparse.ArgumentParser().parse_args();prepare()
