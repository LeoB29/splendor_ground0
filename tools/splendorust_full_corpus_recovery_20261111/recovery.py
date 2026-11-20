"""Immutable predecessor preservation and one explicitly authorized amendment."""
from concurrent.futures import ThreadPoolExecutor
from copy import deepcopy
import hashlib
import os
from pathlib import Path
import time
from . import common as c


def dual_deadline():
    old=c.read(c.ORIGINAL_OUT/'claim.json'); phase=c.read(c.ORIGINAL_OUT/'stages/strength_start.json')
    c.require(old['monotonic_origin']<=time.monotonic()<phase['monotonic_deadline'] and
              old['unix_origin']<=time.time()<phase['unix_deadline'], 'Original remaining strength clocks')


def _paths(base,boundary=lambda:None):
    base=Path(base).resolve(); paths=[]
    for directory,dirs,files in os.walk(base,followlinks=False):
        boundary()
        for name in dirs+files:
            path=Path(directory)/name
            c.require(not path.is_symlink() and not (getattr(path.lstat(),'st_file_attributes',0)&0x400),
                      'No reparse/symlink evidence aliases')
        paths.extend(Path(directory)/name for name in files)
    return sorted(paths)


def inventory_tree(base,path,boundary=lambda:None):
    base=Path(base).resolve();path=Path(path).resolve()
    c.require(not path.is_relative_to(base),'Inventory must not mutate preserved source')
    paths=_paths(base,boundary); count=total=0
    def descriptor(p):
        before=p.stat(); sha=c.sha(p); after=p.stat()
        c.require((before.st_size,before.st_mtime_ns)==(after.st_size,after.st_mtime_ns),'Immutable source during hashing')
        return dict(path=p.relative_to(c.ROOT).as_posix(),bytes=before.st_size,sha256=sha)
    with path.open('xb') as stream,ThreadPoolExecutor(max_workers=4) as pool:
        for start in range(0,len(paths),128):
            boundary()
            for item in pool.map(descriptor,paths[start:start+128]):
                stream.write((c.canonical(item)+'\n').encode());count+=1;total+=item['bytes']
            c.status(c.OUT/'preservation/progress.json',dict(time=c.now(),files=count,total_files=len(paths),bytes=total))
        stream.flush();os.fsync(stream.fileno())
    return dict(root=base.relative_to(c.ROOT).as_posix(),inventory=c.artifact(path),files=count,bytes=total)


def verify_inventory(value,boundary=lambda:None):
    c.verify([value['inventory']],boundary=boundary)
    base=c.ROOT/value['root']; observed=set();count=total=0
    with c.resolve(value['inventory']).open(encoding='utf-8') as stream:
        for line in stream:
            boundary(); item=__import__('json').loads(line)
            path=c.resolve(item)
            c.require(path.is_relative_to(base.resolve()) and item['path'] not in observed,'Unique preserved in-root file')
            c.verify([item],boundary=boundary);observed.add(item['path']);count+=1;total+=item['bytes']
    actual={p.relative_to(c.ROOT).as_posix() for p in _paths(base,boundary)}
    c.require(actual==observed and count==value['files'] and total==value['bytes'],
              'Exact unchanged predecessor inventory, no removed/added files')
    return dict(files=count,bytes=total,status='preserved')


def prepare():
    from . import boundary as boundary_checks
    c.OUT.mkdir(parents=True,exist_ok=True)
    c.require(not (c.OUT/'reuse.json').exists() and not c.PLAN.exists(),'Exclusive new recovery preparation')
    dual_deadline(); authorization=boundary_checks.validate_retry_authorization(c.read(c.AUTHORIZATION))
    source=boundary_checks.check_source_boundary(boundary=dual_deadline,deep=True)
    c.verify(c.read(c.ORIGINAL_PLAN)['inputs'],boundary=dual_deadline)
    physical=inventory_tree(c.ORIGINAL_OUT,c.OUT/'original_inventory.jsonl',dual_deadline)
    value=dict(schema='full_corpus_amended_reuse_v1',time=c.now(),source=source,
        original_inventory=physical,preserved_bytes=physical['bytes'],preserved_files=physical['files'],
        authorization=c.artifact(c.AUTHORIZATION),original_freeze=c.artifact(c.ORIGINAL_PLAN),
        original_owner_absence=c.artifact(c.OUT/'original_owner_absence.json'),
        original_claim=c.artifact(c.ORIGINAL_OUT/'claim.json'),
        original_execution=c.artifact(c.ORIGINAL_OUT/'execution.json'),
        original_failure=c.artifact(c.ORIGINAL_OUT/'failure.json'),
        original_strength_phase=c.artifact(c.ORIGINAL_OUT/'stages/strength_start.json'),
        original_commitment=c.artifact(c.ORIGINAL_OUT/'all_fits_committed.json'),
        original_offline=c.artifact(c.ORIGINAL_OUT/'offline/worker/result.json'),
        completed_main_games=212,completed_operational_games=36,partial_saved_plies=22,
        retry_spec=c.RETRY_SPEC,new_models=0,new_games=0,original_attempt_status='incomplete')
    return c.save(c.OUT/'reuse.json',value)


def validate_reuse(value,boundary=lambda:None):
    from . import boundary as boundary_checks
    c.require(value['schema']=='full_corpus_amended_reuse_v1' and
              value['completed_main_games']==212 and value['completed_operational_games']==36 and
              value['partial_saved_plies']==22 and value['retry_spec']==c.RETRY_SPEC and
              value['new_models']==value['new_games']==0 and value['original_attempt_status']=='incomplete',
              'Exact fixed scientific reuse, no changed sample or model')
    boundary_checks.validate_retry_authorization(c.read(c.AUTHORIZATION))
    c.require(value['authorization']==c.artifact(c.AUTHORIZATION) and
              value['original_freeze']==c.artifact(c.ORIGINAL_PLAN) and c.sha(c.ORIGINAL_PLAN)==c.ORIGINAL_PLAN_SHA,
              'Exact source freeze and explicit retry authorization')
    c.verify(list(c.nested_artifacts(value)),boundary=boundary)
    c.require(value['preserved_bytes']==value['original_inventory']['bytes'] and
              value['preserved_files']==value['original_inventory']['files'],'No reset of existing evidence budget')
    return value


def original_inventory_verify(boundary=lambda:None):
    return verify_inventory(validate_reuse(c.read(c.OUT/'reuse.json'),boundary)['original_inventory'],boundary)


def validate_recovery_claim(claim):
    old=c.read(c.ORIGINAL_OUT/'claim.json')
    c.require(claim['original_claim']==c.artifact(c.ORIGINAL_OUT/'claim.json') and
              claim['reuse']==c.artifact(c.OUT/'reuse.json') and claim['authorization']==c.artifact(c.AUTHORIZATION) and
              claim['freeze']==c.artifact(c.PLAN) and claim['plan_sha256']==c.sha(c.PLAN),
              'Exact separate owner, freeze, authorization and predecessor')
    for key in ('monotonic_origin','unix_origin','total_seconds','model_seconds'):
        c.require(type(claim[key]) is type(old[key]) and claim[key]==old[key],'Original unextended '+key)
    c.require(type(claim['automatic_retries']) is int and claim['automatic_retries']==0 and
              type(claim['authorized_retry_limit']) is int and claim['authorized_retry_limit']==1 and
              claim['retry_spec']==c.RETRY_SPEC and claim['amended_recovered_study'] is True,
              'Only single specifically authorized retry, no automatic retries')
    c.require((claim['pid'],claim['created'])!=(old['pid'],old['created']),'New exact root owner identity')
    return claim


def recover_phase(claim):
    validate_recovery_claim(claim)
    descriptor=c.artifact(c.ORIGINAL_OUT/'stages/strength_start.json')
    phase=deepcopy(c.read(c.resolve(descriptor)))
    phase.update(claim=c.artifact(c.OUT/'claim.json'),original_phase=descriptor)
    from .owner import guard
    guard(claim,phase)
    return c.save(c.OUT/'stages/strength_start.json',phase)


def remaining_blocks():
    # Preserve the original alternating block order. The interrupted block is
    # continued separately (g032..039), never replayed from g030.
    return [(matchup,rep,block) for block in range(0,200,10)
            for matchup in c.MATCHUPS for rep in (0,1)
            if block>=((c.COMPLETED_PREFIXES[matchup][rep]+9)//10)*10]


def expected_live_segments():
    from .owner import segment_id
    rows=[segment_id('prepare',rep,matchup=matchup) for matchup in c.MATCHUPS for rep in (0,1)]
    rows.extend(segment_id('qualification',rep,matchup=matchup) for matchup in c.MATCHUPS for rep in (0,1))
    rows.append(segment_id('resume_partial',1,matchup='terminal_parent'))
    rows.extend(segment_id('main_block',rep,block,matchup) for matchup,rep,block in remaining_blocks())
    return rows


def freeze(tests,reviews):
    from .owner import contract,source_inputs,validate_test_receipt
    from tools.future_training_rules_20261021 import verify_runtime_sources
    c.require(not c.PLAN.exists(),'Separate immutable amended freeze')
    dual_deadline();reuse=validate_reuse(c.read(c.OUT/'reuse.json'),dual_deadline)
    original_inventory_verify(dual_deadline)
    attestation_path=c.OUT/'prelaunch/boundary_runtime_attestation.json'
    attestation=c.read(attestation_path)
    c.require(attestation['status']=='saved_prefix_verified_with_disclosed_dispatch_ambiguity' and
              attestation['models']==attestation['games']==0 and
              attestation['checked_partial_moves']==22 and attestation['raw_requests']==138 and
              attestation['helper_execution']['worker_exit_code']==0 and
              attestation['helper_execution']['identity_absent'] is True and
              attestation['runtime_sources']==c.read(c.ORIGINAL_PLAN)['corrected_source_receipt'],
              'Current actual model-free boundary helper and corrected runtime qualification')
    proof=validate_test_receipt(c.read(tests));c.verify(proof['inputs']+[proof['log']],boundary=dual_deadline)
    code=source_inputs();c.require(proof['inputs']==code,'Tests bind complete current source')
    required=code+[c.artifact(p) for p in (c.PROTOCOL,c.SELECTION,c.AUTHORIZATION,c.OUT/'reuse.json')]
    authors=[]
    for path in reviews:
        review=c.read(path);c.require(review['approved'] is True and not review['blocking_findings'],'Independent reviewer approval')
        c.verify(review['inputs'],boundary=dual_deadline)
        covered={str(c.resolve(x)).casefold():x for x in review['inputs']}
        c.require(all(covered.get(str(c.resolve(x)).casefold())==x for x in required),'Every review binds current recovery')
        authors.append(review['reviewer'])
    c.require(len(set(authors))>=3,'Three independent source-bound recovery reviews')
    value=deepcopy(c.read(c.ORIGINAL_PLAN));value.update(contract())
    value.update(protocol=c.artifact(c.PROTOCOL),original_full_corpus_freeze=c.artifact(c.ORIGINAL_PLAN),
        recovery_reuse=c.artifact(c.OUT/'reuse.json'),retry_authorization=c.artifact(c.AUTHORIZATION),
        saved_boundary_runtime_attestation=c.artifact(attestation_path),
        user_authorization=c.read(c.AUTHORIZATION)['user_reply'],
        tests=c.artifact(tests),reviews=[c.artifact(p) for p in reviews],time=c.now())
    sources=verify_runtime_sources();c.require(sources==value['corrected_source_receipt'],'Unchanged corrected runtime sources')
    runtime=[dict(path=x['absolute_path'],bytes=x['bytes'],sha256=x['sha256']) for x in value['torch_runtime']]
    value['worker_inputs']=code+runtime+[c.artifact(c.PARENT_CHECKPOINT),c.artifact(c.AUTHORIZATION),c.artifact(c.OUT/'reuse.json')]
    entries=code+runtime+c.read(c.ORIGINAL_PLAN)['inputs']+required+list(c.nested_artifacts(reuse))+[c.artifact(attestation_path)]
    for matchup in c.MATCHUPS:
        for rep in (0,1):
            old_local=c.ORIGINAL_OUT/'strength'/matchup/f'r{rep}'/'freeze.json'
            entries += [c.artifact(old_local)]+list(c.nested_artifacts(c.read(old_local)))
    entries += list(c.nested_artifacts(c.read(c.ORIGINAL_OUT/'all_fits_committed.json')))
    for path in [tests,*reviews]:entries += [c.artifact(path)]+list(c.nested_artifacts(c.read(path)))
    unique={}
    for item in entries:
        key=str(c.resolve(item)).casefold();old=unique.setdefault(key,item)
        c.require(old==item,'Consistent immutable recovery source')
    value['inputs']=sorted(unique.values(),key=lambda row:row['path']);c.verify(value['inputs'],boundary=dual_deadline)
    return c.save(c.PLAN,value)


if __name__=='__main__':
    import argparse
    p=argparse.ArgumentParser();p.add_argument('operation',choices=('prepare','verify'))
    a=p.parse_args();print(c.canonical(prepare() if a.operation=='prepare' else original_inventory_verify(dual_deadline)),flush=True)
