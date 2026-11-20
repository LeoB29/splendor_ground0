"""Bounded preservation and prospective freeze, without any scientific execution."""
import argparse
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
import sys
import time
from . import common as c
from tools.splendorust_search128_strength_20261028.review_runtime import ReviewRuntime


def preserve():
    r=ReviewRuntime(c.ROOT,c.SOURCE,900.)
    final=r.read('stages/final_start.json')
    r.shared_deadlines=(final['monotonic_deadline'],final['unix_deadline'])
    c.require(r.pin(c.SOURCE/'freeze.json')['sha256']==c.SOURCE_SHA,'Exact source freeze')
    historical=c.historical_failure(r)
    trees=[]
    with ThreadPoolExecutor(max_workers=8,thread_name_prefix='audit-preservation') as pool:
        for root in c.ROOTS:
            paths=sorted(p for p in root.rglob('*') if p.is_file())
            items=[]
            # At most16 submitted files, retaining the proven reader guards.
            for offset in range(0,len(paths),16):
                r.guard();items.extend(pool.map(r.pin,paths[offset:offset+16]))
            r.exact_tree(root,[x['path'] for x in items])
            trees.append(dict(root=root.relative_to(c.ROOT).as_posix(),artifacts=items))
    integrity=r.revalidate_identities();r.guard()
    manifest=dict(status='preserved_before_new_audit',time=c.now(),source_freeze_sha256=c.SOURCE_SHA,
        trees=trees,total_bytes=sum(x['bytes'] for tree in trees for x in tree['artifacts']),
        total_files=sum(len(tree['artifacts']) for tree in trees),historical_failure=historical,
        identity_validation=integrity,elapsed_seconds=time.monotonic()-r.start,model_calls=0,new_games=0)
    c.require(manifest['total_bytes']+c.output_size()+c.MAX_OUTPUT_BYTES<16*c.GiB,'Combined evidence cap with audit reserve')
    return c.save(c.OUT/'preservation_manifest.json',manifest)


def freeze():
    c.require(not(c.OUT/'freeze.json').exists(),'Never overwrite an audit freeze')
    tests=c.read(c.OUT/'prelaunch/model_free_tests.json');review=c.read(c.OUT/'prelaunch/implementation_review.json')
    c.require(tests['passed'] is True and tests['exit_code']==0 and review['approved'] is True,'Tests and independent implementation review')
    c.verify_pins(tests['inputs']);c.verify_pins(review['inputs'])
    manifest=c.read(c.OUT/'preservation_manifest.json')
    c.require(manifest['source_freeze_sha256']==c.SOURCE_SHA and manifest['status']=='preserved_before_new_audit','Preservation prerequisite')
    paths=sorted(Path(__file__).parent.glob('*.py'))+sorted((c.ROOT/'tests').glob('test_splendorust_search128_saved_audit*20261029.py'))
    paths += [c.PROTOCOL,c.OUT/'prelaunch/model_free_tests.json',c.OUT/'prelaunch/implementation_review.json',c.OWNER_EXIT]
    inputs=[c.artifact(p) for p in paths]
    value=dict(schema='search128_separate_saved_recovery_audit_v1',time=c.now(),source_freeze=c.artifact(c.SOURCE/'freeze.json'),
        implementation_inputs=inputs,preservation_manifest=c.artifact(c.OUT/'preservation_manifest.json'),
        maximum_seconds=c.LIMIT,clipped_to_original_final_and_global=True,original_review_clock_reset=False,
        maximum_rss_bytes=c.MAX_RSS_BYTES,minimum_initial_available_ram_bytes=4*c.GiB,minimum_free_disk_bytes=5*c.GiB,
        maximum_output_bytes=c.MAX_OUTPUT_BYTES,maximum_combined_evidence_bytes=16*c.GiB,
        models=0,new_games=0,scientific_replays=0,bootstrap_runs=0,failed_attempts_remain_unaccepted=True,
        actual_reader_acceptance_owner_exits_required=True,automatic_retries=0,
        python=sys.version,python_executable=dict(path=sys.executable,sha256=c.file_hash(sys.executable)))
    return c.save(c.OUT/'freeze.json',value)


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('mode',choices=['preserve','freeze']);args=p.parse_args()
    print(c.canonical(preserve() if args.mode=='preserve' else freeze()))
