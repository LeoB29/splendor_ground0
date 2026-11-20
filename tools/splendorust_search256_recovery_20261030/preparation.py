"""Finite immutable evidence seal and separate, explicitly authorized recovery."""
import argparse
from copy import deepcopy
from pathlib import Path
import shutil
import sys
import time

from . import common as c, design
from tools.splendorust_search256_strength_20261029_02.preparation import python_dependencies
from tools.future_training_rules_20261021 import verify_runtime_sources

SCHEMA = 'splendorust_search256_disclosed_recovery_v1'
MANIFEST_SCHEMA = 'splendorust_search256_disclosed_recovery_manifest_v1'
MAX_TREE_FILES = 120000
PRESERVATION_SECONDS = 900.


def fixed(path, value):
    if Path(path).exists():
        c.require(c.read(path) == value, 'Existing prepared artifact differs')
        return c.artifact(path)
    return c.save(path, value)


def original_freeze():
    c.require(c.file_hash(c.OLD_OUT/'freeze.json') == c.OLD_FREEZE_SHA, 'Exact interrupted freeze')
    return c.read(c.OLD_OUT/'freeze.json')


def contract():
    old = original_freeze()
    keys = ('phase','max_rss_bytes','initial_available_ram_bytes','minimum_available_ram_bytes',
        'initial_ram_stability_seconds','maximum_ram_wait_seconds','startup_minimum_available_ram_bytes',
        'initial_free_disk_bytes','minimum_free_disk_bytes','seconds_per_move','internal_seconds',
        'research_seconds_per_move','research_internal_seconds','research_log_max_bytes','frozen_log_max_bytes',
        'research_response_max_bytes','frozen_response_max_bytes','request_max_bytes','placement_mode',
        'max_resident_workers','operational_games','main_games','maximum_reproductions','training',
        'seven_hundred_ms_gate','one_second_gate')
    result={key:deepcopy(old[key]) for key in keys}
    result.update(job_seconds=c.JOB_SECONDS,collection_cutoff_seconds=c.COLLECTION_CUTOFF_SECONDS,
        stage_seconds=c.STAGES,independent_review_maximum_seconds=c.INDEPENDENT_REVIEW_SECONDS,
        max_evidence_bytes=c.MAX_EVIDENCE_BYTES,prefinal_evidence_bytes=c.PREFINAL_EVIDENCE_BYTES,
        final_reserve_bytes=c.FINAL_RESERVE_BYTES,max_workers=c.MAX_WORKERS,max_graph_builds=c.MAX_GRAPH_BUILDS,
        declared_worker_lifecycles=design.lifecycle_inventory(),automatic_retries=0,failed_decision_retries=1,
        authorized_interrupted_decision_retries=1,qualification_decisions=134,
        qualification_ordinary_requests=0,qualification_graph_cases=22,
        prior_worker_trees=c.PRIOR_WORKER_TREES,maximum_combined_worker_trees=c.MAX_COMBINED_WORKERS,
        prior_evidence_bytes=c.PRIOR_EVIDENCE_BYTES,prior_ram_wait_seconds=c.PRIOR_RAM_WAIT_SECONDS)
    return result


def seal_prior_tree():
    """Stream file hashes; never load the four-gigabyte evidence corpus in RAM."""
    from .preparation_support import verify_old_absence
    target=c.OUT/'prior_evidence_inventory.json'
    if target.exists():
        return c.artifact(target)
    started=time.monotonic(); deadline=min(started+PRESERVATION_SECONDS,c.ORIGINAL_MAIN_MONOTONIC_DEADLINE)
    def boundary():
        c.require(time.monotonic()<deadline and time.time()<c.ORIGINAL_MAIN_UNIX_DEADLINE,'Bounded preservation deadline')
        c.require(shutil.disk_usage(c.ROOT).free>=5*c.GiB,'Preservation disk reserve')
    identities=verify_old_absence()
    before=sorted(p for p in c.OLD_OUT.rglob('*') if p.is_file())
    c.require(len(before)==93099 and len(before)<=MAX_TREE_FILES,'Exact stopped original file inventory')
    files=[]
    for index,path in enumerate(before):
        boundary(); stat=path.stat(); item=c.artifact(path); after=path.stat()
        c.require((stat.st_size,stat.st_mtime_ns)==(after.st_size,after.st_mtime_ns),'Original evidence changed during seal')
        files.append(item)
        if index and index%10000==0:
            print(c.canonical(dict(event='preservation_progress',files=index,total=len(before))),flush=True)
    c.require(before==sorted(p for p in c.OLD_OUT.rglob('*') if p.is_file()),'Original evidence inventory changed')
    c.require(sum(x['bytes'] for x in files)==4051821726,'Exact stopped original evidence bytes')
    c.require(identities==verify_old_absence(),'Old identity inventory changed')
    return c.save(target,dict(schema='search256_original_tree_preservation_v1',time=c.now(),
        root=c.OLD_OUT.relative_to(c.ROOT).as_posix(),files=files,file_count=len(files),
        total_bytes=sum(x['bytes'] for x in files),owner_identities_absent=identities,
        original_attempt_status='incomplete',actual_owner_exit_codes_unknown=True,
        preservation_seconds=time.monotonic()-started))


def validate_manifest(m):
    values=dict(schema=MANIFEST_SCHEMA,completed_main_games=235,operational_games=6,
        continuation_game_index=235,continuation_ply=30,original_attempt_status='incomplete',
        original_monotonic_origin=c.PRIOR_ORIGIN,original_unix_origin=c.PRIOR_WALL_START,
        main_monotonic_deadline=c.ORIGINAL_MAIN_MONOTONIC_DEADLINE,
        main_unix_deadline=c.ORIGINAL_MAIN_UNIX_DEADLINE,prior_worker_trees=488,
        old_worker_trees=486,new_worker_limit=378,maximum_combined_worker_trees=866,
        prior_ram_wait_seconds=c.PRIOR_RAM_WAIT_SECONDS,prior_evidence_bytes=c.PRIOR_EVIDENCE_BYTES,
        authorized_retries=1)
    c.require(all(m.get(k)==v for k,v in values.items()),'Exact recovery manifest accounting and boundary')
    c.require(m['old_freeze']==c.artifact(c.OLD_OUT/'freeze.json'),'Exact old freeze descriptor')
    r=m['retry']
    c.require(r==dict(game_index=235,ply=30,route='research',old_worker='main_research_g235_challenger',
        old_request_id='request-00000017',new_worker='main_research_g235_challenger',
        new_request_id='request-00000003',unresolved_original_clock_preserved=True,
        input=c.artifact(c.OLD_OUT/'workers/main_research_g235_challenger/inputs/request-00000017.json'),
        prior_last_move=c.artifact(c.OLD_OUT/'games/main/research/p117_g235/moves/0029.json')),
        'Only exact single ambiguous request can be retried')
    claim=c.read(c.ROOT/m['original_claim']['path']);stage=c.read(c.ROOT/m['original_main_start']['path'])
    c.require((claim['monotonic_origin'],claim['unix_origin'])==(c.PRIOR_ORIGIN,c.PRIOR_WALL_START),'Original dual clocks')
    c.require((stage['monotonic_deadline'],stage['unix_deadline'])==
        (c.ORIGINAL_MAIN_MONOTONIC_DEADLINE,c.ORIGINAL_MAIN_UNIX_DEADLINE),'Original main deadline')
    c.require(len(list((c.OLD_OUT/'games/main').rglob('result.json')))==235 and
        len(list((c.OLD_OUT/'workers').iterdir()))==486,'Exact old game and worker count')
    c.require(len(list((c.OLD_OUT/'games/main/research/p117_g235/moves').glob('*.json')))==30,'Exact saved prefix length')
    c.require(not any((c.OLD_OUT/p).exists() for p in ('failure.json','completion.json','execution.json','watchdog_execution.json')),
        'Original missing terminal receipts remain missing, never fabricated')
    return True


def prepare():
    c.require(not(c.OUT/'freeze.json').exists(),'Preparation cannot change frozen recovery')
    old=original_freeze(); c.verify(old['inputs'])
    inventory=seal_prior_tree();tree=c.read(c.ROOT/inventory['path'])
    diagnosis=[c.artifact(p) for p in sorted(c.DIAGNOSIS_OUT.rglob('*')) if p.is_file()]
    c.require(20318908+tree['total_bytes']+sum(x['bytes'] for x in diagnosis)==c.PRIOR_EVIDENCE_BYTES,
        'All prior attempts and diagnosis bytes charged')
    m=dict(schema=MANIFEST_SCHEMA,old_freeze=c.artifact(c.OLD_OUT/'freeze.json'),
        original_claim=c.artifact(c.OLD_OUT/'guardian_claim.json'),original_tree=inventory,
        original_main_start=c.artifact(c.OLD_OUT/'stages/main_start.json'),diagnosis_tree=diagnosis,
        original_main_admission=c.artifact(c.OLD_OUT/'main_admission.json'),
        original_qualification_summary=c.artifact(c.OLD_OUT/'qualification_summary.json'),
        completed_main_games=235,operational_games=6,continuation_game_index=235,continuation_ply=30,
        original_attempt_status='incomplete',original_monotonic_origin=c.PRIOR_ORIGIN,
        original_unix_origin=c.PRIOR_WALL_START,main_monotonic_deadline=c.ORIGINAL_MAIN_MONOTONIC_DEADLINE,
        main_unix_deadline=c.ORIGINAL_MAIN_UNIX_DEADLINE,prior_worker_trees=488,old_worker_trees=486,
        new_worker_limit=378,maximum_combined_worker_trees=866,prior_ram_wait_seconds=c.PRIOR_RAM_WAIT_SECONDS,
        prior_evidence_bytes=c.PRIOR_EVIDENCE_BYTES,authorized_retries=1,
        retry=dict(game_index=235,ply=30,route='research',old_worker='main_research_g235_challenger',
            old_request_id='request-00000017',new_worker='main_research_g235_challenger',
            new_request_id='request-00000003',unresolved_original_clock_preserved=True,
            input=c.artifact(c.OLD_OUT/'workers/main_research_g235_challenger/inputs/request-00000017.json'),
            prior_last_move=c.artifact(c.OLD_OUT/'games/main/research/p117_g235/moves/0029.json')))
    validate_manifest(m)
    return dict(recovery_manifest=fixed(c.OUT/'recovery_manifest.json',m),
        schedule=fixed(c.OUT/'schedule.json',c.read(c.OLD_OUT/'schedule.json')),
        qualification_plan=old['qualification_plan'],profiles=old['profiles'])


def freeze():
    from . import runner
    c.require(not(c.OUT/'freeze.json').exists(),'No replacement freeze')
    prepared=prepare();old=original_freeze()
    tests=c.read(c.OUT/'prelaunch/model_free_tests.json');review=c.read(c.OUT/'prelaunch/implementation_review.json')
    c.require(tests['passed'] is True and tests['exit_code']==0,'Actual passing model-free tests')
    c.verify(tests['inputs']+[tests['log']]);c.verify(review['inputs'])
    c.require(review['approved'] is True and not review['unresolved_blocking_findings'],'Independent implementation approval')
    frozen=deepcopy(old);frozen.update(contract());frozen.update(prepared)
    frozen.update(schema=SCHEMA,time=c.now(),protocol=c.artifact(c.PROTOCOL),
        source_guard=verify_runtime_sources(),original_attempt_status='incomplete',
        amended_recovered_benchmark=True,fresh_full_qualification_and_operational_run=False,
        fresh_recovery_entrypoint_qualification=True,
        recovery=dict(prior_attempt=c.artifact(c.OLD_OUT/'freeze.json'),
            prior_guardian_claim=c.artifact(c.OLD_OUT/'guardian_claim.json'),
            prior_preservation=c.artifact(c.OUT/'prior_evidence_inventory.json'),
            authorized_interrupted_decision_retries=1,prior_worker_trees=488,maximum_combined_worker_trees=866,
            prior_ram_wait_seconds=c.PRIOR_RAM_WAIT_SECONDS,prior_evidence_bytes=c.PRIOR_EVIDENCE_BYTES))
    pins={item['path']:item for item in old['inputs']}
    for item in tests['inputs']+review['inputs']:
        previous=pins.get(item['path'])
        c.require(previous is None or previous==item,'Conflicting tested/reviewed source pin')
        pins[item['path']]=item
    paths=list(Path(__file__).parent.glob('*.py'))+[c.ROOT/'tools/future_training_rules_20261021.py']
    paths=python_dependencies(paths)+list((c.ROOT/'tests').glob('test_splendorust_search256*20261030*.py'))
    paths += [p for p in c.OUT.rglob('*') if p.is_file()]+[c.PROTOCOL,c.OLD_OUT/'freeze.json']
    for path in paths:
        item=c.artifact(path);previous=pins.get(item['path'])
        c.require(previous is None or previous==item,'Conflicting recovery freeze pin')
        pins[item['path']]=item
    frozen['inputs']=sorted(pins.values(),key=lambda x:x['path'])
    c.require('torch' not in sys.modules,'Freeze must not load models')
    runner.validate_freeze_contract(frozen);c.verify(frozen['inputs'])
    c.require(shutil.disk_usage(c.ROOT).free>=c.INITIAL_FREE_DISK_BYTES,'Initial free disk reserve')
    return dict(status='frozen_not_launched',freeze=c.save(c.OUT/'freeze.json',frozen),inputs=len(pins))


if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('action',choices=['prepare','freeze']);args=parser.parse_args()
    print(c.canonical(prepare() if args.action=='prepare' else freeze()),flush=True)
