"""Fresh-process saved-only acceptance readback within the original final clock."""
import argparse
from pathlib import Path
import os
import shutil
import time
import traceback
import psutil
from . import common as c,runner,transport,qualification,analysis,placement
from tools.future_training_rules_20261021 import verify_runtime_sources


def review_deadlines(claim,final,now_mono,now_wall):
    return (min(claim['monotonic_origin']+c.JOB_SECONDS,final['monotonic_deadline'],
                now_mono+c.INDEPENDENT_REVIEW_SECONDS),
            min(claim['unix_origin']+c.JOB_SECONDS,final['unix_deadline'],
                now_wall+c.INDEPENDENT_REVIEW_SECONDS))


class Guard:
    def __init__(self,claim,final):
        self.start=claim['monotonic_origin'];self.end=self.start+c.JOB_SECONDS
        self.deadline,self.wall_deadline=review_deadlines(claim,final,time.monotonic(),time.time())
        self.placement_mode='isolated';self.last=-float('inf');self.peak_rss=0
        self.placement_sequence=len(list((c.OUT/'placement_requests').glob('*.json')))

    def remaining(self):
        return min(self.deadline-time.monotonic(),self.wall_deadline-time.time())

    def effective_deadline(self):
        return time.monotonic()+self.remaining()

    def guard(self):
        c.require(time.monotonic()<self.deadline and time.time()<self.wall_deadline,
                  'Independent reader shares original final/global deadline')
        now=time.monotonic()
        if now-self.last<1:return
        self.last=now;rss=psutil.Process().memory_info().rss;self.peak_rss=max(self.peak_rss,rss)
        c.require(rss<=c.MAX_RSS_BYTES,'Independent reader RSS bound')
        c.require(shutil.disk_usage(c.ROOT).free>=5*c.GiB,'Independent reader free disk')

    def close_worker(self,folder,child=None):
        # All children have already exited under their actual owning handles.
        # The unchanged closed-worker verifier rechecks their saved cleanup and
        # identity absence; this read-only adapter never owns or launches one.
        c.require(child is None,'Saved-only review cannot receive a live worker')
        self.guard()


def check_successful_owners(expected_sha):
    for name in ('execution.json','watchdog_execution.json'):
        value=c.read(c.OUT/name)
        c.require(value['status']=='completed' and value['exit_code']==0 and
            value['freeze_sha256']==expected_sha and not value['cleanup_errors'] and
            value['charged_job_seconds']<c.JOB_SECONDS,'Clean actual runner/watchdog exits required')
    identities=[]
    launch=c.read(c.OUT/'launch.json');identities.append(dict(pid=launch['pid'],created=launch['created']))
    owner=c.read(c.OUT/'guardian_owner.json');identities.append(owner['watchdog'])
    for item in identities:
        try:process=psutil.Process(item['pid'])
        except psutil.NoSuchProcess:continue
        c.require(process.create_time()!=item['created'],'Prior owning identity still active')
    return identities


def review(expected_sha):
    c.require(c.file_hash(c.OUT/'freeze.json')==expected_sha,'Independent exact freeze')
    freeze=c.read(c.OUT/'freeze.json');claim=c.read(c.OUT/'guardian_claim.json')
    c.require(claim['freeze_sha256']==expected_sha,'Shared original owning clock')
    final=c.read(c.OUT/'stages/final_start.json');guard=Guard(claim,final);guard.guard()
    # Exactly once, before any saved-state reconstruction in this new evaluator.
    source_guard=verify_runtime_sources()
    placement.configure_execution(freeze,'isolated')
    runner.verify_runtime(freeze,guard)
    absent=check_successful_owners(expected_sha)
    completion=c.read(c.OUT/'completion.json')
    c.require(completion['status']=='completed_verified' and completion['failure'] is None and
        completion['freeze_sha256']==expected_sha,'No acceptance of an incomplete attempt')
    seal=c.read(c.OUT/'completed_evidence_seal.json')['artifacts'];sealed={}
    for item in seal:
        guard.guard();path=(c.ROOT/item['path']).resolve()
        c.require(path.is_relative_to(c.OUT.resolve()) and item['path'] not in sealed,'Unique sealed study file')
        c.verify([item]);sealed[item['path']]=item
    # No unrepresented additions/deletions in any scientific evidence directory.
    for directory in ('workers','records','entrypoint_checks','closed_workers','closed_worker_costs',
                      'placement_requests','placement_acks','memory_admissions'):
        actual={p.relative_to(c.ROOT).as_posix() for p in (c.OUT/directory).rglob('*') if p.is_file()}
        expected={p for p in sealed if (c.ROOT/p).is_relative_to(c.OUT/directory)}
        c.require(actual==expected,'Exact sealed directory inventory: '+directory)
    panel,matrix,plan=runner.load_inputs(freeze,guard)
    roots={r['root_id']:r for r in panel['roots']};records=[]
    paths=sorted((c.OUT/'records').glob('*.json'))
    c.require([p.name for p in paths]==[f'{i:04d}.json' for i in range(120)],'Full fixed120 inventory')
    represented={name:[] for name in transport.lifecycle_inventory()}
    for path in paths:
        guard.guard();row=c.read(path);transport.verify_saved_record(row,freeze)
        c.require(row['completed'] is True and 'transition_error' not in row,'Every ordinary result successful')
        root=roots[row['spec']['root_id']];after,proof=runner.validate_transition(row,root)
        c.require(row['before']==root['before'] and row['after']==after and row['transition_proof']==proof,
                  'Independent corrected step and saved proof')
        represented[Path(row['worker_folder']).name].append(row['clock']['request_id']);records.append(row)
    exact=analysis.verify_records(records,panel,matrix,complete=True,boundary=guard.guard)
    analysis.verify_saved_analysis(c.read(c.OUT/'analysis.json'),records,panel,matrix,complete=True,boundary=guard.guard)
    c.require(not(c.OUT/'collection_stop.json').exists(),'Full acceptance cannot contain a failed request')
    qualified=c.read(c.OUT/'qualification_summary.json');proofs=[]
    headers=dict(status='passed',scientific_decisions=119,parent_requests=2,graph_cases=22,
                 same_workers_retained_for_repeats=True)
    c.require({k:qualified[k] for k in headers}==headers and
        [x['route'] for x in qualified['receipts']]==['research','frozen'],'Exact two-suite119 qualification summary')
    for name in sorted(represented):
        folder=c.OUT/'workers'/name;guard.guard()
        c.require(len(represented[name])==60 and len(set(represented[name]))==60 and
            set(represented[name])=={p.stem for p in (folder/'inputs').glob('*.json')},'All60 ordinary requests per retained tree')
        proof=transport.verify_closed_worker(folder,freeze,guard)
        c.require(proof==c.read(c.OUT/'closed_workers'/f'{name}.json'),'Fresh full raw/log/closure readback')
        route=c.read(folder/'route_identity.json')['route'];receipt=c.read(c.OUT/'entrypoint_checks'/f'{route}.json')
        qproof=qualification.verify_entrypoint(folder,receipt,plan,freeze,boundary=guard.guard)
        c.require(qproof==next(x['verification'] for x in qualified['receipts'] if x['route']==route),'Fresh119 actual-suite forensic proof')
        proofs.append(proof)
    c.require(c.read(c.OUT/'all_closed_workers.json')==dict(status='passed',workers=2,proofs=proofs),'Combined raw closure proof')
    parents=sum(p['raw']['requests'] for p in proofs)
    coverage=dict(complete=True,planned_requests=120,attempted_requests=120,completed_requests=120,
        failed_requests=0,unattempted_requests=0,qualification_scientific_decisions=119,scientific_decisions=239,
        graph_cases=22,workers=2,parent_requests=124,retries=0,new_games=0,exact_comparisons=exact)
    summary=c.read(c.OUT/'verification_summary.json')
    c.require(parents==124 and c.read(c.OUT/'coverage.json')==completion['coverage']==summary['coverage']==coverage,
              'Independent fixed-workload reconciliation')
    c.require(summary['status']=='passed' and summary['all_raw_placement_logs_graph_science_sources_verified'] is True and
        summary['peak_combined_rss_bytes']<=c.MAX_RSS_BYTES,'Saved full verification and aggregate RSS bound')
    runner.verify_stage_transition(guard,records)
    c.require(runner.verify_exchanges(guard,freeze)==c.read(c.OUT/'all_placement_integrity.json'),'All ownership placement exchanges')
    for stage in c.STAGES:
        begin=c.read(c.OUT/'stages'/f'{stage}_start.json');end=c.read(c.OUT/'stages'/f'{stage}_end.json')
        c.require(begin['monotonic']<=end['monotonic']<begin['monotonic_deadline'] and
            begin['unix_time']<=end['unix_time']<begin['unix_deadline'],'Original stage clocks include closure/verification')
    runner.verify_runtime(freeze,guard)
    for item in seal:guard.guard();c.verify([item])
    c.require(runner.bytes_used()<c.MAX_EVIDENCE_BYTES,'Final physical evidence ceiling')
    guard.guard()
    result=dict(approved=True,status='repeat_saved_evidence_verified',freeze_sha256=expected_sha,
        reviewer=dict(pid=os.getpid(),created=psutil.Process().create_time()),coverage=coverage,
        source_guard=source_guard,source_inputs=len(freeze['inputs']),sealed_files=len(seal),
        seal=c.artifact(c.OUT/'completed_evidence_seal.json'),completion=c.artifact(c.OUT/'completion.json'),
        prior_owner_identities_absent=absent,peak_rss_bytes=guard.peak_rss,
        original_monotonic_origin=claim['monotonic_origin'],original_unix_origin=claim['unix_origin'],
        final_monotonic_deadline=guard.deadline,final_unix_deadline=guard.wall_deadline,
        no_model_calls=True,new_games=0,requires_actual_reader_and_guardian_exit_before_external_acceptance=True)
    return c.save(c.OUT/'independent_review.json',result)


def main():
    parser=argparse.ArgumentParser();parser.add_argument('--freeze-sha256',required=True);args=parser.parse_args()
    try:review(args.freeze_sha256);return 0
    except BaseException:
        c.save(c.OUT/'independent_review_failure.json',dict(error=traceback.format_exc(),time=c.now(),approved=False))
        return 1


if __name__=='__main__':raise SystemExit(main())
