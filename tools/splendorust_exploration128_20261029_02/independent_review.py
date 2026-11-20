"""Separate saved-only acceptance process under the same original owner clock."""
import argparse
from pathlib import Path
import os
import shutil
import time
import traceback
import psutil
from . import common as c,runner,placement
from tools.future_training_rules_20261021 import verify_runtime_sources
def review_deadlines(claim,final,now_mono,now_wall):
    return (min(claim['monotonic_origin']+c.JOB_SECONDS,final['monotonic_deadline'],
                now_mono+c.INDEPENDENT_REVIEW_SECONDS),
            min(claim['unix_origin']+c.JOB_SECONDS,final['unix_deadline'],
                now_wall+c.INDEPENDENT_REVIEW_SECONDS))


class Guard:
    def __init__(self,claim,final):
        self.start=claim['monotonic_origin'];self.wall_start=claim['unix_origin'];self.end=self.start+c.JOB_SECONDS
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


def main():
    parser=argparse.ArgumentParser();parser.add_argument('--freeze-sha256',required=True);args=parser.parse_args()
    try:review(args.freeze_sha256);return 0
    except BaseException:
        c.save(c.OUT/'independent_review_failure.json',dict(error=traceback.format_exc(),time=c.now(),approved=False))
        return 1


def review(expected_sha):
    c.require(c.file_hash(c.OUT/'freeze.json')==expected_sha,'Independent exact freeze')
    freeze=c.read(c.OUT/'freeze.json');claim=c.read(c.OUT/'guardian_claim.json')
    c.require(claim['freeze_sha256']==expected_sha,'Shared original owner clock')
    final=c.read(c.OUT/'stages/final_start.json');guard=Guard(claim,final);guard.guard()
    source_guard=verify_runtime_sources();placement.configure_execution(freeze,'isolated');runner.verify_runtime(freeze,guard)
    absent=check_successful_owners(expected_sha)
    completion=c.read(c.OUT/'completion.json')
    c.require(completion['status']=='completed_verified' and completion['failure'] is None and completion['freeze_sha256']==expected_sha,'No acceptance of incomplete attempt')
    seal=c.read(c.OUT/'completed_evidence_seal.json')['artifacts'];sealed={}
    for item in seal:
        guard.guard();path=(c.ROOT/item['path']).resolve()
        c.require(path.is_relative_to(c.OUT.resolve()) and item['path'] not in sealed,'Unique sealed study file')
        c.verify([item]);sealed[item['path']]=item
    for directory in ('workers','games','entrypoint_checks','qualification_rows','closed_workers','closed_worker_costs','phase_verification','phase_seals','repeated_decisions','replay_samples','placement_requests','placement_acks','memory_admissions'):
        actual={p.relative_to(c.ROOT).as_posix() for p in(c.OUT/directory).rglob('*') if p.is_file()}
        expected={p for p in sealed if(c.ROOT/p).is_relative_to(c.OUT/directory)}
        c.require(actual==expected,'Exact sealed scientific directory inventory: '+directory)
    coverage=runner.verify_saved_attempt(guard,freeze,writing=False)
    c.require(coverage==completion['coverage']==c.read(c.OUT/'verification_summary.json')['coverage'],'Independent full conditional workload reconciliation')
    end=c.read(c.OUT/'stages/final_end.json')
    c.require(final['monotonic']<=end['monotonic']<final['monotonic_deadline'] and final['unix_time']<=end['unix_time']<final['unix_deadline'],'Original final clocks include all saved verification')
    runner.verify_runtime(freeze,guard)
    for item in seal:guard.guard();c.verify([item])
    c.require(runner.bytes_used()<c.MAX_EVIDENCE_BYTES,'Final physical evidence ceiling')
    guard.guard()
    return c.save(c.OUT/'independent_review.json',dict(approved=True,status='exploration_saved_evidence_verified',freeze_sha256=expected_sha,
      reviewer=dict(pid=os.getpid(),created=psutil.Process().create_time()),coverage=coverage,source_guard=source_guard,
      source_inputs=len(freeze['inputs']),sealed_files=len(seal),seal=c.artifact(c.OUT/'completed_evidence_seal.json'),completion=c.artifact(c.OUT/'completion.json'),
      prior_owner_identities_absent=absent,peak_rss_bytes=guard.peak_rss,original_monotonic_origin=claim['monotonic_origin'],original_unix_origin=claim['unix_origin'],
      final_monotonic_deadline=guard.deadline,final_unix_deadline=guard.wall_deadline,no_model_calls=True,new_games=0,
      requires_actual_reader_and_guardian_exit_before_external_acceptance=True))

if __name__=='__main__':raise SystemExit(main())
