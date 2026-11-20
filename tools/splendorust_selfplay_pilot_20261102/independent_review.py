"""Independent saved-only acceptance after actual model-owner exits."""
import argparse,os,time,traceback,shutil
import psutil
from . import common as c,qualification_control as runner
from tools.splendorust_search256_recovery_20261030_02 import placement
from tools.future_training_rules_20261021 import verify_runtime_sources

DIRECTORIES=tuple(name for name in runner.SEALED_NAMES if '.' not in name)


def review_deadlines(claim,final,now_mono,now_wall):
    return (min(claim['monotonic_origin']+c.JOB_SECONDS,final['monotonic_deadline'],now_mono+c.INDEPENDENT_REVIEW_SECONDS),
        min(claim['unix_origin']+c.JOB_SECONDS,final['unix_deadline'],now_wall+c.INDEPENDENT_REVIEW_SECONDS))


class Guard:
    def __init__(self,claim,final):
        self.start=claim['monotonic_origin'];self.wall_start=claim['unix_origin'];self.end=self.start+c.JOB_SECONDS
        self.deadline,self.wall_deadline=review_deadlines(claim,final,time.monotonic(),time.time())
        self.last=-float('inf');self.peak_rss=0;self.placement_mode='isolated'
        self.placement_sequence=len(list((c.OUT/'placement_requests').glob('*.json')))
    def remaining(self):return min(self.deadline-time.monotonic(),self.wall_deadline-time.time())
    def effective_deadline(self):return time.monotonic()+self.remaining()
    def guard(self):
        c.require(self.remaining()>0,'Independent saved-reader original global/final deadline')
        now=time.monotonic()
        if now-self.last<1:return
        self.last=now;rss=psutil.Process().memory_info().rss;self.peak_rss=max(rss,self.peak_rss)
        c.require(rss<=c.MAX_RSS_BYTES and shutil.disk_usage(c.ROOT).free>=5*c.GiB,'Saved reader resources')
    def close_worker(self,folder,child=None):
        c.require(child is None,'Saved-only reader never receives a live model');self.guard()


def check_successful_owners(expected_sha):
    for name in ('execution.json','watchdog_execution.json'):
        value=c.read(c.OUT/name)
        c.require(value['status']=='completed' and value['exit_code']==0 and value['freeze_sha256']==expected_sha and
            not value['cleanup_errors'] and not value['cleanup_receipt']['unconfirmed'] and value['charged_job_seconds']<c.JOB_SECONDS,
            'Actual clean runner/watchdog exits required')
    launch=c.read(c.OUT/'launch.json');identities=[dict(pid=launch['pid'],created=launch['created']),c.read(c.OUT/'guardian_owner.json')['watchdog']]
    for item in identities:
        try:process=psutil.Process(item['pid'])
        except psutil.NoSuchProcess:continue
        c.require(process.create_time()!=item['created'],'Old model owner still alive')
    return identities


def review(expected_sha):
    c.require(c.file_hash(c.OUT/'freeze.json')==expected_sha,'Exact independent diagnostic freeze')
    freeze=c.read(c.OUT/'freeze.json');claim=c.read(c.OUT/'guardian_claim.json');final=c.read(c.OUT/'stages/final_start.json')
    c.require(claim['freeze_sha256']==expected_sha,'Original owner clock identity')
    guard=Guard(claim,final);guard.guard();sources=verify_runtime_sources();placement.configure_execution(freeze,'isolated')
    runner.verify_runtime(freeze,guard);absent=check_successful_owners(expected_sha)
    completion=c.read(c.OUT/'completion.json')
    c.require(completion['status']=='completed_verified' and completion['failure'] is None and completion['freeze_sha256']==expected_sha,
        'Never accept an incomplete diagnostic')
    seal=c.read(c.OUT/'completed_evidence_seal.json')['artifacts'];indexed={}
    for item in seal:
        guard.guard();path=(c.ROOT/item['path']).resolve()
        c.require(path.is_relative_to(c.OUT.resolve()) and item['path'] not in indexed,'Unique current evidence seal')
        c.verify([item]);indexed[item['path']]=item
    for name in DIRECTORIES:
        actual={p.relative_to(c.ROOT).as_posix() for p in(c.OUT/name).rglob('*') if p.is_file()}
        expected={p for p in indexed if(c.ROOT/p).is_relative_to(c.OUT/name)}
        c.require(actual==expected,'Exact complete sealed diagnostic inventory: '+name)
    coverage=runner.verify_saved(guard,freeze,writing=False)
    c.require(coverage==completion['coverage'],'Independent fixed-request reconciliation')
    ending=c.read(c.OUT/'stages/final_end.json')
    c.require(final['monotonic']<=ending['monotonic']<final['monotonic_deadline'] and
        final['unix_time']<=ending['unix_time']<final['unix_deadline'],'Passed original final stage clocks')
    for item in seal:guard.guard();c.verify([item])
    runner.verify_runtime(freeze,guard);c.require(runner.bytes_used()<c.MAX_EVIDENCE_BYTES,'Final physical evidence ceiling')
    return c.save(c.OUT/'independent_review.json',dict(approved=True,status='independently_verified_selfplay_'+c.SCOPE,
        freeze_sha256=expected_sha,reviewer=dict(pid=os.getpid(),created=psutil.Process().create_time()),coverage=coverage,
        source_guard=sources,prior_owner_identities_absent=absent,peak_rss_bytes=guard.peak_rss,
        seal=c.artifact(c.OUT/'completed_evidence_seal.json'),completion=c.artifact(c.OUT/'completion.json'),
        source_inputs=len(freeze['inputs']),no_model_calls=True,new_games=coverage['completed_games'],
        requires_actual_reader_and_guardian_exit_before_external_acceptance=True))


def main():
    parser=argparse.ArgumentParser();parser.add_argument('--freeze-sha256',required=True);args=parser.parse_args()
    try:review(args.freeze_sha256);return 0
    except BaseException:
        c.save(c.OUT/'independent_review_failure.json',dict(approved=False,time=c.now(),error=traceback.format_exc()));return 1


if __name__=='__main__':raise SystemExit(main())
