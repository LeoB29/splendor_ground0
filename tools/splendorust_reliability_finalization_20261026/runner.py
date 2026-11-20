"""Final preservation and receipt acceptance only: no science or models."""
import argparse
import sys
import time
import traceback
import psutil
from . import common as c, preparation, preservation

def configure():
    process=psutil.Process();process.cpu_affinity(list(range(2,20)));process.nice(psutil.NORMAL_PRIORITY_CLASS)
    return dict(pid=process.pid,created=process.create_time(),cpus=process.cpu_affinity(),priority=process.nice())

def execute(freeze,envelope):
    from .receipts import validate_receipts
    # Only authenticate source bytes. Never import/call the earlier full checks.
    envelope.checkpoint('source_preservation_begin',dict(files=len(freeze['source_preservation'])))
    c.verify(freeze['source_preservation'],envelope.guard)
    c.save(c.OUT/'source_preservation.json',dict(status='passed',files=len(freeze['source_preservation']),
        manifest_sha256=c.digest(freeze['source_preservation']),runtime_or_scientific_checks_rerun=False))
    envelope.checkpoint('source_preservation_passed',dict(files=len(freeze['source_preservation'])))
    # The corrected-source startup guard is called once, with its own source
    # already byte-authenticated above. This never instantiates a game/model.
    from tools.future_training_rules_20261021 import verify_runtime_sources
    c.save(c.OUT/'corrected_runtime_sources.json',verify_runtime_sources())
    envelope.checkpoint('passed_receipt_reconciliation_begin')
    reused=validate_receipts(freeze,envelope)
    c.save(c.OUT/'reused_receipts.json',reused)
    envelope.checkpoint('passed_receipt_reconciliation_passed')
    result=preservation.preserve_all(freeze,envelope)
    c.save(c.OUT/'preservation.json',result)
    envelope.checkpoint('final_source_preservation_begin')
    c.verify(freeze['source_preservation'],envelope.guard)
    c.verify(freeze['inputs'],envelope.guard)
    c.verify(freeze['receipt_pins'],envelope.guard)
    owners=preparation.require_owners_absent();envelope.final_guard()
    c.save(c.OUT/'recovered_coverage.json',reused['coverage'])
    summary=dict(status='passed_finalization_checks_pending_clean_owning_exit',accepted=False,
        original_status='incomplete',v1_status='failed',v2_status='incomplete',historical_failures_preserved=True,
        reused_components=reused,preservation=result,source_inputs_preserved=len(freeze['source_preservation']),
        original_owners=owners,new_games=0,new_models=0,new_decisions=0,new_scientific_replays=0,
        larger_data_deferred=True,no_strength_or_causal_claim=True,peak_auditor_rss_bytes=envelope.peak_rss,
        elapsed_seconds=max(time.monotonic()-envelope.start,time.time()-envelope.wall_start))
    c.save(c.OUT/'verification_summary.json',summary);envelope.final_guard()
    return summary

def main():
    parser=argparse.ArgumentParser();parser.add_argument('--freeze-sha256',required=True);args=parser.parse_args()
    sys.dont_write_bytecode=True;sys.addaudithook(c.audit_event)
    claim=c.read(c.OUT/'watchdog_claim.json');envelope=c.Envelope(claim)
    try:
        c.require(claim['freeze_sha256']==args.freeze_sha256,'Owning freeze mismatch')
        c.require(c.artifact(c.OUT/'freeze.json',envelope.guard)['sha256']==args.freeze_sha256,'Finalization freeze changed')
        freeze=c.read(c.OUT/'freeze.json')
        c.require(freeze['job_seconds']==c.JOB_SECONDS==900 and freeze['automatic_retries']==0,'Frozen finalization bound differs')
        c.save(c.OUT/'auditor_placement.json',configure());c.verify(freeze['inputs'],envelope.guard)
        envelope.checkpoint('finalization_startup_passed')
        execute(freeze,envelope);envelope.final_guard()
        c.save(c.OUT/'completion.json',dict(time=c.now(),status='checks_passed_pending_owning_exit',
            freeze_sha256=args.freeze_sha256,original_status='incomplete',automatic_retries=0))
        envelope.final_guard()
    except BaseException:
        c.save(c.OUT/'failure.json',dict(time=c.now(),phase=envelope.phase,error=traceback.format_exc(),previous_attempts_preserved=True))
        if not (c.OUT/'completion.json').exists():
            c.save(c.OUT/'completion.json',dict(time=c.now(),status='incomplete',freeze_sha256=args.freeze_sha256,automatic_retries=0))
        raise

if __name__=='__main__':main()
