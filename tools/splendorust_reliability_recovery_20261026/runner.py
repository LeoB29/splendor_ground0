"""One-shot saved-only auditor; never launch a model or modify original evidence."""
import argparse
import sys
import time
import traceback
import psutil
from . import common as c
from . import preparation

def configure():
    process=psutil.Process();process.cpu_affinity(list(range(2,20)))
    process.nice(psutil.NORMAL_PRIORITY_CLASS)
    c.require(process.cpu_affinity()==list(range(2,20)),'Recovery placement differs')
    return dict(pid=process.pid,created=process.create_time(),cpus=process.cpu_affinity(),priority=process.nice())

def finish_snapshot(freeze,envelope):
    preparation.validate_snapshot(freeze['original_snapshot'],preparation.inventory(c.ORIGINAL,envelope.guard))
    c.verify(freeze['original_snapshot'],envelope.guard)
    preparation.validate_snapshot(freeze['original_snapshot'],preparation.inventory(c.ORIGINAL,envelope.guard))
    c.verify(freeze['inputs'],envelope.guard)
    return dict(status='passed',original_files=len(freeze['original_snapshot']),
        original_bytes=sum(row['bytes'] for row in freeze['original_snapshot']),
        recovery_inputs=len(freeze['inputs']),extra_missing_or_changed_original_files=0)

def execute(freeze,envelope):
    # Import only after the new frozen code is authenticated by main().
    from tools.future_training_rules_20261021 import verify_runtime_sources
    from .runtime import verify_runtime_and_prior
    from .review import run_review
    c.save(c.OUT/'corrected_runtime_sources.json',verify_runtime_sources())
    envelope.checkpoint('corrected_runtime_sources_passed')
    original=c.read(c.ORIGINAL/'freeze.json')
    runtime=verify_runtime_and_prior(original,envelope)
    c.save(c.OUT/'runtime_verification.json',runtime)
    envelope.checkpoint('saved_scientific_review_begin')
    review=run_review(envelope,original,freeze)
    c.require(review['status']=='passed_saved_evidence_review','Saved review did not pass')
    consumed=review.pop('consumed_inputs')
    c.save(c.OUT/'consumed_inputs.json',dict(artifacts=consumed,count=len(consumed)))
    c.save(c.OUT/'saved_review.json',review)
    envelope.checkpoint('saved_scientific_review_passed',dict(games=review['games'],moves=review['live_decisions'],reused_reproductions=review['reproduced_completed_decisions']))
    envelope.checkpoint('final_original_snapshot_readback_begin')
    snapshot=finish_snapshot(freeze,envelope)
    c.save(c.OUT/'original_preservation.json',snapshot)
    envelope.checkpoint('final_original_snapshot_readback_passed',snapshot)
    c.save(c.OUT/'recovered_coverage.json',review['coverage'])
    envelope.guard()
    summary=dict(status='recovered_saved_verification_pending_owning_exit',original_status='incomplete',original_exit_code=1,
        original_freeze_sha256=c.ORIGINAL_SHA,original_interruption=review['original_interruption'],
        games=review['games'],live_decisions=review['live_decisions'],actual_entrypoint_decisions=104,
        reused_completed_decision_reproductions=48,new_games=0,new_models=0,new_reproductions=0,
        worker_count=review['worker_count'],graph_workers=review['graph_workers'],performance=review['performance'],
        original_frozen_inputs_verified=runtime['original_inputs_rechecked_from_start'],
        preceding_qualification_exact_acceptance=runtime['preceding_qualification_exact_acceptance'],
        original_preservation=snapshot,graph_and_raw_scope=review['graph_and_raw_scope'],
        larger_data_deferred=True,no_causal_or_strength_claim=True,original_outputs_written=False,
        peak_auditor_rss_bytes=envelope.peak_rss,requires_clean_owning_exit_and_independent_receipt_readback=True)
    c.save(c.OUT/'verification_summary.json',summary)
    return summary

def main():
    parser=argparse.ArgumentParser();parser.add_argument('--freeze-sha256',required=True);args=parser.parse_args()
    sys.dont_write_bytecode=True
    sys.addaudithook(c.audit_event)
    claim=c.read(c.OUT/'watchdog_claim.json');envelope=c.Envelope(claim)
    try:
        c.require(claim['freeze_sha256']==args.freeze_sha256,'Owning freeze identity mismatch')
        c.require(c.artifact(c.OUT/'freeze.json',envelope.guard)['sha256']==args.freeze_sha256,'Frozen recovery changed')
        freeze=c.read(c.OUT/'freeze.json')
        c.require(freeze['job_seconds']==c.JOB_SECONDS and freeze['automatic_retries']==0,'Recovery fixed bound changed')
        c.save(c.OUT/'auditor_placement.json',configure())
        c.verify(freeze['inputs'],envelope.guard)
        envelope.checkpoint('recovery_startup_passed')
        result=execute(freeze,envelope)
        envelope.final_guard()
        c.save(c.OUT/'completion.json',dict(time=c.now(),status='audit_checks_passed_pending_owning_exit',
            freeze_sha256=args.freeze_sha256,original_status='incomplete',elapsed_seconds=max(time.monotonic()-envelope.start,time.time()-envelope.wall_start),
            verification_status=result['status'],automatic_retries=0))
        envelope.final_guard()
    except BaseException:
        c.save(c.OUT/'failure.json',dict(time=c.now(),phase=envelope.phase,error=traceback.format_exc(),original_attempt_preserved=True))
        if not (c.OUT/'completion.json').exists():
            c.save(c.OUT/'completion.json',dict(time=c.now(),status='incomplete',freeze_sha256=args.freeze_sha256,original_status='incomplete',automatic_retries=0))
        raise

if __name__=='__main__':main()
