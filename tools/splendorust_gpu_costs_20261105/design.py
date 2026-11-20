"""Prospective, model-free binding of the unchanged GPU16 cost diagnostic."""
from copy import deepcopy
from . import common as c

PRIOR = c.ROOT/'outputs/splendorust_gpu_batchfill_20261105_01'
REPORT = c.ROOT/'outputs/splendorust_gpu_batchfill_report_20261105_01'
PLAN_SHA = 'c53b0a0aae4d333e2fbb808eaf1b93a928b83ca1eeca00b95832d45a732fe1aa'
AUDIT_SHA = '507963f3bd6fda1f9a9ccc6750b564514c998f520d89c46a59007b35927ca94d'
CLOSEOUT_SHA = '5f671a8378e6009f2112f9b4536b6a9573992db38ab64103e130f09edd62cfde'


def cells():
    choices = [('oh1_control',1,False,'overhead'), ('oh1_observed',1,True,'overhead'),
               ('oh2_observed',2,True,'overhead'), ('oh2_control',2,False,'overhead'),
               ('diag1',3,True,'diagnostic'), ('diag2',4,True,'diagnostic')]
    return [dict(cell_id=name, configuration='gpu16', block=block, observed=observed, phase=phase,
                 route='gpu', concurrency=16, scheduler='row_fifo32',
                 groups=[list(range(16)),list(range(16,32))])
            for name,block,observed,phase in choices]


def isolated():
    return [dict(cell_id=f'iso{block}_{side}', block=block, side=side, observed=True, route='gpu',
                 scheduler='row_fifo32', physical_batch=8, warmup_cycles=5, measured_cycles=32,
                 batches=[list(range(start,start+8)) for start in range(0,32,8)])
            for block in (1,2) for side in ('pre','post')]


def build_plan():
    from tools.future_training_rules_20261021 import verify_runtime_sources
    from tools.splendorust_gpu_probe_20261105.control import require_absent
    from . import backend
    for path,expected in ((PRIOR/'plan.json',PLAN_SHA),(REPORT/'audit_saved.json',AUDIT_SHA),
                          (REPORT/'closeout.json',CLOSEOUT_SHA)):
        c.require(c.sha(path)==expected,'Authenticated accepted predecessor')
    previous=c.read(PRIOR/'plan.json'); accepted=c.read(REPORT/'audit_saved.json')
    c.require(accepted['status']=='saved_batchfill_diagnostic_verified','Accepted predecessor')
    c.verify(previous['inputs'])
    for identity in accepted['exact_owned_processes_absent']: require_absent(identity)
    sources=verify_runtime_sources(); c.require(sources==previous['source_guard'],'Corrected source identity')
    return dict(schema='splendorust_gpu_costs_v1',time=c.now(),
        authorization='User requested bounded GPU16 execution/preparation/communication/recording cost diagnostic',
        prior=dict(plan=c.artifact(PRIOR/'plan.json'),audit=c.artifact(REPORT/'audit_saved.json'),
            closeout=c.artifact(REPORT/'closeout.json'),frozen_inputs=previous['inputs'],
            preserved_evidence=[c.artifact(p) for p in sorted(PRIOR.rglob('*')) if p.is_file()],
            preserved_reports=[c.artifact(p) for p in sorted(REPORT.rglob('*')) if p.is_file()]),
        runtime=deepcopy(previous['runtime']),cases=deepcopy(previous['cases']),cells=cells(),isolated=isolated(),
        source_guard=sources,observer_source_equivalence=backend.source_equivalence(),
        worker_affinity=previous['worker_affinity'],supervisor_affinity=previous['supervisor_affinity'],
        worker_limits=dict(qualification_seconds=2700,cell_seconds=900,isolated_seconds=900),
        total_seconds=c.TOTAL_SECONDS,model_seconds=c.MODEL_SECONDS,
        maximum_worker_rss_bytes=c.MAX_RSS,maximum_output_bytes=c.MAX_OUTPUT,minimum_free_disk_bytes=c.MIN_DISK,
        available_ram_reserve_bytes=0,forecast_admission_gate=False,group_gap_seconds=0.,
        internal_seconds=c.INTERNAL_SECONDS,outer_seconds=c.OUTER_SECONDS,execution_policy_id=c.EXECUTION_POLICY_ID,
        warmup=deepcopy(previous['warmup']),maximum_diagnostic_decisions=192,qualification_decisions_per_route=67,
        qualification_routes=['gpu'],reused_scheduler_qualification=c.artifact(REPORT/'audit_saved.json'),
        overhead=dict(order=['oh1_control','oh1_observed','oh2_observed','oh2_control'],
            pooled_max_ratio=1.05,each_pair_max_ratio=1.10,minimum_complete_cells=4,
            outcome_if_not_passed='valid negative observer qualification; no full diagnostic or retry'),
        full_stage_order=['iso1_pre','diag1','iso1_post','iso2_pre','diag2','iso2_post'],
        primary_timing=previous['primary_timing'],case_mapping=previous['case_mapping'],
        exact_reproduction=previous['exact_reproduction'],
        timing_scope='Host wall intervals; synchronized network roundtrip includes transfers and blocking output copy, not pure device kernels. Pipe read includes native search and scheduling. Concurrent spans overlap and are not additive wall shares.',
        isolation_scope='Same32 saved root vectors in four full physicalB8 batches under separately loaded isolated workers; live panel includes all hypothetical callbacks. Same-input root subset comparison separately from all live calls.',
        recording_policy='Keep every existing journal/check; buffer only new observer records and export outside measured panel',
        late_completed_responses_accepted=True,descriptive_latency_threshold_seconds=[1.,9.,10.],
        retries=0,new_games=0,training_updates=0,larger_data_deferred=True,
        automatic_followup_games=False,automatic_followup_training=False,protocol=c.artifact(c.PROTOCOL))
