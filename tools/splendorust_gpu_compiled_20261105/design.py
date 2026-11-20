"""Prospective fixed-B8 compilation comparison; no model imports or probes."""
from copy import deepcopy
from . import common as c

PRIOR=c.ROOT/'outputs/splendorust_gpu_costs_20261105_01'
REPORT=c.ROOT/'outputs/splendorust_gpu_costs_report_20261105_01'
PLAN_SHA='3576f6ad2e011c6702139becf782902e1296518de8a0e531ff21027dd2f50dac'
AUDIT_SHA='d414066b23a3e1e73c64079c0aa9cedb602918d8fd52eed760d05739f2437415'
CLOSEOUT_SHA='8f75354017732245f79638e27a77fcd1895a1e2472aeb931efc459b1ebeac78e'


def spec(name,block,implementation,phase):
    return dict(cell_id=name,configuration=implementation,implementation=implementation,block=block,
                phase=phase,route='gpu',concurrency=16,scheduler='row_fifo32',
                groups=[list(range(16)),list(range(16,32))])


def qualification_cell():return spec('qual_compiled',0,'compiled','decision_qualification')


def cells():
    return [spec(name,block,implementation,'timing') for name,block,implementation in (
        ('b1_eager',1,'eager'),('b1_compiled',1,'compiled'),
        ('b2_compiled',2,'compiled'),('b2_eager',2,'eager'))]


def support():
    return dict(physical_batch=8,warmup_calls_per_implementation=5,
                warmup_input='all_zero_physical_batch',fixtures=36,repetitions_per_fixture=2,
                order=['eager','compiled'],full_root_batches=4,single_occupied_batches=32,
                compare_all_float32_rows=True,require_exact_saved_occupied_rows=True)


def build_plan():
    from tools.future_training_rules_20261021 import verify_runtime_sources
    from tools.splendorust_gpu_probe_20261105.control import require_absent
    from . import backend
    for path,expected in ((PRIOR/'plan.json',PLAN_SHA),(REPORT/'audit_saved.json',AUDIT_SHA),
                          (REPORT/'closeout.json',CLOSEOUT_SHA)):
        c.require(c.sha(path)==expected,'Accepted predecessor identity')
    previous=c.read(PRIOR/'plan.json');accepted=c.read(REPORT/'audit_saved.json')
    c.require(accepted['status']=='saved_gpu_cost_diagnostic_verified','Accepted completed cost diagnostic')
    c.verify(previous['inputs'])
    for identity in accepted['exact_owned_processes_absent']:require_absent(identity)
    sources=verify_runtime_sources();c.require(sources==previous['source_guard'],'Corrected source identity')
    return dict(schema='splendorust_gpu_compiled_v1',time=c.now(),
        authorization='User requested implementation and execution of one bounded compiled-GPU test in a separate version',
        prior=dict(plan=c.artifact(PRIOR/'plan.json'),audit=c.artifact(REPORT/'audit_saved.json'),
            closeout=c.artifact(REPORT/'closeout.json'),frozen_inputs=previous['inputs'],
            preserved_evidence=[c.artifact(p) for p in sorted(PRIOR.rglob('*')) if p.is_file()],
            preserved_reports=[c.artifact(p) for p in sorted(REPORT.rglob('*')) if p.is_file()]),
        runtime=deepcopy(previous['runtime']),cases=deepcopy(previous['cases']),
        cells=cells(),qualification_cell=qualification_cell(),support=support(),source_guard=sources,
        compilation=dict(method='torch.jit.trace_then_freeze',physical_batch=8,
            example='all_zero_float32_on_selected_GPU_shape_8_31_48',check_trace=False,strict=True,
            optimize_numerics=False,alternative_options=0,precision='float32',
            primitive_source='tools/splendorust_gpu_probe_20261105/worker.py:make_primitive_forward',
            require_same_mathematical_operator_order=True,require_exact_outputs=True),
        compilation_source_equivalence=backend.source_equivalence(),
        reused_qualification=deepcopy(previous['reused_scheduler_qualification']),
        worker_affinity=previous['worker_affinity'],supervisor_affinity=previous['supervisor_affinity'],
        worker_limits=dict(qualification_seconds=2700,cell_seconds=900),
        total_seconds=c.TOTAL_SECONDS,model_seconds=c.MODEL_SECONDS,
        maximum_worker_rss_bytes=c.MAX_RSS,maximum_output_bytes=c.MAX_OUTPUT,minimum_free_disk_bytes=c.MIN_DISK,
        available_ram_reserve_bytes=0,forecast_admission_gate=False,group_gap_seconds=0.,
        internal_seconds=c.INTERNAL_SECONDS,outer_seconds=c.OUTER_SECONDS,execution_policy_id=c.EXECUTION_POLICY_ID,
        warmup=deepcopy(previous['warmup']),maximum_panel_decisions=160,timing_decisions=128,
        qualification_decisions_per_route=67,qualification_routes=['compiled_gpu'],
        primary_timing=previous['primary_timing'],case_mapping=previous['case_mapping'],
        exact_reproduction=previous['exact_reproduction'],timing_observer=False,
        useful_signal=dict(minimum_throughput_ratio_each_pair=1.05,complete_exact_cells_required=4,
            statistical_significance_claim=False,automatic_promotion=False),
        negative_outcomes=['compile_rejected','runtime_unsupported','exact_output_mismatch','exact_mismatch','deadline_failed'],
        failures='Preserve the first rejection and every released request; no retries, replacements, alternative compile flags or later timing stages after rejection',
        late_completed_responses_accepted=True,descriptive_latency_threshold_seconds=[1.,9.,10.],
        retries=0,new_games=0,training_updates=0,larger_data_deferred=True,
        automatic_followup_games=False,automatic_followup_training=False,protocol=c.artifact(c.PROTOCOL))
