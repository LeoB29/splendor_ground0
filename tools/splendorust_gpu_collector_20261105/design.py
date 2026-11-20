"""Prospective complete-game comparison, with no model work during planning."""
from copy import deepcopy
from . import common as c

PRIOR=c.ROOT/'outputs/splendorust_gpu_compiled_20261105_01'
REPORT=c.ROOT/'outputs/splendorust_gpu_compiled_report_20261105_01'
PLAN_SHA='7da5ff94a0da0f4360eb59ec95f0d481f0afddb189eae891c1e795ff85a1c2e2'
AUDIT_SHA='877452898d6e7a0944c913abb5e47def7b9ce931de7ad0fd47fc02cc8388e204'
CLOSEOUT_SHA='f3becc1433cb025b38cde503fd235380e55610b669a59f0455cc89630989d6b0'


def cells(game_plan=None):
    if game_plan is None:
        from . import collector
        game_plan=collector.build_plan()
    result=[]
    for block,route in ((1,'cpu'),(1,'gpu'),(2,'gpu'),(2,'cpu')):
        indices=list(range((block-1)*16,block*16))
        result.append(dict(cell_id=f'b{block}_{route}',block=block,route=route,
            concurrency=8 if route=='cpu' else 16,
            scheduler='original8' if route=='cpu' else 'row_fifo32',
            implementation='eager' if route=='cpu' else 'compiled',
            game_indices=indices,specifications=[deepcopy(game_plan['specifications'][i]) for i in indices]))
    return result


def build_plan():
    from tools.future_training_rules_20261021 import verify_runtime_sources
    from tools.splendorust_gpu_probe_20261105.control import require_absent
    from . import backend,collector,qualification
    for p,expected in ((PRIOR/'plan.json',PLAN_SHA),(REPORT/'audit_saved.json',AUDIT_SHA),(REPORT/'closeout.json',CLOSEOUT_SHA)):
        c.require(c.sha(p)==expected,'Accepted compiled predecessor identity')
    previous=c.read(PRIOR/'plan.json'); accepted=c.read(REPORT/'audit_saved.json')
    c.require(accepted['status']=='saved_compiled_comparison_verified','Completed exact compiled comparison')
    c.verify(previous['inputs'])
    for identity in accepted['exact_owned_processes_absent']:require_absent(identity)
    sources=verify_runtime_sources();c.require(sources==previous['source_guard'],'Unchanged corrected source identities')
    capture=backend.capture_provenance()
    runtime=deepcopy(previous['runtime']);runtime['native_capture_binary']=deepcopy(capture['binary'])
    game_plan=collector.build_plan()
    return dict(schema='splendorust_compiled_gpu_collector_v1',time=c.now(),
        authorization='User authorized actual collector qualification and bounded full-game compiledGPU16 versus CPU8 comparison',
        prior=dict(plan=c.artifact(PRIOR/'plan.json'),audit=c.artifact(REPORT/'audit_saved.json'),
            closeout=c.artifact(REPORT/'closeout.json'),frozen_inputs=previous['inputs'],
            preserved_evidence=[c.artifact(p) for p in sorted(PRIOR.rglob('*')) if p.is_file()],
            preserved_reports=[c.artifact(p) for p in sorted(REPORT.rglob('*')) if p.is_file()]),
        runtime=runtime,source_guard=sources,cases=deepcopy(previous['cases']),
        compiled_identity=deepcopy(accepted['qualification']['compiled_identity']),
        compilation_fingerprint=accepted['qualification']['compilation_fingerprint'],
        compilation=deepcopy(previous['compilation']),native_capture=capture,
        backend_source_equivalence=backend.source_equivalence(),
        qualification=qualification.build_plan(),games=game_plan,cells=cells(game_plan),
        worker_limits=dict(qualification_seconds=3600,games_seconds=14400),
        total_seconds=c.TOTAL_SECONDS,model_seconds=c.MODEL_SECONDS,
        maximum_worker_rss_bytes=c.MAX_RSS,maximum_output_bytes=c.MAX_OUTPUT,minimum_free_disk_bytes=c.MIN_DISK,
        available_ram_reserve_bytes=0,forecast_admission_gate=False,
        worker_affinity=list(range(19)),supervisor_affinity=[19],
        internal_seconds=c.INTERNAL_SECONDS,outer_seconds=c.OUTER_SECONDS,execution_policy_id=c.EXECUTION_POLICY_ID,
        warmup=deepcopy(previous['warmup']),timing_observer=False,
        maximum_new_games=64,independent_setup_seeds=32,generated_trajectories_per_setup=2,
        maximum_model_replays=192,qualification_decisions_per_route=116,qualification_routes=['cpu','gpu'],
        qualification_fixture_trajectories=2,qualification_extra_continuations=2,
        primary_rate='Naturally completed games / first release through final durable corpus and actual loader acceptance',
        secondary_rates=['Eligible policy/value targets per collection hour','Complete decisions per collection hour',
                         'Verified games and targets per cell wall including startup, warmups, model replays and saved audit'],
        final_independent_audit='Separately report actual final audit time; never silently omit it from total study cost',
        practical_advantage=dict(metric='verified_completed_games_per_hour',minimum_gpu_over_cpu_each_block=1.05,
            complete_exact_cells_required=4,targets_rate_reported_separately=True,statistical_significance_claim=False,
            automatic_promotion=False,never_stops_collection_early=True),
        failures='Preserve first failed decision and all released work; no retries, reset clocks, replacement games or synthetic terminal labels; incomplete cohorts have no completed-cohort throughput claim',
        stop_at_first_failure=True,late_completed_responses_accepted=True,descriptive_latency_threshold_seconds=[1.,9.,10.],
        retries=0,replacements=0,training_updates=0,larger_data_deferred=True,
        corpus_scope='engineering_collector_qualification_only_not_authorized_for_training',
        automatic_followup_training=False,automatic_followup_collection=False,protocol=c.artifact(c.PROTOCOL))
