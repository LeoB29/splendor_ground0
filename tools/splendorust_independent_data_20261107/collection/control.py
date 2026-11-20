"""Model-free collection freeze preparation; root owns all processes and clocks."""
from copy import deepcopy
from pathlib import Path
from . import common as c, collector, qualification, backend, private, storage
from tools.splendorust_gpu_collector_20261105.control import check_registry, DecisionOuterDeadline


def prepare(reservation):
    c.require(not c.PLAN.exists(),'One new collection plan')
    c.verify(list(reservation.values()))
    source=c.ROOT/'outputs/splendorust_gpu_collector_20261105_01/plan.json'
    c.require(c.sha(source)=='45f5aaf9bd7d9a5dc485e1eaaa39c5f908edda887b332e83cc71811dd60b3563','Exact accepted predecessor plan')
    closeout=c.ROOT/'outputs/splendorust_gpu_collector_report_20261105_01/closeout.json'
    c.require(c.sha(closeout)=='2cdb693fd9a86e245ac752666ada49dd9d35b353d1a5da2e5811c80e0750f193','Exact accepted predecessor closeout')
    accepted=c.read(closeout)
    c.require(accepted['status']=='completed_with_recovered_final_verification' and accepted['recovered_verification_passed'] is True
        and accepted['original_study']==c.artifact(source),'Preserved recovered acceptance, not relabelled original audit')
    c.verify([accepted['recovered_audit'],accepted['actual_recovery_exit'],accepted['root_final_checks']])
    previous=c.read(source)
    c.require(previous['schema']=='splendorust_compiled_gpu_collector_v1','Accepted source collector contract')
    capture=backend.capture_provenance()
    runtime=deepcopy(previous['runtime'])
    c.require(runtime['native_capture_binary']==capture['binary'],'Unchanged accepted target bridge')
    from .. import schedule
    c.require(c.canonical(c.read(c.resolve(reservation['collection_schedule'])))==c.canonical(schedule.collection_schedule()),'Frozen production schedule')
    plan=dict(schema='independent_corrected_production_collector_20261107_v1',runtime=runtime,
        source_guard=previous['source_guard'],cases=deepcopy(previous['cases']),native_capture=capture,
        backend_source_equivalence=backend.source_equivalence(),qualification=qualification.build_plan(),
        engineering_games=schedule.qualification_schedule(),games=collector.build_plan(),cells=collector.cells(),
        storage=storage.contract(),worker_affinity=list(range(19)),supervisor_affinity=[19],
        worker_limits=dict(qualification_seconds=4*3600,games_seconds=4*3600),
        total_seconds=c.TOTAL_SECONDS,model_seconds=c.MODEL_SECONDS,
        internal_seconds=c.INTERNAL_SECONDS,outer_seconds=c.OUTER_SECONDS,execution_policy_id=c.EXECUTION_POLICY_ID,
        maximum_worker_rss_bytes=c.MAX_RSS,maximum_output_bytes=c.MAX_OUTPUT,minimum_free_disk_bytes=c.MIN_DISK,
        available_ram_reserve_bytes=0,forecast_admission_gate=False,retries=0,replacements=0,training_updates=0,
        maximum_new_games=1152,excluded_qualification_games=2,maximum_model_replays=3456,
        qualification_routes=['cpu'],qualification_decisions_per_route=116,
        original_source_plan=c.artifact(source),accepted_source_closeout=c.artifact(closeout),
        accepted_source_audit=accepted['recovered_audit'],accepted_source_exit=accepted['actual_recovery_exit'],reservation=reservation,
        warmup=deepcopy(previous['warmup']),timing_observer=False,
        corpus_scope='fresh_corrected_production_only_after_independent_acceptance')
    entries=private.inputs()+list(c.nested_artifacts(plan))
    unique={}
    for item in entries:
        c.require(unique.setdefault(item['path'],item)==item,'Consistent source descriptor')
    plan['inputs']=sorted(unique.values(),key=lambda x:x['path'])
    c.verify(plan['inputs'])
    return c.save(c.PLAN,plan)


def active_registry_paths(folder, mode):
    folder=Path(folder)
    return [folder/('cpu_runtime' if mode=='qualification' else 'runtime')/'active_decisions.json'] if mode in ('qualification','games') else []
