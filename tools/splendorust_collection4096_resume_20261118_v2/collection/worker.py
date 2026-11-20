"""Fresh root-owned CPU collection and saved-audit entrypoints."""
import argparse
import os
from pathlib import Path
import time
import traceback
from . import common as c
from .. import common as root, runtime as owned


def initialize_runtime(operation,destination):
    started,wall=time.perf_counter(),time.time();error=None
    try:return operation()
    except BaseException:
        error=traceback.format_exc();raise
    finally:
        ended,endwall=time.perf_counter(),time.time()
        c.save(destination.parent/'initialization'/(destination.name+'.json'),dict(
            schema='collector_runtime_initialization_v1',route='cpu',implementation='eager',runtime_folder=str(destination),
            status='completed' if error is None else 'failed',error=error,started_perf_counter=started,
            ended_perf_counter=ended,elapsed_seconds=ended-started,started_unix=wall,ended_unix=endwall,
            no_retry=True,inside_collection_timing=False,
            scope='Complete runtime preparation including model load and compiled graph checks; not isolated compilation'))


def validate_plan(plan):
    from . import collector,qualification,storage
    from .. import schedule
    c.require(plan['schema']=='independent_corrected_production_collector_20261115_v1','Explicit production collector plan')
    for key,value in dict(games=collector.build_plan(),cells=collector.cells(),engineering_games=schedule.qualification_schedule(),
        qualification=qualification.build_plan(),storage=storage.contract(),qualification_routes=['cpu'],
        internal_seconds=60.,outer_seconds=75.,execution_policy_id=c.EXECUTION_POLICY_ID,
        total_seconds=168*3600,model_seconds=144*3600,retries=0,replacements=0,training_updates=0,
        maximum_new_games=schedule.COLLECTION_GAMES,excluded_qualification_games=2,
        maximum_model_replays=schedule.COLLECTION_GAMES*3,
        worker_affinity=list(range(19)),supervisor_affinity=[19],
        worker_limits=dict(qualification_seconds=6*3600,games_seconds=4*3600),
        available_ram_reserve_bytes=0,forecast_admission_gate=False).items():
        c.require(c.canonical(plan[key])==c.canonical(value),'Frozen collection contract '+key)
    c.require(plan['runtime']['source_guard']==plan['source_guard'] and
        plan['runtime']['native_capture_binary']==plan['native_capture']['binary'],'Corrected actual runtime/capture bindings')


def verify_full_qualification_gate():
    from tools.splendorust_collection4096_20261115.collection.worker import verify_full_qualification_gate as original
    c.require(root.sha(root.ORIGINAL_OUT/'freeze.json')==root.ORIGINAL_FREEZE_SHA,'Exact original freeze')
    return original()


def execute(folder,mode,expected,cell_id=None):
    from . import backend,collector,qualification,audit,storage
    folder=Path(folder).resolve()
    model_modes=('recovery_qualification','recovery_games','games')
    frozen,boundary,sources=owned.bootstrap(folder,expected,mode,model=mode in model_modes)
    c.require(frozen['collection_plan']==c.artifact(c.PLAN),'Root freezes exact separate collector plan')
    plan=c.read(c.PLAN);validate_plan(plan)
    c.require(sources==plan['source_guard'],'Same corrected startup sources')
    owner=root.read(folder.parent/'owner.json');identity=owner['identity']
    phase=root.read(root.resolve(owner['phase_clock']))
    expected_phase={'recovery_qualification':'supplemental','audit_recovery_qualification':'supplemental',
                    'saved_boundary':'supplemental','recovery_games':'supplemental',
                    'supplemental_audit':'supplemental','games':'collection',
                    'audit_cell':'collection','corpus':'corpus'}[mode]
    c.require(phase['stage']==expected_phase,'Collection role bound to the correct original phase')
    if mode not in model_modes:
        audit.set_boundary(boundary)
        if mode=='saved_boundary':
            from .. import partial135
            result=partial135.audit_saved(plan,boundary)
        elif mode=='audit_recovery_qualification':
            result=audit.audit_recovery_qualification(plan,c.OUT/'recovery_qualification')
        elif mode=='supplemental_audit':
            from .. import partial135
            result=partial135.audit_combined(plan,c.OUT/'recovery_games',boundary)
        elif mode=='audit_cell':
            spec=next(x for x in plan['cells'] if x['cell_id']==cell_id)
            result=audit.audit_cell(plan,spec,c.OUT/'cells'/cell_id)
        elif mode=='corpus':
            from .. import corpus
            result=corpus.accept_all(plan,boundary=boundary)
        else:raise ValueError('Unrecognized saved-audit mode')
        c.save(folder/'saved_audit.json',result)
        c.save(folder/'result.json',dict(status='passed',mode=mode,identity=identity,result=c.artifact(folder/'saved_audit.json'),model_calls=0))
        boundary();return 0
    import psutil
    process=psutil.Process();process.cpu_affinity(plan['worker_affinity'])
    c.save(folder/'worker_identity.json',dict(pid=identity['pid'],created=identity['created'],mode=mode,cell_id=cell_id,
        worker_affinity=process.cpu_affinity(),global_claim=c.artifact(c.CLAIM),execution_policy_id=c.EXECUTION_POLICY_ID))
    if mode in ('games','recovery_games','recovery_qualification'):
        verify_full_qualification_gate()
        gate=c.read(root.ORIGINAL_OUT/'collection/qualification_gate.json');c.verify([gate['audit']])
        c.require(gate['passed'] is True and gate['collection_plan']==c.artifact(root.ORIGINAL_OUT/'collection/plan.json'),
                  'Original independently qualified production admission')
        c.save(folder/'qualification_gate_binding.json',gate)
        audit.verify_gate_binding(plan,folder)
    if mode=='recovery_games':
        for path in ('recovery_boundary','recovery_qualification','recovery_qualification_audit'):
            receipt=c.read(c.OUT/path/'execution.json')
            c.require(receipt['status']=='passed' and receipt['worker_exit_code']==0 and not receipt['cleanup_errors'],
                      'Saved boundary and actual corrected recovery qualifier passed before supplemental models')
    if mode=='games':
        gate=root.read(root.OUT/'recovery_gate.json')
        c.require(gate['status']=='passed','Full recovery qualification completed before unstarted games')
        c.verify(gate['actual_exits'])
        c.require(len(gate['actual_exits'])==5,'All five actual recovery worker exits')
    c.require(backend.source_equivalence()==plan['backend_source_equivalence'] and backend.capture_provenance()==plan['native_capture'],'Original scientific execution source')
    for key in ('OMP_NUM_THREADS','MKL_NUM_THREADS','OPENBLAS_NUM_THREADS','NUMEXPR_NUM_THREADS'):os.environ[key]='1'
    runtime=None;error=None;cleanup=[];result=None
    try:
        destination=folder/('cpu_runtime' if mode=='recovery_qualification' else 'runtime')
        boundary()
        runtime=initialize_runtime(lambda:backend.build_runtime(plan['runtime'],'cpu',destination,
            runtime_sources=sources,scheduler='original8',implementation='eager'),destination)
        if mode=='recovery_qualification':
            c.require(cell_id is None,'Fixture qualifier does not use a production cell')
            fixtures=qualification.run_qualification(plan,'cpu',runtime,folder/'cpu',boundary)
            result=dict(status='qualified',routes={'cpu':fixtures},complete_decisions=116,
                new_games=0,original_engineering_games_reused=2,training_updates=0)
        elif mode=='recovery_games':
            from .. import partial135
            c.require(cell_id is None,'Supplement uses the single frozen interrupted cell')
            result=partial135.run_continuation(plan,runtime,folder/'cell',boundary)
        else:
            matches=[x for x in plan['cells'] if x['cell_id']==cell_id]
            c.require(len(matches)==1,'Exact single frozen production cell')
            result=collector.run_cell(plan,matches[0],runtime,folder/'cell',boundary)
            c.require(result['status']=='completed','Preserved incomplete production cell, no retry/replacement')
    except BaseException:
        error=traceback.format_exc();c.save(folder/'worker_failure.json',dict(error=error,time=c.now(),mode=mode,cell_id=cell_id,no_retry=True))
        if runtime is not None:
            try:runtime.abort()
            except BaseException as caught:cleanup.append(repr(caught))
    finally:
        if runtime is not None:
            try:
                runtime.close()
                c.require(not runtime.active and not runtime.service.pending and not runtime.service.clients,'All original operations closed')
                c.save(runtime.folder/'final_identity.json',dict(time=c.now(),identity=runtime.identity,
                    weights_verified_unchanged=True,service_closed=True,active_operations=runtime.active,
                    completed_requests=runtime.service.completed_requests,physical_batches=runtime.service.batch_count))
                if error is None:
                    storage.seal_journal(runtime.folder/'inference.jsonl')
            except BaseException as caught:cleanup.append(repr(caught))
        c.save(folder/'cleanup.json',dict(time=c.now(),success=not cleanup,errors=cleanup))
    if error or cleanup:return 1
    boundary()
    c.save(folder/'worker_result.json',dict(mode=mode,cell_id=cell_id,status='passed',result=result,exit_code=0,
        no_retry=True,reference_weights_unchanged=True,training_updates=0,execution_policy_id=c.EXECUTION_POLICY_ID))
    c.save(folder/'result.json',dict(status='passed',mode=mode,identity=identity,result=c.artifact(folder/'worker_result.json')))
    return 0


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--mode',required=True,choices=('saved_boundary','recovery_qualification','audit_recovery_qualification',
        'recovery_games','supplemental_audit','games','audit_cell','corpus'))
    parser.add_argument('--plan-sha256',required=True);parser.add_argument('--output',required=True)
    parser.add_argument('--cell-id');args=parser.parse_args()
    return execute(Path(args.output),args.mode,args.plan_sha256,args.cell_id)


if __name__=='__main__':raise SystemExit(main())
