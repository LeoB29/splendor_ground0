"""Explicit owned collection entrypoints; never invoked during preparation."""
from pathlib import Path
import os
import time
import traceback
from . import common as c

MODEL_MODES={'qualification','play','replays'}


def active_registry_paths(folder,request):
    return [Path(folder)/'runtime/active_decisions.json'] if request['mode'] in MODEL_MODES else []


def run(folder,plan,owner,boundary,request):
    from tools import future_training_rules_20261021 as guard
    from .. import common as root, topology
    from . import preparation,backend,collector,qualification,audit,storage
    import psutil
    folder=Path(folder);selected=plan['collection'];mode=request['mode']
    sources=guard.verify_runtime_sources()  # Exactly once in every worker, including saved auditors.
    c.require(sources==selected['source_guard'],'Frozen corrected actual worker sources')
    c.save(folder/'runtime_sources.json',dict(sources=sources,identity=owner['identity'],
        plan=owner['plan'],original_claim=owner['claim'],mode=mode))
    preparation.validate_plan(selected)
    observed=topology.verify_startup(plan['topology'],plan['placement'])
    cpu=request['cpu'];placement=plan['placement']
    if mode=='play':
        expected=list(range(19)) if request['clients']==8 else [placement['model_cpus'][request['process_index']]]
    elif mode=='qualification':expected=[placement['model_cpus'][0]]
    else:expected=placement['audit_cpus']
    c.require(cpu==expected and set(cpu)<=set(placement['original_allowed']),'Frozen exact collection role affinity')
    process=psutil.Process();process.cpu_affinity(cpu)
    c.require(process.cpu_affinity()==cpu,'Actual collection pin selected before model construction')
    c.save(folder/'affinity.json',dict(mode=mode,cpu=cpu,actual=process.cpu_affinity(),observed_topology=observed,
        control_cpu8_original_mask_preserved=mode=='play' and request['clients']==8))
    boundary()
    if mode not in MODEL_MODES:
        if mode=='assemble':
            result=collector.assemble_cell(selected,selected['cells'][request['cell']],Path(request['cell_folder']),
                started=request['started'],gameplay_end=request['gameplay_end'],boundary=boundary)
        elif mode=='audit_qualification':
            result=audit.audit_qualification(selected,Path(request['qualification_folder']),boundary=boundary)
        elif mode=='audit_cell':
            result=audit.audit_cell(selected,selected['cells'][request['cell']],Path(request['cell_folder']),
                request['play_folders'],request['replay_folder'],boundary=boundary)
        elif mode=='identity':
            from .identity import compare_cells
            result=compare_cells(request['control'],request['candidate'],selected['cells'][request['cell']]['specifications'],boundary=boundary)
        else:raise ValueError('Unknown collection role')
        c.save(folder/'saved_result.json',result)
        return result
    c.require(backend.source_equivalence()==selected['backend_source_equivalence'] and
        backend.capture_provenance()==selected['native_capture'],'Unchanged accepted math and native capture sources')
    if mode!='qualification':
        gate_path=c.OUT/'qualification_gate.json';gate=c.read(gate_path);c.verify(gate['executions']+[gate['audit_result']])
        c.require(gate['status']=='passed' and gate['plan']==root.artifact(root.PLAN),'Exact independent actual qualification gate')
        audit_result=root.read(root.resolve(gate['audit_result']))
        c.require(audit_result['status']=='passed' and audit_result['payload']['complete_decisions']==116 and
            audit_result['payload']['engineering_games']==2,'All116 fixtures plus complete engineering trajectories independently checked')
        c.save(folder/'qualification_gate_binding.json',gate)
    for key in ('OMP_NUM_THREADS','MKL_NUM_THREADS','OPENBLAS_NUM_THREADS','NUMEXPR_NUM_THREADS'):os.environ[key]='1'
    runtime=None;error=None;cleanup=[]
    started,wall=time.perf_counter(),time.time()
    try:
        runtime=backend.build_runtime(selected['runtime'],'cpu',folder/'runtime',runtime_sources=sources,scheduler='original8',implementation='eager')
        c.save(folder/'initialization.json',dict(status='passed',started_perf_counter=started,ended_perf_counter=time.perf_counter(),
            started_unix=wall,ended_unix=time.time(),no_retry=True,inside_measured_collection=False))
        if mode=='qualification':
            fixtures=qualification.run_qualification(selected,'cpu',runtime,folder/'cpu',boundary)
            engineering=collector.run_engineering(selected,runtime,folder/'engineering',boundary)
            result=dict(status='passed',fixtures=fixtures,engineering=engineering,complete_decisions=116,engineering_games=2)
        elif mode=='play':
            session=collector.Session(selected,runtime,folder,process_index=request['process_index'],clients=request['clients'],
                cell_folder=request['cell_folder'],queue_path=request['queue_path'],stop_path=request['stop_path'],boundary=boundary)
            result=session.run();c.require(result['status']=='passed','Failed original game preserved; no retries')
        else:
            collector.warmup_runtime(selected,runtime,folder/'warmup')
            result=collector.run_replays(selected,selected['cells'][request['cell']],runtime,Path(request['cell_folder']),boundary)
    except BaseException:
        error=traceback.format_exc();c.save(folder/'worker_failure.json',dict(error=error,mode=mode,no_retry=True))
        if runtime is not None:
            try:runtime.abort()
            except BaseException as caught:cleanup.append(repr(caught))
        raise
    finally:
        if runtime is not None:
            try:
                runtime.close()
                c.require(not runtime.active and not runtime.service.pending and not runtime.service.clients,'All original operations closed')
                c.save(runtime.folder/'final_identity.json',dict(time=c.now(),identity=runtime.identity,
                    weights_verified_unchanged=True,service_closed=True,active_operations=runtime.active,
                    completed_requests=runtime.service.completed_requests,physical_batches=runtime.service.batch_count))
                if error is None:storage.seal_journal(runtime.folder/'inference.jsonl')
            except BaseException as caught:cleanup.append(repr(caught))
        c.save(folder/'cleanup.json',dict(success=not cleanup,errors=cleanup))
        c.require(not cleanup,'Exact collector cleanup and lossless journal closure')
    boundary()
    return result
