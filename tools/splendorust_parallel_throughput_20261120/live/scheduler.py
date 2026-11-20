"""Fixed once-only independent process lanes; owner callback retains OS custody."""
from concurrent.futures import ThreadPoolExecutor,as_completed
from pathlib import Path
import time,traceback
from . import common as c,queue,schedule,audit

def run_stage(folder,plan,owner,boundary,specs,arm,phase,launch_job):
    """launch_job(module, folder, request) blocks and returns actual execution.

    The callback must independently enforce actual exit0 and complete owned
    absence before returning. It must keep the original global/model clocks.
    This helper does not create unowned subprocesses or extend any deadline.
    """
    c.require(arm==schedule.phase_arm(phase) and specs==schedule.phase_specs(phase),'Exact counterbalanced phase')
    folder=Path(folder);folder.mkdir(parents=True,exist_ok=False)
    item=queue.prepare(folder/'queue',specs,arm,phase)
    placement=plan['placement'];count=1 if arm=='A1' else 4
    requests=[dict(kind='lane',mode='lane',queue=item,lane=i,model_cpu=placement['model_cpus'][i],
        controller_cpus=placement['controller_cpus'],original_allowed=placement['original_allowed']) for i in range(count)]
    started=time.monotonic();results=[];executions=[];errors=[]
    with ThreadPoolExecutor(max_workers=count) as executor:
        futures={executor.submit(launch_job,c.PACKAGE+'.worker',folder/f'lane{i}'/'worker',request):i for i,request in enumerate(requests)}
        for future in as_completed(futures):
            lane=futures[future]
            try:
                execution=future.result();executions.append(execution)
                results.append(audit.execution_payload(execution,requests[lane]))
            except BaseException:
                error=traceback.format_exc();queue.fail(item,lane,error);errors.append(error)
    c.require(not errors,'Live phase failed; all original evidence retained')
    boundary();results.sort(key=lambda r:r['lane'])
    c.require([r['lane'] for r in results]==list(range(count)),'Exact independent lane process inventory')
    c.require(len({(r['identity']['pid'],r['identity']['created']) for r in results})==count,
              'Each lane is an independent exact process')
    games=queue.verify(item,results)
    audits=[audit.verify_lane(r,plan,boundary) for r in results]
    sources=[source for value in audits for source in value['decision_sources']]
    c.require(len(sources)==len(set(sources)),'Raw response cannot be reused across independent lanes')
    result=dict(status='passed',arm=arm,phase=phase,queue=item,lanes=results,executions=executions,games=games,audits=audits,
        elapsed_seconds=time.monotonic()-started,includes_startup_graph_replays_exit_and_saved_checks=True)
    return dict(result=result,artifact=c.save(folder/'phase.json',result))
