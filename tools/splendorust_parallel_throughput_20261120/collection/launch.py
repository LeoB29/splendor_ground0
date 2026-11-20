"""Root integration. Calls only after future explicit launch and qualification."""
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
import time
from . import common as c, workflow
from .queueing import create_queue, SharedStop, assignments
from .. import common as root

MODULE=root.PACKAGE+'.collection.worker'


def qualify(plan,jobs,boundary=lambda:None):
    folder=c.OUT/'qualification/worker';audit=c.OUT/'qualification_audit/worker'
    work=jobs.launch_job(MODULE,folder,dict(mode='qualification',cpu=[plan['placement']['model_cpus'][0]]),model=True)
    checked=jobs.launch_job(MODULE,audit,dict(mode='audit_qualification',cpu=plan['placement']['audit_cpus'],qualification_folder=str(folder)),model=False)
    result=root.read(audit/'result.json')
    c.require(result['status']=='passed' and result['payload']['complete_decisions']==116 and result['payload']['engineering_games']==2,'Complete original collector startup qualification')
    return c.save(c.OUT/'qualification_gate.json',dict(status='passed',plan=root.artifact(root.PLAN),executions=[work,checked],audit_result=root.artifact(audit/'result.json')))


def run(plan,jobs,boundary=lambda:None):
    """Run B only; A/live remains owned by the parent pipeline."""
    cells=plan['collection']['cells'];record={};identity=[];lock=__import__('threading').RLock()
    def play(arm,spec):
        cell=spec['cell_id'];folder=c.OUT/'cells'/f'{arm}_{cell}';folder.mkdir(parents=True,exist_ok=False)
        import psutil
        supervisor=[19] if arm=='control' else plan['placement']['controller_cpus']
        with jobs.lock:
            psutil.Process().cpu_affinity(supervisor)
            c.require(psutil.Process().cpu_affinity()==supervisor,'Actual supervisor remains off model cores')
        c.save(folder/'supervisor_affinity.json',dict(arm=arm,cpus=supervisor,
            control_original19_preserved=arm=='control',candidate_distinct_physical_model_cores=arm=='candidate'))
        create_queue(folder/'queue.sqlite',spec['specifications'])
        count,clients=(1,8) if arm=='control' else (4,2)
        workers=[folder/'players'/f'p{i}'/'worker' for i in range(count)]
        def start(index):
            request=dict(mode='play',cell=cell,arm=arm,process_index=index,clients=clients,
                cpu=list(range(19)) if arm=='control' else [plan['placement']['model_cpus'][index]],
                cell_folder=str(folder),queue_path=str(folder/'queue.sqlite'),stop_path=str(folder/'stop.json'))
            return jobs.launch_job(MODULE,workers[index],request,model=True)
        with ThreadPoolExecutor(max_workers=count) as pool:
            futures=[pool.submit(start,i) for i in range(count)]
            try:
                while not all((p/'ready.json').exists() for p in workers):
                    boundary()
                    for future in futures:
                        if future.done():future.result();raise RuntimeError('Service exited before fixed release')
                    time.sleep(.025)
                c.save(folder/'release.json',dict(services=count,clients=8,
                    released_perf_counter=time.perf_counter(),released_unix=time.time(),
                    ready=[c.artifact(p/'ready.json') for p in workers]))
                exits=[future.result() for future in futures]
            except BaseException:
                SharedStop(folder/'stop.json').set()
                for future in futures:
                    try:future.result()
                    except BaseException:pass
                raise
        queue=assignments(folder/'queue.sqlite');c.require(len(queue)==16,'All16 queue entries dispatched exactly once')
        started=dict(started_perf_counter=min(x['assigned_perf_counter'] for x in queue),started_unix=min(x['assigned_unix'] for x in queue))
        c.save(folder/'started.json',dict(cell_id=cell,**started))
        ended=max(c.read(folder/'games'/s['game_id']/'finished.json')['ended_perf_counter'] for s in spec['specifications'])
        assemble=folder/'assembly/worker'
        assembly=jobs.launch_job(MODULE,assemble,dict(mode='assemble',cpu=plan['placement']['audit_cpus'],
            cell=cell,cell_folder=str(folder),started=started,gameplay_end=ended),model=False)
        with lock:record[(arm,cell)]=dict(folder=folder,players=workers,play_exits=exits,assembly=assembly,started=started)
        return dict(status='completed')
    def check(arm,spec):
        cell=spec['cell_id'];info=record[(arm,cell)];folder=info['folder'];replay=folder/'replays/worker';audit=folder/'audit/worker'
        began=time.perf_counter()
        replay_exit=jobs.launch_job(MODULE,replay,dict(mode='replays',cpu=plan['placement']['audit_cpus'],cell=cell,cell_folder=str(folder)),model=True)
        audit_exit=jobs.launch_job(MODULE,audit,dict(mode='audit_cell',cpu=plan['placement']['audit_cpus'],cell=cell,
            cell_folder=str(folder),play_folders=[str(p) for p in info['players']],replay_folder=str(replay)),model=False)
        checked=root.read(audit/'result.json');c.require(checked['status']=='passed' and checked['payload']['status']=='passed','All accepted per-cell audits')
        proof=dict(status='passed',begun_perf_counter=began,verified_perf_counter=time.perf_counter(),
            replay_exit=replay_exit,audit_exit=audit_exit,audit_result=root.artifact(audit/'result.json'))
        if (arm,cell) in (('candidate','X'),('control','Y')):
            identity_folder=c.OUT/'identity'/cell/'worker'
            identity_begun=time.perf_counter()
            receipt=jobs.launch_job(MODULE,identity_folder,dict(mode='identity',cpu=plan['placement']['audit_cpus'],cell=cell,
                control=str(record[('control',cell)]['folder']),candidate=str(record[('candidate',cell)]['folder'])),model=False)
            # Any X mismatch propagates immediately to the coordinator while Y
            # is still running. No continuation waits for a final summary.
            identity.append(receipt);proof['identity_exit']=receipt
            proof['identity_completed_perf_counter']=time.perf_counter()
            proof['shared_identity_seconds']=proof['identity_completed_perf_counter']-identity_begun
        c.save(folder/'verified.json',proof)
        with lock:info['checked']=proof
        return proof
    def stop(arm,spec):SharedStop(c.OUT/'cells'/f'{arm}_{spec["cell_id"]}'/'stop.json').set()
    def quarantine(arm,spec,reason):
        path=c.OUT/'cells'/f'{arm}_{spec["cell_id"]}'/'quarantine.json'
        if not path.exists():c.save(path,dict(status='quarantined',reason=reason,no_retry=True,all_evidence_preserved=True,training_labels_usable=False))
    proof=workflow.run_comparison(cells,play,check,stop,quarantine,boundary)
    cx,cy=record[('candidate','X')],record[('candidate','Y')]
    c.require(cy['started']['started_perf_counter']<cx['checked']['verified_perf_counter'] and
        cx['checked']['begun_perf_counter']<c.read(cy['folder']/'cohort.json')['ended_perf_counter'],
        'Mandatory actual candidateX-check / candidateY-play overlap occurred')
    metrics=measure(record)
    result=dict(status='passed',workflow=proof,identity_exits=identity,metrics=metrics,
        qualifies=metrics['candidate']['verified_games_per_hour']/metrics['control']['verified_games_per_hour']>=1.3,
        adoption_bar=1.3,engineering_only=True,training_use=False)
    c.save(c.OUT/'result.json',result)
    return result


def measure(records):
    from tools.splendorust_gpu_throughput_20261105.audit import timing
    result={}
    identity_seconds=sum(info['checked'].get('shared_identity_seconds',0.) for info in records.values())
    for arm in ('control','candidate'):
        infos=[records[(arm,key)] for key in ('X','Y')];durations=[];plies=0;roots=0
        for info in infos:
            cohort=c.read(info['folder']/'cohort.json');plies+=cohort['plies'];roots+=cohort['eligible_roots']
            for path in (info['folder']/'games').glob('*/clocks/*.json'):
                durations.append(c.read(path)['elapsed_seconds'])
        seconds=(sum(x['checked']['verified_perf_counter']-x['started']['started_perf_counter'] for x in infos) if arm=='control'
            else infos[-1]['checked']['verified_perf_counter']-infos[0]['started']['started_perf_counter'])
        charged=seconds+identity_seconds/2
        result[arm]=dict(games=32,decisions=plies,eligible_roots=roots,verified_seconds=charged,
            pipeline_seconds=seconds,shared_cross_arm_identity_seconds_charged=identity_seconds/2,
            verified_games_per_hour=32*3600/charged,verified_decisions_per_hour=plies*3600/charged,
            complete_response_seconds=timing(durations),
            cell_completion_cadence_seconds=infos[1]['checked']['verified_perf_counter']-infos[0]['checked']['verified_perf_counter'] if arm=='candidate' else None,
            cadence_scope='One observed X-to-Y completion transition; not asymptotic steady-state throughput',
            scope='First actual assignment through replay/audit/actual exits, plus half total cross-arm identity wall time to each arm; conservative candidate charge may overlap Y gameplay')
    return result
