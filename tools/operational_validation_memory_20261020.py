"""Fresh supplemental RAM admission; original operational evidence stays read-only."""
from __future__ import annotations

import hashlib
import json
import math
import os
from pathlib import Path
import time

from tools.local_player_qualification_128.common import ROOT, ResourceHold, memory, now, read, save_new

SOURCE = ROOT/'outputs/live_game_operational_recovery_20261020_02'
BASE = ROOT/'outputs/live_game_operational_recovery_20261020_02_validation_02'
PRIOR_COUNT = 458
CAP, RESERVE, INITIALIZATION = 43200, 600, 60
PHYSICAL_BYTES, COMMIT_BYTES = 2*1024**3, 4*1024**3
INTERVAL, STABLE_SECONDS, REQUIRED_SAMPLES = 10, 120, 13
MAXIMUM_WAIT, CUMULATIVE_WAIT = 7200, 14400
NAME = 'analysis_resume_02'


def require(value, reason):
    if not value:
        raise ResourceHold(reason)


def finite(value):
    return type(value) in (int,float) and math.isfinite(value)


def artifact(path):
    path=Path(path).resolve()
    require(path.is_relative_to(ROOT.resolve()),'Admission artifact outside workspace')
    raw=path.read_bytes()
    return {'path':path.relative_to(ROOT).as_posix(),'sha256':hashlib.sha256(raw).hexdigest(),'bytes':len(raw)}


def prior_memory_state(source_base):
    """Read and bind the complete historical gate inventory, without changing it."""
    source=Path(source_base).resolve()
    require(source==SOURCE.resolve(),'Unexpected source memory evidence')
    paths=sorted((source/'memory_admissions').rglob('*.json'))
    require(len(paths)==PRIOR_COUNT,'Historical memory gate count differs')
    values=[];items=[]
    for path in paths:
        require(path.resolve().is_relative_to((source/'memory_admissions').resolve()),'Historical gate path escape')
        row=read(path)
        require(row['schema']=='operational_memory_admission_v2' and row['status']=='passed' and
                row['failure_category'] is None and row['failure_detail'] is None and
                row['name']==path.relative_to(source/'memory_admissions').with_suffix('').as_posix(),
                'Historical memory gate identity or status')
        numeric=('charged_seconds','started_epoch_seconds','finished_epoch_seconds',
                 'started_monotonic','finished_monotonic','cumulative_before_seconds','cumulative_after_seconds')
        require(all(finite(row[k]) for k in numeric),'Historical memory gate clock type')
        charged=max(row['finished_monotonic']-row['started_monotonic'],
                    row['finished_epoch_seconds']-row['started_epoch_seconds'],0.)
        require(row['charged_seconds']==charged and 0<=charged<MAXIMUM_WAIT and
                0<=row['cumulative_before_seconds']<=row['cumulative_after_seconds']<CUMULATIVE_WAIT and
                row['cumulative_after_seconds']==row['cumulative_before_seconds']+charged and
                type(row['prior_admission_count']) is int,'Historical memory gate charge')
        values.append(row);items.append(artifact(path))
    ordered=sorted(values,key=lambda row:row['prior_admission_count'])
    require([row['prior_admission_count'] for row in ordered]==list(range(PRIOR_COUNT)) and
            ordered[0]['name']=='initial' and ordered[-1]['name']=='analysis','Historical gate ordering')
    previous=[]
    for row in ordered:
        charged=sum(value['charged_seconds'] for value in sorted(previous,key=lambda value:value['name']+'.json'))
        require(math.isclose(row['cumulative_before_seconds'],charged,rel_tol=0.,abs_tol=1e-6),
                'Historical cumulative memory charge differs')
        previous.append(row)
    charged=sum(row['charged_seconds'] for row in values)
    require(charged<CUMULATIVE_WAIT,'Historical memory allowance exhausted')
    return {'count':PRIOR_COUNT,'charged_seconds':charged,'receipts':items}


def guard(launch, resources):
    started=time.monotonic()
    remaining=min(launch['epoch_seconds']+CAP-time.time(),launch['monotonic_start']+CAP-started)
    require(remaining>=RESERVE+INITIALIZATION,'Original execution reserve reached during supplemental memory wait')
    resources()
    remaining=min(remaining-(time.monotonic()-started),launch['epoch_seconds']+CAP-time.time(),
                  launch['monotonic_start']+CAP-time.monotonic())
    require(remaining>=RESERVE+INITIALIZATION,'Original execution reserve reached during supplemental memory wait')
    return remaining


def wait_for_memory(source_base, repair_base, launch, prior_state, *, resources):
    """One new mandatory stable window, charged to the unchanged original deadline."""
    source,base=Path(source_base).resolve(),Path(repair_base).resolve()
    require(source==SOURCE.resolve() and base==BASE.resolve(),'Unexpected supplemental admission paths')
    require(isinstance(launch,dict) and launch.get('maximum_seconds')==CAP and
            finite(launch.get('epoch_seconds')) and finite(launch.get('monotonic_start')) and
            read(source/'launch.json')==launch,'Original launch identity differs')
    path,log=base/'memory_admission.json',base/'memory_admission.jsonl'
    require(not path.exists() and not log.exists(),'Supplemental memory admission already attempted')
    guard(launch,resources)
    require(prior_memory_state(source)==prior_state,'Historical gate inventory or seeded charge changed')
    before=prior_state['charged_seconds']
    started_mono,started_epoch=time.monotonic(),time.time()
    rows=[];streak=0;first_good=None
    status,category,detail='preserved_hold',None,None
    completed_mono=completed_epoch=None
    base.mkdir(parents=True,exist_ok=True)
    try:
        with log.open('x',encoding='utf-8',newline='\n') as stream:
            while True:
                charged=max(time.monotonic()-started_mono,time.time()-started_epoch,0.)
                require(charged<MAXIMUM_WAIT and before+charged<CUMULATIVE_WAIT,'Supplemental memory allowance exhausted')
                remaining=guard(launch,resources)
                read_started=time.monotonic()
                value=memory();stamp,epoch=time.monotonic(),time.time()
                charged=max(stamp-started_mono,epoch-started_epoch,0.)
                require(charged<MAXIMUM_WAIT and before+charged<CUMULATIVE_WAIT,'Supplemental memory allowance exhausted')
                physical,commit=value['available_physical'],value['available_page']
                require(type(physical) is int and type(commit) is int and min(physical,commit)>=0,
                        'Invalid supplemental memory reading')
                good=physical>=PHYSICAL_BYTES and commit>=COMMIT_BYTES
                streak=streak+1 if good else 0
                first_good=(stamp if streak==1 else first_good) if good else None
                row={'epoch_seconds':epoch,'monotonic_seconds':stamp,'elapsed_seconds':stamp-started_mono,
                     'available_physical_bytes':physical,'available_commit_bytes':commit,'admissible':good,
                     'consecutive_admissible_samples':streak,'remaining_seconds':remaining}
                rows.append(row);stream.write(json.dumps(row,sort_keys=True,allow_nan=False)+'\n');stream.flush()
                print(json.dumps({'stage':'supplemental_memory_wait','name':NAME,**row,'outcomes_closed':True},
                                 sort_keys=True,allow_nan=False),flush=True)
                if good and streak>=REQUIRED_SAMPLES and stamp-first_good>=STABLE_SECONDS:
                    break
                left=min(MAXIMUM_WAIT-charged,CUMULATIVE_WAIT-before-charged,
                         remaining-(time.monotonic()-read_started)-RESERVE-INITIALIZATION,
                         launch['epoch_seconds']+CAP-time.time()-RESERVE-INITIALIZATION,
                         launch['monotonic_start']+CAP-time.monotonic()-RESERVE-INITIALIZATION)
                require(left>0,'Original execution reserve reached during supplemental memory wait')
                time.sleep(min(INTERVAL,left))
            os.fsync(stream.fileno())
        guard(launch,resources)
        completed_mono,completed_epoch=time.monotonic(),time.time()
        charged=max(completed_mono-started_mono,completed_epoch-started_epoch,0.)
        require(charged<MAXIMUM_WAIT and before+charged<CUMULATIVE_WAIT,
                'Supplemental memory allowance exhausted during final admission check')
        status='passed'
    except BaseException as error:
        status='preserved_hold';category,detail=type(error).__name__,str(error)
        raise
    finally:
        finished_mono,finished_epoch=((completed_mono,completed_epoch) if status=='passed' else
                                      (time.monotonic(),time.time()))
        charged=max(finished_mono-started_mono,finished_epoch-started_epoch,0.)
        raw=log.read_bytes() if log.exists() else b''
        receipt={'schema':'operational_memory_admission_v2','supplemental':True,'time':now(),'name':NAME,
                 'status':status,'failure_category':category,'failure_detail':detail,'initial':False,'stability_required':True,
                 'started_epoch_seconds':started_epoch,'started_monotonic':started_mono,
                 'finished_epoch_seconds':finished_epoch,'finished_monotonic':finished_mono,
                 'elapsed_seconds':finished_mono-started_mono,'charged_seconds':charged,
                 'prior_admission_count':prior_state['count'],'cumulative_before_seconds':before,
                 'cumulative_after_seconds':before+charged,'prior_gate_inputs':prior_state['receipts'],
                 'maximum_wait_seconds':MAXIMUM_WAIT,'maximum_cumulative_wait_seconds':CUMULATIVE_WAIT,
                 'minimum_available_physical_bytes':PHYSICAL_BYTES,'minimum_available_commit_bytes':COMMIT_BYTES,
                 'sample_interval_seconds':INTERVAL,'stable_seconds':STABLE_SECONDS,'required_consecutive_samples':REQUIRED_SAMPLES,
                 'reserve_seconds':RESERVE,'request_seconds':0,'initialization_seconds':INITIALIZATION,
                 'readings':rows,'readings_log':{'path':log.relative_to(base).as_posix(),
                 'sha256':hashlib.sha256(raw).hexdigest(),'bytes':len(raw)},
                 'no_worker_during_admission':True,'outcomes_closed':True}
        save_new(path,receipt)
    return receipt
