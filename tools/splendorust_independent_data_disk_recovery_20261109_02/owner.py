"""One owner for the fixed collection, four fits and paired strength assessment."""
import argparse
from copy import deepcopy
import os
from pathlib import Path
import shutil
import subprocess
import sys
import time
import traceback
import psutil
from . import common as c
from tools.splendorust_gpu_probe_20261105.control import require_absent, finish_owned_process


def child_environment(repetition=None):
    env = os.environ.copy()
    env.pop('KMP_DUPLICATE_LIB_OK', None)
    env.update(SELFPLAY_JOB_SCOPE='pilot', PYTHONDONTWRITEBYTECODE='1',
        PYTHONPATH=str(c.ROOT/'src')+os.pathsep+str(c.ROOT), OMP_NUM_THREADS='1',
        MKL_NUM_THREADS='1', OPENBLAS_NUM_THREADS='1', NUMEXPR_NUM_THREADS='1',
        CUDA_VISIBLE_DEVICES='', RAYON_NUM_THREADS='1')
    if repetition is not None:
        c.require(type(repetition) is int and repetition in (0, 1), 'Fixed repetition')
        env['INDEPENDENT_DATA_REPETITION'] = str(repetition)
    return env


def contract():
    return dict(schema=c.SCHEMA, total_seconds=c.TOTAL_SECONDS, model_seconds=c.MODEL_SECONDS,
        stage_seconds=c.STAGES, independent_reader_seconds=c.INDEPENDENT_REVIEW_SECONDS,
        training_games=1024, validation_games=64, test_games=64, collection_cells=72,
        games_per_cell=16, excluded_qualification_games=2, teacher_simulations=512,
        worlds=3, cpuct=.4, depth=64, precision='float32', collection_concurrency=8,
        collection_internal_seconds=60., collection_complete_seconds=75.,
        collection_cell_seconds=4*3600, fits=c.fits(), examples_per_fit=3072,
        updates_per_fit=96, batch_size=32, qualification_optimizer_calls=2,
        scientific_optimizer_calls=384, replay_optimizer_calls=4, max_optimizer_calls=390,
        main_games=400, operational_games=12, candidate_repetitions=[0,1], matched_simulations=128,
        internal_seconds=9., complete_response_seconds=10., deployment_seconds_descriptive=1.,
        margin_seconds_descriptive=.7, maximum_plies=2000, maximum_model_reproductions=192,
        maximum_live_lifecycles=850, nominal_live_lifecycles=92, reachable_live_lifecycle_maximum=812,
        collection_model_reproductions=3456, available_ram_gate=False,
        resource_forecasts_may_stop=False, maximum_evidence=c.MAX_EVIDENCE_BYTES,
        prefinal_evidence=c.PREFINAL_EVIDENCE_BYTES, actual_disk_floor=c.MIN_DISK,
        collection_audit_rss=c.COLLECTION_RSS, fit_offline_rss=c.FIT_RSS,
        strength_inference_rss=c.MAX_RSS_BYTES, automatic_retries=0,
        replacement_games=0, automatic_promotion=False, scheduled_monitor=False,
        external_and_six_world_deferred=True, rentals=0)


def source_inputs():
    from tools.splendorust_search512_20261101.preparation import python_dependencies
    sources = python_dependencies((c.ROOT/'tools/splendorust_independent_data_20261107').rglob('*.py'))
    tests = sorted((c.ROOT/'tests').glob('test*independent_data*20261107*.py'))
    return [c.artifact(path) for path in sources+tests]


def guard(claim, phase):
    mono, wall = time.monotonic(), time.time()
    c.require(claim['monotonic_origin'] <= mono < phase['monotonic_deadline']
              and claim['unix_origin'] <= wall < phase['unix_deadline'], 'Original phase/global clock expired')
    c.require(shutil.disk_usage(c.OUT).free >= c.MIN_DISK, 'Actual free disk floor')


def begin_phase(claim, name):
    if name == 'strength':
        return recover_phase(claim)
    c.require(name in ('strength_replay','final'),
              'Fixed study phase')
    mono, wall = time.monotonic(), time.time()
    cap = c.STAGES[name]
    cutoff = c.TOTAL_SECONDS if name == 'final' else c.MODEL_SECONDS
    value = dict(stage=name, claim=c.artifact(c.OUT/'claim.json'), monotonic=mono,
        unix_time=wall, cap_seconds=cap,
        monotonic_deadline=min(mono+cap, claim['monotonic_origin']+cutoff),
        unix_deadline=min(wall+cap, claim['unix_origin']+cutoff))
    guard(claim,value)
    return c.save(c.OUT/'stages'/f'{name}_start.json',value)


def end_phase(claim, descriptor):
    phase = c.read(c.resolve(descriptor)); guard(claim,phase)
    return c.save(c.OUT/'stages'/f"{phase['stage']}_end.json", dict(stage=phase['stage'],
        start=descriptor, monotonic=time.monotonic(), unix_time=time.time()))


def segment_id(mode, repetition, block_start=None):
    c.require(type(repetition) is int and repetition in (0,1), 'Fixed repetition')
    c.require(mode in ('prepare','resume_partial','qualification','operational','main_block','replay','saved',
                      'independent','aggregate','reader'), 'Fixed strength segment')
    if mode=='resume_partial':c.require(repetition==1,'Only the single saved interrupted repetition')
    if mode == 'main_block':
        c.require(type(block_start) is int and block_start in range(0,200,10), 'Fixed ten-game block')
    else:
        c.require(block_start is None, 'Only main blocks have an offset')
    return f'{mode}_r{repetition}' + (f'_g{block_start:03d}' if block_start is not None else '')


def validate_segment_claim(claim, folder, mode, repetition, block_start, expected):
    ident = segment_id(mode,repetition,block_start)
    c.require(type(claim) is dict and set(claim) == {'segment_id','mode','repetition','block_start',
        'claim','phase_clock','freeze_sha256','reader_clock'}, 'Exact strength segment claim fields')
    fields = dict(segment_id=ident,mode=mode,repetition=repetition,block_start=block_start,freeze_sha256=expected)
    c.require(c.canonical({k:claim[k] for k in fields}) == c.canonical(fields)
              and Path(folder).resolve() == c.OUT/'segments'/ident, 'Exact strength role and path')
    c.verify([claim['claim'],claim['phase_clock']])
    c.require(claim['claim'] == c.artifact(c.OUT/'claim.json')
              and c.read(c.OUT/'claim.json')['freeze'] == c.artifact(c.PLAN)
              and c.sha(c.PLAN) == expected, 'Original owner and exact freeze')
    phase = c.read(c.resolve(claim['phase_clock']))
    required = ('strength' if mode in ('prepare','resume_partial','qualification','operational','main_block')
                else 'strength_replay' if mode == 'replay' else 'final')
    c.require(phase['stage'] == required, 'Strength modes share the original phase deadline')
    return claim


def completed_games():
    return 319 + sum(1 for rep in (0,1) for _ in
               (c.OUT/'strength'/f'r{rep}'/'games/main').glob('g*/result.json'))


def run_segment(claim, phase_descriptor, expected, mode, repetition, block_start=None, reader_clock=None):
    from tools.local_player_qualification_128.clock import _WindowsJob
    ident = segment_id(mode,repetition,block_start)
    folder = c.OUT/'segments'/ident; folder.mkdir(parents=True,exist_ok=False)
    local = dict(segment_id=ident,mode=mode,repetition=repetition,block_start=block_start,
        claim=c.artifact(c.OUT/'claim.json'),phase_clock=phase_descriptor,freeze_sha256=expected,reader_clock=reader_clock)
    c.save(folder/'claim.json',local)
    phase = c.read(c.resolve(phase_descriptor))
    module = c.PACKAGE+('.strength.report' if mode in ('aggregate','reader') else '.strength.segment')
    args = [sys.executable,'-B','-u','-m',module,'--mode',mode,'--segment',str(folder),'--freeze-sha256',expected]
    if block_start is not None: args += ['--block-start',str(block_start)]
    proc = job = identity = None
    observed,cleanup,error,status = {},[],None,'failed'
    c.status(c.OUT/'stage.json',dict(time=c.now(),stage=phase['stage'],segment=ident,
        completed_main_games=completed_games(),status='starting'))
    try:
        guard(claim,phase)
        with (folder/'stdout.log').open('xb') as stdout,(folder/'stderr.log').open('xb') as stderr:
            proc = subprocess.Popen(args,cwd=c.ROOT,env=child_environment(repetition),stdout=stdout,stderr=stderr,
                creationflags=getattr(subprocess,'CREATE_NO_WINDOW',0))
            identity = dict(pid=proc.pid,created=psutil.Process(proc.pid).create_time()); job = _WindowsJob(proc.pid)
            c.atomic_save(folder/'owner.json',dict(identity=identity,args=args,
                segment_claim=c.artifact(folder/'claim.json'),created_monotonic=time.monotonic(),
                created_unix=time.time(),windows_job_assigned=True))
            while proc.poll() is None:
                guard(claim,phase)
                if reader_clock: guard(claim,c.read(c.resolve(reader_clock)))
                observe_tree(identity,observed)
                time.sleep(.5)
            guard(claim,phase)
            c.require(proc.returncode == 0,'Strength segment failed without retry: '+ident)
            result = c.read(folder/'result.json')
            c.require(result['status']=='passed' and result['mode']==mode and result['repetition']==repetition,
                      'Completed exact strength segment')
            c.require(result['pid']==identity['pid'] and result['created']==identity['created'],'Actual result owner')
            for item in [*observed.values(),identity]: require_absent(item)
            status='passed'
    except BaseException: error=traceback.format_exc()
    finally: cleanup_owned(proc,job,identity,observed,cleanup)
    if cleanup: status='failed'
    receipt=c.save(folder/'execution.json',dict(status=status,mode=mode,repetition=repetition,
        block_start=block_start,identity=identity,worker_exit_code=None if proc is None else proc.returncode,
        observed_owned_identities=list(observed.values()),cleanup_errors=cleanup,error=error,
        segment_claim=c.artifact(folder/'claim.json'),ended_monotonic=time.monotonic(),ended_unix=time.time()))
    c.require(status=='passed','Failed strength segment preserved: '+ident)
    return receipt


def observe_tree(identity, observed):
    total=0
    try:
        parent=psutil.Process(identity['pid'])
        c.require(parent.create_time()==identity['created'],'Exact owned process identity')
        for child in [parent]+parent.children(recursive=True):
            try:
                item=dict(pid=child.pid,created=child.create_time())
                observed[(item['pid'],item['created'])]=item
                total+=child.memory_info().rss
            except psutil.NoSuchProcess: pass
    except psutil.NoSuchProcess: pass
    return total


def cleanup_owned(proc,job,identity,observed,errors):
    if job:
        try: job.close()
        except BaseException as exc: errors.append(repr(exc))
    if proc:
        try: finish_owned_process(proc,identity)
        except BaseException as exc: errors.append(repr(exc))
    for item in [*observed.values(),*([identity] if identity else [])]:
        try: require_absent(item)
        except BaseException as exc: errors.append(repr(exc))


def output_bytes(folder):
    return sum(p.stat().st_size for p in Path(folder).rglob('*') if p.is_file())


def run_job(claim, phase_descriptor, expected, module, mode, container, *, seconds,
            rss, fit_id=None, extra_args=(), model=True, cell_deadline=None):
    """Fresh single-use process with actual exit/cleanup and clipped parent clocks."""
    from tools.local_player_qualification_128.clock import _WindowsJob
    from tools.splendorust_gpu_live_20261105_03.status_io import read_status_json
    container=Path(container); container.mkdir(parents=True,exist_ok=False)
    folder=container/'worker'; folder.mkdir()
    phase=c.read(c.resolve(phase_descriptor)); begun,wall=time.monotonic(),time.time()
    deadline=min(begun+seconds,phase['monotonic_deadline'])
    wall_deadline=min(wall+seconds,phase['unix_deadline'])
    if cell_deadline:
        deadline=min(deadline,cell_deadline['monotonic_deadline'])
        wall_deadline=min(wall_deadline,cell_deadline['unix_deadline'])
    args=[sys.executable,'-B','-u','-m',module,'--mode',mode,'--plan-sha256',expected,'--output',str(folder)]
    if fit_id is not None: args += ['--fit-id',fit_id]
    args+=list(extra_args)
    proc=job=identity=None
    observed,cleanup,error,status={},[],None,'failed'
    peak=0; registries=set(); deep_at=-float('inf')
    # Closed evidence is counted once. The active subtree is sampled; no repeated full-study walk per move.
    baseline=c.preserved_bytes()+output_bytes(c.OUT)
    cap=c.MAX_EVIDENCE_BYTES if phase['stage']=='final' else c.PREFINAL_EVIDENCE_BYTES
    c.status(c.OUT/'stage.json',dict(time=c.now(),stage=phase['stage'],mode=mode,fit_id=fit_id,
        container=container.relative_to(c.OUT).as_posix(),status='starting'))
    try:
        guard(claim,phase)
        c.require(baseline<cap,'Actual evidence bound before worker')
        with (folder/'stdout.log').open('xb') as stdout,(folder/'stderr.log').open('xb') as stderr:
            proc=subprocess.Popen(args,cwd=c.ROOT,env=child_environment(),stdout=stdout,stderr=stderr,
                creationflags=getattr(subprocess,'CREATE_NO_WINDOW',0))
            identity=dict(pid=proc.pid,created=psutil.Process(proc.pid).create_time());job=_WindowsJob(proc.pid)
            c.atomic_save(container/'owner.json',dict(time=c.now(),identity=identity,args=args,mode=mode,
                fit_id=fit_id,global_claim=c.artifact(c.OUT/'claim.json'),phase_clock=phase_descriptor,
                monotonic_origin=begun,unix_origin=wall,monotonic_deadline=deadline,unix_deadline=wall_deadline,
                maximum_seconds=seconds,maximum_rss_bytes=rss,windows_job_assigned=True,plan_sha256=expected,
                model_work=model))
            while proc.poll() is None:
                guard(claim,phase)
                c.require(time.monotonic()<deadline and time.time()<wall_deadline,'Actual clipped worker clock')
                used=observe_tree(identity,observed);peak=max(peak,used)
                c.require(used<=rss,'Actual aggregate owned worker RSS cap')
                if module==c.PACKAGE+'.collection.worker' and mode in ('qualification','games'):
                    from .collection.control import check_registry, active_registry_paths
                    paths={path for path in active_registry_paths(folder,mode) if path.exists()}
                    c.require(registries<=paths,'Active decision registry cannot disappear')
                    registries|=paths
                    for path in sorted(registries):
                        check_registry(read_status_json(path),identity,time.time(),time.perf_counter())
                if time.monotonic()-deep_at>=5:
                    deep_at=time.monotonic()
                    c.require(baseline+output_bytes(container)<cap,'Actual evidence cap')
                time.sleep(.25)
            guard(claim,phase)
            c.require(time.monotonic()<deadline and time.time()<wall_deadline,'Actual worker clock at exit')
            c.require(proc.returncode==0,'Worker failure preserved without retry')
            result=c.read(folder/'result.json')
            c.require(result['status']=='passed' and result['mode']==mode,'Completed exact worker role')
            c.require(result['identity']==identity,'Completed exact worker identity')
            for item in [*observed.values(),identity]: require_absent(item)
            status='passed'
    except BaseException: error=traceback.format_exc()
    finally: cleanup_owned(proc,job,identity,observed,cleanup)
    if cleanup: status='failed'
    receipt=c.save(container/'execution.json',dict(time=c.now(),status=status,mode=mode,fit_id=fit_id,
        identity=identity,worker_exit_code=None if proc is None else proc.returncode,
        observed_owned_identities=list(observed.values()),cleanup_errors=cleanup,error=error,
        monotonic_origin=begun,unix_origin=wall,monotonic_end=time.monotonic(),unix_end=time.time(),
        maximum_seconds=seconds,monotonic_deadline=deadline,unix_deadline=wall_deadline,
        peak_rss_bytes=peak,plan_sha256=expected,retries=0))
    c.require(status=='passed','Failed worker preserved: '+str(container))
    return receipt



from .recovery import (source_inputs, validate_test_receipt, validate_reuse, validate_recovery_claim, recover_phase, prepare, freeze, run, main)

if __name__ == "__main__":
    raise SystemExit(main())
