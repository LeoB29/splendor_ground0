"""Exclusive bounded owner for a fixed saved-corpus value-target intervention."""
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

def contract():
    return dict(schema=c.SCHEMA,total_seconds=c.TOTAL_SECONDS,model_seconds=c.MODEL_SECONDS,
        stage_seconds=c.STAGES,independent_reader_seconds=c.INDEPENDENT_REVIEW_SECONDS,
        new_collection_games=0,reused_training_games=1024,reused_validation_games=64,
        reused_test_games=64,reused_assessment_is_diagnostic=True,fits=c.fits(),
        value_q_weight=c.VALUE_Q_WEIGHT,value_target_formula=c.VALUE_TARGET_FORMULA,
        examples_per_fit=57022,updates_per_fit=1782,batch_size=32,last_batch_size=30,
        complete_passes=1,dropped_examples=0,padded_examples=0,
        qualification_optimizer_calls=2,scientific_optimizer_calls=7128,
        replay_optimizer_calls=4,max_optimizer_calls=7134,reused_terminal_fits=0,
        main_games=1200,operational_games=36,matchups=c.MATCHUP_ARMS,
        primary_matchup='mixed_terminal',parent_comparisons_family_ci=.975,
        main_games_per_matchup=400,operational_games_per_matchup=12,
        candidate_repetitions=[0,1],matched_simulations=128,worlds=3,cpuct=.4,depth=64,
        precision='float32',internal_seconds=9.,complete_response_seconds=10.,
        deployment_seconds_descriptive=1.,margin_seconds_descriptive=.7,
        maximum_plies=2000,maximum_model_reproductions=576,maximum_live_lifecycles=2550,
        available_ram_gate=False,resource_forecasts_may_stop=False,
        maximum_evidence=c.MAX_EVIDENCE_BYTES,prefinal_evidence=c.PREFINAL_EVIDENCE_BYTES,
        actual_disk_floor=c.MIN_DISK,fit_offline_rss=c.FIT_RSS,
        strength_inference_rss=c.MAX_RSS_BYTES,automatic_retries=0,replacement_games=0,
        automatic_promotion=False,scheduled_monitor=False,external_and_six_world_deferred=True,
        rentals=0,offline_performance_admission_gate=False,best_seed_selection=False,
        amended_recovered_study=True,original_attempt_status='incomplete',
        authorized_retry_limit=1,retry_spec=c.RETRY_SPEC,
        new_optimizer_calls=0,new_offline_model_evaluations=0,reused_fits=4,
        reused_main_games=212,reused_operational_games=36,remaining_main_games=988,
        unchanged_original_deadlines=True,original_freeze_sha256=c.ORIGINAL_PLAN_SHA)


def source_inputs():
    from tools.splendorust_search512_20261101.preparation import python_dependencies
    from tools.splendorust_independent_data_20261107.collection import private
    package=c.ROOT/'tools/splendorust_full_corpus_recovery_20261111'
    sources=python_dependencies(package.rglob('*.py'))
    from .preflight import test_paths
    tests=test_paths()
    entries=[c.artifact(path) for path in sources+tests]+private.inputs()
    unique={}
    for item in entries:
        key=str(c.resolve(item)).casefold()
        old=unique.setdefault(key,item)
        c.require(old==item,'Consistent static and dynamic runtime source closure')
    return sorted(unique.values(),key=lambda item:item['path'])


def validate_test_receipt(proof):
    c.require(proof.get('schema')=='full_corpus_complete_pytest_v1'
              and proof.get('full_suite') is True, 'Complete source-bound test suite required')
    summary=proof['suite_summary']
    c.require(proof['exit_code']==0 and type(proof['exit_code']) is int
              and summary['tests_passed']==summary['tests_collected']>=100
              and summary['failures']==summary['skipped']==0
              and summary['tensor_runtime_loaded'] is False
              and summary['models']==summary['games']==proof['models']==proof['games']==0,
              'All model-free tests pass without skips or model work')
    c.require(proof['runner']==c.artifact(c.ROOT/'tools/splendorust_full_corpus_recovery_20261111/preflight.py'),
              'Exact current preflight runner')
    return proof


def prepare():
    from .recovery import prepare as implementation
    return implementation()


def freeze(tests,reviews):
    from .recovery import freeze as implementation
    return implementation(tests,reviews)


def child_environment(repetition=None,matchup=None):
    env = os.environ.copy()
    env.pop('KMP_DUPLICATE_LIB_OK', None)
    env.update(SELFPLAY_JOB_SCOPE='pilot', PYTHONDONTWRITEBYTECODE='1',
        PYTHONPATH=str(c.ROOT/'src')+os.pathsep+str(c.ROOT), OMP_NUM_THREADS='1',
        MKL_NUM_THREADS='1', OPENBLAS_NUM_THREADS='1', NUMEXPR_NUM_THREADS='1',
        CUDA_VISIBLE_DEVICES='', RAYON_NUM_THREADS='1')
    env.pop('FULL_CORPUS_MATCHUP',None)
    if matchup is not None:
        c.require(matchup in c.MATCHUPS,'Fixed matchup')
        env['FULL_CORPUS_MATCHUP']=matchup
    if repetition is not None:
        c.require(type(repetition) is int and repetition in (0, 1), 'Fixed repetition')
        env['INDEPENDENT_DATA_REPETITION'] = str(repetition)
    return env


def guard(claim, phase):
    mono, wall = time.monotonic(), time.time()
    c.require(claim['monotonic_origin'] <= mono < phase['monotonic_deadline']
              and claim['unix_origin'] <= wall < phase['unix_deadline'], 'Original phase/global clock expired')
    c.require(shutil.disk_usage(c.OUT).free >= c.MIN_DISK, 'Actual free disk floor')


def begin_phase(claim, name):
    c.require(name in ('qualification','collection','corpus','fit','offline','strength','strength_replay','final'),
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


def segment_id(mode, repetition, block_start=None, matchup='mixed_parent'):
    c.require(matchup in c.MATCHUPS,'Fixed matchup')
    c.require(type(repetition) is int and repetition in (0,1), 'Fixed repetition')
    c.require(mode in ('prepare','resume_partial','qualification','operational','main_block','replay','saved',
                      'independent','aggregate','reader'), 'Fixed strength segment')
    if mode == 'main_block':
        c.require(type(block_start) is int and block_start in range(0,200,10), 'Fixed ten-game block')
    else:
        c.require(block_start is None, 'Only main blocks have an offset')
    return f'{mode}_{matchup}_r{repetition}' + (f'_g{block_start:03d}' if block_start is not None else '')


def validate_segment_claim(claim, folder, mode, repetition, block_start, expected, matchup='mixed_parent'):
    ident = segment_id(mode,repetition,block_start,matchup)
    c.require(type(claim) is dict and set(claim) == {'segment_id','mode','repetition','block_start',
        'claim','phase_clock','freeze_sha256','reader_clock','matchup'}, 'Exact strength segment claim fields')
    fields = dict(segment_id=ident,mode=mode,repetition=repetition,block_start=block_start,freeze_sha256=expected,matchup=matchup)
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
    return 212+sum(1 for matchup in c.MATCHUPS for rep in (0,1) for _ in
        (c.OUT/'strength'/matchup/f'r{rep}'/'games/main').glob('g*/result.json'))


def run_segment(claim, phase_descriptor, expected, mode, repetition, block_start=None, reader_clock=None, matchup='mixed_parent'):
    from tools.local_player_qualification_128.clock import _WindowsJob
    ident = segment_id(mode,repetition,block_start,matchup)
    folder = c.OUT/'segments'/ident; folder.mkdir(parents=True,exist_ok=False)
    local = dict(segment_id=ident,mode=mode,repetition=repetition,block_start=block_start,matchup=matchup,
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
            proc = subprocess.Popen(args,cwd=c.ROOT,env=child_environment(repetition,matchup),stdout=stdout,stderr=stderr,
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
            c.require(result['status']=='passed' and result['mode']==mode and result['repetition']==repetition and result['matchup']==matchup,
                      'Completed exact strength segment')
            c.require(result['pid']==identity['pid'] and result['created']==identity['created'],'Actual result owner')
            for item in [*observed.values(),identity]: require_absent(item)
            status='passed'
    except BaseException: error=traceback.format_exc()
    finally: cleanup_owned(proc,job,identity,observed,cleanup)
    if cleanup: status='failed'
    receipt=c.save(folder/'execution.json',dict(status=status,mode=mode,repetition=repetition,matchup=matchup,
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


def run(expected):
    c.require(c.sha(c.PLAN)==expected,'Exact immutable execution freeze')
    plan=c.read(c.PLAN)
    c.require(c.canonical({key:plan.get(key) for key in contract()})==c.canonical(contract()),
              'Exact frozen execution contract')
    c.verify(plan['inputs'])
    from tools.future_training_rules_20261021 import verify_runtime_sources
    c.require(verify_runtime_sources()==plan['corrected_source_receipt'],'Actual root startup sources')
    from tools.splendorust_search256_recovery_20261030_02 import placement
    placed=placement.configure_execution(plan,'isolated')
    from . import recovery
    from .boundary import check_source_boundary
    recovery.validate_reuse(c.read(c.OUT/'reuse.json'))
    check_source_boundary(boundary=recovery.dual_deadline,deep=False)
    old=c.read(c.ORIGINAL_OUT/'claim.json')
    claim=dict(time=c.now(),pid=os.getpid(),created=psutil.Process().create_time(),
        monotonic_origin=old['monotonic_origin'],unix_origin=old['unix_origin'],freeze=c.artifact(c.PLAN),
        plan_sha256=expected,total_seconds=old['total_seconds'],model_seconds=old['model_seconds'],
        placement=placed,automatic_retries=0,authorized_retry_limit=1,retry_spec=c.RETRY_SPEC,
        amended_recovered_study=True,original_attempt_status='incomplete',
        original_claim=c.artifact(c.ORIGINAL_OUT/'claim.json'),
        reuse=c.artifact(c.OUT/'reuse.json'),authorization=c.artifact(c.AUTHORIZATION))
    recovery.validate_recovery_claim(claim)
    c.save(c.OUT/'claim.json',claim)
    receipts,error,status=[],None,'failed'
    try:
        from . import recovery
        fit_module=c.PACKAGE+'.fit'
        phase=recovery.recover_phase(claim)
        for mode in ('prepare','qualification'):
            for matchup in c.MATCHUPS:
                for rep in (0,1):receipts.append(run_segment(claim,phase,expected,mode,rep,matchup=matchup))
            c.save(c.OUT/f'strength_{mode}_gate.json',dict(status='passed',both_repetitions=True,
                all_matchups=list(c.MATCHUPS),phase=phase,actual_exits=receipts[-6:]))
        receipts.append(run_segment(claim,phase,expected,'resume_partial',1,matchup='terminal_parent'))
        for matchup,rep,block in recovery.remaining_blocks():
            receipts.append(run_segment(claim,phase,expected,'main_block',rep,block,matchup=matchup))
        end_phase(claim,phase)
        phase=begin_phase(claim,'strength_replay')
        for matchup in c.MATCHUPS:
            for rep in (0,1):receipts.append(run_segment(claim,phase,expected,'replay',rep,matchup=matchup))
        end_phase(claim,phase)
        phase=begin_phase(claim,'final')
        for matchup in c.MATCHUPS:
            for rep in (0,1):receipts.append(run_segment(claim,phase,expected,'saved',rep,matchup=matchup))
            receipts.append(run_segment(claim,phase,expected,'aggregate',0,matchup=matchup))
        receipts.append(run_job(claim,phase,expected,fit_module,'audit',
            c.OUT/'training_audit',seconds=12*3600,rss=c.COLLECTION_RSS,model=False))
        receipts.append(run_job(claim,phase,expected,fit_module,'analysis',
            c.OUT/'offline_analysis',seconds=12*3600,rss=c.FIT_RSS,model=False))
        final=c.read(c.resolve(phase));mono,wall=time.monotonic(),time.time()
        reader=c.save(c.OUT/'reader_clock.json',dict(monotonic=mono,unix_time=wall,
            cap_seconds=c.INDEPENDENT_REVIEW_SECONDS,
            monotonic_deadline=min(mono+c.INDEPENDENT_REVIEW_SECONDS,final['monotonic_deadline']),
            unix_deadline=min(wall+c.INDEPENDENT_REVIEW_SECONDS,final['unix_deadline']),
            original_phase=phase,claim=c.artifact(c.OUT/'claim.json')))
        for matchup in c.MATCHUPS:
            for rep in (0,1):receipts.append(run_segment(claim,phase,expected,'independent',rep,reader_clock=reader,matchup=matchup))
            receipts.append(run_segment(claim,phase,expected,'reader',0,reader_clock=reader,matchup=matchup))
        receipts.append(run_job(claim,phase,expected,c.PACKAGE+'.audit','final',
            c.OUT/'final_audit',seconds=12*3600,rss=c.COLLECTION_RSS,model=False,
            cell_deadline=c.read(c.resolve(reader))))
        c.verify(plan['inputs']);end_phase(claim,phase)
        status='verified_pending_actual_owner_exit'
    except BaseException: error=traceback.format_exc()
    execution=c.save(c.OUT/'execution.json',dict(status=status,freeze_sha256=expected,segments=receipts,
        error=error,original_claim=c.artifact(c.OUT/'claim.json'),ended_monotonic=time.monotonic(),
        ended_unix=time.time(),automatic_promotion=False,amended_recovered_study=True,original_attempt_status='incomplete',
        original_failed_execution=c.artifact(c.ORIGINAL_OUT/'execution.json'),authorization=c.artifact(c.AUTHORIZATION)))
    c.status(c.OUT/'stage.json',dict(time=c.now(),stage=status,status=status,
        completed_stages=len(receipts),completed_main_games=completed_games()))
    if error:
        c.save(c.OUT/'failure.json',dict(status='incomplete',execution=execution,error=error));return 1
    c.save(c.OUT/'completion.json',dict(status=status,execution=execution,
        independent_reader=c.artifact(c.OUT/'final_audit/worker/result.json'),
        owner_exit_still_requires_observation=True,no_automatic_followup=True,
        amended_recovered_study=True,original_attempt_status='incomplete',authorized_retries=1))
    return 0


def main():
    parser=argparse.ArgumentParser()
    parser.add_argument('operation',choices=('prepare','freeze','run'))
    parser.add_argument('--tests');parser.add_argument('--reviews',nargs='*');parser.add_argument('--freeze-sha256')
    args=parser.parse_args()
    if args.operation=='prepare':print(c.canonical(prepare()));return 0
    if args.operation=='freeze':print(c.canonical(freeze(Path(args.tests),[Path(x) for x in args.reviews])));return 0
    return run(args.freeze_sha256)


if __name__=='__main__':raise SystemExit(main())
