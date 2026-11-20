"""Recover one interrupted stage without changing the original scientific code."""
from __future__ import annotations

import argparse
from contextlib import contextmanager
from functools import partial
import multiprocessing
import os
from pathlib import Path
import re
import subprocess
import sys
import time
import traceback

from tools.structured_proposer_128 import experiment as e, common as c

ROOT, BASE = c.ROOT, c.OUTPUT
OUTPUT = ROOT / 'outputs/structured_proposer_128_recovery_01'
RUNNER = ROOT / 'run_structured_proposer_128_recovery_01.py'
BINDING = '57cd1786992c123e2e8fdedf6e4cf567196588efc8bd1ab9d2bbb24b7908f454'
TARGETS = ['tools/structured_proposer_128_recovery_01/test_recovery.py',
           'tools/structured_proposer_128/test_learning.py::test_interrupted_resume_is_bitwise_equal_every_checkpoint',
           'tools/structured_proposer_128/test_experiment.py::test_current_boundary_accepts_24_hours_and_enforces_96',
           'tools/structured_proposer_128/test_experiment.py::test_cooperative_abort_and_disk_reserve_remain_enforced']


def own_identity():
    paths = sorted(Path(__file__).parent.glob('*.py')) + [RUNNER, Path(__file__).parent / 'PROTOCOL.md']
    return {'files': [c.artifact(p) for p in paths], 'targets': TARGETS,
            'original_binding': BINDING, 'python': sys.version, 'optimized': sys.flags.optimize}


def jobs():
    return [(fit, arm) for fit in c.FITS for arm in c.LEARNING_ARMS]


def upstream(base=BASE):
    return [Path(base)/name for name in ('freeze.json','data_lock.json','training_feature_audit.json')]


def authenticate_stage_launch(arguments, binding, dependencies, base=BASE):
    base = Path(base)
    launch = c.old.read_json(base/'learning_fits_launch.json')
    c.require(launch['manifest'] == binding and launch['stage'] == 'learning' and launch['kind'] == 'fits'
              and launch['expected_indices'] == list(range(len(arguments)))
              and launch['arguments_sha256'] == c.old.digest(arguments)
              and launch['dependencies'] == [c.artifact(path) for path in dependencies],
              'Original fit-stage arguments/dependencies changed')
    for ref in launch['dependencies']: c.verify_hash(ref)
    return launch


def completed_jobs(arguments, binding, base=BASE, expected_indices=None, settings=None, initializers=None):
    """Authenticate bytes/metadata only; never deserialize model/AdamW payloads."""
    base = Path(base)
    settings = e.learning.CONFIG if settings is None else settings
    folder = base/'completed_jobs/learning_fits'
    paths = sorted(folder.glob('*.json'))
    indices = [int(path.stem) for path in paths]
    if expected_indices is not None:
        c.require(indices == list(expected_indices), 'Completed fit receipt inventory differs')
    c.require(len(indices) == len(set(indices)) and all(0 <= i < len(arguments) for i in indices), 'Unknown/duplicate fit index')
    files, evidence = [], []
    for path, index in zip(paths, indices, strict=True):
        fit, arm = arguments[index]
        receipt = c.old.read_json(path)
        ref = receipt['artifact']
        expected_path = base/'fits'/f'fit_{fit}'/arm/'result.json'
        c.require(receipt['manifest'] == binding and receipt['stage'] == 'learning' and receipt['kind'] == 'fits'
                  and receipt['argument_sha256'] == c.old.digest(arguments[index])
                  and (ref['fit'], ref['arm'], ref['index']) == (fit, arm, index)
                  and c.verify_hash(ref).resolve() == expected_path.resolve(), 'Completed job provenance differs')
        result = c.verified(ref)
        initial = e.training_source.initializer(fit, base) if initializers is None else initializers[fit]
        identity = result['identity']
        c.require(result['manifest'] == identity['manifest'] == binding
                  and result['fit'] == identity['fit'] == fit and result['arm'] == identity['arm'] == arm
                  and result['config'] == identity['config'] == settings
                  and result['initializer'] == identity['initializer'] == initial
                  and identity['model_config'] == e.model.MODEL_CONFIG
                  and result['batch_receipt'] == identity['batch_receipt']
                  and result['complete'] is True and result['selected_update'] == settings['updates']
                  and result['assessment_read'] is result['validation_read'] is False,
                  'Completed fit identity/configuration differs')
        c.require([point['update'] for point in result['checkpoints']] == settings['diagnostic_updates']
                  and [row['update'] for row in result['optimizer_checkpoints']] == e.learning._progress_updates(settings)
                  and result['selected_model'] == result['checkpoints'][-1]['checkpoint']
                  and result['selected_training'] == result['checkpoints'][-1]['training'], 'Completed checkpoint inventory differs')
        evidence += [c.artifact(path), ref]
        for point in result['checkpoints']:
            c.require(point['identity_sha256'] == c.old.digest(identity), 'Checkpoint identity differs')
            payload = c.verified(point['receipt'])
            c.require(payload == {k:v for k,v in point.items() if k != 'receipt'}, 'Point receipt differs')
            for key in ('checkpoint','training','receipt'):
                c.verify_hash(point[key]); evidence.append(point[key])
        for item in result['optimizer_checkpoints']:
            c.verify_hash(item); evidence.append(item)
        files.append(ref)
    return files, evidence


def stopped_snapshot():
    c.require(e.validate_freeze(full_sources=True) == BINDING, 'Original frozen experiment changed')
    c.budget_boundary(BASE)
    for name in ('active_run.json','learning_fits_lock.json','learning_fit_audits_launch.json',
                 'learning_complete.json','models_lock.json','final_status.json','abort.json','failure.json','operational_hold.json'):
        c.require(not (BASE/name).exists(), 'Unexpected stage/hold; this recovery is specific: '+name)
    arguments = jobs()
    authenticate_stage_launch(arguments, BINDING, upstream())
    e.training_source.validate_features(BINDING, BASE)
    files, evidence = completed_jobs(arguments, BINDING, expected_indices=range(5))
    fit, arm = arguments[5]
    folder = BASE/'fits'/f'fit_{fit}'/arm
    c.require(not (folder/'result.json').exists(), 'Last fit already completed without expected receipt')
    pointer = c.old.read_json(folder/'resume.json')
    c.require(pointer['update'] == 9216 and c.verify_hash(pointer['checkpoint']).resolve() == (folder/'optimizer_09216.pt').resolve(),
              'Expected interrupted9216 checkpoint differs')
    partial_points = []
    for path in sorted(folder.glob('point_*.json')):
        point = c.old.read_json(path)
        c.require(point['identity_sha256'] == pointer['identity_sha256'] and point['update'] in (0,7136), 'Partial identity differs')
        partial_points += [c.artifact(path), point['checkpoint'], point['training']]
        c.verify_hash(point['checkpoint']); c.verify_hash(point['training'])
    c.require(len(partial_points) == 6, 'Missing partial diagnostic receipts')
    evidence += [c.artifact(BASE/name) for name in ('freeze.json','launch.json','launch_seal.json',
                  'runtime/process.json','learning_fits_launch.json','data_lock.json','training_feature_audit.json')]
    return {'original_binding': BINDING, 'arguments': arguments, 'completed': files, 'evidence': evidence,
            'partial': {'job_index':5, 'argument':arguments[5], 'resume_pointer':pointer,
                        'pointer_at_freeze':c.artifact(folder/'resume.json'), 'point_evidence':partial_points},
            'downtime_charged':True, 'timing_origin':c.artifact(BASE/'launch_seal.json')}


def ensure_no_live_original(ignore=()):
    """Read-only detection; never signal an original or unrelated process."""
    import psutil
    original_pid = c.old.read_json(BASE/'launch.json')['pid']
    known_workers = {int(p.stem.split('_')[0]) for p in (BASE/'workers').glob('*/*_health.json')}
    for process in psutil.process_iter(['pid','name']):
        if process.pid in {os.getpid(), *ignore} or not (process.info['name'] or '').lower().startswith('python'):
            continue
        try: command = process.cmdline()
        except psutil.NoSuchProcess: continue
        text = ' '.join(command)
        parent_match = re.search(r'parent_pid\s*=\s*(\d+)', text)
        own_run = any(Path(arg).name in ('run_structured_proposer_128.py',RUNNER.name) for arg in command) and 'run' in command
        original_worker = bool(parent_match and int(parent_match.group(1)) == original_pid)
        recorded_worker = process.pid in known_workers and 'multiprocessing' in text
        c.require(not (own_run or original_worker or recorded_worker), 'Original/recovery supervisor or worker still live: '+str(process.pid))


def check(folder):
    folder = Path(folder)
    folder.mkdir(parents=True, exist_ok=False)
    before, started = own_identity(), time.perf_counter()
    args = ['-q', *TARGETS, '-p','no:cacheprovider','--basetemp',str(folder/'fixtures')]
    code = 'import torch,pytest;torch.set_num_threads(1);raise SystemExit(pytest.main('+repr(args)+'))'
    with (folder/'stdout.txt').open('x',encoding='utf-8') as out, (folder/'stderr.txt').open('x',encoding='utf-8') as err:
        result = subprocess.run([sys.executable,'-u','-c',code],cwd=ROOT,stdout=out,stderr=err,stdin=subprocess.DEVNULL,
                   timeout=300,creationflags=subprocess.CREATE_NO_WINDOW if os.name == 'nt' else 0)
    c.require(before == own_identity(), 'Recovery code changed during preflight')
    c.old.save_new(folder/'checks.json',{'identity':before,'exit_code':result.returncode,'seconds':time.perf_counter()-started,
                   'logs':[c.artifact(folder/name) for name in ('stdout.txt','stderr.txt')]})
    return result.returncode


def prepare(checks_path):
    ensure_no_live_original()
    checks = c.old.read_json(checks_path)
    c.require(checks['identity'] == own_identity() and checks['exit_code'] == 0, 'Current passing recovery checks required')
    for ref in checks['logs']: c.verify_hash(ref)
    with c.old.process_lock(BASE/'run.lock'), c.old.process_lock(OUTPUT/'launch.lock'):
        snapshot = stopped_snapshot()
        manifest = {'identity':own_identity(),'original_snapshot':snapshot,'checks':c.artifact(checks_path),
                    'owner_authorized':True,'automatic_retry':False,'time':c.old.now()}
        c.old.save_new(OUTPUT/'manifest.json',manifest)
        c.old.save_new(OUTPUT/'freeze.json',{'manifest':c.artifact(OUTPUT/'manifest.json'),'binding':c.old.digest(manifest)})
    return {'prepared':True,'binding':c.old.digest(manifest),'reuse_fits':5,'resume_update':9216}


def validate_recovery(stopped=True):
    freeze = c.old.read_json(OUTPUT/'freeze.json')
    manifest = c.verified(freeze['manifest'])
    c.require(c.old.digest(manifest) == freeze['binding'] and manifest['identity'] == own_identity()
              and manifest['owner_authorized'] is True, 'Recovery freeze changed')
    checks = c.verified(manifest['checks'])
    c.require(checks['identity'] == manifest['identity'] and checks['exit_code'] == 0, 'Recovery checks changed')
    for ref in checks['logs']: c.verify_hash(ref)
    if stopped:
        c.require(c.old.digest(stopped_snapshot()) == c.old.digest(manifest['original_snapshot']),
                  'Original stopped state changed after recovery freeze')
    return freeze['binding'], manifest


def recovered_fit_pool(stage, kind, arguments, binding, dependencies, manifest, original_pool):
    if (stage,kind) != ('learning','fits'):
        return original_pool(stage,kind,arguments,binding,dependencies)
    c.require(binding == BINDING and arguments == jobs(), 'Recovery can only complete the prescribed original six fits')
    authenticate_stage_launch(arguments,binding,dependencies,base=BASE)
    c.budget_boundary(BASE)
    snapshot = manifest['original_snapshot']
    reused = snapshot['completed']
    c.require([ref['index'] for ref in reused] == list(range(5)), 'Recovery completed index inventory changed')
    for ref in snapshot['evidence']: c.verify_hash(ref)
    c.verify_hash(snapshot['partial']['pointer_at_freeze'])
    c.verify_hash(snapshot['partial']['resume_pointer']['checkpoint'])
    started = time.perf_counter()
    runtime = c.windows.configure_spawn()
    e.event('learning_fits',completed=5,total=6,recovery=c.artifact(OUTPUT/'freeze.json'))
    with e.ProcessPoolExecutor(max_workers=c.CONFIG['fit_workers'],mp_context=multiprocessing.get_context('spawn'),
                              initializer=e.worker_initialize,initargs=('learning','fits',binding,BASE)) as pool:
        files = e.dispatch(pool,[arguments[5]],partial(e.run_job,'learning','fits',binding=binding,base=BASE),
                           c.CONFIG['fit_workers'],lambda n:e.event('learning_fits',completed=5+n,total=6),lambda:c.budget_boundary(BASE))
    c.require(len(files) == 1 and files[0]['index'] == 5, 'Recovered job index differs')
    all_files = sorted([*reused,*files],key=lambda ref:ref['index'])
    c.require([ref['index'] for ref in all_files] == list(range(6)), 'Missing/duplicate recovered fits')
    for ref in all_files: c.verify_hash(ref)
    new_receipt = c.old.read_json(BASE/'completed_jobs/learning_fits/000005.json')
    c.require(new_receipt['artifact'] == files[0] and new_receipt['manifest'] == binding
              and new_receipt['argument_sha256'] == c.old.digest(arguments[5]), 'Recovered job completion provenance differs')
    c.old.save_new(BASE/'learning_fits_lock.json',{'manifest':binding,'launch':c.artifact(BASE/'learning_fits_launch.json'),
                  'files':all_files,'elapsed_seconds':time.perf_counter()-started,'runtime':runtime,
                  'recovery':c.artifact(OUTPUT/'freeze.json'),'reused_indices':list(range(5)),
                  'resumed_from':snapshot['partial']['resume_pointer'],'elapsed_scope':'recovery stage only; original96h clock unchanged'})
    return all_files


@contextmanager
def recovery_adapters(manifest):
    original_pool, original_features = e.run_pool, e.training_source.audit_features
    def features(binding,base=None,**kwargs):
        c.require(not kwargs and Path(BASE if base is None else base).resolve() == BASE.resolve() and binding == BINDING,
                  'Unexpected feature reuse request')
        authenticate_stage_launch(jobs(),binding,upstream(BASE),base=BASE)
        e.training_source.validate_features(binding,BASE)
        return c.artifact(BASE/'training_feature_audit.json')
    e.run_pool = lambda stage,kind,args,binding,dependencies: recovered_fit_pool(stage,kind,args,binding,dependencies,manifest,original_pool)
    e.training_source.audit_features = features
    try: yield
    finally:
        e.run_pool, e.training_source.audit_features = original_pool, original_features


def continue_pipeline(binding):
    if not e.build_learning(binding): return
    c.budget_boundary(BASE); e.build_sources(binding)
    c.budget_boundary(BASE); e.build_stage('probe',binding)
    if not e.operational_decision(binding): return
    e.build_stage('main',binding); c.budget_boundary(BASE); e.finish(binding)


def run():
    runtime = OUTPUT/'runtime'; runtime.mkdir(parents=True,exist_ok=True)
    c.windows.assert_windowless_worker(runtime/'stdout.txt',runtime/'stderr.txt')
    with c.old.process_lock(BASE/'run.lock'), c.old.process_lock(OUTPUT/'run.lock'):
        c.require(not (OUTPUT/'launch.json').exists(), 'One-shot recovery; preserve existing evidence')
        ensure_no_live_original()
        recovery_binding, manifest = validate_recovery()
        e.torch.set_num_threads(1); e.torch.use_deterministic_algorithms(True)
        try:
            with e.parent.parent.inhibit_system_sleep() as sleep:
                c.old.save_new(OUTPUT/'launch.json',{'binding':recovery_binding,'manifest':BINDING,'time':c.old.now(),
                              'pid':os.getpid(),'sleep_inhibition':sleep,'original_clock':c.artifact(BASE/'launch_seal.json')})
                c.old.save_new(OUTPUT/'launch_seal.json',{'binding':recovery_binding,'receipt':c.artifact(OUTPUT/'launch.json')})
                c.budget_boundary(BASE)
                with recovery_adapters(manifest): continue_pipeline(BINDING)
                c.old.save_new(OUTPUT/'finished.json',{'binding':recovery_binding,'time':c.old.now(),'pipeline_returned':True})
        except c.BudgetExceeded as error:
            c.old.save_new(OUTPUT/'operational_hold.json',{'binding':recovery_binding,'time':c.old.now(),'reason':str(error)})
            e.event('operational_hold',reason=str(error),recovery=True)
        except BaseException:
            e.abort(RuntimeError('Recovery failed; inspect recovery logs'),BASE)
            c.old.save_new(OUTPUT/'failure.json',{'binding':recovery_binding,'time':c.old.now(),'traceback':traceback.format_exc()})
            raise


def launch():
    with c.old.process_lock(OUTPUT/'launch.lock'):
        c.require(not (OUTPUT/'runtime/process.json').exists() and not (OUTPUT/'launch.json').exists(), 'Recovery already launched')
        ensure_no_live_original()
        with c.old.process_lock(BASE/'run.lock'):
            recovery_binding,_ = validate_recovery()
        runtime=OUTPUT/'runtime';runtime.mkdir(parents=True,exist_ok=True)
        process=c.windows.detached_launch([sys.executable,'-u',RUNNER,'run'],runtime/'stdout.txt',runtime/'stderr.txt',ROOT)
        c.old.save_new(runtime/'process.json',{**process,'binding':recovery_binding,'manifest':BINDING,'time':c.old.now()})
        deadline=time.monotonic()+600
        import psutil
        while not (OUTPUT/'launch_seal.json').exists():
            c.require(psutil.pid_exists(process['pid']) and time.monotonic()<deadline,'No recovery acknowledgement; inspect before any retry')
            time.sleep(.1)
        receipt=c.verified(c.old.read_json(OUTPUT/'launch_seal.json')['receipt'])
        c.require(receipt['pid']==process['pid'] and receipt['binding']==recovery_binding,'Recovery supervisor identity differs')
        c.old.save_new(runtime/'acknowledged.json',{'pid':process['pid'],'time':c.old.now(),'launch':c.artifact(OUTPUT/'launch.json')})
        return {'launched':True,'pid':process['pid'],'binding':recovery_binding,'resume_update':9216}


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('command',choices=('check','prepare','launch','run'))
    parser.add_argument('--checks',type=Path)
    args=parser.parse_args()
    if args.command=='check':
        c.require(args.checks is not None,'Supply new check folder');return check(args.checks)
    if args.command=='prepare':
        c.require(args.checks is not None,'Supply checks.json');result=prepare(args.checks)
    elif args.command=='launch':result=launch()
    else:run();return 0
    print(result,flush=True)
    return 0
