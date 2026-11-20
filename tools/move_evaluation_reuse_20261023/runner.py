"""Bounded exact-behavior preparation reuse, checked against frozen decisions."""
import argparse
import ctypes
import math
import os
from pathlib import Path
import shutil
import sys
import time
import traceback

import psutil

from tools.local_player_qualification_128.clock import processor_topology
from . import common as c

SCIENTIFIC_POLICY_ID = 'entity_onehot_greedy_canonical_atomic_prereveal_cpu_20261023_v1'


def clamp_timeout(limit, deadline, *, clock=time.monotonic):
    remaining = deadline-clock()
    c.require(remaining > 0, "Diagnostic stage deadline reached")
    return min(limit, remaining)


def timing_runtime():
    c.require(hasattr(sys, 'monitoring') and os.name == 'nt', 'Qualified CPython monitoring runtime required')
    buffer = ctypes.create_unicode_buffer(32768)
    size = ctypes.windll.kernel32.GetModuleFileNameW(ctypes.c_void_p(ctypes.pythonapi._handle), buffer, len(buffer))
    c.require(0 < size < len(buffer), 'Cannot locate interpreter DLL for built-in monitoring')
    path = Path(buffer.value).resolve()
    return [{'absolute_path':str(path),'bytes':path.stat().st_size,'sha256':c.file_hash(path),
             'module':'sys.monitoring','implementation':'built-in CPython interpreter'}]


def verify_runtime(freeze):
    c.verify(freeze['inputs'])
    c.require(timing_runtime() == freeze['timing_runtime'], 'Monitoring runtime changed')
    c.require(sys.version == freeze['python'] and c.file_hash(sys.executable) == freeze['python_executable_sha256'],
              'Python executable/version changed')


def inherit_closed_evidence(closeout, inputs):
    """Authenticate completed timing evidence and its closed report, not live handoffs."""
    run = closeout['run_inventory']
    admin = [closeout['inspection_source'],closeout['report']]
    c.require(type(run) is list and type(admin) is list and
              closeout['run_files'] == len(run) and
              closeout['run_bytes'] == sum(item['bytes'] for item in run), 'Prior closeout inventory totals')
    status_dir = c.ROOT/'outputs/move_evaluation_timing_20261023_status_01'
    seen = set()
    for items, boundary in ((run,c.PREVIOUS),([admin[0]],status_dir),([admin[1]],c.ROOT/'docs/engineering')):
        for item in items:
            path = (c.ROOT/item['path']).resolve()
            c.require(path.is_relative_to(boundary.resolve()) and item['path'] not in seen,
                      'Prior closeout evidence must be unique and confined')
            c.require(c.artifact(path) == item, 'Changed prior closeout evidence: '+item['path'])
            c.require(item['path'] not in inputs or inputs[item['path']] == item, 'Inherited input collision')
            inputs[item['path']] = item
            seen.add(item['path'])
    c.require({(c.ROOT/item['path']).resolve() for item in run} ==
              {p.resolve() for p in c.PREVIOUS.rglob('*') if p.is_file()}, 'Prior closed-run inventory changed')


def prepare():
    c.require(not (c.OUT/'freeze.json').exists(), 'Reuse freeze already exists')
    c.require(c.file_hash(c.PREVIOUS/'freeze.json') == c.PREVIOUS_FREEZE_SHA, 'Previous diagnostic freeze changed')
    previous = c.read(c.PREVIOUS/'freeze.json')
    c.verify(previous['inputs'])
    c.require(len(previous['inputs']) == 2133, 'Exact prior frozen dependency inventory')
    status_dir = c.ROOT/'outputs/move_evaluation_timing_20261023_status_01'
    closeout = c.read(status_dir/'closeout.json')
    c.require(closeout['schema'] == 'move_evaluation_timing_administrative_closeout_v1' and
              closeout['status'] == 'closed_completed_diagnosis' and closeout['prior_failures_preserved'] is True and
              closeout['original_120_second_qualification'] == 'failed_unchanged' and
              closeout['freeze'] == c.artifact(c.PREVIOUS/'freeze.json') and
              closeout['frozen_inputs_reauthenticated'] == 2133 and
              closeout['execution']['status'] == 'completed' and closeout['execution']['exit_code'] == 0 and
              closeout['execution']['cleanup_errors'] == [] and
              [row['index'] for row in closeout['roots']] == list(c.ROOT_INDICES), 'Completed timing reference not authenticated')
    inputs = {a['path']:a for a in previous['inputs']}
    inherit_closed_evidence(closeout,inputs)
    c.require(c.read(c.OLD/'failure.json')['status'] == 'held_failure_preserved' and
              c.read(c.OLD/'execution.json')['status'] == 'runner_failed', 'Original failure/closure missing')
    c.require(shutil.disk_usage(c.ROOT).free >= 6*c.GiB, 'Initial disk reserve')
    topology = processor_topology()
    c.require(topology['selected_logical_processor'] == 0, 'CPU0 unavailable')
    lock = c.read(c.PREP/'candidate_lock.json')
    roots = [lock['roots'][i] for i in c.ROOT_INDICES]
    c.require([(r['game'],r['ply'],r['fit']) for r in roots] ==
              [(25,8,161001),(57,14,161002),(58,4,161003)], 'Predetermined root identities')
    c.require(all(len(r['candidates']['actions']) == 4 for r in roots), 'Fixed four candidates')
    references = []
    for index in c.ROOT_INDICES:
        reference = c.read(c.PREVIOUS/'measurements'/f'r{index:02d}_plain.json')['result']['scientific']
        c.verify([c.static_artifact(reference['trace'])])
        c.require(c.read(c.PREVIOUS/'measurements'/f'r{index:02d}_audit.json')['result']['scientific']['status'] == 'passed',
                  'Previous complete root independent replay missing')
        references.append(c.artifact(c.PREVIOUS/'measurements'/f'r{index:02d}_plain.json'))
    paths = set((c.ROOT/'tools/move_evaluation_reuse_20261023').glob('*.py'))
    paths.update(c.ROOT.glob('tests/test_move_evaluation_reuse_*_20261023.py'))
    paths.update(c.OUT.glob('prelaunch_*.json'))
    paths.update([c.PROTOCOL, c.PREVIOUS/'freeze.json', c.PREVIOUS/'execution.json',
        status_dir/'closeout.json',
        c.OLD/'freeze.json', c.OLD/'entrypoint_qualification.json',
        c.OLD/'workers/selection/r00/continuations.jsonl.gz', c.OLD/'failure.json', c.OLD/'execution.json',
        c.OUT/'prelaunch_tests.json'])
    tests = c.read(c.OUT/'prelaunch_tests.json')
    c.require(tests['status'] == 'passed' and tests['exit_code'] == 0, 'Prelaunch tests missing')
    for path in paths:
        item = c.artifact(path)
        c.require(item['path'] not in inputs or inputs[item['path']] == item, 'Inherited input changed')
        inputs[item['path']] = item
    freeze = {'schema':'fixed_evaluator_exact_reuse_v1','time':c.now(),
        'authorization':'Owner approved exact-behavior observer/legal preparation reuse and bounded fixed-root validation.',
        'inputs':sorted(inputs.values(),key=lambda a:a['path']),'timing_runtime':timing_runtime(),
        'python':sys.version,'python_executable_sha256':c.file_hash(sys.executable),
        'topology':topology,'native_binary':previous['native_binary'],
        'previous_freeze':c.artifact(c.PREVIOUS/'freeze.json'),'previous_roots':references,
        'root_indices':list(c.ROOT_INDICES),'roots':[{k:r[k] for k in ('id','game','ply','fit')} for r in roots],
        'job_seconds':c.JOB_SECONDS,'final_reserve_seconds':c.FINAL_RESERVE,
        'implementation_id':c.IMPLEMENTATION_ID,'scientific_policy_id':SCIENTIFIC_POLICY_ID,
        'request_seconds':c.REQUEST_SECONDS,'fixture_seconds':300,'initialization_seconds':60,
        'audit_reserve_seconds_per_root':660,'original_qualification_seconds':120,
        'original_failure_unchanged':True,'previous_instrumentation_failure_unchanged':True,
        'max_evidence_bytes':c.GiB,'max_tree_rss_bytes':2*c.GiB,
        'automatic_retries':0,'larger_data_deferred':True}
    c.save(c.OUT/'freeze.json',freeze)
    print(c.canonical({'freeze_sha256':c.file_hash(c.OUT/'freeze.json'),'inputs':len(inputs),'model_calls':0}),flush=True)


def latency_observation(reference_seconds,current_seconds):
    c.require(all(type(value) in (int,float) and math.isfinite(value) and value > 0
                  for value in (reference_seconds,current_seconds)), 'Finite positive completed-response times')
    return {'reference_complete_response_seconds':reference_seconds,'current_complete_response_seconds':current_seconds,
        'current_over_reference_ratio':current_seconds/reference_seconds,
        'observed_reduction_fraction':1-current_seconds/reference_seconds,
        'ordinary_within_120_seconds':current_seconds <= 120,
        'scope':'Historical reference comparison; host conditions are not paired or controlled.',
        'causal_speedup_claim':False,'full_18_position_qualification':False}


def implementation_metadata(mode):
    c.require(mode in ('qualify_plain','qualify_timed','plain','timed','audit'), 'Explicit reuse worker mode')
    return {'schema':'per_decision_reuse_implementation_v1','optimization_id':c.IMPLEMENTATION_ID,
        'active_opponent':'original_reference' if mode == 'audit' else 'per_decision_reuse',
        'scientific_policy_id':SCIENTIFIC_POLICY_ID,'scientific_policy_unchanged':True,'cross_decision_cache':False}


def check_implementation(value,mode):
    c.require(c.canonical(value) == c.canonical(implementation_metadata(mode)),
              'Declared implementation/reference policy mismatch')


class Envelope:
    def __init__(self, launch):
        self.start = launch['monotonic']
        self.wall_start = launch['unix_time']
        self.end = self.start+c.JOB_SECONDS
        self.work_end = self.end-c.FINAL_RESERVE
        self.waited = 0.
        self.admitted = False
        self.admissions = 0

    def guard(self, deadline, *, memory=True):
        cutoff = min(deadline,self.end)
        c.require(time.monotonic() < cutoff, 'Diagnostic stage/global deadline')
        c.require(time.time() < self.wall_start+(cutoff-self.start), 'Diagnostic wall stage/global deadline')
        c.require(shutil.disk_usage(c.ROOT).free >= 5*c.GiB, 'Free disk reserve')
        used = sum(p.stat().st_size for p in c.OUT.rglob('*') if p.is_file())
        c.require(used < c.GiB, 'New evidence size cap')
        if memory:
            c.require(psutil.virtual_memory().available >= 2*c.GiB, 'RAM shortage during active request')

    def admit(self, deadline):
        self.guard(deadline, memory=False)
        start = time.monotonic()
        available = psutil.virtual_memory().available
        stable = start if available >= 2*c.GiB else None
        fast = self.admitted and stable is not None
        minimum = available
        while not fast:
            self.guard(deadline, memory=False)
            current = time.monotonic()
            available = psutil.virtual_memory().available
            minimum = min(minimum, available)
            if available < 2*c.GiB:
                stable = None
            elif stable is None:
                stable = current
            c.require(self.waited+current-start < 300, 'Cumulative memory admission allowance exhausted')
            if stable is not None and current-stable >= 60:
                break
            time.sleep(.5)
        elapsed = time.monotonic()-start
        self.guard(deadline, memory=False)
        c.require(self.waited+elapsed < 300, 'Cumulative memory admission allowance exhausted')
        self.waited += elapsed
        self.admitted = True
        self.admissions += 1
        c.save(c.OUT/'memory_admissions'/f'{self.admissions:03d}.json', {
            'status': 'passed', 'time': c.now(), 'fast_path': fast, 'wait_seconds': elapsed,
            'cumulative_wait_seconds': self.waited, 'minimum_available_bytes': minimum})


def perform(name, mode, payload, limit, deadline, envelope, freeze):
    c.require('torch' not in sys.modules, 'Supervisor must remain model-free')
    envelope.admit(deadline)
    folder = c.OUT/'workers'/name
    print(c.canonical({'stage': name, 'event': 'starting', 'time': c.now()}), flush=True)
    with c.StudyWorker('tools.move_evaluation_reuse_20261023.worker', folder,
        native_artifact=freeze['native_binary'], affinity=[0], initialization_timeout=clamp_timeout(60, deadline),
        absolute_deadline=deadline, max_message_bytes=32*1024**2, max_request_bytes=64*1024**2,
        max_log_bytes=256*1024**2, max_rss_bytes=2*c.GiB,
        env={'PYTHONPATH': str(c.ROOT/'src')+os.pathsep+str(c.ROOT), 'PYTHONDONTWRITEBYTECODE': '1',
             'CUDA_VISIBLE_DEVICES': '', 'STUDY_MODE': mode, 'STUDY_WORKER_DIR': str(folder),
             'RAYON_NUM_THREADS': '1', 'TOKIO_WORKER_THREADS': '1'},
        resource_guard=lambda: envelope.guard(deadline)) as child:
        check_implementation(c.read(folder/'implementation.json'),mode)
        receipt = child.request(payload, clamp_timeout(limit, deadline))
        c.require(receipt['status'] == 'ok' and receipt['accepted'], 'Request failed: '+c.canonical({
            k: receipt[k] for k in ('status', 'detail', 'receipt_elapsed_seconds')}))
        initialization = child.initialization
        check_implementation(receipt['result']['implementation'],mode)
    before = {(p['pid'], p['created']): p['cpu_seconds'] for p in initialization['telemetry']['process_tree']}
    cpu = {}
    for p in receipt['telemetry']['process_tree']:
        cpu[p['role']] = p['cpu_seconds']-before.get((p['pid'], p['created']), 0.)
    record = {'name': name, 'mode': mode, 'result': receipt['result'],
        'implementation':c.artifact(folder/'implementation.json'),
        'receipt': c.artifact(folder/(receipt['request_id']+'.json')),
        'initialization_seconds': initialization['elapsed_seconds'],
        'complete_response_seconds': receipt['receipt_elapsed_seconds'],
        'sampled_cpu_deltas_by_role': cpu,
        'cpu_scope': 'final receipt sample minus initialization sample; differs slightly from request endpoints',
        'request_tree_cpu_seconds': receipt['telemetry'].get('request_cpu_seconds'),
        'peak_tree_rss_bytes': receipt['telemetry']['peak_sampled_rss_bytes']}
    c.save(c.OUT/'measurements'/(name.replace('/', '_')+'.json'), record)
    print(c.canonical({'stage': name, 'event': 'completed', 'seconds': record['complete_response_seconds']}), flush=True)
    return record


def run(expected_sha):
    from tools.future_training_rules_20261021 import verify_runtime_sources
    launch = {'time':c.now(),'monotonic':float(os.environ['DIAGNOSTIC_START_MONOTONIC']),
              'unix_time':float(os.environ['DIAGNOSTIC_START_UNIX']),
              'pid':os.getpid(),'created':psutil.Process().create_time(),'freeze_sha256':expected_sha}
    c.save(c.OUT/'launch.json',launch)
    envelope = Envelope(launch)
    completed = []
    try:
        c.require(c.file_hash(c.OUT/'freeze.json') == expected_sha, 'Reuse freeze mismatch')
        freeze = c.read(c.OUT/'freeze.json')
        verify_runtime(freeze)
        envelope.guard(envelope.end,memory=False)
        c.require(processor_topology() == freeze['topology'], 'Topology changed')
        c.save(c.OUT/'supervisor_runtime_sources.json',{'sources':verify_runtime_sources(),'time':c.now()})
        from tools.move_evaluation_profile_20261023.analyze import compare_scientific_runs, inspect_old_prefix
        from .checks import validate_fixture, validate_selection
        plain_fixture = perform('qualification_plain','qualify_plain',{'operation':'entrypoint_checks'},
                                300,envelope.work_end,envelope,freeze)
        timed_fixture = perform('qualification_timed','qualify_timed',{'operation':'entrypoint_checks'},
                                300,envelope.work_end,envelope,freeze)
        fixture_check = validate_fixture(plain_fixture['result']['scientific'],timed_fixture['result']['scientific'],
            c.read(c.OLD/'entrypoint_qualification.json'),timed_fixture['result']['measurement'])
        c.require(fixture_check['status'] == 'passed', 'Selected timing fixture verification failed')
        fixture_artifact = c.save(c.OUT/'checks/qualification.json',fixture_check)
        c.save(c.OUT/'entrypoint_qualification.json',{'status':'passed','same_scientific_result_as_original':True,
            'ordinary_measurement':c.artifact(c.OUT/'measurements/qualification_plain.json'),
            'timed_measurement':c.artifact(c.OUT/'measurements/qualification_timed.json'),'checks':fixture_artifact})
        lock = c.read(c.PREP/'candidate_lock.json')
        for index in c.ROOT_INDICES:
            row = lock['roots'][index]
            payload = c.selector_payload(row)
            select_end = envelope.work_end-660
            envelope.guard(select_end)
            plain = perform(f'r{index:02d}/plain','plain',payload,c.REQUEST_SECONDS,select_end,envelope,freeze)
            timed = perform(f'r{index:02d}/timed','timed',payload,c.REQUEST_SECONDS,select_end,envelope,freeze)
            p,q = plain['result']['scientific'],timed['result']['scientific']
            comparison = compare_scientific_runs(p,q)
            timing_check = validate_selection(timed['result']['measurement'],c.ROOT/q['trace']['path'])
            c.require(timing_check['status'] == 'passed', 'Selected timing counts/integrity mismatch')
            timing_artifact = c.save(c.OUT/'timing_summaries'/f'r{index:02d}.json',timing_check)
            previous_record = c.read(c.PREVIOUS/'measurements'/f'r{index:02d}_plain.json')
            previous_comparison = compare_scientific_runs(previous_record['result']['scientific'],p)
            observation = latency_observation(previous_record['complete_response_seconds'],plain['complete_response_seconds'])
            old_prefix = None
            if index == 0:
                old_prefix = inspect_old_prefix(c.OLD/'workers/selection/r00/continuations.jsonl.gz',c.ROOT/p['trace']['path'])
            comparison_artifact = c.save(c.OUT/'comparisons'/f'r{index:02d}.json',
                {'pair':comparison,'previous_verified_ordinary':previous_comparison,'old_failed_prefix':old_prefix,
                 'observational_latency':observation})
            audit = perform(f'r{index:02d}/audit','audit',{'selector':payload,'result':p,
                'trace':c.static_artifact(p['trace'])},c.REQUEST_SECONDS,envelope.work_end,envelope,freeze)
            c.require(audit['result']['scientific']['status'] == 'passed', 'Independent full replay failed')
            completed.append({'index':index,'id':row['id'],'fit':row['fit'],
                'plain':c.artifact(c.OUT/'measurements'/f'r{index:02d}_plain.json'),
                'timed':c.artifact(c.OUT/'measurements'/f'r{index:02d}_timed.json'),
                'audit':c.artifact(c.OUT/'measurements'/f'r{index:02d}_audit.json'),
                'timing_summary':timing_artifact,'comparison':comparison_artifact,
                'plain_seconds':plain['complete_response_seconds'],'timed_seconds':timed['complete_response_seconds'],
                'audit_seconds':audit['complete_response_seconds'],'ordinary_within_120_seconds':observation['ordinary_within_120_seconds'],
                'observational_latency':observation})
            c.save(c.OUT/'completed_roots'/f'r{index:02d}.json',completed[-1])
        envelope.guard(envelope.end)
        c.require(len(completed) == 3, 'Fixed trio incomplete')
        verify_runtime(freeze)
        envelope.guard(envelope.end)
        inventory = [c.artifact(p) for p in sorted(c.OUT.rglob('*')) if p.is_file() and p.name not in
                     ('runner_stdout.log','runner_stderr.log','watchdog.json','launcher_stdout.log','launcher_stderr.log')]
        bound = c.save(c.OUT/'final_inventory.json',{'artifacts':inventory,'time':c.now()})
        envelope.guard(envelope.end)
        c.save(c.OUT/'completion.json',{'status':'completed_exact_reuse_diagnosis_only','roots':completed,'time':c.now(),
            'elapsed_seconds':time.monotonic()-launch['monotonic'],'final_inventory':bound,
            'original_120_second_qualification':'failed_unchanged','previous_instrumentation_failure':'preserved',
            'larger_data_deferred':True,'new_baseline_selection_requests':0,'full_18_position_qualification':False,
            'observational_latency_only':True})
    except BaseException:
        c.save(c.OUT/'failure.json',{'status':'held_failure_preserved','time':c.now(),
            'elapsed_seconds':time.monotonic()-launch['monotonic'],'error':traceback.format_exc(),
            'fully_completed_root_indices':[r['index'] for r in completed],'automatic_retry':False,
            'original_120_second_qualification':'failed_unchanged','fixed_trio_complete':False})
        raise


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('operation',choices=('prepare','run'))
    parser.add_argument('--freeze-sha256')
    args = parser.parse_args()
    prepare() if args.operation == 'prepare' else run(args.freeze_sha256)


if __name__ == '__main__':
    main()




