"""Stage A: bounded inference decomposition; no optimization or full selections."""
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
STUDY_SECONDS = 10800
OWNED_SECONDS = 4500
STAGE_B_SECONDS = 3600
MODES = ('qualify_plain', 'qualify_timed', 'probe_reference', 'probe_plain', 'probe_timed')
CENSUS_SHA256 = 'cf6af76fc1583e627a4e79103859ee0c9315904ed7612da62dfbb0163ae4b5d5'


def new_study_plan():
    return {'schema':'inference_optimization_study_plan_v1', 'time':c.now(),
        'monotonic_origin':time.monotonic(), 'unix_origin':time.time(), 'boot_time':psutil.boot_time(),
        'maximum_wall_seconds':STUDY_SECONDS, 'maximum_owned_job_seconds':OWNED_SECONDS,
        'stage_a_maximum_seconds':900, 'stage_b_maximum_seconds':STAGE_B_SECONDS,
        'maximum_cumulative_ram_wait_seconds':300, 'maximum_evidence_bytes':c.GiB,
        'automatic_retries':0, 'stage_b_requires_closed_stage_a':True,
        'stage_b_requires_separate_linked_freeze':True, 'maximum_optimized_implementations':1,
        'original_120_second_qualification':'failed_unchanged', 'full_18_position_qualification':False}


def validate_study_plan(plan):
    required = new_study_plan()
    for key in ('time', 'monotonic_origin', 'unix_origin', 'boot_time'):
        required.pop(key)
    c.require(set(plan) == set(required) | {'time','monotonic_origin','unix_origin','boot_time'}, 'Study plan fields')
    c.require(c.canonical({k:plan[k] for k in required}) == c.canonical(required), 'Study plan limits changed')
    c.require(all(type(plan[k]) in (int,float) and math.isfinite(plan[k]) and plan[k] > 0
                  for k in ('monotonic_origin','unix_origin','boot_time')), 'Study plan finite origins')
    c.require(plan['boot_time'] == psutil.boot_time(), 'Study cannot resume after reboot')
    return plan


def read_study_plan(freeze=None):
    path = c.OUT/'study_plan.json'
    if freeze is not None:
        c.require(c.artifact(path) == freeze['study_plan'] and
                  freeze['study_plan'] in freeze['inputs'], 'Study plan is not bound to freeze')
    return validate_study_plan(c.read(path))


def study_guard(plan, *, reserve=0):
    c.require(time.monotonic() >= plan['monotonic_origin'], 'Study monotonic origin is in the future')
    c.require(time.monotonic() < plan['monotonic_origin']+STUDY_SECONDS-reserve and
              time.time() < plan['unix_origin']+STUDY_SECONDS-reserve, 'Shared three-hour study deadline')


def implementation_metadata(mode):
    c.require(mode in MODES, 'Explicit inference diagnostic mode')
    from .worker import implementation_metadata as metadata
    return metadata(mode)


def check_implementation(value, mode):
    c.require(c.canonical(value) == c.canonical(implementation_metadata(mode)),
              'Declared inference implementation/reference mismatch')


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
    """Authenticate completed reuse evidence and its closed report, not live handoffs."""
    run = closeout['run_inventory']
    admin = [closeout['inspection_source'],closeout['report']]
    c.require(type(run) is list and type(admin) is list and
              closeout['run_files'] == len(run) and
              closeout['run_bytes'] == sum(item['bytes'] for item in run), 'Prior closeout inventory totals')
    status_dir = c.ROOT/'outputs/move_evaluation_reuse_20261023_status_01'
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
    c.require(not (c.OUT/'freeze.json').exists() and not (c.OUT/'study_plan.json').exists(),
              'Study preparation is one-shot; no clock reset')
    plan = new_study_plan()
    plan_artifact = c.save(c.OUT/'study_plan.json', plan)
    c.require(c.JOB_SECONDS == 900 and c.FINAL_RESERVE == 120 and c.REQUEST_SECONDS == 120,
              'Stage A fixed limits')
    c.require(c.file_hash(c.PREVIOUS/'freeze.json') == c.PREVIOUS_FREEZE_SHA, 'Previous reuse freeze changed')
    previous = c.read(c.PREVIOUS/'freeze.json')
    verify_previous = previous['inputs']
    c.require(len(verify_previous) == 2292, 'Exact prior frozen dependency inventory')
    c.verify(verify_previous)
    study_guard(plan)
    status_dir = c.ROOT/'outputs/move_evaluation_reuse_20261023_status_01'
    closeout = c.read(status_dir/'closeout.json')
    c.require(closeout['schema'] == 'move_evaluation_reuse_administrative_closeout_v1' and
              closeout['status'] == 'closed_completed_exact_reuse_diagnosis' and
              closeout['prior_failures_preserved'] is True and
              closeout['original_120_second_qualification'] == 'failed_unchanged' and
              closeout['freeze'] == c.artifact(c.PREVIOUS/'freeze.json') and
              closeout['frozen_inputs_reauthenticated'] == 2292 and closeout['run_files'] == 154 and
              closeout['execution']['status'] == 'completed' and closeout['execution']['exit_code'] == 0 and
              closeout['execution']['cleanup_errors'] == [] and
              [row['index'] for row in closeout['roots']] == [0,6,12], 'Completed reuse reference not authenticated')
    inputs = {a['path']:a for a in verify_previous}
    c.require(len(inputs) == len(verify_previous), 'Duplicate prior frozen inputs')
    inherit_closed_evidence(closeout, inputs)
    c.require(c.read(c.ORIGINAL/'failure.json')['status'] == 'held_failure_preserved' and
              c.read(c.ORIGINAL/'execution.json')['status'] == 'runner_failed', 'Original failure/closure missing')
    c.require(shutil.disk_usage(c.ROOT).free >= 6*c.GiB, 'Initial disk reserve')
    topology = processor_topology()
    c.require(topology['selected_logical_processor'] == 0, 'CPU0 unavailable')
    census = c.STATUS/'duplicate_input_census.json'
    c.require(c.file_hash(census) == CENSUS_SHA256, 'Prospective callback census changed')
    census_execution = c.read(c.STATUS/'duplicate_input_census_execution.json')
    c.require(census_execution['census_sha256'] == CENSUS_SHA256 and
              census_execution['script_sha256'] == c.file_hash(c.STATUS/'duplicate_input_census.py') and
              census_execution['calculation_completed'] is True and census_execution['output_json_complete'] is True and
              census_execution['models_executed'] == 0 and census_execution['source_artifacts_modified'] is False,
              'Census calculation/source receipt mismatch')
    paths = set((c.ROOT/'tools/move_evaluation_inference_20261023').glob('*.py'))
    paths.update(c.ROOT.glob('tests/test_move_evaluation_inference_*_20261023.py'))
    paths.update(c.OUT.glob('prelaunch_*.json'))
    paths.update([c.PROTOCOL, c.PREVIOUS/'freeze.json', status_dir/'closeout.json', census,
                  c.STATUS/'duplicate_input_census.py', c.STATUS/'duplicate_input_census_execution.json',
                  c.OUT/'study_plan.json', c.OUT/'prelaunch_tests.json'])
    tests = c.read(c.OUT/'prelaunch_tests.json')
    c.require(tests['status'] == 'passed' and tests['exit_code'] == 0, 'Prelaunch tests missing')
    for path in paths:
        item = c.artifact(path)
        c.require(item['path'] not in inputs or inputs[item['path']] == item, 'Inherited input changed')
        inputs[item['path']] = item
    freeze = {'schema':'fixed_evaluator_inference_diagnostic_v1','time':c.now(),
        'authorization':'Owner approved finer inference diagnosis followed by one separately frozen exact optimization.',
        'inputs':sorted(inputs.values(), key=lambda a:a['path']), 'timing_runtime':timing_runtime(),
        'python':sys.version, 'python_executable_sha256':c.file_hash(sys.executable),
        'topology':topology, 'native_binary':previous['native_binary'],
        'previous_freeze':c.artifact(c.PREVIOUS/'freeze.json'), 'previous_closeout':c.artifact(status_dir/'closeout.json'),
        'census':c.artifact(census), 'study_plan':plan_artifact,
        'job_seconds':900, 'final_reserve_seconds':120, 'request_seconds':120, 'initialization_seconds':60,
        'modes':list(MODES), 'implementation_metadata':{mode:implementation_metadata(mode) for mode in MODES},
        'panel_roots':18, 'panel_repetitions':8, 'panel_choices':144,
        'original_qualification_seconds':120, 'original_failure_unchanged':True,
        'max_evidence_bytes':c.GiB, 'max_tree_rss_bytes':2*c.GiB, 'maximum_ram_wait_seconds':300,
        'automatic_retries':0, 'full_selection_requests':0, 'larger_data_deferred':True}
    study_guard(plan)
    c.save(c.OUT/'freeze.json', freeze)
    print(c.canonical({'freeze_sha256':c.file_hash(c.OUT/'freeze.json'),'inputs':len(inputs),'model_calls':0}),flush=True)


class Envelope:
    def __init__(self, launch, study_plan):
        self.plan = validate_study_plan(study_plan)
        self.start = launch['monotonic']
        self.wall_start = launch['unix_time']
        c.require(all(type(x) in (int,float) and math.isfinite(x) for x in (self.start,self.wall_start)) and
                  self.start >= self.plan['monotonic_origin'], 'Invalid owned job origin')
        self.end = min(self.start+c.JOB_SECONDS, self.plan['monotonic_origin']+STUDY_SECONDS)
        self.work_end = self.end-c.FINAL_RESERVE
        self.waited = 0.
        self.admitted = False
        self.admissions = 0

    def guard(self, deadline, *, memory=True):
        cutoff = min(deadline,self.end)
        c.require(time.monotonic() < cutoff, 'Diagnostic stage/global deadline')
        c.require(time.time() < self.wall_start+(cutoff-self.start), 'Diagnostic wall stage/global deadline')
        study_guard(self.plan, reserve=self.end-cutoff)
        c.require(shutil.disk_usage(c.ROOT).free >= 5*c.GiB, 'Free disk reserve')
        used = sum(p.stat().st_size for p in c.OUT.rglob('*') if p.is_file())
        c.require(used < c.GiB, 'New evidence size cap')
        if memory:
            c.require(psutil.virtual_memory().available >= 2*c.GiB, 'RAM shortage during active request')

    def effective_deadline(self, deadline):
        self.guard(deadline)
        cutoff = min(deadline,self.end)
        wall_cutoff = min(self.wall_start+(cutoff-self.start),
                          self.plan['unix_origin']+STUDY_SECONDS-(self.end-cutoff))
        # Capture monotonic first: time spent reading the wall clock grants no refund.
        mono = time.monotonic()
        effective = min(cutoff,mono+wall_cutoff-time.time())
        c.require(effective > mono, 'No dual-clock request allowance remains')
        return effective

    def admit(self, deadline):
        self.guard(deadline,memory=False)
        start = time.monotonic()
        available = psutil.virtual_memory().available
        stable = start if available >= 2*c.GiB else None
        fast = self.admitted and stable is not None
        minimum = available
        while not fast:
            self.guard(deadline,memory=False)
            current = time.monotonic()
            available = psutil.virtual_memory().available
            minimum = min(minimum,available)
            if available < 2*c.GiB:
                stable = None
            elif stable is None:
                stable = current
            c.require(self.waited+current-start < 300, 'Cumulative memory admission allowance exhausted')
            if stable is not None and current-stable >= 60:
                break
            time.sleep(.5)
        elapsed = time.monotonic()-start
        self.guard(deadline,memory=False)
        c.require(self.waited+elapsed < 300, 'Cumulative memory admission allowance exhausted')
        self.waited += elapsed
        self.admitted = True
        self.admissions += 1
        c.save(c.OUT/'memory_admissions'/f'{self.admissions:03d}.json',{
            'status':'passed','time':c.now(),'fast_path':fast,'wait_seconds':elapsed,
            'cumulative_wait_seconds':self.waited,'minimum_available_bytes':minimum})


def perform(name, mode, payload, limit, deadline, envelope, freeze):
    c.require('torch' not in sys.modules, 'Supervisor must remain model-free')
    c.require(mode in MODES and 0 < limit <= c.REQUEST_SECONDS, 'Bounded Stage A request')
    envelope.admit(deadline)
    effective_deadline = envelope.effective_deadline(deadline)
    folder = c.OUT/'workers'/name
    print(c.canonical({'stage': name, 'event': 'starting', 'time': c.now()}), flush=True)
    with c.StudyWorker('tools.move_evaluation_inference_20261023.worker', folder,
        native_artifact=freeze['native_binary'], affinity=[0], initialization_timeout=clamp_timeout(60, effective_deadline),
        absolute_deadline=effective_deadline, max_message_bytes=32*1024**2, max_request_bytes=64*1024**2,
        max_log_bytes=256*1024**2, max_rss_bytes=2*c.GiB,
        env={'PYTHONPATH': str(c.ROOT/'src')+os.pathsep+str(c.ROOT), 'PYTHONDONTWRITEBYTECODE': '1',
             'CUDA_VISIBLE_DEVICES': '', 'STUDY_MODE': mode, 'STUDY_WORKER_DIR': str(folder),
             'RAYON_NUM_THREADS': '1', 'TOKIO_WORKER_THREADS': '1'},
        resource_guard=lambda: envelope.guard(deadline)) as child:
        check_implementation(c.read(folder/'implementation.json'),mode)
        request_deadline = min(effective_deadline,envelope.effective_deadline(deadline))
        receipt = child.request(payload, clamp_timeout(limit, request_deadline))
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
    launch = {'time':c.now(), 'monotonic':float(os.environ['DIAGNOSTIC_START_MONOTONIC']),
        'unix_time':float(os.environ['DIAGNOSTIC_START_UNIX']), 'pid':os.getpid(),
        'created':psutil.Process().create_time(), 'freeze_sha256':expected_sha,
        'study_plan':c.artifact(c.OUT/'study_plan.json')}
    c.save(c.OUT/'launch.json',launch)
    completed = []
    verified = []
    try:
        c.require(c.file_hash(c.OUT/'freeze.json') == expected_sha, 'Inference freeze mismatch')
        freeze = c.read(c.OUT/'freeze.json')
        plan = read_study_plan(freeze)
        envelope = Envelope(launch,plan)
        envelope.guard(envelope.end,memory=False)
        verify_runtime(freeze)
        envelope.guard(envelope.end,memory=False)
        c.require(processor_topology() == freeze['topology'], 'Topology changed')
        c.save(c.OUT/'supervisor_runtime_sources.json',{'sources':verify_runtime_sources(),'time':c.now()})
        from .checks import validate_fixture, validate_panel

        def request(name, mode, payload):
            envelope.guard(envelope.work_end,memory=False)
            value = perform(name,mode,payload,c.REQUEST_SECONDS,envelope.work_end,envelope,freeze)
            completed.append(name)
            return value

        fixture_payload = {'operation':'entrypoint_checks'}
        plain = request('qualification_plain','qualify_plain',fixture_payload)
        timed = request('qualification_timed','qualify_timed',fixture_payload)
        check = validate_fixture(plain['result']['scientific'],timed['result']['scientific'],
            c.read(c.ORIGINAL/'entrypoint_qualification.json'),timed['result']['measurement'])
        c.require(check['status'] == 'passed', 'Actual-model instrumentation qualification failed')
        checked_fixture = c.save(c.OUT/'checks/qualification.json',check)
        c.save(c.OUT/'entrypoint_qualification.json',{'status':'passed',
            'same_scientific_result_as_original':True, 'checks':checked_fixture,
            'ordinary_measurement':c.artifact(c.OUT/'measurements/qualification_plain.json'),
            'timed_measurement':c.artifact(c.OUT/'measurements/qualification_timed.json')})
        verified.append('qualification')

        payload = {'operation':'public_inference_panel','repetitions':8}
        reference = request('probe_reference','probe_reference',payload)
        plain = request('probe_plain','probe_plain',payload)
        timed = request('probe_timed','probe_timed',payload)
        check = validate_panel(reference['result']['scientific'],plain['result']['scientific'],
            timed['result']['scientific'],timed['result']['measurement'])
        c.require(check['status'] == 'passed', 'Public panel equality/count verification failed')
        checked_panel = c.save(c.OUT/'checks/panel.json',check)
        c.save(c.OUT/'probe.json',{'status':'passed','checks':checked_panel,
            'reference':c.artifact(c.OUT/'measurements/probe_reference.json'),
            'plain':c.artifact(c.OUT/'measurements/probe_plain.json'),
            'timed':c.artifact(c.OUT/'measurements/probe_timed.json'),
            'panel_roots':18,'repetitions':8,'choices_per_request':144,
            'timing_scope':'Bounded repeated public-input panel, not a full-planner latency estimate.'})
        verified.append('panel')
        envelope.guard(envelope.end)
        c.require(completed == ['qualification_plain','qualification_timed','probe_reference','probe_plain','probe_timed'] and
                  verified == ['qualification','panel'], 'Fixed diagnostic sequence incomplete')
        verify_runtime(freeze)
        read_study_plan(freeze)
        envelope.guard(envelope.end)
        inventory = [c.artifact(p) for p in sorted(c.OUT.rglob('*')) if p.is_file() and p.name not in
            ('runner_stdout.log','runner_stderr.log','watchdog.json','launcher_stdout.log','launcher_stderr.log')]
        bound = c.save(c.OUT/'final_inventory.json',{'artifacts':inventory,'time':c.now()})
        envelope.guard(envelope.end)
        c.save(c.OUT/'completion.json',{'status':'completed_inference_diagnosis_only','time':c.now(),
            'completed_requests':completed,'verified_stages':verified,'final_inventory':bound,
            'study_plan':freeze['study_plan'], 'elapsed_seconds':time.monotonic()-launch['monotonic'],
            'study_elapsed_seconds':max(time.monotonic()-plan['monotonic_origin'],time.time()-plan['unix_origin']),
            'cumulative_ram_wait_seconds':envelope.waited,
            'original_120_second_qualification':'failed_unchanged','larger_data_deferred':True,
            'optimization_enabled':False,'full_selection_requests':0,'full_18_position_qualification':False,
            'stage_b_started':False,'stage_b_requires_separate_linked_freeze':True})
    except BaseException:
        c.save(c.OUT/'failure.json',{'status':'held_failure_preserved','time':c.now(),
            'elapsed_seconds':time.monotonic()-launch['monotonic'],'error':traceback.format_exc(),
            'completed_requests':completed,'verified_stages':verified,'automatic_retry':False,
            'original_120_second_qualification':'failed_unchanged','stage_b_started':False})
        raise


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('operation',choices=('prepare','run'))
    parser.add_argument('--freeze-sha256')
    args = parser.parse_args()
    prepare() if args.operation == 'prepare' else run(args.freeze_sha256)


if __name__ == '__main__':
    main()


