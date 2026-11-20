"""Stage B: one fixed exact optimization, inherited clocks, gates and full replay."""
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
MODES = ('qualify_plain', 'qualify_timed', 'probe_reference', 'probe_plain', 'probe_timed', 'plain', 'timed', 'audit')
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
    c.require(abs(plan['boot_time']-psutil.boot_time()) <= 1.0, 'Study cannot resume after reboot')
    c.require(time.monotonic() >= plan['monotonic_origin'], 'Original study uptime must continue')
    return plan


def read_study_plan(freeze=None):
    path = c.PLAN_PATH
    c.require(c.file_hash(path) == c.PLAN_SHA, 'Original study plan changed')
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
    c.require(boot_estimate_runtime() == freeze['boot_estimate_runtime'], 'psutil boot-estimate implementation changed')
    c.require(torch_runtime() == freeze['torch_runtime'], 'Pinned Torch runtime changed')


def torch_runtime():
    from .inference import RUNTIME_HASHES
    root = Path(sys.prefix)/'Lib/site-packages/torch'
    result = []
    for relative,expected in RUNTIME_HASHES.items():
        path = (root/relative).resolve()
        c.require(path.is_relative_to(root.resolve()) and c.file_hash(path) == expected, 'Pinned Torch runtime: '+relative)
        result.append({'relative_path':relative,'absolute_path':str(path),'bytes':path.stat().st_size,'sha256':expected})
    return result


def boot_estimate_runtime():
    from psutil import _pswindows
    path = Path(_pswindows.__file__).resolve()
    return {'absolute_path':str(path),'bytes':path.stat().st_size,'sha256':c.file_hash(path),
        'scope':'psutil Windows boot_time documents up to one-second process-to-process fluctuation',
        'maximum_estimate_difference_seconds':1.0}


def load_prior():
    c.require(c.file_hash(c.STAGE_A/'freeze.json') == c.STAGE_A_FREEZE_SHA and
              c.file_hash(c.STAGE_A_STATUS/'closeout.json') == c.STAGE_A_CLOSEOUT_SHA, 'Closed Stage A identities')
    closeout = c.read(c.STAGE_A_STATUS/'closeout.json')
    execution = c.read(c.STAGE_A/'execution.json')
    c.require(closeout['schema'] == 'move_evaluation_inference_administrative_closeout_v1' and
              closeout['status'] == 'closed_completed_inference_diagnosis' and
              closeout['freeze'] == c.artifact(c.STAGE_A/'freeze.json') and
              closeout['original_plan'] == c.artifact(c.PLAN_PATH) and closeout['execution'] == execution and
              closeout['frozen_inputs_reauthenticated'] == 2480 and closeout['run_files'] == 73 and
              closeout['prior_failures_preserved'] is True and closeout['stage_b_started'] is False and
              execution['status'] == 'completed' and execution['exit_code'] == 0 and execution['cleanup_errors'] == [],
              'Stage A must be completely verified and closed')
    owned = execution['charged_owned_job_seconds']
    waited = closeout['cumulative_ram_wait_seconds']
    c.require(type(owned) in (int,float) and math.isfinite(owned) and 0 < owned < 900 and
              owned == max(execution['monotonic_end']-execution['monotonic_origin'],
                           execution['unix_end']-execution['unix_origin']) and
              execution['remaining_owned_job_seconds'] == 4500-owned and
              type(waited) in (int,float) and math.isfinite(waited) and 60 <= waited < 300,
              'Inherited actual time and RAM charges')
    run = closeout['run_inventory']
    c.require(len(run) == len({a['path'] for a in run}) == 73 and
              closeout['run_bytes'] == sum(a['bytes'] for a in run) and
              {a['path'] for a in run} == {p.relative_to(c.ROOT).as_posix() for p in c.STAGE_A.rglob('*') if p.is_file()},
              'Exact closed Stage A file inventory')
    c.verify(run)
    c.verify([closeout['source_inventory']])
    source = c.read(c.ROOT/closeout['source_inventory']['path'])
    c.verify(source['files'])
    c.require({a['path'] for a in source['files']} ==
              {p.relative_to(c.ROOT).as_posix() for p in c.SOURCE.rglob('*') if p.is_file()}, 'Preserved failed-attempt inventory')
    evidence = closeout['run_bytes']+sum(a['bytes'] for a in source['files'])
    c.require(evidence == closeout['shared_stage_a_evidence_bytes'] < c.GiB, 'Inherited combined evidence bytes')
    for identity in closeout['process_identities_closed']:
        try:
            current = psutil.Process(identity['pid'])
            c.require(abs(current.create_time()-identity['created']) >= .01, 'Closed Stage A process is still alive')
        except psutil.NoSuchProcess:
            pass
    return {'owned_seconds':owned,'ram_wait_seconds':waited,'evidence_bytes':evidence,
        'execution':c.artifact(c.STAGE_A/'execution.json'),'closeout':c.artifact(c.STAGE_A_STATUS/'closeout.json')}


def add_input(inputs, item):
    c.require(item['path'] not in inputs or inputs[item['path']] == item, 'Frozen dependency collision')
    inputs[item['path']] = item


def prepare():
    c.require(not (c.OUT/'freeze.json').exists(), 'Stage B preparation is one-shot')
    plan = read_study_plan()
    study_guard(plan,reserve=300)
    prior = load_prior()
    previous = c.read(c.STAGE_A/'freeze.json')
    c.require(len(previous['inputs']) == 2480, 'Exact inherited frozen inputs')
    c.verify(previous['inputs'])
    inputs = {a['path']:a for a in previous['inputs']}
    c.require(len(inputs) == 2480, 'No duplicate inherited inputs')
    closeout = c.read(c.STAGE_A_STATUS/'closeout.json')
    for item in closeout['run_inventory']:
        add_input(inputs,item)
    c.require(c.artifact(c.ROOT/closeout['inspection_source']['path']) == closeout['inspection_source'], 'Stage A inspector changed')
    add_input(inputs,closeout['inspection_source'])
    c.require(shutil.disk_usage(c.ROOT).free >= 6*c.GiB, 'Initial disk reserve')
    topology = processor_topology()
    c.require(topology == previous['topology'] and topology['selected_logical_processor'] == 0, 'Original full CPU topology')
    c.require(c.file_hash(c.PREVIOUS/'freeze.json') == c.PREVIOUS_FREEZE_SHA, 'Frozen three-root reference changed')
    lock = c.read(c.PREP/'candidate_lock.json')
    roots = [lock['roots'][i] for i in c.ROOT_INDICES]
    c.require([(x['game'],x['ply'],x['fit']) for x in roots] == [(25,8,161001),(57,14,161002),(58,4,161003)] and
              all(len(x['candidates']['actions']) == 4 for x in roots), 'Predetermined three fixed-work cases')
    references = []
    for index in c.ROOT_INDICES:
        path = c.PREVIOUS/'measurements'/f'r{index:02d}_plain.json'
        c.verify([c.static_artifact(c.read(path)['result']['scientific']['trace'])])
        c.require(c.read(c.PREVIOUS/'measurements'/f'r{index:02d}_audit.json')['result']['scientific']['status'] == 'passed',
                  'Original-policy root replay reference required')
        references.append(c.artifact(path))
    paths = set((c.ROOT/'tools/move_evaluation_inference_opt_20261023').glob('*.py'))
    paths.update(c.ROOT.glob('tests/test_move_evaluation_inference_opt_*_20261023.py'))
    paths.update(c.OUT.glob('prelaunch_*.json'))
    paths.update([c.PROTOCOL,c.STAGE_A_REPORT,c.STAGE_A_STATUS/'closeout.json',c.PLAN_PATH,
                  c.OUT/'prelaunch_tests.json'])
    tests = c.read(c.OUT/'prelaunch_tests.json')
    c.require(tests['status'] == 'passed' and tests['exit_code'] == 0, 'Stage B preflight required')
    for path in paths:
        add_input(inputs,c.artifact(path))
    freeze = {'schema':'fixed_evaluator_prebound_inference_optimization_v1','time':c.now(),
        'authorization':'One exact implementation selected after closed inference diagnosis, with fixed three-root timing/replay.',
        'inputs':sorted(inputs.values(),key=lambda x:x['path']),'timing_runtime':timing_runtime(),
        'boot_estimate_runtime':boot_estimate_runtime(),'torch_runtime':torch_runtime(),'python':sys.version,
        'python_executable_sha256':c.file_hash(sys.executable),'topology':topology,'native_binary':previous['native_binary'],
        'stage_a_freeze':c.artifact(c.STAGE_A/'freeze.json'),'stage_a_closeout':c.artifact(c.STAGE_A_STATUS/'closeout.json'),
        'stage_a_report':c.artifact(c.STAGE_A_REPORT),'study_plan':c.artifact(c.PLAN_PATH),'prior_stage_a':prior,
        'previous_roots':references,'root_indices':list(c.ROOT_INDICES),
        'roots':[{k:x[k] for k in ('id','game','ply','fit')} for x in roots],
        'job_seconds':3600,'final_reserve_seconds':300,'request_seconds':600,'gate_request_seconds':120,
        'initialization_seconds':60,'audit_reserve_seconds_per_root':660,'modes':list(MODES),
        'implementation_id':c.IMPLEMENTATION_ID,'implementation_metadata':{mode:implementation_metadata(mode) for mode in MODES},
        'max_evidence_bytes':c.GiB,'max_tree_rss_bytes':2*c.GiB,'maximum_cumulative_ram_wait_seconds':300,
        'automatic_retries':0,'full_18_position_qualification':False,'original_failure_unchanged':True,'larger_data_deferred':True}
    study_guard(plan,reserve=300)
    c.save(c.OUT/'freeze.json',freeze)
    print(c.canonical({'freeze_sha256':c.file_hash(c.OUT/'freeze.json'),'inputs':len(inputs),'model_calls':0}),flush=True)


def check_eligibility(path, mode, runtime):
    value = c.read(path)
    if mode in ('audit','probe_reference'):
        expected = {'schema':'entity_prebound_reference_dispatch_v1','active_opponent':'original_reference',
                    'optimization_enabled':False}
        c.require(c.canonical(value) == c.canonical(expected), 'Original-policy eligibility dispatch')
    else:
        from .inference import validate_eligibility_receipt
        validate_eligibility_receipt(value)
        c.require(value['runtime_inputs'] == runtime, 'Eligibility must bind exact frozen Torch paths/hashes')
    return c.artifact(path)


class Envelope:
    def __init__(self, launch, study_plan, prior):
        self.plan = validate_study_plan(study_plan)
        self.prior = prior
        self.start = launch['monotonic']
        self.wall_start = launch['unix_time']
        c.require(all(type(x) in (int,float) and math.isfinite(x) for x in (self.start,self.wall_start)) and
                  self.start >= self.plan['monotonic_origin'], 'Invalid owned job origin')
        self.end = min(self.start+c.JOB_SECONDS, self.start+OWNED_SECONDS-prior['owned_seconds'],
                       self.plan['monotonic_origin']+STUDY_SECONDS)
        self.work_end = self.end-c.FINAL_RESERVE
        self.waited = prior['ram_wait_seconds']
        self.admitted = False
        self.admissions = 0

    def guard(self, deadline, *, memory=True):
        cutoff = min(deadline,self.end)
        c.require(time.monotonic() < cutoff, 'Diagnostic stage/global deadline')
        c.require(time.time() < self.wall_start+(cutoff-self.start), 'Diagnostic wall stage/global deadline')
        study_guard(self.plan, reserve=self.end-cutoff)
        c.require(shutil.disk_usage(c.ROOT).free >= 5*c.GiB, 'Free disk reserve')
        used = self.prior['evidence_bytes']+sum(p.stat().st_size for p in c.OUT.rglob('*') if p.is_file())
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
            'cumulative_wait_seconds':self.waited,'prior_stage_a_wait_seconds':self.prior['ram_wait_seconds'],
            'minimum_available_bytes':minimum})


def perform(name, mode, payload, limit, deadline, envelope, freeze):
    c.require('torch' not in sys.modules, 'Supervisor must remain model-free')
    c.require(mode in MODES, 'Explicit Stage B mode')
    maximum = 120 if mode.startswith(('qualify_', 'probe_')) else 600
    c.require(0 < limit <= maximum, 'Bounded Stage B request')
    envelope.admit(deadline)
    effective_deadline = envelope.effective_deadline(deadline)
    folder = c.OUT/'workers'/name
    print(c.canonical({'stage': name, 'event': 'starting', 'time': c.now()}), flush=True)
    with c.StudyWorker('tools.move_evaluation_inference_opt_20261023.worker', folder,
        native_artifact=freeze['native_binary'], affinity=[0], initialization_timeout=clamp_timeout(60, effective_deadline),
        absolute_deadline=effective_deadline, max_message_bytes=32*1024**2, max_request_bytes=64*1024**2,
        max_log_bytes=256*1024**2, max_rss_bytes=2*c.GiB,
        env={'PYTHONPATH': str(c.ROOT/'src')+os.pathsep+str(c.ROOT), 'PYTHONDONTWRITEBYTECODE': '1',
             'CUDA_VISIBLE_DEVICES': '', 'STUDY_MODE': mode, 'STUDY_WORKER_DIR': str(folder),
             'RAYON_NUM_THREADS': '1', 'TOKIO_WORKER_THREADS': '1'},
        resource_guard=lambda: envelope.guard(deadline)) as child:
        check_implementation(c.read(folder/'implementation.json'),mode)
        eligibility = check_eligibility(folder/'eligibility.json',mode,freeze['torch_runtime'])
        request_deadline = min(effective_deadline,envelope.effective_deadline(deadline))
        receipt = child.request(payload, clamp_timeout(limit, request_deadline))
        c.require(receipt['status'] == 'ok' and receipt['accepted'], 'Request failed: '+c.canonical({
            k: receipt[k] for k in ('status', 'detail', 'receipt_elapsed_seconds')}))
        initialization = child.initialization
        check_implementation(receipt['result']['implementation'],mode)
        c.require(receipt['result']['eligibility'] == eligibility == c.artifact(folder/'eligibility.json'),
                  'Eligibility must match initialized immutable artifact')
        check_eligibility(folder/'eligibility.json',mode,freeze['torch_runtime'])
    before = {(p['pid'], p['created']): p['cpu_seconds'] for p in initialization['telemetry']['process_tree']}
    cpu = {}
    for p in receipt['telemetry']['process_tree']:
        cpu[p['role']] = p['cpu_seconds']-before.get((p['pid'], p['created']), 0.)
    record = {'name': name, 'mode': mode, 'result': receipt['result'],
        'implementation':c.artifact(folder/'implementation.json'),
        'eligibility':eligibility,
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
        'unix_time':float(os.environ['DIAGNOSTIC_START_UNIX']),'pid':os.getpid(),
        'created':psutil.Process().create_time(),'freeze_sha256':expected_sha,
        'study_plan':c.artifact(c.PLAN_PATH)}
    c.save(c.OUT/'launch.json',launch)
    requests,verified,completed = [],[],[]
    try:
        c.require(c.file_hash(c.OUT/'freeze.json') == expected_sha, 'Stage B freeze mismatch')
        freeze = c.read(c.OUT/'freeze.json')
        claim = c.read(c.OUT/'watchdog_claim.json')
        c.require(claim['freeze_sha256'] == expected_sha and claim['monotonic_origin'] == launch['monotonic'] and
                  claim['unix_origin'] == launch['unix_time'], 'Authenticated owned-job origins cannot change')
        c.require(freeze['job_seconds'] == c.JOB_SECONDS == 3600 and freeze['final_reserve_seconds'] == c.FINAL_RESERVE == 300 and
                  freeze['request_seconds'] == c.REQUEST_SECONDS == 600 and freeze['gate_request_seconds'] == 120 and
                  freeze['initialization_seconds'] == 60 and freeze['audit_reserve_seconds_per_root'] == 660,
                  'Fixed Stage B request and reserve limits')
        plan = read_study_plan(freeze)
        prior = load_prior()
        c.require(prior == freeze['prior_stage_a'], 'Inherited actual budgets changed')
        envelope = Envelope(launch,plan,prior)
        envelope.guard(envelope.end,memory=False)
        verify_runtime(freeze)
        envelope.guard(envelope.end,memory=False)
        c.require(processor_topology() == freeze['topology'], 'Topology changed')
        c.save(c.OUT/'supervisor_runtime_sources.json',{'sources':verify_runtime_sources(),'time':c.now()})
        from .checks import validate_fixture,validate_panel,validate_selection
        from tools.move_evaluation_profile_20261023.analyze import compare_scientific_runs,inspect_old_prefix
        from tools.move_evaluation_reuse_20261023.runner import latency_observation

        def request(name,mode,payload,limit=120,deadline=None):
            cutoff = envelope.work_end if deadline is None else deadline
            envelope.guard(cutoff,memory=False)
            value = perform(name,mode,payload,limit,cutoff,envelope,freeze)
            requests.append(name)
            return value

        payload = {'operation':'entrypoint_checks'}
        plain = request('qualification_plain','qualify_plain',payload)
        timed = request('qualification_timed','qualify_timed',payload)
        check = validate_fixture(plain['result']['scientific'],timed['result']['scientific'],
            c.read(c.ORIGINAL/'entrypoint_qualification.json'),timed['result']['measurement'])
        c.require(check['status'] == 'passed', 'Actual-model optimization qualification failed')
        checked_fixture = c.save(c.OUT/'checks/qualification.json',check)
        c.save(c.OUT/'entrypoint_qualification.json',{'status':'passed','same_scientific_result_as_original':True,
            'ordinary_measurement':c.artifact(c.OUT/'measurements/qualification_plain.json'),
            'timed_measurement':c.artifact(c.OUT/'measurements/qualification_timed.json'),'checks':checked_fixture})
        verified.append('qualification')
        payload = {'operation':'public_inference_panel','repetitions':8}
        reference = request('probe_reference','probe_reference',payload)
        plain = request('probe_plain','probe_plain',payload)
        timed = request('probe_timed','probe_timed',payload)
        check = validate_panel(reference['result']['scientific'],plain['result']['scientific'],
            timed['result']['scientific'],timed['result']['measurement'])
        c.require(check['status'] == 'passed', 'Public panel scientific/count verification failed')
        checked_panel = c.save(c.OUT/'checks/panel.json',check)
        c.save(c.OUT/'probe.json',{'status':'passed','checks':checked_panel,
            'reference':c.artifact(c.OUT/'measurements/probe_reference.json'),
            'plain':c.artifact(c.OUT/'measurements/probe_plain.json'),
            'timed':c.artifact(c.OUT/'measurements/probe_timed.json'),
            'panel_roots':18,'repetitions':8,'choices_per_request':144})
        verified.append('panel')
        lock = c.read(c.PREP/'candidate_lock.json')
        for index in c.ROOT_INDICES:
            row = lock['roots'][index]
            payload = c.selector_payload(row)
            select_end = envelope.work_end-660
            plain = request(f'r{index:02d}/plain','plain',payload,600,select_end)
            timed = request(f'r{index:02d}/timed','timed',payload,600,select_end)
            p,q = plain['result']['scientific'],timed['result']['scientific']
            comparison = compare_scientific_runs(p,q)
            timing_check = validate_selection(q,timed['result']['measurement'])
            c.require(timing_check['status'] == 'passed', 'Optimized selection count/integrity check failed')
            timing_artifact = c.save(c.OUT/'timing_summaries'/f'r{index:02d}.json',timing_check)
            previous = c.read(c.PREVIOUS/'measurements'/f'r{index:02d}_plain.json')
            previous_comparison = compare_scientific_runs(previous['result']['scientific'],p)
            observation = latency_observation(previous['complete_response_seconds'],plain['complete_response_seconds'])
            prefix = None if index != 0 else inspect_old_prefix(c.ORIGINAL/'workers/selection/r00/continuations.jsonl.gz',
                                                               c.ROOT/p['trace']['path'])
            comparison_artifact = c.save(c.OUT/'comparisons'/f'r{index:02d}.json',
                {'pair':comparison,'previous_verified_ordinary':previous_comparison,
                 'old_failed_prefix':prefix,'observational_latency':observation})
            audit = request(f'r{index:02d}/audit','audit',{'selector':payload,'result':p,
                'trace':c.static_artifact(p['trace'])},600,envelope.work_end)
            c.require(audit['result']['scientific']['status'] == 'passed', 'Original-policy full replay failed')
            completed.append({'index':index,'id':row['id'],'fit':row['fit'],
                'plain':c.artifact(c.OUT/'measurements'/f'r{index:02d}_plain.json'),
                'timed':c.artifact(c.OUT/'measurements'/f'r{index:02d}_timed.json'),
                'audit':c.artifact(c.OUT/'measurements'/f'r{index:02d}_audit.json'),
                'timing_summary':timing_artifact,'comparison':comparison_artifact,
                'plain_seconds':plain['complete_response_seconds'],'timed_seconds':timed['complete_response_seconds'],
                'audit_seconds':audit['complete_response_seconds'],
                'ordinary_within_120_seconds':observation['ordinary_within_120_seconds'],'observational_latency':observation})
            c.save(c.OUT/'completed_roots'/f'r{index:02d}.json',completed[-1])
        c.require(verified == ['qualification','panel'] and [x['index'] for x in completed] == [0,6,12] and
                  len(requests) == 14, 'Exact two gates and three full cases required')
        envelope.guard(envelope.end)
        verify_runtime(freeze)
        c.require(load_prior() == prior, 'Closed Stage A changed during Stage B')
        read_study_plan(freeze)
        envelope.guard(envelope.end)
        inventory = [c.artifact(p) for p in sorted(c.OUT.rglob('*')) if p.is_file() and p.name not in
            ('runner_stdout.log','runner_stderr.log','watchdog.json','launcher_stdout.log','launcher_stderr.log')]
        bound = c.save(c.OUT/'final_inventory.json',{'artifacts':inventory,'time':c.now()})
        envelope.guard(envelope.end)
        c.save(c.OUT/'completion.json',{'status':'completed_prebound_inference_optimization_diagnosis_only','time':c.now(),
            'completed_requests':requests,'verified_gates':verified,'roots':completed,'final_inventory':bound,
            'study_plan':freeze['study_plan'],'prior_stage_a':prior,'elapsed_seconds':time.monotonic()-launch['monotonic'],
            'study_elapsed_seconds':max(time.monotonic()-plan['monotonic_origin'],time.time()-plan['unix_origin']),
            'cumulative_ram_wait_seconds':envelope.waited,'stage_b_ram_wait_seconds':envelope.waited-prior['ram_wait_seconds'],
            'original_120_second_qualification':'failed_unchanged','prior_failures_preserved':True,
            'larger_data_deferred':True,'full_18_position_qualification':False,'new_baseline_selection_requests':0,
            'observational_latency_only':True,'optimized_implementations_evaluated':1})
    except BaseException:
        c.save(c.OUT/'failure.json',{'status':'held_failure_preserved','time':c.now(),
            'elapsed_seconds':time.monotonic()-launch['monotonic'],'error':traceback.format_exc(),
            'completed_requests':requests,'verified_gates':verified,'fully_completed_root_indices':[x['index'] for x in completed],
            'automatic_retry':False,'original_120_second_qualification':'failed_unchanged','fixed_trio_complete':False})
        raise


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('operation',choices=('prepare','run'))
    parser.add_argument('--freeze-sha256')
    args = parser.parse_args()
    prepare() if args.operation == 'prepare' else run(args.freeze_sha256)


if __name__ == '__main__':
    main()
