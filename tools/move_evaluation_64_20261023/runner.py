"""One separately frozen A64/B64 study; fail closed on any prescribed gate."""
import argparse
from copy import deepcopy
import math
import os
from pathlib import Path
import shutil
import sys
import time
import traceback

import psutil

from . import common as c
from tools.external_strength_screen_20261018 import rules, audit as dictionary
from tools.future_training_rules_20261021 import decode_training_snapshot
from tools.local_player_qualification_128.clock import processor_topology

PREPARATION_SHA = 'a3f5a1f17505fb9858a313966cb7f5462abdb18a90ed0fc7584e9a7d16ec5ac9'
PREPARATION_CLOSEOUT_SHA = '7ac41ba44fd3318bea4a1c4068b37c1851ec6ca9bd2704b053ba15b094407862'
JOB_SECONDS = 28800


def bytes_used():
    return sum(p.stat().st_size for p in c.OUT.rglob('*') if p.is_file())


def clamp_timeout(limit, deadline, *, envelope=None, clock=time.monotonic, wall_clock=time.time):
    remaining = deadline - clock()
    if envelope is not None:
        remaining = min(remaining, envelope.end-clock(),
                        envelope.wall_start+min(deadline-envelope.start, JOB_SECONDS)-wall_clock())
    c.require(math.isfinite(limit) and limit > 0 and remaining > 0, 'Phase deadline reached; no request grace')
    return min(limit, remaining)


def add_input(inputs, item):
    c.require(item['path'] not in inputs or inputs[item['path']] == item, 'Frozen dependency collision')
    inputs[item['path']] = item


def external_observer(observer):
    return {k: rules.OBSERVATION_SCHEMA if k == 'schema' else v for k, v in observer.items() if k != 'context'}


def prepare():
    c.require(not (c.OUT/'freeze.json').exists(), 'Execution freeze already exists')
    c.OUT.mkdir(parents=True, exist_ok=True)
    c.require(c.file_hash(c.PREP/'preparation_receipt.json') == PREPARATION_SHA and
              c.file_hash(c.PREP/'closeout.json') == PREPARATION_CLOSEOUT_SHA, 'Prepared design identity')
    receipt, closed = c.read(c.PREP/'preparation_receipt.json'), c.read(c.PREP/'closeout.json')
    c.verify(receipt['authenticated_inputs']+receipt['outputs']+[receipt['source'], receipt['protocol']])
    c.verify(closed['artifacts']+[closed['protocol']])
    design = c.read(c.PREP/'design.json')
    c.require(design['status'] == 'prepared_design_only' and design['selection_worlds_per_root'] == 64 and
              design['qualification']['total_timed_requests'] == 54, 'Prepared scientific contract changed')
    tests, review = c.read(c.OUT/'prelaunch_tests.json'), c.read(c.OUT/'prelaunch_review.json')
    c.require(tests['status'] == review['status'] == 'passed' and tests['exit_code'] == 0, 'Prelaunch checks/review missing')
    c.verify(tests['source_inventory']+review['source_inventory'])
    c.require(shutil.disk_usage(c.ROOT).free >= 7*c.GiB, 'Initial disk allowance unavailable')
    topology = processor_topology()
    c.require(topology['selected_logical_processor'] == 0, 'Qualified logical CPU0 unavailable')
    old = c.read(c.OLD_PREP/'candidate_lock.json')
    selection = c.read(c.PREP/'selection_lock.json')
    assessment = c.read(c.PREP/'assessment_schedule_parent_only.json')
    locked = deepcopy(old)
    for i, row in enumerate(locked['roots']):
        new = selection['roots'][i]
        c.require(row['id'] == new['id'] == assessment['roots'][i]['id'], 'Root order changed')
        c.require(c.selector_payload(row) | {'selection_schedule': new['payload']['selection_schedule']} == new['payload'],
                  'Original candidates/observer/fit changed')
        expected = deepcopy(row['selection_schedule']); expected['world_seeds'] = expected['world_seeds'][:64]
        c.require(expected == new['payload']['selection_schedule'], '64 worlds must be original prefix')
        row['selection_schedule'] = expected
        row['assessment_world_seeds_parent_only'] = assessment['roots'][i]['world_seeds']
        row.update(control_choice=None, control_action=None, control_means=None, alternative_choice=None)
    locked['schema'] = 'move_evaluation_64_parent_lock_v1'
    locked['assessment_domain'] = assessment['domain']
    c.save(c.OUT/'sealed/parent_lock.json', locked)
    from tools.move_evaluation_20261023.runner import recover_panel_states
    states, imported = recover_panel_states(locked)
    c.save(c.OUT/'sealed/panel_states.json', states)
    from tools.move_evaluation_20261023.qualification import build_fixtures, model_free_checks
    c.save(c.OUT/'fixtures.json', build_fixtures())
    c.save(c.OUT/'model_free_qualification.json', model_free_checks())
    from tools.matched_strength_20261020.statistics import runtime_contract
    c.save(c.OUT/'statistics_runtime.json', runtime_contract())
    build = c.read(c.BUILD/'build_receipt.json')
    c.verify(build['runtime_inputs']+[build['executable']])
    inherited = c.read(c.ROOT/'outputs/move_evaluation_inference_opt_20261023_01/freeze.json')
    external_runtime = inherited['torch_runtime']
    for item in external_runtime:
        p = Path(item['absolute_path'])
        c.require(p.stat().st_size == item['bytes'] and c.file_hash(p) == item['sha256'], 'Pinned Torch runtime changed')
    inputs = {}
    for a in receipt['authenticated_inputs']+receipt['outputs']+closed['artifacts']+imported+build['runtime_inputs']+[build['executable']]:
        add_input(inputs, a)
    paths = set((c.ROOT/'tools/move_evaluation_64_20261023').glob('*.py'))
    paths.update(c.ROOT.glob('tests/test_move_evaluation_64_*_20261023.py'))
    paths.update(c.OUT.glob('prelaunch_*'))
    paths.update([c.PROTOCOL, c.PREP/'closeout.json', c.BUILD/'build_receipt.json',
                  c.ROOT/'docs/engineering/move_evaluation_64_execution_20261023.md',
                  c.OUT/'fixtures.json', c.OUT/'model_free_qualification.json', c.OUT/'statistics_runtime.json',
                  c.OUT/'sealed/parent_lock.json', c.OUT/'sealed/panel_states.json'])
    for path in paths:
        if path.is_file():
            add_input(inputs, c.artifact(path))
    c.verify(list(inputs.values()))
    freeze = dict(schema='bounded_move_evaluation_64_execution_v1', time=c.now(),
        inputs=sorted(inputs.values(), key=lambda a:a['path']), design=c.artifact(c.PREP/'design.json'),
        native_binary=build['executable'], python=sys.version, python_executable_sha256=c.file_hash(sys.executable),
        torch_runtime=external_runtime, topology=topology, job_seconds=JOB_SECONDS,
        pilot_seconds=3600, qualification_seconds=10800, collection_cutoff_seconds=21600,
        initialization_seconds=60, selector_seconds=120, selector_margin_seconds=96, audit_seconds=300,
        max_evidence_bytes=2*c.GiB, automatic_retries=0, larger_data_deferred=True,
        authorization='User approved implementation and execution of the prepared64-world bounded study.')
    c.save(c.OUT/'freeze.json', freeze)
    print(c.canonical(dict(prepared=True, freeze_sha256=c.file_hash(c.OUT/'freeze.json'), inputs=len(inputs), model_calls=0)), flush=True)


class Envelope:
    def __init__(self, launch):
        self.start, self.wall_start = launch['monotonic'], launch['unix_time']
        self.end, self.pilot_end = self.start+JOB_SECONDS, self.start+3600
        self.qual_end, self.collection_end = self.start+10800, self.start+21600
        self.waited, self.admitted, self.admissions = 0., False, 0

    def effective_deadline(self, deadline):
        cutoff = min(deadline, self.end)
        return min(cutoff, time.monotonic()+self.wall_start+(cutoff-self.start)-time.time())

    def guard(self, deadline, *, memory=True):
        cutoff = min(deadline, self.end)
        c.require(time.monotonic() < cutoff and time.time() < self.wall_start+(cutoff-self.start),
                  'Phase/global real-clock deadline')
        c.require(shutil.disk_usage(c.ROOT).free >= 5*c.GiB, 'Free disk reserve')
        c.require(bytes_used() < 2*c.GiB, 'New evidence cap')
        if memory:
            c.require(psutil.virtual_memory().available >= 2*c.GiB, 'RAM shortage during active request')

    def admit(self, deadline):
        self.guard(deadline, memory=False)
        begun = time.monotonic()
        available = psutil.virtual_memory().available
        stable = begun if available >= 2*c.GiB else None
        fast = self.admitted and stable is not None
        minimum = available
        while not fast:
            self.guard(deadline, memory=False)
            current, available = time.monotonic(), psutil.virtual_memory().available
            minimum = min(minimum, available)
            if available < 2*c.GiB:
                stable = None
            elif stable is None:
                stable = current
            c.require(self.waited+current-begun < 1800, 'Cumulative RAM admission allowance exhausted')
            if stable is not None and current-stable >= 60:
                break
            time.sleep(.5)
        elapsed = time.monotonic()-begun
        self.guard(deadline, memory=False)
        c.require(self.waited+elapsed < 1800, 'Cumulative RAM admission allowance exhausted')
        self.waited += elapsed; self.admitted = True; self.admissions += 1
        c.save(c.OUT/'memory_admissions'/f'{self.admissions:05d}.json', dict(status='passed', fast_path=fast,
            wait_seconds=elapsed, cumulative_wait_seconds=self.waited, minimum_available_bytes=minimum, time=c.now()))


def make_worker(name, mode, envelope, deadline, freeze, *, arm=None):
    c.require('torch' not in sys.modules, 'Referee must remain model-free')
    envelope.admit(deadline)
    folder = c.OUT/'workers'/name
    native = None
    if mode in ('qualify', 'select', 'audit_selection') and arm == 'B':
        native = freeze['native_binary']
    elif mode in ('assess', 'qualify_assessor'):
        profile = c.read(c.PROFILE)
        native = next(a for a in profile['artifacts'] if a['path'] == profile['native_binary'])
    print(c.canonical(dict(stage=name, event='starting', time=c.now())), flush=True)
    child = c.StudyWorker('tools.move_evaluation_64_20261023.worker', folder,
        native_artifact=native, affinity=[0], initialization_timeout=clamp_timeout(60, deadline, envelope=envelope),
        absolute_deadline=deadline, max_message_bytes=32*1024**2, max_request_bytes=64*1024**2,
        max_log_bytes=512*1024**2, max_rss_bytes=2*c.GiB,
        env={'PYTHONPATH': str(c.ROOT/'src')+os.pathsep+str(c.ROOT), 'PYTHONDONTWRITEBYTECODE': '1',
             'CUDA_VISIBLE_DEVICES': '', 'STUDY_MODE': mode, 'STUDY_ARM': arm or '', 'STUDY_WORKER_DIR': str(folder),
             'OMP_NUM_THREADS':'1', 'MKL_NUM_THREADS':'1', 'OPENBLAS_NUM_THREADS':'1',
             'RAYON_NUM_THREADS':'1', 'TOKIO_WORKER_THREADS':'1'}, resource_guard=lambda:envelope.guard(deadline))
    child._study_envelope = envelope
    try:
        envelope.guard(deadline)
    except BaseException:
        child.close()
        raise
    return child


def request(child, payload, seconds, deadline):
    envelope = child._study_envelope
    receipt = child.request(payload, clamp_timeout(seconds, deadline, envelope=envelope))
    envelope.guard(deadline)
    c.require(receipt['status'] == 'ok' and receipt['accepted'], 'Request failed: '+c.canonical({
        k:receipt[k] for k in ('status','detail','receipt_elapsed_seconds')}))
    return receipt


def margin_passed(receipt):
    elapsed = receipt.get('receipt_elapsed_seconds')
    return (receipt.get('status') == 'ok' and receipt.get('accepted') is True and
            type(elapsed) in (int,float) and math.isfinite(elapsed) and 0 <= elapsed <= 96)


def prefix_check(row, result, arm, index):
    from .checks import check_archived_prefix
    trace_path = archived_trace_path = None
    if arm == 'A':
        c.verify([row['source_receipt']])
        saved = c.read(c.ROOT/row['source_receipt']['path'])['result']
    elif index in (0,6,12):
        path = c.ROOT/'outputs/move_evaluation_reuse_20261023_01/measurements'/f'r{index:02d}_plain.json'
        saved = c.read(path)['result']['scientific']
        c.verify([{k:saved['trace'][k] for k in ('path','bytes','sha256')}])
        trace_path, archived_trace_path = c.ROOT/result['trace']['path'], c.ROOT/saved['trace']['path']
    else:
        return dict(status='not_available', scope='no_archived_B128_at_this_root')
    return check_archived_prefix(result,saved,arm,trace_path=trace_path,archived_trace_path=archived_trace_path)


def duplicate_check(result, prior):
    from .checks import compare_duplicate
    return compare_duplicate(prior,result)


def selection(envelope, freeze, locked):
    from .checks import check_selection
    for arm in ('A','B'):
        with make_worker('entrypoint_'+arm, 'qualify', envelope, envelope.pilot_end, freeze, arm=arm) as child:
            receipt = request(child, {'operation':'entrypoint_checks'}, 300, envelope.pilot_end)
            c.require(receipt['result']['status'] == 'entrypoint_counterexamples_passed', 'Actual entrypoint qualification failed')
            c.save(c.OUT/f'entrypoint_{arm}.json', receipt['result'])
    schedule = c.read(c.PREP/'selection_lock.json')['timed_request_order']
    states = c.read(c.OUT/'sealed/panel_states.json')
    core, audits, completed, replayed = {}, {}, 0, 0

    def replay(index, arm, deadline):
        nonlocal replayed
        row = locked['roots'][index]; result = core[(index,arm)]
        trace = deepcopy(result['trace'])
        with make_worker(f'audit/r{index:02d}_{arm}', 'audit_selection', envelope, deadline, freeze, arm=arm) as child:
            receipt = request(child, {'selector':c.selector_payload(row), 'result':result, 'trace':trace}, 300, deadline)
            audit = receipt['result']
            c.require(audit['status'] == 'passed' and audit['audited_continuations'] == 64*len(row['candidates']['actions']),
                      'Complete independent replay missing')
            c.save(c.OUT/'audits'/f'r{index:02d}_{arm}.json', audit)
            audits[(index,arm)] = audit; replayed += audit['audited_continuations']

    for ordinal, entry in enumerate(schedule):
        index, arm = entry['index'], entry['arm']
        row = locked['roots'][index]
        deadline = envelope.pilot_end if entry['phase'] == 'pilot' else envelope.qual_end
        payload = c.selector_payload(row)
        privacy = None
        if entry['role'] == 'privacy_repeat':
            from tools.external_strength_screen_20261018_02.runner import hidden_variant
            state = decode_training_snapshot(states[row['id']]); variant = hidden_variant(state)
            before, after = rules.serialize_snapshot(state), rules.serialize_snapshot(variant)
            c.require(before != after, 'Privacy variant unchanged')
            observer = rules.observation_payload(variant)
            dictionary.check_observer(after, observer)
            c.require(observer == external_observer(row['observer']), 'Privacy variant changed lawful observation')
            payload = deepcopy(payload); payload['observer']['observation'] = observer['observation']
            privacy = dict(original_sha256=c.digest(before), variant_sha256=c.digest(after))
        name = f'selection/q{ordinal:02d}_r{index:02d}_{arm}'
        with make_worker(name, 'select', envelope, deadline, freeze, arm=arm) as child:
            receipt = request(child, payload, 120, deadline)
            result = receipt['result']
            checked = check_selection(result,payload,arm)
            c.save(c.OUT/'selections'/f'q{ordinal:02d}.json', dict(entry=entry, result=result,
                receipt=c.artifact(child.output_dir/(receipt['request_id']+'.json')),
                complete_response_seconds=receipt['receipt_elapsed_seconds'], margin_passed=margin_passed(receipt),
                model_free_validation=checked))
        # Preserve the entire response before testing the engineering margin.
        c.require(margin_passed(receipt), 'Completed selector lacks96-second margin; no assessment')
        if entry['role'] == 'core':
            core[(index,arm)] = result
            c.save(c.OUT/'prefix_checks'/f'r{index:02d}_{arm}.json', prefix_check(row,result,arm,index))
            if entry['phase'] != 'pilot':
                replay(index,arm,deadline)
        else:
            duplicate_check(result, core[(index,arm)])
            c.save(c.OUT/'repeat_checks'/f'q{ordinal:02d}.json', dict(status='passed', entry=entry, privacy=privacy))
        completed += 1
        if ordinal == 11:
            for pilot_index in (0,6,12):
                for pilot_arm in ('A','B'):
                    replay(pilot_index,pilot_arm,envelope.pilot_end)
            envelope.guard(envelope.pilot_end)
            c.require(replayed == 1536, 'Pilot replay coverage')
            c.save(c.OUT/'pilot_complete.json', dict(status='passed', requests=12, replayed_continuations=1536,
                elapsed_seconds=time.monotonic()-envelope.start))
        print(c.canonical(dict(stage='selection_and_replay', requests=completed, total_requests=54,
                              replayed_continuations=replayed, time=c.now())), flush=True)
    c.require(completed == 54 and len(core) == len(audits) == 36 and replayed == 9088, 'Full qualification coverage')
    choices = []
    for i,row in enumerate(locked['roots']):
        a,b = core[(i,'A')]['decision'], core[(i,'B')]['decision']
        choices.append(dict(id=row['id'],fit=row['fit'],control_choice=a['record']['choice'],
            alternative_choice=b['record']['choice'],control_action=a['action'],alternative_action=b['action']))
        row.update(control_choice=a['record']['choice'],control_action=a['action'],control_means=a['record']['means'],
                   alternative_choice=b['record']['choice'])
    c.save(c.OUT/'sealed/choices.json', choices)
    c.save(c.OUT/'sealed/evaluated_parent_lock.json', locked)
    from .assessment import qualify_assessor
    qualify_assessor(envelope,freeze,make_worker,request)
    envelope.guard(envelope.qual_end)
    c.save(c.OUT/'qualification_complete.json', dict(status='passed',roots=18,timed_requests=54,
        privacy_requests=12,fully_replayed_outputs=36,fully_replayed_trajectories=9088,
        choices=c.artifact(c.OUT/'sealed/choices.json'),lock=c.artifact(c.OUT/'sealed/evaluated_parent_lock.json')))
    return choices


def verify_runtime(freeze, envelope=None):
    for item in freeze['inputs']:
        if envelope:
            envelope.guard(envelope.end,memory=False)
        c.verify([item])
    c.require(processor_topology() == freeze['topology'] and sys.version == freeze['python'] and
              c.file_hash(sys.executable) == freeze['python_executable_sha256'], 'Runtime/topology changed')
    for item in freeze['torch_runtime']:
        p = Path(item['absolute_path'])
        c.require(p.stat().st_size == item['bytes'] and c.file_hash(p) == item['sha256'], 'Torch runtime changed')


def run(expected_sha):
    # The owning watchdog starts both clocks before source admission and process creation.
    claim = c.read(c.OUT/'watchdog_claim.json')
    launch = dict(time=c.now(), monotonic=claim['monotonic_origin'], unix_time=claim['unix_origin'],
        pid=os.getpid(),created=psutil.Process().create_time(),freeze_sha256=expected_sha)
    c.require(claim['freeze_sha256'] == expected_sha, 'Watchdog freeze claim differs')
    envelope = Envelope(launch)
    c.save(c.OUT/'launch.json', launch)
    try:
        from tools.future_training_rules_20261021 import verify_runtime_sources
        c.save(c.OUT/'supervisor_runtime_sources.json', dict(time=c.now(),sources=verify_runtime_sources()))
        c.require(c.file_hash(c.OUT/'freeze.json') == expected_sha, 'Execution freeze differs')
        freeze = c.read(c.OUT/'freeze.json'); verify_runtime(freeze,envelope)
        locked = c.read(c.OUT/'sealed/parent_lock.json')
        choices = selection(envelope,freeze,locked)
        from .assessment import run_assessment, analyze
        result = run_assessment(envelope,freeze,locked,choices,make_worker,request,bytes_used=bytes_used)
        if result['status'] == 'assessment_collected':
            result = analyze(envelope,freeze,locked,choices,make_worker,request)
        verify_runtime(freeze,envelope)
        envelope.guard(envelope.end,memory=False)
        inventory = []
        excluded = ('runner_stdout.log','runner_stderr.log','watchdog.json')
        for path in sorted(c.OUT.rglob('*')):
            if path.is_file() and path.name not in excluded:
                envelope.guard(envelope.end,memory=False)
                inventory.append(c.artifact(path))
        manifest = c.save(c.OUT/'final_inventory.json', dict(artifacts=inventory,time=c.now(),
            bytes=sum(a['bytes'] for a in inventory),mutable_execution_logs_excluded=list(excluded)))
        envelope.guard(envelope.end,memory=False)
        c.save(c.OUT/'completion.json', dict(**result,time=c.now(),elapsed_seconds=time.monotonic()-envelope.start,
            freeze=c.artifact(c.OUT/'freeze.json'),final_inventory=manifest,larger_data_deferred=True))
        print(c.canonical(dict(status=result['status'],elapsed_seconds=time.monotonic()-envelope.start)),flush=True)
    except BaseException:
        c.save(c.OUT/'failure.json', dict(status='held_failure_preserved',time=c.now(),
            elapsed_seconds=time.monotonic()-envelope.start,error=traceback.format_exc(),
            assessment_result_imputed=False,automatic_retry=False))
        raise


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('operation', choices=('prepare','run'))
    parser.add_argument('--freeze-sha256')
    args = parser.parse_args()
    prepare() if args.operation == 'prepare' else run(args.freeze_sha256)


if __name__ == '__main__':
    main()
