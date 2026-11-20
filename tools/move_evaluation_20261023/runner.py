"""Immutable, fail-closed 8h study: qualification, fixed selection, assessment."""
import argparse
from copy import deepcopy
import json
import os
from pathlib import Path
import shutil
import sys
import time
import traceback

import psutil

from . import common as c
from tools.external_strength_screen_20261018 import rules, audit as dictionary
from tools.future_training_rules_20261021 import decode_training_snapshot, require_official_terminal_target
from tools.local_player_qualification_128.clock import processor_topology


def bytes_used():
    return sum(p.stat().st_size for directory in (c.OUT, c.BUILD) if directory.exists()
               for p in directory.rglob('*') if p.is_file())


def clamp_timeout(limit, deadline, *, clock=time.monotonic):
    remaining = deadline-clock()
    c.require(remaining > 0, 'Phase deadline reached; no request grace')
    return min(limit, remaining)


class Envelope:
    def __init__(self, launch):
        self.start = launch['monotonic']
        self.wall_start = launch['unix_time']
        self.end = self.start+28800
        self.qual_end = self.start+10800
        self.collection_end = self.start+21600
        self.waited = 0.
        self.admitted = False
        self.admissions = 0

    def guard(self, deadline, *, memory=True):
        c.require(time.monotonic() < min(deadline, self.end), 'Phase/global elapsed deadline')
        c.require(time.time() < self.wall_start+28800, 'Absolute wall deadline')
        c.require(shutil.disk_usage(c.ROOT).free >= 5*c.GiB, 'Free disk reserve')
        c.require(bytes_used() < 2*c.GiB, 'New evidence/build size cap')
        if memory:
            c.require(psutil.virtual_memory().available >= 2*c.GiB, 'RAM shortage during active request')

    def admit(self, deadline):
        self.guard(deadline, memory=False)
        begun = time.monotonic()
        stable_since = begun if psutil.virtual_memory().available >= 2*c.GiB else None
        fast = self.admitted and stable_since is not None
        minimum = psutil.virtual_memory().available
        while not fast:
            self.guard(deadline, memory=False)
            current = time.monotonic()
            available = psutil.virtual_memory().available
            minimum = min(minimum, available)
            if available < 2*c.GiB:
                stable_since = None
            elif stable_since is None:
                stable_since = current
            c.require(self.waited+current-begun < 1800, 'Cumulative memory admission allowance exhausted')
            if stable_since is not None and current-stable_since >= 60:
                break
            time.sleep(.5)
        elapsed = time.monotonic()-begun
        self.waited += elapsed
        self.admitted = True
        self.admissions += 1
        c.save(c.OUT/'memory_admissions'/f'{self.admissions:05d}.json',
               {'status': 'passed', 'fast_path': fast, 'wait_seconds': elapsed,
                'cumulative_wait_seconds': self.waited, 'minimum_available_bytes': minimum,
                'time': c.now()})


def recover_panel_states(locked):
    """Reconstruct only the already observed saved roots, without any policy call."""
    source = c.ROOT/'outputs/matched_strength_recovery_20261022_01'
    final = c.read(source/'finalization.json')
    index = {a['path']: a for a in final['artifacts']}
    index.update({a['path']: a for a in c.read(source/'freeze.json')['inputs']})
    consumed, states = {}, {}
    def bound(path):
        a = index[path.relative_to(c.ROOT).as_posix()]
        c.verify([a]); consumed[a['path']] = a
        return c.read(path)
    for row in locked['roots']:
        folder = source/'sealed/games'/f"{row['game']:03d}"
        initial = bound(folder/'initial.json')
        state = decode_training_snapshot(initial['initial'])
        for ply in range(row['ply']):
            move = bound(folder/'moves'/f'{ply:04d}.json')
            before = rules.serialize_snapshot(state)
            c.require(c.digest(before) == move['before_sha256'] and move['status'] == 'completed', 'Saved prefix invalid')
            response = bound(c.ROOT/move['receipt']['path'])['result']
            action = response['decision']['action'] if move['player'] == 'local' else response['action']
            successor = rules.CorrectedEnv().step(state, rules.deserialize_action(action))
            after = rules.serialize_snapshot(successor)
            dictionary.check_step(before, action, after)
            c.require(c.digest(after) == move['after_sha256'], 'Saved prefix transition hash differs')
            state = successor
        dictionary.check_observer(rules.serialize_snapshot(state), external_observer(row['observer']))
        states[row['id']] = rules.serialize_snapshot(state)
    return states, list(consumed.values())


def external_observer(observer):
    return {k: (rules.OBSERVATION_SCHEMA if k == 'schema' else v)
            for k, v in observer.items() if k != 'context'}


def prepare():
    c.require(not (c.OUT/'freeze.json').exists(), 'Execution freeze already exists')
    c.OUT.mkdir(parents=True, exist_ok=True)
    receipt = c.read(c.PREP/'preparation_receipt.json')
    c.verify(receipt['authenticated_inputs']+receipt['outputs']+[receipt['source'], receipt['protocol']])
    design = c.read(c.PREP/'design.json')
    c.require(design['status'] == 'prepared_design_only' and design['resources']['job_seconds'] == 28800,
              'Unexpected prepared design')
    topology = processor_topology()
    c.require(topology['selected_logical_processor'] == 0, 'Qualified logical CPU0 unavailable')
    c.require(shutil.disk_usage(c.ROOT).free >= 7*c.GiB, 'Initial disk allowance unavailable')
    locked = c.read(c.PREP/'candidate_lock.json')
    states, imported = recover_panel_states(locked)
    c.save(c.OUT/'sealed/panel_states.json', states)
    from .qualification import build_fixtures, model_free_checks
    fixtures = build_fixtures()
    c.save(c.OUT/'fixtures.json', fixtures)
    c.save(c.OUT/'model_free_qualification.json', model_free_checks())
    build = c.read(c.BUILD/'build_receipt.json')
    c.verify(build['runtime_inputs']+[build['executable']])
    paths = set(c.ROOT.glob('src/**/*.py')) | set(c.ROOT.glob('tools/**/*.py'))
    paths.update(c.ROOT.glob('tests/test_move_evaluation*20261023.py'))
    paths.update([c.PROTOCOL, c.PREP/'design.json', c.PREP/'candidate_lock.json',
                  c.ROOT/'docs/engineering/move_evaluation_execution_20261023.md',
                  c.PREP/'preparation_receipt.json', c.PREP/'review.json', c.BUILD/'build_receipt.json',
                  c.OUT/'fixtures.json', c.OUT/'model_free_qualification.json', c.OUT/'sealed/panel_states.json'])
    inputs = {a['path']: a for a in receipt['authenticated_inputs']+imported+build['runtime_inputs']+[build['executable']]}
    for path in paths:
        item = c.artifact(path); inputs[item['path']] = item
    import numpy
    from tools.matched_strength_20261020.statistics import runtime_contract
    c.save(c.OUT/'statistics_runtime.json', runtime_contract())
    inputs['outputs/move_evaluation_20261023_01/statistics_runtime.json'] = c.artifact(c.OUT/'statistics_runtime.json')
    freeze = {'schema': 'bounded_move_evaluation_execution_v1', 'time': c.now(),
              'inputs': sorted(inputs.values(), key=lambda a: a['path']), 'design': c.artifact(c.PREP/'design.json'),
              'build_manifest': c.artifact(c.BUILD/'build_receipt.json'), 'native_binary': build['executable'],
              'python': sys.version, 'python_executable_sha256': c.file_hash(sys.executable), 'numpy': numpy.__version__,
              'topology': topology, 'job_seconds': 28800, 'qualification_seconds': 10800,
              'collection_cutoff_seconds': 21600, 'initialization_seconds': 60,
              'entrypoint_fixture_request_seconds': 300, 'new_data': 'corrected_rules_only',
              'max_evidence_bytes': 2*c.GiB, 'automatic_retries': 0,
              'assessment_model_lifetime': 'one_serial_worker_across_roots; at_most_one_setup_per_root; fresh_search_state_every_move',
              'authorization': 'User: implement and run the bounded move-evaluation study, starting with correctness qualification.'}
    c.save(c.OUT/'freeze.json', freeze)
    print(c.canonical({'prepared': True, 'freeze_sha256': c.file_hash(c.OUT/'freeze.json'),
                       'inputs': len(inputs), 'model_calls': 0}), flush=True)


def make_worker(name, mode, envelope, deadline, freeze):
    c.require('torch' not in sys.modules, 'Referee must remain model-free')
    envelope.admit(deadline)
    folder = c.OUT/'workers'/name
    native = None
    if mode in ('qualify', 'select', 'audit_selection'):
        native = freeze['native_binary']
    elif mode in ('assess',):
        profile = c.read(c.PROFILE)
        native = next(a for a in profile['artifacts'] if a['path'] == profile['native_binary'])
    child = c.StudyWorker('tools.move_evaluation_20261023.worker', folder,
        native_artifact=native, affinity=[0], initialization_timeout=clamp_timeout(60, deadline),
        absolute_deadline=deadline, max_message_bytes=32*1024**2, max_request_bytes=64*1024**2,
        max_log_bytes=512*1024**2, max_rss_bytes=2*c.GiB,
        env={'PYTHONPATH': str(c.ROOT/'src')+os.pathsep+str(c.ROOT), 'CUDA_VISIBLE_DEVICES': '',
             'STUDY_MODE': mode, 'STUDY_WORKER_DIR': str(folder), 'RAYON_NUM_THREADS': '1', 'TOKIO_WORKER_THREADS': '1'},
        resource_guard=lambda: envelope.guard(deadline))
    return child


def request(child, payload, seconds, deadline):
    receipt = child.request(payload, clamp_timeout(seconds, deadline))
    c.require(receipt['status'] == 'ok' and receipt['accepted'], 'Request failed: '+c.canonical({
        k: receipt[k] for k in ('status', 'detail', 'receipt_elapsed_seconds')}))
    return receipt


def static_trace_artifact(result):
    return {k: result['trace'][k] for k in ('path', 'sha256', 'bytes')}


def qualify_assessor(envelope, freeze):
    fixtures = c.read(c.OUT/'fixtures.json')
    records = []
    with make_worker('qualify_assessor', 'assess', envelope, envelope.qual_end, freeze) as child:
        for fixture in fixtures:
            snapshot = fixture['snapshot']
            state = decode_training_snapshot(snapshot)
            if state.terminal:
                continue
            observer = rules.observation_payload(state)
            dictionary.check_observer(snapshot, observer)
            for player in ('raw035', 'opponent'):
                receipt = request(child, {'observer': observer, 'player': player}, 1., envelope.qual_end)
                action = receipt['result']['action']
                successor = rules.CorrectedEnv().step(state, rules.deserialize_action(action))
                dictionary.check_step(snapshot, action, rules.serialize_snapshot(successor))
                records.append({'fixture': fixture['id'], 'player': player,
                                'receipt': c.artifact(child.output_dir/(receipt['request_id']+'.json'))})
    c.save(c.OUT/'assessor_qualification.json', {'status': 'passed', 'records': records})


def selection(envelope, freeze, locked):
    with make_worker('entrypoint_qualification', 'qualify', envelope, envelope.qual_end, freeze) as child:
        receipt = request(child, {'operation': 'entrypoint_checks'}, 300, envelope.qual_end)
        c.save(c.OUT/'entrypoint_qualification.json', receipt['result'])
    qualify_assessor(envelope, freeze)
    results = {}
    states = c.read(c.OUT/'sealed/panel_states.json')
    for index, row in enumerate(locked['roots']):
        payload = c.selector_payload(row)
        name = f'selection/r{index:02d}'
        with make_worker(name, 'select', envelope, envelope.qual_end, freeze) as child:
            receipt = request(child, payload, 120, envelope.qual_end)
            result = receipt['result']
            c.require(result['observer_sha256'] == c.digest(payload['observer']), 'Selected observer differs')
        c.save(c.OUT/'selection'/f'r{index:02d}.json', {'id': row['id'], 'result': result,
                'receipt': c.artifact(c.OUT/'workers'/name/'request-00000001.json')})
        with make_worker(f'audit_selection/r{index:02d}', 'audit_selection', envelope, envelope.qual_end, freeze) as child:
            verified = request(child, {'selector': payload, 'result': result, 'trace': static_trace_artifact(result)},
                               300, envelope.qual_end)
            c.save(c.OUT/'selection_audits'/f'r{index:02d}.json', verified['result'])
        if row['id'] in locked['privacy_repeat_roots']:
            from tools.external_strength_screen_20261018_02.runner import hidden_variant
            state = decode_training_snapshot(states[row['id']])
            variant = hidden_variant(state)
            c.require(rules.serialize_snapshot(state) != rules.serialize_snapshot(variant), 'Privacy variant unchanged')
            redacted = rules.observation_payload(variant)
            c.require(redacted == external_observer(row['observer']), 'Hidden-only variant changed lawful observation')
            privacy_payload = deepcopy(payload)
            privacy_payload['observer']['observation'] = redacted['observation']
            with make_worker(f'privacy/r{index:02d}', 'select', envelope, envelope.qual_end, freeze) as child:
                repeated = request(child, privacy_payload, 120, envelope.qual_end)['result']
                c.require(repeated['decision'] == result['decision'] and
                          repeated['trace']['sha256'] == result['trace']['sha256'] and
                          repeated['trace']['records'] == result['trace']['records'], 'Privacy scientific trace mismatch')
            c.save(c.OUT/'privacy'/f'r{index:02d}.json', {'status': 'passed', 'id': row['id'],
                'original_sha256': c.digest(states[row['id']]), 'variant_sha256': c.digest(rules.serialize_snapshot(variant)),
                'trace_sha256': repeated['trace']['sha256']})
        results[row['id']] = result
        print(c.canonical({'stage': 'selection_and_replay', 'complete_roots': index+1, 'total_roots': 18}), flush=True)
    c.require(sum(r['trace']['records'] for r in results.values()) == 9088, 'Full trajectory coverage missing')
    sealed = [{'id': row['id'], 'fit': row['fit'], 'control_choice': row['control_choice'],
               'alternative_choice': results[row['id']]['decision']['record']['choice'],
               'control_action': row['control_action'], 'alternative_action': results[row['id']]['decision']['action']}
              for row in locked['roots']]
    c.save(c.OUT/'sealed/choices.json', sealed)
    envelope.guard(envelope.qual_end)
    c.save(c.OUT/'qualification_complete.json', {'status': 'passed', 'roots': 18, 'privacy_repeats': 6,
                                               'fully_replayed_trajectories': 9088, 'choices': c.artifact(c.OUT/'sealed/choices.json')})
    return sealed


def initial_assessment_world(row, j):
    # This reproduces the observer sampler without importing model-bearing modules into the parent.
    import random
    from splendor_ai.bots.hidden_information import determinize_for_observer
    observer_state = rules.state_from_observation(external_observer(row['observer']))
    world = determinize_for_observer(observer_state, observer_state.current_player,
                                    random.Random(int(row['assessment_world_seeds_parent_only'][j])))
    return decode_training_snapshot(rules.serialize_snapshot(world))


def play_branch(child, row, choice, j, arm, envelope):
    started = time.monotonic()
    deadline = min(started+120, envelope.collection_end)
    state = initial_assessment_world(row, j)
    root_actor = state.current_player
    initial = rules.serialize_snapshot(state)
    action = choice['control_action' if arm == 'A' else 'alternative_action']
    steps = []
    folder = c.OUT/'assessment'/row['id'].replace('/', '_')/f'w{j:02d}_{arm}'
    c.save(folder/'initial.json', {'initial': initial, 'forced_action': action, 'root_actor': root_actor,
                                  'id': row['id'], 'world_index': j, 'arm': arm})
    while not state.terminal:
        envelope.guard(deadline)
        c.require(state.turn_index < 400, 'Assessment turn ceiling; unresolved')
        before = rules.serialize_snapshot(state)
        observer = rules.observation_payload(state)
        dictionary.check_observer(before, observer)
        receipt = None
        if not steps:
            player, received = 'forced', None
        else:
            player = 'raw035' if state.current_player == root_actor else 'opponent'
            received = request(child, {'observer': observer, 'player': player}, 1., deadline)
            action = received['result']['action']
            receipt = c.artifact(child.output_dir/(received['request_id']+'.json'))
        successor = rules.CorrectedEnv().step(state, rules.deserialize_action(action))
        after = rules.serialize_snapshot(successor)
        dictionary.check_step(before, action, after)
        step = {'before': before, 'observer': observer, 'action': action, 'after': after,
                'player': player, 'clock': received, 'receipt': receipt}
        c.save(folder/'steps'/f'{len(steps):04d}.json', step)
        steps.append(step)
        state = successor
    envelope.guard(deadline)
    final = rules.serialize_snapshot(state)
    winner = require_official_terminal_target(final, 'completed')
    score = .5 if winner is None else int(winner == root_actor)
    branch = {'id': row['id'], 'fit': row['fit'], 'world_index': j, 'arm': arm,
              'initial': initial, 'forced_action': choice['control_action' if arm == 'A' else 'alternative_action'],
              'root_actor': root_actor, 'steps': steps, 'final': final, 'score': score,
              'reason': 'completed', 'elapsed_seconds': time.monotonic()-started,
              'branch_started_monotonic': started, 'branch_finished_monotonic': time.monotonic(),
              'branch_deadline_monotonic': deadline}
    c.save(folder/'branch.json', branch)
    return branch


def assessment(envelope, freeze, locked, choices):
    selected = {r['id']: r for r in choices}
    changed = [r for r in locked['roots'] if selected[r['id']]['control_action'] != selected[r['id']]['alternative_action']]
    breadth = {'changed_roots': len(changed), 'by_fit': {str(f): sum(r['fit'] == f for r in changed)
                                                     for f in (161001, 161002, 161003)}}
    c.save(c.OUT/'breadth.json', breadth)
    if len(changed) < 6 or min(breadth['by_fit'].values()) < 1:
        return {'status': 'insufficient_intervention_breadth', 'breadth': breadth, 'assessment_started': False}
    wave_started, wave_bytes = time.monotonic(), bytes_used()
    # One loaded worker satisfies the balanced wave without multiple live workers
    # or a second model setup at any root; the qualified agent resets each move.
    with make_worker('assessment', 'assess', envelope, envelope.collection_end, freeze) as child:
        for phase, worlds in [('wave0', range(1)), ('remaining', range(1, 32))]:
            for index, row in enumerate(changed):
                envelope.admit(envelope.collection_end)
                for j in worlds:
                    for arm in ('A', 'B'):
                        play_branch(child, row, selected[row['id']], j, arm, envelope)
                print(c.canonical({'stage': 'assessment', 'phase': phase, 'complete_roots': index+1, 'changed_roots': len(changed)}), flush=True)
            if phase == 'wave0':
                elapsed = time.monotonic()-wave_started
                added_bytes = bytes_used()-wave_bytes
                remaining_projection = elapsed*31*1.35
                evidence_projection = bytes_used()+added_bytes*31*1.35
                okay = (remaining_projection < envelope.collection_end-time.monotonic() and evidence_projection < 2*c.GiB)
                c.save(c.OUT/'throughput_gate.json', {'status': 'passed' if okay else 'hold',
                    'elapsed_seconds': elapsed, 'added_bytes': added_bytes, 'worlds_per_changed_root': 1,
                    'remaining_seconds_projection_35pct': remaining_projection,
                    'evidence_bytes_projection_35pct': evidence_projection, 'outcomes_inspected_for_gate': False})
                c.require(okay, 'Outcome-blind throughput projection exceeds fixed allowance')
    return {'status': 'assessment_collected', 'changed_roots': len(changed), 'branches': len(changed)*64}


def analyze(envelope, freeze, locked, choices):
    choices = {r['id']: r for r in choices}
    rows = []
    verified_branches = 0
    seen_receipts = set()
    with make_worker('audit_assessment', 'audit_branch', envelope, envelope.end, freeze) as child:
        for row in locked['roots']:
            choice = choices[row['id']]
            item = {k: choice[k] for k in ('id', 'fit', 'control_choice', 'alternative_choice')}
            if choice['control_action'] == choice['alternative_action']:
                item.update(a_scores_halfpoints=None, b_scores_halfpoints=None, differences_halfpoints=[0]*32)
            else:
                scores = {'A': [], 'B': []}
                for j in range(32):
                    for arm in ('A', 'B'):
                        path = c.OUT/'assessment'/row['id'].replace('/', '_')/f'w{j:02d}_{arm}'/'branch.json'
                        branch = c.read(path)
                        c.require((branch['id'], branch['fit'], branch['world_index'], branch['arm']) ==
                                  (row['id'], row['fit'], j, arm), 'Assessment branch identity')
                        c.require(branch['forced_action'] == choice['control_action' if arm == 'A' else 'alternative_action'],
                                  'Assessment sealed intervention differs')
                        c.require(0 <= branch['elapsed_seconds'] <= 120 and
                                  branch['branch_started_monotonic'] < branch['branch_finished_monotonic'] <=
                                  branch['branch_deadline_monotonic'] <= envelope.collection_end and
                                  branch['branch_deadline_monotonic']-branch['branch_started_monotonic'] <= 120,
                                  'Assessment branch exceeded its clock')
                        scientific = {k: branch[k] for k in ('initial','forced_action','root_actor','steps','final','score','reason')}
                        for step in branch['steps'][1:]:
                            identity = step['receipt']['path']
                            c.require(identity not in seen_receipts, 'Assessment response receipt reused across moves/branches')
                            seen_receipts.add(identity)
                        receipt = request(child, {'branch': scientific, 'observer': row['observer'],
                            'world_seed': int(row['assessment_world_seeds_parent_only'][j])}, 120, envelope.end)
                        c.save(path.parent/'audit.json', receipt['result'])
                        scores[arm].append(int(2*branch['score']))
                        verified_branches += 1
                item.update(a_scores_halfpoints=scores['A'], b_scores_halfpoints=scores['B'],
                            differences_halfpoints=[b-a for a, b in zip(scores['A'], scores['B'], strict=True)])
            rows.append(item)
    c.save(c.OUT/'analysis_rows.json', rows)
    with make_worker('statistics', 'statistics', envelope, envelope.end, freeze) as child:
        result = request(child, {'rows': rows, 'candidate_lock': locked}, 600, envelope.end)['result']
    c.save(c.OUT/'independent_analysis.json', result)
    return {'status': 'completed', 'assessment_branches_verified': verified_branches, 'analysis': result}


def run(expected_sha):
    from tools.future_training_rules_20261021 import verify_runtime_sources
    c.save(c.OUT/'supervisor_runtime_sources.json', {'time': c.now(), 'sources': verify_runtime_sources()})
    c.require(c.file_hash(c.OUT/'freeze.json') == expected_sha, 'Execution freeze differs')
    freeze = c.read(c.OUT/'freeze.json')
    c.verify(freeze['inputs'])
    c.require(processor_topology() == freeze['topology'], 'CPU topology changed')
    c.require(c.file_hash(sys.executable) == freeze['python_executable_sha256'], 'Python executable changed')
    launch = {'time': c.now(), 'monotonic': time.monotonic(), 'unix_time': time.time(),
              'pid': os.getpid(), 'created': psutil.Process().create_time(), 'freeze_sha256': expected_sha}
    c.save(c.OUT/'launch.json', launch)
    envelope = Envelope(launch)
    locked = c.read(c.PREP/'candidate_lock.json')
    try:
        choices = selection(envelope, freeze, locked)
        result = assessment(envelope, freeze, locked, choices)
        if result['status'] == 'assessment_collected':
            result = analyze(envelope, freeze, locked, choices)
        envelope.guard(envelope.end)
        for item in freeze['inputs']:
            c.require(time.monotonic() < envelope.end and time.time() < envelope.wall_start+28800,
                      'Final authentication crossed the absolute deadline')
            c.verify([item])
        envelope.guard(envelope.end)
        inventory = []
        for path in sorted(c.OUT.rglob('*')):
            if not path.is_file() or path.name in ('runner_stdout.log', 'runner_stderr.log', 'watchdog.json'):
                continue
            c.require(time.monotonic() < envelope.end and time.time() < envelope.wall_start+28800,
                      'Final artifact binding crossed the absolute deadline')
            inventory.append(c.artifact(path))
        manifest = c.save(c.OUT/'final_inventory.json', {'artifacts': inventory,
            'mutable_execution_logs_excluded': ['runner_stdout.log', 'runner_stderr.log', 'watchdog.json'],
            'time': c.now(), 'bytes': sum(a['bytes'] for a in inventory)})
        envelope.guard(envelope.end)
        c.save(c.OUT/'completion.json', {**result, 'time': c.now(), 'elapsed_seconds': time.monotonic()-launch['monotonic'],
                    'freeze': c.artifact(c.OUT/'freeze.json'), 'final_inventory': manifest, 'larger_data_deferred': True})
        print(c.canonical({'status': result['status'], 'elapsed_seconds': time.monotonic()-launch['monotonic']}), flush=True)
    except BaseException:
        c.save(c.OUT/'failure.json', {'status': 'held_failure_preserved', 'time': c.now(),
            'elapsed_seconds': time.monotonic()-launch['monotonic'], 'error': traceback.format_exc(),
            'assessment_result_imputed': False, 'automatic_retry': False})
        raise


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('operation', choices=('prepare', 'run'))
    parser.add_argument('--freeze-sha256')
    args = parser.parse_args()
    if args.operation == 'prepare':
        prepare()
    else:
        run(args.freeze_sha256)


if __name__ == '__main__':
    main()
