"""Prepare/freeze a fresh matched benchmark; execution requires a separate run command."""
from __future__ import annotations

import argparse
from copy import deepcopy
import os
import secrets
from pathlib import Path
import shutil
import subprocess
import sys
import time
import traceback
import psutil

from tools.matched_strength_20261020 import common as c
from tools.external_strength_screen_20261018 import rules, audit
from tools.external_strength_screen_20261018_02.runner import hidden_variant
from tools.splendorust_120_qualification_20261018.common import ProcessTreeWorker, load_profile
from tools.splendorust_120_qualification_20261018.runner import check_result as check_opponent
from tools.analyze_external_strength_screen_20261018_02 import check_diagnostics
from tools.corrected_planner_qualification_20261018.clock import PlannerWorker
from tools.local_player_qualification_128.clock import processor_topology, WorkerInitializationError

PLAYER_FAILURES = {'timeout', 'worker_exit', 'malformed', 'wrong_request',
                   'illegal', 'unsolicited', 'write_error'}
TESTS = ['tests/test_matched_strength_adapter_20261020.py',
         'tests/test_matched_strength_runner_20261020.py',
         'tests/test_matched_strength_analyzer_20261020.py',
         'tests/test_matched_strength_completion_20261020.py',
         'tests/test_matched_strength_statistics_20261020.py',
         'tests/test_operational_final_validation_repair_20261020.py',
         'tests/test_local_player_qualification_clock.py',
         'tests/test_external_screen_rules_20261018.py',
         'tests/test_external_screen_audit_20261018.py',
         'tests/test_splendorust_120_adapter_20261018.py']


PILOT_HASHES = {
    'freeze.json': 'e5c635c1cd5f0e4e9e499a421ab25f8a34b82076707ee946e10d3f0644f904ba',
    'analyzer_repair_02/freeze.json': 'fa27d136c589f54f0e2e993beb3b706e150520e1e744375e81daf9ee705611e6',
    'independent_analysis_v2.json': 'd3259938221b8e5836bd79d1a7bdbcf30762f3bd5cfc57413ac04c914986a0db',
    'finalization.json': '23fa728c97290b3a9e21edd7401c0e539fee75ef36fc3d33d1d07e74807bcee6',
}


def authenticated_pilot():
    """Authenticate accepted evidence without reopening matchup outcomes."""
    for name, expected in PILOT_HASHES.items():
        c.require(c.file_hash(c.PRIOR_PILOT/name) == expected, 'Prior pilot identity: '+name)
    accepted = c.read(c.PRIOR_PILOT/'independent_analysis_v2.json')
    final = c.read(c.PRIOR_PILOT/'finalization.json')
    c.require(accepted['status'] == 'passed' and final['status'] == 'accepted' and
              accepted['outcomes_closed'] and final['outcomes_closed'] and
              accepted['strength_analysis_games'] == final['strength_analysis_games'] == 0,
              'Prior pilot not accepted with closed outcomes')
    original = c.read(c.PRIOR_PILOT/'freeze.json')
    repair = c.read(c.PRIOR_PILOT/'analyzer_repair_02/freeze.json')
    c.verify(original['inputs']); c.verify(repair['inputs'])
    c.verify([item for item in final['artifacts']
              if item['path'] not in ('AGENT_CURRENT.md', 'AGENT_HANDOFF.md')])
    c.verify(repair['preserved_failure'] + [repair[key] for key in
        ('original_analyzer', 'original_completion', 'original_final_input_check',
         'original_freeze', 'original_launch')])
    return original, repair


def without_result(value):
    return {k: v for k, v in value.items() if k != 'result'}


def seed_inventory(base, schedule):
    command = ['rg', '--files', '--no-ignore', 'outputs', '-g', 'manifest.json',
               '-g', 'schedule.json', '-g', '*seed_registry*.json']
    found = subprocess.run(command, cwd=c.ROOT, text=True, capture_output=True, check=True)
    targets = {int(x['seed']) for x in schedule}
    c.require(len(targets) == c.config()['pair_count'], 'Distinct paired worlds required')
    checked, skipped, collisions = [], [], []
    def walk(value, name):
        if type(value) is int and value in targets: collisions.append(name)
        elif isinstance(value, str) and value.isdecimal() and int(value) in targets: collisions.append(name)
        elif isinstance(value, list):
            for v in value: walk(v, name)
        elif isinstance(value, dict):
            for v in value.values(): walk(v, name)
    for name in sorted(found.stdout.splitlines()):
        p = c.ROOT/name
        if p.resolve().is_relative_to(base.resolve()): continue
        item = c.artifact(p)
        c.require(p.stat().st_size <= 64*1024**2,
                  'Historical seed registry exceeds 64MiB scan ceiling; unscanned input blocks freeze')
        walk(c.read(p), name); checked.append(item)
    c.require(not collisions, 'Historical seed collision')
    return {'namespace': c.PRIVATE_SEED_NAMESPACE, 'derivation': 'full SHA256 canonical JSON [SEED_TAG,private_namespace,private_nonce,pair]',
            'distinct_seeds': c.config()['pair_count'], 'collisions': collisions, 'checked': checked, 'skipped_oversize': skipped,
            'scope': 'Every integer/decimal string in every discovered manifest/schedule/seed registry; any file above64MiB blocks freeze; not every historical game log'}


def build_validation():
    cases = []
    for fixture, game in zip(c.TRANSFER_INDICES, c.VALIDATION_GAMES, strict=True):
        old = c.read(c.PRIOR_LOCAL/'fixtures'/f'{fixture:02d}.json')
        original = old['snapshot']
        variant = rules.serialize_snapshot(hidden_variant(rules.deserialize_snapshot(original)))
        c.require(original != variant, 'Validation hidden variant did not change')
        for mode in ('pilot_transfer', 'operational'):
            context = (c.pilot_context(fixture) if mode == 'pilot_transfer'
                       else c.operational_context(game, original['state']['turn_index']))
            for kind, snapshot in (('original', original), ('hidden_variant', variant)):
                observer = c.make_observer_payload(snapshot, context)
                audit.check_observer(snapshot, c.opponent_observer(observer))
                cases.append({'index': len(cases), 'fixture': fixture, 'mode': mode, 'kind': kind,
                    'fit': context['fit'], 'snapshot': snapshot, 'observer': observer,
                    'reference_receipt': c.artifact(c.PRIOR_PILOT/'workers'/'validation'/f'v{c.PILOT_CASE_INDICES[c.TRANSFER_INDICES.index(fixture)]:03d}'/'request-00000001.json')})
    c.require(len(cases) == 12, 'Twelve validation cases required')
    for i in range(0, 12, 2):
        c.require(cases[i]['observer'] == cases[i+1]['observer'], 'Validation observer privacy mismatch')
    return cases


def accepted_history():
    """Only acceptance/control metadata are read; no operational matchup totals."""
    c.require(not (c.PRIOR_ACCEPTANCE/'hold.json').exists(), 'Prior validation repair has a hold')
    acceptance = c.read(c.PRIOR_ACCEPTANCE/'finalization_validation.json')
    c.require(acceptance['status'] == 'passed' and acceptance['outcomes_closed'] is True,
              'Accepted operational verification required')
    c.verify([acceptance['finalization']])
    final = c.read(c.PRIOR_ACCEPTANCE/'finalization.json')
    c.require(final['status'] == 'operational_checks_passed_with_preserved_timeouts' and
              final['games'] == 18 and final['preserved_local_timeout_forfeits'] == 4 and
              final['outcomes_closed'] is True and final['strength_analysis_games'] == 0,
              'Retained operational failures and closed outcomes required')
    c.verify(final['artifacts'])
    c.require(c.file_hash(c.PRIOR_COMPLETED/'freeze.json') == c.PRIOR_COMPLETED_FREEZE_SHA256,
              'Completed operational source freeze changed')
    original = c.read(c.PRIOR_COMPLETED/'freeze.json')
    c.verify(original['inputs'])
    repair = c.read(c.PRIOR_ACCEPTANCE/'repair_freeze.json')
    c.verify(repair['inputs'])
    inventory = c.read(c.ROOT/repair['source_inventory']['path'])
    actual = {p.relative_to(c.ROOT).as_posix() for p in c.PRIOR_COMPLETED.rglob('*') if p.is_file()}
    c.require(actual == {x['path'] for x in inventory['files']}, 'Completed source inventory changed')
    c.verify(inventory['files'])
    return original, repair, final, inventory


def prepare(config_path):
    """No model/search/game request occurs in preparation. Freeze is exclusive."""
    cfg = c.validate_config(c.read(config_path))
    base = c.OUTPUT
    base.mkdir(parents=True, exist_ok=False)
    try:
        c.require('torch' not in sys.modules, 'Preparation must remain model-free')
        c.save_new(base/'config.json', cfg)
        from tools.matched_strength_20261020.completion import contract
        c.require(c.read(c.PREPARATION/'completion_contract.json') == contract(),
                  'Prospective final-analysis process contract changed')
        from tools.matched_strength_20261020.statistics import runtime_contract
        runtime_path = c.PREPARATION/'statistics_runtime.json'
        runtime = runtime_contract()
        if runtime_path.exists():
            c.require(c.read(runtime_path) == runtime, 'Prospective statistics runtime changed')
        else:
            c.save_new(runtime_path, runtime)
        c.require(shutil.disk_usage(c.ROOT).free >= cfg['evidence_cap_bytes']+c.MIN_FREE,
                  'Preparation requires full evidence allowance plus disk reserve')
        original, repair, final, prior_inventory = accepted_history()
        policy_base = c.ROOT/'outputs/local_timeout_policy_review_20261020_01'
        policy = c.read(policy_base/'review_verification.json')
        c.require(policy['status'] == 'passed_read_only_evidence_review' and policy['outcomes_closed'] is True and
                  policy['benchmark_launched'] is False and policy['new_games'] == 0 and
                  policy['model_inference_executed'] is False, 'Accepted unchanged-player deadline policy review')
        c.verify(policy['artifacts'])
        topology = processor_topology()
        c.require(topology == original['topology'], 'Run preparation with original full supervisor affinity')
        load_profile(c.EXTERNAL_PROFILE_PATH, c.EXTERNAL_PROFILE_SHA256)
        seed_material = {'schema': 'matched_strength_private_seed_material_v1', 'private_nonce': secrets.token_hex(32)}
        c.save_new(base/'sealed/referee_seed_material.json', seed_material)
        schedule = c.make_schedule(seed_material)
        inventory = seed_inventory(base, schedule)
        c.save_new(base/'seed_collision_check.json', inventory)
        c.save_new(base/'sealed/schedule.json', schedule)
        c.save_new(base/'validation_cases.json', build_validation())
        c.save_new(base/'local_profile.json', {'profile_id': c.LOCAL_PROFILE_ID, 'adapter_id': c.ADAPTER_ID, 'schema': c.SCHEMA,
            'ruleset_id': c.RULESET_ID, 'catalogue_sha256': c.CATALOGUE_SHA256,
            'fits': list(c.FITS), 'candidates': 4, 'worlds': 128, 'backend': 'unchanged_serial',
            'parent_profile_id': 'four_candidate128_corrected_live_operational_20261019_v1',
            'public_namespace': c.LIVE_NAMESPACE, 'public_search_tag': c.SEARCH_TAG,
            'fresh_worker_each_decision': True, 'trace_delivery_inside_move_clock': True,
            'deadline_policy': 'hard120_complete_response_or_forfeit_no_fallback'})
        code = 'import psutil,pytest; psutil.Process().cpu_affinity([2]); raise SystemExit(pytest.main('+repr(['-q', *TESTS, '-p', 'no:cacheprovider'])+'))'
        test = subprocess.run([sys.executable, '-B', '-c', code], cwd=c.ROOT,
            env={**os.environ, 'PYTHONPATH': str(c.ROOT/'src')+os.pathsep+str(c.ROOT), 'PYTHONDONTWRITEBYTECODE': '1'},
            text=True, capture_output=True, timeout=300, creationflags=getattr(subprocess, 'CREATE_NO_WINDOW', 0))
        c.save_new(base/'preflight_tests.json', {'returncode': test.returncode, 'stdout': test.stdout, 'stderr': test.stderr,
                                              'cpu': 2, 'models_or_games_executed': False})
        c.require(test.returncode == 0, 'Preflight failed; preserve attempt')
        paths = {x['path'] for source in (original, repair) for x in source['inputs']}
        paths.update(x['path'] for x in final['artifacts'])
        paths.update(x['path'] for x in prior_inventory['files'])
        for directory in (c.PRIOR_ACCEPTANCE, policy_base, c.ROOT/'tools/matched_strength_20261020', c.PREPARATION, base):
            paths.update(p.relative_to(c.ROOT).as_posix() for p in directory.rglob('*')
                         if p.is_file() and '__pycache__' not in p.parts)
        paths.update(TESTS)
        paths.update(x['path'] for x in policy['artifacts'])
        paths.update(x['path'] for x in inventory['checked']+inventory['skipped_oversize'])
        paths.update(('tools/analyze_matched_strength_20261020.py',
                      'docs/engineering/matched_strength_20261020.md',
                      'docs/engineering/matched_strength_20261020_review.md',
                      'docs/engineering/verification_policy.md'))
        c.verify(original['inputs']); c.verify(repair['inputs'])
        c.save_new(base/'freeze.json', {'schema': 'matched_strength_freeze_v1', 'time': c.now(),
            'inputs': [c.artifact(c.ROOT/p) for p in sorted(paths)], 'config': c.artifact(base/'config.json'),
            'prior_acceptance': c.artifact(c.PRIOR_ACCEPTANCE/'finalization_validation.json'),
            'prior_pilot_lineage': [c.artifact(c.PRIOR_PILOT/name) for name in PILOT_HASHES],
            'topology': topology, 'executable': sys.executable, 'ruleset_id': c.RULESET_ID,
            'catalogue_sha256': c.CATALOGUE_SHA256, 'local_profile_id': c.LOCAL_PROFILE_ID, 'adapter_id': c.ADAPTER_ID,
            'opponent_profile': c.artifact(c.EXTERNAL_PROFILE_PATH), 'games': 2*cfg['pair_count'], 'pairs': cfg['pair_count'],
            'seconds_per_move': c.MOVE_SECONDS, 'initialization_seconds': c.INIT_SECONDS,
            'audit_seconds': c.AUDIT_SECONDS, 'execution_cap_seconds': cfg['execution_cap_seconds'],
            'analysis_seconds': c.ANALYSIS_SECONDS, 'cumulative_memory_wait_seconds': cfg['cumulative_memory_wait_seconds'],
            'post_game_reserve_seconds': cfg['post_game_reserve_seconds'], 'final_reserve_seconds': c.FINAL_RESERVE,
            'evidence_cap_bytes': cfg['evidence_cap_bytes'], 'minimum_free_bytes': c.MIN_FREE,
            'max_process_tree_rss_bytes': c.MAX_WORKER_RSS, 'maximum_plies': c.MAX_PLIES,
            'validation_requests': 12, 'sample_worlds': 8, 'audit_seed': c.AUDIT_SEED,
            'audit_domain': c.AUDIT_DOMAIN, 'outcomes_closed': True, 'strength_analysis_games': 0,
            'planned_strength_games': 2*cfg['pair_count'], 'larger_data_deferred': True,
            'prepared_only': True, 'preparation_launches_execution': False})
        print(c.canonical({'prepared': True, 'frozen_inputs': len(paths), 'validation_requests': 12,
                          'planned_strength_games': 2*cfg['pair_count'], 'execution_started': False,
                          'outcomes_closed': True}), flush=True)
    except BaseException:
        c.save_new(base/'preparation_failure.json', {'time': c.now(), 'failure': traceback.format_exc()})
        raise


def worker_options(deadline, cpu, *, reserve, mode, trace=None):
    c.require('torch' not in sys.modules, 'Supervisor must remain model-free')
    admission_started = time.monotonic()
    admission = c.boundary(c.OUTPUT, deadline)
    request_seconds = c.AUDIT_SECONDS if mode == 'audit' else c.MOVE_SECONDS
    if admission['remaining_seconds'] < c.INIT_SECONDS+request_seconds+reserve:
        raise c.ResourceHold('Insufficient dual-clock worker admission reserve')
    effective_deadline = min(deadline-reserve, admission_started+admission['remaining_seconds']-reserve)
    def live_guard():
        receipt=c.boundary(c.OUTPUT,deadline,memory_headroom=False)
        if receipt['remaining_seconds'] <= reserve:
            raise c.ResourceHold('Dual-clock protected reserve reached')
        return receipt
    environment = {'PYTHONPATH': str(c.ROOT/'src')+os.pathsep+str(c.ROOT), 'CUDA_VISIBLE_DEVICES': '',
        'LIVE_WORKER_MODE': mode, 'RAYON_NUM_THREADS': '1', 'TOKIO_WORKER_THREADS': '1'}
    if trace is not None: environment['LIVE_TRACE_PATH'] = str(trace.resolve())
    if mode == 'opponent':
        environment.update(SPLENDOR_OPPONENT_PROFILE=str(c.EXTERNAL_PROFILE_PATH.resolve()),
                           SPLENDOR_OPPONENT_PROFILE_SHA256=c.EXTERNAL_PROFILE_SHA256)
    return {'affinity': [cpu], 'initialization_timeout': c.INIT_SECONDS,
        'absolute_deadline': effective_deadline, 'env': environment,
        'max_message_bytes': 8*1024**2,
        'max_request_bytes': (16*1024**2 if mode == 'audit' else 1024**2),
        'max_log_bytes': 2*1024**2,
        'max_rss_bytes': c.MAX_WORKER_RSS,
        'resource_guard': live_guard}


def completed_result(row):
    receipt = c.read(c.ROOT/row['receipt']['path'])
    c.require(c.artifact(c.ROOT/row['receipt']['path']) == row['receipt'], 'Receipt artifact changed')
    return receipt['result']


def response_has_recognized_illegal_action(result, player, observer, legal, trace_path):
    """Recognize only a complete, bound action response, never a broken adapter.

    This is called after the clock captured the complete response. Recognition
    failures deliberately return False so the ordinary schema/work checks hold
    the experiment; they must never be converted into an illegal-action loss.
    """
    try:
        c.require(type(result) is dict, 'Response object')
        action = result['decision']['action'] if player == 'local' else result['action']
        c.require(type(action) is dict and c.serialize_action(c.deserialize_action(action)) == action,
                  'Canonical action encoding')
        for field in ('tier', 'market_index', 'reserved_index'):
            c.require(action[field] is None or type(action[field]) is int, 'Action index type')
        c.require(action['noble_id'] is None or type(action['noble_id']) is str, 'Noble type')
        for field in ('take_tokens', 'return_tokens', 'spend_tokens'):
            c.require(type(action[field]) is list and all(type(x) is str for x in action[field]), 'Token encoding')
        c.require(type(action['metadata']) is list and all(type(x) is list and len(x) == 2 and
                  all(type(v) is str for v in x) for x in action['metadata']), 'Metadata encoding')
        if player == 'local':
            # Singleton support bypasses only referee membership, not envelope,
            # completed-work, trace identity, or selected-candidate checks.
            c.check_wrapper(result, observer, [action], trace_path)
            record = result['decision']['record']; context = observer['context']
            c.require(record['root_identity'] == context['identity'] and
                      record['profile'] == c.LOCAL_PROFILE_ID and record['ruleset_id'] == c.RULESET_ID and
                      record['catalogue_sha256'] == c.CATALOGUE_SHA256 and
                      record['proposal']['fit'] == context['fit'] and
                      record['proposal']['model_id'] == 'anchor_'+str(context['fit']), 'Local record identity')
            choice = record['choice']; candidates = record['candidates']['actions']
            c.require(type(choice) is int and 0 <= choice < len(candidates) and
                      candidates[choice] == action, 'Selected candidate binding')
        elif player == 'opponent':
            identity = {'player': 'opponent', 'observer_sha256': c.digest(observer),
                'legal_actions_sha256': c.digest(legal), 'legal_count': len(legal),
                'ruleset': c.RULESET_ID, 'catalogue_sha256': c.CATALOGUE_SHA256,
                'profile_sha256': c.EXTERNAL_PROFILE_SHA256}
            c.require(set(result) == set(identity) | {'status', 'action', 'diagnostics'} and
                      all(result[k] == v for k, v in identity.items()) and result['status'] == 'completed',
                      'Opponent completed envelope')
            check_diagnostics(result['diagnostics'], observer, action, legal, c.read(c.EXTERNAL_PROFILE_PATH))
        else:
            return False
        return action not in legal
    except (KeyError, TypeError, ValueError, OSError, IndexError, AttributeError):
        return False


def decision_attempt(name, player, snapshot, observer, deadline, cpu, *, reserve):
    from tools.matched_strength_20261020.memory import wait_for_memory
    admission = wait_for_memory(c.OUTPUT, deadline, name=name, reserve=reserve, request_seconds=c.MOVE_SECONDS)
    if deadline-time.monotonic() < c.INIT_SECONDS+c.MOVE_SECONDS+reserve:
        raise c.ResourceHold('Insufficient move reserve')
    folder = c.OUTPUT/'workers'/name
    trace = folder/'continuations.jsonl.gz'
    row = {'id': name, 'player': player, 'observer': observer, 'status': 'incomplete', 'correct': False,
           'memory_admission': c.artifact(c.OUTPUT/'memory_admissions'/(name+'.json'))}
    started = time.monotonic()
    try:
        audit.check_observer(snapshot, c.opponent_observer(observer) if player == 'local' else observer)
        objects = rules.legal_actions(rules.deserialize_snapshot(snapshot))
        legal = [c.serialize_action(a) for a in objects]
        worker_class = PlannerWorker if player == 'local' else ProcessTreeWorker
        module = ('tools.matched_strength_20261020.worker' if player == 'local'
                  else 'tools.splendorust_120_qualification_20261018.worker')
        payload = {'observer': observer} if player == 'local' else {'player': 'opponent', 'observer': observer}
        with worker_class(module, folder, **worker_options(deadline, cpu, reserve=reserve, mode=player, trace=trace)) as child:
            row['initialization'] = without_result(child.initialization)
            receipt = child.request(payload, c.MOVE_SECONDS, validate=lambda value:
                not response_has_recognized_illegal_action(value, player, observer, legal, trace))
            row['clock'] = without_result(receipt)
        row['initialization_artifact'] = c.artifact(folder/'initialization.json')
        row['receipt'] = c.artifact(folder/'request-00000001.json')
        if receipt['status'] != 'ok':
            if receipt['status'] not in PLAYER_FAILURES:
                raise ValueError('Hard worker or resource failure: '+receipt['status'])
            if receipt['status'] == 'illegal':
                c.require(response_has_recognized_illegal_action(receipt.get('result'), player, observer, legal, trace),
                          'Unrecognized illegal-action receipt; scientific hold')
                row['result_sha256'] = c.digest(receipt['result'])
                if player == 'local': row['trace'] = c.artifact(trace)
            row['status'] = 'forfeit_'+receipt['status']
            return row
        result = receipt['result']; row['result_sha256'] = c.digest(result)
        if player == 'local':
            c.check_wrapper(result, observer, legal, trace)
            row['trace'] = c.artifact(trace)
            action = result['decision']['action']
            after = rules.serialize_snapshot(rules.CorrectedEnv().step(rules.deserialize_snapshot(snapshot), c.deserialize_action(action)))
            audit.check_step(snapshot, action, after)
        else:
            after = check_opponent(result, observer, load_profile(c.EXTERNAL_PROFILE_PATH, c.EXTERNAL_PROFILE_SHA256),
                                   c.EXTERNAL_PROFILE_SHA256, snapshot)
            if after is None:
                row['status'] = 'forfeit_internal_budget_exhausted'; return row
        row.update(status='completed', correct=True, after_sha256=c.digest(after))
        return row
    except WorkerInitializationError as error:
        row['initialization'] = without_result(error.receipt)
        row['status'] = 'initialization_failure'
        raise
    except BaseException as error:
        row['failure_category'] = type(error).__name__
        raise
    finally:
        row['total_seconds'] = time.monotonic()-started
        c.save_new(c.OUTPUT/'decision_attempts'/(name+'.json'), row)


def audit_attempt(name, row, mode, deadline, cpu):
    from tools.matched_strength_20261020.memory import wait_for_memory
    admission = wait_for_memory(c.OUTPUT, deadline, name='audits/'+name, reserve=c.ANALYSIS_SECONDS+c.FINAL_RESERVE, request_seconds=c.AUDIT_SECONDS)
    if deadline-time.monotonic() < c.INIT_SECONDS+c.AUDIT_SECONDS+c.ANALYSIS_SECONDS+c.FINAL_RESERVE:
        raise c.ResourceHold('Insufficient audit/finalization reserve')
    result = completed_result(row); observer = row['observer']
    folder = c.OUTPUT/'audits'/name
    record = {'id': name, 'mode': mode, 'observer_sha256': c.digest(observer),
              'result_sha256': c.digest(result), 'correct': False,
              'memory_admission': c.artifact(c.OUTPUT/'memory_admissions/audits'/(name+'.json'))}
    started = time.monotonic()
    try:
        with PlannerWorker('tools.matched_strength_20261020.worker', folder,
                **worker_options(deadline, cpu, reserve=c.ANALYSIS_SECONDS+c.FINAL_RESERVE, mode='audit')) as child:
            record['initialization'] = without_result(child.initialization)
            receipt = child.request({'observer': observer, 'result': result,
                'traces_path': str((c.ROOT/row['trace']['path']).resolve()), 'mode': mode}, c.AUDIT_SECONDS)
            record['clock'] = without_result(receipt)
        record['initialization_artifact'] = c.artifact(folder/'initialization.json')
        record['receipt'] = c.artifact(folder/'request-00000001.json')
        c.require(receipt['status'] == 'ok', 'Independent continuation audit did not complete')
        c.check_audit_proof(receipt['result'], observer, result, mode)
        record.update(verification=receipt['result'], correct=True)
        return record
    finally:
        record['total_seconds'] = time.monotonic()-started
        c.save_new(c.OUTPUT/'audit_rows'/(name+'.json'), record)


def scientific_projection(decision):
    value = deepcopy(decision)
    value['record'].pop('profile')
    return value


def validate_adapter(deadline, cpu):
    rows, proofs = [], []
    for case in c.read(c.OUTPUT/'validation_cases.json'):
        if deadline-time.monotonic() < 2*c.INIT_SECONDS+c.MOVE_SECONDS+c.AUDIT_SECONDS+c.config()['post_game_reserve_seconds']:
            raise c.ResourceHold('Insufficient validation/operational reserve')
        name = f"validation/v{case['index']:03d}"
        row = decision_attempt(name, 'local', case['snapshot'], case['observer'], deadline, cpu,
                               reserve=c.config()['post_game_reserve_seconds'])
        c.require(row['status'] == 'completed', 'Transfer validation decision failed; games blocked')
        result = completed_result(row)
        if case['mode'] == 'pilot_transfer':
            old = c.read(c.ROOT/case['reference_receipt']['path'])['result']
            c.require(scientific_projection(result['decision']) == scientific_projection(old['decision']),
                      'Pilot scientific transfer mismatch')
            c.require(result['trace'] == old['trace'], 'Pilot full trace transfer mismatch')
        if case['kind'] == 'hidden_variant':
            paired = completed_result(rows[-1])
            c.require(result == paired, 'Hidden-state repeat scientific result mismatch')
        proof = audit_attempt(name, row, 'full', deadline, cpu)
        rows.append(row); proofs.append(proof)
        c.save_new(c.OUTPUT/'validation'/'rows'/f"{case['index']:03d}.json",
                   {'case': case['index'], 'decision': c.artifact(c.OUTPUT/'decision_attempts'/(name+'.json')),
                    'audit': c.artifact(c.OUTPUT/'audit_rows'/(name+'.json')), 'passed': True})
        print(c.canonical({'stage': 'adapter_validation', 'completed': len(rows), 'total': 12,
            'fit': case['fit'], 'mode': case['mode'], 'kind': case['kind'],
            'move_seconds': row['clock']['receipt_elapsed_seconds'], 'full_replay_passed': True}), flush=True)
    summary = {'status': 'passed', 'requests': 12, 'pilot_transfer_requests': 6,
               'new_namespace_requests': 6, 'exact_privacy_pairs': 6, 'all_full_replays': True,
               'worlds_replayed': sum(p['verification']['audited_worlds'] for p in proofs),
               'continuations_replayed': sum(p['verification']['audited_continuations'] for p in proofs)}
    c.save_new(c.OUTPUT/'adapter_validation.json', summary)
    return summary


def play_game(spec, deadline, cpu):
    folder = c.OUTPUT/'sealed'/'games'/f"{spec['game']:03d}"
    env = rules.CorrectedEnv(seed=int(spec['seed'])); state = env.initial_state()
    initial = rules.serialize_snapshot(state); audit.validate_ledger(initial)
    c.save_new(folder/'initial.json', {**spec, 'initial': initial, 'initial_sha256': c.digest(initial)})
    game = {**spec, 'initial_sha256': c.digest(initial), 'moves': [], 'complete': False}
    for ply in range(c.MAX_PLIES):
        if state.terminal: break
        before = rules.serialize_snapshot(state); actor = state.current_player
        player = 'local' if actor == spec['local_seat'] else 'opponent'
        observer = (c.make_observer_payload(before, c.operational_context(spec['game'], state.turn_index))
                    if player == 'local' else rules.observation_payload(state))
        # Per-root lawful redaction remains checked independently of played outcomes.
        alternate = rules.serialize_snapshot(hidden_variant(state))
        alternate_observer = (c.make_observer_payload(alternate, observer['context']) if player == 'local'
                              else rules.observation_payload(rules.deserialize_snapshot(alternate)))
        c.require(observer == alternate_observer, 'Actual-game observation privacy mismatch')
        name = f"games/g{spec['game']:03d}/t{ply:04d}"
        row = {'game': spec['game'], 'pair': spec['pair'], 'fit': spec['fit'], 'ply': ply,
               'actor': actor, 'before_sha256': c.digest(before),
               **decision_attempt(name, player, before, observer, deadline, cpu, reserve=c.config()['post_game_reserve_seconds'])}
        if row['status'] == 'completed':
            result = completed_result(row)
            action = result['decision']['action'] if player == 'local' else result['action']
            state = env.step(state, c.deserialize_action(action))
            after = rules.serialize_snapshot(state)
            c.require(c.digest(after) == row['after_sha256'], 'Actual referee successor mismatch')
            audit.check_step(before, action, after)
        else:
            c.require(row['status'].startswith('forfeit_'), 'Unscored hard failure')
            game.update(complete=True, reason=row['status'], winner=1-actor, failing_player=player)
        path = folder/'moves'/f'{ply:04d}.json'; c.save_new(path, row); game['moves'].append(c.artifact(path))
        print(c.canonical({'stage': 'strength_games', 'game': spec['game'], 'ply': ply,
            'player': player, 'decision_status': row['status'],
            'move_seconds': row['clock'].get('receipt_elapsed_seconds'), 'outcomes_closed': True}), flush=True)
        if game['complete']: break
    if not game['complete']:
        if not state.terminal:
            c.save_new(folder/'unresolved.json', {**game, 'final': rules.serialize_snapshot(state), 'reason': 'ply_ceiling'})
            raise c.ResourceHold('Nonterminal maximum-ply ceiling; no invented game result')
        game.update(complete=True, reason='natural', winner=state.winner)
    game['final'] = rules.serialize_snapshot(state)
    c.save_new(folder/'result.json', game)
    return game


def make_audit_manifest():
    entries = []
    for game in range(2*c.config()['pair_count']):
        folder = c.OUTPUT/'sealed'/'games'/f'{game:03d}'
        for path in sorted((folder/'moves').glob('*.json')):
            row = c.read(path)
            if row['player'] != 'local' or row['status'] != 'completed': continue
            result = completed_result(row); identity = row['observer']['context']['identity']
            support = len(result['decision']['record']['candidates']['actions'])
            entries.append({'id': f"g{game:03d}-t{row['ply']:04d}", 'game': game, 'ply': row['ply'],
                'fit': row['fit'], 'identity': identity, 'observer_sha256': c.digest(row['observer']),
                'move': c.artifact(path), 'receipt': row['receipt'], 'trace': row['trace'],
                'candidate_support': support, 'mode': 'sample', 'world_indices': c.sample_worlds(identity),
                'sample_world_indices': c.sample_worlds(identity),
                'sample_hashes': [{'world_index': wi, 'sha256': c.digest([c.AUDIT_DOMAIN, 'world', c.AUDIT_SEED, identity, wi])}
                                  for wi in c.sample_worlds(identity)],
                'stress_priority_sha256': c.stress_priority(identity),
                'stress_cells': c.stress_cells(row['observer'], support)})
    cells = {}
    for cell in c.expected_stress_cells():
        eligible = [e for e in entries if cell in e['stress_cells']]
        chosen = min(eligible, key=lambda e: (c.stress_priority(e['identity']), e['game'], e['ply'])) if eligible else None
        cells[cell] = {'eligible': [e['id'] for e in eligible], 'selected': chosen['id'] if chosen else None}
    full = sorted({cell['selected'] for cell in cells.values() if cell['selected'] is not None})
    for entry in entries:
        if entry['id'] in full: entry.update(mode='full', world_indices=list(range(128)))
    return {'schema': 'operational_continuation_manifest_v1', 'entries': entries,
            'cells': cells, 'full_stress_ids': full}


def run(expected_freeze_sha256):
    base = c.OUTPUT
    c.require(c.file_hash(base/'freeze.json') == expected_freeze_sha256, 'Expected prepared freeze identity')
    freeze = c.read(base/'freeze.json'); c.verify(freeze['inputs'])
    c.require(Path(sys.executable).resolve() == Path(freeze['executable']).resolve(),
              'Qualified Python executable required')
    from tools.matched_strength_20261020.statistics import runtime_contract
    c.require(c.read(c.PREPARATION/'statistics_runtime.json') == runtime_contract(),
              'Frozen statistical runtime changed before launch')
    c.require(processor_topology() == freeze['topology'], 'Original full supervisor topology required')
    c.require(not (base/'launch.json').exists(), 'Never restart a launched strength attempt')
    started = time.monotonic(); deadline = started+c.config()['execution_cap_seconds']
    c.save_new(base/'launch.json', {'time': c.now(), 'epoch_seconds': time.time(), 'monotonic_start': started,
        'pid': os.getpid(), 'identity': {'pid': os.getpid(), 'created': psutil.Process().create_time()},
        'freeze': c.artifact(base/'freeze.json'), 'maximum_seconds': c.config()['execution_cap_seconds'],
        'torch_loaded_in_supervisor': 'torch' in sys.modules, 'outcomes_closed': True})
    games, validation, failure = [], None, None
    try:
        from tools.matched_strength_20261020.memory import wait_for_memory
        wait_for_memory(base, deadline, name='initial', reserve=c.FINAL_RESERVE, request_seconds=0, initial=True)
        cpu = freeze['topology']['selected_logical_processor']
        validation = validate_adapter(deadline, cpu)
        c.verify(freeze['inputs'])
        c.save_new(base/'games_admission.json', {'time': c.now(), 'adapter_validation': c.artifact(base/'adapter_validation.json'),
            'remaining_seconds': deadline-time.monotonic(), 'outcomes_closed': True})
        for spec in c.read(base/'sealed/schedule.json'):
            games.append(play_game(spec, deadline, cpu))
            print(c.canonical({'stage': 'strength_games', 'completed_games': len(games), 'total': 2*c.config()['pair_count'],
                               'outcomes_closed': True}), flush=True)
        manifest = make_audit_manifest(); c.save_new(base/'audit_manifest.json', manifest)
        for index, entry in enumerate(manifest['entries']):
            row = c.read(c.ROOT/entry['move']['path'])
            audit_attempt(entry['id'], row, entry['mode'], deadline, cpu)
            print(c.canonical({'stage': 'continuation_verification', 'completed_roots': index+1,
                'total_roots': len(manifest['entries']), 'mode': entry['mode'], 'outcomes_closed': True}), flush=True)
        c.boundary(base, deadline, memory_headroom=False)
    except BaseException as error:
        failure = {'category': type(error).__name__, 'detail_path': 'sealed/failure.json'}
        c.save_new(base/'sealed'/'failure.json', {'time': c.now(), 'traceback': traceback.format_exc()})
    try:
        c.verify(freeze['inputs'])
        c.save_new(base/'final_input_check.json', {'time': c.now(), 'all_hashes_and_sizes_match': True,
                                                'files': len(freeze['inputs'])})
        c.boundary(base, deadline, memory_headroom=False)
    except BaseException as error:
        failure = failure or {'category': type(error).__name__, 'detail_path': 'sealed/final_check_failure.json'}
        c.save_new(base/'sealed'/'final_check_failure.json', {'time': c.now(), 'traceback': traceback.format_exc()})
    status = 'completed_pending_independent_analysis' if failure is None else 'preserved_hold'
    result = {'time': c.now(), 'status': status, 'failure': failure, 'validation_passed': validation is not None,
        'completed_games': len(games), 'elapsed_seconds': time.monotonic()-started,
        'outcomes_closed': True, 'strength_analysis_games': 0, 'larger_data_deferred': True}
    c.save_new(base/'operational_progress.json', result)
    c.save_new(base/'completion.json', {**result, 'worker_processes_closed': True,
                                      'progress': c.artifact(base/'operational_progress.json')})
    print(c.canonical(result), flush=True)
    if failure is not None:
        return 1
    from tools.matched_strength_20261020.completion import complete
    return complete(deadline)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(); parser.add_argument('action', choices=('prepare', 'run')); parser.add_argument('--config', type=Path)
    parser.add_argument('--freeze-sha256')
    args = parser.parse_args()
    if args.action == 'prepare':
        if args.config is None: parser.error('prepare requires an explicit prospective --config')
        prepare(args.config)
    else:
        if args.freeze_sha256 is None: parser.error('run requires the reviewed --freeze-sha256')
        with c.execution_lease(): raise SystemExit(run(args.freeze_sha256))
