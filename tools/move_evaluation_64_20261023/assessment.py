"""Parent-only fresh-world assessment, sealed interventions and complete audit.

This module loads no models. Policy calls occur only in supervised observer-only
workers. All exceptions preserve partial evidence and stop; no outcomes are
imputed, retries are not made, and the throughput gate never consumes outcomes.
"""
from collections import Counter
from functools import lru_cache
import importlib.util
from pathlib import Path
import math
import random
import time

from . import common as c
from tools.external_strength_screen_20261018 import rules, audit as dictionary
from tools.future_training_rules_20261021 import decode_training_snapshot, require_official_terminal_target

DOMAIN = 'move_evaluation_64_20261023_assessment_v1'
FITS = (161001, 161002, 161003)
SCIENTIFIC_FIELDS = ('initial', 'forced_action', 'root_actor', 'steps', 'final', 'score', 'reason')


def external_observer(observer):
    return {k: rules.OBSERVATION_SCHEMA if k == 'schema' else v
            for k, v in observer.items() if k != 'context'}


def seed_for(root_id, world_index):
    c.require(type(root_id) is str and type(world_index) is int and 0 <= world_index < 32,
              'Assessment root/world identity')
    return int(c.digest([DOMAIN, root_id, world_index]), 16)


def audit_assessment_initial(observer, seed, snapshot):
    from tools.move_evaluation_20261023 import audit as reference_audit
    from tools.corrected_planner_qualification_20261018 import verify as reference
    ctx = observer['context']
    reference_audit.check_payload({'observer': observer, 'fit': ctx['fit'], 'candidates': {}, 'selection_schedule': {}})
    root_id = f"games/g{ctx['game']:03d}/t{ctx['turn']:04d}"
    seeds = [seed_for(root_id, j) for j in range(32)]
    c.require(type(seed) is int and seed >= 0 and seed in seeds, 'Fresh prescribed integer assessment seed')
    c.require(str(seed) not in reference.schedule(observer)['world_seeds'], 'Assessment/selection collision')
    expected = reference.world_from_observer(observer, seed)
    c.require(c.digest(snapshot) == c.digest(expected), 'Independent assessment initial world differs')
    return {'status': 'passed', 'root_id': root_id, 'world_index': seeds.index(seed), 'world_seed': seed,
            'domain': DOMAIN, 'initial_sha256': c.digest(expected), 'observer_sha256': c.digest(observer)}


def validate_choices(locked, choices):
    roots = locked['roots']
    fields = {'id', 'fit', 'control_choice', 'alternative_choice', 'control_action', 'alternative_action'}
    c.require(type(roots) is list and type(choices) is list and len(roots) == len(choices) == 18 and
              len({r['id'] for r in roots}) == 18 and
              Counter(r['fit'] for r in roots) == Counter({f: 6 for f in FITS}), 'Fixed balanced sealed panel')
    changed = []
    for row, choice in zip(roots, choices, strict=True):
        c.require(type(choice) is dict and set(choice) == fields and
                  (choice['id'], choice['fit']) == (row['id'], row['fit']) and type(choice['fit']) is int,
                  'Ordered sealed choice identity')
        actions = row['candidates']['actions']
        c.require(len({c.digest(a) for a in actions}) == len(actions), 'Duplicate candidate representatives')
        for prefix in ('control', 'alternative'):
            index = choice[prefix+'_choice']
            c.require(type(index) is int and 0 <= index < len(actions) and
                      c.digest(choice[prefix+'_action']) == c.digest(actions[index]), 'Sealed action/index mismatch')
        c.require(type(row['control_choice']) is int and row['control_choice'] == choice['control_choice'] and
                  c.digest(row['control_action']) == c.digest(choice['control_action']), 'Runtime lock must use sealed A64')
        c.require(row['assessment_world_seeds_parent_only'] == [str(seed_for(row['id'], j)) for j in range(32)],
                  'Fixed new-domain parent-only assessment schedule')
        c.require(not set(row['assessment_world_seeds_parent_only']) & set(row['selection_schedule']['world_seeds']),
                  'Selection/assessment seeds collide')
        if choice['control_choice'] != choice['alternative_choice']:
            changed.append(row)
    return changed


def sealed_admission(locked, choices, out):
    changed = validate_choices(locked, choices)
    seal = out/'sealed/choices.json'
    lock_path = out/'sealed/evaluated_parent_lock.json'
    c.require(c.digest(c.read(seal)) == c.digest(choices), 'Caller choices differ from immutable seal')
    c.require(c.digest(c.read(lock_path)) == c.digest(locked), 'Caller parent lock differs from immutable seal')
    qualification = c.read(out/'qualification_complete.json')
    c.require(qualification['status'] == 'passed' and qualification['choices'] == c.artifact(seal) and
              qualification['lock'] == c.artifact(lock_path),
              'All qualification gates and choice binding required before assessment')
    return changed


def qualify_assessor(envelope, freeze, make_worker, request):
    """Exercise both actual assessor routes on the fixed 15 corrected fixtures."""
    from tools.analyze_external_strength_screen_20261018_02 import check_clock, check_diagnostics
    fixtures = c.read(c.OUT/'fixtures.json')
    c.require(type(fixtures) is list and len(fixtures) == 15 and
              len({f['id'] for f in fixtures}) == 15, 'All fifteen assessor fixtures required')
    c.require(c.file_hash(c.PROFILE) == c.PROFILE_SHA, 'PUCT16 profile changed')
    profile, records, terminal = c.read(c.PROFILE), [], []
    expected_fields = {'action', 'diagnostics', 'profile_sha256', 'player', 'observer_sha256',
                       'legal_actions_sha256', 'legal_count', 'ruleset', 'catalogue_sha256'}
    with make_worker('qualify_assessor', 'assess', envelope, envelope.qual_end, freeze) as child:
        for fixture in fixtures:
            envelope.guard(envelope.qual_end)
            snapshot = fixture['snapshot']
            state = decode_training_snapshot(snapshot)
            dictionary.validate_ledger(snapshot)
            if state.terminal:
                winner = require_official_terminal_target(snapshot, 'completed')
                terminal.append({'fixture': fixture['id'], 'snapshot_sha256': c.digest(snapshot),
                                 'official_terminal': True, 'winner': winner})
                continue
            observer = rules.observation_payload(state)
            dictionary.check_observer(snapshot, observer)
            legal = [rules.serialize_action(a) for a in rules.legal_actions(state)]
            for player in ('raw035', 'opponent'):
                receipt = request(child, {'observer': observer, 'player': player}, 1., envelope.qual_end)
                check_clock(receipt, receipt['request_id'], 1., 0)
                result = receipt['result']
                c.require(receipt['status'] == 'ok' and set(result) == expected_fields and
                          result['player'] == player and result['observer_sha256'] == c.digest(observer) and
                          result['legal_actions_sha256'] == c.digest(legal) and
                          type(result['legal_count']) is int and result['legal_count'] == len(legal) and
                          result['ruleset'] == rules.RULESET_ID and
                          result['catalogue_sha256'] == rules.CATALOGUE_SHA256, 'Assessor response identity')
                action = result['action']
                c.require(action in legal, 'Assessor fixture illegal action')
                if player == 'opponent':
                    c.require(result['profile_sha256'] == c.PROFILE_SHA, 'Assessor PUCT16 profile')
                    work = dict(check_diagnostics(result['diagnostics'], observer, action, legal, profile))
                else:
                    diag = result['diagnostics']
                    c.require(result['profile_sha256'] is None and
                              set(diag) == {'policy', 'scores_sha256', 'selected_index', 'legal_count'} and
                              diag['policy'] == 'raw035' and type(diag['selected_index']) is int and
                              diag['selected_index'] == legal.index(action) and diag['legal_count'] == len(legal) and
                              type(diag['scores_sha256']) is str and len(diag['scores_sha256']) == 64 and
                              all(ch in '0123456789abcdef' for ch in diag['scores_sha256']), 'Assessor raw035 diagnostics')
                    work = {'raw035_decisions': 1}
                successor = rules.CorrectedEnv().step(state, rules.deserialize_action(action))
                after = rules.serialize_snapshot(successor)
                dictionary.check_step(snapshot, action, after)
                records.append({'fixture': fixture['id'], 'player': player,
                    'observer_sha256': c.digest(observer), 'before_sha256': c.digest(snapshot),
                    'after_sha256': c.digest(after), 'action': action, 'work': work,
                    'receipt': c.artifact(child.output_dir/(receipt['request_id']+'.json'))})
    c.require(len(records) == 2*(15-len(terminal)), 'Assessor fixture route coverage')
    envelope.guard(envelope.qual_end)
    return c.save(c.OUT/'assessor_qualification.json', {'status': 'passed', 'fixtures': 15,
        'fixture_input': c.artifact(c.OUT/'fixtures.json'), 'records': records, 'terminal_proofs': terminal,
        'complete_move_limit_seconds': 1., 'actual_puct_internal_edges_replayed': False,
        'ruleset_id': rules.RULESET_ID, 'catalogue_sha256': rules.CATALOGUE_SHA256})


@lru_cache(maxsize=1)
def _observer_sampler():
    # The bots package initializer imports Torch. Load the exact frozen pure
    # sampler source without that initializer; the execution freeze binds it.
    path = c.ROOT/'src/splendor_ai/bots/hidden_information.py'
    spec = importlib.util.spec_from_file_location('splendor_64_parent_hidden_information', path)
    c.require(spec is not None and spec.loader is not None, 'Frozen observer sampler unavailable')
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module.determinize_for_observer


def initial_assessment_world(row, j):
    seed = seed_for(row['id'], j)
    c.require(row['assessment_world_seeds_parent_only'][j] == str(seed), 'Parent assessment seed mismatch')
    state = rules.state_from_observation(external_observer(row['observer']))
    world = _observer_sampler()(state, state.current_player, random.Random(seed))
    snapshot = rules.serialize_snapshot(world)
    audit_assessment_initial(row['observer'], seed, snapshot)
    return decode_training_snapshot(snapshot)


def play_branch(child, row, choice, j, arm, envelope, request, *, out=None):
    out = c.OUT if out is None else Path(out)
    c.require(arm in ('A', 'B'), 'Assessment arm')
    started = time.monotonic()
    deadline = min(started+120., envelope.effective_deadline(envelope.collection_end), envelope.end)
    envelope.guard(deadline)
    state = initial_assessment_world(row, j)
    c.require(not state.terminal, 'Assessment root unexpectedly terminal')
    root_actor = state.current_player
    initial = rules.serialize_snapshot(state)
    forced = choice['control_action' if arm == 'A' else 'alternative_action']
    action, steps = forced, []
    folder = out/'assessment'/row['id'].replace('/', '_')/f'w{j:02d}_{arm}'
    c.save(folder/'initial.json', {'initial': initial, 'forced_action': forced, 'root_actor': root_actor,
                                 'id': row['id'], 'world_index': j, 'arm': arm})
    while not state.terminal:
        envelope.guard(deadline)
        c.require(state.turn_index < 400, 'Assessment turn ceiling; unresolved')
        before = rules.serialize_snapshot(state)
        observer = rules.observation_payload(state)
        dictionary.check_observer(before, observer)
        receipt = received = None
        if not steps:
            player = 'forced'
        else:
            player = 'raw035' if state.current_player == root_actor else 'opponent'
            received = request(child, {'observer': observer, 'player': player}, 1., deadline)
            action = received['result']['action']
            receipt = c.artifact(child.output_dir/(received['request_id']+'.json'))
        envelope.guard(deadline)
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
    finished = time.monotonic()
    branch = {'id': row['id'], 'fit': row['fit'], 'world_index': j, 'arm': arm,
              'initial': initial, 'forced_action': forced, 'root_actor': root_actor, 'steps': steps,
              'final': final, 'score': score, 'reason': 'completed', 'elapsed_seconds': finished-started,
              'branch_started_monotonic': started, 'branch_finished_monotonic': finished,
              'branch_deadline_monotonic': deadline}
    artifact = c.save(folder/'branch.json', branch)
    # The complete branch includes durable evidence writing, not only policy time.
    envelope.guard(deadline)
    c.save(folder/'completion.json', {'status': 'completed', 'branch': artifact,
           'finished_monotonic': time.monotonic(), 'elapsed_seconds': time.monotonic()-started})
    envelope.guard(deadline)
    return branch


def throughput_gate(*, elapsed_seconds, added_bytes, current_bytes, remaining_seconds):
    c.require(all(type(v) in (int, float) and math.isfinite(v) for v in (elapsed_seconds, remaining_seconds)) and
              type(added_bytes) is int and type(current_bytes) is int and
              elapsed_seconds > 0 and added_bytes >= 0 and current_bytes >= added_bytes,
              'Invalid throughput accounting')
    seconds = elapsed_seconds*31*1.35
    evidence = current_bytes+added_bytes*31*1.35
    passed = seconds < remaining_seconds and evidence < 2*c.GiB
    return {'status': 'passed' if passed else 'hold', 'elapsed_seconds': elapsed_seconds,
            'added_bytes': added_bytes, 'worlds_per_changed_root': 1,
            'remaining_seconds_projection_35pct': seconds,
            'evidence_bytes_projection_35pct': evidence, 'outcomes_inspected_for_gate': False}


def run_assessment(envelope, freeze, locked, choices, make_worker, request, *, out=None, bytes_used=None):
    out = c.OUT if out is None else Path(out)
    if bytes_used is None:
        bytes_used = lambda: sum(p.stat().st_size for p in out.rglob('*') if p.is_file())
    changed = sealed_admission(locked, choices, out)
    selected = {r['id']: r for r in choices}
    breadth = {'changed_roots': len(changed), 'by_fit': {str(f): sum(r['fit'] == f for r in changed) for f in FITS}}
    c.save(out/'breadth.json', breadth)
    if len(changed) < 6 or min(breadth['by_fit'].values()) < 1:
        return {'status': 'insufficient_intervention_breadth', 'breadth': breadth, 'assessment_started': False}
    envelope.guard(envelope.collection_end)
    wave_started, wave_bytes = time.monotonic(), bytes_used()
    with make_worker('assessment', 'assess', envelope, envelope.collection_end, freeze) as child:
        for phase, worlds in [('wave0', range(1)), ('remaining', range(1, 32))]:
            for index, row in enumerate(changed):
                envelope.admit(envelope.collection_end)
                for j in worlds:
                    for arm in ('A', 'B'):
                        play_branch(child, row, selected[row['id']], j, arm, envelope, request, out=out)
                print(c.canonical({'stage': 'assessment', 'phase': phase,
                      'complete_roots': index+1, 'changed_roots': len(changed)}), flush=True)
            if phase == 'wave0':
                envelope.guard(envelope.collection_end)
                current_bytes = bytes_used()
                gate = throughput_gate(elapsed_seconds=time.monotonic()-wave_started,
                    added_bytes=current_bytes-wave_bytes, current_bytes=current_bytes,
                    remaining_seconds=envelope.effective_deadline(envelope.collection_end)-time.monotonic())
                c.save(out/'throughput_gate.json', gate)
                c.require(gate['status'] == 'passed', 'Outcome-blind throughput projection exceeds fixed allowance')
                envelope.guard(envelope.collection_end)
    envelope.guard(envelope.collection_end)
    return {'status': 'assessment_collected', 'changed_roots': len(changed), 'branches': len(changed)*64}


def validate_branch_files(path, branch, row, choice, j, arm, envelope, seen_receipts):
    c.require((branch['id'], branch['fit'], branch['world_index'], branch['arm']) ==
              (row['id'], row['fit'], j, arm), 'Assessment branch identity')
    c.require(c.digest(branch['forced_action']) == c.digest(choice['control_action' if arm == 'A' else 'alternative_action']),
              'Assessment intervention differs from sealed A64/B64')
    started, finished, deadline = (branch[k] for k in ('branch_started_monotonic', 'branch_finished_monotonic',
                                                     'branch_deadline_monotonic'))
    c.require(0 <= branch['elapsed_seconds'] <= 120 and started < finished <= deadline <= envelope.collection_end and
              deadline <= envelope.end and deadline-started <= 120 and branch['elapsed_seconds'] == finished-started,
              'Assessment branch clock')
    completed = c.read(path.parent/'completion.json')
    c.require(completed['status'] == 'completed' and completed['branch'] == c.artifact(path) and
              finished <= completed['finished_monotonic'] <= deadline and
              branch['elapsed_seconds'] <= completed['elapsed_seconds'] <= 120, 'Durable branch completion clock/binding')
    expected_initial = {'initial': branch['initial'], 'forced_action': branch['forced_action'],
                        'root_actor': branch['root_actor'], 'id': row['id'], 'world_index': j, 'arm': arm}
    c.require(c.digest(c.read(path.parent/'initial.json')) == c.digest(expected_initial), 'Initial evidence differs')
    paths = sorted((path.parent/'steps').glob('*.json'))
    c.require(len(paths) == len(branch['steps']) and paths, 'Branch step file coverage')
    for index, (step_path, step) in enumerate(zip(paths, branch['steps'], strict=True)):
        c.require(step_path.name == f'{index:04d}.json' and c.digest(c.read(step_path)) == c.digest(step),
                  'Branch persisted step differs')
        if index:
            identity = step['receipt']['path']
            c.require(identity not in seen_receipts, 'Assessment response receipt reused across moves/branches')
            seen_receipts.add(identity)


def analyze(envelope, freeze, locked, choices, make_worker, request, *, out=None):
    out = c.OUT if out is None else Path(out)
    changed = sealed_admission(locked, choices, out)
    c.require(len(changed) >= 6 and all(any(r['fit'] == f for r in changed) for f in FITS), 'Assessment breadth missing')
    selected = {r['id']: r for r in choices}
    rows, verified_branches, seen_receipts = [], 0, set()
    with make_worker('audit_assessment', 'audit_branch', envelope, envelope.end, freeze) as child:
        for row in locked['roots']:
            choice = selected[row['id']]
            item = {k: choice[k] for k in ('id', 'fit', 'control_choice', 'alternative_choice')}
            if choice['control_choice'] == choice['alternative_choice']:
                item.update(a_scores_halfpoints=None, b_scores_halfpoints=None, differences_halfpoints=[0]*32)
            else:
                scores = {'A': [], 'B': []}
                for j in range(32):
                    for arm in ('A', 'B'):
                        envelope.guard(envelope.end)
                        path = out/'assessment'/row['id'].replace('/', '_')/f'w{j:02d}_{arm}'/'branch.json'
                        branch = c.read(path)
                        validate_branch_files(path, branch, row, choice, j, arm, envelope, seen_receipts)
                        scientific = {k: branch[k] for k in SCIENTIFIC_FIELDS}
                        receipt = request(child, {'branch': scientific, 'observer': row['observer'],
                            'world_seed': seed_for(row['id'], j)}, 120., envelope.end)
                        proof = receipt['result']
                        c.require(proof['branch']['official_terminal'] is True and
                                  proof['branch']['score_halfpoints'] == int(2*branch['score']), 'Official audited score differs')
                        c.save(path.parent/'audit.json', proof)
                        scores[arm].append(proof['branch']['score_halfpoints'])
                        verified_branches += 1
                item.update(a_scores_halfpoints=scores['A'], b_scores_halfpoints=scores['B'],
                            differences_halfpoints=[b-a for a, b in zip(scores['A'], scores['B'], strict=True)])
            rows.append(item)
    c.require(verified_branches == len(changed)*64, 'Every required branch must be verified')
    c.save(out/'analysis_rows.json', rows)
    with make_worker('statistics', 'statistics', envelope, envelope.end, freeze) as child:
        result = request(child, {'rows': rows, 'candidate_lock': locked}, 600., envelope.end)['result']
    c.save(out/'independent_analysis.json', result)
    envelope.guard(envelope.end)
    return {'status': 'completed', 'assessment_branches_verified': verified_branches, 'analysis': result}
