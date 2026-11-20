"""Frozen, conditional 30-pair external PPO development screen."""
from __future__ import annotations

import argparse
from collections import Counter
from copy import deepcopy
import hashlib
import json
import math
import os
from pathlib import Path
import random
import subprocess
import sys
import time
import traceback

from tools.external_strength_screen_20261018 import common as c, rules, audit
from tools.local_player_qualification_128.clock import PersistentWorker, processor_topology

BASE = c.ROOT / 'outputs/external_strength_screen_20261018_01'
PRIMARY = c.ROOT / 'outputs/component_adjudication_20261018_01'
PRIOR = c.ROOT / 'outputs/local_player_qualification_20261018_recovery_01'
NAMESPACE = 'splendor_external_corrected_rules_20261018_development_v1'
BOOTSTRAP_SEED = 2026101803510
PLAYER_FAILURES = {'timeout', 'worker_exit', 'worker_error', 'malformed', 'wrong_request',
                   'illegal', 'unsolicited', 'write_error'}
TESTS = ['tests/test_external_screen_rules_20261018.py',
         'tests/test_external_screen_ppo_20261018.py',
         'tests/test_external_screen_audit_20261018.py',
         'tests/test_external_screen_runner_20261018.py']


def make_schedule():
    rows = []
    for pair in range(30):
        seed = int.from_bytes(hashlib.sha256(f'{NAMESPACE}:pair:{pair}'.encode()).digest(), 'big')
        for order in range(2):
            rows.append({'game': len(rows), 'pair': pair, 'seed': str(seed),
                         'ppo_seat': (pair + order) % 2})
    return rows


def seed_inventory(base, schedule):
    command = ['rg', '--files', '--no-ignore', 'outputs', '-g', 'manifest.json',
               '-g', 'schedule.json', '-g', '*seed_registry*.json']
    found = subprocess.run(command, cwd=c.ROOT, text=True, capture_output=True, check=True)
    target = {int(row['seed']) for row in schedule}
    checked, skipped, collisions = [], [], []
    def walk(value, path):
        if type(value) is int and value in target:
            collisions.append(path)
        elif isinstance(value, str):
            if value.isdecimal() and int(value) in target:
                collisions.append(path)
        elif isinstance(value, list):
            for v in value: walk(v, path)
        elif isinstance(value, dict):
            for v in value.values(): walk(v, path)
    for name in sorted(found.stdout.splitlines()):
        p = c.ROOT / name
        if p.resolve().is_relative_to(base.resolve()): continue
        if p.stat().st_size > 8 * 1024**2:
            skipped.append(c.artifact(p)); continue
        walk(c.read(p), name)
        checked.append(c.artifact(p))
    if collisions or len(target) != 30:
        raise ValueError('Seed collision')
    return {'namespace': NAMESPACE, 'derivation': 'full SHA256 integer per pair',
            'distinct_initial_seeds': 30, 'collisions': collisions,
            'scope': 'Every integer/decimal-string in discovered historical manifests/schedules/seed registries <=8MiB; not all historical game logs',
            'checked': checked, 'skipped_oversize': skipped}


def prepare(base):
    base.mkdir(parents=True, exist_ok=True)
    if (base / 'freeze.json').exists() or (base / 'launch.json').exists():
        raise ValueError('Preparation already frozen; never overwrite')
    adapter_receipt = c.read(base / 'qualification/ppo_profile_03.json')
    expected_qualification = {'status': 'passed_fixture_qualification', 'ppo_qualified': True,
        'profile': c.PPO_PROFILE, 'checkpoint_sha256': c.PPO_CHECKPOINT_SHA256,
        'ruleset_id': rules.RULESET_ID, 'catalogue_sha256': rules.CATALOGUE_SHA256}
    if any(adapter_receipt.get(k) != v for k, v in expected_qualification.items()):
        raise ValueError('Selected PPO profile is not qualified')
    primary = c.read(PRIMARY / 'independent_component_adjudication.json')
    if primary['counts']['upstream_cards_matching_bga'] != 90 or primary['counts']['local_and_upstream_nobles_matching_bga'] != 10:
        raise ValueError('Primary component qualification failed')
    c.verify(primary['sources'])
    carddb = c.read(PRIMARY / 'bga_tutorial_carddb.json')
    colors = primary['color_decoding']['letter_to_color']
    types = primary['color_decoding']['numeric_type_to_letter']
    corrected = rules.corrected_cards_by_id()
    if set(corrected) != set(primary['full_proposed_local_to_bga_join']):
        raise ValueError('Corrected component identifiers differ')
    for card_id, bga_id in primary['full_proposed_local_to_bga_join'].items():
        actual, source = corrected[card_id], carddb[bga_id]
        face = (source['lvl'], colors[types[str(source['type'])]], source['points'],
                dict(Counter(colors[letter] for letter in source['cost'])))
        if (actual.tier, actual.bonus_color, actual.points, actual.cost) != face:
            raise ValueError('Corrected referee face differs from licensed source: ' + card_id)
    c.verify(c.read(PRIOR / 'freeze.json')['inputs'])
    old = c.frozen_source_check()
    if old['failures']: raise ValueError('Historical inputs changed')
    c.save_new(base / 'original_source_check.json', old)
    selection = c.read(PRIOR / 'selection.json')
    fixtures = []
    for item in selection['core'] + selection['edges']:
        root = c.read(c.ROOT / item['path'])
        fixture = rules.convert_legacy_snapshot(root['snapshot'])
        audit.validate_ledger(fixture)
        fixtures.append({'source': item, 'snapshot': fixture,
                         'meaning': 'Counterfactual corrected-rules fixture; not a replay of the historical game'})
    c.save_new(base / 'fixtures.json', fixtures)
    schedule = make_schedule()
    inventory = seed_inventory(base, schedule)
    c.save_new(base / 'seed_collision_check.json', inventory)
    c.save_new(base / 'schedule.json', schedule)
    c.save_new(base / 'catalogue.json', rules.catalogue_document())
    test = subprocess.run([sys.executable, '-B', '-m', 'pytest', '-q', *TESTS, '-p', 'no:cacheprovider'],
        cwd=c.ROOT, env={**os.environ, 'PYTHONDONTWRITEBYTECODE': '1'}, capture_output=True,
        text=True, timeout=300, creationflags=subprocess.CREATE_NO_WINDOW if os.name == 'nt' else 0)
    c.save_new(base / 'preflight_tests.json', {'returncode': test.returncode, 'stdout': test.stdout, 'stderr': test.stderr})
    if test.returncode: raise ValueError('Preflight failed; preserve report')
    paths = {v['path'] for v in c.read(PRIOR / 'freeze.json')['inputs']}
    paths.update(v['path'] for v in inventory['checked'])
    paths.update(TESTS)
    for directory in [c.ROOT / 'tools/external_strength_screen_20261018', PRIMARY,
                      c.ROOT / 'outputs/external_strength_preparation_20261018_01/upstream',
                      c.ROOT / 'outputs/external_strength_preparation_20261018_01/qualification_sources']:
        paths.update(p.relative_to(c.ROOT).as_posix() for p in directory.rglob('*')
                     if p.is_file() and '__pycache__' not in p.parts)
    paths.add('tools/adjudicate_splendor_components_20261018.py')
    paths.add('docs/engineering/external_strength_screen_20261018.md')
    paths.add('outputs/local_player_qualification_20261018_recovery_01/freeze.json')
    for name in ('fixtures.json', 'seed_collision_check.json', 'schedule.json', 'catalogue.json', 'preflight_tests.json'):
        paths.add((base / name).relative_to(c.ROOT).as_posix())
    # Pinned model/source and all finalized adapter qualification receipts are part of the identity.
    qdir = base / 'qualification'
    if qdir.exists():
        paths.update(p.relative_to(c.ROOT).as_posix() for p in qdir.rglob('*') if p.is_file())
    topology = processor_topology()
    if topology['api'] != 'GetLogicalProcessorInformationEx(RelationProcessorCore)' or len(topology['distinct_efficiency_classes']) < 2:
        raise ValueError('Verified performance-core topology required')
    c.save_new(base / 'freeze.json', {'time': c.now(), 'ruleset_id': rules.RULESET_ID,
        'catalogue_sha256': rules.CATALOGUE_SHA256, 'inputs': [c.artifact(c.ROOT / p) for p in sorted(paths)],
        'topology': topology, 'games': 60, 'pairs': 30, 'seconds_per_move': 1,
        'initialization_seconds': 60, 'maximum_plies': 2000, 'execution_cap_seconds': 14400,
        'bootstrap': {'replicates': 50000, 'seed': BOOTSTRAP_SEED, 'unit': 'paired initial world', 'percentiles': [.025, .975], 'interpolation': 'linear'},
        'players': ['adapted pinned roeey777 PPO', 'raw frozen035 transferred without training to corrected rules'],
        'ppo_profile': c.PPO_PROFILE, 'ppo_qualification': c.artifact(base / 'qualification/ppo_profile_03.json'),
        'screen_gate': 'All preflight/source checks and every one-second dynamic qualification decision must pass',
        'larger_data_deferred': True})
    print(c.canonical({'prepared': str(base), 'fixtures': len(fixtures), 'games_conditional': 60}), flush=True)


def worker(base, name, deadline, cpu):
    if 'torch' in sys.modules: raise ValueError('Referee must not import torch')
    c.boundary(base, deadline)
    if deadline - time.monotonic() < 120: raise c.ResourceHold('Insufficient initialization/finalization reserve')
    return PersistentWorker('tools.external_strength_screen_20261018.worker', base / 'workers' / name,
        affinity=[cpu], initialization_timeout=60, absolute_deadline=deadline - 60,
        env={'PYTHONPATH': str(c.ROOT / 'src') + os.pathsep + str(c.ROOT), 'CUDA_VISIBLE_DEVICES': ''},
        max_message_bytes=256 * 1024, max_log_bytes=16 * 1024**2, max_rss_bytes=2 * 1024**3,
        resource_guard=lambda: c.boundary(base, deadline, memory_headroom=False))


def decision(child, state, player):
    snapshot = rules.serialize_snapshot(state)
    payload = rules.observation_payload(state)
    audit.check_observer(snapshot, payload)
    legal = rules.CorrectedEnv().legal_actions(state)
    expected = [c.serialize_action(a) for a in legal]
    receipt = child.request({'player': player, 'observer': payload}, 1.0,
        validate=lambda value: isinstance(value, dict) and value.get('action') in expected)
    result = receipt.get('result')
    if receipt['status'] == 'ok':
        if result['observer_sha256'] != c.digest(payload) or result['legal_actions_sha256'] != c.legal_digest(legal):
            raise ValueError('Worker observation/legal-action disagreement')
        if result['ruleset'] != rules.RULESET_ID or result['catalogue_sha256'] != rules.CATALOGUE_SHA256:
            raise ValueError('Worker ruleset disagreement')
        if result['player'] != player or result['legal_count'] != len(legal):
            raise ValueError('Worker player/legal-count disagreement')
        if player == 'ppo' and result['diagnostics']['profile'] != c.PPO_PROFILE:
            raise ValueError('Worker PPO profile disagreement')
    return {'player': player, 'observer': payload, 'clock': receipt}


def hidden_variant(state):
    variant = state.copy_shallow()
    for tier in (1, 2, 3):
        variant.hidden_tier_decks[tier].reverse()
    opponent = variant.players[1 - state.current_player]
    for i, card in enumerate(opponent.reserved_cards):
        if opponent.reserved_card_source(i) == 'deck' and variant.hidden_tier_decks[card.tier]:
            opponent.reserved_cards[i], variant.hidden_tier_decks[card.tier][0] = variant.hidden_tier_decks[card.tier][0], card
    return variant


def qualify(base, deadline, cpu):
    fixtures = c.read(base / 'fixtures.json')
    rows = []
    with worker(base, 'qualification', deadline, cpu) as child:
        for index, item in enumerate(fixtures):
            original = rules.deserialize_snapshot(item['snapshot'])
            variant = hidden_variant(original)
            if rules.observation_payload(original) != rules.observation_payload(variant):
                raise ValueError('Hidden-state invariance failed')
            for player in ('raw035', 'ppo'):
                paired = []
                for kind, state in [('original', original), ('hidden_variant', variant)]:
                    row = {'index': len(rows), 'fixture': index, 'kind': kind,
                           'source_root': item['source']['root_id'], **decision(child, state, player)}
                    c.save_new(base / 'dynamic_qualification' / f'{len(rows):03d}.json', row)
                    rows.append(row)
                    if row['clock']['status'] != 'ok':
                        raise ValueError('Dynamic one-second qualification failed: ' + row['clock']['status'])
                    paired.append(row['clock']['result'])
                if paired[0] != paired[1]: raise ValueError('Repeated hidden-state variant changed policy result')
            print(c.canonical({'qualification_fixture': index + 1, 'total': len(fixtures), 'passed_requests': len(rows)}), flush=True)
    summary = {'passed': True, 'fixtures': len(fixtures), 'requests': len(rows), 'seconds_per_move': 1,
               'privacy_pairs': len(rows) // 2, 'max_receipt_seconds': max(r['clock']['receipt_elapsed_seconds'] for r in rows)}
    c.save_new(base / 'dynamic_qualification.json', summary)
    return summary


def play_game(base, spec, deadline, cpu):
    env = rules.CorrectedEnv(seed=int(spec['seed']))
    state = env.initial_state(); initial = rules.serialize_snapshot(state)
    audit.validate_ledger(initial)
    game = {**spec, 'ruleset_id': rules.RULESET_ID, 'catalogue_sha256': rules.CATALOGUE_SHA256,
            'initial': initial, 'initial_sha256': c.digest(initial), 'moves': [], 'complete': False}
    folder = base / 'games' / f'{spec["game"]:03d}'
    c.save_new(folder / 'initial.json', {k: v for k, v in game.items() if k != 'moves'})
    with worker(base, f'game-{spec["game"]:03d}', deadline, cpu) as child:
        for ply in range(2000):
            if state.terminal: break
            actor = state.current_player; player = 'ppo' if actor == spec['ppo_seat'] else 'raw035'
            before = rules.serialize_snapshot(state)
            row = {'ply': ply, 'actor': actor, 'before_sha256': c.digest(before), **decision(child, state, player)}
            status = row['clock']['status']
            if status == 'ok':
                action = row['clock']['result']['action']
                state = env.step(state, c.deserialize_action(action))
                after = rules.serialize_snapshot(state)
                row['after_sha256'] = audit.check_step(before, action, after)
            c.save_new(folder / 'moves' / f'{ply:04d}.json', row)
            game['moves'].append({'file': str((folder / 'moves' / f'{ply:04d}.json').relative_to(c.ROOT)),
                                  'sha256': c.file_hash(folder / 'moves' / f'{ply:04d}.json')})
            if status != 'ok':
                if status not in PLAYER_FAILURES:
                    raise c.ResourceHold('Resource boundary: ' + status)
                detail = str(row['clock'].get('detail', '')) + str(row['clock'].get('error', ''))
                if 'empty' in detail.lower() or 'support' in detail.lower():
                    raise ValueError('Adapter support prerequisite failed during screen: ' + detail)
                game.update(complete=True, reason='player_forfeit_' + status, winner=1 - actor, failing_player=player)
                break
        if not game['complete']:
            if not state.terminal:
                raise c.ResourceHold('Nonterminal 2000-ply ceiling; no invented result')
            game.update(complete=True, reason='natural', winner=state.winner)
    game['final'] = rules.serialize_snapshot(state)
    game['ppo_score'] = .5 if game['winner'] is None else float(game['winner'] == spec['ppo_seat'])
    c.save_new(folder / 'result.json', game)
    return game


def quantile(sorted_values, q):
    if not sorted_values: return None
    x = (len(sorted_values) - 1) * q; lo = int(x); hi = min(lo + 1, len(sorted_values) - 1)
    return sorted_values[lo] + (sorted_values[hi] - sorted_values[lo]) * (x - lo)


def summarize(games):
    if len(games) != 60: raise ValueError('Incomplete fixed screen')
    schedule = {s['game']: s for s in make_schedule()}
    if {g['game'] for g in games} != set(schedule): raise ValueError('Game identities differ')
    for game in games:
        if any(game[k] != v for k, v in schedule[game['game']].items()):
            raise ValueError('Game schedule differs')
        if game['initial_sha256'] != c.digest(game['initial']): raise ValueError('Initial hash differs')
        if not game['complete'] or (game['winner'] is not None and type(game['winner']) is not int):
            raise ValueError('Invalid completed result')
        if game['winner'] not in (None, 0, 1): raise ValueError('Invalid winner')
        score = .5 if game['winner'] is None else float(game['winner'] == game['ppo_seat'])
        if game['ppo_score'] != score: raise ValueError('Score differs from winner')
    pairs = []
    for pair in range(30):
        two = [g for g in games if g['pair'] == pair]
        if len(two) != 2 or {g['ppo_seat'] for g in two} != {0, 1} or two[0]['initial_sha256'] != two[1]['initial_sha256']:
            raise ValueError('Paired world/seat identity failed')
        pairs.append(sum(g['ppo_score'] for g in two) / 2)
    rng = random.Random(BOOTSTRAP_SEED)
    samples = sorted(sum(rng.choices(pairs, k=30)) / 30 for _ in range(50000))
    return {'games': 60, 'pairs': 30, 'ppo_mean_score': sum(pairs) / 30,
        'paired_bootstrap_95_percent_interval': [quantile(samples, .025), quantile(samples, .975)],
        'bootstrap_replicates': 50000, 'bootstrap_seed': BOOTSTRAP_SEED,
        'ppo_wins': sum(g['ppo_score'] == 1 for g in games), 'draws': sum(g['ppo_score'] == .5 for g in games),
        'ppo_losses': sum(g['ppo_score'] == 0 for g in games),
        'by_ppo_seat': {str(seat): sum(g['ppo_score'] for g in games if g['ppo_seat'] == seat) / 30 for seat in (0, 1)},
        'forfeits': [{'game': g['game'], 'reason': g['reason'], 'failing_player': g.get('failing_player')} for g in games if g['reason'] != 'natural'],
        'interpretation': 'Development competitiveness/transfer screen under corrected rules; no elite-strength claim or historical strength carryover'}


def run(base):
    freeze = c.read(base / 'freeze.json'); c.verify(freeze['inputs'])
    start = time.monotonic(); deadline = start + 14400
    c.save_new(base / 'launch.json', {'time': c.now(), 'pid': os.getpid(), 'freeze_sha256': c.file_hash(base / 'freeze.json'),
                                    'seconds_cap': 14400, 'larger_data_deferred': True})
    status, failure, games, qualification = 'failed', None, [], None
    try:
        with c.execution_lease():
            cpu = freeze['topology']['selected_logical_processor']
            qualification = qualify(base, deadline, cpu)
            c.verify(freeze['inputs'])
            for spec in c.read(base / 'schedule.json'):
                game = play_game(base, spec, deadline, cpu); games.append(game)
                print(c.canonical({'game': spec['game'] + 1, 'total': 60, 'pair': spec['pair'],
                                   'reason': game['reason'], 'ppo_score': game['ppo_score'], 'plies': len(game['moves'])}), flush=True)
            summary = summarize(games)
            c.save_new(base / 'screen_summary.json', summary)
            c.verify(freeze['inputs']); audit_all_games(base)
            c.boundary(base, deadline, memory_headroom=False)
            status = 'completed'
    except BaseException:
        failure = traceback.format_exc(); c.save_new(base / 'failure.json', {'time': c.now(), 'failure': failure})
    preserved = False
    try:
        c.verify(freeze['inputs']); preserved = True
    except BaseException:
        final_failure = traceback.format_exc()
        c.save_new(base / 'final_source_failure.json', {'failure': final_failure})
        status = 'failed'
        failure = (failure or '') + '\nFinal source verification failed:\n' + final_failure
    c.save_new(base / 'completion.json', {'time': c.now(), 'status': status, 'failure': failure,
        'qualification': qualification, 'completed_games': len(games), 'elapsed_seconds': time.monotonic() - start,
        'frozen_inputs_unchanged': preserved, 'larger_data_deferred': True})
    print(c.canonical({'status': status, 'completed_games': len(games), 'failure': failure}), flush=True)
    return 0 if status == 'completed' else 1


def check_clock(receipt, request_id, *, limit=1.0):
    if receipt['request_id'] != request_id or receipt['clock'] != 'perf_counter_ns':
        raise ValueError('Clock request/implementation mismatch')
    deadline, elapsed, received = (receipt[k] for k in ('deadline_seconds', 'elapsed_seconds', 'receipt_elapsed_seconds'))
    if not all(type(v) in (int, float) and math.isfinite(v) for v in (deadline, elapsed)):
        raise ValueError('Invalid clock numeric fields')
    if not 0 < deadline <= limit or elapsed < 0:
        raise ValueError('Invalid clock bounds')
    if received is not None and (type(received) not in (int, float) or not math.isfinite(received)):
        raise ValueError('Invalid receipt time')
    if received is not None and received > elapsed:
        raise ValueError('Receipt after clock completion')
    before_deadline = received is not None and 0 <= received <= deadline
    if receipt['received_before_deadline'] is not before_deadline:
        raise ValueError('Incorrect deadline flag')
    if receipt['accepted'] is not (receipt['status'] == 'ok'):
        raise ValueError('Incorrect acceptance flag')
    if receipt['status'] == 'ok' and not before_deadline:
        raise ValueError('Late action accepted')


def audit_game(base, spec):
    folder = base / 'games' / f'{spec["game"]:03d}'
    game = c.read(folder / 'result.json')
    if any(game[k] != v for k, v in spec.items()): raise ValueError('Saved schedule mismatch')
    if not game['complete'] or game['ruleset_id'] != rules.RULESET_ID or game['catalogue_sha256'] != rules.CATALOGUE_SHA256:
        raise ValueError('Game qualification identity mismatch')
    state = game['initial']; audit.validate_ledger(state)
    expected_initial = rules.serialize_snapshot(rules.CorrectedEnv(seed=int(spec['seed'])).initial_state())
    if state != expected_initial or game['initial_sha256'] != c.digest(state):
        raise ValueError('Initial seed/hash mismatch')
    initial_file = c.read(folder / 'initial.json')
    if initial_file['initial'] != state or any(initial_file[k] != v for k, v in spec.items()):
        raise ValueError('Initial receipt mismatch')
    failure = None; count = 0; latency = {'ppo': [], 'raw035': []}
    for ply, item in enumerate(game['moves']):
        path = folder / 'moves' / f'{ply:04d}.json'
        if (c.ROOT / item['file']).resolve() != path.resolve() or c.file_hash(path) != item['sha256']:
            raise ValueError('Saved move identity/hash differs')
        row = c.read(path)
        if state['state']['terminal'] or failure is not None: raise ValueError('Move after game ended')
        actor = state['state']['current_player']
        player = 'ppo' if actor == spec['ppo_seat'] else 'raw035'
        if row['ply'] != ply or row['actor'] != actor or row['player'] != player:
            raise ValueError('Actor/player progression mismatch')
        audit.check_observer(state, row['observer'])
        if c.digest(state) != row['before_sha256']: raise ValueError('Replay chain mismatch')
        receipt = row['clock']; request_id = f'request-{ply + 1:08d}'
        original = c.read(base / 'workers' / f'game-{spec["game"]:03d}' / f'{request_id}.json')
        if receipt != original: raise ValueError('Copied clock differs from worker receipt')
        check_clock(receipt, request_id)
        if receipt['status'] == 'ok':
            result = receipt['result']
            legal = rules.legal_actions(rules.deserialize_snapshot(state))
            if (result['player'] != player or result['observer_sha256'] != c.digest(row['observer']) or
                result['legal_actions_sha256'] != c.legal_digest(legal) or result['legal_count'] != len(legal) or
                result['ruleset'] != rules.RULESET_ID or result['catalogue_sha256'] != rules.CATALOGUE_SHA256 or
                result['action'] not in [c.serialize_action(a) for a in legal]):
                raise ValueError('Worker result binding mismatch')
            if player == 'ppo' and result['diagnostics']['profile'] != c.PPO_PROFILE:
                raise ValueError('Saved PPO profile disagreement')
            state = audit.transition(state, result['action'])
            if c.digest(state) != row['after_sha256']: raise ValueError('Replay after-state mismatch')
            count += 1; latency[player].append(receipt['receipt_elapsed_seconds'])
        else:
            if receipt['status'] not in PLAYER_FAILURES: raise ValueError('Resource failure scored as a game')
            if any(s in str(receipt.get('detail', '')).lower() for s in ('empty', 'support')):
                raise ValueError('Support prerequisite failure scored as a game')
            failure = (receipt['status'], actor, player)
    if state != game['final']: raise ValueError('Final state replay mismatch')
    if failure is None:
        if game['reason'] != 'natural' or not state['state']['terminal'] or game['winner'] != state['state']['winner']:
            raise ValueError('Natural result mismatch')
    elif (game['reason'] != 'player_forfeit_' + failure[0] or game['winner'] != 1 - failure[1] or
          game.get('failing_player') != failure[2] or state['state']['terminal']):
        raise ValueError('Forfeit attribution mismatch')
    score = .5 if game['winner'] is None else float(game['winner'] == spec['ppo_seat'])
    if game['ppo_score'] != score: raise ValueError('Score derivation mismatch')
    return game, count, latency


def audit_all_games(base):
    schedule = c.read(base / 'schedule.json'); count = 0; games = []; latency = {'ppo': [], 'raw035': []}
    if schedule != make_schedule(): raise ValueError('Frozen schedule differs from declared derivation')
    for spec in schedule:
        game, transitions, times = audit_game(base, spec); count += transitions
        for player in latency: latency[player].extend(times[player])
        games.append(game)
    if summarize(games) != c.read(base / 'screen_summary.json'): raise ValueError('Saved statistics differ')
    c.save_new(base / 'independent_replay.json', {'passed': True, 'games': len(games), 'transitions': count,
            'method': 'Separate dictionary ledger transitions/observer redaction, bound clock receipts, independently derived scores; no engine.step during replay',
            'timely_completion_seconds': {p: {'count': len(t), 'mean': sum(t)/len(t) if t else None,
                'median': quantile(sorted(t), .5), 'p95': quantile(sorted(t), .95), 'max': max(t) if t else None}
                for p, t in latency.items()}})


def main():
    parser = argparse.ArgumentParser(); parser.add_argument('mode', choices=['prepare', 'run'])
    parser.add_argument('--output', type=Path, default=BASE); args = parser.parse_args()
    return prepare(args.output) if args.mode == 'prepare' else run(args.output)


if __name__ == '__main__':
    raise SystemExit(main())
