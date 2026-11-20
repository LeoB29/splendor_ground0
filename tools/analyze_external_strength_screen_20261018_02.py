"""Post-completion evidence audit; no arena, model imports or inference.

Transition semantics reuse the frozen independent dictionary ledger. Schedule,
initial-world reconstruction, clock checks, opponent identity checks,
aggregate statistics and integer-multiplicity bootstrap are separate here.
"""
from __future__ import annotations

import argparse
from collections import Counter, defaultdict
from copy import deepcopy
from datetime import datetime, timezone
import hashlib
from itertools import combinations, combinations_with_replacement
import json
import math
import os
from pathlib import Path
import random
import struct
import sys

ROOT = Path(__file__).resolve().parents[1]
BASE = ROOT / 'outputs/external_strength_screen_20261018_02'
NAMESPACE = 'splendor_external_corrected_rules_20261018_stronger02_development_v1'
OPPONENT_ID = 'splendorust'
BOOTSTRAP_SEED = 2026101803510
COLORS = ('white', 'blue', 'green', 'red', 'black')
TOKENS = COLORS + ('gold',)
EXTERNAL = ('black', 'red', 'yellow', 'green', 'blue', 'white')
NORMAL = tuple(c for c in EXTERNAL if c != 'yellow')
FAILURES = {'timeout', 'worker_exit', 'worker_error', 'malformed', 'wrong_request',
            'illegal', 'unsolicited', 'write_error'}


def check(value, message):
    if not value:
        raise ValueError('Independent screen audit: ' + message)


def canonical(value):
    return json.dumps(value, sort_keys=True, separators=(',', ':'), allow_nan=False)


def digest(value):
    return hashlib.sha256(canonical(value).encode()).hexdigest()


def file_hash(path):
    with Path(path).open('rb') as handle:
        return hashlib.file_digest(handle, 'sha256').hexdigest()


def finite(value):
    return type(value) in (int, float) and math.isfinite(value)


def quantile(values, fraction):
    if not values:
        return None
    ordered = sorted(values)
    position = (len(ordered) - 1) * fraction
    lower = math.floor(position)
    upper = min(lower + 1, len(ordered) - 1)
    return ordered[lower] + (ordered[upper] - ordered[lower]) * (position - lower)


def descriptive(values):
    return {'count': len(values), 'mean': sum(values) / len(values) if values else None,
            'median': quantile(values, .5), 'p95': quantile(values, .95),
            'max': max(values) if values else None}


def schedule():
    return [{'game': 2 * pair + order, 'pair': pair,
             'seed': str(int(hashlib.sha256(f'{NAMESPACE}:pair:{pair}'.encode()).hexdigest(), 16)),
             'opponent_seat': (pair + order) % 2}
            for pair in range(30) for order in range(2)]


def initial_snapshot(seed, catalogue, identity):
    rng = random.Random(int(seed))
    visible, hidden = {}, {}
    for tier in range(1, 4):
        cards = sorted((c['card_id'] for c in catalogue['cards'] if c['tier'] == tier),
                       key=lambda key: int(key.split('_')[1]))
        rng.shuffle(cards)
        visible[str(tier)], hidden[str(tier)] = cards[:4], cards[4:]
    nobles = sorted(n['noble_id'] for n in catalogue['nobles'])
    rng.shuffle(nobles)
    return {'schema': 'external_screen_snapshot_v1', **identity, 'state': {
        'current_player': 0, 'turn_index': 0, 'start_player': 0,
        'pending_round_end': False, 'terminal': False, 'winner': None,
        'bank_tokens': {c: 5 if c == 'gold' else 4 for c in TOKENS},
        'visible_tier_cards': visible, 'hidden_tier_decks': hidden,
        'deck_counts': {t: len(v) for t, v in hidden.items()}, 'nobles': nobles[:3],
        'players': [{'player_id': i, 'tokens': dict.fromkeys(TOKENS, 0),
                     'bonuses': dict.fromkeys(COLORS, 0), 'score': 0,
                     'purchased_card_ids': [], 'noble_ids': [],
                     'reserved_card_ids': [], 'reserved_card_sources': []} for i in (0, 1)]}}


def check_clock(receipt, request_id, limit, cpu):
    check(receipt['request_id'] == request_id and receipt['clock'] == 'perf_counter_ns', 'Clock identity')
    elapsed, deadline, received = (receipt[k] for k in ('elapsed_seconds', 'deadline_seconds', 'receipt_elapsed_seconds'))
    check(finite(elapsed) and elapsed >= 0 and finite(deadline) and 0 < deadline <= limit, 'Clock duration')
    check(received is None or finite(received) and received <= elapsed, 'Clock receipt stamp')
    timely = received is not None and 0 <= received <= deadline
    check(receipt['received_before_deadline'] is timely, 'Clock deadline flag')
    accepted = receipt['status'] == 'ok'
    check(receipt['accepted'] is accepted and (not accepted or timely), 'Clock acceptance')
    t = receipt['telemetry']
    check(t['affinity'] == [cpu], 'Worker affinity')
    check(all(finite(t[k]) and t[k] >= 0 for k in ('cpu_seconds', 'request_cpu_seconds')), 'CPU measurement')
    check(all(type(t[k]) is int and 0 <= t[k] <= 2 * 1024**3 for k in ('rss_bytes', 'peak_sampled_rss_bytes')), 'RSS measurement')
    check(t['rss_bytes'] <= t['peak_sampled_rss_bytes'], 'Peak RSS bound')
    check(type(t['observed_threads']) is int and t['observed_threads'] >= 1, 'Observed OS threads')
    check(t['resource_scope'] == 'supervised_worker_and_all_descendants', 'Process-tree resource scope')
    check(type(t['processes_observed']) is int and 1 <= t['processes_observed'] <= 4, 'Unexpected descendant process')
    check(1 <= len(t['process_tree']) <= 4 and sum(p['rss_bytes'] for p in t['process_tree']) == t['rss_bytes'], 'Combined process-tree RSS')
    check(all(p['affinity'] == [cpu] and finite(p['cpu_seconds']) and p['cpu_seconds'] >= 0 and
              type(p['rss_bytes']) is int and p['rss_bytes'] >= 0 for p in t['process_tree']), 'Descendant resource measurements')
    roles = [p['role'] for p in t['process_tree']]
    check(len(set(roles)) == len(roles) and set(roles) <= {'python', 'rust', 'console:python', 'console:rust'}, 'Process roles')
    owners = {p['role']: p['pid'] for p in t['process_tree']}
    for p in t['process_tree']:
        check(type(p['executable_sha256']) is str and len(p['executable_sha256']) == 64, 'Executable identity missing')
        if p['role'].startswith('console:'):
            owner = p['role'].split(':')[1]
            check(p['name'].lower() == 'conhost.exe' and Path(p['executable']).name.lower() == 'conhost.exe' and
                  p['parent_pid'] == owners.get(owner), 'Console helper ownership')
            if 'startup_affinity_before' in p:
                check(p['startup_affinity_adjustment_phase'] == 'initialization' and
                      p['startup_affinity_after'] == [cpu] and
                      p['startup_affinity_adjusted'] is (p['startup_affinity_before'] != [cpu]) and
                      type(p['startup_affinity_before']) is list and
                      all(type(i) is int and i >= 0 for i in p['startup_affinity_before']), 'Console startup affinity adjustment')


def main_action(action):
    kind = action['action_type']
    if kind == 'TAKE_TOKENS':
        return {'kind': 'Take', 'tokens': [action['take_tokens'].count(c) for c in COLORS]}
    if kind in ('BUY_VISIBLE', 'RESERVE_VISIBLE'):
        return {'kind': 'BuyVisible' if kind == 'BUY_VISIBLE' else 'ReserveVisible',
                'index': 4 * (action['tier'] - 1) + action['market_index']}
    if kind == 'BUY_RESERVED': return {'kind': 'BuyReserved', 'index': action['reserved_index']}
    if kind == 'RESERVE_DECK': return {'kind': 'ReserveDeck', 'tier': action['tier'] - 1}
    check(kind == 'PASS', 'Unknown action kind')
    return {'kind': 'Pass'}


def check_diagnostics(d, observer, action, legal, profile):
    """Check reported work and phase/action bindings without rerunning a policy."""
    check(set(d) == {'opponent_id', 'profile_id', 'observer_seed', 'phases', 'inference_calls',
                     'inference_rows', 'forced_pass'}, 'Diagnostic fields')
    check(d['opponent_id'] == OPPONENT_ID and d['profile_id'] == profile['profile_id'] and
          type(d['observer_seed']) is int and d['observer_seed'] == int(digest(observer)[:16], 16), 'Observer-only search seed')
    quota = profile['iterations']
    check(type(quota) is int and 0 < quota <= 2**32-1, 'Frozen simulation quota')
    main = main_action(action)
    forced = main['kind'] == 'Pass'
    check(d['forced_pass'] is forced, 'Forced PASS flag')
    main_choices = {canonical(main_action(a)) for a in legal}
    check(not forced or main_choices == {canonical(main)}, 'Unforced PASS')
    candidates = [a for a in legal if main_action(a) == main]
    expected = [] if forced else [(main, len(main_choices), True)]
    if main['kind'] in ('BuyVisible', 'BuyReserved'):
        payments = {tuple(a['spend_tokens'].count(c) for c in COLORS) for a in candidates}
        chosen = tuple(action['spend_tokens'].count(c) for c in COLORS)
        if len(payments) > 1: expected.append(({'kind': 'Pay', 'tokens': list(chosen)}, len(payments), False))
        candidates = [a for a in candidates if tuple(a['spend_tokens'].count(c) for c in COLORS) == chosen]
    if main['kind'] in ('Take', 'ReserveVisible', 'ReserveDeck'):
        returns = {tuple(a['return_tokens'].count(c) for c in TOKENS) for a in candidates}
        chosen = tuple(action['return_tokens'].count(c) for c in TOKENS)
        if len(returns) > 1: expected.append(({'kind': 'Return', 'tokens': list(chosen)}, len(returns), False))
        candidates = [a for a in candidates if tuple(a['return_tokens'].count(c) for c in TOKENS) == chosen]
    nobles = {a['noble_id'] for a in candidates}
    if len(nobles) > 1:
        check(None not in nobles, 'Mixed compulsory noble choices')
        native_order = 'N04 N05 N10 N09 N03 N06 N01 N08 N07 N02'.split()
        expected.append(({'kind': 'Noble', 'index': native_order.index(action['noble_id'])}, len(nobles), False))
    check(type(d['phases']) is list and len(d['phases']) == len(expected), 'Recorded phase coverage')
    totals = Counter({'forced_pass_decisions': int(forced), 'searched_main_decisions': 0,
                      'late_heuristic_main_decisions': 0, 'single_choice_main_decisions': 0,
                      'auxiliary_heuristic_phases': 0, 'simulations': 0, 'leaf_calls': 0})
    for phase, (selected, count, is_main) in zip(d['phases'], expected):
        check(set(phase) == {'action', 'simulations', 'leaf_calls', 'native_legal_count', 'heuristic_phase'}, 'Phase fields')
        check(phase['action'] == selected and type(phase['native_legal_count']) is int and phase['native_legal_count'] == count,
              'Selected phase action/count binding')
        heuristic = not is_main or count == 1 or observer['observation']['turn_index'] >= 124
        check(phase['heuristic_phase'] is heuristic and type(phase['simulations']) is int and
              type(phase['leaf_calls']) is int, 'Phase work types/classification')
        if heuristic:
            check(phase['simulations'] == phase['leaf_calls'] == 0, 'Heuristic phase reported search work')
            totals['auxiliary_heuristic_phases' if not is_main else
                   'late_heuristic_main_decisions' if observer['observation']['turn_index'] >= 124 else
                   'single_choice_main_decisions'] += 1
        else:
            check(phase['simulations'] == quota and 1 <= phase['leaf_calls'] <= quota + 1, 'Incomplete or impossible fixed search work')
            totals['searched_main_decisions'] += 1
        totals['simulations'] += phase['simulations']; totals['leaf_calls'] += phase['leaf_calls']
    check(type(d['inference_calls']) is int and type(d['inference_rows']) is int and
          d['inference_calls'] == d['inference_rows'] and
          totals['searched_main_decisions'] <= d['inference_calls'] <= totals['leaf_calls'], 'Host scalar inference work')
    totals['host_inference_calls'] = d['inference_calls']
    return totals


def aggregate(games, replicates=50000):
    check(len(games) == 60 and [g['game'] for g in games] == list(range(60)), 'Complete game identities')
    points = [1 if g['winner'] is None else 2 if g['winner'] == g['opponent_seat'] else 0 for g in games]
    pair_units = [sum(points[2*p:2*p+2]) for p in range(30)]
    rng, numerators = random.Random(BOOTSTRAP_SEED), []
    for _ in range(replicates):
        multiplicities = Counter(math.floor(rng.random() * 30) for _ in range(30))
        numerators.append(sum(pair_units[i] * count for i, count in multiplicities.items()))
    values = [n / 120 for n in numerators]
    return {'games': 60, 'pairs': 30, 'opponent_mean_score': sum(points) / 120,
            'paired_bootstrap_95_percent_interval': [quantile(values, .025), quantile(values, .975)],
            'bootstrap_replicates': replicates, 'bootstrap_seed': BOOTSTRAP_SEED,
            'opponent_wins': points.count(2), 'draws': points.count(1), 'opponent_losses': points.count(0),
            'by_opponent_seat': {str(seat): sum(points[i] for i, g in enumerate(games) if g['opponent_seat'] == seat) / 60 for seat in (0, 1)},
            'forfeits': [{'game': g['game'], 'reason': g['reason'], 'failing_player': g.get('failing_player')} for g in games if g['reason'] != 'natural'],
            'pair_score_quarter_units': pair_units,
            'bootstrap_integer_totals_sha256': hashlib.sha256(b''.join(struct.pack('<q', n) for n in numerators)).hexdigest()}


class Audit:
    def __init__(self, base):
        self.base = Path(base).resolve()
        self.evidence = {}
        self.receipts = defaultdict(list)
        self.timely = defaultdict(list)
        self.statuses = defaultdict(Counter)
        self.opponent_decisions = 0
        self.diagnostic_keysets = Counter()
        self.work_counts = Counter()
        self.worker_process_identities = defaultdict(dict)

    def read(self, path):
        path = Path(path).resolve()
        check(path.is_relative_to(ROOT), 'Evidence path outside repository')
        raw = path.read_bytes()
        self.evidence[path.relative_to(ROOT).as_posix()] = {'sha256': hashlib.sha256(raw).hexdigest(), 'bytes': len(raw)}
        return json.loads(raw)

    def freeze(self):
        self.completion = self.read(self.base / 'completion.json')
        check(self.completion['status'] == 'completed' and self.completion['failure'] is None and
              self.completion['completed_games'] == 60 and self.completion['frozen_inputs_unchanged'] is True,
              'Production is not successfully complete; refuse audit of active/failed screen')
        self.frozen = self.read(self.base / 'freeze.json')
        self.launch = self.read(self.base / 'launch.json')
        check(self.launch['freeze_sha256'] == file_hash(self.base / 'freeze.json'), 'Launch freeze binding')
        check(self.frozen['games'] == 60 and self.frozen['pairs'] == 30 and self.frozen['seconds_per_move'] == 1 and
              self.frozen['execution_cap_seconds'] == self.launch['seconds_cap'] == 14400 and
              self.frozen['opponent_id'] == OPPONENT_ID, 'Protocol constants')
        self.identity = {k: self.frozen[k] for k in ('ruleset_id', 'catalogue_sha256')}
        seen, size = set(), 0
        for item in self.frozen['inputs']:
            path = (ROOT / item['path']).resolve()
            check(path.is_relative_to(ROOT) and path not in seen, 'Frozen path identity')
            seen.add(path)
            check(path.stat().st_size == item['bytes'] and file_hash(path) == item['sha256'], 'Frozen hash/size changed: ' + item['path'])
            size += item['bytes']
        self.source_check = {'files': len(seen), 'bytes': size, 'all_hashes_and_sizes_match': True}
        start_path = self.base / 'qualification/execution_start.json'
        self.execution_start = self.read(start_path)
        check(start_path.resolve() in seen and self.frozen['execution_start'] == self.launch['execution_start'] and
              self.frozen['execution_start']['sha256'] == file_hash(start_path), 'Common start freeze binding')
        before = self.launch['elapsed_before_launch_seconds']
        check(self.execution_start['execution_cap_seconds'] == 14400 and finite(self.execution_start['monotonic_start']) and
              self.launch['monotonic_start'] == self.execution_start['monotonic_start'] and
              self.launch['execution_start_time'] == self.execution_start['time'] and
              finite(before) and 0 <= before < 14400 and
              abs(self.launch['screen_monotonic_start'] - self.launch['monotonic_start'] - before) < .1 and
              abs(self.launch['remaining_seconds'] + self.launch['screen_monotonic_start'] - self.launch['monotonic_start'] - 14400) < .1,
              'Qualification/screen common deadline')
        self.profile = self.read(self.base / 'opponent_profile.json')
        self.profile_sha256 = file_hash(self.base / 'opponent_profile.json')
        check(self.profile_sha256 == self.frozen['opponent_profile']['sha256'] and
              self.profile['profile_id'] == self.frozen['opponent_profile_id'] and
              self.profile['opponent_id'] == OPPONENT_ID and self.profile['deterministic_fixed_work'] is True,
              'Frozen deterministic profile identity')
        predecessor = ROOT / 'outputs/external_strength_screen_20261018_01'
        previous_freeze = self.read(predecessor / 'freeze.json')
        check(all((ROOT / item['path']).resolve() in seen for item in previous_freeze['inputs']), 'Missing inherited frozen input')
        check(all(p.resolve() in seen for p in predecessor.rglob('*') if p.is_file() and '__pycache__' not in p.parts), 'Missing prior screen evidence')
        previous_seeds = {r['seed'] for r in self.read(predecessor / 'schedule.json')}
        check(len(previous_seeds) == 30 and not previous_seeds & {r['seed'] for r in schedule()}, 'Previous screen seed overlap')
        self.cpu = self.frozen['topology']['selected_logical_processor']
        self.catalogue = self.read(self.base / 'catalogue.json')
        check(digest(self.catalogue) == self.identity['catalogue_sha256'], 'Catalogue digest')
        self.cards = {c['card_id']: c for c in self.catalogue['cards']}
        check(len(self.cards) == 90 and len(self.catalogue['nobles']) == 10, 'Catalogue size')
        check(self.read(self.base / 'schedule.json') == schedule(), 'Independent paired seed schedule')
        check(self.frozen['bootstrap'] == {'replicates': 50000, 'seed': BOOTSTRAP_SEED, 'unit': 'paired initial world',
              'percentiles': [.025, .975], 'interpolation': 'linear'}, 'Bootstrap protocol')
        check('torch' not in sys.modules, 'Model library loaded')
        for path in (ROOT, ROOT / 'src'):
            if str(path) not in sys.path:
                sys.path.insert(0, str(path))
        from tools.external_strength_screen_20261018 import audit, rules
        self.ledger, self.rules = audit, rules
        check('torch' not in sys.modules, 'Audit import loaded model library')

    def process_identity(self, worker, receipt, python_executable=None):
        owners = self.worker_process_identities[worker]
        for p in receipt['telemetry']['process_tree']:
            role, identity = p['role'], (p['pid'], p['created'])
            check(role not in owners or owners[role]['identity'] == identity, 'Worker process replacement')
            executable = Path(p['executable']).resolve()
            if role == 'rust':
                check(executable == (ROOT / self.profile['native_binary']).resolve(), 'Rust executable path')
                artifacts = [v for v in self.profile['artifacts'] if (ROOT / v['path']).resolve() == executable]
                check(len(artifacts) == 1 and p['executable_sha256'] == artifacts[0]['sha256'], 'Rust executable digest')
            elif role == 'python' and python_executable is not None:
                check(executable == Path(python_executable).resolve(), 'Python executable path')
            elif role.startswith('console:'):
                check(executable == (Path(os.environ['SystemRoot']) / 'System32/conhost.exe').resolve(), 'OS console executable path')
            check(role not in owners or owners[role]['executable'] == p['executable'] and
                  owners[role]['sha256'] == p['executable_sha256'], 'Executable identity changed')
            startup = {k: v for k, v in p.items() if k.startswith('startup_affinity_')}
            check(role not in owners or owners[role]['startup_affinity'] == startup, 'Console startup record changed')
            owners[role] = {'identity': identity, 'executable': p['executable'], 'sha256': p['executable_sha256'],
                            'startup_affinity': startup}
        check(receipt['telemetry']['processes_observed'] == len(owners), 'Observed process identity coverage')

    def initialize(self, worker):
        row = self.read(self.base / 'workers' / worker / 'initialization.json')
        check_clock(row, None, 60, self.cpu)
        check(row['status'] == 'ok' and row['affinity'] == [self.cpu], 'Initialization rejected')
        check(row['result']['pid'] == row['identity']['pid'] and row['result']['threads'] == 1, 'Initialization process identity')
        check(all(v == '1' for v in row['thread_environment'].values()) and len(row['thread_environment']) == 6, 'Numerical thread environment')
        self.process_identity(worker, row, row['command'][0])
        self.receipts[worker].append(row)

    def request(self, worker, number, row, snapshot, player, scope):
        request_id = f'request-{number:08d}'
        receipt = self.read(self.base / 'workers' / worker / f'{request_id}.json')
        check(receipt == row['clock'], 'Copied receipt differs from original')
        check_clock(receipt, request_id, 1, self.cpu)
        self.process_identity(worker, receipt)
        self.receipts[worker].append(receipt)
        self.ledger.check_observer(snapshot, row['observer'])
        check(row['player'] == player, 'Requested policy identity')
        self.statuses[scope + ':' + player][receipt['status']] += 1
        if receipt['status'] == 'ok':
            result = receipt['result']
            state = self.rules.deserialize_snapshot(snapshot)
            legal = [json.loads(json.dumps(self.rules.serialize_action(a))) for a in self.rules.legal_actions(state)]
            check(result['player'] == player and result['observer_sha256'] == digest(row['observer']) and
                  result['legal_actions_sha256'] == digest(legal) and result['legal_count'] == len(legal) and
                  result['ruleset'] == self.identity['ruleset_id'] and result['catalogue_sha256'] == self.identity['catalogue_sha256'] and
                  result['action'] in legal, 'Worker result legal/input/identity binding')
            self.timely[scope + ':' + player].append(receipt['receipt_elapsed_seconds'])
            if player == 'opponent':
                diagnostics = result['diagnostics']
                check(diagnostics['opponent_id'] == OPPONENT_ID and
                      diagnostics['profile_id'] == self.profile['profile_id'] and
                      result['profile_sha256'] == self.profile_sha256, 'Opponent profile identity')
                work = check_diagnostics(diagnostics, row['observer'], result['action'], legal, self.profile)
                if scope == 'screen':
                    self.opponent_decisions += 1
                    self.work_counts.update(work)
                    self.diagnostic_keysets[tuple(sorted(diagnostics))] += 1
        else:
            check(receipt['status'] in FAILURES, 'Resource failure scored as player failure')
            detail = str(receipt.get('detail', '')) + str(receipt.get('error', ''))
            check(not any(s in detail.lower() for s in ('empty', 'support', 'adapter_prerequisite')), 'Support failure cannot be scored')
        return receipt

    def qualification(self):
        fixtures = self.read(self.base / 'fixtures.json')
        check(len(fixtures) == 23, 'Fixed qualification fixture count')
        self.initialize('qualification')
        maximum, pairs, rows = 0., 0, 0
        for fixture, item in enumerate(fixtures):
            snapshot = item['snapshot']
            self.ledger.validate_ledger(snapshot)
            for player in ('raw035', 'opponent'):
                results = []
                for kind in ('original', 'hidden_variant'):
                    row = self.read(self.base / 'dynamic_qualification' / f'{rows:03d}.json')
                    check(row['index'] == rows and row['fixture'] == fixture and row['kind'] == kind and
                          row['source_root'] == item['source']['root_id'], 'Qualification case identity')
                    receipt = self.request('qualification', rows + 1, row, snapshot, player, 'qualification')
                    check(receipt['status'] == 'ok', 'Qualification failure')
                    maximum = max(maximum, receipt['receipt_elapsed_seconds'])
                    results.append((row['observer'], receipt['result']))
                    rows += 1
                check(results[0] == results[1], 'Qualification privacy/repeat pair')
                pairs += 1
        summary = {'passed': True, 'fixtures': 23, 'requests': rows, 'seconds_per_move': 1,
                   'privacy_pairs': pairs, 'max_receipt_seconds': maximum}
        check(summary == self.read(self.base / 'dynamic_qualification.json') == self.completion['qualification'], 'Qualification summary')
        check(len(list((self.base / 'dynamic_qualification').glob('*.json'))) == 92, 'Additional qualification rows')
        return summary

    def game(self, spec):
        folder = self.base / 'games' / f'{spec["game"]:03d}'
        worker = f'game-{spec["game"]:03d}'
        self.initialize(worker)
        game = self.read(folder / 'result.json')
        check(all(game[k] == v for k, v in spec.items()) and game['complete'] is True and
              all(game[k] == v for k, v in self.identity.items()), 'Game schedule/identity')
        expected = initial_snapshot(spec['seed'], self.catalogue, self.identity)
        check(game['initial'] == expected and game['initial_sha256'] == digest(expected), 'Independent initial shuffle')
        initial = self.read(folder / 'initial.json')
        check(all(initial[k] == game[k] for k in initial if k != 'complete') and initial['complete'] is False, 'Initial receipt binding')
        state, failure, transitions = expected, None, 0
        self.ledger.validate_ledger(state)
        check(0 < len(game['moves']) <= 2000, 'Game ply bounds')
        for ply, entry in enumerate(game['moves']):
            path = folder / 'moves' / f'{ply:04d}.json'
            check((ROOT / entry['file']).resolve() == path and file_hash(path) == entry['sha256'], 'Move file chain')
            row = self.read(path)
            check(not state['state']['terminal'] and failure is None, 'Move after game end')
            actor = state['state']['current_player']
            player = 'opponent' if actor == spec['opponent_seat'] else 'raw035'
            check(row['ply'] == ply and row['actor'] == actor and row['before_sha256'] == digest(state), 'Actor/ply/prestate chain')
            receipt = self.request(worker, ply + 1, row, state, player, 'screen')
            if receipt['status'] == 'ok':
                state = self.ledger.transition(state, receipt['result']['action'])
                check(row['after_sha256'] == digest(state), 'Independent successor hash')
                transitions += 1
            else:
                failure = (receipt['status'], actor, player)
        check(len(list((folder / 'moves').glob('*.json'))) == len(game['moves']), 'Extra game moves')
        check(state == game['final'], 'Final ledger')
        if failure is None:
            check(game['reason'] == 'natural' and state['state']['terminal'] and
                  game['winner'] == state['state']['winner'], 'Natural winner')
        else:
            check(not state['state']['terminal'] and game['reason'] == 'player_forfeit_' + failure[0] and
                  game['winner'] == 1 - failure[1] and game['failing_player'] == failure[2], 'Forfeit winner')
        check(game['winner'] is None or type(game['winner']) is int and game['winner'] in (0, 1), 'Winner type')
        score = .5 if game['winner'] is None else float(game['winner'] == spec['opponent_seat'])
        check(game['opponent_score'] == score, 'Game score')
        return game, transitions

    def finish(self, games, transitions, qualification):
        calculated = aggregate(games)
        stored = self.read(self.base / 'screen_summary.json')
        for key, value in calculated.items():
            if key not in ('pair_score_quarter_units', 'bootstrap_integer_totals_sha256'):
                check(stored[key] == value, 'Aggregate disagreement: ' + key)
        for pair in range(30):
            a, b = games[2*pair:2*pair+2]
            check(a['initial'] == b['initial'] and {a['opponent_seat'], b['opponent_seat']} == {0, 1}, 'Paired-world equality')
        replay = self.read(self.base / 'independent_replay.json')
        check(replay['passed'] is True and replay['games'] == 60 and replay['transitions'] == transitions, 'Production replay count')
        for player in ('opponent', 'raw035'):
            check(replay['timely_completion_seconds'][player] == descriptive(self.timely['screen:' + player]), 'Production latency summary')
        elapsed = self.completion['elapsed_seconds']
        wall = (datetime.fromisoformat(self.completion['time']) - datetime.fromisoformat(self.execution_start['time'])).total_seconds()
        check(finite(elapsed) and 0 < elapsed < 14400 and abs(wall - elapsed) < 3, 'Execution wall cap/UTC reconciliation')
        screen_wall = (datetime.fromisoformat(self.completion['time']) - datetime.fromisoformat(self.launch['time'])).total_seconds()
        screen_elapsed = self.completion['screen_elapsed_seconds']
        check(finite(screen_elapsed) and 0 < screen_elapsed <= elapsed and abs(screen_wall - screen_elapsed) < 3 and
              abs(elapsed - screen_elapsed - self.launch['elapsed_before_launch_seconds']) < .1, 'Screen/common elapsed accounting')
        check({p.name for p in (self.base / 'workers').iterdir() if p.is_dir()} == set(self.receipts), 'Worker directory identities')
        check({p.name for p in (self.base / 'games').iterdir() if p.is_dir()} == {f'{i:03d}' for i in range(60)}, 'Game directory identities')
        telemetry = []
        final_cpu = 0.
        for worker, receipts in self.receipts.items():
            check(len(list((self.base / 'workers' / worker).glob('request-*.json'))) == len(receipts) - 1, 'Extra worker requests')
            cpu_values = [r['telemetry']['cpu_seconds'] for r in receipts]
            check(cpu_values == sorted(cpu_values), 'Nonmonotonic worker CPU')
            final_cpu += max(cpu_values)
            telemetry.extend(r['telemetry'] for r in receipts)
        before_bytes = sum(p.stat().st_size for p in self.base.rglob('*') if p.is_file())
        check(before_bytes < 256 * 1024**2, 'Evidence size cap')
        check('torch' not in sys.modules and 'tools.external_strength_screen_20261018_02.splendorust_adapter' not in sys.modules, 'Unexpected model import')
        return {'status': 'passed', 'time': datetime.now(timezone.utc).isoformat(),
                'analyzer_sha256': file_hash(__file__), 'rules': self.identity, 'profile': self.profile['profile_id'],
                'freeze_sha256': file_hash(self.base / 'freeze.json'), 'source_preservation': self.source_check,
                'qualification': qualification, 'statistics': calculated, 'transitions': transitions,
                'screen_decision_statuses': {k: dict(v) for k, v in sorted(self.statuses.items())},
                'timely_receipt_seconds': {k: descriptive(v) for k, v in sorted(self.timely.items())},
                'opponent_decision_audit': {'accepted_decisions': self.opponent_decisions,
                    'reported_work_counts_independently_checked': dict(self.work_counts),
                    'fixed_simulations_per_searched_main_decision': self.profile['iterations'],
                    'diagnostic_keysets': [{'keys': list(k), 'decisions': v} for k, v in sorted(self.diagnostic_keysets.items())],
                    'model_or_search_recomputed': False,
                    'identity_and_authoritative_legal_action_verified': True},
                'resources': {'worker_count': len(self.receipts), 'initialization_seconds': descriptive([r[0]['receipt_elapsed_seconds'] for r in self.receipts.values()]),
                    'peak_sampled_rss_bytes': max(t['peak_sampled_rss_bytes'] for t in telemetry),
                    'sum_request_sampled_cpu_seconds': sum(t['request_cpu_seconds'] for t in telemetry),
                    'sum_worker_last_observed_cpu_seconds': final_cpu,
                    'maximum_observed_os_threads': max(t['observed_threads'] for t in telemetry),
                    'maximum_processes_observed_per_worker': max(t['processes_observed'] for t in telemetry),
                    'console_helpers_with_initialization_affinity_adjustment': sum(
                        p['startup_affinity'].get('startup_affinity_adjusted', False)
                        for roles in self.worker_process_identities.values() for p in roles.values()),
                    'scope': 'Combined Python, persistent Rust and their exact OS console helpers, sampled on one assigned core',
                    'execution_elapsed_seconds': elapsed, 'utc_elapsed_seconds': wall,
                    'screen_elapsed_seconds': screen_elapsed, 'elapsed_before_screen_launch_seconds': self.launch['elapsed_before_launch_seconds'],
                    'execution_cap_includes_source_qualification_and_timing_pilots': True,
                    'evidence_bytes_before_this_receipt': before_bytes,
                    'limitations': 'RSS/CPU are sampled. OS thread count is not numerical-thread parallelism. No continuous free-memory/disk trace was saved.'},
                'scope': {'no_models_loaded_or_inference': True, 'no_games_or_training_run': True,
                    'aggregate_and_bootstrap_independent': True, 'dictionary_transition_audit_reused': True,
                    'timeout_completion_latencies_not_imputed': True,
                    'interpretation': 'Fixed development transfer screen against the adapted external policy, not elite strength or an unchanged native championship search result.'},
                'bootstrap_scope': {'all_observed_pair_scores_identical': len(set(calculated['pair_score_quarter_units'])) == 1,
                    'interval_is_degenerate': len(set(calculated['paired_bootstrap_95_percent_interval'])) == 1,
                    'interpretation': 'A degenerate empirical bootstrap interval reflects the observed resampling distribution; it does not establish universal zero uncertainty or a zero future win/loss probability.'},
                'evidence_files': [{'path': p, **v} for p, v in sorted(self.evidence.items())]}


def analyze(base):
    audit = Audit(base)
    audit.freeze()
    qualification = audit.qualification()
    games, transitions = [], 0
    for spec in schedule():
        game, count = audit.game(spec)
        games.append(game)
        transitions += count
    return audit.finish(games, transitions, qualification)


def self_test():
    check(len(schedule()) == 60 and len({r['seed'] for r in schedule()}) == 30, 'Synthetic schedule')
    check(quantile([0, 1, 2, 3], .25) == .75 and quantile([], .5) is None, 'Synthetic quantiles')
    games = [{**s, 'winner': s['opponent_seat'] if s['game'] % 3 == 0 else None if s['game'] % 3 == 1 else 1 - s['opponent_seat'], 'reason': 'natural'} for s in schedule()]
    got = aggregate(games, 17)
    check((got['opponent_wins'], got['draws'], got['opponent_losses']) == (20, 20, 20), 'Synthetic WDL')
    pairs = [v / 4 for v in got['pair_score_quarter_units']]
    rng = random.Random(BOOTSTRAP_SEED)
    reference = sorted(sum(rng.choices(pairs, k=30)) / 30 for _ in range(17))
    check(got['paired_bootstrap_95_percent_interval'] == [quantile(reference, .025), quantile(reference, .975)], 'Independent bootstrap stream')
    telemetry = {'affinity': [1], 'cpu_seconds': 0., 'request_cpu_seconds': 0., 'rss_bytes': 1,
                 'peak_sampled_rss_bytes': 1, 'observed_threads': 1,
                 'resource_scope': 'supervised_worker_and_all_descendants', 'processes_observed': 1,
                 'process_tree': [{'pid': 1, 'created': 1., 'rss_bytes': 1, 'cpu_seconds': 0., 'affinity': [1], 'observed_threads': 1,
                                   'role': 'python', 'executable': 'synthetic/python.exe', 'name': 'python.exe',
                                   'parent_pid': 0, 'executable_sha256': '0' * 64}]}
    good = {'request_id': 'r', 'clock': 'perf_counter_ns', 'elapsed_seconds': .2, 'deadline_seconds': 1.,
            'receipt_elapsed_seconds': .1, 'received_before_deadline': True, 'accepted': True,
            'status': 'ok', 'telemetry': telemetry}
    check_clock(good, 'r', 1, 1)
    for field, value in [('receipt_elapsed_seconds', 2.), ('deadline_seconds', True), ('accepted', False)]:
        bad = {**good, field: value}
        try:
            check_clock(bad, 'r', 1, 1)
        except ValueError:
            pass
        else:
            raise AssertionError('Malformed clock accepted: ' + field)
    print('Independent analyzer synthetic checks passed; no production evidence read.')


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--base', type=Path, default=BASE)
    parser.add_argument('--output', type=Path)
    parser.add_argument('--self-test', action='store_true')
    args = parser.parse_args()
    if args.self_test:
        self_test()
        return
    result = analyze(args.base)
    destination = args.output or args.base / 'independent_analysis.json'
    encoded = json.dumps(result, sort_keys=True, indent=2, allow_nan=False) + '\n'
    if destination.resolve().is_relative_to(args.base.resolve()):
        check(result['resources']['evidence_bytes_before_this_receipt'] + len(encoded.encode()) < 256 * 1024**2, 'Final evidence ceiling')
    with destination.open('x', encoding='utf-8', newline='\n') as handle:
        handle.write(encoded)
    print(canonical({'status': result['status'], 'games': 60, 'transitions': result['transitions'],
                     'output': str(destination), 'statistics': result['statistics']}))


if __name__ == '__main__':
    main()
