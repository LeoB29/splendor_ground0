"""Post-completion evidence audit; no arena, model imports or inference.

Transition semantics reuse the frozen independent dictionary ledger. Schedule,
initial-world reconstruction, clock checks, selected PPO index interpretation,
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
from pathlib import Path
import random
import struct
import sys

ROOT = Path(__file__).resolve().parents[1]
BASE = ROOT / 'outputs/external_strength_screen_20261018_01'
NAMESPACE = 'splendor_external_corrected_rules_20261018_development_v1'
PROFILE = 'official_rules_buy_and_token_alias_v1'
CHECKPOINT = '7fdb58b46d00305af42fad818b05549692e27ff1243cb63a76c6a858ba348df4'
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
             'ppo_seat': (pair + order) % 2}
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


def index_vocabulary():
    """Independent positional enumeration; no PPO module or neural dependency."""
    keys = []
    def add(kind, take=None, returned=None, position=None, noble=None):
        keys.append((kind, dict(take or {}), dict(returned or {}), position, noble))
    nobles = (None, 0, 1, 2, 3, 4)
    for noble in nobles:
        add('PASS', noble=noble)
    for tier in range(3):
        for slot in range(4):
            for noble in nobles:
                place = (tier, slot, -1)
                add('RESERVE', position=place, noble=noble)
                add('RESERVE', {'yellow': 1}, position=place, noble=noble)
                for color in NORMAL:
                    add('RESERVE', {'yellow': 1}, {color: 1}, place, noble)
    for color in NORMAL:
        other = [c for c in EXTERNAL if c != color]
        for noble in nobles:
            add('COLLECT_SAME', {color: 2}, noble=noble)
            for a, b in combinations(other, 2):
                add('COLLECT_SAME', {color: 2}, {a: 1, b: 1}, noble=noble)
            for amount in (1, 2):
                for returned in other:
                    add('COLLECT_SAME', {color: 2}, {returned: amount}, noble=noble)
    for size in (1, 2, 3):
        for selected in combinations(NORMAL, size):
            for noble in nobles:
                take = dict.fromkeys(selected, 1)
                add('COLLECT_DIFF', take, noble=noble)
                other = [c for c in EXTERNAL if c not in selected]
                for amount in range(1, size + 1):
                    for returned in combinations_with_replacement(other, amount):
                        add('COLLECT_DIFF', take, Counter(returned), noble=noble)
    for slot in range(3):
        for noble in nobles:
            add('BUY_RESERVE', position=(-1, -1, slot), noble=noble)
    for tier in range(3):
        for slot in range(4):
            for noble in nobles:
                add('BUY_AVAILABLE', position=(tier, slot, -1), noble=noble)
    check(len(keys) == 3510, 'PPO vocabulary size')
    return keys


VOCABULARY = index_vocabulary()


def external_counts(tokens):
    return dict(Counter('yellow' if c == 'gold' else c for c in tokens))


def verify_selected_index(snapshot, result, legal, cards):
    d = result['diagnostics']
    check(d['profile'] == PROFILE and d['checkpoint_sha256'] == CHECKPOINT, 'PPO identity')
    check(d['policy'] == 'adapted_pinned_ppo_v1', 'PPO policy label')
    index = d['action_index']
    check(type(index) is int and 0 <= index < 3510, 'PPO index bounds')
    check(type(d['support']) is int and 1 <= d['support'] <= 3510, 'PPO nonempty support')
    check(type(d['token_alias_count']) is int and 0 <= d['token_alias_count'] <= d['support'], 'Alias count bounds')
    check(type(d['selected_token_alias']) is bool, 'Alias flag type')
    check(d['authoritative_legal_actions'] == len(legal), 'PPO authoritative legal count')
    for name in ('features_sha256', 'mask_sha256', 'output_sha256'):
        check(isinstance(d[name], str) and len(d[name]) == 64 and all(c in '0123456789abcdef' for c in d[name]), 'PPO digest shape')
    kind, take, returned, position, noble = VOCABULARY[index]
    s = snapshot['state']; player = s['players'][s['current_player']]
    action = result['action']; native = action['action_type']
    check((None if noble is None else s['nobles'][noble]) == action['noble_id'], 'PPO noble index')
    actual_take, actual_return = external_counts(action['take_tokens']), external_counts(action['return_tokens'])
    if kind.startswith('COLLECT'):
        check(native == 'TAKE_TOKENS', 'Collect index action family')
        bank = {('yellow' if c == 'gold' else c): n for c, n in s['bank_tokens'].items()}
        held = {('yellow' if c == 'gold' else c): n for c, n in player['tokens'].items()}
        total, available = sum(held.values()), sum(bank[c] > 0 for c in NORMAL)
        if kind == 'COLLECT_SAME':
            check(len(take) == 1 and all(bank[c] >= 4 and n == 2 for c, n in take.items()), 'Pinned two-same availability')
        else:
            minimum = min(3 if total <= 7 else 2 if total == 8 else 1, available)
            check(minimum <= len(take) <= min(3, available) and all(n == 1 and bank[c] >= 1 for c, n in take.items()), 'Pinned different take availability')
        check(sum(returned.values()) == max(total + sum(take.values()) - 10, 0), 'Pinned return quantity')
        check(all(c not in take and n <= held[c] for c, n in returned.items()), 'Pinned return source')
        check(all(take.get(c, 0) - returned.get(c, 0) == actual_take.get(c, 0) - actual_return.get(c, 0) for c in EXTERNAL), 'Selected index token effect')
        alias = (take != actual_take or returned != actual_return)
        check(d['selected_token_alias'] is alias, 'Selected alias flag')
        if alias:
            candidates = [a for a in legal if a['action_type'] == 'TAKE_TOKENS' and a['noble_id'] == action['noble_id'] and all(
                external_counts(a['take_tokens']).get(c, 0) - external_counts(a['return_tokens']).get(c, 0) == take.get(c, 0) - returned.get(c, 0) for c in EXTERNAL)]
            check(action == min(candidates, key=canonical), 'Canonical alias representative')
            check(d['token_alias_count'] > 0, 'Selected alias absent from mask count')
    else:
        check(d['selected_token_alias'] is False, 'Non-token alias forbidden')
        if kind in ('BUY_AVAILABLE', 'RESERVE'):
            check(native == ('BUY_VISIBLE' if kind == 'BUY_AVAILABLE' else 'RESERVE_VISIBLE') and
                  position == (action['tier'] - 1, action['market_index'], -1), 'Market positional index')
            card_id = s['visible_tier_cards'][str(action['tier'])][action['market_index']]
        elif kind == 'BUY_RESERVE':
            check(native == 'BUY_RESERVED' and position == (-1, -1, action['reserved_index']), 'Reserved positional index')
            card_id = player['reserved_card_ids'][action['reserved_index']]
        else:
            check(kind == native == 'PASS', 'Pass positional index')
            card_id = None
        if kind.startswith('BUY'):
            cost = cards[card_id]['cost']
            need = {c: max(cost.get(c, 0) - player['bonuses'][c], 0) for c in COLORS}
            payment = {c: min(need[c], player['tokens'][c]) for c in COLORS}
            payment['gold'] = sum(need[c] - payment[c] for c in COLORS)
            check(Counter(action['spend_tokens']) == Counter({c: n for c, n in payment.items() if n}), 'Canonical minimum-gold payment')
        else:
            check(take == actual_take and returned == actual_return, 'Literal reserve/pass token meaning')
    return index, d['selected_token_alias'], d['token_alias_count']


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


def aggregate(games, replicates=50000):
    check(len(games) == 60 and [g['game'] for g in games] == list(range(60)), 'Complete game identities')
    points = [1 if g['winner'] is None else 2 if g['winner'] == g['ppo_seat'] else 0 for g in games]
    pair_units = [sum(points[2*p:2*p+2]) for p in range(30)]
    rng, numerators = random.Random(BOOTSTRAP_SEED), []
    for _ in range(replicates):
        multiplicities = Counter(math.floor(rng.random() * 30) for _ in range(30))
        numerators.append(sum(pair_units[i] * count for i, count in multiplicities.items()))
    values = [n / 120 for n in numerators]
    return {'games': 60, 'pairs': 30, 'ppo_mean_score': sum(points) / 120,
            'paired_bootstrap_95_percent_interval': [quantile(values, .025), quantile(values, .975)],
            'bootstrap_replicates': replicates, 'bootstrap_seed': BOOTSTRAP_SEED,
            'ppo_wins': points.count(2), 'draws': points.count(1), 'ppo_losses': points.count(0),
            'by_ppo_seat': {str(seat): sum(points[i] for i, g in enumerate(games) if g['ppo_seat'] == seat) / 60 for seat in (0, 1)},
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
        self.indices, self.aliases = Counter(), Counter()
        self.alias_games, self.alias_available_requests = set(), 0

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
              self.frozen['ppo_profile'] == PROFILE, 'Protocol constants')
        self.identity = {k: self.frozen[k] for k in ('ruleset_id', 'catalogue_sha256')}
        seen, size = set(), 0
        for item in self.frozen['inputs']:
            path = (ROOT / item['path']).resolve()
            check(path.is_relative_to(ROOT) and path not in seen, 'Frozen path identity')
            seen.add(path)
            check(path.stat().st_size == item['bytes'] and file_hash(path) == item['sha256'], 'Frozen hash/size changed: ' + item['path'])
            size += item['bytes']
        self.source_check = {'files': len(seen), 'bytes': size, 'all_hashes_and_sizes_match': True}
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

    def initialize(self, worker):
        row = self.read(self.base / 'workers' / worker / 'initialization.json')
        check_clock(row, None, 60, self.cpu)
        check(row['status'] == 'ok' and row['affinity'] == [self.cpu], 'Initialization rejected')
        check(row['result']['pid'] == row['identity']['pid'] and row['result']['threads'] == 1, 'Initialization process identity')
        check(all(v == '1' for v in row['thread_environment'].values()) and len(row['thread_environment']) == 6, 'Numerical thread environment')
        self.receipts[worker].append(row)

    def request(self, worker, number, row, snapshot, player, scope):
        request_id = f'request-{number:08d}'
        receipt = self.read(self.base / 'workers' / worker / f'{request_id}.json')
        check(receipt == row['clock'], 'Copied receipt differs from original')
        check_clock(receipt, request_id, 1, self.cpu)
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
            if player == 'ppo':
                index, alias, available = verify_selected_index(snapshot, result, legal, self.cards)
                if scope == 'screen':
                    self.indices[index] += 1
                    self.aliases[str(alias).lower()] += 1
                    if alias:
                        self.alias_games.add(worker)
                    self.alias_available_requests += available > 0
        else:
            check(receipt['status'] in FAILURES, 'Resource failure scored as player failure')
            detail = str(receipt.get('detail', '')) + str(receipt.get('error', ''))
            check(not any(s in detail.lower() for s in ('empty', 'support')), 'Support failure cannot be scored')
        return receipt

    def qualification(self):
        fixtures = self.read(self.base / 'fixtures.json')
        check(len(fixtures) == 23, 'Fixed qualification fixture count')
        self.initialize('qualification')
        maximum, pairs, rows = 0., 0, 0
        for fixture, item in enumerate(fixtures):
            snapshot = item['snapshot']
            self.ledger.validate_ledger(snapshot)
            for player in ('raw035', 'ppo'):
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
            player = 'ppo' if actor == spec['ppo_seat'] else 'raw035'
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
        score = .5 if game['winner'] is None else float(game['winner'] == spec['ppo_seat'])
        check(game['ppo_score'] == score, 'Game score')
        return game, transitions

    def finish(self, games, transitions, qualification):
        calculated = aggregate(games)
        stored = self.read(self.base / 'screen_summary.json')
        for key, value in calculated.items():
            if key not in ('pair_score_quarter_units', 'bootstrap_integer_totals_sha256'):
                check(stored[key] == value, 'Aggregate disagreement: ' + key)
        for pair in range(30):
            a, b = games[2*pair:2*pair+2]
            check(a['initial'] == b['initial'] and {a['ppo_seat'], b['ppo_seat']} == {0, 1}, 'Paired-world equality')
        replay = self.read(self.base / 'independent_replay.json')
        check(replay['passed'] is True and replay['games'] == 60 and replay['transitions'] == transitions, 'Production replay count')
        for player in ('ppo', 'raw035'):
            check(replay['timely_completion_seconds'][player] == descriptive(self.timely['screen:' + player]), 'Production latency summary')
        elapsed = self.completion['elapsed_seconds']
        wall = (datetime.fromisoformat(self.completion['time']) - datetime.fromisoformat(self.launch['time'])).total_seconds()
        check(finite(elapsed) and 0 < elapsed < 14400 and abs(wall - elapsed) < 3, 'Execution wall cap/UTC reconciliation')
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
        check('torch' not in sys.modules and 'tools.external_strength_screen_20261018.ppo' not in sys.modules, 'Unexpected model import')
        return {'status': 'passed', 'time': datetime.now(timezone.utc).isoformat(),
                'analyzer_sha256': file_hash(__file__), 'rules': self.identity, 'profile': PROFILE,
                'freeze_sha256': file_hash(self.base / 'freeze.json'), 'source_preservation': self.source_check,
                'qualification': qualification, 'statistics': calculated, 'transitions': transitions,
                'screen_decision_statuses': {k: dict(v) for k, v in sorted(self.statuses.items())},
                'timely_receipt_seconds': {k: descriptive(v) for k, v in sorted(self.timely.items())},
                'ppo_selected_indices': {str(k): v for k, v in sorted(self.indices.items())},
                'ppo_token_alias_selection': {'selected': self.aliases['true'], 'literal': self.aliases['false'],
                    'games_with_selected_alias': len(self.alias_games), 'decisions_with_alias_available': self.alias_available_requests,
                    'selected_index_semantics_and_alias_flag_independently_checked': True,
                    'full_mask_and_network_outputs_recomputed': False},
                'resources': {'worker_count': len(self.receipts), 'initialization_seconds': descriptive([r[0]['receipt_elapsed_seconds'] for r in self.receipts.values()]),
                    'peak_sampled_rss_bytes': max(t['peak_sampled_rss_bytes'] for t in telemetry),
                    'sum_request_sampled_cpu_seconds': sum(t['request_cpu_seconds'] for t in telemetry),
                    'sum_worker_last_observed_cpu_seconds': final_cpu,
                    'maximum_observed_os_threads': max(t['observed_threads'] for t in telemetry),
                    'execution_elapsed_seconds': elapsed, 'utc_elapsed_seconds': wall,
                    'evidence_bytes_before_this_receipt': before_bytes,
                    'limitations': 'RSS/CPU are sampled. OS thread count is not numerical-thread parallelism. No continuous free-memory/disk trace was saved.'},
                'scope': {'no_models_loaded_or_inference': True, 'no_games_or_training_run': True,
                    'aggregate_and_bootstrap_independent': True, 'dictionary_transition_audit_reused': True,
                    'timeout_completion_latencies_not_imputed': True,
                    'interpretation': 'Fixed development transfer screen against the adapted external policy, not elite or unchanged-published-agent strength.'},
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
    games = [{**s, 'winner': s['ppo_seat'] if s['game'] % 3 == 0 else None if s['game'] % 3 == 1 else 1 - s['ppo_seat'], 'reason': 'natural'} for s in schedule()]
    got = aggregate(games, 17)
    check((got['ppo_wins'], got['draws'], got['ppo_losses']) == (20, 20, 20), 'Synthetic WDL')
    pairs = [v / 4 for v in got['pair_score_quarter_units']]
    rng = random.Random(BOOTSTRAP_SEED)
    reference = sorted(sum(rng.choices(pairs, k=30)) / 30 for _ in range(17))
    check(got['paired_bootstrap_95_percent_interval'] == [quantile(reference, .025), quantile(reference, .975)], 'Independent bootstrap stream')
    telemetry = {'affinity': [1], 'cpu_seconds': 0., 'request_cpu_seconds': 0., 'rss_bytes': 1,
                 'peak_sampled_rss_bytes': 1, 'observed_threads': 1}
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
    check(len(VOCABULARY) == 3510 and VOCABULARY[0][0] == 'PASS' and VOCABULARY[-1][0] == 'BUY_AVAILABLE', 'Synthetic index order')
    index = next(i for i, key in enumerate(VOCABULARY) if key == ('COLLECT_DIFF', {'black': 1}, {'yellow': 1}, None, None))
    action = {'action_type': 'TAKE_TOKENS', 'noble_id': None,
              'take_tokens': ['black', 'red', 'green'], 'return_tokens': ['red', 'green', 'gold']}
    snapshot = {'state': {'current_player': 0, 'nobles': [],
        'bank_tokens': {'white': 0, 'blue': 0, 'green': 1, 'red': 1, 'black': 1, 'gold': 2},
        'players': [{'tokens': {'white': 0, 'blue': 0, 'green': 3, 'red': 3, 'black': 3, 'gold': 1}}]}}
    diagnostics = {'profile': PROFILE, 'checkpoint_sha256': CHECKPOINT, 'policy': 'adapted_pinned_ppo_v1',
        'action_index': index, 'support': 15, 'token_alias_count': 15, 'selected_token_alias': True,
        'authoritative_legal_actions': 1, **{k: '0' * 64 for k in ('features_sha256', 'mask_sha256', 'output_sha256')}}
    result = {'action': action, 'diagnostics': diagnostics}
    check(verify_selected_index(snapshot, result, [action], {}) == (index, True, 15), 'Synthetic token alias interpretation')
    result['diagnostics']['selected_token_alias'] = False
    try:
        verify_selected_index(snapshot, result, [action], {})
    except ValueError:
        pass
    else:
        raise AssertionError('Incorrect alias flag accepted')
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
