"""Exhaustive corrected-rule continuation audit, independent of rollout stepping.

Shared trusted dependencies: corrected legal enumeration/serialization, frozen
feature encoders and neural models. Every state transition uses the separate
dictionary ledger; world reconstruction, proposal grouping and choice arithmetic
are implemented here. No production execution or file mutation occurs on import.
"""
from __future__ import annotations

from collections import Counter
from copy import deepcopy
import gzip
import hashlib
import json
import math
from pathlib import Path
import random

from tools.external_strength_screen_20261018 import rules, audit as ledger

COLORS = ('white', 'blue', 'green', 'red', 'black')
TOKENS = COLORS + ('gold',)
FITS = (161001, 161002, 161003)
OUTCOME_FIELDS = ('reason', 'winner', 'turns', 'trace_sha256', 'final_state_sha256')
PROFILE_ID = 'four_candidate128_corrected_rules_20261018_v1'
LOCAL_PROFILE = PROFILE_ID


def require(condition, message):
    if not condition:
        raise ValueError('Corrected planner independent audit: ' + message)


def digest(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(',', ':'), allow_nan=False).encode()).hexdigest()


def file_hash(path):
    with Path(path).open('rb') as stream:
        return hashlib.file_digest(stream, 'sha256').hexdigest()


def envelope(state):
    return {'schema': rules.SNAPSHOT_SCHEMA, 'ruleset_id': rules.RULESET_ID,
            'catalogue_sha256': rules.CATALOGUE_SHA256, 'state': state}


def observer_envelope(observer):
    return {'schema': rules.OBSERVATION_SCHEMA, 'ruleset_id': rules.RULESET_ID,
            'catalogue_sha256': rules.CATALOGUE_SHA256, 'observation': deepcopy(observer['observation'])}


def check_context(context):
    require(set(context) == {'fit', 'stream', 'identity'}, 'Public context fields')
    identity = context['identity']
    require(type(identity) is list and len(identity) == 3 and identity[0] == 'main' and
            type(identity[1]) is int and 0 <= identity[1] < 402 and type(identity[2]) is int and 0 <= identity[2] < 400,
            'Public root identity')
    pair = identity[1] // 2
    assignment = sorted(range(201), key=lambda p: digest(['proposal-breadth-arena-128-fit-v1', 'main', p]))
    require(type(context['fit']) is int and context['fit'] == FITS[assignment.index(pair) % 3] and
            context['stream'] == digest(['proposal-breadth-arena-128-stream-v1', 'main', pair]), 'Fit/stream assignment')


def world_from_observer(observer, seed=None):
    """Rebuild each unknown pool directly from the catalogue and public ledger."""
    view = deepcopy(observer['observation'])
    cards = rules.corrected_cards_by_id()
    known, unknown = set(), {tier: [] for tier in (1, 2, 3)}
    for values in view['visible_tier_cards'].values():
        known.update(values)
    for seat, player in enumerate(view['players']):
        known.update(player['purchased_card_ids'])
        reserves = player.pop('reserves')
        player['reserved_card_ids'], player['reserved_card_sources'] = [], []
        for i, reserve in enumerate(reserves):
            private = seat != view['current_player'] and reserve['source'] != 'visible'
            require(reserve['source'] in ('visible', 'deck'), 'Unqualified legacy reserve provenance')
            if private:
                require(reserve['card_id'] is None, 'Opponent private identity leaked')
                unknown[reserve['tier']].append((seat, i))
                player['reserved_card_ids'].append(None)
            else:
                require(reserve['card_id'] in cards and cards[reserve['card_id']].tier == reserve['tier'], 'Known reserve face/tier')
                known.add(reserve['card_id']); player['reserved_card_ids'].append(reserve['card_id'])
            player['reserved_card_sources'].append(reserve['source'])
    rng = random.Random(seed) if seed is not None else None
    view['hidden_tier_decks'] = {}
    for tier in (1, 2, 3):
        pool = sorted(key for key, card in cards.items() if card.tier == tier and key not in known)
        require(len(pool) == view['deck_counts'][str(tier)] + len(unknown[tier]), 'Unknown pool conservation')
        if rng is not None: rng.shuffle(pool)
        for seat, i in unknown[tier]:
            view['players'][seat]['reserved_card_ids'][i] = pool.pop()
        view['hidden_tier_decks'][str(tier)] = pool
    result = envelope(view)
    ledger.validate_ledger(result)
    ledger.check_observer(result, observer_envelope(observer))
    return result


def belief_key(observer):
    view = observer['observation']; actor = view['current_player']
    def tokens(value): return tuple((c, value[c]) for c in TOKENS)
    def player(p, own):
        reserves = tuple(sorted((r['source'], r['card_id'] if own or r['source'] == 'visible' else r['tier']) for r in p['reserves']))
        return (tokens(p['tokens']), tuple((c, p['bonuses'][c]) for c in COLORS), p['score'],
                tuple(sorted(p['purchased_card_ids'])), tuple(sorted(p['noble_ids'])), reserves)
    return ('splendor-public-belief-v1', True, view['start_player'] == actor, view['turn_index'],
            view['pending_round_end'], False, None, tokens(view['bank_tokens']),
            tuple((t, view['deck_counts'][str(t)]) for t in (1, 2, 3)), tuple(sorted(view['nobles'])),
            tuple((t, tuple(sorted(view['visible_tier_cards'][str(t)]))) for t in (1, 2, 3)),
            player(view['players'][actor], True), player(view['players'][1-actor], False))


def schedule(observer):
    belief = digest(belief_key(observer)); stream = observer['context']['stream']
    return {'belief_sha256': belief, 'candidate_nonce': digest(['155-candidate-v1', stream, belief]),
            'world_seeds': [f'155-world-v1|{stream}|{belief}|{i:02d}' for i in range(128)]}


def action_key(snapshot, action):
    state = snapshot['state']; kind = action['action_type']; target = None
    if kind in ('BUY_VISIBLE', 'RESERVE_VISIBLE'):
        target = state['visible_tier_cards'][str(action['tier'])][action['market_index']]
    elif kind == 'BUY_RESERVED':
        target = state['players'][state['current_player']]['reserved_card_ids'][action['reserved_index']]
    net = Counter(action['take_tokens']); net.subtract(action['return_tokens']); net.subtract(action['spend_tokens'])
    return repr((kind, target, action['tier'] if kind == 'RESERVE_DECK' else None,
                 tuple((c, net[c]) for c in TOKENS), action['noble_id']))


def finite_scores(values, count):
    require(type(values) is list and len(values) == count and all(type(v) in (int, float) and math.isfinite(v) for v in values), 'Finite complete score vector')


def derive_proposal(observer, snapshot, legal, raw, logits, model_artifact):
    finite_scores(raw, len(legal)); finite_scores(logits, len(legal))
    keys, members, representatives = [], {}, {}
    for index, action in enumerate(legal):
        key = action_key(snapshot, action)
        if key not in members:
            keys.append(key); members[key] = []; representatives[key] = action
        members[key].append(index)
    order, top_representatives = [], {}
    for index in sorted(range(len(legal)), key=lambda i: (-raw[i], i)):
        key = action_key(snapshot, legal[index])
        if key not in top_representatives:
            order.append(key); top_representatives[key] = legal[index]
    root_schedule = schedule(observer)
    chosen = order[:2]
    representatives.update({key: top_representatives[key] for key in chosen})
    remaining = [key for key in keys if key not in chosen]
    rng = random.Random(f"149-candidates-v1|155-live-v1|{root_schedule['candidate_nonce']}")
    chosen += rng.sample(remaining, min(2, len(remaining)))
    reference = {'keys': chosen, 'actions': [representatives[k] for k in chosen], 'canonical_support': len(keys), 'legal_support': len(legal)}
    group_scores = [max(logits[i] for i in members[k]) for k in keys]
    offered = list(chosen)
    if len(chosen) == 4:
        available = [i for i, key in enumerate(keys) if key not in chosen[:3]]
        selected = min(available, key=lambda i: (-group_scores[i], i))
        offered = chosen[:3] + [keys[selected]]
    candidates = {**reference, 'keys': offered, 'actions': [representatives[k] for k in offered]}
    return {'schedule': root_schedule, 'reference': reference, 'candidates': candidates,
            'model_id': f"anchor_{observer['context']['fit']}", 'model_sha256': model_artifact['sha256'],
            'fit': observer['context']['fit'], 'group_keys': keys, 'legal_actions': legal,
            'learned_logits': logits, 'learned_group_scores': group_scores,
            'anchor_candidates': candidates, 'root_identity': observer['context']['identity'],
            'breadth': False, 'added_indices': {'ranked': [], 'random': []}}


def root_scores(state, legal, model_id):
    from . import local
    import torch
    from splendor_ai.encoding import encode_public_observation_tensor, encode_legal_action_features, fit_feature_vector
    model, artifact = local.registered_model(model_id)
    obs = encode_public_observation_tensor(state, state.current_player)
    features = encode_legal_action_features(state=state, legal_actions=legal, player_id=state.current_player)
    x = torch.tensor(fit_feature_vector(obs.vector, model.config.observation_size), dtype=torch.float32)
    y = torch.tensor([fit_feature_vector(row.vector, model.config.action_feature_size) for row in features], dtype=torch.float32)
    with torch.no_grad():
        logits, _ = model(x[None], y[None], torch.ones((1, len(legal)), dtype=torch.bool))
    return list(map(float, local.score_actions(state, legal))), logits[0].tolist(), artifact


def signature(snapshot):
    value = deepcopy(snapshot['state']); value['turn_index'] %= 2
    return digest(value)


def progress(before, after, action):
    if action['action_type'].startswith(('BUY', 'RESERVE')): return True
    actor = before['state']['current_player']
    p, q = before['state']['players'][actor], after['state']['players'][actor]
    other_p, other_q = before['state']['players'][1-actor], after['state']['players'][1-actor]
    return (any(len(p[k]) != len(q[k]) for k in ('purchased_card_ids', 'reserved_card_ids', 'noble_ids')) or
            any(p[k] != q[k] for k in ('score', 'bonuses')) or any(other_p[k] != other_q[k] for k in ('score', 'bonuses')))


def adjudicate(snapshot):
    ranks = [(p['score'], -len(p['purchased_card_ids'])) for p in snapshot['state']['players']]
    return None if ranks[0] == ranks[1] else (0 if ranks[0] > ranks[1] else 1)


def audit_trajectory(world, forced_action, row, boundary=lambda: None, scorer=None):
    if scorer is None:
        from .local import score_actions as scorer
    require(row['initial_state_sha256'] == digest(world['state']), 'Initial world trace hash')
    require(type(row['steps']) is list and row['steps'], 'Empty trajectory')
    current, seen, forced, stalled, raw_choices = deepcopy(world), Counter(), 0, 0, 0
    hasher = hashlib.sha256(); reason = None; winner = None
    for index, step in enumerate(row['steps']):
        boundary()
        s = current['state']; seen[signature(current)] += 1
        require(not s['terminal'] and seen[signature(current)] < 4 and s['turn_index'] < 400 and stalled < 60 and forced < 2,
                'Trace continued past terminal/cutoff')
        state = rules.deserialize_snapshot(current)
        legal_objects = rules.legal_actions(state)
        legal = [rules.serialize_action(a) for a in legal_objects]
        require(legal and set(step) == {'state_hash', 'action', 'turn', 'actor', 'legal_count', 'selected_index', 'forced', 'scores_sha256'}, 'Trace step fields/support')
        require(step['state_hash'] == digest(s) and type(step['turn']) is int and step['turn'] == s['turn_index'] and
                type(step['actor']) is int and step['actor'] == s['current_player'] and type(step['legal_count']) is int and step['legal_count'] == len(legal), 'Trace state/legal identity')
        require(step['forced'] is (index == 0), 'Forced action position')
        if index == 0:
            require(step['scores_sha256'] is None and forced_action in legal, 'Forced candidate legal identity')
            selected = legal.index(forced_action)
        else:
            scores = list(map(float, scorer(state, legal_objects))); finite_scores(scores, len(legal))
            require(digest(scores) == step['scores_sha256'], 'Raw035 score digest differs on replay')
            selected = max(range(len(legal)), key=lambda i: scores[i]); raw_choices += 1
        require(type(step['selected_index']) is int and step['selected_index'] == selected and step['action'] == legal[selected], 'Raw035 argmax/first-tie action')
        hasher.update(digest(step['action']).encode())
        successor = ledger.transition(current, step['action'])
        changed = progress(current, successor, step['action'])
        forced = forced + 1 if len(legal) == 1 and legal[0]['action_type'] == 'PASS' and not changed else 0
        stalled = 0 if changed else stalled + 1
        current = successor
        if forced >= 2:
            reason, winner = 'forced_pass_deadlock', adjudicate(current)
        elif current['state']['terminal']:
            reason, winner = 'completed', current['state']['winner']
        if reason is not None: require(index == len(row['steps']) - 1, 'Extra trace after natural termination')
    require(reason is not None and stalled < 60, 'Incomplete/non-natural trajectory')
    expected = {'reason': reason, 'winner': winner, 'turns': len(row['steps']), 'trace_sha256': hasher.hexdigest(),
                'final_state_sha256': digest(current['state'])}
    require(all(row[k] == v and (k not in ('turns', 'winner') or row[k] is None or type(row[k]) is int) for k, v in expected.items()), 'Trace outcome mismatch')
    return expected, raw_choices


def select_outcomes(worlds, support):
    require(len(worlds) == 128 and 1 <= support <= 4, 'Complete128-world support')
    totals = [0] * support
    for world in worlds:
        require(len(world['outcomes']) == support, 'Incomplete candidate assessments')
        for i, row in enumerate(world['outcomes']):
            require(type(row['wdl']) in (int, float) and row['wdl'] in (-1, 0, 1), 'WDL domain')
            totals[i] += int(row['wdl'])
    selected = max(range(support), key=lambda i: totals[i])
    return selected, [n / 128 for n in totals]


def audit_decision(observer, result, traces_path, boundary=lambda: None):
    """Re-execute all policy choices and independently replay every continuation."""
    boundary(); check_context(observer['context'])
    require(set(observer) == {'schema', 'context', 'ruleset_id', 'catalogue_sha256', 'observation'} and
            observer['schema'] == 'corrected_planner_observer_v1' and observer['ruleset_id'] == rules.RULESET_ID and
            observer['catalogue_sha256'] == rules.CATALOGUE_SHA256 and observer['observation']['terminal'] is False and
            observer['observation']['winner'] is None, 'Observer envelope/live identity')
    path = Path(traces_path)
    wrapper = result
    require(set(wrapper) == {'decision', 'trace', 'ruleset_id', 'catalogue_sha256', 'profile_id', 'observer_sha256'} and
            wrapper['profile_id'] == PROFILE_ID and wrapper['ruleset_id'] == rules.RULESET_ID and wrapper['catalogue_sha256'] == rules.CATALOGUE_SHA256 and
            wrapper['observer_sha256'] == digest(observer), 'Wrapper observer/rules identity')
    require(path.stat().st_size == wrapper['trace']['bytes'] and file_hash(path) == wrapper['trace']['sha256'], 'Compressed trace identity')
    decision = wrapper['decision']; record = decision['record']
    require(set(decision) == {'action', 'record'} and set(record) == {'root_identity', 'planner', 'privacy', 'schedule',
            'candidates', 'proposal', 'scores', 'worlds', 'profile', 'ruleset_id', 'catalogue_sha256', 'choice', 'means'} and
            record['profile'] == LOCAL_PROFILE and record['ruleset_id'] == rules.RULESET_ID and
            record['catalogue_sha256'] == rules.CATALOGUE_SHA256, 'Decision/profile fields')
    root = world_from_observer(observer)
    require(observer['observation']['turn_index'] == observer['context']['identity'][2], 'Turn/context identity')
    state = rules.deserialize_snapshot(root)
    legal_objects = rules.legal_actions(state); legal = [rules.serialize_action(a) for a in legal_objects]
    raw, logits, artifact = root_scores(state, legal_objects, f"anchor_{observer['context']['fit']}")
    proposed = derive_proposal(observer, root, legal, raw, logits, artifact)
    require(record['proposal'] == proposed and record['scores'] == raw and record['candidates'] == proposed['candidates'] and
            record['schedule'] == proposed['schedule'] and record['root_identity'] == observer['context']['identity'], 'Proposal/model/schedule reconstruction')
    privacy = {name: sum(r['source'] != 'visible' for r in observer['observation']['players'][seat]['reserves'])
               for name, seat in [('own', state.current_player), ('opponent', 1-state.current_player)]}
    require(record['privacy'] == privacy and record['planner'] == 'reference', 'Planner/privacy role')
    support = len(proposed['candidates']['actions'])
    require(len(record['worlds']) == 128 and wrapper['trace']['records'] == 128 * support, 'Delivered world/trace count')
    traces, transitions, choices = 0, 0, 0
    with gzip.open(path, 'rt', encoding='utf-8') as stream:
        for wi, (seed, saved) in enumerate(zip(proposed['schedule']['world_seeds'], record['worlds'])):
            boundary(); world = world_from_observer(observer, seed)
            require(saved['seed'] == seed and saved['state_sha256'] == digest(world['state']) and len(saved['outcomes']) == support, 'World seed/state coupling')
            for ci, action in enumerate(proposed['candidates']['actions']):
                line = stream.readline(8 * 1024**2)
                require(line and line.endswith('\n'), 'Missing/truncated/oversized trace row')
                row = json.loads(line)
                require(type(row['world_index']) is int and row['world_index'] == wi and type(row['candidate_index']) is int and
                        row['candidate_index'] == ci and row['seed'] == seed, 'Trace canonical identity/order')
                expected, raw_count = audit_trajectory(world, action, row, boundary)
                wdl = 0.0 if expected['winner'] is None else (1.0 if expected['winner'] == state.current_player else -1.0)
                require(saved['outcomes'][ci] == {**expected, 'wdl': wdl}, 'Recorded continuation disagrees with independent replay')
                traces += 1; transitions += expected['turns']; choices += raw_count
        require(stream.read(1) == '', 'Extra continuation traces')
    choice, means = select_outcomes(record['worlds'], support)
    require(type(record['choice']) is int and record['choice'] == choice and record['means'] == means and
            decision['action'] == proposed['candidates']['actions'][choice], 'Root means/first-maximum/action')
    boundary()
    return {'status': 'passed', 'profile_id': PROFILE_ID, 'ruleset_id': rules.RULESET_ID,
            'catalogue_sha256': rules.CATALOGUE_SHA256, 'observer_sha256': digest(observer), 'result_sha256': digest(wrapper),
            'decision_sha256': digest(decision), 'trace_sha256': file_hash(path), 'trace_bytes': path.stat().st_size,
            'audited_worlds': 128, 'candidate_support': support, 'audited_continuations': traces,
            'audited_transitions': transitions, 'raw035_choices_reexecuted': choices,
            'proposal_model_reexecuted': True, 'proposal_selection_independent': True,
            'all_continuations_replayed': True, 'transition_oracle': 'frozen_external_screen_dictionary_ledger',
            'shared_dependencies': ['corrected legal enumeration/catalogue/serialization', 'frozen feature encoders and checkpoints'],
            'old_outcomes_used_as_ground_truth': False}
