"""Public-information, corrected-rules adapter for the pinned Breck policy.

No model is loaded here. The upstream encoding is preserved, including its
token-relative noble features. Its game engine is never used as the referee.
"""
from __future__ import annotations

from collections import Counter
from copy import deepcopy
import hashlib
import itertools
import json

import numpy as np

from tools.external_strength_screen_20261018 import rules
from tools import future_training_rules_20261021 as training_rules

PROFILE_ID = 'breck_numpy_policy_corrected_public_v1'
COMPLETION_DOMAIN = 'breck_public_completion_20261026_v1'
RETURN_DOMAIN = 'breck_prereveal_returns_20261026_v1'
COLORS = ('white', 'blue', 'green', 'red', 'black')
TOKENS = COLORS + ('gold',)


def require(condition, message):
    if not condition:
        raise ValueError(message)


def digest(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(',', ':'),
                                    allow_nan=False).encode()).hexdigest()


def array_digest(value):
    return hashlib.sha256(np.ascontiguousarray(value).tobytes()).hexdigest()


def _table():
    result = []
    for family, combinations, width in (
            ('take_three', itertools.combinations(COLORS, 3), 4),
            ('take_double', ((c, c) for c in COLORS), 3),
            ('take_two', itertools.combinations(COLORS, 2), 3),
            ('take_one', ((c,) for c in COLORS), 2)):
        for combo in combinations:
            for returns in range(width):
                result.append(dict(index=len(result), family=family,
                                   take_tokens=list(combo), return_count=returns))
    result.append(dict(index=95, family='forced_pass_extension'))
    for tier in (1, 2, 3):
        for slot in range(4):
            for gold in (False, True):
                result.append(dict(index=len(result), family='buy_visible', tier=tier,
                                   market_index=slot, with_gold=gold))
    for slot in range(3):
        for gold in (False, True):
            result.append(dict(index=len(result), family='buy_reserved', reserved_index=slot,
                               with_gold=gold))
    for tier in (1, 2, 3):
        for slot in range(5):
            result.append(dict(index=len(result), family='reserve_deck' if slot == 4 else 'reserve_visible',
                               tier=tier, market_index=None if slot == 4 else slot))
    assert len(result) == 141
    return tuple(result)


ACTION_TABLE = _table()
ACTION_TABLE_SHA256 = digest(ACTION_TABLE)


def lawful_completion(observer):
    """Exactly one observation-seeded allocation, never actual private data.

    Hash-sort each tier's unseen IDs, assign opponent blind slots in slot order,
    and retain the remaining IDs as a dummy deck. No hidden draw is inspected.
    """
    state = rules.state_from_observation(observer)  # Strict privacy/schema gate.
    require(not state.terminal, 'Breck cannot decide a terminal position')
    observation_hash = digest(observer)
    view = observer['observation']
    cards = rules.corrected_cards_by_id()
    assigned = []
    for tier in (1, 2, 3):
        slots = [(seat, slot) for seat, player in enumerate(view['players'])
                 for slot, reserve in enumerate(player['reserves'])
                 if reserve['card_id'] is None and reserve['tier'] == tier]
        unseen = [c.card_id for c in state.hidden_tier_decks[tier]]
        unseen += [state.players[s].reserved_cards[i].card_id for s, i in slots]
        unseen.sort(key=lambda card_id: (digest([COMPLETION_DOMAIN, observation_hash, tier, card_id]), card_id))
        for (seat, slot), card_id in zip(slots, unseen):
            state.players[seat].reserved_cards[slot] = cards[card_id]
            assigned.append(dict(seat=seat, slot=slot, tier=tier, card_id=card_id))
        state.hidden_tier_decks[tier] = [cards[key] for key in unseen[len(slots):]]
    snapshot = rules.serialize_snapshot(state)
    training_rules.decode_training_snapshot(snapshot)
    require(rules.observation_payload(state) == observer, 'Completion changes public information')
    return state, snapshot, assigned


def _card_vector(card, effective):
    reward = np.zeros(5, dtype=int)
    reward[COLORS.index(card.bonus_color)] = 1
    cost = np.array([card.cost.get(c, 0) for c in TOKENS], dtype=int)
    return np.concatenate((reward, [card.points / 15], np.maximum(cost - effective, 0)[:5] / 4))


def _player_vector(player):
    gems = np.array([player.tokens[c] for c in TOKENS], dtype=int)
    cards = np.array([player.bonuses.get(c, 0) for c in TOKENS], dtype=int)
    vector = np.zeros(47, dtype=np.float32)
    vector[:6] = gems / 4.0
    vector[5] /= 1.25
    vector[6] = gems.sum() / 10.0
    vector[7:13] = cards
    vector[12] = cards.sum() / 10
    for index, card in enumerate(player.reserved_cards):
        vector[13 + 11*index:24 + 11*index] = _card_vector(card, gems + cards)
    vector[-1] = player.score / 15
    return vector


def encode_state(state):
    """Exact source arithmetic and packing, including zero-padding empty slots."""
    actor = state.players[state.current_player]
    effective = np.array([actor.tokens[c] + actor.bonuses.get(c, 0) for c in TOKENS], dtype=int)
    gems = np.array([state.bank_tokens[c] for c in TOKENS], dtype=int)
    board = np.zeros(157, dtype=np.float32)
    board[:6] = gems / 4.0
    board[5] /= 1.25
    board[6] = gems.sum() / 10.0
    for tier in (1, 2, 3):
        for slot, card in enumerate(state.visible_tier_cards[tier]):
            start = 7 + ((tier-1)*4 + slot)*11
            board[start:start+11] = _card_vector(card, effective)
    for slot, noble in enumerate(state.nobles):
        cost = np.array([noble.requirement.get(c, 0) for c in TOKENS], dtype=int)
        board[139+6*slot:145+6*slot] = np.concatenate(([noble.points], np.maximum(cost-effective, 0)[:5])) / 4.0
    return np.concatenate((board, _player_vector(actor), _player_vector(state.players[1-state.current_player]))).astype(np.float32)


def _spend(actor, card):
    cost = [max(0, card.cost.get(c, 0) - actor.bonuses[c]) for c in COLORS]
    payment = {c: min(actor.tokens[c], n) for c, n in zip(COLORS, cost)}
    payment['gold'] = sum(cost) - sum(payment.values())
    return tuple(c for c in TOKENS for _ in range(payment[c]))


def _returns(actor, gain, observation_hash, index):
    """Source preference and uniform-choice algorithm with a private local RNG.

    The seed is fixed by lawful observation and output index; no global RNG,
    referee hidden state, market refill or blind-reserve identity is consulted.
    """
    seed = int(digest([RETURN_DOMAIN, observation_hash, index])[:8], 16)
    rng = np.random.RandomState(seed)
    gain_counts = Counter(gain)
    held = {c: actor.tokens[c] + gain_counts[c] for c in TOKENS}
    result = []
    for _ in range(max(0, sum(held.values())-10)):
        eligible = [c for c in COLORS if held[c] > 0 and gain_counts[c] == 0]
        if not eligible:
            eligible = [c for c in COLORS if held[c] > 0]
        require(bool(eligible), 'No source-compatible token return')
        color = eligible[int(rng.choice(len(eligible)))]
        held[color] -= 1
        result.append(color)
    return tuple(sorted(result, key=TOKENS.index))


def source_mask(state):
    actor = state.players[state.current_player]
    result = np.zeros(141, dtype=bool)
    for descriptor in ACTION_TABLE:
        i, family = descriptor['index'], descriptor['family']
        if family.startswith('take_'):
            gain = descriptor['take_tokens']
            enough = (state.bank_tokens[gain[0]] >= 4 if family == 'take_double'
                      else all(state.bank_tokens[c] >= n for c, n in Counter(gain).items()))
            result[i] = enough and descriptor['return_count'] == max(0, actor.token_count + len(gain)-10)
        elif family == 'forced_pass_extension':
            result[i] = actor.token_count == 10  # Original standalone discard support, not our final mask.
        elif family.startswith('buy_'):
            if family == 'buy_visible':
                values, slot = state.visible_tier_cards[descriptor['tier']], descriptor['market_index']
            else:
                values, slot = actor.reserved_cards, descriptor['reserved_index']
            if slot < len(values):
                needed = Counter(_spend(actor, values[slot]))['gold']
                result[i] = needed <= actor.tokens['gold'] if descriptor['with_gold'] else needed == 0
        else:
            if len(actor.reserved_cards) < 3:
                result[i] = (state.deck_counts[descriptor['tier']] > 0 if family == 'reserve_deck'
                             else descriptor['market_index'] < len(state.visible_tier_cards[descriptor['tier']]))
    return result


def prepare(observer):
    state, completion, assigned = lawful_completion(observer)
    observation_hash = digest(observer)
    actor = state.players[state.current_player]
    legal = rules.legal_actions(state)
    require(bool(legal), 'No corrected legal decisions')
    serialized = [rules.serialize_action(action) for action in legal]
    upstream_mask = source_mask(state)
    actions = [None] * 141
    for descriptor in ACTION_TABLE:
        i, family = descriptor['index'], descriptor['family']
        if family == 'forced_pass_extension':
            eligible = [a for a in legal if a.action_type.name == 'PASS']
        elif not upstream_mask[i]:
            continue
        else:
            action_type = {'buy_visible': 'BUY_VISIBLE', 'buy_reserved': 'BUY_RESERVED',
                           'reserve_visible': 'RESERVE_VISIBLE', 'reserve_deck': 'RESERVE_DECK'}.get(family, 'TAKE_TOKENS')
            eligible = [a for a in legal if a.action_type.name == action_type]
            for key in ('tier', 'market_index', 'reserved_index'):
                if key in descriptor:
                    eligible = [a for a in eligible if getattr(a, key) == descriptor[key]]
            if family.startswith('buy_'):
                card = (state.visible_tier_cards[descriptor['tier']][descriptor['market_index']]
                        if family == 'buy_visible' else actor.reserved_cards[descriptor['reserved_index']])
                spend = Counter(_spend(actor, card))
                eligible = [a for a in eligible if Counter(a.spend_tokens) == spend]
            else:
                gain = descriptor['take_tokens'] if family.startswith('take_') else (['gold'] if state.bank_tokens['gold'] else [])
                returns = Counter(_returns(actor, gain, observation_hash, i))
                eligible = [a for a in eligible if Counter(a.take_tokens) == Counter(gain) and Counter(a.return_tokens) == returns]
        if eligible:
            # A single compulsory noble: stable lexical local ID, before hidden reveal.
            eligible.sort(key=lambda a: (a.noble_id or '', digest(rules.serialize_action(a))))
            actions[i] = rules.serialize_action(eligible[0])
    mask = np.array([a is not None for a in actions], dtype=bool)
    require(bool(mask.any()), 'Corrected legal set has no supported Breck output')
    features = encode_state(state)
    require(features.shape == (251,) and features.dtype == np.float32 and np.isfinite(features).all(), 'Invalid Breck features')
    return dict(features=features, legal_mask=mask, source_mask=upstream_mask, actions=tuple(actions),
                observation_sha256=observation_hash, completion=completion,
                completion_sha256=digest(completion), completion_assignments=assigned,
                feature_sha256=array_digest(features), mask_sha256=array_digest(mask),
                legal_actions_sha256=digest(serialized), legal_count=len(legal))


def _result(prepared, logits):
    values = np.asarray(logits)
    require(values.shape == (141,) and values.dtype == np.float32 and np.isfinite(values).all(),
            'Policy must return exactly141 finite float32 logits')
    selected = int(np.argmax(np.where(prepared['legal_mask'], values, -np.inf)))
    return dict(player='breck', observer_sha256=prepared['observation_sha256'],
                legal_actions_sha256=prepared['legal_actions_sha256'], legal_count=prepared['legal_count'],
                action=deepcopy(prepared['actions'][selected]), diagnostics={
                    'schema': 'breck_public_trace_v1', 'profile_id': PROFILE_ID,
                    'ruleset_id': rules.RULESET_ID, 'catalogue_sha256': rules.CATALOGUE_SHA256,
                    'turn_contract': training_rules.TURN_CONTRACT,
                    'action_table_sha256': ACTION_TABLE_SHA256,
                    'completion_domain': COMPLETION_DOMAIN, 'return_domain': RETURN_DOMAIN,
                    'completion_sha256': prepared['completion_sha256'],
                    'completion_assignments': prepared['completion_assignments'],
                    'feature_sha256': prepared['feature_sha256'], 'mask_sha256': prepared['mask_sha256'],
                    'features': prepared['features'].tolist(), 'mask': prepared['legal_mask'].tolist(),
                    'source_mask': prepared['source_mask'].tolist(), 'logits': values.tolist(),
                    'logits_sha256': array_digest(values), 'selected_index': selected})


def decide(observer, predict):
    prepared = prepare(observer)
    return _result(prepared, predict(prepared['features']))


def validate_result(result, observer):
    """Model-free trace reconstruction; does not certify neural arithmetic."""
    require(isinstance(result, dict) and isinstance(result.get('diagnostics'), dict), 'Invalid Breck result')
    logits = result['diagnostics'].get('logits')
    require(isinstance(logits, list) and len(logits) == 141 and
            all(type(value) in (int, float) and np.isfinite(value) for value in logits), 'Invalid recorded logits')
    values = np.asarray(logits, dtype=np.float32)
    require(values.tolist() == logits, 'Recorded logits are not exact float32 values')
    expected = _result(prepare(observer), values)
    require(digest(result) == digest(expected), 'Breck trace/action/observation binding differs')
    return {'status': 'passed', 'scope': 'saved_encoding_mask_argmax_legality_no_neural_forward',
            'selected_index': expected['diagnostics']['selected_index']}


validate_trace = validate_result
