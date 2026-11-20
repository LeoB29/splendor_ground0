"""Per-choice public preparation, preserving the pinned bridge semantics.

The caller's observer is copied before validation. The context retains only
immutable serialized public data and frozen Actions; it contains no referee
state, future draws, or cross-choice cache. Native projections returned to
callers and callbacks are detached from the context.
"""
from copy import deepcopy
from dataclasses import dataclass, field
import hashlib
import json

from tools.external_strength_screen_20261018 import rules
from tools.external_strength_screen_20261018_02 import bridge as old

IMPLEMENTATION_ID = 'entity_per_decision_public_preparation_reuse_20261023_v1'
require = old.require
canonical = old.canonical
COLORS, TOKENS, NONE = old.COLORS, old.TOKENS, old.NONE
CARD_TO_RUST, NOBLE_TO_RUST, RUST_CARD_IDS = old.CARD_TO_RUST, old.NOBLE_TO_RUST, old.RUST_CARD_IDS


@dataclass(frozen=True, slots=True)
class Prepared:
    """Opaque, per-decision context; caches themselves are immutable strings."""
    _observer_json: str = field(repr=False)
    legal: tuple = field(repr=False)
    _native_json: str | None = field(default=None, init=False, repr=False)
    _main_json: str | None = field(default=None, init=False, repr=False)

    @property
    def observer_sha256(self):
        return hashlib.sha256(self._observer_json.encode()).hexdigest()


def prepare(observer):
    require(type(observer) is dict and set(observer) == {'schema', 'ruleset_id', 'catalogue_sha256', 'observation'},
            'observer envelope allowlist')
    snapshot = deepcopy(observer)
    require(rules.CATALOGUE_SHA256 == old.CATALOGUE_SHA256, 'corrected catalogue changed')
    state = rules.state_from_observation(snapshot)
    require(not state.terminal, 'terminal state has no decision')
    legal = tuple(rules.legal_actions(state))
    require(bool(legal), 'authoritative legal set is empty')
    return Prepared(canonical(snapshot), legal)


def encode_prepared(context):
    require(type(context) is Prepared, 'unrecognized prepared context')
    if context._native_json is None:
        view = json.loads(context._observer_json)['observation']
        start, actor = view['start_player'], view['current_player']
        players, counts = [old._empty_player() for _ in range(4)], [0]*4
        for seat, source in enumerate(view['players']):
            index = (seat-start) % 2
            reserves = [{'card': NONE if r['card_id'] is None else CARD_TO_RUST[r['card_id']],
                         'tier': r['tier']-1, 'public': r['source'] == 'visible'} for r in source['reserves']]
            counts[index] = len(reserves)
            players[index] = {'tokens': [source['tokens'][c] for c in TOKENS],
                'bonuses': [source['bonuses'][c] for c in COLORS], 'score': source['score'],
                'owned': sorted(CARD_TO_RUST[c] for c in source['purchased_card_ids']),
                'nobles': sorted(NOBLE_TO_RUST[n] for n in source['noble_ids']),
                'reserved': reserves+[old._empty_reserve() for _ in range(3-len(reserves))]}
        native = {'count': 2, 'viewer': (actor-start) % 2, 'current': (actor-start) % 2,
            'bank': [view['bank_tokens'][c] for c in TOKENS],
            'market': [value for tier in ('1', '2', '3')
                       for value in [CARD_TO_RUST[c] for c in view['visible_tier_cards'][tier]]+
                       [NONE]*(4-len(view['visible_tier_cards'][tier]))],
            'remaining': [view['deck_counts'][str(t)] for t in (1, 2, 3)],
            'nobles': sorted(NOBLE_TO_RUST[n] for n in view['nobles']), 'players': players,
            'reserved_counts': counts, 'turns': view['turn_index'], 'phase': {'kind': 'Main'},
            'final_round': view['pending_round_end']}
        object.__setattr__(context, '_native_json', canonical(native))
    return json.loads(context._native_json)


def main_prepared(context):
    require(type(context) is Prepared, 'unrecognized prepared context')
    if context._main_json is None:
        values = {canonical(old._main(a)): old._main(a) for a in context.legal}
        result = sorted(values.values(), key=old._main_order)
        if any(a['kind'] == 'Pass' for a in result):
            require(result == [{'kind': 'Pass'}], 'PASS permitted only when forced')
        object.__setattr__(context, '_main_json', canonical(result))
    return json.loads(context._main_json)


def resolve_prepared(context, selected_main, select_aux):
    """Pinned compound resolution using one original support and fresh projection."""
    require(type(context) is Prepared and context._native_json is not None and context._main_json is not None,
            'prepare, encode and Main construction must precede resolution')
    legal = context.legal
    selected_main = deepcopy(selected_main)
    require(canonical(selected_main) in {canonical(a) for a in json.loads(context._main_json)}, 'invalid Main choice')
    candidates = [a for a in legal if canonical(old._main(a)) == canonical(selected_main)]
    # Parsing the immutable cache gives this resolution an independent mutable
    # copy; it cannot alter Main observation hashes or another auxiliary call.
    observation = json.loads(context._native_json)
    actor = observation['current']
    player = observation['players'][actor]
    kind = selected_main['kind']
    card_id = None
    if kind in ('BuyVisible', 'BuyReserved'):
        source_kind = 'Market' if kind == 'BuyVisible' else 'Reserved'
        index = selected_main['index']
        observation['phase'] = {'kind': 'Payment', 'source': {'kind': source_kind, 'index': index}}
        payments = sorted({tuple(old._vector(a.spend_tokens, COLORS)) for a in candidates}, reverse=True)
        chosen = old._choose_aux(observation, [{'kind': 'Pay', 'tokens': list(p)} for p in payments], select_aux)
        candidates = [a for a in candidates if old._vector(a.spend_tokens, COLORS) == chosen['tokens']]
        paid = old._vector(candidates[0].spend_tokens)
        for color, amount in enumerate(paid):
            player['tokens'][color] -= amount
            observation['bank'][color] += amount
        if kind == 'BuyVisible':
            card_id = observation['market'][index]
            observation['market'][index] = NONE
        else:
            card_id = player['reserved'].pop(index)['card']
            player['reserved'].append(old._empty_reserve())
            observation['reserved_counts'][actor] -= 1
        card = rules.corrected_cards_by_id()[RUST_CARD_IDS[card_id]]
        player['bonuses'][COLORS.index(card.bonus_color)] += 1
        player['score'] += card.points
        player['owned'] = sorted(player['owned']+[card_id])
    elif kind in ('Take', 'ReserveVisible', 'ReserveDeck'):
        gained = old._vector(candidates[0].take_tokens)
        for color, amount in enumerate(gained):
            player['tokens'][color] += amount
            observation['bank'][color] -= amount
        if kind != 'Take':
            if kind == 'ReserveVisible':
                index = selected_main['index']
                card_id, tier, public = observation['market'][index], index//4, True
                observation['market'][index] = NONE
            else:
                card_id, tier, public = NONE, selected_main['tier'], False
                observation['remaining'][tier] -= 1
            slot = observation['reserved_counts'][actor]
            player['reserved'][slot] = {'card': card_id, 'tier': tier, 'public': public}
            observation['reserved_counts'][actor] += 1
        returns = sorted({tuple(old._vector(a.return_tokens)) for a in candidates})
        if any(returns[0]):
            observation['phase'] = {'kind': 'Return'}
            chosen = old._choose_aux(observation, [{'kind': 'Return', 'tokens': list(r)} for r in returns], select_aux)
            candidates = [a for a in candidates if old._vector(a.return_tokens) == chosen['tokens']]
            for color, amount in enumerate(chosen['tokens']):
                player['tokens'][color] -= amount
                observation['bank'][color] += amount
    else:
        require(kind == 'Pass' and all(a.action_type.name == 'PASS' for a in legal), 'unforced PASS')
    noble_ids = {a.noble_id for a in candidates}
    if noble_ids != {None}:
        require(None not in noble_ids, 'inconsistent compulsory noble alternatives')
        observation['phase'] = {'kind': 'Noble'}
        values = [{'kind': 'Noble', 'index': n} for n in sorted(NOBLE_TO_RUST[n] for n in noble_ids)]
        chosen = old._choose_aux(observation, values, select_aux)
        candidates = [a for a in candidates if NOBLE_TO_RUST[a.noble_id] == chosen['index']]
    require(len(candidates) == 1 and candidates[0] in legal, 'compound choice is ambiguous or illegal')
    return candidates[0]
