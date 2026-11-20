"""Pinned upstream encoder and policy reference on the same lawful surrogate.

Only selected class ASTs are compiled: no GUI, deck construction, shuffling or
upstream game transitions. Inference loading/forward are explicit lazy calls.
"""
from __future__ import annotations

import ast
from functools import lru_cache
import hashlib
import itertools
from pathlib import Path
from types import SimpleNamespace

import numpy as np

from . import adapter
from tools.external_strength_screen_20261018 import rules

ROOT = Path(__file__).resolve().parents[2]
ASSET_ROOT = ROOT / 'external/breck_webapp_6ec66ef'
COMPONENTS = 'Splendor/Environment/Splendor_components/'
SOURCE_HASHES = {
    COMPONENTS+'Board_components/deck.py': 'da3bf4635fa335d9d0e451c6da909de499c745a77faa705c654add9a9cc6b2b9',
    COMPONENTS+'Board_components/board.py': '3723b7675fd487828664d48b9df5660f630d39edb7d942be1e1c0f4b03dbaf7b',
    COMPONENTS+'Board_components/splendor_cards_data.py': '6a398e2f60a39760ba9d35bed97390593648f8534395aa751dc7681302142a89',
    COMPONENTS+'Player_components/player.py': 'ae4ff8022b56e28e1d529b5315dcf1b37a300967d798c7f2202e3018336c5a0f',
    'Splendor/Environment/gui_game.py': '36efd3111434a815079ea86a194f5a759c9f0d772de8b060a5e837883e03245b',
    'Splendor/RL/inference_model.py': 'f5f593a88996f658b73470304dbc44b012349d407b4b76d7aee68c34b756becf',
}
WEIGHTS = ASSET_ROOT / 'Splendor/RL/trained_agents/inference_model.npz'
WEIGHTS_SHA256 = '5044383cadd98aa6c753b9d0a2bd8557b1ad017c21b6d8ba762027f63277438a'


def _tree(relative):
    raw = (ASSET_ROOT / relative).read_bytes()
    adapter.require(hashlib.sha256(raw).hexdigest() == SOURCE_HASHES[relative], 'Pinned source changed: '+relative)
    return ast.parse(raw.decode('utf-8'), filename=str(ASSET_ROOT / relative))


def _classes(relative, names):
    tree = _tree(relative)
    nodes = [node for node in tree.body if isinstance(node, ast.ClassDef) and node.name in names]
    adapter.require({node.name for node in nodes} == set(names), 'Missing pinned source classes')
    module = ast.Module(body=[ast.ImportFrom(module='__future__', names=[ast.alias(name='annotations')], level=0)]+nodes,
                        type_ignores=[])
    namespace = {'np': np, 'ndarray': np.ndarray, 'it': itertools}
    exec(compile(ast.fix_missing_locations(module), str(ASSET_ROOT / relative), 'exec'), namespace)
    return namespace


@lru_cache(maxsize=1)
def catalogue_mapping():
    tree = _tree(COMPONENTS+'Board_components/splendor_cards_data.py')
    assignment = next(node for node in tree.body if isinstance(node, ast.Assign) and
                      any(isinstance(target, ast.Name) and target.id == 'PRELOADED_CARD_DATA' for target in node.targets))
    source = ast.literal_eval(assignment.value)
    colors = adapter.COLORS
    cards = {(c.tier-1, colors.index(c.bonus_color), c.points, tuple(c.cost.get(k, 0) for k in colors)): c.card_id
             for c in rules.corrected_cards_by_id().values()}
    nobles = {(n.points, tuple(n.requirement.get(k, 0) for k in colors)): n.noble_id
              for n in rules.corrected_nobles_by_id().values()}
    mapped_cards, mapped_nobles = {}, {}
    for tier in range(3):
        for entry in source[tier]:
            key = (tier, entry['gem'], entry['points'], tuple(entry['cost']))
            adapter.require(key in cards, 'Upstream card face is not in corrected catalogue')
            local = cards[key]
            adapter.require(local not in mapped_cards, 'Nonbijective upstream card catalogue')
            mapped_cards[local] = dict(entry, tier=tier)
    for entry in source['Noble']:
        key = (entry['points'], tuple(entry['cost']))
        adapter.require(key in nobles, 'Upstream noble face differs')
        local = nobles[key]
        adapter.require(local not in mapped_nobles, 'Nonbijective upstream nobles')
        mapped_nobles[local] = dict(entry, tier='Noble')
    adapter.require(len(mapped_cards) == 90 and len(mapped_nobles) == 10, 'Incomplete upstream catalogue')
    return mapped_cards, mapped_nobles


@lru_cache(maxsize=1)
def _encoding_classes():
    deck = _classes(COMPONENTS+'Board_components/deck.py', ['Card', 'Noble'])
    player = _classes(COMPONENTS+'Player_components/player.py', ['Player'])['Player']
    board = _classes(COMPONENTS+'Board_components/board.py', ['Board'])['Board']
    gui = _tree('Splendor/Environment/gui_game.py')
    methods = [node for cls in gui.body if isinstance(cls, ast.ClassDef) for node in cls.body
               if isinstance(node, ast.FunctionDef) and node.name == 'to_state']
    adapter.require(len(methods) == 1, 'Ambiguous upstream game encoder')
    namespace = {'np': np}
    exec(compile(ast.Module(body=methods, type_ignores=[]), str(ASSET_ROOT/'Splendor/Environment/gui_game.py'), 'exec'), namespace)
    return deck['Card'], deck['Noble'], player, board, namespace['to_state']


def surrogate(observer):
    state, _, _ = adapter.lawful_completion(observer)
    Card, Noble, Player, Board, _ = _encoding_classes()
    card_mapping, noble_mapping = catalogue_mapping()
    card_objects = {local: Card(**record) for local, record in card_mapping.items()}
    noble_objects = {local: Noble(**record) for local, record in noble_mapping.items()}
    players = []
    for seat, local in enumerate(state.players):
        player = Player('reference'+str(seat), None, seat)
        player.gems = np.array([local.tokens[c] for c in adapter.TOKENS], dtype=int)
        player.cards = np.array([local.bonuses.get(c, 0) for c in adapter.TOKENS], dtype=int)
        player.reserved_cards = [card_objects[c.card_id] for c in local.reserved_cards]
        player.points = local.score
        players.append(player)
    board = Board.__new__(Board)  # Never runs Deck/SystemRandom or any game action.
    board.gems = np.array([state.bank_tokens[c] for c in adapter.TOKENS], dtype=int)
    board.cards = [[card_objects[c.card_id] for c in state.visible_tier_cards[t]] +
                   [None]*(4-len(state.visible_tier_cards[t])) for t in (1, 2, 3)]
    board.nobles = [noble_objects[n.noble_id] for n in state.nobles] + [None]*(3-len(state.nobles))
    board.decks = [SimpleNamespace(cards=[card_objects[c.card_id] for c in state.hidden_tier_decks[t]]) for t in (1, 2, 3)]
    return SimpleNamespace(board=board, active_player=players[state.current_player],
                           inactive_player=players[1-state.current_player])


def encode(observer):
    game = surrogate(observer)
    return _encoding_classes()[-1](game)


def source_mask(observer):
    game = surrogate(observer)
    return game.active_player.get_legal_moves(game.board)


def load_policy():
    """Explicit post-freeze only entry point. No load/forward on import."""
    adapter.require(hashlib.sha256(WEIGHTS.read_bytes()).hexdigest() == WEIGHTS_SHA256, 'Pinned weights changed')
    cls = _classes('Splendor/RL/inference_model.py', ['InferenceModel'])['InferenceModel']
    return cls(str(WEIGHTS))


def predict(features, policy=None):
    policy = load_policy() if policy is None else policy
    return policy._forward(features).astype(np.float32, copy=False)


def validate_mapping_contract():
    """All141 static descriptors, source dimensions and catalogue; no inference."""
    _, _, Player, _, _ = _encoding_classes()
    p = Player('mapping', None, 0)
    adapter.require((p.take_dim, p.buy_dim, p.reserve_dim, p.action_dim) == (96, 30, 15, 141), 'Source dimensions differ')
    for descriptor in adapter.ACTION_TABLE[:95]:
        index = descriptor['index']
        if index < 40:
            gain, count = p.all_takes_3[index//4], index % 4
        elif index < 55:
            gain, count = p.all_takes_2_same[(index-40)//3], (index-40) % 3
        elif index < 85:
            gain, count = p.all_takes_2_diff[(index-55)//3], (index-55) % 3
        else:
            gain, count = p.all_takes_1[(index-85)//2], (index-85) % 2
        adapter.require(CounterTokens(descriptor['take_tokens']) == tuple(gain) and descriptor['return_count'] == count,
                        'Take index mapping differs')
    for index in range(96, 126):
        d = adapter.ACTION_TABLE[index]
        offset = index-96
        expected = ({'family': 'buy_visible', 'tier': offset//8+1, 'market_index': (offset%8)//2}
                    if offset < 24 else {'family': 'buy_reserved', 'reserved_index': (offset-24)//2})
        adapter.require(all(d[k] == v for k, v in expected.items()) and d['with_gold'] == bool(offset % 2), 'Buy index differs')
    for index in range(126, 141):
        d, offset = adapter.ACTION_TABLE[index], index-126
        adapter.require(d['tier'] == offset//5+1 and d['market_index'] == (None if offset%5 == 4 else offset%5) and
                        d['family'] == ('reserve_deck' if offset%5 == 4 else 'reserve_visible'), 'Reserve index differs')
    adapter.require(adapter.ACTION_TABLE[95] == {'index': 95, 'family': 'forced_pass_extension'}, 'Forced-pass extension differs')
    catalogue_mapping()
    return {'status': 'passed', 'static_indices_checked': 141, 'cards': 90, 'nobles': 10,
            'upstream_discard_index_95_replaced_only_for_forced_pass': True}


def CounterTokens(values):
    return tuple(values.count(c) for c in adapter.TOKENS)
