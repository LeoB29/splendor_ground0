"""Live boundary/transfer fixtures with fake neural models only."""
from copy import deepcopy
import gzip
import json
from types import SimpleNamespace

import pytest
import torch

from tools.live_game_operational_20261018 import common as c, local, worker
from tools.corrected_planner_qualification_20261018 import local as old
from tools.external_strength_screen_20261018 import rules, audit as ledger


class FakeRaw:
    def score_actions(self, state, legal):
        return tuple(0.0 for _ in legal)


class FakeProposal:
    config = SimpleNamespace(observation_size=500, action_feature_size=200)
    def __call__(self, observation, features, mask):
        return torch.zeros((1, features.shape[1])), torch.zeros(1)


@pytest.fixture
def fake_models(monkeypatch):
    monkeypatch.setattr(old, '_INITIALIZED', True)
    monkeypatch.setattr(old, '_RAW', FakeRaw())
    monkeypatch.setattr(old, '_MODELS', {f'anchor_{fit}': FakeProposal() for fit in c.FITS})
    monkeypatch.setattr(old, '_ARTIFACTS', {
        f'anchor_{fit}': {'sha256': str(fit), 'source_fit': fit} for fit in c.FITS})


def near_terminal():
    state = rules.CorrectedEnv(seed=20261018).initial_state()
    selected = [card for card in rules.corrected_cards_by_id().values() if card.points == 5][:3]
    ids = {card.card_id for card in selected}
    for tier in (1, 2, 3):
        for groups in (state.visible_tier_cards, state.hidden_tier_decks):
            groups[tier] = [card for card in groups[tier] if card.card_id not in ids]
        while len(state.visible_tier_cards[tier]) < 4 and state.hidden_tier_decks[tier]:
            state.visible_tier_cards[tier].append(state.hidden_tier_decks[tier].pop())
        state.deck_counts[tier] = len(state.hidden_tier_decks[tier])
    for card in selected:
        state.players[0].purchased_cards.append(card)
        state.players[0].score += card.points
        state.players[0].bonuses[card.bonus_color] += 1
    state.current_player, state.turn_index, state.pending_round_end = 1, 1, True
    ledger.validate_ledger(rules.serialize_snapshot(state))
    return state


def observe(state=None, game=0):
    state = near_terminal() if state is None else state
    return local.make_observer_payload(rules.serialize_snapshot(state), c.operational_context(game, state.turn_index))


def write_trace(path, rows):
    with path.open('wb') as raw:
        with gzip.GzipFile(filename='', mode='wb', fileobj=raw, mtime=0, compresslevel=3) as stream:
            for row in rows: stream.write((c.canonical(row) + '\n').encode())


def wrapper(observer, decision, path, rows):
    write_trace(path, rows)
    return {'decision': decision, 'trace': {'sha256': c.file_hash(path), 'bytes': path.stat().st_size,
            'records': len(rows)}, 'ruleset_id': c.RULESET_ID, 'catalogue_sha256': c.CATALOGUE_SHA256,
            'profile_id': c.LOCAL_PROFILE_ID, 'observer_sha256': c.digest(observer)}


def test_all_three_models_keep_pair_fit_and_separate_game_streams():
    assert [c.local_seat(g) for g in range(6)] == [1, 0, 0, 1, 1, 0]
    for pair, fit in enumerate(c.FITS):
        first, second = [c.operational_context(2*pair+leg, 0) for leg in (0, 1)]
        assert first['fit'] == second['fit'] == fit
        assert first['stream'] != second['stream']
        assert first['pair'] == second['pair'] == pair


@pytest.mark.parametrize('damage', ['fit', 'stream', 'pair', 'identity', 'turn', 'namespace', 'extra', 'actor'])
def test_live_context_and_observer_boundary_fail_closed(damage):
    payload = observe()
    if damage == 'actor': payload['observation']['current_player'] = 0
    elif damage == 'extra': payload['context']['referee_seed'] = 42
    elif damage == 'fit': payload['context']['fit'] = c.FITS[1]
    elif damage == 'stream': payload['context']['stream'] = '0'*64
    elif damage == 'pair': payload['context']['pair'] = True
    elif damage == 'identity': payload['context']['identity'][0] = 'main'
    elif damage == 'turn': payload['observation']['turn_index'] += 1
    else: payload['context']['namespace'] = 'different'
    with pytest.raises(ValueError): local.state_from_observer_payload(payload)


def test_legacy_transfer_admits_only_bound_observations_and_exact_old_context():
    for index in c.TRANSFER_INDICES:
        original = deepcopy(c.legacy_observer(index))
        payload = {**original, 'schema': c.SCHEMA, 'context': c.legacy_context(index)}
        local.state_from_observer_payload(payload)
        bad = deepcopy(payload); bad['context']['stream'] = '0'*64
        with pytest.raises(ValueError): local.state_from_observer_payload(bad)
        bad = deepcopy(payload); bad['observation']['turn_index'] += 2
        with pytest.raises(ValueError): local.state_from_observer_payload(bad)
    with pytest.raises(ValueError): c.legacy_context(1)


@pytest.mark.parametrize('game', [0, 3, 4])
def test_full128_orchestration_preserves_policy_ties_and_hidden_order(fake_models, game):
    state = near_terminal(); observer = observe(state, game)
    traces = []
    decision = local.choose(observer, trace_sink=traces.append)
    assert decision['record']['proposal']['model_id'] == f"anchor_{c.FITS[game//2]}"
    assert len(decision['record']['worlds']) == 128 and len(traces) == 512
    assert decision['record']['choice'] == 0
    assert all(row['turns'] == 1 and len(row['steps']) == 1 for row in traces)
    changed = deepcopy(state)
    for values in changed.hidden_tier_decks.values(): values.reverse()
    assert observe(changed, game) == observer


def test_identical_legacy_primitives_transfer_except_profile_label(fake_models, monkeypatch, tmp_path):
    # Synthetic legacy fixture exercises transfer numerically without saved-root rollouts.
    state = near_terminal()
    pair = next(p for p in range(201) if old.historical._public_fit_and_stream(p)[0] == c.FITS[0])
    fit, stream = old.historical._public_fit_and_stream(pair)
    context = {'fit': fit, 'stream': stream, 'identity': ['main', 2*pair, state.turn_index]}
    old_observer = old.make_observer_payload(rules.serialize_snapshot(state), context)
    monkeypatch.setattr(c, 'legacy_observer', lambda index: deepcopy(old_observer))
    real_read = c.read
    monkeypatch.setattr(c, 'read', lambda path: deepcopy(old_observer)
        if 'observations' in str(path) else real_read(path))
    live_observer = local.make_observer_payload(rules.serialize_snapshot(state), c.legacy_context(0))
    before_trace, after_trace = [], []
    before = old.choose(old_observer, False, trace_sink=before_trace.append)
    after = local.choose(live_observer, trace_sink=after_trace.append)
    assert after['record']['profile'] == c.LOCAL_PROFILE_ID
    after['record']['profile'] = before['record']['profile']
    assert after == before and after_trace == before_trace
    one, two = tmp_path/'one.gz', tmp_path/'two.gz'
    write_trace(one, before_trace); write_trace(two, after_trace)
    assert one.read_bytes() == two.read_bytes()


def test_worker_writes_exact_trace_inside_result_and_rejects_reuse(tmp_path, monkeypatch):
    monkeypatch.setattr(c, 'OUTPUT', tmp_path)
    path = tmp_path/'continuations.jsonl.gz'
    monkeypatch.setenv('LIVE_TRACE_PATH', str(path))
    decision = {'action': {'synthetic': True}, 'record': {}}
    def choose(payload, *, boundary, trace_sink):
        boundary(); trace_sink({'test': 'trace'}); return decision
    context = {'mode': 'local', 'local': SimpleNamespace(choose=choose), 'used': False}
    result = worker.handle(context, {'observer': {'public': 'fixture'}})
    assert result['decision'] == decision and result['trace']['records'] == 1
    assert result['trace']['sha256'] == c.file_hash(path)
    assert gzip.decompress(path.read_bytes()) == b'{"test":"trace"}\n'
    with pytest.raises(ValueError, match='Exactly one'): worker.handle(context, {'observer': {}})


def test_worker_trace_boundary_never_overwrites_or_leaves_output(tmp_path, monkeypatch):
    monkeypatch.setattr(c, 'OUTPUT', tmp_path/'allowed')
    for path in (tmp_path/'continuations.jsonl.gz', tmp_path/'allowed'/'other.gz'):
        with pytest.raises(ValueError): worker.trace_path(path)
    path = tmp_path/'allowed'/'continuations.jsonl.gz'; path.parent.mkdir(); path.write_bytes(b'preserve')
    monkeypatch.setenv('LIVE_TRACE_PATH', str(path))
    context = {'mode': 'local', 'local': SimpleNamespace(choose=lambda *_: pytest.fail('Overwritten')), 'used': False}
    with pytest.raises(FileExistsError): worker.handle(context, {'observer': {}})
    assert path.read_bytes() == b'preserve'


def test_worker_initialization_rejects_unknown_mode_before_models(monkeypatch):
    monkeypatch.setenv('LIVE_WORKER_MODE', 'opponent')
    monkeypatch.setattr(local, 'initialize', lambda: pytest.fail('Models initialized'))
    with pytest.raises(ValueError): worker.initialize()
