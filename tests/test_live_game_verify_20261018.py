"""Sampled/full assurance and tamper checks using one-step fake-model roots."""
from copy import deepcopy
import gzip
import json

import pytest

from tools.live_game_operational_20261018 import common as c, local, verify as v, worker
from test_live_game_adapter_20261018 import fake_models, near_terminal, observe, wrapper


@pytest.fixture
def evidence(fake_models, tmp_path):
    observer = observe(); traces = []
    decision = local.choose(observer, trace_sink=traces.append)
    path = tmp_path/'continuations.jsonl.gz'
    result = wrapper(observer, decision, path, traces)
    return observer, result, path, traces


def test_sample_is_identity_only_eight_whole_worlds_and_matches_common():
    for game in range(6):
        identity = c.operational_context(game, 18)['identity']
        indices = v.sample_worlds(identity)
        assert indices == c.sample_worlds(identity) and len(indices) == len(set(indices)) == 8
        assert indices == sorted(indices) and all(0 <= i < 128 for i in indices)
    assert v.sample_worlds(['x', 1, 2]) != v.sample_worlds(['x', 1, 3])


def test_full_and_sampled_proofs_distinguish_recorded_from_replayed(evidence):
    observer, result, path, traces = evidence
    sample = v.audit_decision(observer, result, path, mode='sample')
    assert sample['recorded_worlds'] == 128 and sample['recorded_continuations'] == len(traces) == 512
    assert sample['audited_worlds'] == 8 and sample['audited_continuations'] == 32
    assert sample['audited_transitions'] == 32 and sample['raw035_choices_reexecuted'] == 0
    assert sample['all_continuations_replayed'] is False
    c.check_audit_proof(sample, observer, result, 'sample')
    full = v.audit_decision(observer, result, path, mode='full')
    assert full['audited_world_indices'] == list(range(128))
    assert full['audited_continuations'] == full['audited_transitions'] == len(traces)
    assert full['all_continuations_replayed'] is True
    c.check_audit_proof(full, observer, result, 'full')


def test_two_ply_sample_reexecutes_raw_scores_and_rejects_corruption(fake_models, tmp_path):
    state = v.rules.CorrectedEnv(seed=911).initial_state()
    cards = list(v.rules.corrected_cards_by_id().values())
    chosen = [card for card in cards if card.points == 5][:2] + [card for card in cards if card.points == 4][:1]
    bonuses = {color: sum(card.bonus_color == color for card in chosen) for color in v.frozen.COLORS}
    for color, count in state.nobles[0].requirement.items():
        chosen += sorted([card for card in cards if card.points == 0 and card.bonus_color == color],
                         key=lambda card: card.card_id)[:max(0, count-bonuses[color])]
    ids = {card.card_id for card in chosen}
    for tier in (1, 2, 3):
        for groups in (state.visible_tier_cards, state.hidden_tier_decks):
            groups[tier] = [card for card in groups[tier] if card.card_id not in ids]
        while len(state.visible_tier_cards[tier]) < 4 and state.hidden_tier_decks[tier]:
            state.visible_tier_cards[tier].append(state.hidden_tier_decks[tier].pop())
        state.deck_counts[tier] = len(state.hidden_tier_decks[tier])
    for card in chosen:
        state.players[0].purchased_cards.append(card)
        state.players[0].score += card.points
        state.players[0].bonuses[card.bonus_color] += 1
    v.frozen.ledger.validate_ledger(v.rules.serialize_snapshot(state))
    observer = observe(state, 1); traces = []
    decision = local.choose(observer, trace_sink=traces.append)
    assert all(row['turns'] == 2 for row in traces)
    path = tmp_path/'continuations.jsonl.gz'
    result = wrapper(observer, decision, path, traces)
    proof = v.audit_decision(observer, result, path, mode='sample')
    assert proof['audited_continuations'] == 32 and proof['audited_transitions'] == 64
    assert proof['raw035_choices_reexecuted'] == 32
    index = v.sample_worlds(observer['context']['identity'])[0] * 4
    traces[index]['steps'][1]['scores_sha256'] = '0'*64
    result = wrapper(observer, decision, path, traces)
    with pytest.raises(ValueError, match='score digest differs'):
        v.audit_decision(observer, result, path, mode='sample')


@pytest.mark.parametrize('damage', ['seed', 'world_hash', 'missing_outcome', 'winner_bool', 'wdl',
    'termination', 'turns', 'outcome_hash', 'choice', 'means', 'proposal', 'extra_trace', 'missing_trace',
    'trace_actor', 'trace_forced', 'trace_first_choice', 'trace_digest'])
def test_all_world_record_checks_reject_unsampled_corruption(evidence, damage):
    observer, result, path, traces = evidence
    selected = set(v.sample_worlds(observer['context']['identity']))
    wi = next(i for i in range(128) if i not in selected)
    saved = result['decision']['record']['worlds'][wi]
    row = traces[wi*4]
    if damage == 'seed': saved['seed'] += 'different'
    elif damage == 'world_hash': saved['state_sha256'] = '0'*64
    elif damage == 'missing_outcome': saved['outcomes'].pop()
    elif damage == 'winner_bool': saved['outcomes'][0]['winner'] = False
    elif damage == 'wdl': saved['outcomes'][0]['wdl'] *= -1
    elif damage == 'termination': saved['outcomes'][0]['reason'] = 'max_turns'
    elif damage == 'turns': saved['outcomes'][0]['turns'] = 0
    elif damage == 'outcome_hash': saved['outcomes'][0]['final_state_sha256'] = 'short'
    elif damage == 'choice': result['decision']['record']['choice'] = True
    elif damage == 'means': result['decision']['record']['means'][0] += 1/128
    elif damage == 'proposal': result['decision']['record']['proposal']['learned_logits'][0] += 1
    elif damage == 'extra_trace': traces.append(deepcopy(traces[-1]))
    elif damage == 'missing_trace': traces.pop()
    elif damage == 'trace_actor': row['steps'][0]['actor'] = 0
    elif damage == 'trace_forced': row['steps'][0]['forced'] = False
    elif damage == 'trace_first_choice': row['steps'][0]['selected_index'] += 1
    else: row['trace_sha256'] = '0'*64
    updated = wrapper(observer, result['decision'], path, traces)
    with pytest.raises((ValueError, KeyError)): v.audit_decision(observer, updated, path, mode='sample')


def test_sampled_branch_mismatch_rejected_even_if_headers_and_aggregates_agree(evidence):
    observer, result, path, traces = evidence
    wi = v.sample_worlds(observer['context']['identity'])[0]
    row = traces[wi*4]; outcome = result['decision']['record']['worlds'][wi]['outcomes'][0]
    row['final_state_sha256'] = outcome['final_state_sha256'] = '0'*64
    updated = wrapper(observer, result['decision'], path, traces)
    with pytest.raises(ValueError, match='outcome mismatch'):
        v.audit_decision(observer, updated, path, mode='sample')


def test_unsampled_trajectory_is_explicitly_not_verified_but_full_replay_catches_it(evidence):
    observer, result, path, traces = evidence
    selected = set(v.sample_worlds(observer['context']['identity']))
    wi = next(i for i in range(128) if i not in selected)
    traces[wi*4]['final_state_sha256'] = result['decision']['record']['worlds'][wi]['outcomes'][0]['final_state_sha256'] = '0'*64
    updated = wrapper(observer, result['decision'], path, traces)
    proof = v.audit_decision(observer, updated, path, mode='sample')
    assert proof['all_continuations_replayed'] is False
    with pytest.raises(ValueError, match='outcome mismatch'):
        v.audit_decision(observer, updated, path, mode='full')


@pytest.mark.parametrize('damage', ['fit', 'seat', 'turn', 'stream', 'legacy_outside'])
def test_verifier_context_is_independent_and_fail_closed(damage):
    observer = observe()
    if damage == 'fit': observer['context']['fit'] = c.FITS[1]
    elif damage == 'seat': observer['observation']['current_player'] = 0
    elif damage == 'turn': observer['context']['turn'] += 1
    elif damage == 'stream': observer['context']['stream'] = '0'*64
    else: observer['context'] = {**c.legacy_context(0), 'fixture_index': 1}
    with pytest.raises(ValueError): v.check_context(observer['context'], observer['observation'])


def test_frozen_legacy_contexts_admit_exact_saved_lawful_views():
    for index in c.TRANSFER_INDICES:
        view = c.legacy_observer(index)['observation']
        assert v.check_context(c.legacy_context(index), view) == c.legacy_context(index)


def test_worker_audit_mode_passes_exact_mode_and_trace_binding(tmp_path, monkeypatch):
    monkeypatch.setattr(c, 'OUTPUT', tmp_path)
    path = tmp_path/'continuations.jsonl.gz'; path.write_bytes(b'synthetic')
    calls = []
    def audit(observer, result, traces_path, *, mode, boundary):
        calls.append((observer, result, traces_path, mode)); return {'status': 'fixture'}
    monkeypatch.setattr(v, 'audit_decision', audit)
    context = {'mode': 'audit', 'used': False}
    payload = {'observer': {}, 'result': {}, 'traces_path': str(path), 'mode': 'full'}
    assert worker.handle(context, payload) == {'status': 'fixture'}
    assert calls == [({}, {}, path, 'full')]
    with pytest.raises(ValueError, match='Exactly one'): worker.handle(context, payload)
