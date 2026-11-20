"""Model-free fixture and qualification guard tests; never load checkpoints."""
from copy import deepcopy

import pytest

from tools.move_evaluation_20261023 import qualification as q
from tools.external_strength_screen_20261018 import rules


@pytest.fixture(scope='module')
def fixtures():
    return q.build_fixtures()


def test_all_required_structural_fixtures_are_strictly_decodable(fixtures):
    assert len(fixtures) == 15
    assert {f['id'] for f in fixtures if 'corrected_face' in f['tags']} == {
        'face_' + key for key in rules.CORRECTED_COSTS}
    for fixture in fixtures:
        state = q.guard.decode_training_snapshot(fixture['snapshot'])
        assert rules.serialize_snapshot(state) == fixture['snapshot']


def test_semantic_edges_include_natural_noble_finish_and_gold(fixtures):
    checked = q._semantic_edges(fixtures)
    assert {'noble_final_round', 'gold_payment', 'gold_return', 'multiple_nobles'} <= set(checked)


def test_every_concrete_fixture_action_has_compound_and_independent_transition_path(fixtures):
    phases, count = set(), 0
    for fixture in fixtures:
        n, observed = q._compound_all(q.guard.decode_training_snapshot(fixture['snapshot']))
        count += n
        phases.update(observed)
    assert count > 100
    assert phases == {'Payment', 'Return', 'Noble'}


def test_all_six_faces_reach_actual_python_feature_offsets(fixtures):
    for fixture in fixtures:
        if 'corrected_face' in fixture['tags']:
            report = q._encoded_costs(q.guard.decode_training_snapshot(fixture['snapshot']))
            assert fixture['tags'][1] in report['visible_cards']


def test_old_or_malformed_identity_and_unfinished_targets_fail_closed(fixtures):
    rejected = q.identity_rejections(fixtures[0])
    assert len(rejected) == 15
    assert 'data hash mismatch' in rejected
    assert 'bare historical snapshot' in rejected


def test_hidden_reserve_variant_preserves_observer_not_referee(fixtures):
    fixture = next(f for f in fixtures if f['id'] == 'private_reserves')
    state = q.guard.decode_training_snapshot(fixture['snapshot'])
    variant = q._private_variant(state)
    assert rules.observation_payload(state) == rules.observation_payload(variant)
    assert rules.serialize_snapshot(state) != rules.serialize_snapshot(variant)


def test_reference_mapping_is_complete_and_does_not_use_auxiliary_logits():
    policy = list(range(81))
    mapped = q.canonical_logits(policy)
    assert len(mapped) == 67
    assert mapped[1] == 30 and mapped[3] == 35 and mapped[7] == 45
    assert mapped[31] == -1e8
    assert mapped[32:37] == policy[55:60]
    assert mapped[37:49] == policy[12:24]
    assert mapped[49:52] == policy[24:27]
    assert mapped[52:64] == policy[:12]
    assert mapped[64:67] == policy[27:30]
    assert all(value < 60 for value in mapped)
    with pytest.raises(ValueError): q.canonical_logits([0.] * 80)
    with pytest.raises(ValueError): q.canonical_logits([float('nan')] * 81)


def test_first_max_reference_preserves_ties_in_legal_order():
    actions = [{'kind': 'BuyVisible', 'index': 2}, {'kind': 'ReserveDeck', 'tier': 1}]
    logits = q.canonical_logits([0.] * 81)
    assert max(range(2), key=lambda i: logits[q.canonical_index(actions[i])]) == 0
    assert q.canonical_index({'kind': 'Take', 'tokens': [0, 2, 0, 0, 0]}) == 33
    assert q.canonical_index({'kind': 'Take', 'tokens': [1, 0, 1, 0, 1]}) == 21


def test_mutated_afterstate_is_rejected_by_independent_audit(fixtures):
    state = q.guard.decode_training_snapshot(fixtures[0]['snapshot'])
    action = rules.legal_actions(state)[0]
    after = rules.serialize_snapshot(rules.CorrectedEnv().step(state, action))
    bad = deepcopy(after); bad['state']['bank_tokens']['gold'] += 1
    with pytest.raises(ValueError):
        q.guard.check_training_transition(fixtures[0]['snapshot'], rules.serialize_action(action), bad)


def test_model_free_receipt_never_claims_production_admission():
    receipt = q.model_free_checks()
    assert receipt['status'] == 'model_free_checks_passed'
    assert receipt['production_entrypoint_qualified'] is False
    assert len(receipt['runtime_sources']) == 8


def test_actual_mixed_trajectory_route_reaches_both_policies_and_natural_terminal(fixtures, monkeypatch):
    from tools.move_evaluation_20261023 import evaluator
    class FakeGreedy:
        def choose(self, observer):
            state = rules.state_from_observation(observer)
            return rules.legal_actions(state)[0], {'fixture': 'model_free'}
    monkeypatch.setattr(evaluator.q, 'score_actions', lambda state, legal: [-float(i) for i in range(len(legal))])
    report = q.verify_actual_trajectory(FakeGreedy(), fixtures)
    assert report['status'] == 'passed'
    assert report['independent_transitions'] == 3
    assert [row['policy'] for row in report['trace']] == ['forced', 'greedy_entity', 'raw035']
