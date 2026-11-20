"""Arm dispatch and independent natural-ending qualification, no checkpoints."""
import pytest
from tools.move_evaluation_64_20261023 import qualification as q
from tools.external_strength_screen_20261018 import rules


class FakeGreedy:
    def choose(self, observer):
        state = rules.state_from_observation(observer)
        return rules.legal_actions(state)[0], {'fixture': 'model_free'}


@pytest.mark.parametrize('arm', ['A', 'B'])
def test_new_arm_trajectory_fixture_covers_both_seats_and_independent_edges(monkeypatch, arm):
    monkeypatch.setattr(q.local, 'score_actions', lambda state, legal: [-float(i) for i in range(len(legal))])
    receipt = q.verify_actual_trajectory(None if arm == 'A' else FakeGreedy(), q.old.build_fixtures(), arm=arm)
    assert receipt['independent_transitions'] == 3 and receipt['result']['reason'] == 'completed'
    if arm == 'A':
        assert all('policy' not in row for row in receipt['trace'])
        assert all(row['scores_sha256'] is not None for row in receipt['trace'][1:])
    else:
        assert [row['policy'] for row in receipt['trace']] == ['forced', 'greedy_entity', 'raw035']


def test_actual_entrypoint_refuses_uninitialized_models(monkeypatch):
    monkeypatch.setattr(q.local, '_INITIALIZED', False)
    with pytest.raises(ValueError, match='models not initialized'):
        q.run_entrypoint_checks(None, arm='A')


def test_actual_selector_contract_calls_new_reproduce_and_rejects_all_variants(monkeypatch):
    from copy import deepcopy
    from tools.move_evaluation_64_20261023 import common as c
    original_read = c.read
    lock = original_read(c.PREP/'selection_lock.json')
    original_roots = [r['payload'] for r in lock['roots']]
    accepted = []
    def reproduce(payload):
        if payload not in original_roots:
            raise ValueError('fixture rejected')
        accepted.append(payload['fit'])
        seeds = payload['selection_schedule']['world_seeds']
        return None, None, {'schedule': {**deepcopy(payload['selection_schedule']), 'world_seeds': seeds+['extension']*64}}, None
    monkeypatch.setattr(q.evaluator, 'reproduce', reproduce)
    receipt = q.selector_contract_checks()
    assert accepted == [161001, 161002, 161003]
    assert receipt['extra_panel_rollouts'] == 0
    assert all(len(r['rejected']) == 7 for r in receipt['actual_reproduction_roots'])
