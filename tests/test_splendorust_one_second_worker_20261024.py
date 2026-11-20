"""Model-free counterexamples for observer routing and complete-response guards."""
from copy import deepcopy
import json
from pathlib import Path
from types import SimpleNamespace

import pytest

from tools.external_strength_screen_20261018 import rules
from tools.external_strength_screen_20261018.common import ROOT
from tools.splendorust_one_second_20261024 import adapters, worker


def observer():
    return rules.observation_payload(rules.CorrectedEnv(seed=240124).initial_state())


class StubPolicy:
    request_deadline = None
    calls = 0

    def __init__(self, arm='greedy', illegal=False):
        self.arm, self.illegal = arm, illegal

    def choose(self, payload, *, diagnostic_capture=False):
        self.calls += 1
        state = rules.state_from_observation(payload)
        action = rules.CorrectedEnv().legal_actions(state)[0]
        if self.illegal:
            action = None
        return action, {'profile_id': adapters.GREEDY_PROFILE_ID if self.arm == 'greedy'
                        else adapters.SEARCH_PROFILE_ID, 'capture': diagnostic_capture}


def context(policy):
    return {'adapters': {'greedy': policy, 'search': policy},
            'diagnostic_capture': False, 'request_started': None}


def test_all_saved_positions_route_through_corrected_observer():
    fixtures = json.loads((ROOT / 'outputs/external_strength_screen_20261018_02/fixtures.json').read_text())
    assert len(fixtures) == 23
    for fixture in fixtures:
        state = rules.deserialize_snapshot(fixture['snapshot'])
        payload = rules.observation_payload(state)
        assert rules.observation_payload(worker.decode_observer(payload)) == payload


@pytest.mark.parametrize('mutate', [
    lambda p: p.pop('ruleset_id'),
    lambda p: p.update(ruleset_id='historical'),
    lambda p: p.update(catalogue_sha256='0' * 64),
    lambda p: p.update(referee_seed=123),
    lambda p: p.update(context={}),
    lambda p: p['observation'].update(hidden_tier_decks={}),
])
def test_invalid_observer_never_reaches_policy(mutate):
    payload = observer()
    mutate(payload)
    policy = StubPolicy()
    with pytest.raises((ValueError, KeyError)):
        worker.handle(context(policy), {'player': 'greedy', 'observer': payload})
    assert policy.calls == 0


@pytest.mark.parametrize('payload_request', [
    {'player': 'raw035', 'observer': {}},
    {'player': 'greedy', 'observer': {}, 'seed': 1},
    {'operation': 'train'},
    [], None,
])
def test_request_allowlist_rejects_extra_secrets_and_unknown_operations(payload_request):
    policy = StubPolicy()
    with pytest.raises(ValueError, match='allowlist'):
        worker.handle(context(policy), payload_request)
    assert policy.calls == 0


@pytest.mark.parametrize('arm', ['search', 'greedy'])
def test_complete_response_binds_corrected_identity_and_legal_support(arm):
    policy = StubPolicy(arm)
    payload = observer()
    result = worker.handle(context(policy), {'player': arm, 'observer': payload})
    assert result['player'] == arm and result['observer_sha256'] == worker.digest(payload)
    assert result['ruleset'] == rules.RULESET_ID
    assert result['catalogue_sha256'] == rules.CATALOGUE_SHA256
    assert result['turn_contract'] == worker.TURN_CONTRACT
    assert result['legal_count'] > 0
    assert result['profile_sha256'] == (adapters.PROFILE_SHA256 if arm == 'search' else None)
    assert policy.request_deadline is None


def test_illegal_complete_action_fails_closed_and_clears_deadline():
    policy = StubPolicy(illegal=True)
    with pytest.raises(ValueError, match='PLAYER_ILLEGAL_ACTION'):
        worker.handle(context(policy), {'player': 'greedy', 'observer': observer()})
    assert policy.request_deadline is None


def test_worker_final_evidence_generation_is_inside_internal_allowance(monkeypatch):
    policy = StubPolicy()
    ctx = context(policy)
    ctx['request_started'] = 10.0
    monkeypatch.setattr(worker.time, 'perf_counter', lambda: 10.900001)
    with pytest.raises(TimeoutError, match='complete response'):
        worker.handle(ctx, {'player': 'greedy', 'observer': observer()})
    assert policy.request_deadline is None


def test_greedy_keeps_logits_outputs_and_frozen_trace_without_position_tokens(monkeypatch):
    expected = {'phases': [{'canonical_logits': [1.0] * 67,
                          'inference': [{'tokens': [7.0], 'output83': [2.0] * 83,
                                         'input_sha256': 'x', 'output_sha256': 'y'}]}]}
    monkeypatch.setattr(adapters.GreedyOpponent, 'choose',
        lambda self, payload, diagnostic_capture: ('action', deepcopy(expected)))
    policy = object.__new__(adapters.SharedGreedyOpponent)
    action, compact = policy.choose({})
    assert action == 'action'
    assert compact['phases'][0]['canonical_logits'] == [1.0] * 67
    assert compact['phases'][0]['inference'][0]['output83'] == [2.0] * 83
    assert 'tokens' not in compact['phases'][0]['inference'][0]
    assert policy.choose({}, diagnostic_capture=True)[1] == expected
    assert policy.reference_choose({}, diagnostic_capture=True)[1] == expected
    assert policy.active_deadline is None


def test_search_calls_exact_frozen_forward_and_captures_only_first_input(monkeypatch):
    outputs = [[float(i) for i in range(83)]]
    calls = []
    def frozen(self, tokens, batch):
        calls.append((tokens, batch))
        return outputs
    monkeypatch.setattr(adapters.Opponent, '_infer', frozen)
    policy = object.__new__(adapters.SearchOpponent)
    policy.capture, policy.callbacks = True, []
    assert policy._infer([1.0], 1) is outputs
    policy._infer([2.0], 1)
    assert calls == [([1.0], 1), ([2.0], 1)]
    assert policy.callbacks[0]['tokens'] == [1.0]
    assert policy.callbacks[0]['output83'] == outputs[0]
    assert 'tokens' not in policy.callbacks[1]


def test_adapter_guard_rejects_expired_parent_deadline_before_native_call(monkeypatch):
    calls = []
    monkeypatch.setattr(adapters.Opponent, 'choose', lambda *args: calls.append(args))
    policy = object.__new__(adapters.SearchOpponent)
    policy.request_deadline = 0.0
    with pytest.raises(TimeoutError, match='ONE_SECOND_INTERNAL_DEADLINE'):
        policy.choose({})
    assert not calls


def test_batch_memory_admission_failure_never_reaches_native_operation(monkeypatch):
    calls = []
    monkeypatch.setattr(adapters.Opponent, 'choose', lambda *args: calls.append(args))
    policy = object.__new__(adapters.SearchOpponent)
    def shortage():
        raise RuntimeError('STUDY_RESOURCE: available RAM')
    policy.admission_guard = shortage
    with pytest.raises(RuntimeError, match='STUDY_RESOURCE'):
        policy.choose({})
    assert not calls


def test_original_search_deadline_remains_an_attributable_deadline_failure(monkeypatch):
    def original_timeout(*args):
        raise TimeoutError('Internal fixed-work deadline exceeded')
    monkeypatch.setattr(adapters.Opponent, 'choose', original_timeout)
    policy = object.__new__(adapters.SearchOpponent)
    with pytest.raises(TimeoutError, match='ONE_SECOND_INTERNAL_DEADLINE') as error:
        policy.choose({})
    assert isinstance(error.value.__cause__, TimeoutError)
    assert policy.active_deadline is None and policy.capture is False


def test_readiness_rejects_missing_native_even_when_role_was_previously_owned():
    with pytest.raises(ValueError, match='three exact live compute roles'):
        adapters.validate_ready_metadata({}, expected_sources={}, worker_pid=10,
            process_tree=[{'role': 'python', 'pid': 10}, {'role': 'greedy', 'pid': 12}])


def test_readiness_binds_model_arrangement_sources_and_native_pids(monkeypatch, tmp_path):
    class FakeSearch:
        def __init__(self, profile):
            self.model, self.reference = SimpleNamespace(training=False), {}
            self.proc = SimpleNamespace(pid=11)
        def close(self): pass
    class FakeGreedy:
        def __init__(self, folder, search):
            self.model, self.reference = search.model, search.reference
            self.proc = SimpleNamespace(pid=12)
    monkeypatch.setattr(adapters, 'SearchOpponent', FakeSearch)
    monkeypatch.setattr(adapters, 'SharedGreedyOpponent', FakeGreedy)
    _, identity = adapters.initialize_adapters(tmp_path)
    metadata = {**identity, 'sources': {'runtime.py': 'expected'}, 'pid': 10, 'threads': 1,
                'ruleset_id': worker.RULESET_ID, 'catalogue_sha256': worker.CATALOGUE_SHA256,
                'turn_contract': worker.TURN_CONTRACT}
    tree = [{'role': 'python', 'pid': 10}, {'role': 'search', 'pid': 11}, {'role': 'greedy', 'pid': 12}]
    assert adapters.validate_ready_metadata(metadata, expected_sources={'runtime.py': 'expected'},
        worker_pid=10, process_tree=tree)['status'] == 'passed'
    for field, value in [('model_arrangement', 'two independent models'),
                         ('sources', {'runtime.py': 'changed'}),
                         ('native_pids', {'search': 111, 'greedy': 12}), ('threads', True)]:
        bad = {**metadata, field: value}
        with pytest.raises(ValueError, match='ready metadata differs'):
            adapters.validate_ready_metadata(bad, expected_sources={'runtime.py': 'expected'},
                worker_pid=10, process_tree=tree)
