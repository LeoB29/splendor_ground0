"""Adapter boundaries with synthetic observers/fake inference; no opponent search."""
from copy import deepcopy
from dataclasses import replace
import json
from pathlib import Path
from types import SimpleNamespace
import sys

import pytest

from tools.external_strength_screen_20261018 import rules
from tools.external_strength_screen_20261018_02 import common as frozen_common
from tools.local_player_qualification_128.clock import PersistentWorker
from tools.splendorust_120_qualification_20261018 import common as c, worker


@pytest.fixture(scope='module')
def baseline():
    return c.base_profile()


@pytest.fixture
def profile(baseline, monkeypatch):
    monkeypatch.setattr(c, 'base_profile', lambda: deepcopy(baseline))
    return c.make_profile(256)


@pytest.mark.parametrize('quota', c.QUOTA_LADDER)
def test_profile_changes_only_declared_work_deadlines_and_identity(baseline, monkeypatch, quota):
    monkeypatch.setattr(c, 'base_profile', lambda: deepcopy(baseline))
    value = c.make_profile(quota)
    assert c.validate_profile(value) is value
    assert value['artifacts'] == baseline['artifacts']
    assert value['artifacts'] is not baseline['artifacts']
    assert {k: v for k, v in value.items() if k in baseline and k not in c.PROFILE_CHANGES} == {
        k: v for k, v in baseline.items() if k not in c.PROFILE_CHANGES}
    assert value['internal_budget_seconds'] == 108 and value['external_budget_seconds'] == 120


@pytest.mark.parametrize('quota', [True, 0, -1, 16, 255, 257, 1024.0, '256', 6401, 2**32])
def test_unregistered_quotas_rejected(profile, quota):
    with pytest.raises(ValueError): c.make_profile(quota)
    profile['iterations'] = quota
    with pytest.raises(ValueError): c.validate_profile(profile)


@pytest.mark.parametrize('field,value', [
    ('internal_budget_seconds', 109), ('internal_budget_seconds', 108.0),
    ('external_budget_seconds', 121), ('external_budget_seconds', True),
    ('profile_id', 'old-profile'), ('schema', 'external_opponent_profile_v1'),
    ('depth', 65), ('cpuct', .5), ('world_pool', 4), ('chance_universes', 4),
    ('dynamic_fpu', False), ('fpu_reduction', .03), ('uniform_prior', 1),
    ('inference_batch_size', 8), ('fresh_agent_per_turn', False),
    ('deterministic_fixed_work', 1), ('checkpoint', 'other'), ('native_binary', 'other'),
    ('model_definition', 'other'), ('source_pin', 'other'), ('opponent_id', 'other'),
    ('artifacts', []), ('base_profile', {}),
])
def test_changed_configuration_or_artifacts_fail(profile, field, value):
    profile[field] = value
    with pytest.raises(ValueError): c.validate_profile(profile)


def test_unknown_or_missing_profile_fields_fail(profile):
    profile['secret_extra_search_option'] = True
    with pytest.raises(ValueError): c.validate_profile(profile)
    del profile['secret_extra_search_option']; del profile['world_pool']
    with pytest.raises(ValueError): c.validate_profile(profile)


def test_loader_binds_profile_hash_then_all_inherited_artifacts(tmp_path, monkeypatch, profile):
    monkeypatch.setattr(c, 'ROOT', tmp_path)
    path = tmp_path / 'profile.json'
    path.write_text(json.dumps(profile), encoding='utf-8')
    verified = []
    monkeypatch.setattr(c, 'verify', lambda items: verified.append(deepcopy(items)))
    assert c.load_profile(path, c.file_hash(path)) == profile
    assert verified == [profile['artifacts']]
    with pytest.raises(ValueError, match='hash changed'): c.load_profile(path, '0' * 64)
    assert len(verified) == 1


def test_frozen_loader_still_rejects_extended_deadline(tmp_path, profile):
    path = tmp_path / 'profile.json'; path.write_text(json.dumps(profile), encoding='utf-8')
    with pytest.raises(ValueError, match='Invalid deterministic'): frozen_common.load_profile(path)


def test_clock_inherits_exact_frozen_sampler_request_and_kill_paths():
    assert c.ProcessTreeWorker._sample is frozen_common.ProcessTreeWorker._sample
    assert c.ProcessTreeWorker.request is PersistentWorker.request
    assert c.ProcessTreeWorker.close is PersistentWorker.close


def test_new_constructor_uses_new_loader_without_initializing_old_loader(monkeypatch, profile):
    observed = {}
    def read_profile(path, sha):
        observed['profile'] = (path, sha); return profile
    def fake_initialize(instance, *args, **kwargs):
        observed['initializing'] = instance._initializing
        observed['native'] = instance._native_executable
        observed['kwargs'] = kwargs
    monkeypatch.setattr(c, 'load_profile', read_profile)
    monkeypatch.setattr(c, 'file_hash', lambda path: 'f' * 64)
    monkeypatch.setattr(PersistentWorker, '__init__', fake_initialize)
    monkeypatch.setattr(frozen_common, 'load_profile', lambda *args: pytest.fail('Old profile loader called'))
    value = c.ProcessTreeWorker('dummy', 'unused', env={
        'SPLENDOR_OPPONENT_PROFILE': 'new-profile', 'SPLENDOR_OPPONENT_PROFILE_SHA256': 'new-sha'})
    assert observed['profile'] == ('new-profile', 'new-sha') and observed['initializing'] is True
    assert observed['native'] == (c.ROOT / profile['native_binary']).resolve()
    assert value._initializing is False


@pytest.fixture
def decision(profile):
    state = rules.CorrectedEnv(seed=20261018).initial_state()
    observer = rules.observation_payload(state)
    legal = rules.legal_actions(state)
    diagnostic = {'opponent_id': 'splendorust', 'profile_id': profile['profile_id'],
                  'observer_seed': int(c.digest(observer)[:16], 16), 'phases': [],
                  'inference_calls': 0, 'inference_rows': 0, 'forced_pass': False}
    calls = []
    def choose(payload):
        calls.append(payload); return legal[0], diagnostic
    context = {'opponent': SimpleNamespace(choose=choose), 'profile': profile,
               'profile_sha256': 'f' * 64, 'used': False}
    return context, {'player': 'opponent', 'observer': observer}, legal, diagnostic, calls


def test_success_retains_exact_action_diagnostics_and_observer_identity(decision):
    context, request, legal, diagnostics, calls = decision
    result = worker.handle(context, request)
    assert result == {'status': 'completed', 'action': c.serialize_action(legal[0]),
        'diagnostics': diagnostics, 'profile_sha256': 'f' * 64, 'player': 'opponent',
        'observer_sha256': c.digest(request['observer']), 'legal_actions_sha256': c.legal_digest(legal),
        'legal_count': len(legal), 'ruleset': c.RULESET_ID, 'catalogue_sha256': c.CATALOGUE_SHA256}
    assert result['diagnostics'] is diagnostics and calls == [request['observer']]
    with pytest.raises(ValueError, match='Exactly one'): worker.handle(context, request)
    assert len(calls) == 1


def test_internal_exhaustion_returns_no_partial_action_or_diagnostics(decision):
    context, request, _, _, _ = decision
    def exhausted(_): raise TimeoutError(worker.INTERNAL_TIMEOUT_MESSAGE)
    context['opponent'].choose = exhausted
    result = worker.handle(context, request)
    assert result['status'] == 'internal_budget_exhausted'
    assert result['internal_budget_seconds'] == 108
    assert result['profile_id'] == context['profile']['profile_id']
    assert result['observer_sha256'] == c.digest(request['observer'])
    assert 'action' not in result and 'diagnostics' not in result
    with pytest.raises(ValueError, match='Exactly one'): worker.handle(context, request)


@pytest.mark.parametrize('error', [TimeoutError('unrelated transport timeout'),
    ValueError('ADAPTER_PREREQUISITE: malformed support'), RuntimeError('native exited'), KeyError('field')])
def test_non_budget_errors_are_not_classified_as_exhaustion(decision, error):
    context, request, _, _, _ = decision
    def fail(_): raise error
    context['opponent'].choose = fail
    with pytest.raises(type(error), match=str(error).strip("'")): worker.handle(context, request)


@pytest.mark.parametrize('damage', ['raw035', 'extra', 'hidden_state', 'profile', 'illegal'])
def test_input_and_output_boundary_fails_closed(decision, damage):
    context, request, legal, diagnostics, calls = decision
    if damage == 'raw035': request['player'] = 'raw035'
    elif damage == 'extra': request['environment_seed'] = 42
    elif damage == 'hidden_state': request['observer']['hidden_actual_state'] = {}
    elif damage == 'profile': diagnostics['profile_id'] = 'changed'
    else:
        invalid = replace(legal[0], noble_id='N99')
        context['opponent'].choose = lambda payload: (invalid, diagnostics)
    with pytest.raises(ValueError): worker.handle(context, request)
    if damage in ('raw035', 'extra', 'hidden_state'): assert not calls


def test_initialize_authenticates_before_importing_or_constructing_model(monkeypatch, profile):
    from tools.external_strength_screen_20261018_02 import splendorust_adapter
    events = []
    monkeypatch.setenv('SPLENDOR_OPPONENT_PROFILE', 'new-profile')
    monkeypatch.setenv('SPLENDOR_OPPONENT_PROFILE_SHA256', 'f' * 64)
    def load(path, sha): events.append('authenticate'); return profile
    monkeypatch.setattr(c, 'load_profile', load)
    monkeypatch.setitem(sys.modules, 'torch', SimpleNamespace(
        set_num_threads=lambda n: events.append(('threads', n)),
        set_num_interop_threads=lambda n: events.append(('interop', n))))
    def construct(document):
        events.append('opponent'); assert document is profile; return object()
    monkeypatch.setattr(splendorust_adapter, 'Opponent', construct)
    context = worker.initialize()
    assert events == ['authenticate', ('threads', 1), ('interop', 1), 'opponent']
    assert set(context) == {'opponent', 'profile', 'profile_sha256', 'used'} and context['used'] is False
