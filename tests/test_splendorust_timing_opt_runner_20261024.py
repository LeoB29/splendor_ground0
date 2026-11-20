"""Bounded orchestration and two-search-player replay counterexamples; no models."""
from copy import deepcopy
from pathlib import Path
from types import SimpleNamespace

import pytest

from tools.splendorust_timing_opt_20261024 import runner


def test_operational_schedule_is_fresh_paired_and_both64_players_are_candidate():
    rows = runner.operational_schedule()
    assert len(rows) == 12 and len({row['seed_decimal'] for row in rows}) == 6
    for first, second in zip(rows[::2], rows[1::2], strict=True):
        assert first['seed_decimal'] == second['seed_decimal']
        assert first['challenger_seat'] == 1-second['challenger_seat']
        assert first['challenger_mode'] == 'candidate'
        if first['stage'] == 'ops32':
            assert first['challenger_quota'] == first['opponent_quota'] == 32
            assert first['opponent_mode'] == 'baseline'
        else:
            assert first['challenger_quota'] == 64 and first['opponent_quota'] == 32
            assert first['opponent_mode'] == 'candidate'
    assert rows == runner.operational_schedule()


@pytest.mark.parametrize('value', [{}, {'qualified': 1}, {'timing_qualified': None}, {'passed': 'true'}])
def test_gate_rejects_missing_or_nonboolean_admission(value):
    with pytest.raises(ValueError):
        runner.gate_passed(value)


class Envelope:
    def __init__(self):
        self.stage, self.events = None, []
    def begin(self, stage):
        self.stage = stage
        self.events.append(('begin', stage))
    def finish(self):
        self.events.append(('finish', self.stage))
    def guard(self):
        self.events.append(('guard', self.stage))
    def admit(self):
        self.events.append(('admit', self.stage))
    def remaining(self): return 100.
    def timeout(self, allowance): return allowance


@pytest.mark.parametrize('gates, stages, admitted', [
    ([False], ['sampler', 'qualification32', 'final'], False),
    ([True, False], ['sampler', 'qualification32', 'operational32', 'final'], False),
    ([True, True, False], ['sampler', 'qualification32', 'operational32', 'qualification64', 'final'], False),
    ([True, True, True, False], ['sampler', 'qualification32', 'operational32', 'qualification64', 'operational64', 'final'], False),
    ([True, True, True, True], ['sampler', 'qualification32', 'operational32', 'qualification64', 'operational64', 'final'], True),
])
def test_failed_gate_skips_later_collection_but_always_finishes_verification(monkeypatch, gates, stages, admitted):
    envelope, answers, saved = Envelope(), iter(gates), []
    calls = []
    monkeypatch.setattr(runner, 'qualify_sampler', lambda *args: calls.append('sampler'))
    monkeypatch.setattr(runner, 'qualify32', lambda *args: {'qualified': next(answers)})
    monkeypatch.setattr(runner, 'qualify64', lambda *args: {'qualified': next(answers)})
    monkeypatch.setattr(runner, 'operational', lambda *args: {'qualified': next(answers)})
    monkeypatch.setattr(runner, 'final_checks', lambda *args: calls.append(('final', deepcopy(args[-1]))))
    monkeypatch.setattr(runner.c, 'save', lambda path, value: saved.append((path.name, value)))
    result = runner.run_stages(envelope, {}, {}, [])
    assert [stage for event, stage in envelope.events if event == 'begin'] == stages
    assert calls[0] == 'sampler' and calls[-1][0] == 'final'
    assert calls[-1][1] == result
    outcome = saved[-1][1]
    assert outcome['optimized64_qualified'] is admitted
    assert outcome['strength_games_launched'] == 0
    assert outcome['next_strength_comparison'] == ('64versus32' if admitted else None)


def test_sampler_startup_clock_is_not_reset(monkeypatch):
    envelope = Envelope()
    envelope.begin('sampler')
    monkeypatch.setattr(runner, 'qualify_sampler', lambda *args: None)
    monkeypatch.setattr(runner, 'qualify32', lambda *args: {'qualified': False})
    monkeypatch.setattr(runner, 'final_checks', lambda *args: None)
    monkeypatch.setattr(runner.c, 'save', lambda *args: None)
    runner.run_stages(envelope, {}, {}, [], sampler_started=True)
    assert envelope.events.count(('begin', 'sampler')) == 1


def test_hard_failure_stops_without_retry_or_later_model_work(monkeypatch):
    envelope, calls = Envelope(), []
    monkeypatch.setattr(runner, 'qualify_sampler', lambda *args: None)
    def failure(*args):
        calls.append('once')
        raise TimeoutError('one-second failure')
    monkeypatch.setattr(runner, 'qualify32', failure)
    monkeypatch.setattr(runner, 'operational', lambda *args: pytest.fail('Must not continue'))
    with pytest.raises(TimeoutError, match='one-second'):
        runner.run_stages(envelope, {}, {}, [])
    assert calls == ['once']


def test_failed_request_is_saved_before_assertion_or_clock_guard(monkeypatch):
    events = []
    monkeypatch.setattr(runner.c, 'save', lambda path, value: events.append(('saved', value)))
    envelope = SimpleNamespace(guard=lambda: events.append(('guard', None)))
    row = {'completed': False, 'clock': {'status': 'timeout', 'detail': 'preserved'}}
    with pytest.raises(ValueError, match='retained without retries'):
        runner.save_and_check(row, Path('unused'), envelope)
    assert events == [('saved', row), ('guard', None)]


def test_failed_actual_entrypoint_is_saved_and_not_retried(monkeypatch):
    saved, calls = [], []
    receipt = {'status': 'worker_error', 'detail': 'counterexample failed'}
    child = SimpleNamespace(request=lambda payload, allowance: calls.append((payload, allowance)) or receipt)
    monkeypatch.setattr(runner.c, 'save', lambda path, value: saved.append(value))
    with pytest.raises(ValueError, match='entrypoint failed without retry'):
        runner.actual_entrypoint(child, Envelope(), 'candidate')
    assert calls == [({'operation': 'entrypoint_checks'}, 300)] and saved == [receipt]


def test_no_new_move_begins_with_only_partial_stage_allowance(monkeypatch, tmp_path):
    from tools.splendorust_search_scale_20261024 import qualification
    state = qualification.reference._initial()
    child = SimpleNamespace(_counter=0, output_dir=tmp_path, challenger_quota=32,
                            request=lambda *args, **kwargs: pytest.fail('Partial move clock forbidden'))
    envelope = Envelope()
    monkeypatch.setattr(envelope, 'remaining', lambda: .999)
    monkeypatch.setattr(runner.c, 'save', lambda *args: None)
    with pytest.raises(ValueError, match='complete one-second move'):
        runner.decision(child, state, envelope)


def mock_game(monkeypatch):
    states = [{'state': {'current_player': i % 2, 'terminal': i == 2, 'winner': 1 if i == 2 else None}}
              for i in range(3)]
    spec = dict(stage='ops64', pair_index=0, game_index=0, challenger_seat=0,
                challenger_quota=64, opponent_quota=32, challenger_mode='candidate', opponent_mode='candidate')
    rows = []
    for i in range(2):
        action, trace = {'move': i}, {'immutable': i}
        rows.append(dict(ply=i, seat=i, arm='search', game_role='challenger' if i == 0 else 'opponent',
            quota=64 if i == 0 else 32, mode='candidate', stage='ops64', pair_index=0, game_index=0,
            before=states[i], after=states[i+1], observer={'seat': i}, action=action, trace=trace,
            completed=True, clock={'status': 'ok', 'result': {'action': action, 'diagnostics': trace}}))
    ending = dict(**spec, plies=2, reason='natural', winner=1, final=states[-1])
    validations = []
    monkeypatch.setattr(runner, 'decode_training_snapshot', lambda value: value)
    monkeypatch.setattr(runner.ledger, 'check_observer', lambda *args: None)
    monkeypatch.setattr(runner.ledger, 'transition', lambda before, action: states[action['move']+1])
    monkeypatch.setattr(runner.audit, 'check_clock', lambda receipt: None)
    monkeypatch.setattr(runner.audit, 'validate_result', lambda result, observer, arm, expected_quota:
                        validations.append((arm, expected_quota)))
    return states, spec, rows, ending, validations


def test_independent_game_replay_uses_search64_and_search32_not_reference16(monkeypatch):
    states, spec, rows, ending, validations = mock_game(monkeypatch)
    result = runner.replay_game(states[0], rows, ending, spec)
    assert result['status'] == 'passed' and result['transitions'] == 2
    assert validations == [('search', 64), ('search', 32)]


@pytest.mark.parametrize('field, value', [('quota', 16), ('mode', 'baseline'), ('arm', 'reference')])
def test_operational_replay_rejects_wrong_opponent_identity(monkeypatch, field, value):
    states, spec, rows, ending, _ = mock_game(monkeypatch)
    rows[1][field] = value
    with pytest.raises(ValueError, match='player identity changed'):
        runner.replay_game(states[0], rows, ending, spec)


def test_operational_replay_rejects_disconnected_action_and_response(monkeypatch):
    states, spec, rows, ending, _ = mock_game(monkeypatch)
    rows[1]['action'] = {'move': 0}
    with pytest.raises(ValueError, match='result coupling changed'):
        runner.replay_game(states[0], rows, ending, spec)


def test_sampler_mismatch_is_preserved_with_independent_values_before_failure(monkeypatch):
    class Child:
        _failure = None
        def __enter__(self): return self
        def __exit__(self, *args): pass
        def _sample(self):
            return {'process_tree': [{'pid': 3, 'created': 4., 'parent_pid': 2, 'observed_threads': 1}]}
    class Process:
        pid = 3
        def create_time(self): return 4.
        def ppid(self): return 2
        def num_threads(self): return 9
    saved = []
    monkeypatch.setattr(runner, 'make_worker', lambda *args, **kwargs: Child())
    monkeypatch.setattr(runner.psutil, 'Process', lambda pid: Process())
    monkeypatch.setattr(runner.c, 'save', lambda path, value: saved.append(value))
    with pytest.raises(ValueError, match='exact poll evidence preserved'):
        runner.qualify_sampler(Envelope(), {})
    assert len(saved) == 1 and saved[0]['status'] == 'failed'
    comparison = saved[0]['independent_public_psutil'][0]
    assert comparison['sampled']['observed_threads'] == 1
    assert comparison['public']['observed_threads'] == 9
    assert comparison['fresh_created_after'] == 4.
