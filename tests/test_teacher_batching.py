import copy
import random

import pytest
import torch

from splendor_ai.alphazero import teacher_batching as b


@pytest.fixture(scope="module", autouse=True)
def initialize():
    b.old.initialize_worker()


def states():
    env = b.old.SplendorEnv(seed=2000950000)
    first = env.initial_state()
    state = first
    for _ in range(2):
        reserve = next(a for a in env.legal_actions(state) if a.action_type.name == "RESERVE_DECK" and a.tier == 2)
        state = env.step(state, reserve)
    rich = copy.deepcopy(state)
    for player in rich.players:
        player.score = 14
        player.bonuses = {c: 5 for c in player.bonuses}
    return first, state, rich


def test_padded_scores_empty_rows_and_batch_one_equivalence():
    env = b.old.SplendorEnv()
    policy = b.Batched035(b.old._RAW)
    cases = [(s, env.legal_actions(s)) for s in states()]
    cases += [(states()[0], [])]
    scores = policy.scores_many(cases)
    assert scores[-1] == () and policy.choose_many(cases)[-1] is None
    for (state, legal), row in zip(cases[:-1], scores[:-1]):
        # Arithmetic closeness is diagnostic; chosen move equality is required.
        reference = b.old._RAW.score_actions(state, legal)
        torch.testing.assert_close(torch.tensor(row), torch.tensor(reference), rtol=1e-4, atol=1e-5)
        assert legal[max(range(len(row)), key=lambda i: row[i])] == b.old._RAW.choose_action(env, state, legal)
        assert policy.scores_many([(state, legal)])[0] == reference
    assert policy.metrics["padded_rows"] > 0


def test_hidden_world_changes_do_not_change_batched_legal_inputs():
    state = states()[1]
    env = b.old.SplendorEnv()
    worlds = [b.old.determinize_for_observer(state, state.current_player, random.Random(i)) for i in (3, 19)]
    assert b.old.serialize_state_snapshot(worlds[0]) != b.old.serialize_state_snapshot(worlds[1])
    legal = [env.legal_actions(s) for s in worlds]
    assert legal[0] == legal[1]
    scores = b.Batched035(b.old._RAW).scores_many(list(zip(worlds, legal)))
    assert scores[0] == scores[1]


def test_first_maximum_tie_and_supplied_action_order():
    bot = copy.deepcopy(b.old._RAW)
    with torch.no_grad():
        for parameter in bot._model.parameters():
            parameter.zero_()
    state = states()[0]
    legal = list(reversed(b.old.SplendorEnv().legal_actions(state)))
    assert b.Batched035(bot).choose_many([(state, legal), (state, legal[:2])]) == [legal[0], legal[0]]


@pytest.mark.parametrize("size", (1, 2, 8))
def test_full_trajectories_match_original_with_forced_roots_and_hidden_reserves(size):
    env = b.old.SplendorEnv()
    rich = states()[2]
    cutoff = copy.deepcopy(rich)
    cutoff.turn_index = 400
    work = [(rich, None), (rich, env.legal_actions(rich)[0]), (cutoff, None)]
    before = [b.old.serialize_state_snapshot(s) for s, _ in work]
    expected = [b.old.trajectory(copy.deepcopy(s), a, include_trace=True) for s, a in work]
    observed = b.trajectories(work, b.Batched035(b.old._RAW), size, include_trace=True)
    assert observed == expected
    assert before == [b.old.serialize_state_snapshot(s) for s, _ in work]
    assert observed[-1]["reason"] == "max_turns"
    terminal = b.old.deserialize_state_snapshot(expected[0]["final_state"])
    assert b.trajectories([(terminal, None)], b.Batched035(b.old._RAW), size, True) == [
        b.old.trajectory(terminal, include_trace=True)]


def test_scheduler_repetition_and_no_progress_match_original(monkeypatch):
    state = states()[0]
    env = b.old.SplendorEnv()
    class Serial:
        def choose_many(self, rows):
            return [b.old._RAW.choose_action(env, s, legal) for s, legal in rows]
    monkeypatch.setitem(b.old.CONFIG, "no_progress_limit", 1)
    expected = b.old.trajectory(copy.deepcopy(state), include_trace=True)
    assert b.trajectories([(state, None), (state, None)], Serial(), 2, True) == [expected, expected]
    monkeypatch.setitem(b.old.CONFIG, "no_progress_limit", 60)
    monkeypatch.setattr(b.old, "state_signature", lambda state: "same")
    expected = b.old.trajectory(copy.deepcopy(state), include_trace=True)
    assert expected["reason"] == "repetition_cutoff"
    assert b.trajectories([(state, None)], Serial(), 1, True) == [expected]


def test_full_length_scheduler_with_serial_oracle_keeps_each_job_independent():
    env = b.old.SplendorEnv()
    class Serial:
        def choose_many(self, rows):
            return [b.old._RAW.choose_action(env, s, legal) for s, legal in rows]
    work = [(s, None) for s in states()[:2]]
    expected = [b.old.trajectory(copy.deepcopy(s), a, include_trace=True) for s, a in work]
    observed = b.trajectories(work, Serial(), 2, include_trace=True)
    assert observed == expected and all(r["turns"] > 10 for r in observed)


def test_invalid_batch_inputs_fail_without_fallback():
    policy = b.Batched035(b.old._RAW)
    assert b.trajectories([], policy) == []
    for size in (0, -1, True, 1.5):
        with pytest.raises(ValueError):
            b.trajectories([], policy, size)
    with pytest.raises(ValueError):
        b.Batched035(object())
    bot = copy.deepcopy(b.old._RAW)
    bot._model.train()
    with pytest.raises(ValueError):
        b.Batched035(bot)
