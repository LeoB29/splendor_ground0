"""Adapter checks without checkpoints, saved outcomes or simulation workloads."""
from __future__ import annotations

from types import SimpleNamespace
from concurrent.futures import ThreadPoolExecutor

import pytest

from tools.large_corpus_optimization_128 import engine
from tools.simulation_speed_probe_128_v2.encoder import Fast035
from . import backend


@pytest.fixture
def reference(monkeypatch):
    model = SimpleNamespace(training=False,
                            config=SimpleNamespace(observation_size=295, action_feature_size=79))
    raw = SimpleNamespace(_device="cpu", _model=model, score_actions=lambda *args: "original scores")
    monkeypatch.setattr(engine.old, "_RAW", raw)
    yield raw
    assert backend._ACTIVE is None


def call_block(indices=(0, 2), seeds=(11, 12, 13)):
    return engine.run_block("state", [object() for _ in indices], list(seeds),
                            {"candidate_indices": list(indices), "worlds": []}, "isolated-base")


@pytest.mark.parametrize("name", ["reference", "fast035"])
def test_scope_preserves_root_scoring_auditor_and_counts(monkeypatch, reference, name):
    observed, guards = [], []
    def block(state, actions, seeds, saved, base):
        observed.append(engine.old._RAW)
        assert (state, base) == ("state", "isolated-base")
        assert saved["candidate_indices"] == [0, 2]
        return "completed"
    monkeypatch.setattr(engine, "run_block", block)
    with backend.use_backend(engine, name, boundary=lambda: guards.append(engine.old._RAW)) as usage:
        assert engine.old._RAW is reference
        assert engine.old._RAW.score_actions(None, None) == "original scores"
        assert call_block() == "completed"
        assert engine.old._RAW is reference
        assert engine.old._RAW._model is reference._model
        assert usage == {"backend": name, "blocks_attempted": 1, "blocks_completed": 1,
                         "candidate_worlds_attempted": 6, "candidate_worlds_completed": 6,
                         "blocks": [{"candidate_indices": [0, 2], "worlds": 3,
                                     "candidate_worlds": 6, "completed": True}]}
    assert engine.run_block is block and engine.old._RAW is reference
    assert guards == [reference, reference]
    if name == "fast035":
        assert isinstance(observed[0], Fast035) and observed[0].model is reference._model
    else:
        assert observed == [reference]


@pytest.mark.parametrize("name", ["reference", "fast035"])
@pytest.mark.parametrize("failure", [RuntimeError("fixture failure"), KeyboardInterrupt()])
def test_block_failure_restores_and_propagates(monkeypatch, reference, name, failure):
    def block(*args):
        raise failure
    monkeypatch.setattr(engine, "run_block", block)
    with pytest.raises(type(failure)) as caught:
        with backend.use_backend(engine, name) as usage:
            call_block()
    assert caught.value is failure
    assert engine.run_block is block and engine.old._RAW is reference
    assert usage["blocks_attempted"] == 1 and usage["blocks_completed"] == 0
    assert usage["candidate_worlds_attempted"] == 6 and usage["candidate_worlds_completed"] == 0


@pytest.mark.parametrize("at", [1, 2])
def test_entry_or_exit_guard_failure_restores(monkeypatch, reference, at):
    calls = []
    def guard():
        calls.append("guard")
        if len(calls) == at:
            raise RuntimeError("resource hold")
    def block(*args):
        return "done"
    monkeypatch.setattr(engine, "run_block", block)
    with pytest.raises(RuntimeError, match="resource hold"):
        with backend.use_backend(engine, "fast035", boundary=guard) as usage:
            call_block()
    assert engine.run_block is block and engine.old._RAW is reference
    assert usage["blocks_completed"] == (0 if at == 1 else 1)


@pytest.mark.parametrize("mutation", ["device", "training", "observation", "action"])
def test_unsupported_model_fails_before_install(reference, mutation):
    original_block = engine.run_block
    if mutation == "device":
        reference._device = "cuda"
    elif mutation == "training":
        reference._model.training = True
    elif mutation == "observation":
        reference._model.config.observation_size = 294
    else:
        reference._model.config.action_feature_size = 80
    with pytest.raises(ValueError):
        with backend.use_backend(engine, "fast035"):
            pytest.fail("Unsupported model entered context")
    assert engine.run_block is original_block and engine.old._RAW is reference


def test_nested_context_rejected_outer_still_works(monkeypatch, reference):
    def block(*args):
        return "done"
    monkeypatch.setattr(engine, "run_block", block)
    with backend.use_backend(engine, "fast035"):
        outer = engine.run_block
        with pytest.raises(RuntimeError, match="already active"):
            with backend.use_backend(engine):
                pytest.fail("Nested context entered")
        assert engine.run_block is outer and engine.old._RAW is reference
        assert call_block() == "done"
    assert engine.run_block is block


def test_cross_thread_block_rejected_without_mutation(monkeypatch, reference):
    def block(*args):
        pytest.fail("Cross-thread collector called")
    monkeypatch.setattr(engine, "run_block", block)
    with backend.use_backend(engine, "fast035"):
        with ThreadPoolExecutor(max_workers=1) as pool:
            with pytest.raises(RuntimeError, match="collection thread"):
                pool.submit(call_block).result()
        assert engine.old._RAW is reference


def test_changed_policy_rejected_before_work(monkeypatch, reference):
    with backend.use_backend(engine, "fast035") as usage:
        with monkeypatch.context() as patch:
            other = object()
            patch.setattr(engine.old, "_RAW", other)
            with pytest.raises(RuntimeError, match="Reference policy changed"):
                call_block()
            assert engine.old._RAW is other
        assert usage["blocks_attempted"] == 0 and engine.old._RAW is reference


def test_input_callback_reaches_only_fast_policy(monkeypatch, reference):
    seen = []
    callback = lambda *args: seen.append(args)
    def block(*args):
        engine.old._RAW.on_position("state", "actions", "tensors")
    monkeypatch.setattr(engine, "run_block", block)
    with backend.use_backend(engine, "fast035", on_position=callback):
        call_block()
    assert seen == [("state", "actions", "tensors")]
    assert engine.old._RAW is reference


@pytest.mark.parametrize("kwargs", [{"name": "unknown"}, {"boundary": 1}, {"on_position": 1},
                                    {"on_position": lambda *args: None}])
def test_invalid_arguments_do_not_install(reference, kwargs):
    original = engine.run_block
    with pytest.raises((ValueError, TypeError)):
        with backend.use_backend(engine, **kwargs):
            pytest.fail("Invalid context entered")
    assert engine.run_block is original and engine.old._RAW is reference


def test_wrong_collector_rejected(reference):
    with pytest.raises(ValueError, match="historical LCO"):
        with backend.use_backend(SimpleNamespace(), "fast035"):
            pytest.fail("Wrong collector entered")


def test_uninitialized_policy_rejected(monkeypatch):
    original = engine.run_block
    monkeypatch.setattr(engine.old, "_RAW", None)
    with pytest.raises(ValueError, match="Initialize"):
        with backend.use_backend(engine, "fast035"):
            pytest.fail("Uninitialized context entered")
    assert engine.run_block is original and backend._ACTIVE is None
