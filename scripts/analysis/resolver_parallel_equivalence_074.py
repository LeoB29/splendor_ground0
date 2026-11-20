"""Exact serial/parallel resolver comparison preregistered for experiment 074."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
import time
from typing import Any

import torch

from splendor_ai.bots import LegalActionAttentionScorerBot
from splendor_ai.engine.env import SplendorEnv
from splendor_ai.eval.run_paired_public_belief_resolver_gameplay import (
    FrozenPublicBeliefResolverBot,
    ResolverGameplayConfig,
)
from splendor_ai.training.replay import serialize_action


TRAJECTORY_SEED = 810_000
TARGET_TURNS = (0, 25, 50)
PARALLEL_WORKERS = 10
OUTPUT = Path("outputs/resolver_parallel_equivalence_074/result.json")


class _CapturingResolver(FrozenPublicBeliefResolverBot):
    def __init__(self, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)
        self.hook: dict[str, Any] | None = None

    def _on_resolved_decision(self, **kwargs) -> None:
        self.hook = {
            "kind": "resolved",
            "decision_index": kwargs["decision_index"],
            "parent": serialize_action(kwargs["parent"]),
            "selected": serialize_action(kwargs["selected"]),
            "resolution": kwargs["resolution"].to_payload(),
        }

    def _on_fallback_decision(self, **kwargs) -> None:
        self.hook = {
            "kind": "fallback",
            "decision_index": kwargs["decision_index"],
            "parent": serialize_action(kwargs["parent"]),
            "reason": kwargs["reason"],
        }


def _trajectory_states(cfg: ResolverGameplayConfig) -> dict[int, Any]:
    env = SplendorEnv(seed=TRAJECTORY_SEED)
    state = env.initial_state()
    bots = (
        LegalActionAttentionScorerBot(cfg.checkpoint, cfg.device),
        LegalActionAttentionScorerBot(cfg.checkpoint, cfg.device),
    )
    selected: dict[int, Any] = {}
    while not state.terminal and state.turn_index <= max(TARGET_TURNS):
        if state.turn_index in TARGET_TURNS:
            selected[state.turn_index] = state
        legal = env.legal_actions(state)
        action = bots[state.current_player].choose_action(env, state, legal)
        state = env.step(state, action)
    missing = sorted(set(TARGET_TURNS) - set(selected))
    if missing:
        raise RuntimeError(f"Frozen trajectory ended before target turns: {missing}")
    return selected


def _projection(bot: _CapturingResolver, action) -> dict[str, Any]:
    return {
        "selected_action": serialize_action(action),
        "telemetry": bot.telemetry,
        "hook": bot.hook,
    }


def _stable_bytes(payload: dict[str, Any]) -> bytes:
    return json.dumps(
        payload,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=True,
    ).encode("utf-8")


def main() -> None:
    torch.set_num_threads(1)
    cfg = ResolverGameplayConfig()
    states = _trajectory_states(cfg)
    comparisons = []
    for decision_index, turn in enumerate(TARGET_TURNS):
        state = states[turn]
        env = SplendorEnv()
        legal = env.legal_actions(state)
        common = {
            "config": cfg,
            "engine_seed": TRAJECTORY_SEED,
            "actor_seat": state.current_player,
        }
        serial = _CapturingResolver(**common, continuation_workers=1)
        parallel = _CapturingResolver(**common, continuation_workers=PARALLEL_WORKERS)
        try:
            serial_started = time.perf_counter()
            serial_action = serial.choose_action(env, state, legal)
            serial_seconds = time.perf_counter() - serial_started

            parallel_started = time.perf_counter()
            parallel_action = parallel.choose_action(env, state, legal)
            parallel_seconds = time.perf_counter() - parallel_started

            serial_projection = _projection(serial, serial_action)
            parallel_projection = _projection(parallel, parallel_action)
            serial_bytes = _stable_bytes(serial_projection)
            parallel_bytes = _stable_bytes(parallel_projection)
            comparison = {
                "turn": turn,
                "phase": "early" if turn <= 19 else "mid" if turn <= 49 else "late",
                "actor_seat": state.current_player,
                "serial_seconds": serial_seconds,
                "parallel_seconds": parallel_seconds,
                "speedup": serial_seconds / parallel_seconds,
                "serial_sha256": hashlib.sha256(serial_bytes).hexdigest(),
                "parallel_sha256": hashlib.sha256(parallel_bytes).hexdigest(),
                "byte_identical": serial_bytes == parallel_bytes,
                "serial": serial_projection,
                "parallel": parallel_projection,
            }
            comparisons.append(comparison)
            print(
                f"turn={turn} phase={comparison['phase']} "
                f"serial={serial_seconds:.2f}s parallel={parallel_seconds:.2f}s "
                f"speedup={comparison['speedup']:.2f}x "
                f"identical={comparison['byte_identical']}",
                flush=True,
            )
        finally:
            serial.close()
            parallel.close()

    result = {
        "experiment": "074_resolver_parallel_equivalence",
        "protocol": "docs/experiments/human_resolver_ceiling_calibration_074.md",
        "trajectory_seed": TRAJECTORY_SEED,
        "target_turns": list(TARGET_TURNS),
        "parallel_workers": PARALLEL_WORKERS,
        "comparisons": comparisons,
        "passed": len(comparisons) == len(TARGET_TURNS)
        and all(item["byte_identical"] for item in comparisons)
        and all(item["serial"]["hook"] is not None for item in comparisons),
    }
    OUTPUT.parent.mkdir(parents=True, exist_ok=True)
    temporary = OUTPUT.with_suffix(".json.tmp")
    temporary.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    temporary.replace(OUTPUT)
    digest = hashlib.sha256(OUTPUT.read_bytes()).hexdigest()
    print(f"result={OUTPUT} sha256={digest} passed={result['passed']}", flush=True)
    if not result["passed"]:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
