"""Measure resolver wall time per decision, as an interactive opponent sees it.

The 192 continuations of one resolver decision run **serially**: both
`ProcessPoolExecutor` uses in `run_paired_public_belief_resolver_gameplay.py`
parallelise over games, not within a decision. A single interactive move therefore
costs roughly a minute rather than the few seconds implied by dividing `051`'s
aggregate CPU time by the worker count.

This script intentionally retains the one-worker reference path. Experiment 074
added an opt-in within-decision process pool for interactive play and verified it
against this serial behavior. Use
`scripts/analysis/resolver_parallel_equivalence_074.py` for the byte-identical
serial/parallel comparison and measured interactive speedup.

Run from the repository root:

    python scripts/analysis/resolver_move_cost.py
"""

from __future__ import annotations

import time

from splendor_ai.bots import LegalActionAttentionScorerBot
from splendor_ai.engine.env import SplendorEnv
from splendor_ai.eval.run_paired_public_belief_resolver_gameplay import (
    FrozenPublicBeliefResolverBot,
    ResolverGameplayConfig,
)


SEED = 990_001
DECISIONS = 8
ASSUMED_RESOLVER_MOVES_PER_GAME = 31


def main() -> None:
    config = ResolverGameplayConfig()
    branches = (
        config.determinizations
        * config.root_actions
        * config.opponent_actions
        * config.discovery_continuations
    )
    print(
        "frozen contract:"
        f" determinizations={config.determinizations}"
        f" root_actions={config.root_actions}"
        f" opponent_actions={config.opponent_actions}"
        f" discovery_continuations={config.discovery_continuations}"
        f" => {branches} branches/search"
    )

    resolver = FrozenPublicBeliefResolverBot(config, engine_seed=SEED, actor_seat=0)
    opponent = LegalActionAttentionScorerBot(config.checkpoint, device=config.device)

    env = SplendorEnv(seed=SEED)
    state = env.initial_state()
    timings: list[float] = []
    turn = 0
    while not state.terminal and len(timings) < DECISIONS:
        actions = env.legal_actions(state)
        if not actions:
            break
        actor = resolver if state.current_player == 0 else opponent
        started = time.perf_counter()
        chosen = actor.choose_action(env, state, actions)
        elapsed = time.perf_counter() - started
        if actor is resolver:
            timings.append(elapsed)
            print(
                f"  resolver decision {len(timings)} at turn {turn}:"
                f" {elapsed:.2f}s -> {chosen.action_type.name}"
            )
        state = env.step(state, chosen)
        turn += 1

    mean = sum(timings) / len(timings)
    print(
        f"\nmean {mean:.2f}s  min {min(timings):.2f}s  max {max(timings):.2f}s"
        f"  (n={len(timings)}; a fallback decision costs ~0.02s)"
    )
    per_game = mean * ASSUMED_RESOLVER_MOVES_PER_GAME
    print(f"projected wait per game: {per_game/60:.1f} min")
    print(f"projected wait for 10 games: {per_game*10/60:.0f} min")


if __name__ == "__main__":
    main()
