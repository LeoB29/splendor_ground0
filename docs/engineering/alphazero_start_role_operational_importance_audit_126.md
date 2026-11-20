# AlphaZero Start-Role Operational Importance Audit 126

Date frozen: 2026-08-28

## Purpose

Experiment 117 proved that changing only `start_player` can materially change a
1,024-simulation teacher on deliberately selected late pre-round-end states.
Experiments 118--125 did not produce a network with enough raw role uptake.
Before more representation work, measure how often the missing variable matters
across ordinary reachable positions at actual project search budgets.

This is a no-training sensitivity audit. It does not estimate a general
first-player advantage and does not claim that each flipped-role copy is itself
on the recorded trajectory. Each original state must be exactly reconstructed
and reachable; the flipped copy is used only as a controlled rules-sensitivity
comparison.

## Frozen inputs

- completed experiment-117 result SHA-256:
  `a95081b4170604f5e7a162a532016c0eb2c4b5a67b36e31643fe5c50caf2a283`;
- completed experiment-125 result SHA-256:
  `7e11e49ed022bdb81a87a1f9be2e8c1c04a2c8e91be33297d56f7b0e7068c3f5`;
- exact generation-11 checkpoint SHA-256:
  `c97e7e06910639c0edf61fdd878fd665e390298790830e0e792061d9107354d0`;
- untouched 32-game / 1,956-row holdout SHA-256:
  `33bb2d8f92d418f62063864f1fea7473ea5b323f1005e887aee01ed61c78be88`;
- retained installed generation-2 checkpoint SHA-256:
  `60ca410b7d0d42c7beda66165df161cdc239a5e39a606d967e79adcf620534dc`.

Verify the hashes, completed decisions, normal outcomes, generation-11
lineage, and unchanged installation before evidence.

## Reachable-state reconstruction and sampling

Exactly reconstruct all 32 untouched games from their initial seeds and replay
actions. At every row require exact turn, player, canonical entity tensor,
unique legal selected action, final scores, winner, and normal completion.

For a game with `n` recorded decision states, select eight octile-midpoint rows
with zero-based indices `floor((2*k + 1) * n / 16)` for `k=0..7`. Require eight
unique states per game and exactly 256 original states in game-seed then octile
order. Record octile, normalized progress, turns remaining in the recorded
game, scores, pending-round-end, current player, and start role.

For each original, create a shallow counterfactual changing only
`start_player`. Require a changed canonical public-belief key, identical v1
entity tensor, and identical semantic legal action set. Raw generation-11
logits and value must agree within `1e-7`.

## Coupled search comparison

Run original and flipped states at both 64 and 256 simulations with exact
generation 11, `c_puct=2.25`, real mode, zero root noise, and maximum depth 96.
Use the same seed within a pair:

`1126000 + 10000 * budget_index + panel_index`,

where budgets are ordered `(64, 256)`. Use four one-thread workers with exact
resumable shard identities.

For every pair/budget report visit-distribution total variation, visit-argmax
change, absolute and signed search-root-value difference, and search telemetry.
Also report the fractions with TV at least 0.05, absolute value difference at
least 0.05, and either threshold or an argmax change.

## Summaries and uncertainty

At each budget report overall, each octile, original-role strata, first four
versus last four octiles, and final two octiles. Overall and combined-octile
summaries use 50,000-sample game-cluster bootstrap intervals for mean TV,
argmax-change rate, and mean absolute value difference. Bootstrap seed bases
are 1,126,100 for 64 simulations and 1,126,200 for 256 simulations.

The 256-simulation result is operative. The 64-simulation result is diagnostic
for search-budget dependence.

## Frozen interpretation

Integrity requires the exact 32 games, 256 reachable originals, eight unique
octiles per game, 512 budgeted comparisons, exact search counts, alias/action
checks, raw equality, finite metrics, and unchanged inputs.

For any stated stratum, policy materiality requires all of:

- mean visit TV at least 0.05;
- visit-argmax change rate at least 10%;
- game-cluster CI95 lower bound for mean TV above 0.01.

Value materiality requires both:

- mean absolute root-value difference at least 0.05;
- game-cluster CI95 lower bound above 0.01.

At 256 simulations:

1. If either policy or value is material overall, decide
   `start_role_operationally_material_prioritize_minimal_representation_repair`.
2. Otherwise, if either is material in the final two octiles, decide
   `start_role_material_only_near_game_end_deprioritize_general_role_repair`.
3. Otherwise decide
   `start_role_not_operationally_material_close_representation_branch`.

An integrity failure decides
`start_role_operational_audit_integrity_failed_no_conclusion`.

No threshold may be changed after results. The audit performs no training,
optimizer step, parameter mutation, target generation for learning, candidate
checkpoint write, self-play, gameplay arena, installation, promotion, external
`035` comparison, or automatic continuation.
