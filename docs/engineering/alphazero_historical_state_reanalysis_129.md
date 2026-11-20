# AlphaZero Historical-State Reanalysis Pilot 129

Date frozen: 2026-08-29

## Purpose

Experiments 105--110 found replay forgetting and no external improvement from
simply retaining or enlarging stale self-play data. Experiments 113--128 then
showed that deeper search is stronger, while more data, more capacity,
continued deep-search distillation, start-role representation, and a fixed
search-value blend do not reliably convert that strength at the ordinary
budget.

Experiment 129 tests the strongest remaining AlphaZero hypothesis: the learner
is trained on too narrow a succession of states. Historical games will supply
only reachable starting states. Exact generation 11 will then play both sides
from those states with fresh 1,024-simulation search, producing new visit and
terminal targets. No historical action or stale historical target enters the
new replay after the starting state.

This is population-anchored state reanalysis, not asymmetric league play. The
search model and actual continuation policy remain the same on both sides.

## Frozen inputs

- experiment-128 result SHA-256:
  `21e506ea5792543330da90a7cd2880c0e85b1c7d46a721ade1ac57028a5c9170`;
- exact generation-11 checkpoint SHA-256:
  `c97e7e06910639c0edf61fdd878fd665e390298790830e0e792061d9107354d0`;
- experiment-114 terminal-control checkpoint SHA-256:
  `93e548a35f2a42194c70602928fa1ae9398fe042dbad2b2c876694fa97a9c4bd`;
- retained installed generation-2 checkpoint SHA-256:
  `60ca410b7d0d42c7beda66165df161cdc239a5e39a606d967e79adcf620534dc`;
- generation-2 actor replay SHA-256:
  `855b9ae3cb4e05da7b26b759ce8fa4e62c8c44aea4a0dcee2da0e01a1790de56`;
- generation-5 actor replay SHA-256:
  `0f9393fb020cf01c4d06f829c03e7426913dcc29d6afde49091cb9def57772e1`;
- generation-8 actor replay SHA-256:
  `96f0d346743f30bd29e329f4bf3e2a7a361e778391c6d6e5833ca2a06af9be10`;
- generation-11 actor / ordinary-control training replay SHA-256:
  `59f1dfeb36fafc4af8b6fcf8b90b78f91a751be86ef4328ba0067c9514b84c7a`;
- experiment-114 ordinary holdout SHA-256:
  `33bb2d8f92d418f62063864f1fea7473ea5b323f1005e887aee01ed61c78be88`.

Verify every hash, completed 128 failure, exact checkpoint lineage, unchanged
installation, source generations, normally completed games, ordered contiguous
turns, legal selected actions, and exact state tensors before evidence.

## Historical starting-state schedule

Use four source populations: generation-2, generation-5, generation-8, and
generation-11 actor replay. The first three sources have 256 clean games; the
sealed generation-11 experiment-114 training partition has 224. Every source
therefore contains the complete ordinal range required below.

For training collection, scan source-game ordinals 0--79 in original replay
order and seal the first 64 normally completed fresh continuations. The desired
accepted slots cycle `early`, `middle`, `late`; their target progress fractions
are 0.25, 0.50, and 0.75. For each source game choose the nearest replay state
that is not `pending_round_end`, ties toward the earlier turn. If a fresh
continuation does not end through ordinary completion, exclude it without
looking at winner or score and use the next source game for the same accepted
slot. At most 16 replacements per source are available.

This yields exactly 256 clean historical continuations: 64 per source and,
within each source, 22 early, 21 middle, and 21 late starts. Every source state
is reconstructed from its original seed and complete public action prefix, and
its canonical entity tensor must exactly reproduce the stored row.

Historical-start gameplay uses disjoint fixed source-game ordinals 96--111:
16 per source, again cycling early/middle/late. No source game used or attempted
for training may enter this 64-state panel.

## Fresh continuation search

From every selected training state, exact generation 11 controls both players.
Use 1,024 simulations, `c_puct=2.25`, real information-legal mode, root-noise
fraction 0.25, search seed namespace `1129000`, and the established temperature,
repetition, no-progress, forced-pass, and 200-turn contracts. Each continuation
is independently atomic, resumable, and hashed.

Write generation-11 AlphaZero rows with fresh visit distributions and terminal
WDL. Preserve source generation, source seed, source turn, source phase, and
synthetic continuation identity as provenance fields. Historical policy
targets, Q values, selected actions, and outcomes are forbidden after the
starting state.

## Matched-size mixed replay and training

The experiment-114 terminal-control candidate is the fixed ordinary-data A/B
control. It trained from 13,960 ordinary generation-11 rows.

After the 256 historical continuations are sealed, add whole ordinary games
from the beginning of the sealed 114 training replay until historical plus
ordinary rows first reach 13,960; choose the closer of including or excluding
the final whole game, ties excluding it. This row-count rule is outcome-blind.
The mixed candidate's total row count must differ from 13,960 by no more than
one ordinary game's row count.

Start from exact generation 11 and train one same-architecture generation-12
candidate on the historical and selected ordinary paths. Use the exact
experiment-114 parent-safe recipe: up to 12 epochs, batch 256, AdamW learning
rate and weight decay `1e-4`, policy weight 1.0, value weight 0.5, deterministic
whole-game validation fraction 0.2, patience 3, minimum improvement 0.001,
bucket batches, and epoch-zero parent eligibility. Use seed `1129000`. Do not
change the policy target, terminal-WDL value target, architecture, or search.

## Ordinary holdout gate

Evaluate exact generation 11, the 114 terminal-control checkpoint, and the new
candidate on every row of the sealed 32-game ordinary holdout. Gameplay opens
only if a nonzero epoch was selected, all metrics are finite, candidate policy
CE and combined objective strictly improve generation 11, value MSE regresses
by at most 0.025 versus generation 11, and relative to the 114 control candidate
policy CE is at most 0.01 worse, combined objective at most 0.02 worse, and
value MSE at most 0.025 worse.

## Gameplay gates

Only an offline pass opens gameplay. Both agents use 256 simulations,
`c_puct=2.25`, real mode, zero root noise, and maximum 200 turns.

First run the 64-state disjoint historical-start panel. For every state, play
the new candidate and 114 control with the candidate assigned once to each
seat, for 128 games. Bootstrap the 64 state clusters with 50,000 samples. Pass
requires every game to complete normally, candidate score share at least 55%,
positive margin, at least 50% from each candidate seat, and CI95 lower bound
strictly above 50%.

Only that pass opens the ordinary initial-state A/B: new candidate versus 114
control on seeds `2720000..2720199`, both seats, for 400 games in four exact
workers. Apply the same strength, seat, termination, and confidence gates.

## Decisions

- Reconstruction or collection integrity failure:
  `historical_reanalysis_integrity_failed_no_strength_conclusion`.
- Offline training or ordinary holdout failure:
  `historical_reanalysis_offline_failed_no_gameplay_no_install`.
- Historical-start gameplay failure:
  `historical_reanalysis_failed_diverse_state_gameplay_no_install`.
- Historical-start pass but ordinary A/B failure:
  `historical_reanalysis_diverse_gain_not_ordinary_robust_no_install`.
- Both gameplay panels pass:
  `historical_reanalysis_transfers_authorize_population_loop_design_no_install`.

A pass authorizes only a separately frozen multi-generation population-state
reanalysis loop. A failure substantially closes the present compact
policy/value AlphaZero training design and triggers review of distributional or
belief-aware value modelling. No `035` evaluation, installation, promotion, or
automatic continuation is authorized.

## Launch guard

Default invocation verifies inputs, streams all source identities, reconstructs
the exact training-attempt and disjoint gameplay schedules, validates one
temporary continuation target construction, and removes it. It instantiates no
optimizer, writes no durable replay, trains no checkpoint, and plays no
strength game. The long run requires explicit `--launch`. The owner's
2026-08-29 request to do experiment 129 authorizes launch only after focused
tests and preflight pass.
