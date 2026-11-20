# AlphaZero Start-Role Representation Audit 117

Date frozen: 2026-08-27

## Purpose

Experiment 116 increased total capacity from 356,450 to 991,828 parameters and
improved every untouched offline objective, but reached only 51.625% gameplay
with a 58.75%/44.5% seat split. The canonical entity tensor records normalized
turn index and pending round end, but not whether the observer is the starting
player. Existing contract tests prove that states differing only in that public
final-round role collide in the v1 tensor.

Experiment 117 tests whether the strong 1,024-simulation gen11 teacher is
materially sensitive to this missing variable before any role-aware architecture
is authorized. It performs no training or gameplay arena.

## Frozen inputs

- completed 114 result SHA-256:
  `dd8bbaa7876b881db69d44fb4242624e1f227981b4628da392b698491c231ad5`;
- exact 114 same-architecture candidate SHA-256:
  `93e548a35f2a42194c70602928fa1ae9398fe042dbad2b2c876694fa97a9c4bd`;
- completed 116 result SHA-256:
  `5c50ca7989381ad3c55b32f42e97a9ab67b4992085c53e1578227d932bde5718`;
- exact 116 capacity candidate SHA-256:
  `bb3bf65b102e5ec4c46a351c3958dba250c0f4c94f240847b7c3bc5871a57565`;
- exact generation-11 checkpoint SHA-256:
  `c97e7e06910639c0edf61fdd878fd665e390298790830e0e792061d9107354d0`;
- untouched 114 holdout SHA-256:
  `33bb2d8f92d418f62063864f1fea7473ea5b323f1005e887aee01ed61c78be88`;
- retained installed generation-2 checkpoint SHA-256:
  `60ca410b7d0d42c7beda66165df161cdc239a5e39a606d967e79adcf620534dc`.

Verify exact completed decisions, checkpoint lineage, holdout size and normal
outcomes, retained installation, and every hash before evidence.

## Existing-holdout stratification

Evaluate exact gen11, the 114 candidate, and the 116 candidate on all 1,956
untouched rows. Report policy loss, value loss, weighted objective, target-top
agreement, target-top probability, policy entropy, value MAE, and row count for:

- all rows;
- observer is starting player / is not starting player;
- turns 0--19, 20--39, and 40+;
- every start-role by phase cell.

The engine starts every source game at seat 0. Validate that `player_id` and
turn parity agree before deriving role. This stage is descriptive and does not
authorize architecture work by itself.

## Exact state reconstruction and frozen schedule

Reconstruct all 32 untouched trajectories from `SplendorEnv(game_seed)` by
matching each recorded selected action payload uniquely against the legal action
set and applying it. At every turn require exact player, turn, canonical entity
tensor, legal selected action, terminal outcome, and final scores.

For each game select its latest two pre-terminal states for which
`pending_round_end` is false. This freezes 64 pairs in game-seed then turn order.
Require exactly two states per game and both observer start-role classes across
the panel.

For each selected state create a counterfactual shallow copy changing only
`start_player` to the other seat. Require the original and counterfactual to
have different canonical public-belief keys but identical v1 entity tensors and
identical legal semantic action payloads. Successor-state keys themselves are
expected to differ because they correctly include start role.

## Coupled teacher comparison

Run exact gen11 on original and flipped-role states with 1,024 simulations,
`c_puct=2.25`, real mode, zero root noise, maximum depth 96, and the same
operative seed within each pair: `1117000 + 100 * panel_index`. Use four
concurrent one-thread shards of 16 scheduled pairs. Each shard seals its exact
identity and resumes only by an exact completed-pair prefix.

Require raw network logits/priors and values to agree within `1e-7`, confirming
the representational alias. Compare search visit distributions over serialized
semantic actions and report per pair:

- visit total variation;
- visit-argmax action change;
- absolute and signed search-root-value difference;
- original observer start role and turn;
- search depth, inference calls, and elapsed time.

Report overall and per-role summaries plus 50,000-sample game-cluster bootstrap
intervals for mean visit TV, top-action change rate, and mean absolute root-value
difference, using seed `1117100`.

## Frozen interpretation gate

Integrity requires all 64 exact scheduled pairs, 32 games, both role classes,
all alias/action-set/raw-prediction checks, finite metrics, and 1,024 simulations
on both sides.

Policy role sensitivity passes when top-action change rate is at least 10%, mean
visit TV is at least 0.05, and the game-cluster CI95 lower bound for mean TV is
above 0.01. Value role sensitivity passes when mean absolute search-root-value
difference is at least 0.05 and its game-cluster CI95 lower bound is above 0.01.

If integrity passes and either policy or value sensitivity passes, review an
identity-preserving start-role-aware joint policy/value pilot. Otherwise, do not
rebuild for start role; proceed to training-objective and multi-generation-loop
design review.

No result automatically trains, generates self-play, runs `035`, installs,
promotes, or continues any model.
