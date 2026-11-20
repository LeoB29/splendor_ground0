# AlphaZero Identity-Preserving Start-Role Joint Pilot 118

Date frozen: 2026-08-27

## Purpose

Experiment 117 proved that changing only public start role materially changes
the exact-gen11 1,024-search teacher while v1 raw policy and value remain
identical. Experiment 116 already ruled out a simple shortage of role-blind
capacity. Experiment 118 asks the narrow causal question: can the smallest
explicit role bridge absorb paired teacher policy and value contrasts and
convert them into ordinary-budget gameplay?

This is one bounded role-only pilot. It generates no self-play and cannot alter
the installed checkpoint.

## Frozen inputs

- completed 117 result SHA-256:
  `a95081b4170604f5e7a162a532016c0eb2c4b5a67b36e31643fe5c50caf2a283`;
- exact generation-11 checkpoint SHA-256:
  `c97e7e06910639c0edf61fdd878fd665e390298790830e0e792061d9107354d0`;
- exact 114 role-blind candidate SHA-256:
  `93e548a35f2a42194c70602928fa1ae9398fe042dbad2b2c876694fa97a9c4bd`;
- exact 116 role-blind candidate SHA-256:
  `bb3bf65b102e5ec4c46a351c3958dba250c0f4c94f240847b7c3bc5871a57565`;
- 114 partition manifest SHA-256:
  `61ef0dd0a4b0f83e2a9aba7cd3cc83d9e62f7770798287b7d2f672bacdb1c4dc`;
- 224-game training replay SHA-256:
  `59f1dfeb36fafc4af8b6fcf8b90b78f91a751be86ef4328ba0067c9514b84c7a`;
- 32-game role-confirmation replay SHA-256:
  `33bb2d8f92d418f62063864f1fea7473ea5b323f1005e887aee01ed61c78be88`;
- retained installed generation-2 checkpoint SHA-256:
  `60ca410b7d0d42c7beda66165df161cdc239a5e39a606d967e79adcf620534dc`.

Verify all hashes, the exact 117 decision and sensitivity pass, gen11 lineage,
the failed 116 gameplay gate, the exact disjoint 224/32 partition, and retained
installation before evidence.

## Architecture bridge

Preserve every v1 encoder and checkpoint unchanged. Keep all 356,450 exact
gen11 parameters frozen. Add a two-entry, 96-wide
`observer_is_start_player` embedding to the canonical global entity after the
existing entity input projection and before the shared entity transformer.
Both rows start at exact zero. Train only these 192 parameters, with no other
adapter, new head, or residual tower.

Deployment derives the explicit input as
`state.current_player == state.start_player`. Replay derivation must validate
fixed-start turn parity before using `player_id == 0`. The shared transformed
state continues into both action scoring and value, so the intervention is
joint.

On every internal-validation orientation and all 128 orientations of the 117
panel, epoch zero must reproduce exact gen11 legal logits and value to maximum
absolute error `1e-7`. The parent must remain frozen, there must be exactly 192
trainable parameters, and checkpoint reload must reproduce predictions.

## Paired teacher-target generation

Reconstruct all 224 exact 114 training trajectories, matching every turn,
canonical tensor, selected semantic action, terminal outcome, and final score.
For each game select its latest two states before `pending_round_end`, in game
seed then turn order. Require 448 states, 224 games, two states per game, and
both original observer-role classes.

For every state make a shallow counterfactual changing only `start_player`.
Require a changed canonical public-belief key, identical v1 tensor, and
identical semantic legal-action set. Run original and flipped orientations with
exact gen11, 1,024 simulations, `c_puct=2.25`, real mode, zero root noise,
maximum depth 96, and coupled seed `1118000 + 100 * panel_index`. Use four
one-thread shards of 112 pairs, each sealing its identity before its first
search and resuming only by an exact completed-pair prefix.

Persist semantic action keys, visit distributions, search-root value, raw
priors/value, and search telemetry for both orientations. Require exactly 896
oriented targets and reproduce all role-alias invariants. This stage creates no
game and uses no terminal outcome as a training target.

## Whole-game split and role-only fitting

Split the 224 source games deterministically with seed `1118100`: 45 validation
games and 179 training games. Both orientations of a source state remain in the
same split. Require 358 training pairs / 716 rows and 90 validation pairs / 180
rows.

Use policy cross-entropy to the 1,024-visit distribution plus `0.5` times MSE to
the search-root value. Train for exactly six epochs, batch size 128, AdamW
learning rate `3e-4`, zero weight decay, gradient clipping 5, and shuffle seed
`1118118`. These are the existing 037 role-embedding schedule and the current
joint loss weights; no sweep occurs. Include exact epoch zero and select the
lowest validation weighted objective, with earlier epoch winning ties. Require
a nonzero selected epoch and at least 0.5% relative validation-objective
improvement over epoch zero.

## Held-out paired role confirmation

The 64 exact 117 states from the disjoint 32-game holdout are forbidden for
target fitting and epoch selection. Reconstruct their schedule exactly and
rerun the original/flipped exact-gen11 searches with the original 117 seeds
`1117000 + 100 * panel_index`. The reproduced mean visit TV, top-action-change
rate, and mean absolute root-value difference must match the sealed 117 result
within `1e-9`.

Evaluate exact gen11, frozen 114, frozen 116, and the selected 118 candidate raw
predictions against both oriented teacher targets. For each endpoint choose the
lowest-loss frozen role-blind baseline among gen11/114/116, using the complete
panel and without reference to candidate 118.

Offline efficacy passes only if all conditions hold:

- candidate policy cross-entropy and value MSE each improve over their selected
  role-blind baseline overall and separately in both role orientations;
- the 50,000-sample 32-game cluster-bootstrap CI95 upper bound for the overall
  candidate-minus-baseline delta is below zero for policy CE, value MSE, and
  their weighted objective, using seeds `1118120..1118122`;
- candidate paired raw-policy TV averages at least 0.025 and its cluster CI95
  lower bound is above 0.01;
- candidate paired absolute raw-value difference averages at least 0.05 and its
  cluster CI95 lower bound is above 0.01;
- all metrics are finite and both embedding rows differ from exact zero.

The separation thresholds retain only about 10.7% and 16.0% of 117's observed
teacher policy/value contrast. They demand material uptake without requiring a
192-parameter bridge to reproduce full search.

## Reachable-state safety panel

Evaluate the original 1,956 reachable holdout rows once with candidate 118 and
exact gen11. Report the full 117 phase-by-role breakdown. Candidate-minus-gen11
weighted objective must be no greater than `+0.005` overall, for start role, and
for nonstart role; all metrics must be finite. This repeatedly inspected panel
is a safety/nonregression check, not a fresh efficacy claim.

Any identity, target-generation, training, paired-confirmation, or safety
failure stops before gameplay.

## Conditional ordinary-budget arena

Only a complete offline pass opens exact candidate 118 versus exact gen11. Both
use 256 simulations, `c_puct=2.25`, real mode, zero root noise, and operative
seed `2 * game_seed + seat`. Use seeds `2180000..2180199`, both candidate seats,
for 400 games in four concurrent one-thread shards of 50 paired seeds.

Every shard seals protocol/checkpoint/search/schedule identity and resumes only
by exact completed-game prefix. Report W-D-L, share, margin, both seats,
terminations, and a 50,000-sample paired bootstrap with seed `1118200`.
Gameplay passes only at score share at least 55%, positive mean margin, both
seats at least 50%, all normal terminations, and CI95 lower bound strictly above
50%.

## Decisions and exclusions

- Role uptake failure:
  `minimal_role_bridge_failed_direct_uptake_review_richer_interaction_no_gameplay`.
- Role uptake passes but safety fails:
  `role_bridge_uptake_with_reachable_regression_no_gameplay`.
- Offline passes but gameplay fails:
  `role_omission_real_but_not_sufficient_close_simple_role_repair_no_install`.
- Full gameplay pass:
  `minimal_start_role_joint_bridge_confirmed_review_short_continuation_no_install`.

Do not tune embedding width, learning rate, pair selection, thresholds, or
search settings after a failure. No branch authorizes automatic self-play,
`035`, installation, promotion, or continuation. Exact generation 2 remains
installed.

The runner defaults to preflight-only and must require an explicit launch flag
before the first 1,024-search target is generated. Preflight may reconstruct
states, validate hashes/schedules, construct the zero bridge, and run identity
tests; it may not generate teacher targets or optimize parameters.
