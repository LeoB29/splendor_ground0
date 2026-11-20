# AlphaZero Search-Value Distillation Pilot 128

Date frozen: 2026-08-29

## Purpose

Experiment 113 proved that exact generation 11 searched at 1,024 simulations
is materially stronger than the same network searched at 256 simulations.
Experiments 114, 116, and 127 then improved held-out teacher-target fit without
converting that improvement into reproducible ordinary-budget gameplay.

The current AlphaZero policy target already distils the 1,024-simulation visit
distribution, but its value target remains the terminal WDL from one stochastic
hidden-information game. Experiment 128 tests the smallest credible transfer
repair: retain the exact policy target and architecture, while replacing the
training value target with an equal, fixed blend of terminal WDL and the
already serialized 1,024-simulation root value.

This is a single guarded A/B pilot. It generates no self-play, performs no
`035` evaluation, and does not authorize installation, promotion, or automatic
continuation.

## Frozen inputs

- experiment-114 result SHA-256:
  `dd8bbaa7876b881db69d44fb4242624e1f227981b4628da392b698491c231ad5`;
- experiment-127 result SHA-256:
  `ba7b76de3b201a2222d43e59c997d8b4ae8d743ff3bc59f1a0fd9f995fcda44d`;
- exact generation-11 checkpoint SHA-256:
  `c97e7e06910639c0edf61fdd878fd665e390298790830e0e792061d9107354d0`;
- existing experiment-114 terminal-WDL control checkpoint SHA-256:
  `93e548a35f2a42194c70602928fa1ae9398fe042dbad2b2c876694fa97a9c4bd`;
- retained installed generation-2 checkpoint SHA-256:
  `60ca410b7d0d42c7beda66165df161cdc239a5e39a606d967e79adcf620534dc`;
- sealed 224-game training replay SHA-256:
  `59f1dfeb36fafc4af8b6fcf8b90b78f91a751be86ef4328ba0067c9514b84c7a`;
- sealed 32-game holdout replay SHA-256:
  `33bb2d8f92d418f62063864f1fea7473ea5b323f1005e887aee01ed61c78be88`.

Verify all hashes, exact 114 train/holdout membership, generation lineage,
1,024-simulation replay identity, completed 127 failure, and unchanged
installation before evidence.

## Stage A: training-replay search-value audit

Use only the sealed 224-game training replay. Do not open the 32-game holdout
for this gate. For every row, require finite terminal, raw-network, root-Q, and
root-search values in `[-1, 1]`, exact 1,024 root visits, and reproduction of
`search_root_value` as the visit-weighted mean of serialized root Q values to
within `1e-9`.

Define the fixed target

`blended_value = 0.5 * terminal_WDL + 0.5 * search_root_value`.

The audit qualifies only if all structural checks pass, mean absolute
root-search correction relative to the raw network is at least 0.02, at least
10% of rows have an absolute correction of at least 0.05, and root-search MSE
against terminal WDL is no more than 0.05 worse than raw-network MSE. These are
signal and gross-calibration guards, not coefficient selection. The 0.5 blend
is frozen regardless of the audit values. Audit failure stops before training.

## Stage B: one parent-safe blended-value successor

Start from exact generation 11 and train one same-architecture generation-12
candidate on the sealed 224-game replay. Preserve every experiment-114 policy
target and the complete experiment-114 optimizer/selection recipe: up to 12
epochs, batch 256, AdamW learning rate and weight decay `1e-4`, policy weight
1.0, value weight 0.5, deterministic whole-game internal validation fraction
0.2, patience 3, minimum improvement 0.001, bucket batches, and epoch-zero
parent eligibility. Use training seed `1128000`. The only learning change is
the fixed blended value target above.

After selection, evaluate exact generation 11, the existing experiment-114
terminal-WDL control, and the new candidate once on every row of the sealed
32-game holdout. Report policy cross-entropy and top agreement plus terminal,
root-search, and blended value MSE/MAE.

Gameplay opens only if all metrics are finite, a nonzero epoch was selected,
the new candidate strictly improves generation 11 on policy CE, root-search
MSE, and blended MSE, and it strictly improves the existing 114 control on
blended MSE. Its policy CE may be at most 0.005 worse than the 114 control, and
its terminal-WDL MSE may be at most 0.025 worse than both generation 11 and the
114 control. Failure seals without gameplay.

## Stage C: fresh ordinary-budget causal gameplay

Both panels use 256 simulations, `c_puct=2.25`, real mode, zero root noise,
maximum 200 turns, both seats, four exact workers, and a 50,000-sample paired
bootstrap.

The primary causal A/B compares the new blended-value candidate with the
existing experiment-114 terminal-WDL control on seeds
`2600000..2600199`, for 200 paired seeds / 400 games. Only a primary pass opens
the secondary strength panel.

The secondary compares the new candidate with exact generation 11 on seeds
`2610000..2610199`, again for 200 paired seeds / 400 games.

Each panel passes only if every game completes normally, score share is at
least 55%, mean score margin is positive, each seat is at least 50%, and the
paired-bootstrap CI95 lower bound is strictly above 50%.

## Decisions

- Audit failure:
  `search_value_signal_audit_failed_no_training_no_install`.
- Training or holdout failure:
  `search_value_blend_offline_failed_no_gameplay_no_install`.
- Primary A/B failure:
  `search_value_blend_failed_direct_control_close_fixed_blend_no_install`.
- Primary pass but secondary failure:
  `search_value_blend_beats_terminal_control_not_parent_robust_no_install`.
- Both panels pass:
  `search_value_blend_transfers_deep_search_authorize_population_pilot_no_install`.
- Any identity or integrity failure:
  `search_value_distillation_integrity_failed_no_strength_conclusion`.

A pass authorizes only design of a separately frozen population-anchored
multi-generation AlphaZero pilot. A failure closes this exact fixed blended
target, not every distributional, uncertainty-aware, or belief-aware value
formulation. No result uses `035` as a teacher or benchmark, changes the
installed generation-2 checkpoint, or continues automatically.

## Launch guard

Default invocation performs frozen-input, replay-streaming, lineage,
configuration, output-path, and one-batch target-construction preflight only.
It must not run the Stage-A evidence audit, instantiate an optimizer, train a
checkpoint, or play a game. The long run requires explicit `--launch`. The
owner's 2026-08-29 request to do experiment 128 authorizes that launch only
after focused tests and preflight pass.
