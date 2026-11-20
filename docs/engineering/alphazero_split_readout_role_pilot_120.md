# AlphaZero Split-Readout Start-Role Pilot 120

Date frozen: 2026-08-28

## Purpose

Experiment 117 showed that start role materially changes the deep-search
teacher. Experiment 118 showed that adding role information to one early global
entity barely learned that signal. Experiment 119 found that the same signal
reaches direct policy/value readouts about 13.3 and 10.0 times more strongly.

Experiment 120 asks one narrow question: can the simplest direct split-readout
bridge learn the already sealed role-sensitive teacher targets, confirm on the
untouched 117 panel, remain safe on ordinary reachable states, and only then
improve ordinary-budget gameplay?

## Frozen inputs

- completed 119 result SHA-256
  `6256cc23f24e7c6b601884bbdbbe0fb9f9c4126e64700818db03f692f6fd1828`;
- completed 118 result SHA-256
  `4846b79e1fbcd6344e759155befc469463a9100423f8785ff0e352fa9127fbc1`;
- completed 117 result SHA-256
  `a95081b4170604f5e7a162a532016c0eb2c4b5a67b36e31643fe5c50caf2a283`;
- exact generation-11 checkpoint SHA-256
  `c97e7e06910639c0edf61fdd878fd665e390298790830e0e792061d9107354d0`;
- role-blind candidates 114 and 116 SHA-256
  `93e548a35f2a42194c70602928fa1ae9398fe042dbad2b2c876694fa97a9c4bd`
  and
  `bb3bf65b102e5ec4c46a351c3958dba250c0f4c94f240847b7c3bc5871a57565`;
- 114 partition manifest SHA-256
  `61ef0dd0a4b0f83e2a9aba7cd3cc83d9e62f7770798287b7d2f672bacdb1c4dc`;
- 224-game training and 32-game holdout replay SHA-256
  `59f1dfeb36fafc4af8b6fcf8b90b78f91a751be86ef4328ba0067c9514b84c7a`
  and
  `33bb2d8f92d418f62063864f1fea7473ea5b323f1005e887aee01ed61c78be88`;
- sealed 118 training-target manifest SHA-256
  `278d22ab7497c0b4921c86a700c5fc88c3ba1d73982ff645ccbb169d499df79f`;
- retained installed generation-2 checkpoint SHA-256
  `60ca410b7d0d42c7beda66165df161cdc239a5e39a606d967e79adcf620534dc`.

Verify the exact 117--119 decisions, all hashes, gen11 lineage, target schedule
and action semantics, the disjoint 224/32 partition, and retained installation
before evidence.

## Architecture

Keep all 356,450 exact-gen11 parameters frozen and preserve every v1 encoder.
After the frozen entity encoder and mean pool produce state vector `h`, derive
signed public role `s=+1` when the observer is the start player and `s=-1`
otherwise. Add exactly two zero-initialized trainable 96-vectors:

- policy state `h_p = h + s*p`; `h_p` feeds both the frozen
  state-to-action context and frozen score head;
- value state `h_v = h + s*v`; `h_v` feeds only the frozen value head.

Train only `p` and `v`: exactly 192 parameters and 356,642 total. Policy loss
must have no gradient into `v`; value loss must have no gradient into `p`.
Epoch zero and checkpoint reload must reproduce exact gen11 legal logits and
values within `1e-7` on every orientation of the 448 fitting states and 64
protected states.

## Existing paired targets and development fitting

Do not generate a new fitting target. Load all four completed 118 target shards,
verify their exact manifest, protocol identity, 448-pair schedule, 1,024-search
telemetry, semantic action order, role alias invariants, and the sealed 118
summary. Reconstruct the exact 224 source trajectories and 896 oriented rows.

Reuse the exact 118 whole-game split: seed `1118100`, 179 fitting games / 358
pairs / 716 rows and 45 development games / 90 pairs / 180 rows. Because 119
inspected gradients on this split, it is a development gate rather than fresh
confirmation.

Use the unchanged objective and schedule: policy cross-entropy to the visit
distribution plus `0.5` times search-root-value MSE; six epochs; batch size 128;
AdamW learning rate `3e-4`; zero weight decay; gradient clipping 5; shuffle seed
`1118118`. No sweep occurs. Select the lowest development weighted objective,
with earlier epoch winning ties. Require a nonzero epoch, at least 0.5% relative
improvement from epoch zero, both role vectors nonzero, finite metrics, frozen
parent, and exact endpoint gradient separation.

## Protected paired confirmation

The 64 states from the disjoint 32-game holdout remain fitting- and
selection-forbidden. Only a development pass may reconstruct their exact 117
schedule and regenerate original/flipped exact-gen11 searches with 1,024
simulations, `c_puct=2.25`, real mode, zero root noise, maximum depth 96, and
coupled seed `1117000 + 100 * panel_index`. Four one-thread shards of 16 pairs
must seal identity and resume only by exact prefix. Their teacher summary must
reproduce 117 within `1e-9`.

Evaluate raw gen11, 114, 116, and candidate 120 against those oriented targets.
For each endpoint select the lowest-loss role-blind baseline without reference
to candidate 120. Candidate policy CE and value MSE must improve overall and in
both role classes; 50,000-sample 32-game bootstrap CI95 upper bounds must be
below zero for policy, value, and weighted objective, using seeds
`1120120..1120122`. Candidate raw paired policy TV must average at least 0.025
with CI95 lower bound above 0.01. Raw paired absolute value difference must
average at least 0.05 with CI95 lower bound above 0.01. Both role vectors must
remain nonzero and all metrics finite.

## Reachable-state safety

Only a confirmation pass opens the original 1,956-row reachable holdout safety
panel. Report the 117 phase-by-role breakdown. Candidate-minus-gen11 weighted
objective may be no greater than `+0.005` overall, for start role, and for
non-start role. This is a safety check, not a fresh efficacy claim.

## Conditional gameplay

Only a complete development, protected-confirmation, and safety pass opens
candidate 120 versus exact gen11. Both use 256 simulations, `c_puct=2.25`, real
mode, zero root noise, and operative seed `2 * game_seed + seat`. Use fresh
seeds `2200000..2200199`, both seats, for 400 games in four one-thread shards.

Require score share at least 55%, positive mean margin, both seats at least 50%,
all normal terminations, and a 50,000-sample paired-bootstrap CI95 lower bound
strictly above 50% with seed `1120200`.

## Decisions and exclusions

- Development uptake failure:
  `split_readout_role_failed_development_uptake_no_confirmation`.
- Protected confirmation failure:
  `split_readout_role_failed_protected_confirmation_no_gameplay`.
- Safety failure:
  `split_readout_role_confirmed_but_reachable_regression_no_gameplay`.
- Gameplay failure:
  `split_readout_role_offline_gain_not_converted_close_role_only_repair_no_install`.
- Gameplay pass:
  `split_readout_role_confirmed_review_short_continuation_no_install`.

No result authorizes automatic self-play, external `035`, installation,
promotion, or continuation. Exact generation 2 remains installed. Do not tune
the architecture, optimizer, split, thresholds, or search settings after seeing
evidence.

The runner is preflight-only by default and requires explicit `--launch` before
the first optimizer step. Preflight may read and validate existing 118 targets,
reconstruct states, build the zero bridge, check identity, endpoint separation,
and checkpoint round-trip. It may not optimize, access the protected
confirmation targets, generate a search, or run gameplay.
