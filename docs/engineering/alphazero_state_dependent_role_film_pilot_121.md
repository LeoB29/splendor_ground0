# AlphaZero State-Dependent Start-Role FiLM Pilot 121

Date frozen: 2026-08-28

## Purpose

The v1 input already contains turn index, pending-round-end status, and both
players' scores, but it omits start role. Experiment 117 proved that start role
changes deep-search decisions. Experiments 118 and 120 showed that constant role
offsets learn some of that signal but not enough. Experiment 119 independently
showed that a split feature-wise linear modulation (FiLM) bridge has exact
identity, stable role gradients, and no policy/value gradient leakage.

Experiment 121 tests whether allowing start role to interact with the complete
learned state vector is sufficient. This is the final bounded small role-adapter
pilot; it is not permission to tune 120 after failure.

## Frozen inputs

- completed 120 result SHA-256
  `3e1ef85c2b4a8231694a60e089e0b366ceba299c5a1925de116657d420ada3cd`;
- completed 119 result SHA-256
  `6256cc23f24e7c6b601884bbdbbe0fb9f9c4126e64700818db03f692f6fd1828`;
- completed 118 and 117 result SHA-256
  `4846b79e1fbcd6344e759155befc469463a9100423f8785ff0e352fa9127fbc1`
  and
  `a95081b4170604f5e7a162a532016c0eb2c4b5a67b36e31643fe5c50caf2a283`;
- exact generation-11 checkpoint SHA-256
  `c97e7e06910639c0edf61fdd878fd665e390298790830e0e792061d9107354d0`;
- role-blind candidates 114 and 116 SHA-256
  `93e548a35f2a42194c70602928fa1ae9398fe042dbad2b2c876694fa97a9c4bd`
  and
  `bb3bf65b102e5ec4c46a351c3958dba250c0f4c94f240847b7c3bc5871a57565`;
- exact 114 partition manifest, 224-game replay, and 32-game holdout SHA-256
  `61ef0dd0a4b0f83e2a9aba7cd3cc83d9e62f7770798287b7d2f672bacdb1c4dc`,
  `59f1dfeb36fafc4af8b6fcf8b90b78f91a751be86ef4328ba0067c9514b84c7a`,
  and
  `33bb2d8f92d418f62063864f1fea7473ea5b323f1005e887aee01ed61c78be88`;
- sealed 118 training-target manifest SHA-256
  `278d22ab7497c0b4921c86a700c5fc88c3ba1d73982ff645ccbb169d499df79f`;
- retained installed generation-2 checkpoint SHA-256
  `60ca410b7d0d42c7beda66165df161cdc239a5e39a606d967e79adcf620534dc`.

Verify every hash and decision, exact checkpoint lineage, target schedule and
semantic action order, the disjoint partition, and retained installation before
evidence.

## Architecture

Keep all 356,450 generation-11 parameters and every v1 encoder frozen. Let `h`
be the frozen 96-wide state vector after entity encoding and mean pooling, and
let signed public role `s` be `+1` for the start player and `-1` otherwise.

For policy and value separately, add zero-initialized 96-wide gain and bias
vectors:

- `h_p = h * (1 + s*g_p) + s*b_p`;
- `h_v = h * (1 + s*g_v) + s*b_v`.

`h_p` feeds the frozen action context and score head. `h_v` feeds only the
frozen value head. Train exactly 384 parameters; total parameters are 356,834.
Policy loss must have exactly zero gradient into value FiLM parameters and value
loss exactly zero gradient into policy FiLM parameters. All new parameters
start at zero, so epoch zero must reproduce exact gen11 logits and values within
`1e-7`, live and after checkpoint reload.

The gain terms make role effects depend on the complete learned state, which
already includes turn timing, pending-round status, scores, and other public
position information. No new hand-designed feature or encoder version is added.

## Existing targets and development fitting

Reuse and fully validate the sealed 448-pair / 896-row 118 target panel. Generate
no fitting target and no self-play. Reuse the exact whole-game split: 358 fitting
pairs / 716 rows and 90 development pairs / 180 rows from 179/45 games.
As in 120, this design-exposed split is only a development gate.

Use exactly the unchanged 118/120 training schedule: policy cross-entropy plus
`0.5` search-root-value MSE, six epochs, batch size 128, AdamW learning rate
`3e-4`, zero weight decay, gradient clipping 5, shuffle seed `1118118`, earliest
minimum-development-objective epoch. No sweep occurs.

Require a nonzero selected epoch, at least 0.5% relative development objective
improvement, nonzero gain and bias groups for both endpoints, finite metrics,
frozen parent, and exact endpoint separation. Failure stops before protected
confirmation.

## Protected confirmation and safety

Only a development pass may reconstruct the disjoint 64-state 117 schedule and
regenerate its original/flipped exact-gen11 searches: 1,024 simulations,
`c_puct=2.25`, real mode, zero root noise, maximum depth 96, coupled seed
`1117000 + 100 * panel_index`, four exact-prefix-resumable shards. Reproduce
117's teacher summary within `1e-9`.

Against the lowest-loss role-blind baseline among gen11/114/116, require
candidate policy CE and value MSE improvements overall and in both role classes;
50,000-sample 32-game bootstrap CI95 upper bounds below zero for policy, value,
and weighted objective with seeds `1121120..1121122`; raw paired policy TV at
least 0.025 with CI95 lower bound above 0.01; raw absolute value difference at
least 0.05 with CI95 lower bound above 0.01; all four FiLM groups nonzero; and
finite metrics.

Only confirmation pass opens the 1,956-row reachable-state safety panel.
Candidate-minus-gen11 weighted objective may be no greater than `+0.005`
overall, for start role, and for non-start role.

## Conditional gameplay

Only full development, confirmation, and safety passes open candidate 121
versus exact gen11 at 256 simulations, `c_puct=2.25`, real mode, zero root
noise, and operative seed `2 * game_seed + seat`. Use fresh seeds
`2210000..2210199`, both seats, 400 games in four one-thread shards.

Require score share at least 55%, positive mean margin, both seats at least 50%,
all normal terminations, and 50,000-sample paired-bootstrap CI95 lower bound
strictly above 50% with seed `1121200`.

## Decisions and exclusions

- Development failure:
  `state_dependent_role_film_failed_development_close_small_role_adapters`.
- Protected confirmation failure:
  `state_dependent_role_film_failed_confirmation_close_small_role_adapters`.
- Safety failure:
  `state_dependent_role_film_confirmed_but_reachable_regression_no_gameplay`.
- Gameplay failure:
  `state_dependent_role_film_offline_gain_not_converted_no_install`.
- Gameplay pass:
  `state_dependent_role_film_confirmed_review_short_continuation_no_install`.

Do not change epoch count, learning rate, threshold, architecture, split, or
search settings after evidence. Failure of development or confirmation closes
small role-only adapters and returns the project to a broader trainable
representation redesign. No result authorizes automatic self-play, external
`035`, installation, promotion, or continuation. Exact generation 2 remains
installed.

The runner is preflight-only without `--launch`. Preflight may validate existing
targets and identity/separation/round-trip contracts, but may not optimize,
access protected confirmation targets, search, or play. The user's 2026-08-28
instruction to carry out the stated next step authorizes this frozen 121 launch
after successful preflight.
