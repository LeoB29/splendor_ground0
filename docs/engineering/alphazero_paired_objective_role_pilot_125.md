# AlphaZero Paired-Objective Start-Role Pilot 125

Date frozen: 2026-08-28

## Purpose

Experiment 122 showed that a limited jointly trainable state representation can
fit common deep-search targets but still ignore start role. Experiments 123--124
found that explicit paired contrast gradients are stable and selected a
scale-native summed policy-difference loss. Experiment 125 is one guarded
training pilot of that frozen architecture and objective.

## Frozen inputs

- completed 124 result SHA-256
  `29437a929887c9e34a51c52de218df82fb3c2b0f06298c92e240c001364b18d9`;
- exact generation-11 checkpoint SHA-256
  `c97e7e06910639c0edf61fdd878fd665e390298790830e0e792061d9107354d0`;
- sealed 118 fitting-target manifest SHA-256
  `278d22ab7497c0b4921c86a700c5fc88c3ba1d73982ff645ccbb169d499df79f`;
- exact 224-game fitting replay SHA-256
  `59f1dfeb36fafc4af8b6fcf8b90b78f91a751be86ef4328ba0067c9514b84c7a`;
- exact 32-game holdout replay SHA-256
  `33bb2d8f92d418f62063864f1fea7473ea5b323f1005e887aee01ed61c78be88`;
- role-blind candidates 114 and 116 SHA-256
  `93e548a35f2a42194c70602928fa1ae9398fe042dbad2b2c876694fa97a9c4bd`
  and
  `bb3bf65b102e5ec4c46a351c3958dba250c0f4c94f240847b7c3bc5871a57565`;
- retained installed generation-2 SHA-256
  `60ca410b7d0d42c7beda66165df161cdc239a5e39a606d967e79adcf620534dc`.

Only after a development pass may the runner open the already generated exact
122 confirmation-target manifest SHA-256
`e9289abc0d8068558534596aca30e3c480a413aab822bff96b50aca44e8aa0fe`.
Those targets must reproduce the original 117 confirmation schedule and exact
teacher summary; generate no replacement confirmation search.

Verify the finalized 117--124 lineage, every hash, target/action order, exact
whole-game split, and installed checkpoint before optimization. Do not read
confirmation target contents or reconstruct the protected state panel during
preflight or development fitting.

## Architecture and objective

Use exact experiment-122 architecture from generation-11 identity: add one
zero-initialized signed 96-wide role vector to the canonical global entity and
jointly train only that vector plus the complete final entity-transformer block.
Exactly 74,880 parameters are trainable; all other 281,666 parent parameters
remain frozen. Epoch zero must reproduce exact generation-11 outputs within
`1e-7`, live and after checkpoint reload.

Every optimizer batch must contain complete original/flipped pairs. For each
pair compute:

- ordinary per-orientation policy cross-entropy `P`;
- ordinary per-orientation root-value MSE `V`;
- summed policy contrast `PC = sum_actions ((p+ - p-) - (t+ - t-))^2`,
  averaged over pairs;
- value contrast `VC = ((v+ - v-) - (z+ - z-))^2`, averaged over pairs.

The frozen joint objective is:

`J = P + 0.5*V + 702.476696552303*PC +
0.5*39.4241527478651*VC`.

No coefficient recalculation, sweep, annealing, or alternative arm occurs.

## Training and development selection

Reuse exactly 358 fitting pairs and 90 development pairs from the existing
179/45 whole-game split. Use 12 epochs, 64 pairs per batch (128 oriented rows),
AdamW learning rate `1e-4`, weight decay `1e-4`, gradient clipping 5, and shuffle
seed `1118118`. Select the earliest epoch with minimum development `J`; epoch
zero is the safe fallback.

The selected epoch opens confirmation only if all checks pass:

- selected epoch is nonzero;
- ordinary development objective `P + 0.5*V` improves at least 0.5% from epoch
  zero;
- canonical per-action-averaged policy contrast MSE improves at least 10%;
- value contrast MSE improves at least 10%;
- mean raw paired candidate policy TV is at least 0.025;
- mean raw paired candidate absolute value difference is at least 0.05;
- selected summed policy contrast improves, role vector and final block move,
  every frozen parameter remains bit-exact, trainable scope is exact, and all
  metrics are finite.

Failure stops without reading confirmation target contents, safety, or gameplay.
Do not add epochs, alter coefficients, lower a gate, or select another epoch
after evidence.

## Conditional confirmation and safety

Only a development pass may reconstruct the 64 protected states and load the
four exact 122 confirmation-target shards. Verify their manifest, protocol,
generation-11 teacher identity, schedule, semantic actions, and 117 teacher
summary within `1e-9`.

Apply the unchanged protected confirmation gates against the lowest-loss
role-blind endpoint baselines among generation 11, candidate 114, and candidate
116: policy and value improve overall and in both role classes; 50,000-sample
32-game bootstrap CI95 upper bounds below zero for policy, value, and ordinary
weighted objective using seeds `1125120..1125122`; raw paired policy TV at least
0.025 with CI95 lower bound above 0.01; raw absolute value difference at least
0.05 with CI95 lower bound above 0.01; finite metrics and nonzero role input.

Only confirmation pass opens the unchanged 1,956-row reachable-state safety
panel. Candidate-minus-generation-11 ordinary weighted objective may be no
greater than `+0.005` overall, for start role, and for non-start role.

## Conditional gameplay

Only development, confirmation, and safety passes open candidate 125 versus
exact generation 11 at 256 simulations, `c_puct=2.25`, real mode, zero root
noise, and operative seed `2 * game_seed + seat`. Use fresh seeds
`2250000..2250199`, both seats, 400 games in four one-thread shards.

Require score share at least 55%, positive mean margin, both seats at least 50%,
all normal terminations, and 50,000-sample paired-bootstrap CI95 lower bound
strictly above 50% using seed `1125200`.

## Decisions and exclusions

- Development failure:
  `paired_objective_role_pilot_failed_development_no_confirmation`.
- Protected confirmation failure:
  `paired_objective_role_pilot_failed_protected_confirmation`.
- Safety failure:
  `paired_objective_role_pilot_confirmed_but_reachable_regression_no_gameplay`.
- Gameplay failure:
  `paired_objective_role_pilot_offline_gain_not_converted_no_install`.
- Gameplay pass:
  `paired_objective_role_pilot_confirmed_review_role_aware_continuation_no_install`.

No result authorizes new self-play, external `035`, installation, promotion, or
automatic continuation. Exact generation 2 remains installed.

The runner is preflight-only without `--launch`. Preflight may validate fitting
targets, paired batches, identity, gradients, and checkpoint round-trip, but may
not optimize or access confirmation target contents. The user's 2026-08-28
instruction to perform this next step authorizes launch after successful
preflight.
