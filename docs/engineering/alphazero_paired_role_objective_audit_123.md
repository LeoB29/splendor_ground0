# AlphaZero Paired Start-Role Objective Audit 123

Status: frozen before 123 evidence

Date frozen: 2026-08-28

## Question

Experiment 122 gave a signed start-role input and the final entity-transformer
block enough capacity to improve ordinary development objective by 2.7653%, but
protected confirmation showed almost no role-dependent output. Determine,
without training, whether an explicit paired role-contrast objective has stable,
properly scaled access to both the role input and shared state block, and define
a development role-uptake gate before any further pilot.

## Frozen inputs

- finalized 122 candidate checkpoint SHA-256
  `843e2f0d69cea9f799491552259bb867357a4a25b3faa08b78f9ef13e88961d6`;
- exact generation-11 checkpoint SHA-256
  `c97e7e06910639c0edf61fdd878fd665e390298790830e0e792061d9107354d0`;
- exact 224-game fitting replay SHA-256
  `59f1dfeb36fafc4af8b6fcf8b90b78f91a751be86ef4328ba0067c9514b84c7a`;
- sealed 118 fitting-target manifest SHA-256
  `278d22ab7497c0b4921c86a700c5fc88c3ba1d73982ff645ccbb169d499df79f`.

Reconstruct only the 448 fitting pairs / 896 oriented rows and the exact 179/45
whole-game split. Do not accept a protected-result, holdout-replay, 117 target,
search, or arena path. The finalized 122 checkpoint may be evaluated only on
these fitting/development pairs to test the proposed gate.

## Losses at exact identity

Create the exact epoch-zero 122 model: copied generation 11, zero signed
96-wide role input, final entity-transformer block trainable, and every other
parent parameter frozen. Do not update it.

For each pair, let `+` denote the start-player orientation and `-` the
non-start orientation.

- Ordinary policy loss is mean per-orientation teacher cross-entropy.
- Ordinary value loss is mean per-orientation root-value MSE.
- Policy contrast loss is mean squared error between predicted and teacher
  `(p+ - p-)` action-probability vectors, averaged over legal semantic actions.
- Value contrast loss is mean squared error between predicted and teacher
  `(v+ - v-)` differences.

For ordinary policy, ordinary value, policy contrast, and value contrast,
compute loss and gradient without mutation on the 358-pair training split,
90-pair development split, and all 448 pairs. Report gradient L2 and RMS
separately for the role input, final entity block, and their concatenation;
finite/nonzero status; train-versus-development cosine for every applicable
group; and ordinary-versus-contrast cosine on training.

At exact identity, the paired contrast gradient must be nonzero in the role
input but structurally negligible in the final shared block: until the role
input moves, both orientations are identical at that block and their difference
gradient cancels apart from floating-point residue. Require final-block gradient
RMS no greater than `0.001` times role-input gradient RMS. Evaluate the same
contrast gradients at finalized candidate 122, whose role input is nonzero, and
there require finite nonzero role and final-block gradients plus positive
train-versus-development cosine. Candidate 122 is diagnostic only and cannot
alter coefficients or thresholds.

## One fixed normalization rule

Use training-split combined-gradient parity, separately by endpoint:

- `lambda_policy = ||grad ordinary_policy|| / ||grad policy_contrast||`;
- `lambda_value = ||grad ordinary_value|| / ||grad value_contrast||`.

The proposed future objective is
`ordinary_policy + 0.5*ordinary_value + lambda_policy*policy_contrast +
0.5*lambda_value*value_contrast`.

This is one deterministic calculation, not a coefficient sweep. Report the
normalized contrast contribution and summed endpoint gradient on train and
development. The identity model must remain exact and every parameter must be
bit-identical after the audit.

## Objective qualification

The paired objective qualifies only if:

- all input, identity, trainable-scope, frozen-parent, determinism, and
  no-mutation checks pass;
- both ordinary raw loss gradients are finite and nonzero in the role input and
  final block on train, development, and all pairs;
- at identity, both contrast gradients are finite and nonzero in the role input
  and combined group, structurally negligible in the final block under the
  `0.001` RMS-ratio rule, and have strictly
  positive role and combined train-versus-development cosine;
- at finalized candidate 122, both contrast gradients are finite and nonzero in
  the role input and final block on every split and have strictly positive
  train-versus-development cosine in all three groups;
- both coefficients are finite, positive, and at most `10000`;
- after normalization, each contrast role-gradient L2 is at least its ordinary
  counterpart on training;
- each summed normalized endpoint gradient has positive
  train-versus-development cosine and L2 no greater than 2.5 times its ordinary
  training gradient;
- development teacher mean policy TV and mean absolute value difference are
  each at least 0.15, confirming a robust fitting-only role signal.

## Prospectively fixed future development gate

If the objective qualifies, any separately frozen training pilot must retain
the existing at-least-0.5% aggregate development-objective improvement and also
require all of:

- at least 10% relative reduction from epoch zero in development policy
  contrast MSE;
- at least 10% relative reduction in development value contrast MSE;
- mean raw paired candidate policy TV at least 0.025;
- mean raw paired candidate absolute value difference at least 0.05;
- nonzero selected epoch, exact frozen scope, nonzero role input, and finite
  metrics.

As a diagnostic only, evaluate finalized candidate 122 on the fitting and
development pairs and verify that this new gate rejects it. Do not use that
candidate to alter coefficients or thresholds.

## Decisions and exclusions

- Qualification:
  `paired_role_contrast_objective_qualifies_freeze_pilot_no_training`.
- Failure:
  `paired_role_contrast_objective_not_stable_no_training`.

The runner must not expose a launch switch, instantiate an optimizer, mutate a
parameter, save a candidate checkpoint, read or regenerate protected 117 data,
generate a search target, run self-play or gameplay, access external `035`,
install, promote, or continue automatically. A qualification authorizes only a
separately reviewed and frozen pilot.
