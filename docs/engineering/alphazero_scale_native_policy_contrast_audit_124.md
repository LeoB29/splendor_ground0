# AlphaZero Scale-Native Policy Contrast Audit 124

Status: frozen before 124 evidence

Date frozen: 2026-08-28

## Question

Experiment 123 found stable paired role gradients and a usable value-contrast
coefficient, but its policy probability-difference MSE was diluted by averaging
over legal actions and required `lambda_policy=19558.0513`, above the frozen
10,000 cap. Determine, without training, whether one of two prospectively fixed
scale-native policy losses is stable and bounded enough for a future pilot.

## Frozen inputs

- completed 123 result SHA-256
  `12a7332ec08de864f7a55fb5d57d189924078a6513149cf4e020f410f39bce36`;
- finalized 122 candidate checkpoint SHA-256
  `843e2f0d69cea9f799491552259bb867357a4a25b3faa08b78f9ef13e88961d6`;
- exact generation-11 checkpoint SHA-256
  `c97e7e06910639c0edf61fdd878fd665e390298790830e0e792061d9107354d0`;
- exact 224-game fitting replay SHA-256
  `59f1dfeb36fafc4af8b6fcf8b90b78f91a751be86ef4328ba0067c9514b84c7a`;
- sealed 118 fitting-target manifest SHA-256
  `278d22ab7497c0b4921c86a700c5fc88c3ba1d73982ff645ccbb169d499df79f`.

Reconstruct only the 448 fitting pairs / 896 oriented rows and exact 358/90
whole-game split. Verify that 123 completed without training and failed only its
policy coefficient bound. Accept no 117 target, protected result, holdout replay,
search, or arena path.

## Fixed policy-loss arms

For each pair, let `d_pred = p_start - p_nonstart` and
`d_teacher = t_start - t_nonstart` over the shared ordered legal actions.

1. `summed_probability_difference_mse` is the simple primary arm:
   mean over pairs of `sum_actions (d_pred - d_teacher)^2`. It removes only the
   action-count divisor that diluted experiment 123.
2. `teacher_energy_normalized_mse` is the fallback arm:
   mean over pairs of
   `sum_actions (d_pred - d_teacher)^2 /
   max(sum_actions d_teacher^2, E_floor)`, where `E_floor` is the standard median
   training-pair teacher energy, fixed once before gradients. The median floor
   prevents tiny teacher differences from receiving unbounded weight.

The exact epoch-zero 122 model and finalized candidate 122 are diagnostic only.
No parameter may move.

## No-training measurements

For both arms and the ordinary policy CE reference, compute loss and gradient on
the 358-pair training split, 90-pair development split, and all 448 pairs.
Report gradient L2/RMS separately for the role input, final entity block, and
combined trainable parameters.

At exact identity, require finite nonzero role and combined gradients, with
final-block RMS no greater than `0.001` times role-input RMS. At finalized
candidate 122, require finite nonzero role and final-block gradients. Report
train/development cosine for every applicable group and ordinary-versus-contrast
training cosine.

For each arm independently, set
`lambda_policy = ||grad ordinary_policy_train|| /
||grad policy_contrast_train||`. Combine its normalized gradient with ordinary
policy and report role contribution, total norm ratio, and
train/development cosine. This is a deterministic calculation, not a coefficient
sweep.

Reproduce 123's fitting-only teacher signal and its future development gate on
candidate 122. Retain `lambda_value=39.4241527478651` and the exact value
contrast formulation; do not recalibrate either.

## Qualification and simplicity rule

An arm qualifies only if:

- all input, identity, trainable-scope, determinism, finite-metric, and
  no-mutation checks pass;
- its exact-identity role and combined gradients are finite/nonzero on every
  split and its final-block RMS satisfies the `0.001` structural-residue rule;
- its exact-identity role and combined train/development cosines are positive;
- at candidate 122, role and final-block gradients are finite/nonzero on every
  split and all three train/development cosines are positive;
- `lambda_policy` is finite, positive, and at most 10,000;
- normalized contrast role-gradient L2 is at least ordinary role-gradient L2;
- the summed normalized policy gradient has positive train/development cosine
  and training L2 no greater than 2.5 times ordinary policy;
- the training teacher-energy floor is finite and positive, development teacher
  policy TV remains at least 0.15, and the retained future gate rejects 122.

Apply the simplicity rule prospectively: select
`summed_probability_difference_mse` if it qualifies, even if the normalized arm
also qualifies. Consider `teacher_energy_normalized_mse` only if the summed arm
fails. If neither qualifies, close probability-difference MSE policy contrast
and review a different bounded policy objective.

## Decisions and exclusions

- Summed arm qualifies:
  `summed_policy_contrast_qualifies_freeze_joint_objective_pilot_no_training`.
- Normalized fallback qualifies:
  `teacher_normalized_policy_contrast_qualifies_freeze_joint_objective_pilot_no_training`.
- Neither qualifies:
  `scale_native_policy_contrast_not_qualified_no_training`.

The runner must have no launch switch or optimizer path. It may not mutate or
save a model, train, read/regenerate protected 117 data, generate targets, run
self-play/gameplay, access external `035`, install, promote, or continue
automatically. Qualification authorizes only a separately reviewed and frozen
training pilot.
