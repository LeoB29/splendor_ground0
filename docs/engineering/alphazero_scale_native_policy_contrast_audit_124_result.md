# AlphaZero Scale-Native Policy Contrast Audit 124 Result

Date completed: 2026-08-28

## Outcome

Experiment 124 completed normally without training. Both fixed policy-contrast
arms qualified. The prospective simplicity rule selected
`summed_probability_difference_mse`, so a separately frozen joint-objective
training pilot is justified.

The audit used only the sealed 448-pair / 896-row fitting panel and its exact
358/90 pair split. No protected or holdout path was accepted. No optimizer was
instantiated, no parameter changed, and no checkpoint was written.

## Selected policy loss

The selected loss is mean over pairs of the sum across shared legal actions of
the squared error between candidate and teacher start-minus-nonstart
probability differences. Removing the per-action divisor reduced the required
training-gradient-parity coefficient from experiment 123's 19,558.0513 to:

`lambda_policy = 702.476696552303`.

This is below the frozen 10,000 cap. The normalized policy contrast role
gradient was 447.251 times the ordinary policy role gradient. Adding it to the
ordinary policy gradient produced total training L2 only 1.4158 times ordinary,
with train/development cosine 0.801980.

The exact-identity role-gradient train/development cosine was 0.957346. At
finalized candidate 122, the summed loss reached both the role input and shared
state block with train/development cosines 0.889451 and 0.855035. Every summed
arm integrity and qualification check passed.

## Fallback arm

The teacher-energy-normalized fallback also passed with
`lambda_policy = 14.4438957087457`. Its training teacher-energy floor was the
median 0.0129433. It is not selected because the simpler summed loss already
qualified, as required by the frozen rule.

Retain experiment 123's value-difference MSE and
`lambda_value = 39.4241527478651`. Retain its future development gates: at least
0.5% aggregate improvement, at least 10% policy and value contrast-error
reductions, raw policy TV at least 0.025, and raw absolute value difference at
least 0.05.

## Implementation correction

The first invocation incorrectly compared the full-precision saved value
coefficient `39.42415274786513` bit-for-bit with the protocol's rounded
`39.4241527478651`, marking global integrity false even though both arms passed
their own checks. The comparison was corrected to accept at most `1e-12`
serialization roundoff and covered by a regression test. The unchanged frozen
protocol, loss formulas, coefficients, thresholds, and input data were rerun
from scratch. The corrected result above is the operative result.

## Next step

Freeze and preflight one guarded training pilot using the exact experiment-122
architecture from generation-11 identity and the joint objective:

`ordinary_policy + 0.5*ordinary_value +
702.476696552303*summed_policy_contrast +
0.5*39.4241527478651*value_contrast`.

Training batches must preserve whole original/flipped pairs. Keep the 122
optimizer schedule and the new role-uptake development gates. Qualification of
124 does not itself authorize training, protected confirmation, gameplay,
installation, promotion, or continuation.

Decision:
`summed_policy_contrast_qualifies_freeze_joint_objective_pilot_no_training`.

Focused experiments 118--124 tests pass 26/26, the 102--124 AlphaZero band
passes 99/99, and the full repository suite passes 822/822 in 173.88 seconds.

## Artifacts

- Frozen protocol SHA-256:
  `52e1db9043b9c9b9a0e0f2a023f736e8dd2c7362c8c9ed8ec8dc009972152a6b`.
- Raw result SHA-256:
  `29437a929887c9e34a51c52de218df82fb3c2b0f06298c92e240c001364b18d9`.
