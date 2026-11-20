# AlphaZero Paired Start-Role Objective Audit 123 Result

Date completed: 2026-08-28

## Outcome

Experiment 123 completed normally without training. The proposed paired
probability-MSE/value-MSE objective did not qualify because its policy contrast
term is too weakly scaled. This was the only failed qualification check.

The audit used only the sealed 448-pair / 896-row experiment-118 fitting panel
and its exact 358/90 pair split. It accepted no 117 result, protected target,
holdout replay, search, or arena path. It instantiated no optimizer, changed no
parameter, and wrote no checkpoint.

## Gradient audit

The exact epoch-zero experiment-122 model reproduced generation-11 logits and
values exactly on all 896 rows. Its 74,880-parameter trainable scope and frozen
boundary were exact and deterministic.

At identity, paired policy and value contrast losses produced stable nonzero
role-input gradients. The shared final-block gradients were only floating-point
residue, as expected before the role input moves. At finalized candidate 122,
whose role input is nonzero, both contrast losses produced finite nonzero
gradients in the role input and final block. Candidate train/development
cosines were 0.750489/0.741170 for policy role/block and
0.999903/0.997939 for value role/block.

The fitting-only development teacher signal is strong: mean policy TV 0.268446
and mean absolute value difference 0.316020, both above the frozen 0.15
requirement.

## Normalization result

Training-gradient parity produced:

- `lambda_policy = 19558.0513`;
- `lambda_value = 39.4242`.

The value coefficient is finite and comfortably bounded. The policy coefficient
exceeds the prospectively fixed maximum of 10,000, so the combined objective
does not qualify. This large number comes from averaging probability-difference
MSE over every legal action, which dilutes an already small first-order policy
signal.

Apart from that coefficient, the normalization behaved well. Normalized policy
and value contrast role gradients were 447.25 and 157.70 times their ordinary
role gradients. Summed endpoint gradient norms were only 1.416 and 1.419 times
ordinary, with train/development cosines 0.740931 and 0.988830.

## Proposed gate diagnostic

The prospectively defined future development gate correctly rejects finalized
candidate 122 even though its aggregate objective improved 2.7653%:

- policy contrast error improved only 0.0316%, below 10%;
- value contrast error improved only 0.9989%, below 10%;
- raw policy TV was 0.000390, below 0.025;
- raw absolute value difference was 0.003383, below 0.05.

This confirms that aggregate fitting alone cannot be the development gate for a
role-aware pilot.

## Interpretation and next step

Do not train this objective and do not raise the coefficient cap after seeing
the result. The value contrast formulation is usable. The next bounded step is
a no-training audit of a scale-native paired policy contrast, such as summed
rather than per-action-averaged probability-difference error and a
teacher-normalized alternative. Freeze a simplicity rule before comparing them,
retain the value coefficient and the new development gate, and use only the
sealed fitting panel. Protected 117 must remain untouched.

Decision: `paired_role_contrast_objective_not_stable_no_training`.

No training, protected evaluation, search, self-play, gameplay, external `035`,
installation, promotion, or automatic continuation occurred.

Focused experiments 118--123 tests pass 22/22, the 102--123 AlphaZero band
passes 95/95, and the full repository suite passes 818/818 in 233.66 seconds.

## Artifacts

- Frozen protocol SHA-256:
  `94f26d42a6b13009cd43277e7a817881e0607c390764f1a61d3eeec248f3c43f`.
- Raw result SHA-256:
  `12a7332ec08de864f7a55fb5d57d189924078a6513149cf4e020f410f39bce36`.
