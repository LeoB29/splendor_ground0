# Resolver Distillation Disagreement and Logit Audit (`052a`)

Frozen after the completed `052` result and before computing any row-level
disagreement or logit-movement result. This is a diagnostic of the rejected
`052` student. It performs no training, gameplay, corpus generation, model
selection, or promotion.

## Question and fixed evidence

`052` produced a clean resolver-every-move corpus and reduced confirmation
resolved-target KL from `1.064529` to `0.528818`, but it failed the offline
gate because resolved target-top agreement improved by only `0.6404`
percentage points and fallback-anchor KL was `0.186057`. This audit asks why
the large distributional improvement rarely changed the selected action.

Require these exact frozen inputs:

- `052` result SHA-256
  `60458c1bea4328b64f33ad8238529d074a25d1034fa2845e0b339deee3827153`;
- parent `035` SHA-256
  `4422a82b593340eecd02af3a0941f24f0318cc7453e6a0739fd9f524539af2cd`;
- rejected selected-student SHA-256
  `32236a090513dd2e7a4bb4e249f11f8f5a3de8f4387a6feadb9b1684dea3773b`;
- train-target SHA-256
  `6827848ca5773799a698c62712ee04f3c40b2e2a33fd1ec3c9b5e3828d3af7e5`;
- validation-target SHA-256
  `865926c3e3221d8840bcbb1cdac43913bbd6efcd48b65637efca68e315a2d052`;
- confirmation-target SHA-256
  `8625d9049957ce266fa544896f0262cb37ccea6d168ed0c60a847ef5325cce8f`.

The train split is in-sample diagnostic evidence only. Validation and
confirmation remain separately reported. No result may be used to declare the
failed `052` gate passed.

## Fixed populations

Reconstruct every row through the acting-player public observation and legal
action encoders. Verify regenerated legal indices, recomputed parent argmax,
stored parent probabilities, and all referenced actions before admitting a
row.

Report the following populations independently:

1. **Resolved rows:** every `target_available=true` row.
2. **Raw resolver disagreements:** the recorded resolver action differs from
   raw `035`.
3. **Canonical resolver disagreements:** raw resolver disagreements where the
   two actions do not share a stored canonical successor. Alias-only switches
   are counted separately.
4. **Blended-target disagreements:** the top action of the exact stored
   `0.75 resolver + 0.25 p035` distribution differs from raw `035`. This is the
   population directly relevant to the failed `052` top-action gate.
5. **Fallback rows:** reported only to localize anchor drift; they are not
   combined with resolved disagreement metrics.

Tie-breaking is the existing legal-row order, matching `052`.

## Fixed measurements

For each disagreement pair, use raw model action scores, not probabilities, to
compute an additive-invariant pairwise gap:

`gap = score(preferred action) - score(parent action)`.

For raw/canonical resolver disagreements, the preferred action is the recorded
resolver action. For blended-target disagreements, it is the stored target
argmax. Report:

- parent and student gap mean, median, and fixed quantiles;
- `gap_delta = student_gap - parent_gap`;
- fraction with positive gap delta, fraction crossing from non-positive to
  positive, and fraction whose student argmax is the preferred action;
- fraction retaining the parent argmax and fraction moving to a third action;
- preferred-action rank under parent and student;
- stored target probability advantage over the parent action;
- where defined, adversarial resolved-value advantage of the resolver action
  over the parent action.

Report these for all three splits and separately by phase, acting seat,
winner/loser role, resolver action type, and fixed parent-gap bins
`(-inf,-1]`, `(-1,-0.5]`, `(-0.5,-0.25]`, `(-0.25,-0.1]`,
`(-0.1,-0.05]`, `(-0.05,0]`. Also report deterministic seed-ordered split
halves and game-cluster bootstrap 95% intervals for mean gap delta, directional
movement, and preferred-action capture. Use 20,000 bootstrap samples and seed
`520052`.

For fallback rows, report `KL(p035 || student)`, action movement, and
student-minus-parent entropy, split by phase and seat.

## Interpretation boundaries

The audit may distinguish:

- **target conflict:** many raw/canonical resolver choices are not actually the
  blended-target argmax or have non-positive target advantage;
- **insufficient optimization:** the target prefers the resolver action but
  train-split pairwise gaps rarely move in the correct direction;
- **generalization failure:** train movement/capture is materially stronger
  than both validation and confirmation;
- **argmax-margin failure:** gaps move correctly and KL improves, but movement
  is too small to cross the parent decision boundary;
- **representation/capacity pressure:** direction and crossing remain weak
  even in-sample despite aligned, nontrivial targets. This audit can suggest
  that explanation but cannot prove capacity is the unique cause.

This is descriptive evidence with no pass/fail or promotion gate. It cannot
authorize gameplay, retraining, target tuning, additional corpus generation,
or GPU rental. Any follow-up experiment requires a new prospective protocol.
