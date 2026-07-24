# Adaptive Resolver Teacher Audit (`057`)

Frozen after `056` and before implementation or any new resolver evidence.
This is a no-training, CPU-only fixed-state diagnostic. It performs no
confirmation access, game collection, gameplay gate, promotion, or model
fitting.

## Question

`056` showed that independent resolver averaging converges strongly overall,
but uniform 8x computation oversolves mid/late positions while early labels
remain unstable. `057` asks whether a deterministic confidence-based
best-arm allocation can return a stable point action or value-equivalence set
at less than the uniform 8x budget, especially in early positions.

Require the frozen `056` result SHA-256
`bc20cde5436d904a4e63a381a8173b000fcc5b66462d71194134bdab77dac081`,
with full integrity and decision
`budget_convergence_positive_but_8x_insufficient_redesign_teacher`.
Also require:

- `055` result SHA-256
  `4aa3b54e77a4856c2c5cd1bb18b1a4088755bda7679bd52a7ce58627ba622714`;
- `055` selection SHA-256
  `4c8c9f9c499b3d8cae6a7a001c3c9b8a1f438eb466fcf0be35acb3e34072102a`;
- frozen `035` SHA-256
  `4422a82b593340eecd02af3a0941f24f0318cc7453e6a0739fd9f524539af2cd`.

The `052` confirmation split and game outcomes are forbidden.

## Fixed states and independent estimates

`056` used the first two persisted `055` selections in each of the 18
seat x phase x original-gap cells. Select the remaining two rows per cell,
without inspecting resolver values, completion, metrics, or outcomes. Require
36 unique states, two per cell, spanning at least 15 games, and no overlap
with `056`.

Reuse the completed frozen `055` estimates A and B. Generate C through P with
the exact frozen resolver contract:

- three information-legal determinizations;
- four root actions and four opponent responses;
- four stochastic continuations per leaf, exactly 192 continuations/search
  and 48 continuations/root action;
- epsilon `0.1`, temperature `0.5`, top-k `3`;
- loop-safe score-margin rollout, scale `20`, turn cap `400`, and at most 64
  model-loop fallbacks.

To be disjoint from `056`, new labels C-P use namespace indices 17-30:
belief-root start `10,000,000 * index + 1,000,000` and continuation-seed
start `1,000,000,000,000,000 * index`. Attempt all 504 new searches with ten
CPU workers and persistent per-search progress. Repeat the first C search
from zero and require an identical deterministic projection.

## Frozen adaptive rule

Treat A-H and I-P as independent eight-sample streams. Within each stream,
each resolver action value is one 48-continuation root unit. The offline
replay may inspect a root value only when that unit is allocated; uninspected
values cannot affect allocation or output.

For every action, consume its first two samples initially: eight root units,
384 continuations. At every later step:

1. Compute each action's sample mean and sample standard deviation.
2. Form a clipped interval `mean +/- t * standard_error`, using the fixed
   one-sided-95% Student critical values for degrees of freedom 1-7:
   `3.078, 1.886, 1.638, 1.533, 1.476, 1.440, 1.415`.
3. Let `best_lower` be the largest lower bound. The active equivalence set is
   every action whose upper bound is at least `best_lower - 0.025`.
4. Stop only when every active action has at least three samples, the active
   set has been unchanged for four allocation steps, and its largest interval
   radius is at most `0.075`.
5. Otherwise allocate the next unused sample to the active action with the
   largest radius, breaking ties by fewer samples and then lower root index.
   Recompute all intervals after every unit; a previously inactive action may
   re-enter. If no active action has an unused sample, stop at the uniform 8x
   ceiling of 32 root units/1,536 continuations.

Return the highest-mean action (lower index breaks ties), the final active
equivalence set, per-action means/intervals/counts, and a diagnostic
distribution obtained by softmaxing means at temperature `0.05` within the
active set. The equivalence set, not this diagnostic distribution, is the
primary non-point target.

The compute-matched uniform baseline averages all eight samples/action in
each stream. It always consumes 32 units/1,536 continuations.

## Metrics

For adaptive and uniform pairs report:

- exact best-action agreement;
- flattened Pearson and MAE, and mean within-state Spearman;
- both directional cross-regrets, mean symmetric regret, p90, and fractions
  at most `0.025` and `0.05`;
- mutual value-equivalent agreement within `0.025`;
- game-cluster bootstrap CI95 for exact agreement, equivalence, and regret,
  with 20,000 samples and seed `570057`;
- phase, seat, and original-gap breakdowns.

For adaptive also report exact set agreement, mean set Jaccard, mean
distribution Jensen-Shannon divergence, stop reasons, root units and
continuations used overall and by stratum, and the fraction saved versus
uniform 8x. Report paired adaptive-minus-uniform changes on every state.

## Integrity

Require:

- exact prerequisite hashes and frozen configuration;
- exactly 36 unique outcome-blind non-056 states, two per cell and at least
  15 games;
- exact ordered root-key and legal-action alignment;
- frozen A/B reproduction and exactly 504 new C-P attempts;
- all 36 states eligible in both streams;
- exactly 192 continuations per completed new search, no more than 96,768
  new continuations, all resolved;
- zero illegal actions and hard caps, loop-adjudication fraction at most
  `0.001`;
- adaptive replay consumes only prefix samples and 8-32 root units per arm;
- finite values and metrics;
- byte-identical first-state C repeat.

Any failure yields `adaptive_teacher_audit_invalid_no_learning_decision`.

## Gates and decisions

The point-teacher gate requires all of:

- adaptive exact agreement at least `0.70`, CI95 lower at least `0.60`;
- Pearson at least `0.85`, mean Spearman at least `0.65`;
- mean regret at most `0.025`, p90 at most `0.075`;
- equivalence at least `0.75`, CI95 lower at least `0.65`;
- every phase exact at least `0.55` and equivalence at least `0.60`;
- every seat equivalence at least `0.60`;
- mean compute at most `0.90` of uniform 8x;
- exact agreement no more than `0.03` below same-state uniform and mean
  regret no more than `0.005` above it.

The equivalence-teacher gate does not require the point exact/Spearman
thresholds, but requires:

- equivalence at least `0.75`, CI95 lower at least `0.65`;
- mean regret at most `0.025`, p90 at most `0.075`;
- every phase and seat equivalence at least `0.60`;
- mean set Jaccard at least `0.70`;
- mean diagnostic-distribution Jensen-Shannon divergence at most `0.15`;
- mean compute at most `0.90` of uniform 8x;
- equivalence no more than `0.03` below same-state uniform and mean regret no
  more than `0.005` above it.

If the point gate passes, authorize only a separately preregistered adaptive
point-teacher corpus pilot. If only the equivalence gate passes, authorize
only a separately preregistered uncertainty/equivalence-target corpus pilot.
If uniform 8x independently passes the old point or equivalence stability
gate but adaptive allocation does not, direct the next decision toward a
fixed-budget teacher rather than claiming adaptive success. Otherwise reject
this allocation rule and redesign the search teacher. No result directly
authorizes training, gameplay, promotion, or GPU rental.
