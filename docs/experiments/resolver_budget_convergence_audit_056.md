# Resolver Search-Budget Convergence Audit (`056`)

Frozen after `055` and before implementing or executing any additional
resolver search. This is a no-training CPU audit. It performs no confirmation
access, game collection, gameplay gate, promotion, or model fitting.

## Question and prerequisites

`055` proved that one 192-continuation resolver call has repeatable coarse
values but unstable within-state ordering. `056` tests whether independent
search averaging economically converts that strong procedure into a stable
teacher target.

Require:

- `055` result SHA-256
  `4aa3b54e77a4856c2c5cd1bb18b1a4088755bda7679bd52a7ce58627ba622714`,
  with full integrity and decision
  `resolver_targets_too_noisy_block_direct_distillation`;
- `055` selection SHA-256
  `4c8c9f9c499b3d8cae6a7a001c3c9b8a1f438eb466fcf0be35acb3e34072102a`;
- `052` validation SHA-256
  `865926c3e3221d8840bcbb1cdac43913bbd6efcd48b65637efca68e315a2d052`;
- frozen `035` SHA-256
  `4422a82b593340eecd02af3a0941f24f0318cc7453e6a0739fd9f524539af2cd`.

The `052` confirmation split and all outcomes are forbidden.

## Fixed states

Within each of the 18 `055` seat x phase x original-gap selection cells, keep
the first two rows in persisted `055` selection order. Require 36 unique
states, two per cell, spanning at least 15 games. Selection may not inspect
`055` replicate values, completion, outcomes, or metrics.

Reuse exact completed `055` replicates A and B for those states. Regenerate and
verify their ordered four canonical root keys before using them.

## Sixteen independent estimates

For each state, obtain aligned resolver estimates A through P. A and B are the
frozen `055` searches. C through P are new searches with the exact frozen
`050`/`051` resolver contract:

- three information-legal determinizations;
- four root actions and four opponent responses;
- four stochastic continuations per leaf, exactly 192 per completed search;
- epsilon `0.1`, temperature `0.5`, top-k `3`;
- loop-safe score-margin rollout, scale `20`, turn cap `400`, and maximum 64
  model-loop fallbacks.

Estimate index `i` from 1 through 16 uses belief-root start
`(10,000,000 * i) + 1,000,000` and continuation-seed start
`1,000,000,000,000,000 * i`. Thus A/B reproduce `055` starts
11,000,000/21,000,000 and 1e15/2e15; C-P are disjoint from all prior
namespaces.

Attempt C-P for every state: 504 new searches and at most 96,768 new
continuations. A changed determination may produce an exact frozen structural
fallback; record it rather than replacing the state. Every completed search
must have exact 192-branch accounting. Use ten CPU workers and persist every
state/estimate independently for restart safety. Repeat C on the first state
from zero and require identical deterministic projection.

## Matched budget ensembles

For a completed set of estimates, average aligned four-action values within
fixed disjoint blocks:

- 1x: compare A vs I, B vs J, ..., H vs P (eight comparisons/state);
- 2x: compare means AB vs IJ, CD vs KL, EF vs MN, GH vs OP;
- 4x: compare means ABCD vs IJKL and EFGH vs MNOP;
- 8x: compare mean A-H versus mean I-P.

An ensemble comparison is eligible only when all searches in both blocks
completed. Effective budgets are 192, 384, 768, and 1,536 continuations per
side. No ensemble may use the original 052 estimate as ground truth.

## Fixed metrics

At each budget report:

- eligible state-comparisons and unique states/games;
- flattened four-value Pearson and MAE;
- mean within-state Spearman with average ranks for ties;
- exact best-action agreement;
- both directional cross-regrets, mean symmetric regret, and p90;
- directional regret fractions at most `0.025` and `0.05`;
- mutual value-equivalent agreement at `0.025`;
- phase, seat, and original-gap breakdowns;
- game-cluster bootstrap CI95 with 20,000 samples and seed
  `560056 + budget` for exact agreement, equivalence, and mean regret.

Report changes from 1x to 2x/4x/8x. As an external descriptive check only,
compare the original 052 action against the full mean of all completed A-P
estimates; it cannot enter a gate.

Also report wall time, aggregate worker CPU time when available, searches per
second, and continuation multiplier. No speed threshold is a learning gate.

## Integrity

Require:

- exact hashes and configuration;
- 36 unique outcome-blind states, two per cell, at least 15 games;
- exact ordered root-key and legal-action alignment;
- all frozen A/B rows reproduced from the `055` result;
- exactly 504 new C-P attempts;
- at least one eligible 8x state in every selection cell and at least 80% of
  all selected states eligible at 8x;
- new continuations equal 192 times completed new searches and are at most
  96,768, all resolved;
- zero illegal actions and hard caps;
- loop adjudication fraction at most `0.001`;
- finite values/metrics;
- byte-identical first-state C repeat.

Failure yields `budget_convergence_audit_invalid_no_learning_decision`.

## Decision

Use the following priority.

### 1. Stable averaged point target

Decision is
`averaged_resolver_targets_stable_authorize_teacher_protocol` only if at 8x:

- exact best agreement at least 70%, bootstrap lower at least 60%;
- flattened Pearson at least `0.85`;
- mean Spearman at least `0.65`;
- mean symmetric regret at most `0.025` and p90 at most `0.075`;
- mutual value-equivalent agreement at least 75%, bootstrap lower at least
  65%;
- at least 80% of directional choices have regret at most `0.05`;
- every phase exact agreement at least 55% and equivalence at least 60%;
- versus 1x, exact agreement improves by at least 15 percentage points,
  Spearman by at least `0.15`, and mean regret falls by at least 40%.

This authorizes only preregistration of synchronous collection using averaged
resolver targets and a compute/cost plan. It does not authorize collection or
training itself.

### 2. Stable equivalence/uncertainty target

If exact point stability fails, decision is
`averaged_resolver_equivalence_stable_authorize_uncertainty_protocol` only if
at 8x:

- equivalence at least 75%, bootstrap lower at least 65%;
- mean regret at most `0.035`, p90 at most `0.075`;
- every phase and seat equivalence at least 60%;
- equivalence improves at least 15 points and mean regret falls at least 30%
  versus 1x.

This authorizes only an uncertainty/equivalence-target protocol.

### 3. Positive but insufficient convergence

If neither gate passes but 8x exact agreement improves at least 10 points and
mean regret falls at least 25% versus 1x, decision is
`budget_convergence_positive_but_8x_insufficient_redesign_teacher`.

### 4. No useful convergence

Otherwise decision is
`budget_averaging_does_not_stabilize_resolver_targets`.

The last two decisions close brute-force averaging at this budget and direct
the project toward a search teacher with converged visit distributions.

No decision authorizes GPU rental, gameplay, promotion, or self-play
deployment. GPU acceleration remains irrelevant to Python environment
rollouts; rented CPU parallelism becomes worth pricing only if a stable target
gate passes.
