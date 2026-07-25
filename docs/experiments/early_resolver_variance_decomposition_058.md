# Early Resolver Variance Decomposition (`058`)

Frozen after `057` and before implementation or any new resolver evidence.
This is a no-training, CPU-only diagnostic. It performs no confirmation
access, corpus collection, gameplay gate, promotion, or model fitting.

## Question

`056` and `057` independently showed that uniform 8x averaging stabilizes
mid/late resolver targets but not early targets. `057` also showed that
stopping root actions from marginal confidence intervals saves computation
but loses action quality. `058` attributes early root-value variance among:

1. sampled public-belief determinizations;
2. stochastic continuation rollouts;
3. discontinuous opponent-response minimization.

The result chooses the architecture of the next search-teacher prototype; it
does not authorize teacher collection or training.

Require frozen:

- `057` result SHA-256
  `72d935d5b3e4430a362e365b4e782bcb04aca41c0c3c2ad84549f82e859a4754`,
  full integrity, and decision
  `adaptive_allocation_failed_redesign_search_teacher`;
- `055` selection SHA-256
  `4c8c9f9c499b3d8cae6a7a001c3c9b8a1f438eb466fcf0be35acb3e34072102a`;
- `052` validation SHA-256
  `865926c3e3221d8840bcbb1cdac43913bbd6efcd48b65637efca68e315a2d052`;
- frozen `035` SHA-256
  `4422a82b593340eecd02af3a0941f24f0318cc7453e6a0739fd9f524539af2cd`.

Game outcomes and the `052` confirmation split are forbidden.

## Fixed early states

Take every early row from the persisted outcome-blind `055` selection:
exactly 24 unique states, four in each actor-seat x original-gap cell, from
at least 12 games. Preserve persisted order. Root actions remain the first
four distinct canonical successors ranked by frozen `035`.

## Crossed evidence matrix

For every state evaluate a fully crossed `4 x 4` matrix:

- belief batches `b=0..3`; each contains three information-legal
  determinizations;
- continuation replicates `r=0..3`;
- four frozen root actions and the top four frozen-035 opponent responses;
- two continuations per determinization/root/response leaf.

Each `(b,r)` cell therefore uses 96 continuations; each state uses exactly
1,536 and the full matrix at most 36,864. Use the frozen continuation policy:
epsilon `0.1`, temperature `0.5`, top-k `3`, score-margin scale `20`,
turn cap `400`, and at most 64 loop-fallback triggers.

Belief root indices are
`580,000,000 + sample_index * 100 + b`. Continuation seeds are
`58,000,000,000,000,000 + sample_index * 1,000,000,000
+ r * 1,000,000 + determinization * 10,000 + continuation`.
The seed deliberately does not include `b`, root, or response: continuation
random numbers are paired across belief batches and semantic branches as far
as diverging game histories permit. All namespaces are disjoint from prior
experiments.

Use ten CPU workers and persist each completed state atomically in a JSONL
progress file. Repeat the first full state from zero and require an identical
deterministic projection.

## Hard-min and smooth-min values

For every `(b,r)` cell store all leaf continuation values and the selected
opponent action in every public information set.

Compute the deployed hard-min root values with `resolve_two_ply`. Also compute
a preregistered smooth-min counterfactual independently in each opponent
information set:

`softmin(v) = -tau * log(mean(exp(-v/tau)))`, with `tau=0.05`.

Weight each information-set contribution by its determinization count exactly
as in the hard resolver. Smooth-min changes only the opponent aggregation;
belief states, candidate responses, continuations, and leaf values are
identical.

## Frozen metrics

For hard-min and smooth-min separately:

- belief split: average `b=0,1` over every `r` versus `b=2,3` over every `r`;
- continuation split: average `r=0,1` over every `b` versus `r=2,3` over
  every `b`;
- exact best agreement, Pearson, MAE, within-state Spearman, mutual
  value-equivalence within `0.025`, symmetric cross-regret, p90 regret, and
  seat/gap breakdowns;
- game-cluster bootstrap CI95 with 20,000 samples and seeds `580057` for
  belief splits and `580058` for continuation splits.

For every state/action perform a balanced two-way decomposition of the 16
cell values. Report sums of squares and aggregate fractions attributable to
belief main effect, continuation main effect, and interaction. Report the
same fractions by seat and gap.

For hard-min opponent selections report, per state/root, the fraction not
equal to the modal response signature across the 16 cells, its aggregate
mean, and the fraction of state/roots with any switch. Report smooth-minus-
hard changes in split exact agreement, equivalence, Spearman, and regret.

## Integrity

Require:

- exact prerequisite hashes and frozen configuration;
- exactly 24 unique outcome-blind early states, four per seat x gap cell and
  at least 12 source games;
- exact root-key and legal-action alignment;
- all 24 states structurally complete across all 16 cells;
- exactly 36,864 attempted and resolved continuations;
- zero hard caps and illegal actions;
- loop-adjudication fraction at most `0.001`;
- stable opponent semantic action sets within each belief batch;
- finite leaf, root, and summary values;
- byte-identical first-state repeat.

Failure yields `early_variance_decomposition_invalid_no_design_decision`.

## Attribution decisions

Let aggregate hard-min ANOVA fractions be `B`, `R`, and `I`. Let the average
hard split regret be the mean of belief- and continuation-split regret.

Opponent minimization is declared primary when all hold:

- modal-response deviation is at least `0.20`;
- smooth-min improves the mean of the two split exact agreements by at least
  `0.10`;
- smooth-min mean split regret is at most `0.75` of hard-min.

Otherwise belief sampling is primary when:

- `B >= 0.45` and `B >= R + 0.15`;
- hard belief-split regret is at least `1.25` times hard continuation-split
  regret.

Otherwise continuation sampling is primary under the symmetric conditions:

- `R >= 0.45` and `R >= B + 0.15`;
- hard continuation-split regret is at least `1.25` times hard belief-split
  regret.

Decisions:

- opponent primary:
  `hard_min_response_switching_primary_design_soft_or_visit_response_teacher`;
- belief primary:
  `belief_sampling_primary_design_broader_public_belief_search`;
- continuation primary:
  `continuation_noise_primary_improve_rollout_evaluator`;
- otherwise:
  `mixed_early_variance_design_joint_public_belief_visit_search`.

Regardless of decision, no corpus, training, gameplay, promotion, or GPU
rental is authorized. The next prototype must be separately preregistered.
