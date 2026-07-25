# Compute-Matched Belief-Breadth Audit (`059`)

Frozen after `058` and before implementation or any new resolver evidence.
This is a no-training, CPU-only search-allocation audit. It performs no
confirmation access, corpus collection, gameplay gate, promotion, or model
fitting.

## Question

`058` attributed 61.74% of early root-value variance to narrow sampled public
belief, versus 9.23% to continuation randomness. `059` asks whether the
existing 192-continuation resolver budget becomes a stable teacher when it is
reallocated from repeated continuations to broader determinizations.

Require frozen:

- `058` result SHA-256
  `9369c7a6ace8cacd0907fa4dd6d2dd72a48495aae3b09725cd5664d0ffa74a64`,
  full integrity, and decision
  `belief_sampling_primary_design_broader_public_belief_search`;
- `055` selection SHA-256
  `4c8c9f9c499b3d8cae6a7a001c3c9b8a1f438eb466fcf0be35acb3e34072102a`;
- `052` validation SHA-256
  `865926c3e3221d8840bcbb1cdac43913bbd6efcd48b65637efca68e315a2d052`;
- frozen `035` SHA-256
  `4422a82b593340eecd02af3a0941f24f0318cc7453e6a0739fd9f524539af2cd`.

Game outcomes and the `052` confirmation split are forbidden.

## Fixed states

Use persisted outcome-blind `055` selection order:

- all 24 early states, four per seat x original-gap cell;
- the first two mid states per seat x gap cell, 12 total;
- the first two late states per seat x gap cell, 12 total.

Require 48 unique states from exactly 17 games. Early is the primary phase;
mid and late are guardrails.

## Fixed candidate configurations

Each estimate ranks the same four canonical root successors and the top four
frozen-035 opponent responses. Compare:

- `3x4`: 3 determinizations and 4 continuations/leaf;
- `6x2`: 6 determinizations and 2 continuations/leaf;
- `12x1`: 12 determinizations and 1 continuation/leaf.

Every completed estimate uses exactly
`determinizations x 4 roots x 4 responses x continuations/leaf = 192`
continuations. All determinizations are pooled into one call to
`resolve_two_ply`; do not separately minimize small batches and then average.

Keep the frozen continuation policy: epsilon `0.1`, temperature `0.5`, top-k
`3`, score-margin scale `20`, turn cap `400`, and at most 64 loop-fallback
triggers.

## Nested paired namespaces

Generate independent left and right estimates for every configuration.
Within one state/side, all three configurations use the same belief root
index and `deterministic_belief_range`: `3x4` consumes the first three of the
same 12 determinizations, `6x2` the first six, and `12x1` all twelve.

Belief root indices are:

`590,000,000 + sample_index * 10 + side`, where left=`0`, right=`1`.

Each configuration has twelve continuation slots per semantic leaf. Pair slot
randomness across configurations with:

`slot = determinization * continuations_per_leaf + continuation`.

Continuation seeds are:

`59,000,000,000,000,000 + sample_index * 1,000,000,000
+ side * 100,000,000 + slot`.

The seed excludes configuration, root action, and opponent response. Left and
right are disjoint; all namespaces are disjoint from earlier experiments.

Attempt all 288 estimates (`48 x 3 x 2`) using ten CPU workers, persisting
each completed state in a restart-safe JSONL progress file. Repeat the first
state's left estimate for all three configurations from zero and require
identical deterministic projections.

## Metrics

For each configuration, compare left versus right on:

- eligible states/games and structural fallback reasons;
- exact best-action agreement;
- flattened Pearson and MAE;
- mean within-state Spearman;
- directional and symmetric cross-regret, p90, and regret fractions at most
  `0.025` and `0.05`;
- mutual value-equivalent agreement within `0.025`;
- phase, seat, and original-gap breakdowns;
- game-cluster bootstrap CI95 for exact agreement, equivalence, and mean
  regret, using 20,000 samples and seeds `590060`, `590061`, and `590062`.

For `6x2` and `12x1`, report paired candidate-minus-`3x4` changes overall and
by phase. On early states, game-cluster bootstrap the exact-agreement gain,
equivalence gain, and regret change with 20,000 samples and seeds `590160`
and `590161`.

Report estimate wall/CPU time and throughput per configuration. Runtime is
descriptive, not a learning gate.

## Integrity

Require:

- exact prerequisite hashes and frozen configuration;
- exactly 48 unique outcome-blind states, phase/cell counts
  `early=4`, `mid=2`, `late=2`, and exactly 17 source games;
- exact root-key and legal-action alignment;
- exactly 288 estimate attempts;
- exactly 192 continuations for every completed estimate and no more than
  55,296 total;
- all attempted continuations resolved;
- zero hard caps and illegal actions;
- loop-adjudication fraction at most `0.001`;
- stable opponent semantic action sets within every completed estimate;
- nested belief roots and paired continuation-slot seeds exactly as frozen;
- finite action values and metrics;
- identical first-state left repeats for all three configurations.

Failure yields `belief_breadth_audit_invalid_no_design_decision`.

Structural ineligibility is a substantive configuration result, not an audit
integrity failure, provided it is recorded before any continuation from that
estimate and every attempted estimate is accounted for.

## Candidate gates

Evaluate `12x1` first, then `6x2`. A candidate passes only if all hold:

- both replicates complete on at least 90% of all states and at least 90% of
  each phase;
- early exact agreement at least `0.55`;
- early value-equivalent agreement at least `0.60`;
- early mean Spearman at least `0.50`;
- early mean regret at most `0.035` and p90 at most `0.075`;
- versus matched `3x4` early states, exact gain at least `0.15` with
  bootstrap CI95 lower bound at least `0.00`;
- early equivalence gain at least `0.10` with CI95 lower bound at least
  `0.00`;
- early mean regret at most `0.70` of `3x4`, and paired regret-change
  bootstrap upper bound at most `0.00`;
- in both mid and late, exact and equivalence are no more than `0.10` below
  `3x4`, and mean regret is no more than `0.01` above `3x4`;
- total hard-cap and loop-adjudication integrity passes.

Decisions:

- passing `12x1`:
  `twelve_by_one_belief_breadth_stable_authorize_separate_resolver_pilot`;
- otherwise passing `6x2`:
  `six_by_two_belief_breadth_stable_authorize_separate_resolver_pilot`;
- neither:
  `belief_breadth_reallocation_insufficient_move_to_public_belief_visit_search`.

No decision directly authorizes collection, training, gameplay, promotion,
or GPU rental. Any resolver pilot must be separately preregistered.
