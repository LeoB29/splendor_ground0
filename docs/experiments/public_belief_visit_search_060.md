# Public-Belief Visit-Search Diagnostic (`060`)

Frozen after `059` and before implementation or any new search evidence. This
is a no-training, CPU-only feasibility diagnostic. It performs no
confirmation access, corpus collection, gameplay gate, promotion, or model
fitting.

## Question

`059` showed that twelve belief particles at fixed compute substantially
reduce value regret but still do not produce a reproducible scalar winning
action. `060` keeps the successful twelve-particle public belief while
replacing uniform hard-choice evaluation with information-set-keyed PUCT and
a complete root visit distribution.

This is a bounded two-ply public-belief visit-search prototype, not yet a
full-game IS-MCTS bot. It tests whether visit distributions provide a stable,
value-safe teacher signal before further search engineering.

Require frozen:

- `059` result SHA-256
  `14dad95d3e41cf2b2f108a44177fe237fa12d84deddad90e58ae6acaf7cc0d47`,
  full integrity, and decision
  `belief_breadth_reallocation_insufficient_move_to_public_belief_visit_search`;
- `055` selection SHA-256
  `4c8c9f9c499b3d8cae6a7a001c3c9b8a1f438eb466fcf0be35acb3e34072102a`;
- `052` validation SHA-256
  `865926c3e3221d8840bcbb1cdac43913bbd6efcd48b65637efca68e315a2d052`;
- frozen `035` SHA-256
  `4422a82b593340eecd02af3a0941f24f0318cc7453e6a0739fd9f524539af2cd`.

Game outcomes and the `052` confirmation split are forbidden.

## Fixed states and independent searches

Reuse the exact 48-state `059` outcome-blind selection:

- all 24 early states;
- 12 balanced mid guardrails;
- 12 balanced late guardrails;
- exactly 17 source games.

Run independent left and right searches for every state. Each search uses 12
information-legal determinizations and exactly 192 simulations, cycling
particles `simulation mod 12`, so every particle receives 16 simulations.
Each simulation performs one continuation rollout.

Belief root indices are
`600,000,000 + sample_index * 10 + side`, left=`0`, right=`1`.
Continuation seeds are
`60,000,000,000,000,000 + sample_index * 1,000,000,000
+ side * 100,000,000 + simulation`.
Namespaces are disjoint across sides and all earlier experiments.

Use ten CPU workers with restart-safe per-state JSONL progress. Repeat both
first-state searches from zero and require identical deterministic
projections.

## Public-belief two-ply tree

The root contains the first four distinct canonical successors ranked by
frozen `035`. Its prior is softmax over their frozen scores at temperature
`1.0`, mixed with `0.05` uniform probability.

Each simulation:

1. Select a root action using shared-root maximizing PUCT.
2. Apply it in the scheduled determinization.
3. Key the opponent node by the root semantic action and
   `canonical_public_belief_key` from the opponent's legal observation.
4. Rank the opponent's first four distinct canonical successors with frozen
   `035`, build the same softmax-plus-uniform prior, and select a response
   with minimizing PUCT from the root player's perspective.
5. Run one frozen continuation and backpropagate normalized final score
   margin to both the shared root edge and that opponent information-set
   edge.

For any node, force each edge to be visited once in descending prior order
before normal PUCT. Thereafter:

- root score:
  `Q + 1.5 * prior * sqrt(total_visits) / (1 + edge_visits)`;
- opponent selection score:
  `-Q + 1.5 * prior * sqrt(total_visits) / (1 + edge_visits)`.

Tie-break by higher prior and then lower frozen semantic index. Opponent
semantic action sets and priors must be identical whenever the same public
information-set key is revisited. Distinct opponent private observations
remain distinct nodes; no hidden card or determinization identity enters a
node key.

The rollout contract remains epsilon `0.1`, temperature `0.5`, top-k `3`,
score-margin scale `20`, turn cap `400`, and at most 64 loop-fallback
triggers.

Store root priors, counts, probabilities, Q values, the 96-simulation
halfway snapshot, final top-visit action, all opponent-node counts, and
structural/rollout counters. Final root counts must sum to 192.

## Metrics

For independent left/right searches report:

- eligible states/games and fallback reasons;
- raw root-distribution Jensen-Shannon divergence and total variation;
- halfway-to-final JSD within each search;
- top-visit exact agreement and visit-rank Spearman;
- root-Q Pearson, MAE, within-state Spearman, directional and symmetric
  cross-regret, p90 regret, and mutual value-equivalence within `0.025`;
- phase, seat, and original-gap breakdowns;
- game-cluster bootstrap CI95 with 20,000 samples and seed `600060`.

Define the mutual equivalence group in each state as root actions within
`0.025` of the best Q under both searches. If nonempty, collapse its visit
mass into one bin and leave other roots as separate bins; otherwise retain
the raw four bins. Report equivalence-collapsed JSD overall and by phase.

To detect a prior-only result, report each search's JSD and total variation
from its frozen root prior, the fraction whose top-visit action differs from
the prior top action, visit entropy, and effective action count.

Compare the visit search descriptively against frozen `059` `12x1` on the
same states, including phase changes in exact agreement, equivalence, and
regret. The frozen scalar result cannot enter search Q calculations.

## Integrity

Require:

- exact prerequisite hashes and frozen configuration;
- exact frozen 48-state balance and 17 games;
- 96 search attempts;
- exact root/legal alignment and information-legal determinizations;
- all completed searches have 192 root visits and 192 continuations;
- attempted continuations at most 18,432 and all resolved;
- zero hard caps and illegal actions;
- loop-adjudication fraction at most `0.001`;
- stable semantic action sets and priors on every revisited opponent node;
- every particle used exactly 16 times per completed search;
- finite priors, visits, Q values, and metrics;
- identical first-state left/right repeats.

Any failure yields `public_belief_visit_search_invalid_no_design_decision`.
Structural ineligibility is substantive if recorded with exact partial
accounting; it does not invalidate the audit by itself.

## Feasibility gates

Require all:

- both searches complete on at least 90% overall and in every phase;
- mean raw JSD at most `0.15`, early at most `0.18`;
- mean equivalence-collapsed JSD at most `0.10`, early at most `0.12`;
- mean halfway-to-final JSD at most `0.10`;
- early top-visit exact agreement at least `0.55`;
- early value-equivalent agreement at least `0.65`;
- early mean Q Spearman at least `0.50`;
- early mean cross-regret at most `0.030`, p90 at most `0.075`;
- mid and late equivalence at least `0.65` and mean regret at most `0.035`;
- mean search-to-prior total variation at least `0.10`;
- at least 10% of states change prior-top action in one or both searches;
- mean raw JSD is at least `0.01` lower than the mean prior-to-search JSD,
  ensuring independent searches agree more with each other than either
  merely agrees with the parent prior.

Passing decision:
`public_belief_visit_targets_stable_authorize_separate_full_search_pilot`.

Failure decision:
`public_belief_visit_targets_unstable_pause_search_teacher_branch`.

Neither decision directly authorizes self-play, collection, training,
gameplay, promotion, or GPU rental. A passing result authorizes only a
separately preregistered full-search feasibility pilot.
