# Resolver Target Repeatability and Noise-Ceiling Audit (`055`)

Frozen after `054` and before implementing or executing any new resolver
replicate. This is a no-training, CPU-only diagnostic on already frozen
development states. It performs no confirmation access, gameplay, model
selection, promotion, or self-play collection.

## Question and prerequisites

`054` showed that a public-history model fit more exact resolver-value detail
in-sample but did not improve held-out action ordering. Before attributing that
failure to architecture or collecting a larger teacher corpus, `055` asks how
repeatable the resolver's four action values and preferred action are when the
same information-legal state is searched under independent belief and rollout
randomness.

Require these exact artifacts:

- broad resolver result `051` SHA-256
  `1a731e5edf3c63cffb2b46d8adaebba9fc50b81e490367bb0c8448ef6a885ed4`;
- `052` validation targets SHA-256
  `865926c3e3221d8840bcbb1cdac43913bbd6efcd48b65637efca68e315a2d052`;
- `054` result SHA-256
  `2e5c062e3ed12688ba66a031c1aff368f34cf62d8f9cb21b7c9ed59cdd1a7945`,
  with full integrity and decision
  `resolver_targets_not_learnable_with_current_frozen_features`;
- raw `035` SHA-256
  `4422a82b593340eecd02af3a0941f24f0318cc7453e6a0739fd9f524539af2cd`.

The old `052` confirmation corpus is forbidden. Final outcomes, winner labels,
and `054` model predictions may not be used for selection or resolver input.

## Outcome-blind state selection

Use only the 1,135 completed-search rows in the frozen `052` validation file.
For each row, define the original resolver best-minus-second value gap:

- `close`: gap at most `0.025`;
- `medium`: gap above `0.025` and at most `0.075`;
- `clear`: gap above `0.075`.

Form all 18 cells from seat `{0,1}`, phase `{early,mid,late}`, and gap stratum
`{close,medium,clear}`. Within each cell, sort rows by SHA-256 of
`"55:{game_seed}:{turn_index}"` and select the first four. Require exactly 72
unique states, four per cell, at least 18 source games, and no use of outcome
fields. Persist the selected state snapshots, original four canonical
successor keys/values, game seed, acting seat, decision index, turn, phase, and
gap stratum.

## Frozen resolver replicates

Regenerate legal actions and exact frozen `035` rankings from each state.
Require the same four ordered canonical successor keys recorded by `052`.

Run the exact `050`/`051` discovery resolver twice per selected state. Keep
every substantive parameter frozen:

- three information-legal determinizations;
- four canonical root actions;
- four information-set-consistent opponent responses;
- four stochastic continuations per leaf;
- epsilon `0.1`, temperature `0.5`, top-k `3`;
- loop-safe score-margin rollout, scale `20`, max turn `400`, and maximum 64
  model-loop fallbacks.

Only the random namespaces differ:

- replicate A: belief root-index start `11,000,000`, continuation seed start
  `1,000,000,000,000,000`;
- replicate B: belief root-index start `21,000,000`, continuation seed start
  `2,000,000,000,000,000`.

Both namespaces are disjoint from the original resolver and each other.
Attempt both replicates on all 72 states. A changed belief sample may
legitimately trigger one of the exact frozen structural fallback reasons even
though the original namespace resolved that state; record this as target
instability, not infrastructure failure. Every completed search must use
exactly 192 continuations, so total continuations must equal 192 times the
number of completed fresh searches and may be at most 27,648. Run with ten CPU
workers. Repeat replicate A for the first selected state from zero and require
its deterministic projection--whether a completed search or a structural
fallback--to be byte-identical.

## Fixed metrics

First report structural resolution repeatability: completion rate for each
replicate, both-completed fraction, completion/fallback agreement, fallback
reasons, and cell breakdowns. The primary value comparison is fresh A versus
fresh B on states where both completed. Original-versus-A and
original-versus-B use their respective completed states and are secondary
checks. For each value comparison report:

- flattened four-value Pearson correlation and MAE;
- mean within-state Spearman correlation, using average ranks for ties;
- exact selected-best agreement;
- directional cross-regret: the other estimate's best value minus its value
  for the action selected by this estimate;
- symmetric mean cross-regret, 90th percentile, and the fraction of
  directional choices with regret at most `0.025` and `0.05`;
- value-equivalent agreement, defined as both directional cross-regrets at
  most `0.025`;
- phase, seat, and original-gap-stratum breakdowns.

For A versus B, compute game-cluster bootstrap CI95 intervals with 20,000
samples and seed `550055` for exact best agreement, value-equivalent
agreement, and mean symmetric cross-regret.

Also report all-three consensus across original/A/B: exact unanimous best
fraction, at-least-two best fraction, all-three value-equivalent fraction, and
per-action value standard deviation.

## Integrity

Require:

- all prerequisite hashes and the exact frozen configuration;
- exactly 72 unique selected states and four in every selection cell;
- unique legal action codec indices and exact ordered root-key alignment;
- exactly 144 fresh search attempts with complete structural accounting;
- at least two both-completed states in every selection cell;
- total continuations exactly 192 times completed searches and at most 27,648,
  all resolved;
- zero illegal actions and hard caps;
- loop-adjudicated fraction at most `0.001`;
- finite values and metrics;
- byte-identical first-state replicate-A repeat;
- no training, confirmation, gameplay, or model mutation.

Integrity failure yields
`repeatability_audit_invalid_no_learning_decision`.

## Decision

Use fresh A versus B and the following fixed priority.

### 1. Exact point targets are repeatable

Decision is
`resolver_point_targets_repeatable_authorize_richer_representation_protocol`
only if all hold:

- exact best agreement at least 70%;
- both-completed fraction at least 80%;
- its game-bootstrap CI95 lower bound at least 60%;
- flattened value Pearson at least `0.70`;
- mean within-state Spearman at least `0.60`;
- mean symmetric cross-regret at most `0.025`;
- at least 80% of directional choices have regret at most `0.05`;
- clear-gap exact best agreement at least 80%;
- every phase and seat has exact best agreement at least 55%.

This authorizes only preregistration of a richer end-to-end
state/action/history representation on existing development data.

### 2. Disagreements are mostly value-equivalent

If the exact point-target gate fails, decision is
`resolver_equivalence_targets_authorize_uncertainty_aware_protocol` only if:

- value-equivalent agreement is at least 75%;
- both-completed fraction at least 70%;
- its bootstrap CI95 lower bound is at least 65%;
- mean symmetric cross-regret is at most `0.035`;
- 90th-percentile symmetric cross-regret is at most `0.075`;
- every phase and seat has value-equivalent agreement at least 60%.

This authorizes only preregistration of equivalence-set or uncertainty-aware
teacher targets, not a larger exact-value corpus.

### 3. Resolver point labels are too noisy

Otherwise decision is
`resolver_targets_too_noisy_block_direct_distillation`. Direct resolver-value
or hard-best-action distillation remains blocked. The resolver remains a
proven strong search agent; this decision does not weaken `051`.

No outcome authorizes fresh collection, GPU rental, gameplay, promotion, or
self-play deployment.
