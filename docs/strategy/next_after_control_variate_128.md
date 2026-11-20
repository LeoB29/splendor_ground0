# Next step after the control-variate feasibility result

2026-10-11. Recommendation only. The owner requested a considered next step,
not execution. No implementation, fitting, inference, simulation, resampling,
new source games or launch was performed. AGENT_CURRENT.md was read first;
three independent reviewers examined learning, search and programme direction.
All frozen work and the intentionally dirty checkout remain preserved.

## Recommendation

Run one matched experiment testing explicit action-to-card resource relationships
inside the already useful learned proposer, with a prospectively specified fresh
decision-quality test. Confidence is moderate. This tests whether making a
particular relationship easier to learn improves transfer; it does not presume
that representation has been established as the cause of previous failures.

Retain the complete original 161 networks, frozen. Add a small shared residual
action scorer whose output starts at exactly zero. Compare explicit before/after
resource shortfalls for individual known cards with an otherwise matched scorer
given the constituent raw costs and resources. Use all three original fits in
both arms. Keep the existing unweighted R1 BROAD training recipe fixed.

Unlike the earlier runner-up proposal, fresh evaluation belongs in this same
prospective experiment. The repeatedly inspected 168 validation roots cannot
provide its primary success claim. Historical validation remains diagnostic.

## What the evidence says

The completed recovery02 control-variate experiment found a predicted
variance-times-CPU ratio of 1.00351077, versus a target at most 0.80. Its paired
95% interval is [1.00330118, 1.00372322], conditional on the selected TRAIN rule
and timings. It excludes timing uncertainty and is not a measured player
slowdown. The chosen recipe uses 127 full trajectories and 128 total prefixes:
effectively no substitution. No game win rate or deployed estimator resulted.

The failure has a mechanism beyond an arbitrary allocation boundary. From the
saved TRAIN pooled moments, covariance squared divided by the product of the
two variances is 0.00319764, 0.02283685 and 0.13376547 at 4, 8 and 16 plies.
These are post-hoc descriptive calculations from existing aggregates. For the
same pooled global-linear control, an exactly known prefix mean available for
free would remove only about 0.32%, 2.28% or 13.38% of variance at unchanged
full-trajectory count. Real prefixes cost CPU and their means must be estimated.
Thus even that idealization misses the 20% target. Short prefixes supply little
relevant paired signal; longer ones consume much of the original computation.
This closes the tested frozen143 evaluator/linear-prefix recipe, not all control
variates or learned evaluators. It adds no causal evidence for a neural
representation bottleneck.

The learning evidence is separate and more directly relevant to the next step:

- The original learned proposer won 65.796% against the original 128-world
  planner over 402 fresh games. Learning can improve planning in this programme.
- The independent-game coverage experiment acquired about 97% of the added
  training regret, but BROAD-minus-anchor was -0.1183 percentage points on 360
  fresh positions, with paired95% interval [-0.6825, +0.4449]. The teacher
  positive control still gained +1.4659pp [0.9071, 2.0692]. Fourfold breadth of
  that added component was insufficient; this does not refute general data
  scaling or population changes.
- Planning-opportunity weighting acquired 93-95% of the relevant added training
  regret. On reused validation it gained only +0.020926pp against BROAD,
  interval [-0.413876, +0.432478]. The teacher remained +1.327660pp above anchor.
- More dose, changed learning rate, soft targets, frozen-head training and
  opportunity weights have not delivered another reliable planning improvement.
  This lowers the priority of another nearby optimization or weighting sweep.
  Frozen-head underacquisition does not establish that the frozen representation
  lacks useful information.

These results favor investigating transfer of acquired guidance. They do not
uniquely separate representation, noisy finite-world labels, limited independent
coverage, long-horizon interactions or continuation/population mismatch.

Historical architecture negatives matter. Experiment149's afterstate learner
and151's shared-card attention learner could fit training data without useful
new-position move gains. Experiment153's compact public potential also failed
and merged distinguishable actions. Therefore a generic new card encoder or
another public scalar critic would have a weak rationale. The proposed contrast
is specifically explicit arithmetic versus the same underlying raw information,
as a correction to a proven proposer, on the larger current corpus and proposal
objective. It must not be advertised as the first test of card sharing.

Experiment134 lost decisively under its outcome-blind owner runtime amendment;
the automated operational-invalid flag is not the final scientific interpretation.
Experiment137 failed its refill-coverage mechanism gate and does not reject
chance-node reasoning generally. Neither supports returning automatically to
those search changes. More recent eight-candidate search did improve gameplay,
but its equal-budget eight-by64 alternative scored about 49.75%; more search
coverage is not a reliable efficiency gain by itself.

## The single proposed experiment

1. Preserve each original 161 policy and value network exactly. Train only the
   added residual scorer, with exact initial equivalence to the original logits.
   Pair three treatment fits with three raw-component controls. Match all
   architecture, parameter count, initialization, trainability and optimizer
   choices between those two families.
2. Treatment explicitly expresses how each legal move changes colored-resource
   deficits and shared-gold-adjusted shortfalls to individual known cards.
   The raw control receives the corresponding before/after resources, discounts
   and costs, with identical card attributes, locations, masks and action context.
   Fix channel layout and model size before fitting. Preserve the raw information
   needed to recover the relationships; do not manufacture an informationally
   weaker control or collapse concrete legal-action distinctions.
3. Use only observer-available information and deterministic consequences of the
   proposed action. Do not read future market refills, unseen deck order or
   private opposing cards. Account correctly for payments, returns, new discounts,
   gold, card removal/availability and reservation ownership. Do not spend the same
   gold independently on every color. Existing rules handle actual transitions;
   the new branch must not silently simulate an unknown refill as known.
4. Keep the exact unweighted R1 BROAD corpus, soft-target temperature .05, batch
   streams, optimizer and final 1,784-update endpoint. No opportunity weighting,
   teacher replacement, additional training corpus, checkpoint selection or dose
   rescue. Authenticate inherited evidence rather than rerunning historical fits.
5. Record training acquisition for both arms and all fits. Old 2,489 validation
   rows and the saved 336 conditional roots may diagnose behavior, but cannot
   choose a model, feature revision, checkpoint or whether an otherwise valid
   treatment receives the planned fresh test. An acquisition failure limits the
   interpretation of a later null; it is not evidence against relations generally.
6. Seal all six final models, unchanged anchors, population, sample, randomness,
   analysis and decision rules before fresh evaluation. A proposed primary panel
   is 360 positions from 360 distinct new games, balanced across the same 18
   source-fit/seat/phase cells as the independent-game coverage experiment.
   Use its declared fast161 source population and outcome-blind position selection;
   establish disjoint source ancestry and random streams. Retain private-reserve
   cases and short action supports. This population does not represent every
   state visited by the deployed search champion.
7. Evaluate the treatment's and control's unrestricted lawful fourth proposals
   with the common first three in the unchanged 128-world planner. Include the
   unchanged anchor and eight-candidate teacher as controls. Share the 128
   selection worlds across the union of candidate actions. Seal every arm's
   selected action before collecting a separate 128-world assessment block for
   the distinct selected actions. No assessment world may select a move.
8. Primary investment gate: treatment-minus-anchor at least +0.5pp conditional
   expected score with a paired95% lower endpoint above zero, and treatment-minus-
   raw-control paired95% lower endpoint above zero. Freeze both contrasts in
   advance and require them jointly. Preserve all three fits; report per-fit
   results without picking a winner. Resample whole source games within the
   declared cells, retaining all arms jointly. Preserve unknown-outcome bounds.
   This threshold is an investment judgment, not a power guarantee or win-rate
   promise. Small uncertain effects remain inconclusive.

The encoder already includes target-card affordability, token flows and aggregate
affordability changes (src/splendor_ai/encoding/action_features.py). The addition
is granular action-by-card relationships across the currently known choices,
especially for token-taking moves with no designated target card. It adds
computational accessibility, not newly observable facts. The owner's reservation
denial insight stays a separate parked hypothesis; this experiment is not its
frequency test or a rule telling the player always to reserve first.

## What each outcome would change

- A fresh joint pass supports this particular explicit-relations correction and
  earns a separately proposed complete-game comparison. Conditional continuation
  scores are not full-game win rates, broad strength or superhuman performance.
- Similar useful improvement by both residual families supports the common
  residual architecture/training constraint, not the explicit arithmetic claim.
  Report that honestly without replacing the prespecified treatment post hoc.
- Good acquisition but no useful fresh planning gain weighs against this local
  transfer remedy. Deprioritize further nearby compression tweaks and return to
  a programme-level choice about teacher/population/long-horizon mechanisms.
- Poor acquisition closes the fixed training recipe; it cannot refute the
  broader representation idea. No automatic longer training or feature sweep.

The strongest alternative is further search breadth against a matched-compute
precision control. It has real empirical motivation, but is likely to buy more
expensive search rather than resolve repeatable learning. Independent strong
opposition remains essential to eventual superhuman claims. It is not yet a
more localized test of the demonstrated transfer problem.

## Execution boundary and assurance

This is a recommendation, not a frozen implementation protocol or authorization.
If authorized, finalize the exact feature layout, residual size, seed namespaces,
analysis and prospective resource allowance before any production learning.
Use the existing verification policy: check all changed inputs, proposals and
source decisions, reproduce the new optimizer trajectories, audit all statistics,
and use prespecified sampled continuation replay plus bounded stress cases for
unchanged simulation machinery. Do not repeat exhaustive historical replays.

The earlier 45-90 minute estimate applied only to a saved-data screen; it does
not cover this fresh version. Price the fresh collection and audit workload with
an outcome-blind probe before the main collection. No GPU, long arena, recurring
monitor or automatic successor is part of this recommendation. Later execution
should be hidden and checkpointed, with an explicit interruption recovery path.

## Principal evidence

- outputs/control_variate_128_recovery_02/{result,selection_lock,assessment_result}.json
- outputs/learned_proposal_128/report.md
- outputs/independent_game_coverage_128_recovery_01/report.md
- outputs/planning_opportunity_128/report.md
- outputs/frozen_representation_128/report.md
- docs/engineering/{afterstate_advantage_149,shared_cards_151,public_feature_advantage_153}_result.md
- docs/engineering/independent_game_coverage_128.md
- docs/engineering/expanded_root_resolver_fresh_champion_comparison_134_owner_runtime_amendment.md
- outputs/expanded_root_resolver_fresh_champion_comparison_134/result.json
- outputs/alphazero_stratified_root_refill_search_137/result.json
- docs/strategy/independent_superhuman_review_after_134_137.md
- docs/strategy/next_after_equal_budget_breadth_128.md
- docs/engineering/verification_policy.md
