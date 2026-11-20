# Draft whole game test of learned proposals

18 October 2026. Design preparation only. The candidate, sample size and launch
decision remain open until the current studies are complete and interpreted.
The question is whether changing the fourth proposal improves complete-game
score against the established anchor planner at the same simulation budget.

The running structured study and its frozen dose, acquisition requirements,
success criteria, verification and resources remain unchanged. Its +0.5-point
criterion does not become a requirement for the alternative rankers, which keep
their two Bonferroni-adjusted 97.5% intervals. This draft authorizes no fitting,
prediction, evaluation, simulation, benchmark, restart or monitoring.

## Two integration routes

**Route A: choose within a small reconstructed candidate set.** At each real
decision, retain the anchor planner's first three canonical proposals. Rebuild
the alternative study's outsider set: the original 161 anchor's best remaining
action, up to two next-ranked actions and up to two distinct deterministic
random alternatives. A fixed ridge or boosting model selects one of these at
most five outsiders as the fourth proposal. Only the first three plus that
choice receive planner simulations; constructing the alternatives requires no
teacher rollouts. Preserve ordered ties, concrete representatives and short
supports. Freeze a lawful mapping from game and decision identity to new random
seed domains, disjoint from consumed studies; offline panel indices cannot
simply serve as live game IDs.

This is the closest deployment to the alternative study's measured ranking
task. After implementation and information-boundary checks, it can receive a
whole-game comparison without first demonstrating selection over all legal
actions. A favorable ranking result motivates that test but does not establish
playing strength or guarantee that its gain survives the planner's final choice.

**Route B: choose from all legal actions outside the first three.** Score every
eligible canonical action and use the highest-scoring one as the fourth
proposal. For a structured raw or relations model, preserve maximum concrete
logit aggregation; its current study already assesses this proposal mechanism.
For an alternative ranker, preserve its trained mean-feature aggregation within
canonical groups. Applying it to the full legal set is a broader, unvalidated
extension. Recorded outsider outcomes cannot assess an unrecorded proposal.
Any additional proposal assessment needs fresh evidence and its own prospective
design; whether a separate conditional screen is worth its cost remains open.
Neither route changes the continuation policy or substitutes the learner's
score for simulated terminal outcomes.

## Shared whole game comparison

- **Comparator and controls.** Use the original 161 anchor fourth proposal in
  the otherwise identical planner. The proposed comparison keeps four candidates
  at full support and 128 worlds per candidate, unchanged greedy-035
  continuations, legal world construction, canonical representatives and final
  tie rules. Preserve the existing behavior when fewer than four groups exist.
  Bind exact checkpoint and planner identities in the eventual protocol.

- **Fit coverage and pairing.** Retain all three original source-fit contexts;
  do not select the best individual fit. For a neural candidate, pair each model
  with its own anchor. For a pooled alternative model, use that same model across
  all three anchor contexts. Use distinct fresh seed panels across fits and play
  each seed twice with seats swapped, keeping assigned models fixed across the
  pair. Reconstruct opponents' choices from their own lawful inputs.
  Derive shared search streams from pair identity and lawful observations using
  fixed domain tags, avoiding policy-dependent random-number consumption.

- **Endpoint and uncertainty.** Measure actual terminal game score: win 1,
  draw 0.5, loss 0. Average within each seat-swapped pair and equally across fit
  strata. Resample complete pairs within strata; if candidates share a seed,
  retain all its matchups together. Turns and simulated worlds are not independent
  games. Report superiority over
  50% separately from the chosen practical target. If multiple candidates or
  routes receive confirmatory tests, freeze multiplicity handling in advance.

- **Cost and verification.** Record delivered continuation counts and total
  decision time, including candidate generation and model inference; report
  mean, median and upper-tail latency. Equal simulations do not imply equal wall
  time. Check every actual game and decision, lawful inputs, proposals and final
  statistics; prescribe sampled continuation replay plus full stress cases under
  the verification policy. Freeze failure, unresolved-outcome, resource and
  recovery rules; preserve failed evidence and never count a technical failure
  as a draw. A separate operational probe prices the complete authorized design.

The accepted fast continuation backend may be explicitly bound for both players
only in a compatible newly frozen experiment, with any necessary integration
checks. It remains optional; this draft starts no further speed campaign.

## Decisions after the current studies

Preserve the blind sequence: check that the main study's `final_status.json` and
`main_complete.json` both exist without reading their contents, validate the
alternative study's identity, create and authenticate its prediction seal, then
open outcomes and evaluate both fixed rankers. If the required audited panel
does not exist, leave the add-on unevaluated. This sequence is not launched here.

Consider effect sizes, uncertainty, whole-game relevance and cost together;
proceed, defer for a specific validation, or stop this line. Current results may
inform the next design, but any panel used for that choice is development
evidence for the new candidate. Confirm it on fresh whole games.

Before launching, freeze the selected route and model family, checkpoint mapping,
integration code, seed schedule, pair count, analysis, practical target,
verification and resource limits. Choose the pair count from the whole-game
effect worth detecting, paired variance assumptions and affordable verification.
Do not convert conditional-score percentage points into an assumed game win
rate, copy an earlier arena's size automatically, or extend the sample after
seeing its results. Success would support this player against this comparator
and budget; broader strength and promotion require their own evidence.

Sources: [structured proposer protocol](../engineering/structured_proposer_128.md),
[alternative ranker protocol](../engineering/alternative_ranker_128.md),
[candidate reconstruction](../../tools/alternative_ranker_128/prediction.py),
[earlier paired arena protocol](../engineering/learned_proposal_arena.md),
[verification policy](../engineering/verification_policy.md).
