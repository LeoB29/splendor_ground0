# Next decision after the structured and alternative assessments

18 October 2026. This recommendation was prepared before the diagnostic was run.
The owner subsequently authorized execution, completed at 13:01 Paris. See the
[diagnostic result and current recommendation](../engineering/ranker_planner_bridge_128_result.md).
Both gains were small and uncertain; the next preferred learning proposal is
a material independent-position learning curve, priced before commitment, with
external strength calibration prepared alongside it. No successor was launched.
The earlier rationale below remains a record of the pre-calculation decision.

**Do one bounded diagnostic using saved worlds for both frozen alternative
rankers before choosing an arena.** The immediate question is whether their
fourth proposals improve the planner's final choice after competing with its
unchanged first three. This is closer to deployment than another outsider-only
ranking estimate and requires no training or new simulations.

## Why this step

The structured study passed acquisition but did not establish fresh planner
improvement or an advantage attributable to its relation channels. The two
alternative recipes remain inconclusive: ridge's restricted-ranking gain is
+0.4340 percentage points, with adjusted interval [-0.3678,+1.2354]; boosting's
is +0.5143, [-0.1655,+1.1990]. Their means do not establish which model is better.
These are conditional score differences, not whole-game win-rate gains. See the
[completed assessment report](../engineering/alternative_ranker_128_result.md).

A fresh whole-game test would be legitimate even without a significant preceding
conditional result. The reason to do the cheap diagnostic first is its value
relative to the cost of an arena, not a new significance requirement. Previous
402-game arenas took approximately 78 and 91 hours overall at eight workers.
Those are historical costs, not a forecast for a new player or worker count.

## Concrete diagnostic

Use every one of the 720 audited roots and both already-sealed outsider choices.
For each ranker, compare the common first three plus its fourth proposal against
the common first three plus the anchor's fourth proposal. Select final actions
on saved worlds 0–63 and evaluate those actions on worlds 64–127. Reverse the
halves, then average the two paired differences within each root.

Report proposal changes, final-action changes, how often the fourth proposal is
selected, and paired score differences with descriptive uncertainty. Preserve
all roots, original strata, support order and ties. No split search, model
selection, tuning or subgroup search is part of this step. The
[calculation and resource specification](../engineering/ranker_planner_bridge_128.md)
sets the exact scope and checks needed before execution.

This is post hoc development analysis of already-opened data with a **64-world
selector**. A fixed plan cannot restore untouched-test status. It neither
reassesses the completed studies nor establishes the performance of the
128-world planner or whole-game strength. Original independent assessment records
do not cover every action that these new planners might choose; using only their
covered actions would produce a biased comparison.

## What follows from it

A favorable diagnostic would support considering a fresh restricted-support
arena. A weak diagnostic would reduce enthusiasm for that investment, without
proving that a 128-world or whole-game gain is impossible. Neither outcome
automatically launches anything or promotes a model. Consider effect size,
uncertainty, integration effort and affordable precision together.

If an arena is justified, Route A in the existing
[two-route design](learned_proposer_whole_game_bridge_2026-10-18.md) is the closest
deployment to what the rankers learned. Freeze a candidate, fresh paired games,
whole-game target, sample size, exact four-candidate/128-world comparison,
verification and resource limits for that separate study. Price its actual
implementation with an outcome-blind operational probe. Do not automatically
copy 402 games or convert conditional gains into win rates.

The full-legal route remains an unvalidated extension for these alternative
rankers. It requires a distinct proposal assessment or a separately justified
direct arena. The accepted faster continuation backend remains optional for a
compatible future freeze; its collector timing does not establish arena timing.

If further investment in learning is chosen instead, a material increase in
independent training positions with a measured learning curve is more informative
than another modest extension of the same recipe. Compare its cost with other
levers such as search allocation, continuation policy and external strength
calibration. Another ranking-only panel for these unchanged models is lower
priority because it leaves planner and whole-game translation unresolved.

Historical cost evidence: [broader planner arena report](../../outputs/proposal_breadth_arena_128/report.md),
[its overall duration](next_after_proposal_breadth_arena_128.md), and
[completed equal-budget arena](../../outputs/equal_budget_breadth_arena_128/recovery_03/result.json).
Interpretation also follows the accepted revision 2 of the
[independent review](independent_review_2026-10-16_generalization_and_calibration.md).
