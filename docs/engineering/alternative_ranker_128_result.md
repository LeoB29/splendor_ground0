# Structured proposer and alternative ranker results

18 October 2026. The structured proposer study completed at 09:59 Paris. The
owner then authorized the deferred alternative-ranker assessment. Its frozen
prediction and evaluation commands completed without new training or simulations.

Neither study establishes a fresh learned improvement under its registered
decision rule. The structured model misses its joint success criterion. Both
alternative rankers have positive point estimates, but their adjusted intervals
include zero. Useful teacher headroom remains evident.

All numbers below are percentage points of expected conditional score
(win 1, draw 0.5, loss 0). They are neither Splendor prestige points nor observed
whole-game win-rate changes. The two studies measure different interventions and
their effect sizes should not be compared as equivalent planner gains.

## Main planner assessment

All six fits completed 28,544 updates and independent reproduction. Every fit
passed the original acquisition gates: added-TRAIN regret capture ranged from
97.5634% to 98.0711%, and old-TRAIN capture from 97.1714% to 97.7559%.
The fresh panel contains 720 distinct source games, balanced across 18 cells.
Planner choices were assessed on 128 worlds disjoint from selection.

| Contrast | Mean gain | Paired 95% interval |
| --- | ---: | ---: |
| Relations minus anchor | +0.1297 | [-0.2365, +0.4861] |
| Relations minus raw | +0.1552 | [-0.0217, +0.3456] |
| Raw minus anchor | -0.0255 | [-0.3776, +0.3158] |
| Teacher minus anchor | +1.1627 | [+0.7785, +1.5750] |
| Teacher minus relations | +1.0330 | [+0.6071, +1.4979] |

The mean relations gain is below the required +0.5 points, and neither required
95% lower confidence endpoint is above zero. Thus the original classification
is `practical_progress_not_established`, with no established attribution to the
explicit relation channels. Strong training acquisition did not establish fresh
transfer in this recipe; teacher gains cannot rescue its failed primary criteria.

All prescribed original audits passed. Main verification replayed 86,912 of
922,112 recorded continuations, including 29 full stress roots, alongside
structural checks at all 720 positions. The 18-position probe received full
replay. Main replay was sampled, not exhaustive. No unresolved continuation
outcomes were recorded.

## Alternative restricted ranking assessment

Both already-fitted models were evaluated on all 720 positions. Each chose among
at most five recorded teacher outsiders beyond the common first three proposals.
Their fixed choices were assessed using the saved 128 selection-world outcomes;
these outcomes had not selected the models or their choices. This measures outsider ranking,
not the final four-candidate planner decision measured above.

| Frozen recipe | Mean gain over assigned anchor | Bonferroni-adjusted 97.5% interval | Conclusion |
| --- | ---: | ---: | --- |
| Ridge regression | +0.4340 | [-0.3678, +1.2354] | Inconclusive |
| Histogram gradient boosting | +0.5143 | [-0.1655, +1.1990] | Inconclusive |

Both classifications are `restricted_ranking_gain_not_established`. There is no
+0.5-point gate for this add-on; boosting's point estimate above that number
does not change its conclusion. Both intervals remain compatible with
deterioration, zero effect or useful improvement.

The registered calculation uses 50,000 joint stratified root bootstraps, seed
2713000000, and 40 roots in each of 18 cells. Its two 97.5% percentile intervals
provide nominal Bonferroni family coverage of 95%, without a finite-sample
coverage guarantee. All 720 roots remain in the denominator. Outsider counts
0/1/2/3/4/5 occur at 13/15/14/22/8/648 roots respectively; the 28 empty or singleton
cases retain zero paired gain. Unknown selection outcomes: zero.

Supplementary empirical anchor regret is 5.8675 points. Ridge and boosting
recover 7.3971% and 8.7656% of that denominator respectively. This denominator
uses noisy within-support maxima; these are not unbiased fractions of true
remaining skill and are not success criteria.

## Blind sequence and evidence

The alternative binding was validated before prediction. Both original completion
files were checked by existence only. Predictions for all 720 roots were sealed
at 12:12:46 Paris and separately authenticated at 12:13:05, before conditional
outcomes or main classifications were opened. The frozen evaluator wrote its
result at 12:13:18. Source-game containers contain terminal metadata; prediction
used the selected snapshots and no fresh conditional payoffs.

The evaluator authenticated the completed original panel and audit links, then
checked every recorded outsider support, fixed choice and score calculation.
An additional independent check of saved integer rows and the exact registered
bootstrap is recorded separately. This assessment reused original verification;
it performed no new trajectory replay.

- [Alternative result](../../outputs/alternative_ranker_128/result.json)
- [Prediction seal](../../outputs/alternative_ranker_128/predictions_seal.json)
- [Seal authentication](../../outputs/alternative_ranker_128/assessment_execution_20261018_01/seal_authentication.json)
- [Independent statistics check](../../outputs/alternative_ranker_128/assessment_execution_20261018_01/independent_statistics_check.json)
- [Original main report](../../outputs/structured_proposer_128/report.md)
- [Original final audit](../../outputs/structured_proposer_128/final_audit.json)

Alternative result SHA256:
`ca16e70028b166a767ccea4ea2d36a9295b7d9be77ee365d566a71ffc80be821`.

## Consequence for the programme

The findings add negative evidence against the tested fully trainable relational
remedy while leaving the two alternative recipes inconclusive. They do not show
that learning is impossible, that more independent data cannot help, or that
teacher improvements are all cheaply learnable. No learner is promoted and no
whole-game experiment starts automatically. The existing
[two-route gameplay design](../strategy/learned_proposer_whole_game_bridge_2026-10-18.md)
remains a draft; candidate choice, further validation, sample size and launch
require a separate decision informed by these results and their uncertainty.
