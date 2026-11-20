# Saved-game postmortem: improve move evaluation first

22 October 2026. Post hoc development analysis of the accepted **amended**
60-game benchmark. The original frozen attempt remains incomplete. The retained
timeout loss and interrupted attempt remain unchanged. This report authorizes no
new games, model promotion, training or larger-data collection.

## Recommendation

Prioritize **move evaluation, especially the strength and pace of the simulated
future players**, before expanding candidate coverage. This is a research
allocation decision, not proof that a particular alternative move would win.

The planner was usually capable of considering an immediately scoring action.
Its actual play nevertheless built more development cards and nobles while
falling far behind in prestige. Its simulated futures use greedy raw035 for both
players and generally allow substantially more time than these actual games did.
Improving the decisions predicted by that evaluator is therefore the most useful
next hypothesis to test.

Candidate coverage remains unresolved for longer-term plans, defensive moves and
payment variants. We have no recorded outcomes for excluded moves. The present
evidence does not justify declaring four candidates sufficient in general.

## Scope and method

- Read all **60 accepted game records**, every actual move, and all **1,496
  completed local decisions**. The other local request was the preserved timeout
  at game 39 / ply 7. Natural-ending economic comparisons use 59 games; the
  timeout's 0–0 board is not a draw or a natural final margin.
- Fix the detailed panel before computing its metrics: **18 positions from 18
  distinct games and 16 seed pairs**, one per model × local seat × phase. Phases
  are absolute zero-based plies 0–15, 16–31 and 32 onward. Within cells, use the
  minimum hash of a fixed domain/game/ply string, excluding games already chosen.
  The known benchmark result and one schema example preceded this selection;
  this is never presented as preregistered confirmation.
- Use the recorded 128 worlds only. Select on the first 64 and assess on the last
  64, reverse, then average. Retain first-candidate tie breaking. This measures
  ranking under the recorded continuation policy, not external playing strength.
- Read all **9,088 saved candidate trajectories** at the panel roots. This is
  trace inspection, not new simulation or a repeated transition audit. The
  accepted benchmark's prior replay scope and its limitations remain unchanged.
- No model inference, new games, rollout generation or training. The main
  calculation completed in 10.91 seconds at about 157 MiB final sampled RSS,
  within its 15-minute/1-GiB process limits. New artifacts are capped at 64 MiB.
  Inputs consumed by the main calculation were hashed before use and rechecked
  afterward; independent calculations use separate source and arithmetic.

The independent checker passed all 1,496 decision calculations, 60 game
summaries and 18 panel selections, including integer split-half arithmetic,
action details and the restricted reply comparisons. It rechecked 4,569 consumed
files unchanged. The separate whole-game review authenticated 6,051 inputs.
These are overlapping provenance counts, not additional games or samples.

The corrected runtime source guard ran at worker startup. This arithmetic-only
analysis does not decode training datasets or qualify a new training pipeline.
All outputs are separate from the accepted benchmark directories.

## What happened in the games

| Measure, 59 natural games | Our four-candidate player | SplendoRust |
|---|---:|---:|
| Mean final prestige | 7.03 | 15.63 |
| Mean purchased cards | 11.85 | 9.08 |
| Zero-point share of purchases | 69.4% | 36.4% |
| Tier-three purchases, total | 18 | 119 |
| Nobles claimed, total | 21 | 4 |
| First prestige gain, median own turn | 13 | 8 |
| Mean prestige after round 20 | 2.25 | 8.66 |

Our player bought more cards in 51/59 games. It was behind in 57/59 games at
round 20; the other two were tied. Natural games lasted a median of 50 plies,
with a median final deficit of nine points. This pattern appears across all three
models and both seats. It does not establish that buying any individual
zero-point card, taking a noble or reserving a card was a mistake.

Reservations need a careful interpretation. Our player reserved more in its first
eight turns, but **fewer over the whole game**: 261 versus 293. It finished with
156 unbought reserved cards versus 67, and all three slots occupied in 40/59
games versus 2/59. An unbought reserve may still provide gold or denial value;
these counts do not measure wasted moves.

## Candidate coverage versus evaluation

| Recorded decision property | Count |
|---|---:|
| At least one purchase legal | 1,008 / 1,496 |
| Purchase legal but none offered | 18 / 1,008 |
| Immediate positive prestige legal | 411 / 1,496 |
| Positive prestige legal but none offered | **0 / 411** |
| Positive prestige offered, but zero immediate prestige chosen | **185 / 411** |
| Raw035 top action changed by search | 836 / 1,496 |
| Learned fourth candidate selected | 268 / 1,496 |

At every recorded root, an offered action attained the **maximum immediate
prestige available in the full legal list**. This includes card points and a
three-point noble where the action claims one. None of these roots had a legal
action that could immediately take our score to 15. Therefore the evidence does
not show a missed immediate win, and it does not say scoring immediately is
always strategically best.

Coverage is nonetheless narrow: median offered canonical coverage was one third
of legal distinct successors. At game 8 / ply 9, all four candidates buy one of
two zero-point cards using different payments, while token taking and buying a
reserved card are absent. Those payments are legitimate distinct successors,
not duplicate-record bugs. Better multi-turn coverage may still matter.

The two 64-world halves selected the same move at **816/1,496 roots (54.5%)**,
including ties. With full-best ties excluded, agreement was 676/1,292 (52.3%).
At the 18 panel roots it was 8/18. This reveals sampling-sensitive rankings but
does not establish that simply buying more worlds would close the strength gap.
In particular, all 59 final local roots had matching half choices, but 57 had a
tied full maximum: their apparent stability largely reflects losing-value ties.

Cross-half selection improved the recorded expected-score objective over raw035's
first candidate by **2.46 percentage points per root**, or **2.50 points when
equally weighting the 59 natural games**. All three models show similar positive
conditional gains. This is useful evidence that the search does exploit its own
signal; it is not an improvement in win rate against SplendoRust. Related roots
are not independent samples, and no new confidence interval or success gate is
introduced.

## Simulated futures across the games and fixed panel

The saved simulations finish later than the actual games. Excluding timeout
games and roots with any internal deadlock ending, the average excess remaining
length was **11.81 plies early, 10.14 in the middle, and 5.12 late**, across
468/472/548 eligible roots respectively, not just the 18-position panel. Early
continuations therefore allowed about six extra rounds relative to observed
play. Actual and simulated futures use different policies and random worlds;
this comparison supports investigating pace mismatch, not a causal effect size.

Only four panel positions permit an especially clean next-reply comparison:
our chosen move draws no card and neither player has a hidden deck reserve.
At each, all 128 sampled continuations agree on raw035's next action, score hash,
legal count and selected index. SplendoRust chose a different action at **three
of these four positions**. These are four policy comparisons, not 512 independent
observations; disagreement alone does not prove which reply is better.

Two useful fixed-panel cases:

- **Game 28 / ply 25:** all three legal canonical successors are offered. Both
  halves choose taking tokens over a one-point purchase, with recorded expected
  score 82.42%. SplendoRust's next reply differs from the simulated reply on the
  same lawful observation. Missing support cannot explain this particular
  selection, but the unplayed purchase is not proved better.
- **Game 54 / ply 40:** all four candidates have zero recorded expected score.
  The selected reservation is the first-candidate tie fallback. Agreement across
  halves here is not evidence of a well-resolved strategic choice.

The complete fixed panel is below. Forecast is the selected action's expected
score **under the saved raw035 continuations**, not a calibrated probability of
beating SplendoRust. Immediate points describe the selected action. Repeated
payment variants remain separate legitimate candidates.

| Game / ply | Model | Seat | Phase | Offered / canonical | Selected action | Points | Forecast | Halves agree |
|---|---:|---:|---|---:|---|---:|---:|---|
| 25 / 8 | 161001 | 0 | early | 4 / 13 | buy reserved T1_19 | 0 | 61.33% | no |
| 30 / 22 | 161001 | 0 | middle | 4 / 4 | buy visible T1_73 | 0 | 66.02% | yes |
| 54 / 40 | 161001 | 0 | late | 4 / 16 | reserve visible T2_67 | 0 | 0.00% | yes, tie |
| 36 / 1 | 161001 | 1 | early | 4 / 29 | take tokens | 0 | 54.30% | no |
| 7 / 19 | 161001 | 1 | middle | 4 / 64 | buy visible T1_56 | 0 | 37.89% | no |
| 19 / 47 | 161001 | 1 | late | 4 / 8 | buy reserved T2_47 | 2 | 92.19% | yes |
| 57 / 14 | 161002 | 0 | early | 4 / 24 | buy reserved T1_75 | 0 | 46.09% | yes |
| 2 / 18 | 161002 | 0 | middle | 4 / 12 | take tokens | 0 | 75.78% | no |
| 26 / 44 | 161002 | 0 | late | 4 / 38 | buy visible T1_20 | 0 | 18.75% | no |
| 8 / 9 | 161002 | 1 | early | 4 / 10 | buy visible T1_55 | 0 | 51.56% | no |
| 51 / 27 | 161002 | 1 | middle | 4 / 28 | buy visible T1_19 | 0 | 72.66% | yes |
| 56 / 49 | 161002 | 1 | late | 4 / 21 | reserve visible T2_64 | 0 | 7.42% | no |
| 58 / 4 | 161003 | 0 | early | 4 / 27 | reserve visible T1_80 | 0 | 75.78% | yes |
| 22 / 24 | 161003 | 0 | middle | 4 / 5 | take tokens | 0 | 44.53% | yes |
| 46 / 36 | 161003 | 0 | late | 4 / 13 | buy visible T1_78 | 0 | 12.50% | no |
| 11 / 3 | 161003 | 1 | early | 4 / 26 | reserve visible T2_82 | 0 | 72.66% | no |
| 28 / 25 | 161003 | 1 | middle | 3 / 3 | take tokens | 0 | 82.42% | yes |
| 59 / 41 | 161003 | 1 | late | 4 / 34 | buy visible T2_09 | 1 | 3.12% | no |

## Next controlled research step

Prepare one **fixed-candidate evaluator comparison**. Retain the current four
proposals, lawful observations, all three models and matched random worlds;
change only how credible future play or continuation value is assessed. The
first candidate mechanism should address stronger replies and the faster
prestige race. Price and qualify its complete decision cost before selecting an
implementation: running full 120-second SplendoRust searches inside every
rollout is not assumed affordable, and its neural network is not automatically
a validated drop-in value function.

Use held-out assessment worlds to separate selection from assessment, and an
assessment policy or outcome criterion independent of the candidate evaluator.
Separate worlds alone do not remove a shared continuation-policy or value-model
bias. This panel can be used for development; it is no longer fresh evidence.
A promising intervention must then face fresh paired whole-game tests against
the frozen current player, with SplendoRust as the external check, at a declared
clock. Do not claim progress from a local proxy alone. Candidate widening becomes
the next competing intervention if improving evaluation fails to change useful
decisions or wider support demonstrates independently assessed value.

Keep larger-data collection deferred until there is a credible stronger target
and a falsifiable plan for learning it. Future collection, fitting and fresh
evaluation must satisfy the corrected-rules policy end to end; historical targets
cannot be relabelled. No generally migrated training entry point is qualified by
this postmortem.

## Rules and evidence limits

There were five internal forced-pass-deadlock endings among 725,248 recorded
candidate continuations, none in the panel. They remain internal conventions,
not official completed-game targets; they do not account for a broad numerical
pattern here. No actual natural game contained a pass. None of our selected blind
reserves returned tokens, but that narrow observation does not clear all known
pre-reveal return/noble restrictions. These remain separate documented follow-ups.

The benchmark establishes the strength of the combined SplendoRust player under
these settings, not the separate effect of its search or neural network. The
postmortem cannot rank excluded actions, prove that an observed losing move was
avoidable, or estimate the causal benefit of a new evaluator.

## Reproducible artifacts

- [Bounded plan and fixed panel](../../outputs/matched_postmortem_20261022_01/plan.json)
- [All-root summary](../../outputs/matched_postmortem_20261022_01/summary.json)
- [Detailed 18-position decisions and traces](../../outputs/matched_postmortem_20261022_01/panel_analysis.json)
- [All-game independent findings](../../outputs/matched_postmortem_20261022_01/independent/whole_game_findings_v1.md)
- [Independent notes on all 18 positions](../../outputs/matched_postmortem_20261022_01/independent/panel_interpretation_v1.md)
- [Independent arithmetic check](../../outputs/matched_postmortem_20261022_01/checks/independent_check.json)
- [Duration and interpretation arithmetic](../../outputs/matched_postmortem_20261022_01/interpretation_arithmetic.json)
- [Execution, source recheck and output bindings](../../outputs/matched_postmortem_20261022_01/completion.json)

The accepted benchmark files and player implementations were not modified.
