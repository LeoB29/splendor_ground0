# Next proposal after 145: independently trainable value features

Owner approval update, 2026-09-08: the subsequent "ok do it," authorizes this
proposal as experiment 146. The exact design is recorded in
`docs/engineering/value_representation_146.md`; see AGENT_CURRENT for execution
status. The recommendation and its original authorization status below are
preserved as historical context.

Completion update: 146 failed both primary prediction requirements. Trainable
early/middle error improved .1473% versus public (3% required), without an
established benefit over the frozen-encoder control. All 512 fresh games and
independent checks passed; no protected arena followed. See
`docs/engineering/value_representation_146_result.md` for results and important
limits from the short, validation-selected training dose. No successor approved.

2026-09-08. Recommendation only; no training, new games, arena, promotion or
successor has been launched or authorized. Read alongside the 143--145 results.

## Recommendation

Run one bounded value-representation pilot on the strong-035 branch. Preserve
035's policy exactly. Give the evaluator its own trainable board encoder and
train it to improve on the frozen public score/resource baseline. Use an
otherwise comparable frozen-encoder learner as the representation control.
This is a change to what value learning can learn, not another increase in
the number of games testing the unchanged 145 bot.

A useful concrete implementation is a fixed public baseline plus a learned
residual, with the residual output initially zero. Both representation arms
then start from the same prediction. One arm freezes a copied 035 state encoder;
the other trains its separate copy along with the value head. Neither modifies
the original policy model. Keep inputs, initialization, data, loss, maximum
training budget and validation-selection rule matched where applicable.
This isolates the trainability of value features within that recipe; it does
not isolate an abstract representation-information effect from optimization.

Reuse only the 143 development training/validation partitions for fitting and
selection. The existing test sets are consumed. Evaluate locked candidates on
fresh held-out games, including a prespecified early/middle-game endpoint
(turns <40), an overall error/late-game nonregression check, and inference cost.
The exact sample size, architecture, regularization, practical margins and
selection rules must be frozen before any new protected outcomes. Keep neutral,
the frozen public evaluator and the comparable frozen-encoder learner as
references. Do not select early/middle subgroups retrospectively.

Only if the new value representation produces a material, independently tested
gain over the frozen references should the same bounded experiment proceed to
a fresh seat-paired gameplay comparison against public-value search, with the
same .4-second allowance. Fix that schedule, primary endpoint, precision and
success rule before the protected pipeline starts. Prediction improvement is
a gate to testing, not a claim of stronger play. Stop if the prediction gate
fails. Even a gameplay win in this duel would not itself establish superiority
over 035 or 084, and would not authorize a promotion or learning cycle.

## Evidence and limits

143 trained only a 33,025-parameter value head on a frozen 035 encoder. Moving
from 256 to 4,096 training games improved test error modestly: the large head
reduced neutral-predictor MSE by 1.73%, while basic public features achieved
15.59%. This supports testing the value-learning interface; it does not prove
that the encoder discarded information or that a separate network will work.

The public reference's descriptive early/middle/late MSEs were
.979266/.909708/.635634. Most predictive gain arrived late. This motivates
evaluating useful information before the endgame explicitly in a new test;
it does not prove that early positions permit a large predictable advantage.

144 did not establish superiority over raw 035; 145 did not establish a public
versus neutral value benefit at matched .4-second allowances. Both intervals
still permit useful effects. Do not pool the two endpoints or call them proof
of equivalence. An unchanged larger arena would primarily refine a small
uncertain effect and has lower programme value than testing a substantive
remaining learning hypothesis.

Higher search compute has worked on a different branch (113), so depth is still
a live alternative. The mean maximum depth of about 2.3 in 145 describes a
shallow regime but does not establish why the public value failed to show a
gain. The present recommendation chooses value trainability because of the
143 comparison; it does not claim that search allocation/depth is already solved.

Earlier 112/116/132 tests also limit interpretation. This is not the first
separate value path or trainable residual tower anywhere in the programme.
Those trials used other checkpoints, targets, data scales and training recipes.
Their lack of demonstrated gameplay gains is why a new offline success must
still face a prospective gameplay test. This pilot concerns the currently
strong 035 policy and the specific frozen-feature restriction in 142/143.

## Read-only post-145 trace diagnostic

All 800 existing 145 records were scanned, without new search or games. For
each saved root, count a choice as off the top prior only when the selected
canonical edge's `network_prior` is strictly below the maximum root prior.
Ties at the maximum are not counted as deviations. Aggregate across decisions,
with descriptive phases <20, 20--39 and >=40 turns.

| Diagnostic | Public | Neutral |
|---|---:|---:|
| Decisions | 25,479 | 25,479 |
| Choice below maximum prior | 5.4712% | .5730% |
| Early choice below maximum prior | 6.45% | 0% |
| Middle choice below maximum prior | 4.5125% | 0% |
| Late choice below maximum prior | 5.4542% | 1.5402% |
| Mean largest prior probability | .613352 | .610547 |
| Fraction with largest prior >.95 | 20.4757% | 20.2637% |
| Mean visited canonical classes | 7.2149 | 5.3634 |
| Mean legal canonical classes | 20.9300 | 20.8138 |

This confirms limited behavioral intervention rather than zero intervention.
It does not prove that priors are too sharp, that more disagreement would help,
or that public and neutral choose differently at these rates on the SAME states:
the arms follow different trajectories. These are post-hoc, decision-weighted
descriptions, not new strength tests, causal estimates or passing criteria.

Sources: `docs/engineering/value_generalization_scale_143_result.md`,
`docs/engineering/public_value_search_144_result.md`,
`docs/engineering/public_value_ablation_145_result.md`,
`src/splendor_ai/training/model.py` (LegalActionAttentionScorerModel),
`src/splendor_ai/encoding/observation.py`, and
`outputs/public_value_ablation_145/games/game_0000.json` through `game_0799.json`.
