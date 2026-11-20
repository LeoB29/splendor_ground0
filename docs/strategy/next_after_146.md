# Recommendation after 146: controlled training-exposure test

Authorization update: the owner subsequently approved this proposal and the
runtime estimate with "ok great, can you do it". Exact experiment 147 design:
`docs/engineering/value_training_exposure_147.md`. Follow AGENT_CURRENT for
execution state; the original proposal and status below are historical.

Completion update: 147 finished both full schedules and the independent audit.
The new trainable model selected update 1152, but its fresh early/middle gain
was .6977% over public (3% required); gains over frozen-long and trainable-short
were not established. Both fits reached 99.959% unique-position coverage and
the final trainable model overfit substantially. No protected arena followed.
See `docs/engineering/value_training_exposure_147_result.md`. This closes the
proposed exposure check; no further schedule or successor is authorized.

2026-09-08. Recommendation only. No training, new games, search, arena or
successor has been launched or authorized by this review. Preserve all dirty
and frozen work; champions remain 084 research / 035 fast.

## Single next experiment

Run one bounded test of the training schedule on the existing strong-035
residual-value recipe. Keep model architecture, initialization, development
data, batch stream, optimizer, loss, validation metric and checkpoint-selection
threshold fixed. Remove the early termination at four stale validations and
give both frozen-encoder and trainable-encoder fits a fixed longer budget.

Suggested budget: 16,384 updates at batch 128, or 2,097,152 position draws
(about eight corpus-equivalents). Sampling remains game-uniform then
position-uniform with replacement. This does NOT guarantee eight visits to
each position; report actual unique game/position coverage.

Retain 146's .001 required validation improvement and 128-update validation
cadence. This keeps the causal question on training exposure and opportunity
for later improvement, without simultaneously relaxing model selection.
Step zero stays eligible. Do not force a late checkpoint into evaluation if
the unchanged selection rule prefers an early checkpoint. No learning-rate
grid, new architecture, new training games or altered target is included.

Keep both 146 short-run selected checkpoints as frozen references. Reuse
training seed 146010 and require deterministic reproduction of each 146 fit's
first 640 updates and selected update-128 tensors before treating the longer
run as valid. Thus the comparison has four controlled cells: frozen versus
trainable encoder, each under the original short schedule versus the fixed
long schedule. Only the two long fits are new; original 146 remains closed.

Lock every candidate using development data only, then use one fresh protected
prediction set (suggested 512 games, disjoint seed namespace). No 143 or 146
protected outcomes enter fitting or checkpoint selection. Evaluate all four
cells on the same new games, plus public/neutral references. Prespecify the
long-minus-short contrasts and the long-trainable-minus-long-frozen contrast;
resample whole games with paired models.

The candidate would still need a material early/middle gain over public and
the matched frozen learner, with overall/late and cost guardrails. Keep 146's
3% versus public and 1% versus frozen targets; require established improvement
over the short trainable reference as well. Freeze the exact decision rule,
sample sizes, seed schedules, uncertainty method and any conditional gameplay
stage before new outcomes. A conditional fresh .4-second arena against public
search is appropriate only after this gate passes; it would not authorize
promotion or prove superiority to raw 035 or 084.

This is a proposed design, not a frozen execution protocol. Any later approval
should be turned into a fully specified, bounded protocol before launch.

## Why this is the highest-value next check

146's selected model used 16,384 draws, 15,855 unique positions (6.08% of the
260,904 available), spread across 4,011 games. Both fits stopped at update 640
and selected 128. Broad game coverage is therefore already present, but
substantial within-game training exposure is not. The actual fits took about
3.1 and 3.8 seconds, so a bounded schedule test uses existing data at relatively
low training cost. Overall execution and audit costs still need preflight.

This is a limitation of the experiment, not evidence that early stopping was
wrong. The validation trajectory did not establish a large later gain. Simply
substituting update 384 after looking at the result would be an invalid rescue
and its validation change was far smaller than the practical target. Preserve
146's result and test the schedule prospectively on a new holdout.

142's eight-epoch overfit cautions against assuming more training helps; it used
205 training games and a different bootstrap. 143 mostly varied data scale
while freezing the encoder. 146 tested trainability with short selected exposure.
None is this controlled schedule test on the 4,096-game strong-035 corpus.

Repeated-return targets alone already failed in 132's compact frozen-head
setting. More search helped the gen11 branch in 113, but does not establish
compute headroom for the current 035/public-value planner. 144/145 did not
establish the relevant gameplay gains. Another larger arena or another
architecture change would leave this cheap training-exposure ambiguity open.

## Interpretation and stopping

- If the long trainable arm improves materially over its short reference and
  the long frozen control, additional optimization exposure supports value
  representation learning in this recipe. Gameplay remains the next necessary
  test, not a consequence already established by better predictions.
- If both long fits improve similarly, exposure may help value fitting without
  supporting an encoder-trainability advantage. Do not claim the latter.
- If selection stays early, or fresh gains remain negligible, stop this
  unchanged recipe and deprioritize insufficient update count as its practical
  remedy. That does not rule out different optimization or targets universally.
  The next strategic review should revisit target/state distribution and
  decision-relevant search signal rather than automatically extending training.

The long-term objective remains a planner/learner that improves actual playing
strength across iterations. This one bounded check resolves a specific training
ambiguity; it is not a claim that prediction MSE alone defines progress toward
superhuman play. No automatic second schedule, extra samples or promotion.

Sources: `docs/engineering/value_representation_146_result.md`,
`outputs/value_representation_146/trainable_training.json`,
`outputs/value_representation_146/training_dose_diagnostic.json`,
`src/splendor_ai/alphazero/run_value_representation.py`,
`docs/engineering/strong_policy_pilot_142_result.md`,
`docs/engineering/alphazero_search_headroom_113_result.md`,
`docs/engineering/alphazero_capacity_pilot_116_result.md`, and
`docs/engineering/alphazero_deadlock_consistent_repeated_return_value_ab_132_result.md`.
