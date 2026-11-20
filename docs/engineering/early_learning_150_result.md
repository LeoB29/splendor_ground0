# Experiment 150: early prediction gains, development screen failed

2026-09-09. **Complete and independently audited. Neither representation passed
the development screen.** The bounded pipeline ran 19:48:51–19:50:41 Europe/Paris,
109.83 seconds excluding preparation and tests. Parent 8844/session 76222 and
auditor 14564 exited successfully; no experiment process remains running.

All six models selected a previously unmeasured checkpoint at updates 12–77.
The extra checks therefore changed the observed learning picture, while
preserving every original optimization trajectory. Earlier selections reduced
validation prediction error and improved average selected outcomes relative
to the original selections, but neither delivered the required +.02 WDL gain
over 035. These are development observations on reused validation outcomes,
not independent evidence of stronger play. Experiment 149 remains closed and unchanged.

## Paired development results

Same 192 validation states, one per source game; four fixed candidates and the
existing 64-world outcomes under exact greedy 035 continuations. Each model
chooses independently; average all three seeds equally within a representation.
Lower MSE is better. The zero predictor keeps 035 and has MSE .099504683.

| Representation | Original-grid MSE | Dense-grid MSE | Reduction vs original | Reduction vs zero |
|---|---:|---:|---:|---:|
| Control | .099917092 | .096169514 | 3.751% | 3.352% |
| Afterstate | .099021156 | .096116394 | 2.933% | 3.405% |

| Representation | Original selected WDL minus 035 | Dense selected WDL minus 035 | Dense expected-score change vs 035 |
|---|---:|---:|---:|
| Control | -.018174913 | +.001898872 | +.095 percentage points |
| Afterstate | -.017415365 | -.004638672 | -.232 percentage points |

The required move gain was +.02 WDL, or +1 percentage point expected score under
these fixed continuations. Both missed it. These means have no confirmatory
confidence intervals: validation also selected the checkpoints. Do not call
the control a proven improvement or the afterstate a statistically established
regression from this diagnostic. These are not arena win rates.

| Development criterion | Control | Afterstate |
|---|---|---|
| At least two seeds select updates 1–127 | Pass: 3/3 | Pass: 3/3 |
| MSE reduction vs original at least 3% | Pass | Fail: 2.933% |
| MSE reduction vs zero at least 3% | Pass | Pass |
| Selected WDL gain over 035 at least +.02 | Fail | Fail |
| Selected WDL improves over original selection | Pass | Pass |
| All criteria | **Fail** | **Fail** |

The afterstate narrowly missed the MSE cutoff, but the decision is not solely
an artifact of that boundary: its average selected move gain was negative.
The control met both MSE rules but missed the move requirement. Prediction fit
alone therefore cannot justify a fresh strength test under the frozen screen.

## What happened in the early interval

| Model | Original selected update | Dense selected update | Dense validation MSE | Dense WDL minus 035 |
|---|---:|---:|---:|---:|
| Control 149001 |0|77|.093792408|+.014973958|
| Control 149002 |0|12|.097730694|+.000488281|
| Control 149003 |0|20|.096985440|-.009765625|
| Afterstate 149001 |2560|30|.095150011|-.001464844|
| Afterstate 149002 |0|13|.097369950|-.010416667|
| Afterstate 149003 |0|16|.095829221|-.002034505|

These selections correspond to 1.0–6.4 average training-set passes, before the
original first trained check at 10.7 passes. They are the best observed points
on the expanded grid, not a claim about an unmeasured continuous optimum.
Every model still completed all 4096 updates. The later histories and final
weights reproduced 149 exactly; added evaluation consumed no training RNG.

Dense selection changed 59.03% of control choices and 50.69% of afterstate choices
relative to the respective original selections, averaged across seeds. Both
dense arms kept 035 on 40.45% of choices. This records substantial intervention
on validation states; it does not establish agreement between the two models
or causal strength effects in a deployed game.

![Early validation curves and conditional move outcomes](C:/Users/leoba/Desktop/splendor_ground0/outputs/early_learning_150/analysis/early_learning_verified.png)

The curves show why the original statement that five models selected initial
weights needs its checkpoint-grid qualification. They do not overturn 149's
valid fresh-test failure. Selection among more checkpoints can fit validation
noise; neither these lower minima nor the selected-move means are fresh tests.

## Verification and stopping

All 38 focused 150/149 tests passed in 13.82s. The full suite passed 1134 tests in
307.50s. Preflight validated both development panels and provenance before any
real-data fit. No preparation failure, target change or post-freeze code edit.

The six new fits took 37.90–38.08s each concurrently; main fit/analysis stage
took 40.93s. Independent audit took 66.50s. It separately aggregated all 960
reused training/validation target records, reproduced every prediction,
descriptive statistic and screen condition, and reproduced all six full fits
using a separate training loop. Each fit matched all 160 history rows, both
selected tensors and final tensors. All 33 original coarse history rows and
original selected/final tensors also matched 149 exactly.

All 977 bound artifacts from 149, its 281 dependencies and 4 inputs, plus 280
dependencies and 5 inputs from 148 remained unchanged. No source games or continuations were generated
or replayed. No protected test panel or outcomes entered this diagnostic.

`final_status.json` records complete=true, development_only=true,
development_screen_passed=false, new_continuations=0, fresh_test=false,
arena=false, promotion=false and automatic_successor=false. Stop here. The
missed-early-window remedy failed the declared development threshold for this
recipe; this does not rule out useful early learning, other optimization,
different representations or afterstate learning generally. No best-seed,
checkpoint, sample-size or threshold rescue follows. Champions remain 084/035.

Manifest binding:
`04c878af0d94292c1b4b834d4a19f63760d6f7ffbf29fc9934e0aa89026308a3`.
Manifest file SHA-256:
`8a1fe332890fcc939c69ca08feaf40a27eb9b739562a4b89af4a2f4f60484f57`.
Result SHA-256:
`785ce2d054cecc3affaed40c58bbd8776e3b3bc29bb30691627d1eae8d3aeeea`.
Audit SHA-256:
`26a3cff7145f60715556515a5830efd60ad916da371975e6bdedf740697dc0ad`.

[Protocol](C:/Users/leoba/Desktop/splendor_ground0/docs/engineering/early_learning_150.md),
[preparation](C:/Users/leoba/Desktop/splendor_ground0/docs/engineering/early_learning_150_preparation.md),
[result JSON](C:/Users/leoba/Desktop/splendor_ground0/outputs/early_learning_150/result.json),
[audit](C:/Users/leoba/Desktop/splendor_ground0/outputs/early_learning_150/audit.json),
[extracted histories](C:/Users/leoba/Desktop/splendor_ground0/outputs/early_learning_150/analysis/training_histories.json).
