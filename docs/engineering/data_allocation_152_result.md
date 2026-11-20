# Experiment 152 result: development screen failed, fully audited

2026-09-10 Paris. Complete and independently audited. Spreading the target
simulation budget across four times as many positions improved validation
prediction error by only **0.81%** versus the existing setup, below the required
3%. Selected move outcomes remained close to the baseline. Two of four primary
development criteria failed, so this recipe did not establish a material gain.

All 73,656 new continuations passed independent replay. All six complete
4,096-update refits reproduced their histories, selected/final weights and
exposure counts exactly. Data, target arithmetic, frozen controls, predictions,
statistics and final hashes passed. The process exited successfully; no
experiment processes remain. No fresh performance test or successor ran.

[Frozen protocol](C:/Users/leoba/Desktop/splendor_ground0/docs/engineering/data_allocation_152.md)
and [final status](C:/Users/leoba/Desktop/splendor_ground0/outputs/data_allocation_152/final_status.json).
All intentionally dirty work and frozen predecessors were preserved.

## Question and controls

Does spreading a fixed nominal target-simulation budget across more independent
positions improve the existing shared-card action-advantage learner?

| Arm | Training positions | Worlds per position | Actual represented continuations | Models |
|---|---:|---:|---:|---|
| A |768|32|98,304|Three frozen 151 selected models|
| B |Same 768|First 8 of existing 32|24,576|Three new full fits|
| C |B plus 2,304 =3,072|8|98,232|Three new full fits|

C versus A reallocates coverage and target precision together. C versus B
adds positions at fixed precision. B versus A lowers precision on the same
positions. The unchanged learner uses the same three seeds and 4096 updates,
with 64 uniformly sampled states per update. This holds updates fixed, not
epochs or elapsed time. Selected-checkpoint and final unique-state exposure
for every seed are reported below.

The original 192-state, 64-world validation panel is reused for checkpoint
selection and this development screen. There is no confirmatory test,
whole-game strength estimate or superhuman-performance claim. All three seeds
enter each mean; no seed, arm or checkpoint is rescued after seeing results.

## Collection and integrity

The selected panel contains 2,304 fresh source-game-disjoint positions, 384 in
each phase/actor cell. The first feasible prefix uses 4,727 source games; all
4,728 attempted games, including one unused prefetched suffix, were preserved
and independently replayed. The pre-label audit checked 301,480 source moves,
reconstructed selection and all new inputs, and passed in 381.79 seconds.
Main labels started at 2026-09-09T21:57:34.494276+00:00, after that audit passed.

Candidate support is four for 2,299 positions, three for two, two for two, and
one for one. All positions are retained. Consequently main label collection
used 73,656 paths rather than the 73,728 maximum; C's total represented
training-label budget is 72 paths (0.07324%) below A's. This follows the frozen
support/padding rule, not an outcome-dependent exclusion or budget extension.
The single baseline-only state contributes zero under the frozen loss rule.

Collection completed at 00:24:41 Paris in 27m 07s: 73,656 continuations and 2,514,298
moves, all natural (73,651 completed and five valid forced-pass deadlocks).
No cutoff, replacement, exclusion or source/budget extension. Every new
continuation and all six full refits passed their independent audits.

The separately budgeted 96-path timing probe already passed independent replay
(3,172 moves; all normal completions). Its six-state 6.2494-second timing was
startup-sensitive; actual collection took 27m 07s, superseding its crude projection.

## Development result

| Arm | Mean validation advantage MSE | Mean selected WDL minus 035 | Selected updates |
|---|---:|---:|---|
| A:768*32 |.095864088|+.000189887|16/17/24|
| B:768*8 |.099250799|-.013156467|6/10/9|
| C:3072*8 |.095088978|+.001356337|18/39/32|

| Primary development criterion | Observed | Frozen requirement | Result |
|---|---:|---:|---|
| C MSE reduction versus A |0.808551%|At least 3%|Fail|
| C MSE reduction versus zero |4.437685%|At least 3%|Pass|
| C selected WDL minus 035 |+0.001356337|At least +0.02|Fail|
| C selected WDL minus A |+0.001166450|Strictly positive|Pass|

The overall development screen failed. Zero-advantage MSE was 0.099504683.
All three seeds remain included, with equal weight and independent move choices.

| Contrast | MSE difference (negative is better) | Selected WDL difference (positive is better) |
|---|---:|---:|
| C minus A |-0.000775110|+0.001166450|
| C minus B |-0.004161821|+0.014512804|
| B minus A |+0.003386711|-0.013346354|

C passed the separate mechanism diagnostic by improving both means versus B.

On this reused development panel, adding positions recovered the deterioration
from reducing label replication, with only a small mean improvement beyond A.
The result does not establish a material move-selection gain or a scaling law.
It weighs against this specific fourfold data-allocation change being sufficient
with the frozen learner/schedule. It does not rule out all benefits of coverage,
different training regimes or value learning in general.

| Seed | A MSE / WDL gain | B MSE / WDL gain | C MSE / WDL gain |
|---|---|---|---|
|149001|.094938405 / +.010986328|.099601272 / -.018961589|.094547254 / +.011474609|
|149002|.096055543 / -.008463542|.098763527 / -.007649740|.096894223 / -.001953125|
|149003|.096598316 / -.001953125|.099387597 / -.012858073|.093825457 / -.005452474|

C beats B on both metrics for each paired seed. C versus A is mixed across
seeds: one C model has higher MSE, and one has lower WDL. The mean WDL gain
over 035 remains close to zero. These are descriptive development comparisons,
not independent statistical replications or whole-game win-rate estimates.
C changes 54.6875% of baseline choices on average, so its small mean gain is
not explained by always retaining the baseline action. Its WDL gain corresponds
to only 0.067817 percentage points of expected score under the stored
continuations; this conversion is descriptive, not a gameplay strength estimate.

## Learning behavior and exposure

All six new fits ran 4,096 updates; checkpoint selection is not early stopping.
Selected C models had sampled more unique positions than A's entire 768-position
pool, but only 31–55% of C's 3,072-position pool. All positions were sampled by
the final checkpoint. Later training worsened validation error, so simply
forcing the learner to see the full pool did not improve the selected result.

| Arm/seed | Selected update | Draws at selection | Unique positions at selection | Final training MSE | Final validation MSE |
|---|---:|---:|---:|---:|---:|
| A 149001 |16|1024|575|.000237902|.104410857|
| A 149002 |17|1088|569|.000010870|.124457352|
| A 149003 |24|1536|663|.000399335|.106139816|
| B 149001 |6|384|304|.000266730|.128771573|
| B 149002 |10|640|441|.000030538|.125430644|
| B 149003 |9|576|412|.000411065|.129425704|
| C 149001 |18|1152|962|.004467858|.136590824|
| C 149002 |39|2496|1687|.005917611|.133547872|
| C 149003 |32|2048|1479|.004812198|.155567780|

Final exposure is 262,144 draws/model, all 768 unique B positions or all 3,072
unique C positions. B/C histories, selected/final tensors and these exposure
counts passed independent full-refit reproduction. A's exposure is reconstructed
from its frozen sampler for reporting; its training audit remains the bound 151
audit. The generalization gap remains
under this exact recipe; it does not uniquely identify target noise, limited
coverage, objective, representation or optimization as its cause.

## Frozen decision and reporting limits

All four primary gates must pass: C's mean advantage MSE at least 3% below A
and zero, C's selected WDL minus 035 at least +0.02, and selected WDL strictly
above A. The C-versus-B joint MSE/WDL improvement is also reported as the
mechanism diagnostic. Neither a development pass nor failure authorizes an
automatic successor. All authorized collection, fitting and audits are complete.
Experiment 152 is closed. Champions 084 (research) and 035 (fast) remain unchanged.

Raw B/C training errors use 8-world labels, whereas A used 32-world labels;
differences between those training errors mix target noise with fitting.
Validation errors are evaluated against the same 64-world labels. Interpret
learning curves with that distinction and the selected exposure counts.

## Preparation and provenance

- Full repository suite: 1,173 passed in 324.24s. Final focused suite: 77 passed in 17.22s.
  The preparation record documents the final test-only fixture adjustment.
- Initial 960-state feature/target checks and 96-path timing replay passed.
- The full final audit passed all six refits, 160 history rows per fit and every
  selected/final tensor. All target replays, feature/lineage checks, independent
  statistics and exact frozen-A prediction reproduction passed.
- Frozen 152 binding:
  `015aef64dac1775b8e1403092d6f630514dcc7900e9b7348810be78d8499c1a8`.
- Pre-label panel audit SHA:
  `9055e9d010a9ae3a4c85b86a22dbe898807a30f866512ade0731db8ba814405c`.
- Run: 23:44:38 September 9 to 01:35:43 September 10, Paris time. Total
  **6,664.63 seconds (1h 51m 05s)**, excluding preparation/tests. Development
  completed after 3,825.40 seconds; final audit took 2,836.57 seconds (47m 17s).
- Result SHA:
  `e6b2e69de55ed20040dc5c4d2d5b6f36854d009ffab03d1cd27ad5f0b54f10bb`.
- Final audit SHA:
  `e430a94ae2c307ab532ebf0314434eec5a81dbc6fad3ece059b10c8bc52e765f`.
- Selection lock SHA:
  `0630133922a9423909eb5b7eccc46f050fc03b025d2c58d0879166cc2a1c24cd`.
- Final status SHA:
  `c4e90234ab86c46a96022101221fa6739f248577459fb0092c692fd9ce8ee079`.
- Post-run verification independently matched the frozen identity and all final
  bindings. Parent 33648/session 3026 and auditor 68288 exited successfully;
  no experiment process or failure artifact remains.
- Complete controls, labels, models, predictions, statistics and final audit
  remain hash-bound. No frozen implementation changes, fresh performance
  test, arena, promotion or automatic successor.

[Derived histories and exposure](C:/Users/leoba/Desktop/splendor_ground0/outputs/data_allocation_152/analysis/learning_histories.json)
were extracted from the hash-verified models without new inference or fitting.
The figure was rendered in a separate plotting process and visually inspected.
Exposure markers use circles/squares/triangles for seeds 149001/149002/149003.
History curves use the original float32 reductions; headline statistics use
the independently checked float64 reductions.

![Audited learning curves and selected exposure](C:/Users/leoba/Desktop/splendor_ground0/outputs/data_allocation_152/analysis/learning_curves.png)
