# Experiment 149: afterstate advantage-learning pilot

2026-09-09. **Complete and independently audited; advance gate failed.**
The corrected run started 2026-09-08 at 22:40:10 and finished 2026-09-09 at 01:21:17
Europe/Paris. Pipeline duration 2 h 41 m 07 s, excluding preparation. No arena,
promotion or successor. Parent 1912/session 73622 and auditor 40416 exited cleanly.

The fixed advance rule failed every statistical improvement requirement. The
afterstate models selected worse moves than raw 035 under the specified greedy 035
continuations, and their held-out advantage prediction MSE was worse than the
matched control. Their small selection gain over the control was not established.
This rejects advancement of this particular recipe; it does not reject every
afterstate architecture or demonstrate a whole-game strength difference.

## Fresh conditional move results

192 fresh source states, one per game, balanced across phase/actor seat; four
candidate actions and 128 independent worlds per state. Each model's choices
were locked before test continuations. Average the three paired model-seed
outcomes within each source state; bootstrap whole source games, 50,000 replicates.

| Contrast | Mean WDL difference | Paired CI95 | Result |
|---|---:|---:|---|
| Afterstate minus control | +0.007094 | [-0.006510,+0.020061] | Required gain not established |
| Afterstate minus raw 035 | -0.042169 | [-0.070055,-0.016845] | Worse under fixed continuations |
| Control minus raw 035 | -0.049262 | [-0.076199,-0.025011] | Worse under fixed continuations |

Afterstate versus raw 035 corresponds to -2.108 percentage points of expected
terminal score, 95% CI [-3.503, -0.842], **under these conditional continuations**.
It is not an arena win rate or a prediction of the same full-game difference.
Afterstate versus control is +0.355 percentage points of expected terminal score,
95% CI [-0.326, +1.003].

Afterstate advantage MSE 0.088044 versus control 0.084892: difference +0.003152,
95% CI [+0.001061, +0.005277], a 3.712% increase in error. Lower is better. This is
against independently sampled test outcome estimates, not optimal-play truth.

Both primary selection means had to reach +0.02 WDL with positive CI lower bounds;
neither passed. Afterstate-control CI half-width 0.013285 meets the <=0.02 precision
rule, but afterstate-baseline half-width 0.026605 does not. The latter interval
nevertheless establishes a negative conditional effect: failure of the precision
criterion must not be mistaken for absence of evidence of harm. The MSE criterion
also fails, with the difference established in the wrong direction. The +0.02
afterstate-control target is barely inside its interval; do not claim equivalence
or exclusion of that effect.

| Paired seed | Afterstate minus 035 | Control minus 035 | Afterstate minus control |
|---|---:|---:|---:|
|149001| -0.024536 | -0.028076 | +0.003540 |
|149002| -0.058472 | -0.061808 | +0.003337 |
|149003| -0.043498 | -0.057902 | +0.014404 |

These are robustness diagnostics, not three extra independent game samples.
No seed or checkpoint was selected using the test. No agreement with a noisy
empirical best action was treated as an oracle metric.

## What training did

All six fresh 320001-parameter networks completed 4096 updates, using the same
paired initialization, batches, optimizer and validation rule. Validation
preferred initial weights for five models; only afterstate 149001 selected a
trained checkpoint, at update 2560. Initial checkpoints were eligible in the
frozen protocol. Selected update 0 does not mean training was skipped.

| Model | Selected update | Final training MSE | Final validation MSE |
|---|---:|---:|---:|
| Control 149001 |0|0.001699|0.106769|
| Afterstate 149001 |2560|0.000208|0.099648|
| Control 149002 |0|0.000707|0.106555|
| Afterstate 149002 |0|0.001955|0.105278|
| Control 149003 |0|0.000882|0.104418|
| Afterstate 149003 |0|0.000072|0.115289|

Initial validation MSEs were 0.098307--0.101182. The selected afterstate 149001
checkpoint had validation MSE 0.096426 and training MSE 0.001131. Training error
fell sharply while validation generally deteriorated. Explicit deterministic
consequences did not reliably solve the generalization problem in this fresh,
compact, flat-network recipe. Because five selected models are untrained, do
not describe the final comparison as a contest between six successfully learned
strategies or infer a universal limit on afterstate learning.

The validation grid was update 0 and multiples of 128. With 64 source states per
update and 768 training states, the first evaluated trained checkpoint followed
8192 state draws, about 10.7 average passes through the dataset. An earlier useful
learning window remains unmeasured. The evidence concerns the frozen recipe and
checkpoint grid; no extra checkpoint selection or test-based rescue was performed.

![Independently reproduced training and validation curves](C:/Users/leoba/Desktop/splendor_ground0/outputs/afterstate_advantage_149/analysis/training_curves_verified.png)

## Descriptive target uncertainty

After the frozen test, estimate each action-advantage mean's Monte Carlo variance
as the sample variance of paired world-level WDL differences divided by the world
count. Average across nonbaseline candidates/source states. This descriptive
calculation changes no model, selection or advance criterion.

| Split | Mean estimator variance | Zero-advantage predictor MSE | Estimated reduction from pairing |
|---|---:|---:|---:|
| Training |0.036729|0.101540|6.59%|
| Validation |0.019048|0.099505|7.50%|
| Test |0.009565|0.085915|7.12%|

The comparison for pairing is the independent-world variance estimated from the
same baseline/candidate marginal variances. It is an estimate, not an additional
uncoupled simulation arm. Sharing the initial world offered only modest variance
reduction here. The estimator-variance/zero-predictor-MSE ratios are approximately
36.2%, 19.1%, 11.1% across the three splits. These ratios concern finite Monte Carlo
target estimation and include no claim that afterstates remove future variance.
They do not prove that adding worlds alone would fix generalization. Artifact:
[target_uncertainty.json](C:/Users/leoba/Desktop/splendor_ground0/outputs/afterstate_advantage_149/analysis/target_uncertainty.json).

## Data and integrity accounting

| Split | Source states | Source games examined | Continuations | Moves |
|---|---:|---:|---:|---:|
| Training |768|1608|98304|3355494|
| Validation |192|425|49152|1672599|
| Fresh test |192|458|98304|3379123|
| Total |1152|2491|245760|8407216|

245,753 continuations reached normal completion; 7 ended in valid forced-two-PASS
deadlock under the established scoring tie-break. There were no cutoffs or
invalid targets. All states had four distinct canonical candidates. Root
RESERVE_DECK was included 664 times across the 1152 panels, accounting for 35744
continuations. No held blind reserve was allowed at the source snapshot; blind
reserves remained legal at the root and throughout continuation play.

The afterstate/chance wrapper is isolated from the frozen engine. Tests and final
preflight cover composition, payments/returns, gold, public/private draws, nobles,
terminal handling, exhausted decks and hidden-world feature invariance. Full
suite 1122 passed; focused 26 passed. Final preflight 426 composition comparisons
and 96 natural timing continuations passed. No learned dynamics model was used.

The first main launch stopped before targets/fits/test games because cyclic
assignment of source games to cells was too restrictive. A read-only audit
proved sufficient eligible data existed. The corrected run uses deterministic
balanced matching, with complete development panels locked in preflight. The
failed run and two earlier preparation attempts are preserved with manifests
and code snapshots; see the [preparation history](C:/Users/leoba/Desktop/splendor_ground0/docs/engineering/afterstate_advantage_149_preparation.md) and the protocol.
No learning/test outcome informed those engineering repairs. All scientific
quotas, target budgets, learning recipes and advance gates stayed fixed.

## Audit and closeout

**Passed:** separate-process verification of all source selections/features,
locks and predictions; all 245760 world/action continuations and 8407216 moves;
and independent recomputation of statistics/gates. All six full 4096-update fits,
histories, selected weights and final tensors were exactly reproduced. All 281
current and 280 prior frozen dependencies matched. No failure artifact exists in
the corrected run. `final_status.json` records complete=true and
advance_gate_passed=false, with no arena, promotion or automatic successor.

Collection plus fits took 5140.29 s (85 m 40 s), including fresh test source generation.
The audit took 4525.00 s (75 m 25 s); complete pipeline 9667.31 s (2 h 41 m 07 s). No experiment
jobs remain running. All 458 fresh test source games (seeds 1490000000--1490000457)
and their selected 192-state evaluation are now consumed holdout material.

Main manifest binding:
`66cf58f8dc8b49ee9caf8e804f7af33c9213239a259eadc08fee016496f00407`.
Test result SHA-256:
`2d0738324ea2294d416c26fe8d3cc156c3dfe28fc36fcad8f2b54ebf510d821a`.
Audit SHA-256:
`8b67ea968a3b09ec226a681ccc212dd3496b00b5a03d0ed787942d01e1ebedbc`.
Manifest file SHA-256:
`05fa10e2ea15b92b3670f6ba5876fd22cf6fe63979c2fc20c770e5e69cc7d188`.
Artifacts: [experiment output](C:/Users/leoba/Desktop/splendor_ground0/outputs/afterstate_advantage_149/); protocol:
[afterstate_advantage_149.md](C:/Users/leoba/Desktop/splendor_ground0/docs/engineering/afterstate_advantage_149.md).

No champion change: 084 research / 035 fast. No gameplay arena, new training recipe,
sample expansion, selected-seed rescue or automatic successor follows this result.
