# Experiment 153 result: public-feature control failed, fully audited

2026-09-10 Paris. Complete. The fixed 135-coefficient quadratic public-feature
model failed **all four development gates**. Its validation prediction error
was 2.13% higher than the three-model 152-C mean. Its selected moves had mean
terminal WDL minus raw 035 of -0.005778, against the required +0.02.

All 26 main fits and 26 independent SVD fits completed. Every public feature,
target, fold, prediction, choice and reported metric passed the independent
audit. All 1,196 repository tests passed. No new label simulations, neural
refits, protected-panel predictions, fresh performance test, arena, promotion
or automatic successor ran. Champions remain 084 research / 035 fast.

[Frozen protocol](C:/Users/leoba/Desktop/splendor_ground0/docs/engineering/public_feature_advantage_153.md)
and [final status](C:/Users/leoba/Desktop/splendor_ground0/outputs/public_feature_advantage_153/final_status.json).
All intentionally dirty work and frozen predecessor files were preserved.

## What was tested

The model assigns a potential to each deterministic afterstate using 15 public
actor-minus-opponent quantities: score, bonuses, tokens, gold and card/reserve/
noble counts. The basis contains those quantities and their 120 quadratic
products. Advantage is candidate potential minus raw-035 baseline potential.
The original actor's perspective, pending blind-reserve count, real candidate
masks, baseline-first ties and all original positions were retained.

Training reused exactly 152-C's 3,072 states and eight-world advantage targets.
Evaluation used the existing 192-state, 64-world development validation.
Controls are zero/raw 035 and the individual predictions of all three frozen
C models; their metrics are averaged, not their predictions. Validation was
not used to select this model's penalty. It remains repeatedly reused
development evidence, not a fresh test or a whole-game strength estimate.

## Validation results

WDL is the paired terminal outcome score on the existing continuations. The
numbers below are WDL differences, not game win-rate percentage points.

| Model | Advantage MSE | Selected WDL minus 035 | Baseline choice fraction |
|---|---:|---:|---:|
| Zero / raw 035 | 0.099504683 | 0 | 100% |
| 152-C seed 149001, update 18 | 0.094547254 | +0.011474609 | 44.271% |
| 152-C seed 149002, update 39 | 0.096894223 | -0.001953125 | 45.833% |
| 152-C seed 149003, update 32 | 0.093825457 | -0.005452474 | 45.833% |
| C equal-seed mean | 0.095088978 | +0.001356337 | 45.313% |
| **153 public quadratic** | **0.097116269** | **-0.005777995** | **72.396%** |

The treatment's MSE and selected WDL were descriptively worse than each C seed.

| Treatment minus control | MSE difference | Selected WDL difference |
|---|---:|---:|
| C 149001 | +0.002569015 | -0.017252604 |
| C 149002 | +0.000222045 | -0.003824870 |
| C 149003 | +0.003290812 | -0.000325521 |
| C mean | +0.002027291 | -0.007134332 |

| Preregistered gate | Observed | Result |
|---|---:|---|
| MSE reduction versus C mean >=3% | -2.131993% | Fail |
| MSE reduction versus zero >=3% | +2.400303% | Fail |
| Selected WDL minus 035 >=+0.02 | -0.005777995 | Fail |
| Selected WDL strictly above C mean | -0.007134332 difference | Fail |

## Training-only selection

Five source-game folds contained 615, 615, 614, 614 and 614 states. Each phase/
actor cell contained 512 states and was balanced across folds to within one.
All candidates and worlds from a source stayed together. The full fold map
was sealed before feature extraction and fitting.

| Ridge penalty | Pooled out-of-fold state-mean MSE |
|---|---:|
| **0.001, selected** | **0.231698829** |
| 0.01 | 0.232903008 |
| 0.1 | 0.234978164 |
| 1 | 0.235651088 |
| 10 | 0.235736790 |

Training zero MSE was 0.235746596. Selected out-of-fold MSE improved on that by
1.7170%. The final all-training fit had MSE 0.228400754, selected WDL minus
035 -0.001302083 and baseline fraction 74.870%. Its in-sample MSE reduction
versus zero was 3.1160%.

The selected penalty is the smallest in the frozen grid; this does not establish
an optimum outside that grid. No weaker penalty or additional feature was
tested after seeing the result. Training and validation labels average eight
and 64 worlds respectively, so their raw MSE levels should not be treated as
a directly comparable train/validation generalization gap.

Every fold metric is retained in the
[selection lock](C:/Users/leoba/Desktop/splendor_ground0/outputs/public_feature_advantage_153/selection_lock.json).
All 26 coefficient vectors and all five out-of-fold prediction arrays are in
[fits.pt](C:/Users/leoba/Desktop/splendor_ground0/outputs/public_feature_advantage_153/fits.pt).

## Representation collisions

Exact equality of the 15 public quantities also implies equality of the whole
135-entry basis. Counts below exclude padding and retain every real candidate.

| Diagnostic | Training | Validation |
|---|---:|---:|
| States | 3,072 | 192 |
| Real nonbaseline candidates | 9,207 | 576 |
| Candidate pairs | 18,410 | 1,152 |
| Identical-feature pairs | 4,895 | 316 |
| States with at least one identical-feature pair | 1,867 | 122 |
| Nonbaseline candidates identical to baseline | 2,761 | 184 |
| States where every candidate has identical features | 284 | 18 |
| Baseline-only states, included in preceding row | 1 | 0 |

On validation, 63.54% of positions contain an alias, 31.94% of nonbaseline
candidates are indistinguishable from baseline, and 9.375% of positions offer
no feature distinction between any candidates. The model cannot distinguish
these moves regardless of the fitted coefficients. This quantifies a real
information loss but does not establish how much of the failure it caused.
No aliased examples or baseline-only state were removed.

## Integrity, timing and scope

The independent auditor rebuilt public effects directly from each prestate and
action. It did not use the main projection, engine step, chance resolution or
hidden card identity. It verified 3,264 states and 13,047 real actions: 479
reserved purchases, 2,515 visible purchases, 1,890 blind reservations, 3,779
visible reservations and 4,384 token takes, including 189 noble-taking actions.
All features, polynomial ordering, root perspectives, reserve counts, masks,
source/fold identities and paired target arithmetic matched exactly.

Previously audited source/continuation replays were bound by their hashes.
No unchanged continuation was resimulated. The final auditor reconstructed
all 25 CV fits and the final fit through augmented SVD least squares, separately
from the main normal equations. Maximum coefficient difference was
7.994e-15; maximum augmented normal residual was 3.553e-17. All frozen numeric
tolerances passed. Selected penalty, all choices and all four gates agreed
exactly; controls matched the completed 152 result.

| Stage | Measured time |
|---|---:|
| Full repository tests | 319.17 s, 1,196 passed |
| Preparation process, including data preflight | 23.98 s |
| Independent data preflight, within preparation | 14.72 s |
| Main process, including final audit | 18.36 s |
| Experiment run after identity check, within main process | 16.22 s |
| Final independent audit, within run | 13.42 s |

Preparation plus the main process took 42.34 seconds; implementation, focused
tests, reporting and the full repository suite are additional. The main launch
was 11:21:16 Paris and completion 11:21:32 on September 10. Both processes
exited 0. No failure artifact remains. Post-run identity, input, control,
test-evidence and final result/audit hash checks passed.

- Binding: `0627752a8399ac74484584025135282dcbff41e376122a362a8cf32218554d3f`.
- Fold map SHA: `ff6f7e4909e24896903440ae9c5c4f0e186b8e8417d927256fbfa9ceac4bf3f9`.
- Result SHA: `fa2a64c1692dd7995743112f99a898450e664c6e8f5ac2a30beea406ff423668`.
- Audit SHA: `128a026b4ce8df849d64b8673b12ac32f02c26f7a789a4262a77ced392f8e7c4`.
- Fit payload SHA: `a72793a7942824845464ba93fb5e20dc5f60dfe28a8e68b62074b3e856a55fcd`.

[Independent audit](C:/Users/leoba/Desktop/splendor_ground0/outputs/public_feature_advantage_153/final_audit.json),
[machine-readable result](C:/Users/leoba/Desktop/splendor_ground0/outputs/public_feature_advantage_153/result.json),
and [post-run verification](C:/Users/leoba/Desktop/splendor_ground0/outputs/public_feature_advantage_153/post_run_verification.json).

## Scientific decision

Close this specific compact control. The ordinary-value success of 143 did
not transfer into useful action-advantage performance under this fixed public
quadratic representation, loss and penalty grid. The result gives no basis
for promotion, a fresh test of this fitted model or an automatic scale-up.

It does not rule out richer public representations, different useful targets,
different policy-learning regimes or better planning. Nor does it prove the
existing full-state inputs are unlearnable. Capacity, representation,
regularization and optimization changed together, and the development panel
was reused. Preserve this negative result and review the programme before
choosing a substantially different hypothesis. No successor is launched here.
