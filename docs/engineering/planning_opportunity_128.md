# Planning-opportunity-weighted teaching

Owner authorized on2026-10-09 after the completed fixed-representation test.
This protocol is fixed before production fits or new-model diagnostics. One
CPU saved-data development experiment is authorized, including preparation,
independent optimizer reproduction, supported planning assessment and closure.
No new source games, rollouts, architecture, sweep, arena, promotion, fresh
confirmation or automatic successor. Preserve all dirty and frozen work.

## Hypothesis and matched control

The current soft outsider loss gives every row equal weight even when its
extra candidates cannot improve the retained first three under recorded teacher
estimates. About half the old and added multiple-outsider training rows have
this property. Does prioritizing empirical opportunities improve the fourth
proposal's contribution to actual supported planning?

This tests allocation of finite learning effort. When true values are known,
maximizing outsider value already maximizes the augmented shortlist's value.
The existing objective is not mathematically inconsistent with that optimum.
Noisy finite positive gaps can amplify noise, reduce effective coverage and
allow drift on unweighted positions. A positive result does not uniquely
establish the cause of the earlier failures.

Use the original three161 anchors and the completed recovery01 BROAD models
as matched controls. Train the full original policy, with only value-head
parameters frozen. Do not continue the failed head-only models. Data, seed,
architecture, parameter mask, optimizer, target probabilities, batches, update
count and final-only selection match BROAD. Only the per-row loss weight changes.

## Fixed training rule

For each saved row, let C be the maximum recorded WDL sum among its actual
first up-to-three candidates; let O be the maximum among the remaining
recorded outsiders. Define integer opportunity units

`u = max(0, O - C)` when outsiders exist, otherwise `u = 0`.

The corresponding expected-score opportunity is u/256 because each candidate
has128 recorded terminal worlds. This uses only old TRAIN selection outcomes.
No independent assessment or validation outcome enters weights or fitting.

Reconstruct the exact saved R1 BROAD schedule, retaining the narrow partitions
that determine its paired length bucketing. For each fit and each pool p
(old and added separately), calculate once before optimization:

`N_p = prescribed number of exposures from p`

`S_p = sum(exposure_count_for_row * u_row)`

`weight_row = u_row * N_p / S_p`.

Require positive S in both pools; otherwise stop before fitting. Each pool's
exposure-weighted mean weight is exactly one, preserving the intended50/50
mixture. Retain zero-weight rows, all batch slots and all AdamW steps. Mean the
weighted per-row losses over the original whole batch length, not over positive
rows or their summed weights. Singleton outsiders retain their mathematical
opportunity but their unchanged outsider CE is zero; report this separately.

The target is UNCHANGED `softmax(q_outsider/.05)`, q=(128+sumWDL)/256. Do not
clip target values to opportunity, filter outsiders, change temperature, add an
anchor penalty or resample positions. This is one weight intervention.

Use9,461 old TRAIN rows from159 source pairs and all720 additional positions
from distinct games. Each fit has1,784 updates and114,044 exposures,57,022 from
each pool, with1,772 batches of64 and12 of53. Exact row order and exposure
receipts must equal the saved BROAD control. All720 consumed panels stay retired
from future evaluation. Original2,489 VALIDATION rows remain excluded.

Use the original three seeds, fresh AdamW lr1e-4, weight decay1e-4, gradient
clip1, float32 CPU, deterministic operations and one Torch thread. Freeze only
value_head parameters, matching BROAD; shared state features can change.
Always use update1,784. Initial/final diagnostics are not a selection rule.
No alternative checkpoint, seed, weight transformation or coefficient rescue.

Record raw integer units, exposure counts, normalizing sums, normalized weights,
zero-row/exposure shares, maximum weight, usable multiple-outsider mass and
weight concentration. Report effective sample counts separately for exposures,
distinct rows, and independent source units (old whole pairs / added games).
These are concentration diagnostics, not statistical sample sizes for inference.

Independently reconstruct weights, canonical groups, schedule and every optimizer
update; verify final tensors, diagnostics, frozen value head and full-policy
trainable membership. Seal all three models only after passing reproduction.

## Relevant regret and development assessment

For a model-proposed recorded outsider j, define marginal shortlist regret:

`r = max(C, O) - max(C, sumWDL_j)`, in integer units; score regret is r/256.

When no outsider exists, regret is zero. This measures the recorded quality lost
by using this fourth candidate alongside the retained first three. It does not
model finite-sampling mistakes in the final planner; independent planning below
remains primary. Summarize initial/final training regret for old/added pools and
capture `1 - final_sum_regret / initial_sum_regret`; zero initial regret yields
undefined capture, never an invented success. Report ordinary outsider regret,
loss, target KL, agreement and policy movement as descriptive diagnostics.

After all model seals, evaluate all2,489 original VALIDATION rows for all three
models of each family (anchor, BROAD, weighted). Give source pairs equal weight.
Evaluate the saved336-root planning panel using one assigned source-fit model
per root:168 TRAIN roots from42 pairs and168 VALIDATION roots from42 different
pairs,14 pairs per fit per split. Three fits do not triple the sample size.

Keep full-legal model inputs; restrict the proposed fourth to recorded outsiders,
using canonical max-logit grouping and original first-maximum tie rules. Retain
the first three and choose among the up-to-four supported moves using the saved
128 selection worlds. The full recorded teacher shortlist remains a control.
Seal every prediction and selection choice before reading the existing128
independent assessment worlds, which cover every recorded candidate. Preserve
unknown-outcome bounds and equal-action cancellation; do not impute draws or
score an unrecorded full-legal proposal from missing outcomes.

All positions/outcomes are reused DEVELOPMENT evidence. The validation corpus
was excluded from training but has been inspected in earlier studies. A pass
cannot establish fresh generalization, unrestricted deployment or game strength.

Use50,000 whole-pair bootstrap replicates, stratified by assigned fit within
each split, seed2273000000. Report paired95% intervals and per-fit mean contrasts.
Independently reproduce all changed predictions, public/private invariance,
choices, outcome arithmetic, diagnostics and statistics. Authenticate historical
simulation audits without rerunning their trajectories or outcomes.

## Fixed investment screen

All requirements must pass:

1. Each fit removes at least50% of its original added720 TRAIN marginal shortlist
   regret. The previous unweighted outsider-regret floor is not used here.
2. Each fit has strictly lower pair-weighted marginal shortlist regret on the
   original2,489 VALIDATION rows than both its initializer and matched BROAD.
3. On168 VALIDATION planning roots, weighted-minus-anchor mean lower bound is
   at least+.005 expected score (+.5 percentage points), with paired95% lower
   bound strictly positive. Weighted-minus-BROAD paired95% lower bound is also
   strictly positive.
4. Each assigned-fit validation stratum has nonnegative mean lower planning
   contrasts against both controls.

Show each component. Better weighted loss, outsider accuracy, one fit, or TRAIN
performance cannot rescue a failed gate. If learning acquisition fails, do not
interpret a planning null as proof that weighting cannot generalize. A pass
only justifies considering separately specified fresh full-legal confirmation.
No conditional fresh stage or adaptive extension is authorized in this run.

## Execution and preservation

New namespace `tools/planning_opportunity_128`, runner
`run_planning_opportunity_128.py`, output `outputs/planning_opportunity_128`.
Pass targeted source, optimizer, assessment/statistics, orchestration and real
hidden-worker tests before freezing. Verify original/R1 and recent frozen code
identities; never patch predecessor files. Source conversion and historical
simulation assurance are inherited by hash, not replayed unnecessarily.

Resolve failure destinations at call time; fixtures must not write production
failure markers. Reject stale production runtime evidence before/after preflight,
prepare and launch. Use one hidden detached pythonw supervisor, at most three
one-thread CPU workers, no GPU or terminal windows. Inhibit system sleep while
running; display sleep is allowed. No recurring monitor.

Expected execution20-30 minutes after preparation, subject to actual throughput.
Hard operational ceiling two hours and5GiB disk reserve. Preserve all partial
evidence on failure/hold and do not silently retry. Finish with all prescribed
audits, result, readable report and final status; no automatic successor.
After freeze, only unbound handoff notes may change. Report conditional expected
scores accurately; they are not full-game win rates.
