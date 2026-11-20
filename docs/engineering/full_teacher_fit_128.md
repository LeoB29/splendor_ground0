# Full-corpus focused teacher learning

Authorized by the owner on 2026-10-08 after the saved learning-curve review.
Implementation: tools/full_teacher_fit_128; runner: run_full_teacher_fit_128.py;
outputs: outputs/full_teacher_fit_128. Preserve the intentionally dirty worktree
and all frozen predecessors. This authorizes one bounded learning experiment;
no automatic dose extension, model selection, fresh simulations or arena follows.

## Question and fixed intervention

Can the established outsider learner acquire useful transferable preferences
when focused practice covers the complete existing breadth TRAIN corpus?
The small focused experiment learned its168 positions but did not demonstrate
transfer, and its13-stage saved learning curve found no convincing early window.
The older mixed specialist saw the breadth rows only four times, interleaved
with historical supervision. This experiment combines broader coverage with
concentrated outsider training; it is not a pure causal test of diversity.

Use all9,461 breadth TRAIN rows from159 whole source pairs, excluding all19,047
historical TRAIN rows. Preserve the original2,489 breadth VALIDATION rows from42
pairs outside optimization and checkpoint selection. Source rows, original
split and targets come from the closed proposal_breadth_learning_128 and
proposal_specialist_128 evidence. The original168 focused TRAIN positions are
an exact subset; separately report acquisition on them and the added9,293 rows.
Do not reselect source identities or filter by confidence, margin or outcomes.

Authenticate prior complete source/export/model and prescribed-audit evidence.
Inherit authenticated mature tensor/game/trajectory audits; do not replay old
training or simulations. Verify the new breadth-only transformation, original
split, target and tensor identity, and assessment-panel correspondence. Training
payloads contain lawful tensors and OLD128-world targets only. Saved independent
assessment outcomes never enter optimization or checkpoint selection.

The target is the first maximum original terminal WDL among recorded candidates
outside the first three. Keep the existing cross-entropy, maximum concrete-action
logit within each canonical group, offered-order tie breaking, empty/singleton
zero-loss behavior and unchanged batch denominator. Unoffered actions get no
invented labels. Architecture, lawful inputs and continuation policy are unchanged.

## Three fixed fits

Start from the original161001/161002/161003 initializers, never the tiny-fit final
models. Each sees the same pooled9,461 TRAIN rows. Exactly1,784 optimizer updates,
batch64 with the last short batch, fresh AdamW, lr1e-5, weight decay1e-4, clip1,
float32 deterministic CPU and one Torch thread per worker. Freeze value-head
parameters while training the policy/shared representation. Seeds223700001/2/3
and tag focused-teacher-fit-128-training-v1 remain as in the tiny experiment.

Each epoch shuffles the sealed row order with Python Random seeded by
tag|training_seed|epoch, stable-sorts by feature length, forms consecutive batches
of64 and shuffles those batches. Stop exactly at update1,784, even mid-epoch.
There are148 batches per complete epoch:12 complete passes and eight additional
batches. Record exact stream digest and per-row/total exposures (approximately
114,000), not a rounded epoch label. Optimizer updates match previous studies;
exposures, padding, FLOPs and wall time do not.

Save checkpoints0,1,2,4,8,16,32,64,128,256,512,1024,1784. At0 and1784 record all
TRAIN old-target loss, first-winner and any-tied-maximum agreement, regret, and
original168/added9,293 subgroup metrics. Intermediate checkpoint diagnostics use
only the original168 TRAIN probe to keep cost bounded. Scope is explicit in each
report. Diagnostic batches may be sorted by feature length for efficiency.
All diagnostics are descriptive; always select final1784, with no best-fit choice.
Independently reproduce all three optimizer trajectories, checkpoints, metrics,
exposures and frozen value-head identities before sealing the final models.

After model sealing, evaluate old-target metrics on all2,489 VALIDATION examples
for the original anchors and fixed final focused models, with the saved mixed
specialists as an additional control. Each validation row uses its original
assigned source-fit model, yielding2,489 rows per family. This differs from the
TRAIN diagnostics, where each of three models is evaluated on all9,461 rows.
These metrics never select a checkpoint.

## Saved-outcome development assessment

Retain all336 main teacher_signal_capture_128 positions:168 TRAIN and168 VALIDATION,
42 whole source pairs each, four roots per pair. Each position uses its original
assigned source-fit model. Three models do not triple the statistical sample.
Infer full lawful legal-action logits, then restrict the proposed extra action
to the recorded outsiders. Choose the supported move using old128-world scores
among the first three plus the proposed outsider. Retain empty/short supports
and structural-zero effects. This restriction cannot test unrecorded proposals.

Apply the identical restriction and supported-decision rule to the earlier saved
mixed-data SPECIALIST. Keep original anchor, teacher and prior BROAD/NARROW
controls. The completed tiny-fit result is historical context only. Authenticate
all control models and their completed audits; no old fits are rerun. Predictions,
model assignments and choices for all roots/arms must be globally sealed before
opening the saved independent128-world assessment matrices. Independent lawful
input/private-information checks, inference, choices, outcome arithmetic and
statistics cover every newly evaluated model decision.

The single primary investment endpoint is VALIDATION focused-supported minus
unchanged anchor. The development screen requires mean at least+0.5 percentage
points and paired95% lower bound strictly abovezero. Specialist contrasts and
outsider quality are diagnostic; beating the tiny model alone does not count.
Report TRAIN, VALIDATION and between-split differences, including all short
supports. Use50,000 paired bootstrap draws, seed2233000000, chunks1000, within
six fit-by-split strata of14 whole pairs each. Keep all four roots and all
contrasts together; resample TRAIN and VALIDATION independently. Independently
reconstruct statistics. Secondary intervals are descriptive, not a multiple-
comparison discovery rule. Fixed fits and reused development outcomes limit
inference. These conditional continuation scores are NOT game win rates.

Strong TRAIN acquisition with weak validation lowers the priority of this
unchanged learning recipe. Weak TRAIN acquisition leaves optimization/capacity
unresolved. Outsider improvement without supported-planning improvement is
insufficient. A positive screen motivates separately specified fresh-position
confirmation using actual full-legal proposals; no automatic launch or promotion.

## Verification and operations

Follow docs/engineering/verification_policy.md. No new games or continuations
are generated, so inherit the predecessor's authenticated prescribed simulation
audits. Do not describe inherited sampled replay as exhaustive replay. Verify
all new fits and assessment decisions independently, source transformations,
split/firewall integrity, complete whole-pair statistics and model/code hashes.

Preflight includes malicious/tampered split/target/control records, first-max
ties and short supports, exposure/final-checkpoint rules, bounded real fit and
independent reproduction, full source identity checks, saved-outcome access
separation and an actual hidden supervisor/model-loading worker. Freeze code,
tests, dependencies, protocol, source and controls only after checks pass.

Run three hidden CPU workers, no GPU, console windows or recurring monitor.
Keep system sleep inhibited while allowing display sleep. One-shot launch,
two-hour execution ceiling and5GiB free-disk reserve; preserve partial evidence
on failure and stop without automatic retry. Preparation is separate. Earlier
mixed-data timings suggest tens of minutes execution, but diagnostic and source
checks may change this estimate. Report measured progress after launch.
