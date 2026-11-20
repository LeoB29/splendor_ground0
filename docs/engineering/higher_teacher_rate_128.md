# Higher learning rate on the complete teacher corpus

Authorized by the owner on 2026-10-08 after reviewing the completed
full_teacher_fit_128 experiment. Implementation: tools/higher_teacher_rate_128;
runner: run_higher_teacher_rate_128.py; output: outputs/higher_teacher_rate_128.
Preserve all dirty work, frozen predecessors and partial evidence. One bounded
experiment is authorized; no automatic sweep, extension, successor or promotion.

## Question and intervention

The completed full-corpus learner acquired little of its own training signal:
about 10.76% reduction in old-target regret, and roughly two percentage points
more exact teacher agreement. Its independent conditional validation improvement
was +0.330171pp versus the anchor, below the fixed +0.5pp development hurdle,
and no demonstrated improvement over the earlier mixed-data specialist. This
does not establish strong acquisition followed by poor transfer. Test whether
the conservative learning rate is limiting acquisition at the existing budget.

Change AdamW learning rate from 1e-5 to 1e-4. All other scientific training
settings match the completed full-corpus control: original three 161 initializers,
9,461 TRAIN rows, row order and targets, batch stream, seeds, objective, optimizer,
update count and fixed final selection. Ten times the rate is not ten times the
updates. An unstable or unsuccessful run rejects this specific recipe and does
not establish a capacity limit or that further optimization cannot help.

## Source and control inheritance

Authenticate the completed full_teacher_fit_128 binding, manifests, final result
and prescribed audits, its three selected models and learning records, and exact
sealed train.pt, validation.pt and panel.pt tensors. Inherit its authenticated
raw-root conversion, split, encoding and old simulation audits. Reuse those exact
tensor bytes and identities; do not reconvert the 4.9GB original root corpus,
retrain saved controls or replay old simulations. Recheck tensor semantics,
split separation, panel correspondence, saved model identity and audit linkage.

Each fit uses all 9,461 breadth TRAIN rows from 159 pairs. All 2,489 original
VALIDATION rows from 42 pairs remain excluded from training and selection. No
historical examples, new positions, filtering or confidence-based selection.
Preserve the 453 empty and 262 singleton TRAIN outsider rows and their existing
zero-loss and optimizer-denominator semantics. The 168 original TRAIN probe
positions and remaining 9,293 rows are reported separately as well as pooled.

Three saved lower-rate full-corpus models are the matched intervention control.
Original anchors and three earlier mixed-data specialists remain context controls.
Saved independent assessment matrices never enter optimization or selection.

## Fixed training and verification

For each original 161001/161002/161003 initializer: fresh AdamW, learning rate
1e-4, weight decay 1e-4, gradient clipping norm 1, batch 64, deterministic float32
CPU, one Torch thread per worker, policy/shared representation trainable and
value head frozen. Use seeds 223700001/2/3 and the unchanged batch tag
focused-teacher-fit-128-training-v1. Exactly 1,784 updates, final always selected.

Retain the first old-score maximum among recorded outsiders beyond the first
three candidates as cross-entropy target, maximum concrete-action logit within
canonical groups, offered-order ties and unchanged short-support denominator.
Epoch shuffling, feature-length sorting, consecutive batches and batch shuffling
must match the lower-rate run exactly: 148 batches per full epoch, 12 full passes
and eight further batches, 114,044 row exposures per fit. Authenticate and compare
initializer, training and probe identity, exact batch digest, per-row exposures,
seed, final update and all scientific settings against the saved control. Only
learning rate may differ; comparison-control metadata is nontraining metadata.

Save checkpoints 0,1,2,4,8,16,32,64,128,256,512,1024,1784. Full TRAIN metrics at
0 and final; intermediate diagnostics use only the fixed original 168 TRAIN
probe. Record exact first-winner and any-tied-best agreement, old-target regret
and cross-entropy. Record all preclip gradient norms, finite checks, mean, maximum
and fraction above the clipping threshold as diagnostics without changing updates.
Independently reproduce all three complete trajectories, every checkpoint,
training metric, gradient diagnostic and exposure count before model sealing.
No best checkpoint, seed, rate or training duration is selected from the results.

## Conditional planning and validation

Use the same saved 336-position panel: 168 TRAIN and 168 VALIDATION positions,
42 whole source pairs per split, four roots per pair. Each position retains its
originally assigned fit. Infer lawful full-legal logits, restrict each proposed
extra action to the recorded outsiders, then choose among the first three plus
that outsider using old 128-world scores. Retain short supports and zero effects.
Apply exactly the same rule to new higher-rate and saved lower-rate/specialist
models. Globally seal every new prediction and choice before reading saved
independent 128-world assessment outcomes. Independently verify lawful inputs,
private-information invariance, inference, choices, arithmetic and statistics.

Re-inferred lower-rate choices, assessment rows and conditional point estimates
must reproduce the completed control exactly; tolerate only declared numerical
logit tolerance, never a changed choice. After sealing, compute all 2,489 old-label
VALIDATION diagnostics for anchor, higher-rate, lower-rate and specialist families,
using each row's originally assigned fit. These are diagnostic old-label metrics,
not independent utility, and never select a checkpoint.

The sole primary remains VALIDATION supported-planning score minus the anchor.
Development screen: mean at least +0.5 percentage points and paired 95% lower
bound strictly above zero. Report direct higher-minus-lower-rate supported and
outsider contrasts and the specialist controls. Passing the anchor screen alone
does not establish that a higher learning rate helped. Compare full TRAIN
acquisition and independent VAL utility together; secondary intervals remain
descriptive, with no multiple-comparison discovery rule or automatic action.

Use 50,000 whole-pair bootstrap draws with seed 2233000000 and chunks of 1,000,
within six fit-by-split strata of 14 pairs. Keep each pair's four roots and all
arms together, resampling TRAIN and VAL independently. Independently reconstruct
all statistics. Three fits share a corpus and do not triple the statistical
sample. Reused development panels and restricted recorded proposals cannot
establish fresh-state, full-legal or game strength. Scores are not game winrates.

Stronger TRAIN acquisition with unchanged or weaker independent VAL utility
would shift attention toward targets, representation and generalization. Little
acquisition despite larger stable steps leaves optimization/capacity unresolved.
A positive planning screen may motivate separately specified fresh full-legal
confirmation, never automatic simulation, arena, promotion or a successor.

## Operations and preflight

Follow docs/engineering/verification_policy.md. No new games or continuations;
inherit completed prescribed simulation audits without claiming fresh exhaustive
replay. Preflight covers sealed-source reuse and tampering, split/target/control
identity, exact matched batches and optimizer recipe, stability reproduction,
bounded real fits, saved-outcome separation, low-rate identity and paired
statistics, and an actual hidden supervisor/model-loading worker. Freeze code,
tests, protocol and dependencies only after the complete preflight passes.

Run three hidden CPU workers, no GPU or visible terminals, with system sleep
inhibited and display sleep permitted. Use a detached hidden supervisor so chat
activity is unnecessary. No recurring monitoring. One-shot launch, 600-second
startup acknowledgement allowance, two-hour execution ceiling from launch and
5GiB free-disk reserve. Preserve failures and stop without automatic retries.
Expected execution including prescribed verification is roughly 25-40 minutes,
based on the completed 25m14s lower-rate study; preparation is additional.
