# Finite score-aware outsider targets

Owner authorized implementation and execution on 2026-10-09 after reviewing the
completed higher-rate checkpoint curve. This is one bounded experiment in
tools/soft_teacher_targets_128, entered through run_soft_teacher_targets_128.py,
with outputs under outputs/soft_teacher_targets_128. Preserve all intentionally
dirty work and frozen predecessors. No automatic successor or promotion follows.

## Question and intervention

Can finite preferences derived from the recorded action scores transfer better
than a probability-one winner label? Useful teacher guidance survived independent
reassessment, stronger fitting acquired substantial TRAIN signal, and the saved
checkpoint curves found no convincing earlier planning advantage. These facts
motivate the intervention; they do not identify noisy labels as the cause.

Use the exact 9,461 breadth TRAIN positions from 159 whole source pairs and keep
all 2,489 VALIDATION positions from 42 pairs outside optimization and checkpoint
selection. Reuse the authenticated full_teacher_fit_128 tensors and their saved
old_sums_wdl. Do not add historical replay, new states, games or simulations.

Let U be the recorded candidates outside the actual first three. For each j in U,
q_j=(128+old_sums_wdl[j])/256 is the mean terminal score on the OLD 128 worlds,
where win=1, draw=.5 and loss=0. Fix t=softmax(q/.05) and minimize
-sum(t_j*log_softmax(z)_j), using the existing maximum concrete-action logit for
each canonical group. Stable softmax arithmetic must preserve finite targets.
Equal old scores receive equal target probabilities; no arbitrary first winner
breaks a target tie. No outcomes for unrecorded actions are invented.

Temperature .05 is a declared five-percentage-point score scale: a 5pp difference
means e:1 target odds and a 1pp difference means approximately 1.22:1. It is not
calibrated uncertainty or a fitted noise estimate. There is no temperature sweep,
confidence filter, action filter, additional regularizer or architecture change.
The earlier gap-weighted hard-sign ranking loss still favored unlimited margins
and used the two actions outside the top two; this finite target uses up to five
actions outside the actual first three. Previous failed learning variants remain
relevant cautions, not proofs that this intervention will succeed.

Empty and singleton outsider sets give differentiable zero loss. Keep those rows,
batch denominators, optimizer steps and exposure counts. Models always receive
full lawful legal-action inputs; only loss and proposal support are restricted.

## Fixed fits and authenticated controls

Start from the original three 161 initializers. Match the completed
higher_teacher_rate_128 recipe: 1,784 updates, batch64 including short final
batches, fresh AdamW at lr1e-4, weight decay1e-4, norm clipping1, deterministic
float32 CPU and one Torch thread per worker. Keep the same seeds, exact batches,
114,044 row exposures per fit and frozen value-head parameters. The policy and
shared representation remain trainable. Only the target objective changes.

Always use the final update1784. Saved updates0,1,2,4,8,16,32,64,128,256,512,1024,
1784 support TRAIN diagnostics only; they never select, stop or extend fitting.
Retain all three fits and original source-fit assignment at assessment.

Reuse the completed hard-target HIGH-rate models as the exact recipe control and
the LOW-rate final models as the practical reference. Retain original anchors and
the earlier specialist as context. Authenticate prior source/model/fit/audit
receipts and prove the high-rate initializer, batches, dose and non-objective
settings match. Do not rerun old training or simulation trajectories.

Independently reconstruct the new target arithmetic and all three complete new
optimizer trajectories, checkpoints, metrics, exposures, clipping diagnostics
and frozen value-head identities before sealing the final models. Report soft
target cross-entropy, entropy and target KL alongside the unchanged hard-label
agreement, tied-best agreement and old-score regret. Preserve explicit metric
names: raw cross-entropy under different targets is not directly comparable.
Report full TRAIN diagnostics at0/final, the original168-row TRAIN probe at other
stages, and full VALIDATION metrics after final-model sealing. Acquisition and
parameter/policy movement are diagnostics, not causal proof of a noise mechanism.

## Saved independent-outcome development screen

Keep the original336 assessment roots:168 TRAIN and168 VALIDATION, each42 whole
pairs with four roots per pair. Use the assigned source-fit model for each root;
three fits do not triple the statistical sample. Infer full legal-action logits,
restrict the extra proposal to recorded outsiders, and choose the supported move
using OLD128-world scores among the first three plus that proposal. Preserve
candidate ordering, ties, empty supports and structural-zero effects.

Globally seal all predictions, control identities, old-world choices and full
VALIDATION diagnostics before opening the saved independent assessment matrices.
Re-infer both hard-target controls, reproducing their completed choices and
conditional point estimates. Independently verify new inference, lawful inputs,
private-information invariance, choices, outcome arithmetic and statistics.

Use50,000 whole-pair bootstrap draws, seed2233000000, within the original six
fit-by-split strata of14 pairs. All arms/contrasts stay paired; TRAIN and VALIDATION
are resampled independently. Independently reconstruct the statistics.

Further investment requires BOTH:

- VALIDATION new-supported minus anchor mean >=+.005 score (+.5pp), with paired
  95% lower endpoint strictly above zero.
- VALIDATION new-supported minus LOW-rate-supported paired95% lower endpoint
  strictly above zero.

The direct new-minus-HIGH-rate comparison is mandatory diagnostic reporting for
objective attribution. If it is uncertain, do not claim that softened targets
caused an improvement, even if the practical screen passes. Lower training loss,
less overconfidence, outsider quality or better TRAIN performance alone cannot
pass the gate. No fit, subgroup or intermediate checkpoint may rescue a miss.

This is reused development evidence on restricted recorded support, NOT game
win rates, fresh-position confirmation or proof of full-legal proposal strength.
A pass motivates a separately specified fresh-position/full-legal confirmation;
none starts automatically. A null with substantial target acquisition lowers
this fixed recipe's priority; weak acquisition remains unresolved. No nearby
temperature or optimizer rescue follows. Independent-game diversity and
representation remain competing explanations.

## Verification and operations

Follow docs/engineering/verification_policy.md. Inherit authenticated completed
simulation/source checks rather than replaying them. Verify every new target,
fit, evaluation decision, control identity and paired statistic. Preflight must
exercise ties, near ties, extreme scores, short supports, target gradients, split
and outcome firewalls, matched recipe/exposures, fixed final selection, full
control reproduction, composite gate failures and an actual hidden worker.

Freeze this protocol, code, tests, dependencies, inputs and passing preflight
before production. Run one detached windowless supervisor with three hidden CPU
workers; no GPU, visible terminal windows or recurring monitor. Inhibit system
sleep while allowing display sleep. The process is independent of active chat.
Use a two-hour execution ceiling and5GiB disk reserve. Preserve all partial
evidence on failure; no automatic retry, extension, successor or promotion.
Expected execution30-45 minutes including verification, preparation additional,
based on the preceding22-25 minute three-fit experiments; this is an estimate.
