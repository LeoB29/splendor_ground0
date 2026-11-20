# Recommendation after 149: inspect the missing early learning window

Completion update:150 completed and was fully audited. All six models selected
updates12–77, lowering validation MSE, but both development screens failed the
required+.02 WDL gain over035. No fresh confirmation followed. See
[the result](C:/Users/leoba/Desktop/splendor_ground0/docs/engineering/early_learning_150_result.md).
The original proposal and authorization notes below are historical.

Authorization update: the owner subsequently approved this proposal with "ok
do it". Execute the bounded development diagnostic as150 under
[the frozen execution protocol](C:/Users/leoba/Desktop/splendor_ground0/docs/engineering/early_learning_150.md).
See AGENT_CURRENT for status. The original proposal below remains historical;
no fresh test, new continuations or automatic successor is included.

2026-09-09. Recommendation only. No training, inference, continuations, games or
tests were launched during this review. Preserve all dirty and frozen work.
Champions remain 084 research / 035 fast.

## Decision and rationale

Recommend one small development diagnostic, provisionally experiment 150:
**does the existing learner briefly acquire useful information before it
overfits, during the interval that 149 did not measure?** Reuse only 149's
training and validation data and reproduce its six fixed training trajectories.
Add validation at every update from 1 through 127; retain every original check.
Compare original versus expanded checkpoint selection on the SAME trajectories.

149 completed all six schedules, but five models selected initial weights.
The first trained check followed 8192 sampled states, about 10.7 average passes
through the 768-state training set. The saved curves show strong training fit
and poor validation transfer without resolving the first passes. A useful
early checkpoint is possible, not an established explanation for the failure.

Include a zero-advantage reference: all candidate predictions tie, so the fixed
tie rule keeps 035's move. Descriptively on the consumed 149 test, its MSE was
.085915, versus .088044 afterstate and .084892 control. The control's slightly
better MSE coexisted with worse chosen moves. Thus both prediction error and
selected-move outcomes must be reported; prediction fit alone is insufficient.

Longer training already produced substantial overfit in 147 and 149. Four times
the search allowance in 148 did not establish strength headroom, although a
useful effect remains possible. More return averaging has negative evidence
from 130/132. A new architecture or regularizer would change the learner before
this cheap measurement gap is closed. 134 and 138–141 also warn that conditional
move gains cannot substitute for full-game improvement.

This differs from 147's extra optimization exposure on another value learner:
every optimization step here stays fixed. Only the eligible checkpoint set
expands inside the missing interval. This diagnostic includes no new simulation
collection, fresh test or arena, and cannot reopen 149's failed result.

## Proposed bounded design

Use both original representations and all three paired seeds. Keep features,
source-game splits, targets, initial weights, batches, optimizer, learning rate,
clipping, loss, 4096 updates and earliest-minimum-MSE selection unchanged. Do not
zero-initialize the head, shrink the model, tune hyperparameters or select a
best seed. Validate at updates 0 through 128, then every 128 through 4096.

Additional evaluation must not change RNG state or training. Require exact
reproduction of every original coarse history value and original selected/final
tensors, plus independent reproduction of the early histories and new selected
tensors. Freeze a separate protocol and explicit dependency inventory before
reruns; keep 149's source untouched. Reuse existing targets without replaying
their continuations again, verifying their original audited hashes instead.

Report full early curves and original/expanded selections per seed and as
equal-seed averages for each representation. Compare validation MSE against
the original selection and zero predictor. Report validation WDL gain of
selected moves over 035, its change versus the original selection, and move
change rates, using all 192 fixed validation states and candidate panels.

For a concrete development screen, propose requiring at least two of three
seeds in a representation to select updates 1–127; equal-seed validation MSE
must improve at least 3% over BOTH that representation's coarse reference and
zero. Its equal-seed validation move gain over 035 must reach +.02 WDL and
improve over the coarse reference. These practical screening rules are not
significance tests. Freeze them before new fits. Report both representations;
a control-only pass supports no special afterstate benefit.

These measurements reuse development outcomes for selection. They cannot
establish out-of-sample superiority. Do not advertise validation confidence
intervals as unbiased post-selection evidence. Do not evaluate newly selected
models on 149's consumed test or any other historical protected outcomes.

If the screen fails, close the missed-early-window remedy for this recipe at
the declared development threshold; do not generalize to every optimizer or
afterstate architecture. If it passes, freeze the selection procedure and
propose a fresh source-game-disjoint conditional move test with coarse and
035 comparators. Even that later test cannot establish whole-game strength.
No fresh test, arena, promotion or automatic successor is included here.

This should cost much less than 149 because its continuations are reused.
Original fits took about a minute each when run concurrently. Extra validation,
preparation and independent reproduction add time; no new runtime is measured
yet. Do not promise a precise duration from those historical fit timings.

## Evidence

[149 results](C:/Users/leoba/Desktop/splendor_ground0/docs/engineering/afterstate_advantage_149_result.md),
[149 protocol](C:/Users/leoba/Desktop/splendor_ground0/docs/engineering/afterstate_advantage_149.md),
[147 results](C:/Users/leoba/Desktop/splendor_ground0/docs/engineering/value_training_exposure_147_result.md),
[148 results](C:/Users/leoba/Desktop/splendor_ground0/docs/engineering/search_teacher_headroom_148_result.md),
and the [programme review](C:/Users/leoba/Desktop/splendor_ground0/docs/strategy/independent_superhuman_review_after_134_137.md).

Early stopping acts as regularization in settings such as kernel regression;
that does not guarantee a benefit for this neural network
([Raskutti, Wainwright and Yu, 2014](https://www.jmlr.org/papers/v15/raskutti14a.html)).
Checkpoint selection can itself overfit validation, which motivates separate
fresh confirmation after any development screen
([Cawley and Talbot, 2010](https://www.jmlr.org/papers/v11/cawley10a.html)).
