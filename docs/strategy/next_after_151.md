# Recommendation after151: test breadth versus label precision

Completion update, 2026-09-10: Experiment 152 finished with all audits passing
and its development screen failing. C improved mean MSE only 0.808551% versus
A (required 3%); selected WDL minus 035 was +0.001356337 (required +0.02).
C improved both means versus B, but did not establish a material primary gain.
All 73,656 new continuations and six full fits independently reproduced.
See [the complete result](C:/Users/leoba/Desktop/splendor_ground0/docs/engineering/data_allocation_152_result.md).
Experiment 152 is closed; no fresh test or automatic successor was launched.
The authorization and recommendation below are preserved as historical context.

Authorization update2026-09-09: owner subsequently said "ok do it". Implement
and run152 under `docs/engineering/data_allocation_152.md`, including all audits.
Development only; no new performance test, arena, promotion or automatic successor.
The recommendation below predates that authorization and remains preserved.

2026-09-09. Recommendation only. The owner asked what to do next; no new
experiment is authorized or launched. This review read existing results and
checked151's frozen identity. No fits, model inference, new games, continuations
or test runner ran. Preserve all dirty work and frozen experiments.

## Decision

Recommend one bounded data-allocation experiment, provisionally152. Hold151's
shared-card model and learning rule fixed. Test whether the same total target
simulation budget works better when spread over four times as many independent
source positions. Include a smaller control that separates increased coverage
from the effect of reducing simulations per label.

| Arm | Source positions | Worlds per position | Four-action continuations represented | Role |
|---|---:|---:|---:|---|
| A: current |768|32|98,304|Frozen151 models and original labels|
| B: fewer worlds |768|8|24,576|Same positions, lower label precision|
| C: broader coverage |3,072|8|98,304|Same label precision as B; same total target budget as A|

The primary comparison is C versus A: the effect of reallocating a fixed
simulation budget. C versus B tests the effect of augmenting the training pool
at fixed per-position precision. B versus A tests reduced target replication
on the same positions. C versus A alone must not be described as isolating
coverage, because position count and label precision both change.

All three151 seeds remain paired and retained. A is the three frozen selected
models; B/C each get three fresh full fits and independent complete refits.
This is one experiment with two necessary comparisons, not an architecture sweep.

## Evidence and alternatives

151's model fits its768 training labels closely, but its best validation
checkpoints occur at16/17/24 updates. Selected MSE improves only0.2625% over150's
flat control, and selected WDL minus035 is+.000189887. Card sharing and attention
did not solve this recipe's generalization gap. This does not establish that
coverage alone is responsible. The parameter/example ratio is not causal proof.

149's existing paired-target uncertainty analysis already estimated training
mean-label variance.036729 against zero-predictor MSE.101540, about36.2%; the
validation ratio is19.1%. These are ratios of Monte Carlo variance to target
second moment, not percentages of decision errors explained. There is no reason
to repeat a generic noise audit.130/132 already found high trajectory variance
and little benefit from averaging returns alone in a different frozen value head.
Here, B makes precision's effect observable within the exact current advantage
task, without purchasing additional replication of the same768 states.

The stored post150 split-world selector gained+.050863 train /+.087240 validation
WDL when choosing on one half and scoring on the other. That is suggestive
decision signal, not evidence that a network can recover it from768 examples.
It uses extra simulation information at selection and is reused-development
evidence. It cannot establish an attainable neural ceiling or a strength gain.

More data is not an untried general hypothesis:143's256-to4096-game frozen value
head comparison gave modest predictive improvement. It used ordinary state
values, frozen features and one realized game outcome, unlike this paired
action-advantage learner. That result is a reason to demand measured move
improvement here, not to promise that scaling will solve the problem.

147's long fits and150's dense early checkpoint checks lower the priority of
another schedule extension.151 lowers the priority of another small encoder
substitution.132 lowers the priority of simply buying more worlds per old state.
148 did not establish the desired stronger shared-tree teacher, and the earlier
offline/gameplay discrepancies still prevent promoting an offline predictor.
These findings concern specific tested recipes; they do not reject all search,
value learning or eventual large-scale improving self-play.

The statistical premise is modest: under the same sampling distribution, an
8-world mean and a32-world mean estimate the same conditional expected advantage,
with different estimator variance. Squared-loss training can in principle learn
conditional means from unbiased noisy targets; its population target is not
changed merely by their variance. This does not guarantee finite-data learning
or correct move ordering. The general argument appears in section2 of
[Lehtinen et al., Noise2Noise, ICML2018](https://proceedings.mlr.press/v80/lehtinen18a/lehtinen18a.pdf).
No image-denoising algorithm is being proposed for Splendor.

## Proposed bounded implementation

Keep all768 original training states in C. For B and that part of C, use the
first8 stored149 worlds in their original order, with no outcome-dependent
subsampling. Add exactly2,304 fresh source-game-disjoint states for C,384 per
phase/actor cell, giving512 per cell after the original128. One state per game.
Use new source/world seed namespaces, never any old protected source or outcome.

Match the original143 development source behavior: exact035 scoring, with .2
probability of sampling among the top3 actions at temperature.5 in the first20
turns and greedy otherwise. Do not silently replace this by wholly greedy
source games. Candidate roots and all target continuations retain149's exact
greedy035 policy and four canonical candidate selection rule. Match149's
eligibility restriction on already-held blind reservations; new root/future
blind reservations remain legal. No outcome or model-disagreement selection.

Use the same balanced augmenting-path source assignment and outcome-independent
within-cell position hash rule. Propose an8,192-source-game maximum for the new
2,304 states, stop at the first feasible balanced prefix, preserve every attempted
source, fail on non-natural source/target or exhausted budget. Freeze exact seed
namespaces/hash tags and collector implementation before generation. Verify the
complete source panel and lock all features before collecting its labels.

Model architecture,327297 parameters, initialization seeds149001/2/3, AdamW
settings, batch size64,4096-update maximum/full schedule, validation grid and
earliest-strict-minimum selection all stay as151. Match optimizer-update budget,
not epochs. B uses the original sampler stream over768 states; C uses the same
seeded sampling rule over3072. The sampled state indices cannot be identical
across differently sized pools. Report actual unique state/game exposure at each
selected checkpoint, alongside all histories and every seed's results.

Keep the existing192-state/64-world validation panel fixed. This is another
development screen on reused data; positive results require later fresh
confirmation. Do not re-evaluate149's consumed protected panel or treat new
training-source collection as a fresh performance test.

Propose requiring C's equal-seed validation advantage MSE at least3% below BOTH
A and zero, selected WDL minus035 at least+.02, and selected WDL strictly above A.
Report B, C-minus-B and every seed regardless of outcome. Attribution to broader
coverage also needs C to improve on B; if only B improves, investigate target
replication/optimization effects rather than crediting added positions. These
development contrasts do not establish statistical superiority or a scaling law.

No arena or fresh conditional performance test is included in this initial
experiment. If the primary screen fails, stop. If it passes, propose a separately
frozen fresh conditional test of the locked models, followed only if justified
by gameplay. Do not select a post-hoc winning arm/seed or change the thresholds.
No automatic model promotion, self-play launch or scientific successor.

## Cost and integrity

Only the2,304 additional states need new labels:2,304*8*4 = **73,728 new
continuations**, plus independent replay of all73,728. A/B's existing targets
are reused and independently reaggregated; the old full replay audit stays bound.
This is75% of149's training-collection count, while C's total training-label
budget equals A's. Equal continuation counts are not an exact wall-time match;
path lengths and wider-pool training cost differ.

149 took34.31min to collect98,304 training continuations. A simple same-throughput
projection gives25.73min for the new labels, before replay, source generation,
six main fits, six refits, preparation and tests. A rough planning allowance is
around1--2 hours overall, not a measured152 runtime or deadline. An outcome-blind
timing preflight must check this before main label collection; its small fixed
probe budget and independent replay must be frozen separately.

All source generation, features, input privacy, labels, controls, selected/final
weights, predictions, statistics and chronology must be audited. Freeze152 in
new files; do not edit151 or other frozen dependencies. No extra worlds, state
extension, architecture search or validation-selected seed rescue.

## Local evidence

- [151 results](C:/Users/leoba/Desktop/splendor_ground0/docs/engineering/shared_cards_151_result.md)
- [149 results and target uncertainty](C:/Users/leoba/Desktop/splendor_ground0/docs/engineering/afterstate_advantage_149_result.md)
- [150 recommendation and stored split-world diagnostic](C:/Users/leoba/Desktop/splendor_ground0/docs/strategy/next_after_150.md)
- [143 data-scale results](C:/Users/leoba/Desktop/splendor_ground0/docs/engineering/value_generalization_scale_143_result.md)
- [143 source behavior](C:/Users/leoba/Desktop/splendor_ground0/docs/engineering/value_generalization_scale_143.md)
- [132 repeated-return results](C:/Users/leoba/Desktop/splendor_ground0/docs/engineering/alphazero_deadlock_consistent_repeated_return_value_ab_132_result.md)
- [147 exposure results](C:/Users/leoba/Desktop/splendor_ground0/docs/engineering/value_training_exposure_147_result.md)
- [148 search-budget results](C:/Users/leoba/Desktop/splendor_ground0/docs/engineering/search_teacher_headroom_148_result.md)
