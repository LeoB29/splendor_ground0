# Recommendation after 150: a shared card encoder for action advantages

Authorization update2026-09-09: owner subsequently said "ok, let's do it".
Implement151 under `docs/engineering/shared_cards_151.md`; development gate
first and fresh384 conditional test only on audited pass. The recommendation
record below predates authorization and remains preserved.

Completion update2026-09-09:151 is complete and independently audited; two of
four development gates failed. MSE reduction vs flat0.2625% (<3%) and selected
WDL gain over035+.000189887 (<+.02). No fresh conditional stage, arena or promotion.
See `docs/engineering/shared_cards_151_result.md`. No successor is authorized.

2026-09-09. Recommendation only. No training, model inference, new games,
continuations or test runner launched. This review read the current state and
results, inspected historical alternatives, and computed the descriptive
development-data check below. Preserve all dirty work and frozen experiments.
Champions remain 084 research / 035 fast.

## Decision

Recommend one bounded representation pilot, provisionally 151: **can a learner
that shares its card-processing rules turn the existing action-advantage signal
into useful move choices?** Compare a new, fully trainable shared card encoder
against the frozen dense-selected flat afterstate models from 150. Keep the
afterstate, target, training data, candidate actions and training schedule fixed.

The new model should encode each card using its costs, points, bonus, tier and
legally known location/ownership, then combine those representations with player
resources and the complete candidate action using a shared encoder and attention
pooling. This is a concrete inductive-bias change: learning about the usefulness
of a card's costs or bonus can transfer to other cards. It is a hypothesis,
not evidence that the current inputs omit information or that attention will win.

150 closes the immediate checkpoint-timing remedy: all six models found better
early validation minima, yet both failed the material selected-move criterion.
Plainly extending training, changing checkpoint cadence again, or promoting
the best seed would have low scientific value now.

## New read-only evidence from the existing development data

For each 149 training/validation state, split its stored independent worlds by
even/odd world index. Select the highest mean-WDL action on one half, breaking
ties toward the baseline. Evaluate that fixed action minus 035 on the other
half. Repeat in reverse and average the two directions equally within each
source state, then average source states. No network is involved and no worlds
were generated. The two directions are not separate independent state samples.

| Existing split | States | Worlds per half | Apparent discovery gain, WDL | Opposite-half gain, WDL | Opposite-half expected-score gain |
|---|---:|---:|---:|---:|---:|
| Training |768|16|+.191366|+.050863|+2.543 pp|
| Validation |192|32|+.170329|+.087240|+4.362 pp|

The direct selector changed about 51.50% and 54.95% of choices respectively.
Both opposite-half means were independently recomputed with a second arithmetic
implementation.150's complete frozen identity still matched. Artifact:
[post150 development diagnostic](C:/Users/leoba/Desktop/splendor_ground0/outputs/early_learning_150/analysis/post150_development_target_split_diagnostic.json).

This is post-hoc, reused-development evidence, not a new holdout, optimal oracle
or gameplay result. The selector obtains fresh simulation information relative
to each half; a network does not have that information at inference. A positive
selector result therefore does not prove that these advantages are learnable
from 768 examples. The train/validation difference also cannot be attributed
to 16 versus 32 worlds, because the source states differ. Selection optimism
remains substantial even though positive mean gains survive this split.

Nevertheless, this finding favors testing generalization over assuming the
four-action target panel contains no useful deviations. It motivates the
encoder experiment without establishing a unique cause of the learning failures.

## Relationship to accumulated evidence

This is not the programme's first entity model or generic action-value fit.
Earlier entity actors, counterfactual-Q learning,053/054 and the frozen-feature
preference probe 003 already give negative evidence.116 added a residual tower
around a frozen gen 11 model and improved offline fit without a gameplay win.
Those findings require a controlled comparison and a decision-quality gate here;
they do not establish the adequacy of 149's fresh flat encoding for its specific
greedy 035, paired terminal-WDL targets.

The new intervention is shared semantic card processing on that controlled
afterstate/advantage task. It is not another frozen 035 feature head, a public
history addition, a larger residual tower, generic PPO or an MSE-only promotion.
The old clean LOO actor 041 already changed actions and failed to establish a
gain over 035; another plain policy-gradient update is not an untried shortcut.

148 delivered more search compute without establishing the desired teacher
headroom.134 and 138–141 show why sampled-root gains cannot substitute for actual
games. None of these results proves that search, value learning or large-scale
self-play cannot work. The long-term requirement remains repeated improvement
in complete playing strength, not a succession of better fitted targets.

## Proposed bounded design

Use the same 768/192 development states and outcomes, four candidates, exact
greedy 035 continuation objective and observable afterstate interface. Both
representations receive the same legally available information. Deriving card
attributes from the public catalog is allowed; hidden deck order, unseen
opponent cards, actual future draws and new outcome information are not.

Implement a small shared card encoder with action-conditioned attention pooling
and the necessary global/player/noble/pending-event fields. Retain complete
action distinctions, payments, returns and target identity. Arbitrary internal
entity-list ordering must not change predictions; this is not permission to
permute game colors, market slots or temporal events as if they were proven
game symmetries. Check observer privacy and information coverage explicitly.

Target roughly 320,000 trainable parameters, within 10% of the flat control.
Freeze the exact token schema, widths, architecture and parameter count before
looking at any new fit. No architecture grid or extra engineered lookahead.
Use three seeds, all retained, with the same minibatch streams, AdamW settings,
4096-update budget and dense-early validation grid as 150. Different architectures
cannot share identical weight tensors; pairing concerns seeds/data/batches.
Predict scalar(candidate) minus scalar(baseline), as before, with baseline-first
ties. Keep the three frozen dense-selected afterstate models from 150 as controls.

First run a development screen. Propose requiring equal-seed validation MSE
at least 3% below BOTH the flat afterstate control and zero, selected WDL gain
over 035 at least+.02, and a positive selected-WDL change versus the flat control.
Do not require selection in the early interval: that was 150's mechanism test,
not a requirement that another architecture learn at the same speed. Report
every seed and criterion. This screen is developmental, not a fresh strength test.

Only a complete development pass proceeds to a prespecified fresh conditional
move test. Suggested scope:384 source-game-disjoint states,64 per phase/seat
cell, generated after all model choices are locked; four fixed actions and 128
independent worlds per state, at most 196,608 continuations plus an independent
replay audit. Retain the original eligibility/privacy restrictions, complete
natural termination and explicit forced-pass deadlock handling. None of the
protected 143–150 panels or outcomes may be reused as fresh evaluation.

Freeze this conditional stage, namespaces, selection rules, timing preflight
and statistics before training. Proposed fresh requirements: treatment-minus 035
and treatment-minus-flat selected WDL both at least+.02, both paired 95% CI lower
bounds positive and half-widths at most.02, plus advantage-MSE difference versus
flat with CI upper below zero and all integrity checks passed. Bootstrap whole
source games, averaging seeds within each state. Wider intervals mean unresolved
precision; no extensions or threshold changes to obtain a pass.

All three fits must be independently reproduced; verify frozen inputs and all
new features, predictions and statistics. If the development screen fails,
stop before new continuations. If the fresh stage fails, stop this encoder
recipe. Even a fresh conditional pass would only justify a subsequent gameplay
proposal, not a promotion or self-play launch. No full-game arena is included.
The initial stage reuses expensive data; the conditional stage is substantial
and requires an outcome-blind timing estimate. No measured runtime is claimed.

## Sources

[150 results](C:/Users/leoba/Desktop/splendor_ground0/docs/engineering/early_learning_150_result.md),
[149 results](C:/Users/leoba/Desktop/splendor_ground0/docs/engineering/afterstate_advantage_149_result.md),
[116 results](C:/Users/leoba/Desktop/splendor_ground0/docs/engineering/alphazero_capacity_pilot_116_result.md),
[054 protocol](C:/Users/leoba/Desktop/splendor_ground0/docs/experiments/resolver_public_history_representation_054.md),
and the historical 003/041 records in
[AGENT_CURRENT](C:/Users/leoba/Desktop/splendor_ground0/AGENT_CURRENT.md).

Attention over sets is an established way to model entity interactions while
preserving invariance to internal list order. This supports the architecture's
form, not its effectiveness in Splendor:
[Set Transformer, Lee et al., ICML2019](https://proceedings.mlr.press/v97/lee19d.html).
