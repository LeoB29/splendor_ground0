# Recommendation after the local generalization panel

2026-09-17. Review and recommendation only. No successor implemented, inference,
simulation, training, profiling or new statistical analysis performed. Preserve
the intentionally dirty worktree and all frozen experiment dependencies. The
completed panel binds the whole existing src tree; a later implementation must
use new files outside that inventory. Monitoring remains off.

## Decision

The highest-value immediate experiment is a **saved-data, equal-budget test of
successive halving in the proven terminal-rollout planner**. Test one fixed rule
against uniform sampling, then stop. It needs no new simulations or training.
Its purpose is to decide whether a fresh, expensive planner arena is worth
proposing. It cannot itself establish stronger full-game play.

Confidence is moderate. Two independent read-only reviews identified learned
root proposals as the strongest missing learning-to-search mechanism; the search
review also favored this saved-data screen for immediate information per compute.
This recommendation prioritizes that unusually cheap decision before another
paid learning or planning study. It does not identify sampling allocation as the
dominant bottleneck, nor abandon learned proposals.

## What the accumulated evidence supports

| Evidence | Consequence |
|---|---|
| 155/156 established strong full-game terminal-rollout planning with 035 continuations. | Preserve that validated mechanism while testing a specific change. |
| 160's 128-world planner scored 65.5% against its 32-world counterpart, with a paired 95% interval of 60.875--70.125%. Mean root time increased about fourfold, from 19.482 to 78.101 seconds. | More sampling has demonstrated value. This is not equal-time superiority, and does not prove adaptive allocation works. |
| 157 learned useful restricted action rankings; 161's stronger-teacher students had positive comparisons with the 32-world students and 157. | The broad claim that the teacher has no learnable signal is contradicted. Useful compression exists. |
| 158 did not establish better planning from stronger learned continuations; its partial owner-stopped audit remains provisional. | Direct policy improvement does not automatically improve this planner through continuation substitution. |
| 162 refresh scored 50.4167% against equal-dose replay, and 53.0208% against unchanged 161. | Fresh examples have no established advantage over matched extra practice. The modest direct gain is real evidence, but failed the frozen practical gate. |
| The completed 5,400-game panel scored anchor/refresh/replay at 69.4444/67.6111/69.4722%. Refresh minus anchor was -1.8333 pp, 95% [-4.9444,+1.3056]; replay minus anchor +0.0278 pp, [-2.6111,+2.7500]. | Neither update established broader improvement across these local opponents. Regression and equivalence are also unproved. Another unchanged refresh has a weak immediate case. |
| Every 162 final checkpoint had worse validation loss than its selected checkpoint; refresh had better selected validation loss than replay in every fit. | Neither more passes nor better target fit supplies an obvious strength intervention. The loss or capacity bottleneck is unidentified. |

Do not multiply these win rates, select the best fit, or infer external/human
strength from the local panel. The network-to-stronger-search feedback loop is
still unproved. No failure here rules out self-play learning, larger models,
different objectives or more general search in principle.

## One concrete screen

Use the already identity-selected 600 roots from 200 source seed pairs in
`outputs/world_budget_160/review_sampling_20260916/roots.json`. There are three
phase positions per pair, from one previously hash-assigned source seat. Keep
all roots; do not select by gaps, results, phase or intervention size.

Each source root stores terminal results for up to four canonical actions under
128 sampled worlds. These are complete action-by-world outcome vectors, so the
counterfactual below needs arithmetic and artifact verification only.

For a root with four candidates, compare exactly:

1. **Uniform reference:** evaluate all four on the first 32 selection worlds;
   choose the action with the highest mean terminal score.
2. **Successive halving:** evaluate all four on the first 16 selection worlds;
   retain the two highest means; evaluate those two on the next 32 selection
   worlds; choose between them using all 48 of their sampled results.

Both consume 128 action continuations: 4 x 32 versus 4 x 16 + 2 x 32. Preserve
canonical action representatives and original-order tie breaking at both stages.
Do not change continuations, outcome utility, candidate support, or world law.
For one, two or three candidates, both arms use the unchanged 32-world uniform
rule and contribute zero paired difference. Report their frequency and retain
them in the main estimate. Equal continuation counts do not prove equal wall
time: trajectory lengths and scheduling can differ.

Before calculating any new comparison, bind one analysis specification and
implementation. Use 64 fixed permutations of the 128 world indices, generated
with NumPy default_rng seeds 2026091700 through 2026091763. In every permutation:

- only the first 48 worlds may inform either decision;
- the uniform arm uses the first 32 of those 48;
- judge both locked choices on the remaining 80 worlds;
- preserve the full vector of action results when permuting a world;
- average all 64 partitions; never select a favorable partition or rule.

Those repeated partitions reduce dependence on one fortunate sample split.
They do not create 64 independent experiments. An outcome may enter selection
in one partition and evaluation in another, but the two sets are disjoint for
each individual decision. No pooled evaluation result may affect a decision.

Primary endpoint: held-out terminal score of the adaptive choice minus that of
the uniform choice, where score is win=1, draw=0.5, loss=0. Average partitions
within root, phases within source pair, then equally across the 200 pairs. Use
50,000 whole-pair bootstrap resamples with seed 2026091764 for a descriptive
paired 95% interval. The independent units are the 200 source pairs, not roots,
worlds or partitions. Phase, seat, support size, choice changes and elimination
behavior are descriptive only; they cannot choose another allocation rule.

The analysis should read only its allocated cells when constructing each
decision, record those accesses and independently reconstruct the final choices
and endpoint. Verify original source hashes, candidate alignment, score domain,
equal per-root budgets and partition separation. Use small hand-checkable
fixtures to catch tie, survivor and held-out leakage errors. Do not rerun the
original expensive simulations or claim a new full inner replay of 160.

The source root-list SHA256 is
`3c762d22dfcd412239c1f485826fe6ef72e802a5b7f340d2732fa1e305763755`;
source result SHA256 is
`2e3efa2c524f8d8c68284fad0f3400365bcb08f4b598aecf121039a4c45b19e5`.
Recheck these and the source game/root hash chain before analysis. The old
diagnostic summary and its script are evidence, not files to overwrite.

## Interpretation and stopping rule

This is explicitly a **development screen on reused data**. The roots come from
the 160 matchup and their outcomes have already supported sampling diagnostics
and training. Disjoint evaluation within a partition protects against selecting
and judging a move on the same outcomes; it does not turn the overall study into
a fresh confirmation or test the all-128-world teacher against ground truth.

A positive estimate with its paired interval above zero supports proposing a
fresh equal-budget adaptive-versus-uniform gameplay experiment. Report the
effect size, not just its sign, when deciding whether the likely value warrants
that expense. No model or planner is promoted from this screen.

A negative or inconclusive result does not justify an automatic larger sample,
another halving schedule, a phase-specific rescue or a sweep of thresholds. Set
aside this particular rule and reconsider the learned-proposal experiment. It
would not rule out all adaptive allocation. Conversely, a positive root result
may fail to compound into better full-game play, as earlier programme history
has repeatedly shown.

The incremental simulation and training cost is zero. File verification,
implementation and checks still take time; no measured runtime estimate exists
for this new analysis. This should avoid another multi-hour teacher collection.
There is no automatic successor or monitoring attached to the recommendation.

## Strongest objection: adaptation already failed once

Experiment 057's confidence-based stopping saved 34.46% of continuation work but
lost 11.11 points of exact action agreement and increased cross-regret by
0.00919 relative to its uniform comparator. Early elimination can discard good
actions because their first estimates are noisy. That risk remains here.

The present comparison differs in both operator and budget: it uses direct
terminal WDL means under the validated 155/160 greedy-035 continuation contract,
spends the full reference budget on survivors, and evaluates choices on disjoint
worlds. The older rule stopped early using marginal intervals over sampled
minimizing-resolver values. These distinctions justify one cheap test; they do
not cancel the negative historical evidence.

The design draws on fixed-budget best-arm identification, including
[Karnin, Koren and Somekh (2013)](https://proceedings.mlr.press/v28/karnin13.html).
Theory for bandits is not a strength guarantee for correlated action outcomes
or these policy-dependent simulated futures.

## Why the other leading hypotheses wait

**Learned proposals are the strongest next paid mechanism hypothesis.** Every
current successful student still chooses within 035's top-two-plus-two-random
set. Adding that chosen move back does not expand search. A real test should
keep three common baseline candidates (035 top two plus the first random
alternative), then compare a fourth uniform alternative against the highest
161-scored canonical group outside the common three. A third arm uses the
highest 035-ranked group outside the common three to control for replacing
random exploration with extra deterministic exploitation. Keep all three 161
fits, four slots, the same world budget and 035 continuations. Full-legal scores
are available, but their maxima were not trained or validated for this role;
tail-ranking errors are a substantive risk. This deserves its own fresh-world
screen and later gameplay design, not an automatic launch alongside this study.

**Retaining older teacher examples is the best remaining nearby learning
alternative.** 161/162 omit the original 155/156 training corpus. Adding it back
at matched dose could help, but forgetting has not been demonstrated, and its
32-world labels change both label noise and source distribution. It tests a
useful data mixture, not a clean diagnosis of forgetting. This is more grounded
than an arbitrary loss or network-size change, but weaker than the nearly free
sampling test as the immediate decision.

**Teacher batching is a useful engineering enabler.** The prototype exists, but
representative numerical equivalence and speedup remain unmeasured. A separate
bounded timing/equivalence probe could reduce future study costs. It does not
itself establish better decisions or identify a learning bottleneck.

The long-term aim remains stronger planning at a practical computation budget,
followed by learning that reliably improves its next generation. Independent
opponents and eventual prospective strong-human comparisons remain necessary
for a superhuman claim. Another local screen alone cannot establish that claim.
