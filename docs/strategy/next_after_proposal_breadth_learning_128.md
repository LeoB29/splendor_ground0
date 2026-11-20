# Next experiment after broader supervision failed to improve planning

2026-10-02. Recommendation only. Test whether explicitly teaching the network
to choose the best additional move outside the planner's existing first three
can improve four-candidate, 128-world planning. Use the broader teacher's saved
comparisons and a matched overall-winner learning control. Confidence in this
allocation is moderate; objective mismatch is a hypothesis, not a diagnosis.

AGENT_CURRENT was read first, followed by the latest protocol and results,
earlier learning, ranking, breadth and gameplay evidence, and the wider strategy
including 134/137. Three independent reviews challenged the recommendation.
Two preferred the targeted learning intervention; one preferred a frozen-model
shortlist diagnostic. That disagreement is material and is discussed below.
No new inference, training, simulation, resampling, tests or experiment
implementation ran. Small descriptive arithmetic used saved validation records.
All frozen predecessors and existing dirty work remain intact.

## What the evidence now supports

The broader teacher scored 56.2189% against the original four-candidate anchor
in 402 games, with paired 95% interval [51.6169%, 60.6965%]. Its earlier fresh
position screen gained +1.2316 percentage points over that anchor and +1.0829
over equal-budget extra precision. Those results establish useful additional
candidate coverage for this mixture. The gameplay gain used approximately
1.94 times the decision time; it does not establish equal-time superiority.

The completed broader-supervision learner gained only +0.0749 percentage
points over the anchor on 360 fresh positions, with interval [-0.2474,+0.4243].
Against matched narrow supervision it gained +0.1823, interval [-0.0315,+0.4384].
These are conditional terminal-score differences, not arena win rates. All
prescribed checks passed. The primary interval's upper end is below the preset
+0.5-point investment hurdle on this population. This closes the fixed recipe;
it neither proves exact zero nor rules out useful teacher information.

The broader learner changed 124 of 353 eligible fourth proposals and 44 of 360
final choices. All three fits improved the measured validation loss, from
1.41824/1.41782/1.41942 to 1.40030/1.39865/1.39824; all selected their final
update 1784. Inactivity and failure to improve that criterion are weak leading
explanations. Final-checkpoint selection leaves further loss optimization open,
but supplies no evidence that more optimization would improve planning.

Earlier winner imitation also failed to establish a second useful generation,
and the ranking pilot changed proposals without useful independently assessed
decision improvement. Its fixed fourth proposals had better descriptive values
on selection worlds, but that did not become better final decisions. This makes
the relationship between a proposal and the existing candidates important.
It does not uniquely identify redundant proposals, noise, extrapolation or
continuation bias as the cause.

Preserve the positive first learned-proposal generation: learned128 scored
65.796% against original128. Learning can improve this search. What remains
unestablished is a repeatable route for subsequent useful improvement.

## Why the proposed target is materially different

Deployment retains three common candidates: the original035 top two and its
first random alternative. The learned model supplies one additional candidate
outside that set. Overall-winner imitation does not directly teach which of
the remaining moves is best when its teacher winner is already in the common
three. It still provides indirect gradients; it is not zero supervision.

For example, if A, B and C are already available and the teacher prefers A to
everything, learning to favor A can improve ordinary imitation. The deployed
proposer must nevertheless decide whether D, E or F is the useful extra choice.
The proposed target explicitly compares those remaining observed alternatives.
The unchanged search still decides whether to play any of them.

The earlier ranking pilot did address alternatives, so this is not a claim that
all earlier training ignored them. Its four-candidate records contained only
ONE move outside the actual first three. Consequently it trained the ordering
of the two actions outside the TOP TWO. Its protocol explicitly acknowledged
that limitation. The new eight-candidate records now provide up to FIVE measured
alternatives outside the actual first three. Exact role-specific supervision
was degenerate on the older support and is now possible.

A bounded read of the already-used development validation records found 1,414
of 2,489 new breadth rows (56.8%) with the teacher winner in the common first
three. There were 2,291 rows with at least two outsiders and 2,034 with all eight
candidates. These describe this validation partition, not a fresh estimate of
all gameplay or proof of a causal bottleneck. The checked source is
outputs/proposal_breadth_learning_128/fits/fit_161001/broad/validation_0000.json,
SHA256 33e8b23ed73da456ab517d3a12c1d91ebc7848199814b6a5cfcda7ccbda2ece4.

## One controlled learning pilot

Start one new specialist fit from each original161 initializer, retaining all
three fit lineages. Use the exact latest corpus, lawful inputs, canonical group
representatives and whole-source-pair splits: 23,936 historical rows plus 11,950
breadth rows. Do not add the latest evaluation panel as training data.

On historical rows retain the inherited offered-group winner cross-entropy.
On each new breadth row let C be its exact recorded first three candidates
(or the complete support when smaller), and U its recorded candidates outside
C. The specialist loss is cross-entropy over U, targeting the first maximum
recorded 128-world terminal-score mean within U. Use the established maximum
concrete-action logit for each canonical group and original candidate tie order.

For empty or singleton U, define the new-row specialist loss as zero. Keep the
row in its original batch and exposure accounting; do not skip optimizer steps.
Keep zero rows in batch denominators. An all-zero batch needs a differentiable
zero that preserves the prescribed optimizer, momentum and weight-decay semantics.
Do not invent comparisons for absent actions, assign them zero value, filter
rows by outcomes or margins, or oversample rows where an outsider wins overall.
The target is a noisy best-observed-outsider label, not an optimal action or an
exact marginal-utility estimate. It also ranks outsiders when all seem worse
than the retained candidates; search continues to provide the final safeguard.

Keep the latest fixed dose: four passes, batch64, AdamW learning rate1e-5,
weight decay1e-4, gradient clipping1, fresh optimizer, frozen value head,
deterministic float32 CPU. Preserve the exact per-fit batch streams and checkpoint
opportunities: 1,784 updates and 114,032 training-row exposures per fit. No loss
weight, temperature, architecture, optimizer or dose sweep accompanies this test.

The matched learning control is the completed broader overall-winner training
trajectory. Reuse its saved checkpoints after proving data, initializer, batches,
optimizer, exposure and dependency identities; do not train those models again.
The unchanged original161 models remain the practical reference. This comparison
tests the new support-and-target objective as a package, not target labels alone.
Scalar loss size need not be equal for the two objectives.

Select BOTH learning families using one prospectively fixed validation criterion:
historical CE on old validation rows plus outsider CE on new validation rows,
mean within each whole source pair and then equally across pairs. Include all
rows with the same zero/singleton convention. Use the inherited checkpoint
schedule, the initial checkpoint and earliest strict minimum. Independently
recompute this criterion for every reused control checkpoint and every new
specialist checkpoint. Do not compare specialist-selected models against a
control selected by a different criterion.

This common new criterion may select different control checkpoints from the
closed pilot. Record both identities. It does not rewrite that pilot's result
or claim a new result for its original selections. Keep all three fits and seal
the new six selected models plus the three anchors before fresh testing.
Development losses, proposal agreement and support coverage are diagnostics;
they cannot substitute for the actual planner endpoint.

## Fresh evaluation and decision rule

Use a new fixed panel of 360 source-game-disjoint positions, 120 per original
fit and 20 per fit by acting-seat by phase cell. Preserve the existing fast161
source construction, phase boundaries, private-reserve cases and short supports,
with fresh source seeds and world namespaces. Freeze the 1,080-game main source
pool and separate 54-game probe pool, identity sampling and shortage response.
This population has previously demonstrated useful breadth headroom, although
that does not guarantee the same effect on every new panel or establish the
distribution reached by the strongest planner.

At each position keep the original three proposals unchanged. The anchor,
specialist and matched overall-winner control each propose their highest-scored
remaining legal canonical group. With at most three legal groups, every arm
retains the complete original support without an empty-set argmax.
Evaluate the deduplicated union, at most six
actions, on 128 common lawful selection worlds with unchanged greedy035 terminal
continuations. Each arm chooses from its own up-to-four support with original
ties. Lock all choices before assessing only their distinct selected actions,
at most three, on 128 disjoint assessment worlds.

The primary endpoint is specialist-minus-anchor independently assessed terminal
score. The attribution endpoint is specialist-minus-matched-control. Report
control-minus-anchor and per-fit summaries descriptively. Use the established
50,000 paired bootstrap resamples within the 18 fixed cells, equally weighting
source games and retaining all arms together, with a new frozen bootstrap seed.
Keep the same practical investment rule:

- Specialist minus anchor mean at least +0.5 percentage points and paired95%
  lower endpoint above zero.
- Specialist minus matched control paired95% lower endpoint above zero.
- Every prescribed integrity and operational condition passes.

This is a compute-allocation rule, not a power guarantee or a predicted arena
win rate. A pass supports proposing fresh complete-game confirmation. A gain
over anchor without the attribution contrast leaves the source of improvement
unresolved. A miss closes this fixed specialist recipe; do not add positions,
choose a fit or phase, or move to another nearby target or dose automatically.
Another null would materially reduce the priority of this local imitation family
and justify reconsidering a different mechanism. It would not falsify learning.

## Competing recommendations and remaining risks

The strongest diagnostic alternative uses the already trained broader learner
without changing weights: restrict its fourth proposal to the five outsiders
in the original eight-candidate mixture, compare with unrestricted deployment
and the unchanged anchor, and retain the eight-candidate teacher as a positive
control. The original anchor's fourth is already in that shortlist. A direct
restricted-minus-unrestricted gain would establish the practical value of that
restriction and implicate the action-support change. The independent reviewer
favoring this study reasonably emphasized repeated failures of improved
validation fit to improve planning.

I give the specialist pilot priority because the new corpus directly enables
the previously unavailable target, fitting is inexpensive, and the hypothesis
addresses the desired repeatable learning loop. The diagnostic could be equally
valuable if unrestricted extrapolation is the main problem, but neither its
success nor its failure uniquely establishes the right training intervention.
This is a judgment under uncertainty, not consensus that objective mismatch is
proven. Do not add that diagnostic as an automatic extra stage to this pilot.

The strongest search alternative is a fixed-budget breadth allocator: evaluate
eight candidates on32 worlds, retain four, give those64 additional worlds and
choose using their96 total outcomes. At full support this spends512 continuations,
matching four times128. Its early-elimination and lower-precision risks differ
from the learning question. The old halving screen's +0.1088pp result was a small
development result at another budget. A fresh bounded screen could be useful,
but no current evidence guarantees this schedule. No multi-day arena is earned
for either alternative yet.

Independent strong opposition remains necessary before broader-strength or
superhuman claims. The existing related fast-opponent panel is not an established
unsaturated benchmark for these planners. Designing that benchmark matters, but
a tournament alone would not resolve this learning-transfer mechanism. Larger
models, more diverse data and new continuation/value estimators remain open;
the accumulated nulls constrain their tested recipes, not the entire classes.

Even a successful specialist can overfit noisy 128-world winners, extrapolate
poorly to unoffered legal moves, or exploit greedy035 continuation errors.
Explicit outsider ranking does not optimize the exact expected value of the
noisy final selection process. The first three vary with a random proposal,
while the scorer is not newly conditioned on that set. These limitations are
why actual selected-move assessment and later gameplay remain necessary.

Earlier134's 31.875% gameplay result and later139 warn against treating conditional
gains as strength certificates.137 constrained its root-refill stratification
intervention; it did not disprove broader chance search. Likewise,158/159 did
not establish that a stronger direct policy is a better continuation evaluator,
and142-153 did not rule out all useful value or representation learning.

Research on sampled-action policy improvement supports distinguishing learning
over sampled comparisons from behavior over the full action space
([Hubert et al., 2021](https://proceedings.mlr.press/v139/hubert21a.html)).
That is context for the unresolved support issue, not evidence that this
Splendor intervention will work or an imported convergence guarantee.

## Cost and verification

The expensive teacher outcomes already exist. The last pilot completed all six
fits, independent reproductions, sources, 360 decisions and verification in
2.0455 hours. Its main collection took74.94 minutes and sampled audit7.85.
This design needs three new fits, new target checks and common validation of
saved control checkpoints. Expect roughly2-4 hours of execution, predominantly
CPU, after implementation/preflight; this remains an extrapolation.

Propose the existing eight-hour execution ceiling, five-GiB free-space reserve
and separate18-position outcome-blind runtime/integrity probe. Forecast all
remaining work and verification before main collection; a cost shortfall is a
hold, not permission to extend the deadline. Main continuation ceiling is
360 times128 times(6+3) =414,720 before deduplication; probe adds at most20,736.

Follow docs/engineering/verification_policy.md. Check every source decision,
exported target, split, batch identity, proposal, budget, choice and statistic.
Independently reproduce the three new training trajectories and new validation
selection. Inherit the sealed old fit reproductions after identity checks;
do not rerun them. Fully replay the bounded probe and prescribed stress roots,
and use frozen sampled continuation replay elsewhere on the mature engine.
Do not repeat full historical audits or all new continuations by default.

If authorized later, implement in new files outside frozen inventories and run
with hidden workers, durable logs and no recurring monitor. No launch, promotion,
full-game arena, automatic successor or extension is authorized by this note.

## Local evidence

- outputs/proposal_breadth_learning_128/{report.md,result.json,final_audit.json,
  learning_complete.json,fits/}; docs/engineering/proposal_breadth_learning_128.md
- outputs/proposal_breadth_arena_128/report.md; outputs/proposal_breadth_128/report.md
- outputs/proposal_ranking_128/report.md; docs/engineering/proposal_ranking_128.md
- outputs/proposal_iteration_128/report.md; outputs/proposal_diagnostic_128/report.md
- outputs/learned_proposal_128/report.md
- docs/strategy/next_after_proposal_breadth_arena_128.md
- docs/strategy/next_after_proposal_ranking_128.md; next_after_proposal_diagnostic_128.md
- docs/strategy/next_after_generalization_panel.md
- docs/strategy/independent_superhuman_review_after_134_137.md

Only this new recommendation and unbound handoff notes are changed by the review.
