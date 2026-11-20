# Next after the completed proposal diagnostic

2026-09-27. Recommendation only, requested after the diagnostic finished.
AGENT_CURRENT.md was read first. Three independent read-only reviews considered
learning, search alternatives and statistical design. No new fit, model inference,
simulation, test, source collection or experiment was started. Preserve all dirty
work and frozen evidence. This is a new unbound strategy note.

## Decision

Test one **gap-weighted ranking update for the alternative-proposal role**, using
the full rollout outcomes already recorded. Compare it with matched winner-label
training and the unchanged useful proposer. Evaluate actual planner decisions on
one fresh, bounded panel before committing to another multi-day strength arena.

Confidence is moderate. This is the best next intervention, not a diagnosis that
the loss is definitely responsible. Candidate discovery is the closest rival.
Do not turn a null result into a succession of nearby loss/temperature/LR sweeps.

## What the completed evidence changes

The previous full-game update scored 51.4925%, paired 95% interval
[47.2637,55.7214], against the unchanged learned proposer in 128-world planning.
It did not establish the required progress. The new diagnostic now finds:

| Contrast | Conditional score gain, pp | Paired 95% interval, pp |
|---|---:|---:|
| Latest update minus anchor | +0.1419 | [-0.1632,+0.4800] |
| Latest update minus replay update | +0.3770 | [-0.1749,+1.2943] |
| Random fourth minus anchor | -0.9970 | [-2.1708,-0.0583] |
| First three only minus anchor | -1.5644 | [-2.7752,-0.6102] |
| Larger union minus anchor | +0.3090 | [-0.0175,+0.6860] |

These are selected-move terminal scores under greedy035 continuations, not arena
win rates. Only the first contrast was primary; the others remain diagnostic.
The independent assessment, prescribed sampled replay and final closure passed.

The treatment changes its fourth proposal in 45/188 eligible states, and changes
the final move in 18/201 states. Nine percent decision movement is not inherently
too low: a few excellent changes can matter. What matters is the small, uncertain
assessed benefit, combined with the preceding inconclusive arena. Neither result
proves equivalence, harm, universal saturation or inability to improve iteratively.

The anchor's learned fourth remains useful. The original learned128 planner's
65.7960% result against original128 still establishes one productive
search-to-learning-to-stronger-search cycle. The new evidence weakens repeating
the same winner-imitation update as the default way to obtain the next cycle.

Better offered-choice cross-entropy is not sufficient evidence for better
proposals. All three latest treatments improved that criterion on old and new
validation data; useful fourth recognition improved too. Failure to fit the
measured loss, gross measured forgetting, an unchanged model and blanket
suppression of fourth moves are weak leading explanations. More fitting or a
larger network has no comparably specific mechanism justification yet.

## The information currently being discarded

At each four-candidate root we already simulated the original035 top two and
two alternatives. Winner-only training says which of all four came first. If
one of the top two wins, it does not explicitly preserve which alternative is
better, or whether their difference is negligible or large. It still provides
indirect relative learning; the claim is missing explicit information, not zero
learning about alternatives.

The proposed update learns that ordering directly. This is especially relevant
because deployment asks the model for an extra legal candidate after three
common candidates have been excluded. It does not ask the model to choose the
already-recorded overall winner from four offered moves.

Important distinction: train between the TWO recorded actions outside the
original top TWO, namely offered slots 2 and 3 (zero-based). There is only ONE
recorded action outside the actual first THREE; a restricted loss there would be
degenerate. The new objective is therefore closer to the proposal role, not an
exact solution to marginal complementarity or all-legal candidate discovery.

## One fixed learning comparison

Use the existing combined 23,936 rows (19,047 training; 4,889 validation), with
the existing whole-source-pair splits. Obtain full outcome differences from their
sealed source roots; old tensor exports alone contain only winner labels. Use
the original three selected161 models as paired initializers. Do not add the
latest arena/diagnostic states to training or start from the latest update.

For a support-four row, let d be mean terminal score of slot2 minus slot3 over
the same recorded128 worlds, where win=1, draw=0.5, loss=0. Let z2 and z3 be the
unchanged maximum-over-concrete-actions canonical group logits. Proposed loss:

    L = abs(d) / M * softplus(-sign(d) * (z2 - z3))

M is the mean abs(d) over the fixed TRAINING rows, assigning zero to rows with
fewer than four candidates. Compute it once before fits and require M>0. This
normalization avoids accidentally shrinking the overall objective scale solely
because empirical score differences are small. It does not make gradients or
effective information identical to cross-entropy.

Exact ties and short supports contribute zero ranking loss. Keep their rows,
batch accounting and optimizer-step schedule. Do not treat unoffered actions as
negative examples. No gap threshold, uncertainty filter, temperature search,
extra anchoring term, architecture change or new continuation policy.

Keep four passes, the existing batch streams and checkpoint schedule, fresh
AdamW state, learning rate 1e-5, weight decay 1e-4, gradient clip1, architecture
and frozen value-head parameters. Retain all three fits and initial checkpoints;
never choose the strongest fit after observing outcomes.

The matched control can reuse the COMPLETE saved treatment/CE trajectories from
proposal_iteration_128: their corpus, initializers, optimizer, batch streams and
exposure match this counterfactual. Verify that identity before reuse. Train
three new ranking trajectories and independently reproduce those new fits;
there is no need to regenerate the matching CE trajectories merely to use them.

Select checkpoints for BOTH families on the same whole-pair-mean validation
ranking loss, using training-derived M and earliest strict minimum. The old
selected CE checkpoint is not automatically the matched control: changing the
selection criterion may select a different saved checkpoint. This compares the
two training objectives under a common role-oriented selection rule. The untouched
161 proposers remain the practical anchor. Validation is reused development
material, not strength evidence; no model is selected from the fresh test.

Gap weighting gives tiny empirical differences less influence, but it does not
remove rollout noise or provide calibrated confidence. The broader rationale
of weighting policy mistakes by action-value differences has precedent in
[Lazaric, Ghavamzadeh and Munos, JMLR2016](https://www.jmlr.org/beta/papers/v17/10-364.html).
Its guarantees do not transfer automatically to our partial-support, noisy,
imperfect-information proposer. Earlier052/053 and149-153 failures also warn
against treating a new target as a guaranteed remedy.

## One fresh decision test, with a fixed stopping rule

Recommend 360 main positions from360 distinct new source-game seeds:120 per
frozen161 source fit,20 per fit x acting-seat x phase cell. Phases remain0-19,
20-39,40+. Use cheap, unchanged161 restricted-policy self-play to produce source
games, then assign/select roots by identities only. This is a test on student-
generated positions, not the exact stronger-planner state distribution.

Generate a fixed pool with a declared cap, for example1,080 source games
prospectively assigned across the18 cells, and select20 eligible games per cell
by identity. Each selected game contributes one root. Include private reserves,
small supports, agreements, wins and losses. Some fast games may end before
turn40; freeze eligibility/shortage handling before collection and hold if a
cell is short. Never extend the pool based on outcomes or select disagreements.

The201 inspected diagnostic positions are development evidence. Fresh worlds
at those same positions would not constitute fresh-position generalization.
Lock all models, exact source/world namespaces and the complete test plan before
new evaluation outcomes. A separate18-state timing/integrity probe, one per cell,
must use disjoint source seeds and cannot tune scientific choices.

At every root retain the same common first three, concrete representatives,
observer-information contract, canonical scoring and tie rules. Compare only
three fourth proposers: new ranking update, matched CE update, untouched161.
Their deduplicated union has at most SIX actions. Each conceptual arm uses its
unchanged four-candidate128-world decision rule.

Evaluate the union on128 shared fresh selection worlds. Lock each arm's choice
before assessment. Evaluate only the distinct LOCKED choices (at most three)
on128 disjoint fresh assessment worlds. This is enough for the stated decision
endpoint and avoids spending additional simulations on every unchosen fourth.
Do not add random/union arms or another post-hoc mechanism study to this test.

Primary progress comparison: independently assessed ranking-minus-anchor score.
Required attribution comparison: ranking-minus-matched-CE score. Average all
360 source games equally. A fixed50,000-replicate paired bootstrap resamples
whole source games within the18 fixed cells, keeping arms together. These
intervals condition on the three fits; they are not optimizer-seed population
claims. Report phase/fit/seat/privacy and movement descriptively, not for rescue.

Proposed compute-allocation gate, frozen before fits: ranking-minus-anchor point
gain at least +0.5 percentage points AND its95% lower endpoint above zero, plus
ranking-minus-CE95% lower endpoint above zero. Both comparisons are required.
The0.5pp hurdle is a practical investment choice; it neither predicts55% arena
strength nor excludes valuable smaller effects. A strict cheap gate deliberately
risks missing a small real improvement. No precision guarantee follows from360.

If the gate fails or is inconclusive, close this recipe and retain the anchor.
No additional states, epochs, checkpoints or loss variants to seek a pass. If it
passes, the next justified investment is a separately specified fresh128-world
full-game comparison against the anchor, with matched-control attribution also
planned. A conditional screen alone does not establish stronger play. This
recommendation's initial bounded pilot does not authorize that multi-day arena.

## Cost and verification

No new teacher TRAINING rollouts are needed. Training itself previously took
roughly4-6minutes per comparable combined-corpus fit, though export, checkpoint
scoring, reproducibility and contention add work. This is not a promise of the
new fit duration. Simulation for fresh EVALUATION remains necessary and CPU-bound.

Main ceiling:360 *128 *(6 selection candidates +3 assessed choices)=414,720
continuations. Probe ceiling20,736. The last diagnostic completed253,952 main
and15,872 probe continuations plus prescribed audits in1h53. A rough expectation
for the new pilot is several hours (provisionally3-6h execution), not three days;
source distribution, inference/reproduction and trajectory lengths can differ.

Propose an8h total execution ceiling, including data preparation, fits,
reproduction, source collection, probe, main collection and verification.
Implementation/preflight is additional. Use an outcome-blind runtime/storage
forecast before main collection; hold if it does not fit. No gate is changed
because of promising early outcomes. Run hidden independent CPU workers without
monitoring if later authorized.

Follow the accepted verification policy. Independently rebuild all training
targets, splits and control identities; reproduce the new training and common
checkpoint selection; verify every source state/proposal/choice and independence
boundary. Use prescribed sampled replay for mature rollout paths with fixed full
stress cases, not an automatic full replay of all existing teacher simulations.
Freeze exact coverage before collection. Preserve shared-primitive limitations.

## Why this beats the alternatives now

Broader challenger discovery is a CLOSE second. Random replacement's loss does
not show that adding random candidates is useless. Read-only exploratory
arithmetic found that22 of28 union-induced decision changes involve the random
addition; the union's independent gain remains uncertain and costs more. Existing
two-outsider comparisons may be too sparse or too anchored to learn better
unseen alternatives. A null ranking intervention raises this hypothesis's
priority, rather than licensing endless new objectives on the same support.

More worlds, stronger continuations, another tree architecture or a larger model
remain possible routes. They currently change less localized mechanisms or cost
more.128-versus32 sampling already helped;158/159 did not establish learned-
continuation improvement. The latest diagnostic does not identify a fresh reason
to repeat continuation replacement or imply that selection noise erased this
update's benefit. Earlier134/137 constrain their particular search recipes, not
all future search. Efficiency/batching remains an enabler, not this mechanism test.

A broader opponent panel is still required for general-strength claims, and
declared-resource strong-human evaluation eventually matters for superhuman
performance. Neither supplies the immediate learning intervention. This pilot
tests whether the demonstrated useful proposal role can be improved through a
more relevant learning signal. Its result must determine whether a full-game
investment is worthwhile; lower training loss cannot substitute for that result.

## Evidence

- outputs/proposal_diagnostic_128/{report.md,result.json,final_audit.json,final_status.json}
- outputs/proposal_iteration_128/{report.md,result.json,models_lock.json,fits/}
- tools/proposal_iteration_128/learning.py; src/splendor_ai/alphazero/fast_selector_157.py
- docs/strategy/next_after_proposal_iteration_128.md; next_during_proposal_iteration_128.md
- docs/strategy/next_after_161.md; next_after_162.md; next_after_generalization_panel.md
- docs/strategy/independent_superhuman_review_after_134_137.md
- docs/engineering/learned_proposal_screen_result.md; docs/engineering/verification_policy.md

Recommendation only. No implementation, training, inference, source games,
rollouts, scheduled monitoring or successor has been launched by this review.
