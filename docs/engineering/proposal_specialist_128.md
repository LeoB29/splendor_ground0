# Learning the additional fourth proposal

Owner authorized implementation and execution on 2026-10-02 following
docs/strategy/next_after_proposal_breadth_learning_128.md. This protocol covers
one bounded pilot in tools/proposal_specialist_128, entered through
run_proposal_specialist_128.py, with outputs under outputs/proposal_specialist_128.
Preserve all dirty work and frozen predecessors. No automatic full-game arena,
promotion, training successor, extension or recurring monitor follows.

## Question and comparison

Does explicitly learning the best observed alternative outside the planner's
retained first THREE proposals improve actual four-candidate, 128-world planning?
The roles are anchor, specialist and control. Retain original161 initializers
161001/update563, 161002/update488 and 161003/update563. Train exactly three new
specialist trajectories. Reuse the completed broader overall-winner learning
trajectories as the matched control; never retrain them or select a best fit.

This tests a support-and-target objective as a package. It does not isolate
target labels alone, establish that objective mismatch caused previous nulls,
or add architecture, value, continuation-policy or deployed-budget changes.
The original eight-candidate teacher's gameplay gain remains a separate result.

## Closed corpus and inherited control evidence

Reuse the exact sealed 35,886-row corpus from outputs/proposal_breadth_learning_128,
binding bf916c18fc25e5be4bb2b7ff13317157d155da82668ae4d8a86308544b5ca72a.
It comprises 23,936 historical rows and 11,950 candidate-side breadth-arena main
decisions. There are 28,508 training rows and 7,378 validation rows. Do not add
the completed learning pilot's evaluation positions, probe records or reference
side decisions. Preserve every row, lawful full-legal input, canonical group
representative, candidate order, source identity, split and exposure slot.

Historical targets, supports and splits remain unchanged. New breadth source
pairs retain the completed pilot's split: digest ranking using
proposal-breadth-learning-128-source-pair-split-v1, with 14 validation and 53
training pairs from each source fit's 67 pairs. Pool the same 42 validation and
159 training pairs for all fits. Both seat-reversed games and every turn stay
together. Source-corpus pair namespaces prevent collisions with historical rows.
No margin, outcome, disagreement, outside-winner or support-size filtering.

The retained historical source is outputs/proposal_iteration_128, binding
1a3c53142516a5d8f08376d0a7227ae9199952a9e077e1309d13fdcdf6d40ace. The breadth
teacher source is outputs/proposal_breadth_arena_128, binding
a8395f535998379a2062ebce8fed79cc6a26e5cd1deb699c96bd4b9dca06deb5. Its offered
action-by-world records already provide the 128-world means needed here.
Verify closed completion/audit receipts, frozen dependency identities, every
linked data shard and row/target arithmetic on reuse. Inherit their sealed
source trajectory verification; no old rollout or optimizer replay is added.

For each control fit, prove identity of its initializer, exact corpus and split,
training-row order, full-legal representations, checkpoint opportunities,
training seed and batch stream, optimizer configuration, dose, frozen value
parameters and dependencies. Verify all reused checkpoint hashes and the linked
closed independent fit-reproduction receipt. Preserve the old selected update
as provenance. Its new selection may differ; do not overwrite or reinterpret
the completed pilot's selection or result.

## Specialist objective and exact exposure

For each new breadth row, C is its original first min(3, support size)
candidates, and U is the remaining recorded support in original candidate order.
Target the first maximum recorded 128-world terminal WDL mean within U.
The monotone score transform (WDL+1)/2 does not change this target. Compute each
canonical group's score as the maximum concrete-action logit in that group,
then cross-entropy only over U. Historical rows retain their inherited offered
support and overall-winner CE. Unoffered actions receive no measured negative
label or invented value.

For empty or singleton U, new-row loss is exactly zero. Preserve that row's
batch position and denominator, every optimizer step and all exposure counts.
The zero is differentiable so an all-zero batch retains prescribed optimizer
momentum and weight-decay semantics. Never drop short-support rows, oversample
outsider wins, reweight losses or skip an optimizer step because loss is zero.
Targets are noisy best-observed-outsider labels, not optimal moves or exact
marginal utility. Outsiders are compared even when all appear worse than C;
the unchanged search still chooses among its complete deployed support.

Use four passes, batch64 including the last short batch, fresh AdamW with
learning rate1e-5 and weight decay1e-4, gradient norm clipping1, float32
deterministic CPU, one Torch thread and a frozen value head. Keep the original
training seeds 212700001, 212700002 and 212700003 for fits161001-161003.
Do NOT replace these with the new experiment's source/world seed namespace.

In preserved training-corpus order, shuffle indices with Python Random seeded
by proposal-breadth-learning-128-batches-v1|training_seed|epoch, stably sort by
feature length, divide into consecutive64-row batches, then shuffle those
batches with the same generator. The specialist and reused control must have
identical per-fit sequence digests: 446 batches per pass, 1,784 updates and
114,032 row exposures each. Scalar loss magnitudes need not be equal between
objectives. No additional loss weight, temperature, model, dose or seed sweep.

Checkpoint opportunities include update zero, updates1/2/4/8/16/32/64 within
the dose, and epoch*B+ceil(q*B/4) for epoch0-3 and q1-4, where B=446. Deduplicate
and sort. Finish all prescribed steps even if an earlier checkpoint is best.

## Common new validation and model seal

Select BOTH learning families using the same new criterion: historical offered
winner CE on old validation rows and outsider CE on new validation rows, with
the same empty/singleton zero convention. Include every validation row; average
within whole source pair, then equally across pairs. Select the earliest strict
minimum over every checkpoint opportunity, including initialization. This is
development model selection, not evidence of fresh planner improvement.

Independently reproduce the three new specialist trajectories, every saved
checkpoint, frozen value parameters and validation calculation. Independently
recompute common new validation at every reused control checkpoint and reproduce
each new control selection. Inherit the already completed control optimizer
reproductions after identity verification; do not train those trajectories
again. Independent selection must not merely accept stored specialist labels
or collection-side validation records. Record each control's old and new
selected-update identities.

Seal six newly selected learning models plus three unchanged anchors before
fresh source collection or evaluation. Reload model kind, configuration, tensor
keys and tensors when checking identity with initialization. Private inference
registries must not mutate frozen ancestors' model or output globals.

After every prescribed fit and selection audit, an exact identity exception
may close the pilot only if ALL THREE selected specialist models are exactly
their anchors in kind, configuration, tensor keys and every tensor. The same
deterministic deployed rule then implies zero specialist-minus-anchor effect.
Fail the practical gate and skip fresh source/evaluation; report no empirical
interval and no measured fresh specialist-minus-control contrast. Similar
validation losses, proposals or observed actions do not establish identity.
If any specialist differs, complete the fixed panel without outcome stopping.

## Fresh source population and fixed panels

Collect 1,080 fresh main source games and a separate54-game probe pool. Assign
games by identity hash using proposal-specialist-128-source-cell-v1, rotating
the18 cells ordered by original fit, acting seat0/1 and phase0/1/2:60 main or
three probe games per cell. Both actors use that cell's unchanged original161
restricted student and lawful fast-selector inference with original035
candidate construction. Assignment does not constrain actual moves or outcomes.

Environment seeds are base plus source-game index: main2135000000,
probe2136000000, isolated test2138000000. Source streams use
[proposal-specialist-128-source-stream-v1, stage, index]. Fixture overrides
are forbidden in production and cannot target production or frozen output.
Independently reconstruct every source game and actual decision. Preserve
failed games; do not replace seeds. Natural completion or the established
forced-pass deadlock is required. Invalid400-turn, fourfold-repetition or
60-turn no-progress cutoffs cause failure, not accepted outcomes.

Within each assigned cell, eligibility is a pre-decision state at the assigned
seat and phase: turns0-19,20-39,40+. Identity-rank eligible games using
[proposal-specialist-128-panel-v1, game, stage, index], choose20 main or one
probe game per cell, and identity-select one eligible turn per selected game.
Order selected states by source-game ID for dense panel indices. Main has360
distinct source games,120 per fit and20 per cell; probe has18 distinct games.
Retain private reserves and short supports. If the fixed pool lacks a cell,
write a source-shortage hold without extension, alternate seeds or fallback.

Selection is blind to winners, logits, margins, proposal disagreement, rollout
outcomes and execution times. Independently reconstruct eligibility and panel
selection from audited source records. Seal panel identities and audit samples
before rollouts. This fast161-source population is not an independent strong
opponent benchmark or the reached-state distribution of the strongest planner.

## Deployed decisions and independent assessment

Preserve the original035 top two groups and first random alternative, original
ordering, concrete representatives and source stream. Each anchor/specialist/
control adds its highest-scored remaining legal canonical group, with inherited
canonical-order ties. For at most three legal groups retain the complete
original support without an empty-set argmax. Each arm deploys up to four moves.

Build the deduplicated research union in order: common first three, anchor
fourth, specialist fourth, control fourth. There are at most six candidates.
Evaluate every union candidate on128 common observer-legal selection worlds,
using unchanged greedy035 terminal continuations. Each arm chooses its own
first-offered maximum terminal-WDL mean. Persist complete selection evidence
and a lock of all three choices BEFORE assessment simulation.

Assess only the sorted distinct locked union indices, at most three actions,
on128 disjoint fresh observer-legal worlds, with explicit sparse index mapping.
Assessment cannot change action, model, fit, support, checkpoint or sample size.
Conceptual deployment stays four candidates times128; shared research work does
not change either arm's budget. Assessment redraws hidden worlds including deck
order rather than preserving selection's hidden state.

World seed =2130000000+stage_offset+512*panel_index+block_offset+world.
Stage offsets are probe0, main1000000, isolated test2000000; block offsets are
selection0 and assessment256; world indices0-127. Verify disjoint ranges and
separation from source seeds and all earlier experiment namespaces. At most
360*128*(6+3)=414,720 main and18*128*(6+3)=20,736 probe continuations are recorded.

## Statistics and decision rule

Primary endpoint: specialist-minus-anchor independently assessed terminal score.
Required attribution: specialist-minus-control. Report control-minus-anchor
and fit/seat/phase/privacy/support/proposal summaries descriptively. The score
is simulated terminal win1/draw0.5/loss0, not a complete-game arena win rate.

Weight all360 source games equally. Use50,000 paired bootstrap resamples
within the18 fixed fit/seat/phase cells, retaining every arm together.
Seed2133000000, NumPy default_rng,50 chunks of1,000. Within each chunk visit
cells in that order, draw integers[0,20) with shape(1000,20) over panel-index
ordered rows, and use common draws for every contrast. Report linear2.5/97.5
percentiles. Independent units are source games, not worlds or candidate actions;
inference conditions on the three fixed fits and this source population.

Pass only when specialist-minus-anchor mean>=.005 and its paired95% lower bound
is>0, AND specialist-minus-control paired95% lower bound is>0, with all required
integrity and resource conditions met. For the integer/256 score grid, compare
the practical threshold exactly: summed specialist-anchor numerator*200
>=360*256. Use classification positive_conditional_specialist_contribution on
pass and practical_progress_not_established otherwise. Independent raw-record
index resampling and audit-receipt histogram reconstruction must agree within
absolute1e-12 on means and intervals, with identical gates and classification.

The+.5 percentage-point hurdle is a compute investment rule, not a power
guarantee or arena prediction. A pass supports proposing separately authorized
fresh complete-game confirmation. A gain over anchor without the attribution
contrast leaves improvement's source unresolved. A miss closes this recipe;
do not add samples, choose a phase or fit, change targets/dose or launch a nearby
study automatically. It would reduce priority for this local imitation family
without proving no learnable teacher signal, all learning impossible, or exact
zero gain. Continued extrapolation and greedy035 continuation bias remain open.

## Verification and immutable evidence

Apply docs/engineering/verification_policy.md. Check every inherited/new data
identity, target, partition, exposure, checkpoint and model selection; every
new source move; and every actual evaluation proposal, lawful input, canonical
member, world seed, outcome record, aggregate, budget, sparse mapping, choice,
score and stage link. Independent evaluation reconstructs source states and
decision arithmetic without the collection decision helper. Mature rules,
model and continuation primitives remain shared trusted dependencies.

Fully replay every continuation at all18 probe states. Main: at every state
replay all evaluated candidates in eight of128 identity-selected worlds in
EACH block, plus all worlds at the identity-minimum state in each fit/seat/phase
cell(18), fit/privacy category(12) and fit/canonical-support1/2/3 cell(9).
Deduplicate overlap and report absent cells without replacements. At most39
main states receive full stress replay. Seed2134000000, tags
proposal-specialist-128-audit-world-v1 and proposal-specialist-128-audit-stress-v1.
Freeze selected identities and counts before collection.

Every unreplayed continuation receives structural/outcome-arithmetic checks;
this does not independently verify its trajectory. Reconcile every root and
count with the sealed sample. Check selection evidence and lock precede
assessment. Closure verifies model/artifact hashes, links, counts and independent
statistics without another historical or all-main continuation replay. Reports
must explicitly distinguish sampled main replay from fully replayed probe and
stress roots. A mismatch blocks acceptance; preserve evidence and do not sample
until success or retry production automatically.

Preflight covers exact outsider masks/targets and ties, empty/singleton loss
and all-zero optimizer behavior, historical target preservation, source-pair
identity, exact inherited streams/dose, every common checkpoint selection,
specialist reproduction and inherited control closure, exact identity exception,
lawful hidden inputs, source selection, sparse assessment and locks, corruption,
statistics/gates, finalization, resource holds and actual hidden workers.
Freeze current passing checks, all new code/configuration, this protocol,
accepted strategy, runtime/source/dependency identities and model hashes before
production preparation. Never edit a frozen implementation or predecessor.

## Hidden execution, resources and failure handling

Use one detached hidden supervisor, at most three fitting/reproduction workers
and eight source/evaluation/audit workers, one deterministic Torch thread each,
and the mature serial continuation backend. Windows children use the matching
windowless executable with durable logs, health checks and launch acknowledgement.
No visible terminal or active chat is required. Scoped system-sleep inhibition
permits display sleep and ends with the supervisor; shutdown interrupts work.
No automatic retry/restart, extension or recurring monitor.

Order: closed evidence/data/control identity checks and new targets; three
specialist fits and common control validation; independent new fits and all new
selections; model seal/identity exception; fixed source pools and source audits;
panel/audit seals;18-root collection/full replay; outcome-blind cost gate;
360-root collection/prescribed audit; independent final statistics and closure.

The eight-hour total execution ceiling starts at the original launch.json epoch
and includes preparation, validation/reproduction, sources, probe, main and
closure. Implementation/preflight is additional. Maintain five GiB free space.
Check budgets at data, optimizer/validation, source-move and rollout-world
boundaries, job completions and finalization. An active trajectory may finish
before its next check; the probe cannot extend the deadline.

Before main, forecast selection/assessment using delivered probe continuation
timings plus structural root cost, scaled to maximum276,480 selection and
138,240 assessment continuations. Forecast audit blocks from measured replay
rates, the sealed main sample and maximum six/three supports, including structural
root cost. Divide worker-seconds by eight, add35% allowance and300seconds for
closure, and include elapsed launch time. Require total<=eight hours. Forecast
storage using probe bytes per recorded continuation at main maximum, add50%
allowance plus512MiB and retain five-GiB reserve. No strength outcome enters
this operational decision.

Resource expiry or source shortage writes an operational hold; integrity or
worker failure writes failure evidence. Cancel pending work and preserve every
completed and partial strict artifact. Advisory status contention uses the
inherited resilient reporter and does not invalidate successful scientific
work. A complete run, including a failed scientific gate, writes final result,
audit, report and status. Nothing further launches automatically.
