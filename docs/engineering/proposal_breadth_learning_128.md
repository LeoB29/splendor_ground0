# Controlled learning from broader search

Owner authorized implementation and execution on 2026-10-02, following
docs/strategy/next_after_proposal_breadth_arena_128.md. This protocol covers one
bounded pilot in tools/proposal_breadth_learning_128, entered through
run_proposal_breadth_learning_128.py, with outputs under
outputs/proposal_breadth_learning_128. Preserve all existing dirty work and
frozen predecessors. No automatic full-game arena, promotion or successor.

## Question and fixed learning intervention

Can the broader teacher's already recorded comparisons improve the learned
fourth proposal while keeping deployed search at four candidates and 128 worlds?
Compare broader-supervision and matched narrow-supervision updates against the
unchanged original161 anchor. Retain all three initializers, 161001/update563,
161002/update488 and 161003/update563. There are six new fits, with roles broad
and narrow for each initializer; the final inference registry also includes
the three unchanged anchors. Never select the best fit from evaluation outcomes.

This tests a broader-supervision package, including offered support and its
winner. It does not isolate the winner label from the additional comparisons.
There is no architecture, value objective, continuation-policy or search-budget
change. No unoffered action receives a measured negative label or zero value.

## Closed sources, corpus and split

Retain exactly the previous 23,936 rows from outputs/proposal_iteration_128,
binding 1a3c53142516a5d8f08376d0a7227ae9199952a9e077e1309d13fdcdf6d40ace.
These contain 11,985 original160 rows and 11,951 learned128 rows, with the
existing 19,047 training and 4,889 validation assignments and original targets.
Do not substitute ranking-pilot targets or reselect historical partitions.

Add all 11,950 candidate-side main decisions from the completed breadth arena,
outputs/proposal_breadth_arena_128, binding
a8395f535998379a2062ebce8fed79cc6a26e5cd1deb699c96bd4b9dca06deb5.
They contain 88,384 offered action estimates backed by 11,313,152 recorded
continuations. Exclude the operational probe and reference-side decisions.
Keep losses, agreements, ties, private reserves and short supports. Do not
filter on whether additions win, outcome magnitude or learner disagreement.
The combined corpus contains 35,886 rows before its fixed training/validation
partition; optimizer fits use only the training partition.

At every new row, broad support is the original ordered up-to-eight candidate
list. Narrow support is its exact original up-to-four anchor prefix, including
concrete representatives. Both targets are the first maximum mean terminal WDL
within their respective supports over the same 128 recorded worlds. The score
transform (WDL+1)/2 preserves this ranking. A new maximum is a training label,
not an independently assessed action value or optimal-play claim.

Within each SOURCE fit's 67 environment-seed pairs, rank pair IDs by the digest
of [proposal-breadth-learning-128-source-pair-split-v1, source_fit, pair]. Assign
the first 14 pairs to validation and the other 53 to training. Pool the same
159 new training and 42 new validation pairs for all six fits. Both reversed-seat
games and every turn of a pair stay together. Namespace pair identifiers by
source corpus so historical and new pair numbers cannot collide. Preserve every
historical split and source row order. Seal row identities, source hashes,
partitions, counts, representations and both labels before fitting.

Bind closed source completion/audit receipts, source game/root links, original
model hashes and the frozen035 continuation model. Check each source root on
use. Independent export checks reconstruct actual source states, lawful full-legal
inputs, canonical membership and the complete action-by-world target arithmetic.
Inherited prescribed trajectory audits remain the simulation evidence; do not
rerun old rollouts or old optimizer trajectories merely to read their data.

## Identical exposure and common checkpoint selection

Both families use full-legal model inputs and maximum concrete-action logits
within canonical groups. Their offered-group cross-entropy differs only in
the new row's support and corresponding winner: broad compares up to eight,
narrow compares its original up to four. Historical rows retain their original
support and label for both families. Narrow training does not insert the extra
groups into its softmax and penalize actions its teacher did not consider.

Four complete training passes, batch size 64 including the last short batch,
fresh AdamW, learning rate 1e-5, weight decay 1e-4, gradient clipping at 1,
float32 deterministic CPU and one Torch thread. Freeze value-head parameters.
Use training seeds 212700001, 212700002 and 212700003 for the corresponding
original161 initializers. Begin with training-row indices in preserved corpus
order. Shuffle with Python Random seeded by
proposal-breadth-learning-128-batches-v1|training_seed|epoch, stably sort by
feature length, partition into consecutive batches, then shuffle the batches
with the same generator. This retains the inherited feature-length bucketing.
Broad and narrow share the exact sequence and batch slots for a given seed. No
oversampling, replacement, gradient reweighting, seed selection or dose sweep.

The fixed checkpoint schedule includes update zero, updates 1, 2, 4, 8, 16,
32 and 64 when within the dose, and the inherited quarter-epoch endpoints:
epoch*B + ceil(q*B/4), for epochs 0-3, q=1,2,3,4 and B=ceil(training_rows/64).
Deduplicate and sort through the final update. Both families use the same
schedule. Finish every prescribed optimizer step even if an earlier checkpoint
currently has the best validation score.

Checkpoint selection uses the mean of source-pair mean offered-group winner
cross-entropies, with every validation pair weighted equally. On new validation
rows, BOTH families are judged on the full broader support and broader winner;
historical validation rows retain their original support/target. Choose the
earliest strict minimum, including update zero. This common development
criterion cannot substitute for fresh planner utility. No fresh source or
evaluation outcome participates in checkpoint selection.

Independently reproduce all six fits, every saved checkpoint and its common
validation calculation, selected/final tensors and frozen value parameters.
Seal all nine inference models with selected-update and initializer-equality
metadata before fresh source collection or testing. Private registries must
not mutate the model/output globals of frozen ancestors.

The prospective exact-identity exception applies only after all six fit audits:
if ALL THREE selected broad models reload with exactly their anchor's model
kind, configuration, tensor keys and every tensor, the unchanged deterministic
proposal/selection rule implies zero broad-minus-anchor effect. Close with
the progress gate failed, skip fresh source/evaluation stages and report that
no empirical interval or fresh broad-minus-narrow contrast was measured. If
any broad model differs, run the complete fixed panel without outcome-based
early stopping. Do not infer equality from similar validation scores or actions.

## Fresh source population and selected positions

Use 1,080 new main source games and a separate 54-game probe pool. Hash-rank
game IDs using proposal-breadth-learning-128-source-cell-v1, then rotate the
18 cells ordered by source fit, acting seat 0/1 and phase 0/1/2. This allocates
60 main and three probe games per cell. Both source actors use that cell's
unchanged original161 restricted student, lawful fast-selector inference and
the original035 candidate construction. Source assignments do not restrict
the actual game's moves or outcomes.

Environment seed bases plus source-game index: main 2125000000, probe
2126000000, isolated test 2128000000. Source streams are digests of
[proposal-breadth-learning-128-source-stream-v1, stage, index]. Fixture snapshot
overrides are forbidden in production and cannot target production output.
All source games and actual decisions are independently reconstructed; invalid
400-turn, fourfold-repetition or 60-turn no-progress cutoffs are not accepted
results. Only natural completion or the established forced-pass deadlock is
valid. A failed game is preserved, not replaced with another seed.

Within its assigned cell, an eligible game contains a pre-decision state for
the assigned seat and phase, turns 0-19, 20-39 or 40+. Hash-rank eligible games
with [proposal-breadth-learning-128-panel-v1, game, stage, index], choose 20 main
or one probe game per cell, then identity-hash-select one eligible turn. Sort
selected states by source-game ID to obtain dense panel indices. Main has 360
distinct source games, 120 per fit and 20 per cell; probe has 18 distinct games.
Keep small supports and private reserves. Log a source-shortage hold if the
fixed pool lacks a cell; no outcome-dependent pool extension or fallback.

Selection never uses winners, logits, margins, proposal disagreement or rollout
outcomes. Independently reconstruct eligibility and selection from the audited
source records. Seal selected identities and audit sampling before rollouts.
This is a fresh fast161-source population, not the full reached-state
distribution of stronger planners or an independent-opponent benchmark.

## Exact deployed comparison and assessment

At each state preserve the original035 top two proposals and first random
alternative, concrete representatives, source stream and original ordering.
For each sealed anchor/broad/narrow model, take its highest-scored remaining
canonical group as the fourth proposal, with canonical-order ties. For fewer
than four groups retain the full original support. Deduplicate the research
union ordered common first three, anchor fourth, broad fourth, narrow fourth:
at most six candidates, with each arm retaining its own up-to-four choices.

Selection evaluates every union candidate on 128 fresh observer-legal worlds,
using shared hidden-world draws, unchanged greedy035 terminal continuations,
terminal WDL and first-offered maximum ties. Persist complete selection evidence,
identity and each arm's chosen union index BEFORE any assessment simulation.

Assessment uses 128 disjoint fresh observer-legal worlds and evaluates only the
sorted unique union indices of the three locked choices, at most three actions.
Store the explicit sparse index mapping. Assessment cannot select another action,
model, fit, sample size or subgroup. Conceptual deployment remains four by 128;
research shares overlapping continuation work across arms.

World seed = 2120000000 + stage_offset + 512*panel_index + block_offset + world.
Stage offsets: probe 0, main 1000000, isolated test 2000000. Selection offset 0,
assessment offset 256, world indices 0-127. Check disjoint ranges, source seed
separation and fresh namespaces. Assessment resamples hidden worlds, including
deck order, rather than preserving selection's hidden state. Main maximum
recorded continuations: 360*128*(6+3) = 414,720; probe maximum: 20,736.

## Statistics, gates and stopping interpretation

Primary: broad-minus-anchor independently assessed terminal score. Required
attribution: broad-minus-narrow. Secondary narrow-minus-anchor and fit, seat,
phase, privacy and choice-change summaries are descriptive; they cannot rescue
the primary decision or select a replacement winner. Score is win 1, draw 0.5,
loss 0 in the simulated continuation, not a whole-game arena result.

Average all 360 source games equally. Bootstrap 50,000 paired resamples within
the 18 fixed fit/seat/phase cells, keeping all arms together. Seed 2123000000,
NumPy default_rng, 50 chunks of 1,000. Within each chunk, iterate cells in that
order and draw integers [0,20) with shape (1000,20), selecting rows ordered by
dense panel index. Use common draws for every contrast and linear 2.5/97.5
percentiles. The independent units are source games, not worlds or actions;
the inference conditions on these three fixed fits and the source population.

The pilot passes only if broad-minus-anchor mean is at least .005 and its
95% lower bound exceeds zero, AND broad-minus-narrow's 95% lower bound exceeds
zero, with all integrity/resource conditions met. Score means lie on the
integer/256 grid; the practical comparison uses exact integers:
summed broad-anchor numerator * 200 >= 360*256.
Classification is positive_conditional_broader_supervision_contribution when
all gates pass, otherwise practical_progress_not_established.

Raw-record indexed resampling and independently reconstructed audit-receipt
histogram accumulation must agree within absolute 1e-12 for means/intervals,
with identical gates/classification. Report every estimate and uncertainty.
The .5 percentage-point hurdle is a compute-investment threshold, not a power
guarantee, arena prediction or proof that smaller gains are worthless.

A pass supports proposing separately authorized full-game confirmation at the
original budget. It does not establish retention of the broader teacher's whole
advantage, efficient repeated improvement or superhuman strength. A miss closes
this fixed recipe without adding states, selecting another fit, changing loss,
extending exposure or automatically launching another study.

## Verification and immutable evidence

Apply docs/engineering/verification_policy.md. Check every new source move,
exported row/target/split, selected model, lawful input, canonical membership,
proposal, world identity, outcome field, aggregate, budget, sparse mapping,
choice, score and stage link. Independently reconstruct each selected source
state and the proposal and choice arithmetic without collection decision
helpers. Source and model primitives remain mature shared dependencies.

Probe: full continuation replay at all 18 states. Main: replay all evaluated
candidates in eight of 128 identity-selected worlds in EACH block at EVERY
state, plus full replay of the identity-minimum state in each fit/seat/phase
cell (18), fit/privacy category (12), and fit/canonical-support1/2/3 cell (9).
Deduplicate overlaps, report absent cells and never replace them using outcomes.
At most 39 main full-stress roots. Audit seed 2124000000, tags
proposal-breadth-learning-128-audit-world-v1 and
proposal-breadth-learning-128-audit-stress-v1. Freeze sampling before collection.

Unreplayed trajectories receive complete structural/outcome-arithmetic checks.
Reconcile every root receipt and recorded/replayed count with the frozen sample.
Check that the saved selection lock and evidence precede assessment. Final
closure rechecks models, artifact hashes/links, counts and independent statistics
without an additional historical or full new-continuation replay. Explicitly
report sampled main replay; a stress/probe receipt may be fully replayed.

Preflight covers authentic broad/narrow masks, inherited target preservation,
ties and short supports, pooled pair splits, exact batch exposure, common
validation, all-six fit reproduction and identity exception, lawful hidden
inputs, sparse assessment, selection sealing, corrupt records, statistical
gates, finalization, resource holds and actual hidden workers. Freeze current
passing checks, all new code/configuration, this protocol, source/runtime
identities, provenance and model hashes before production preparation.

## Hidden execution, budgets and failure handling

One detached hidden supervisor; up to three fit/reproduction workers and eight
source/evaluation/audit workers, one deterministic Torch thread each. Use the
mature serial continuation backend. Every Windows child uses the matching
windowless executable, with worker health checks, durable logs and launch
acknowledgement. No visible terminal, active chat or recurring monitor is needed.
Scoped sleep inhibition allows display sleep and ends when the supervisor exits;
shutdown still interrupts work. No automatic retry or restart.

Order: export/independent checks; six fits and six fit audits; model seal and
identity exception; fixed source pools and complete source audits; selected
panel and audit seals; 18-root collection/full replay; outcome-blind cost gate;
360-root collection/prescribed audit; final independent statistics and closure.

The eight-hour execution ceiling begins at the original launch.json epoch and
includes preparation, all data/fits/source work, probe, main and finalization.
Implementation/preflight is additional. Maintain five GiB free space. Check
budgets at export, optimizer/validation, source-move and rollout-world boundaries,
job completions and finalization. An active trajectory can finish before the next
check. A passing probe cannot extend the total deadline.

Before main, use probe selection/assessment timing per delivered continuation,
plus structural collection time per root, scaled to maximum 276,480 selection
and 138,240 assessment branches. Forecast audit blocks from their measured
replay rates, the sealed main sample's replay counts at maximum six/three
supports, and structural time per root. Divide worker-seconds by eight, apply
35% time allowance and add 300 seconds for finalization. Include all elapsed
launch time. Require elapsed plus forecast <= eight hours. Project storage
from measured probe bytes per recorded continuation at the main maximum, with
50% allowance plus 512 MiB, retaining the five-GiB reserve. No strength endpoint
may determine the operational gate.

Resource expiry or source shortage writes an operational hold; integrity or
worker failure writes failure evidence. Cancel pending work and preserve every
completed or partial strict artifact. Advisory status-file contention uses the
existing resilient reporter and must not invalidate successful scientific work.
Do not overwrite frozen sources or failed attempts. A complete run, including
a failed scientific gate, writes result, final audit, report and final status.
No automatic full-game arena, training successor, promotion or monitoring follows.
