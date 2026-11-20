# Experiment162: teacher feedback on student-visited positions

Owner authorized execution on2026-09-17: "ok, lets do it", following
`docs/strategy/next_after_161.md`. One bounded pipeline, no automatic extension,
promotion, successor or monitoring. Preserve frozen predecessors and dirty work.
This protocol becomes immutable with the manifest after all repository tests.

## Question and fixed parents

Does a small refresh of teacher labels on current student self-play positions
improve fast play beyond equal-dose practice on the existing teacher corpus,
and does it improve on the unchanged161 student?

Use all three selected161128-label students as source players and initializers:

|New fit seed|Parent fit|Update|Checkpoint SHA256|
|---|---|---:|---|
|162001|161001|563|8d83d69b9691569f187beb93f2636635ab419c835442d02f0eb6fc511866fa62|
|162002|161002|488|b06e37d3640adb846de562daaaae4d526bf586a848a1e1783f2bc4d2c3899c96|
|162003|161003|563|8d4a258f625003e0fad9241560db0f606c23d1d8fab3a9c1afd3bb0c2f11a66a|

Bind161's scientific identity
`f8f1fda4884cdcde518ad079430528b95abaf19e5b47946630dd5c4e5f22eeeb`,
recovery identity
`b3ec4a136483f0c0859edf6cd85e9751573bc26b719467d0b75042b11a43fe0a`,
completed result
`813596b6228a2e3530a8b73cd577b557bf211cfd76ed433411ae2313074b2113`,
selected models, old export/audit locks and completion records. Recursively check
its frozen predecessors.161's failed practical-progress gate remains unchanged.

## Source population, sampling and split

Generate exactly384 fresh games:128 initial-seed clusters times three fixed
source students. Each student plays itself on both sides. Index=3*cluster+fit;
environment seed1621000000+cluster. Across the three models use the same stream
digest `["162-source-stream-v1", cluster]`. Each model retains frozen035 as its
top-two-plus-two-random canonical proposer and its161 scorer chooses only among
those groups. No learned continuations, new support or teacher calls during play.
Full omniscient snapshots are replay evidence only; model inputs remain lawful.

Select six distinct actual decision positions per game,2,304 in total. For each
actor0/1 and phase0/1/2, select the minimum digest
`["162-position-sample-v1", game_index, turn]` within that cell. Phases are
turn<20,20<=turn<40,turn>=40. If cells are empty, fill remaining slots from all
unused decisions in the same hash order. If fewer than six total decisions exist,
stop with preserved evidence; no replacement games or sample-size extension.
Do not inspect winners, chosen action, disagreement, confidence, action family,
support size or teacher return when selecting. Keep one-option positions.

The exact count2,304 replaces the recommendation's approximate2,400 while giving
equal source-model counts and prospective seat/phase coverage. Six roots per game
limit within-game concentration. Source games across models share128 initial
seeds; they are not384 independent initial seeds.

Hash-rank the128 clusters by `["162-source-split-v1", cluster]`; the first26 are
validation, the remaining102 training. Both acting seats and all three models
stay together. Thus468 fresh validation rows and1,836 fresh training rows.
Pool fresh examples across the three source models for each fit. Retain161's
old160 split exactly:9,577 training and2,408 validation rows. No validation row
ever appears in training or old-row replacement.

## Teacher labels, timing probe and verification scope

Query the exact128-world160 teacher at each sampled root using that root's
recorded source proposal stream. This reproduces its actual canonical proposals
and representatives. Extend155's lawful world schedule to128 exactly as160;
greedy035 plays both simulated actors, terminal WDL is averaged over shared
worlds and the first maximum wins. The teacher does not execute the student move
before evaluating alternatives. No outcome/phase/confidence filter or retry.

First label18 prospectively selected roots: one from each source-model/actor/
phase cell, minimizing `["162-timing-probe-v1", root_index]`. Empty cells use
remaining hash order. These18 belong to the2,304 fixed roots and retain their
original train/validation membership; they are not queried twice. Audit them,
then pass an integrity/storage-only preflight before the remaining2,286 labels.
No teacher quality statistic is a preflight gate. Available disk must exceed
max(10GiB,1.5*128*probe_teacher_bytes+5GiB). Report127 times probe collection plus
audit duration as a provisional remaining-teacher estimate, with startup and
small-probe uncertainty disclosed. Do not use this to change scientific budgets.

Independently replay every source decision and natural game ending, including
features, proposals, scorer output, hidden-information invariance and snapshots.
Independently reconstruct every teacher proposal, every sampled lawful world,
all outcome-field/WDL checks, every mean, label and exported tensor. Full inner
trajectory replay covers exactly two worlds/root: the two lowest digests
`["162-audit-world-v1", root_index, world_index]`, all offered actions in each.
This is prospectively sampled inner verification, NOT full replay of every
teacher continuation. Save every world outcome and trace/final-state digest.
Shared primitives include frozen engine/encoders, Torch model and independently
written157/160 audit routines; successor sampling, batches and loops are separate.

## Matched learning

Arms `refresh` and `replay`, three paired fits each. Every refresh epoch sees all
11,413 combined training rows exactly once, uniformly. Four epochs give716
updates (179/epoch),45,652 total row exposures; batch64 with last batch21.

Construct one shared batch-slot plan per fit seed/epoch: shuffle combined indices
with `random.Random("162-batches-v1|seed|epoch")`, stable-sort by concrete legal
feature count, chunk into64, then shuffle chunks with the same RNG. The refresh
arm uses the actual combined rows. In the replay arm each old row stays in its
exact slot; each fresh-row slot is replaced by one uniformly sampled old training
row, with RNG `"162-old-replacement-v1|seed|epoch|fresh_row_id"`. Replacement is
with replacement and independent of labels. Every old row is seen once per epoch
plus the control's additional old-only draws. Both arms have identical update
counts, batch sizes and slot order, with differing row IDs only at fresh slots.
The treatment's new-data weight is fixed by corpus count,1,836/11,413 (~16.1%);
no replay-ratio sweep. Three fixed fit seeds, no best-seed selection.

Warm-start each from its frozen161 parent; fresh AdamW, lr1e-5, weight decay1e-4,
gradient clipping1, float32 CPU, one Torch thread, deterministic algorithms and
three fit workers. All policy/shared-trunk parameters train; value-head parameters
remain frozen and there is no value objective. Use exact157's offered-group
cross-entropy: model sees every legal concrete action, each offered canonical
group receives its maximum concrete logit, only offered groups enter the loss.
Single-group loss is zero; preserve those rows and their accounting. No inferred
negative labels on unoffered moves. Deployment is unchanged035 proposals plus
the updated restricted selector, with no expensive teacher call.

Both arms select checkpoints on the SAME2,876-row validation set. Average CE
within each source cluster/pair, then equally across40 old and26 fresh clusters.
This deliberately gives equal weight to source clusters, not rows or source
families. The old validation data are reused development evidence; the new panel
is held out from training but used for selection and is not the final test.

Observe initial0, early1/2/4/8/16/32/64 and quarter-epoch points using157's exact
ceil rule. Minimum common validation CE selects; ties use earliest update and
initial0 is eligible. No gameplay, subgroup or agreement selection. Independently
replay each entire fit and every saved checkpoint bitwise, including frozen value
head, validation predictions and checkpoint selection. Retain training evidence
for both arms regardless of apparent validation quality.

## Fresh game test

Two direct comparisons for each fixed fit: refresh versus replay, and refresh
versus unchanged161 anchor. Use400 new initial-seed clusters, all three fit pairs
and both seats per comparison:4,800 main games total. Environment1620000000+pair;
shared proposer stream digest `["162-proposer-stream-v1", "main", pair]`.
Freeze selected checkpoints before any main game. A separate48-game/four-cluster
pilot uses1629000000+pair and stage `probe`, is fully replayed, and only gates
integrity/storage (same10GiB minimum,1.5*100*pilot_bytes+5GiB). Its results never
choose models or change the schedule. No source, pilot and main seeds overlap.

For each comparison average both seats and all three fixed fits within a seed.
Bootstrap400 WHOLE clusters jointly across both comparisons,50,000 resamples,
NumPy default_rng seed1623000000. Use integer halfpoints until final division;
independent statistics reproduce the result. The intervals condition on these
fixed initializers/order streams and are pointwise95% intervals. Clear success
requires BOTH: lower bound>50% against replay, and score>=55% with lower bound>50%
against unchanged161. No five-point transfer gate, seat veto, margin veto or
seed-specific gate. Report their estimates, margins and decision latency anyway.
Do not round a miss to a pass. Report `clear_success_not_established` if either
gate fails, without equating this with proof of no improvement. No extra games.

Every actual game is fully independently replayed. Retain score/fewest-card
adjudication after two consecutive sole-legal-PASS turns; normal/equal-turn
endings are valid. Fourfold repetition,400 global turns,60 no-progress turns,
illegal/missing actions and exceptions preserve evidence and stop the pipeline;
never assign replacement outcomes or silently drop a game/root.

## Operation and limits

The pipeline runs source games/audit, sample seal,18-root teacher probe/audit,
timing/storage preflight,remaining teachers/audit, six fits/full fit audits,
selected-model seal,48 fast pilot games/audit,4,800 main games/audit, final report.
All artifacts are immutable with bound dependency locks. Build selected-model
dependencies once when they are also the direct upstream, preventing161's
duplicate-alias bookkeeping failure while retaining strict file uniqueness.

Require all repository tests, including real hidden Windows-spawn imports and
launcher-lifetime checks, before freeze. Record test logs and exact dependency
hashes; preserve the existing161 test-only160 guard isolation. Hidden pythonw
supervisor and workers, eight simulation/game workers, three fit workers, durable
parent acknowledgement before launcher exits. No visible terminal windows,
scheduled monitoring, automatic restart, checkpoint promotion or next experiment.
A failure cancels queued jobs, preserves already running work and records the
exception; recovery would be separate from the frozen scientific experiment.

Old+fresh versus equal-dose old-only tests the value of this refresh. It does not
isolate student-state distribution from simply obtaining extra independent data.
Source phase stratification and fixed small budget limit scope. No claim of
teacher equivalence, a better value function, learning-to-stronger-search feedback,
general opponent superiority or superhuman performance follows from a pass.
