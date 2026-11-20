# Learned fourth-proposal screen

Owner authorized implementation and the bounded experiment on2026-09-17. The
recommendation is `../strategy/next_after_saved_halving.md`. This new study uses
isolated files outside the frozen src tree. Preserve all prior work. No training,
batching, halving, model promotion, arena or successor is included.

## Population and comparisons

Main:288 positions from288 distinct fresh exact-greedy035 source games,48 per
early/middle/late by acting-seat cell. Source seed2030000000+index, a fixed1,024
games. Probe:18 distinct positions,3 per cell, from128 separately seeded source
games2039000000+index. Every generated source, including the unselected suffix,
is replayed and must end naturally. An incomplete quota or invalid game fails
the stage; no replacement, exclusion or source extension.

Use154's existing no-private-reserve eligibility: at the selected position both
players' reserves are visible. Phase is turn<20,20--39,>=40. Select the first
feasible capacity-constrained matching in ascending source index, one position
per source. Within each source/cell choose the minimum digest of
`[learned-proposal-position-v1,stage,index,turn]`. Outcomes do not select positions.
New seed namespaces were checked against current source/protocol files before
implementation. Synthetic test seeds2039500000+ are separate from evidence.

All three frozen161128-world students act on every position:

| Seed | Checkpoint update | SHA256 |
|---|---:|---|
|161001|563|8d83d69b9691569f187beb93f2636635ab419c835442d02f0eb6fc511866fa62|
|161002|488|b06e37d3640adb846de562daaaae4d526bf586a848a1e1783f2bc4d2c3899c96|
|161003|563|8d4a258f625003e0fad9241560db0f606c23d1d8fab3a9c1afd3bb0c2f11a66a|

Construct original155 candidates using its unchanged candidate nonce/split and
canonical grouping. Reference keeps all four original candidates. Every other
arm keeps the first three:035top2 plus its first random alternative. Fourth is
the highest035-ranked group outside those three (raw control), or the highest
all-legal group score from one frozen161 fit. All logits see the complete lawful
legal-action set; group score is the maximum concrete-action logit. Learned ties
follow original canonical group order. The raw ranking follows descending035
concrete scores, original legal index on ties, then first occurrence of each group.

Preserve exact035 concrete representatives for the original top two. All other
groups keep the existing first-legal canonical representative, including a new
proposal. This retains the reference and avoids changing representatives across
arms. Below four canonical groups all arms retain the complete original support.
Retain all coincident proposals and choices. No model or subset is selected.

## Simulation and choice locks

Each conceptual arm gets32 worlds per offered action, up to128 continuations.
All arms share the same sampled worlds and greedy035 terminal continuation policy
for both players. Use WDL+1/0/-1 from the root actor, terminal completion or the
existing two-forced-pass adjudication only. Repetition4, no progress60, turn400,
stalled or illegal action are failures, never usable targets.

Share simulations across the union of candidate groups, at most8. Every arm's
choice is its largest mean WDL, first offered candidate on ties. Seal the entire
stage's choices and full selection-replay receipts before evaluation begins.
Evaluation uses128 new independent worlds and only the union of locked winning
actions, at most5. Every choice is judged on common evaluation worlds. The
maximum is896 continuations/root,258,048 main and16,128 probe, before full replay.
Shared research computation does not enlarge any conceptual arm's budget.

World seed is base+1,000*position_index+world_index. Bases:
probe selection2039100000, probe evaluation2039200000;
main selection2031000000, main evaluation2032000000. Namespaces are disjoint.
Original155 root-schedule world labels are recorded as proposal provenance only;
actual collected worlds always use these new selection/evaluation schedules.
Privacy probes use2039300000+(1,000 for main)+position_index.

## Statistics and conclusions

Score=(WDL+1)/2. Average each fit's evaluation score and then equally average the
three fits per root. Primary: learned mean minus reference. Required attribution
contrast: learned mean minus035-ranked control.50,000 paired source-game bootstrap
replicates within six48-position cells; seed2033000000,1,000-replicate chunks.
Both contrasts use the same draws, keeping every arm/fit together. Equal cell
weight, NumPy linear2.5/97.5% quantiles. Fits are fixed, not independent roots.

Only if both95% lower bounds exceed zero classify
`positive_conditional_learning_contribution`; otherwise
`learning_contribution_not_established`. This is a conjunctive requirement,
not separate claims chosen after seeing data. Report both effect sizes/intervals.
No practical whole-game gain is inferred from statistical significance here.

Per-fit scores, cells, proposal/choice overlap, ranks and timing are descriptive.
No subgroup, best fit, proposal-score threshold or further sampling is selected.
A negative result rejects this exact unrestricted-max proposer, not all learned
proposal mechanisms. A positive result warrants considering fresh gameplay; no
arena is automatically authorized. Source-distribution/no-private-reserve limits,
conditional035 futures and equal-count-versus-equal-time limits must be stated.

## Integrity and execution

Entry `run_learned_proposal_screen.py`. Core and independent verifier are
`scripts/analysis/learned_proposal_screen.py` and
`scripts/analysis/verify_learned_proposal_screen.py`. Output is
`outputs/learned_proposal_screen`.

The new verifier independently reconstructs source actions/states, outcome-blind
matching, complete legal/group inputs, all proposal rankings and representatives,
hidden-world invariance, sampled worlds, chosen moves and both statistics. Every
new continuation is fully replayed using149's separate replay routine, including
independent afterstate/refill composition checks. The rules engine, canonical
grouping and neural arithmetic are trusted shared dependencies, not independently
reimplemented Splendor rules. This is full replay of this study, unlike160's
sampled inner replay.

Freeze the whole existing src inventory, all new study files, the imported149/154
verifiers, exact models/selected-fit audit chain, current passing test evidence
and CPU/Python/NumPy/Torch/windowless-runtime identity. Verify recorded inventories
of completed studies without adding new files to their historical identities.
No already-bound file is edited. New files and selected checkpoints become frozen.

Explicit commands are `check --checks <new directory>`, `prepare --checks
<checks.json>`, and `launch`. Launch creates a detached pythonw supervisor and
holds its parent until startup acknowledgement. Each pool uses eight verified
pythonw workers, one deterministic Torch thread. Every stage has a write-once
launch, input hashes, output lock and audit receipts. No duplicate launch or
automatic retry. Failures and partial evidence remain available.

Probe pipeline includes full source/proposal/selection/evaluation verification.
Its outcome-free cost estimate uses the maximum main continuation budget, the
larger aggregate CPU/task-wall cost, source scaling and25% overhead. Proceed
automatically within the owner's authorization only if forecast<=8hours and
projected remaining disk>=5GiB. Otherwise record an operational hold for review.
This ceiling is a cost safeguard, not a strength-based gate. The initial3--5hour
estimate remains provisional; implementation/checking time is additional.

The supervisor then runs the fixed main study and all prescribed audits, writes
the final report and exits. Probe strength is never summarized or used to choose
settings. No scheduled monitoring or unattended successor is attached.
