# Experiment 154: fresh validation of direct rollout move selection

2026-09-10. Owner authorized `docs/strategy/next_after_153.md` with "ok, do it".
Scope: preparation, tests, immutable freeze, timing, 768 fresh source positions,
32 discovery worlds/candidate, 128 independent evaluation worlds for selected
versus baseline, and full audits. No fitted model, arena, promotion or successor.
Both a pass and a failure stop after audit. Preserve all dirty/frozen work.

## Scientific question and population

Does this fixed finite-budget terminal-rollout selector obtain a material
conditional WDL gain over exact greedy 035 on fresh baseline positions?
It tests the prospective signal suggested by the post-150 development split.
It does not test learning, first establish search headroom (084 already did),
certify stable exact labels, compare with 084 or establish whole-game strength.

Generate exact greedy-035 versus itself games on the fixed source schedule.
Starting player is the unchanged engine default; sampled acting seat is balanced.
Choose one eligible nonterminal state per source game, 128 in each of six cells:
cell = 2*phase + actor; phases are turn<20, 20<=turn<40, turn>=40. Both players'
already-held reservations must all be publicly visible. Blind reservation is
legal at the root and in all continuations; disclose the source restriction.

The fresh source policy is greedy 035, as in 149's fresh test. Development
143/152 sources used early exploration, so this is an explicit population
difference from the motivating post-150 result. No earlier protected panel is
read for new predictions, selection, targets or pooling.

## Frozen source and candidate construction

Source index 0..4095, no extension. Dispatch consecutive batches of eight;
consume records in source order. A final batch may contain at most seven unused
prefetched games, all preserved and audited. Stop on any invalid or cutoff game,
including unused suffixes. Never replace a source on its outcome or difficulty.

Apply the unchanged 149 deterministic augmenting-path balanced matching rule
to eligible cells, in increasing cell/source order. For each feasible game/cell,
choose the state minimizing the existing canonical JSON SHA256 digest of
`["154-position-v1", source_index, turn_index]`. Stop at the first prefix filling
all six quotas. If 4096 sources cannot fill them, fail rather than change design.
Store every attempted source trace, source hash, eligibility counts, selected
snapshot, identity and the selected prefix. Audit all sources before discovery.

For each selected state, retain the exact 149 candidate algorithm under the
new split string `154-main`: rank all concrete legal actions by frozen 035 score,
original index on ties; keep the greedy action and highest-ranked distinct
canonical successor; sample two remaining canonical groups uniformly without
replacement, preserving existing representative actions. Python RNG seed is
`149-candidates-v1|154-main|source_index` with the numeric index interpolated.
Canonical support insertion order is unchanged. Keep all actions if fewer than
four exist, including baseline-only states. No candidate expansion or filtering.

Freeze the complete panel and candidate actions before any discovery outcome.
The independent source audit reconstructs greedy choices, matching and candidate
selection; checks root policy scores under a resampled legal world; and verifies
all snapshots, traces, legal actions, termination and identities.

## Randomness schedule

| Purpose | Seed rule |
|---|---|
| Main source | 1540000000 + source index |
| Discovery | 1541000000 + 1000*state index + world index (0..31) |
| Evaluation | 1542000000 + 1000*state index + world index (0..127) |
| Bootstrap | 1543000000 |
| Two off-panel timing source games | 1549000000 + index (0..1) |
| Probe discovery | 1549100000 + 1000*probe index + world index (0..3) |
| Probe evaluation | 1549200000 + 1000*probe index + world index (0..3) |
| Panel privacy check, no continuations | 1549300000 + state index |

These namespaces are mutually disjoint and beyond prior experiment namespaces.
Probe positions are the six already-frozen 149 preflight snapshots, not its test
or development evaluation panel. Their exact file hash is bound in the manifest.
Use their original candidate contract, seed string `149-candidates-v1|probe|index`
with each original source index. No probe result enters primary statistics.

## Discovery, selection and evaluation

For every discovery world resample observer-legal unknown cards, including deck
order. Share that initial world across all real candidates, apply each action
and follow exact greedy 035 for both players to natural terminal WDL. Unknown
card identities/order must not influence root candidate selection or policy
observations. Main world generator is frozen; auditor independently reconstructs
unknown pools from the public catalog minus legally known card identities.

WDL is +1 win, 0 draw, -1 loss from the original root actor's perspective. Use
equal-turn endings and score/card-count tie-breaks. Preserve 149 termination
rules: valid natural completion or two forced PASS turns with established score
adjudication. Fourfold repetition, turn>=400, 60 no-progress turns, missing
legal action and other invalid outcomes stop the experiment; no replacement,
exclusion or favorable reassignment. Root terminal outcomes remain valid.

Average 32 WDL outcomes per action. Select the largest mean, original candidate
index on exact ties (baseline first). No significance screen, extra worlds,
alternate budget, learned evaluator, optimism bonus or confidence filter.
Write the entire selection lock, including exact concrete/semantic actions,
discovery means, target lock and panel hashes, before any evaluation collection.

Independently replay every discovery continuation and reconstruct all selections
before allowing evaluation. This audit is retained and bound at final closure;
it is not needlessly rerun after the evaluation results arrive.

For each state evaluate only [baseline, selected] on 128 entirely new worlds.
If selected equals baseline, compute [baseline] once per world and retain an
exact zero-gain state. No state is dropped. Both branches otherwise share their
initial evaluation world and continue with exact greedy 035. The action lock
must precede every evaluation file; evaluation cannot reselect an action.

## Statistics and gates

Primary observation is the paired state mean WDL(selected)-WDL(baseline).
Display the equal-state mean across 768 source games. All six cells have equal
weight. Bootstrap 50,000 complete source-game resamples within each cell, not
individual continuations. NumPy default_rng seed1543000000; 50 chunks of1000;
within each chunk generate a1000x128 index array for cells0..5 in order. Average
sampled states within each cell, then cells. Percentiles2.5/97.5 use NumPy's
linear interpolation. Independent audit uses occurrence-count arithmetic and
manually interpolated sorted quantiles with the same frozen index stream.

Pass requires mean>=+.02, CI95 lower>0, half-width<=.02, and every integrity
audit passing. Compare numeric statistics to1e-12 absolute tolerance, choices
and gate classifications exactly. A wide interval is precision-limited and
does not authorize more states. +.02 WDL equals one point of expected terminal
score percentage under these continuations, not a full-game win-rate increase.

Report apparent discovery gain, independent gain, their difference, switch
fraction, coincident choices, baseline-only states, phase/seat/cell means and
all actual work counts regardless of success. No prediction-MSE or exact-label
agreement gate: this experiment fits no model. No pooling with older results.

## Preflight, runtime and verification

Eight CPU workers, one Torch thread/worker, deterministic Torch algorithms;
unchanged original 035 checkpoint. No NumPy BLAS with Torch. RNG and array
reductions are permitted. No install or new model dependency.

Before main sources: full repository tests; freeze protocol, new runner,
entrypoint, verifier, tests and explicit parent identities. Parent153 binding
`0627752a8399ac74484584025135282dcbff41e376122a362a8cf32218554d3f`, audit SHA
`128a026b4ce8df849d64b8673b12ac32f02c26f7a789a4262a77ced392f8e7c4`.
Do not expand the historic dynamic149 identity or edit frozen ancestor files.

Preparation runs two fresh off-panel exact-035 source games for source timing,
then four discovery worlds per candidate and four evaluation worlds for selected
versus baseline on the six fixed probe states. Probe selection is automatic under
the fixed rule; do not inspect gains or use outcomes to alter design. All probe
sources, worlds, paths, choices and locks receive full independent verification.

Time each source/target task in CPU and wall seconds. Project the maximum main
collection cost from per-continuation task time divided by8 workers; separately
project replay cost from auditor CPU time, and source ceiling from the two fresh
source probes. Record these as estimates with small-probe/contention caveats,
not promises or an outcome-dependent sample-size selector. Startup, matching
and hashing add overhead. Budget/population are not altered to meet a duration.

Main caps: discovery98,304; evaluation196,608; total294,912 continuations.
Actual work is smaller with shorter candidate support or coincident selections.
Source generation, at most144 probe continuations, and their replay audits are
additional. Full source audit precedes discovery. Full discovery audit precedes
evaluation. Full evaluation replay and independent final statistics/identity
checks precede closure, including after a statistical failure.

The auditor reuses the frozen repository engine, canonical grouping and 149's
independent greedy replay routine; it is not an independent Splendor rules
implementation. World reconstruction, source trajectory checking, matching,
candidate ordering, selection and statistics are independently implemented here.

## Execution, persistence and stopping

`run_rollout_selector_154.py prepare` freezes and runs preflight. `run` requires
passed preflight, then performs all fixed stages. Immutable JSON files refuse
overwrite; the advisory process lock prevents a duplicate run. `progress.json`
is mutable stage/count telemetry and contributes no scientific decisions.

An explicit `resume` may recover an externally interrupted, identity-matching
run by reusing complete source/target files and validated stage locks. No
completed simulations are repeated during collection; every file is ultimately
hash-checked and replayed by the full audit. Deterministic completed intermediate
records must compare equal. A recorded failure requires review and cannot be
automatically resumed. Never tune or edit the frozen experiment after outcomes.

Final status includes result and audit hashes, actual continuations and time,
statistical and integrity decisions. Both outcomes close154 for review. A pass
supports only a later gameplay proposal; no arena, learner, promotion, large
self-play run or automatic successor is included.
