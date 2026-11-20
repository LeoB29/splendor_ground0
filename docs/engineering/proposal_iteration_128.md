# Controlled second proposal-learning generation at128 worlds

2026-09-24. The owner explicitly requested starting the complete proposed staged
experiment. This protocol operationalizes `docs/strategy/next_after_learned128.md`.
It is frozen with code, tests, runtime, source lineage and both arena schedules
before any production extraction or fitting. No old source, checkpoint, result,
protocol or frozen inventory is edited. New code lives in
`tools/proposal_iteration_128`, with `run_proposal_iteration_128.py` as entrypoint.

## Question and scope

Does one update from the newly improved learned128 teacher make a better fourth
move proposer inside a128-world planner? If so, does it beat an update given the
same amount of practice on old examples? This is one fixed feedback package:
the new states, candidate support and128-world targets change together. It does
not isolate teacher quality on identical states or establish broader/human strength.

The accepted parent is `outputs/learned_proposal_128`, binding
`a9a541cb44599e4ed6c6076f05c5b8b29692859ad9c667979b17cf16dd754773`, with final result
SHA256 `35769af8ecc525851f2b3da1f4e7aa8ab1699e4db6b5590206d06b212d329d71`.
That completed study checked every actual game and sampled inner continuations;
its recorded targets are finite-sample preferences, not optimal-action labels.

## Data and six fixed fits

Retain all11,985 original160 exported decisions, their independently verified
161 tensor exports, and the original9,577 training/2,408 validation split.
Use all11,951 candidate-side decisions from the learned128 main arena's402 games
and201 starting-seed pairs. Both seats and every decision from a pair stay
together. Within each of the three source-fit strata of67 pairs, the14 lowest
hash priorities under `proposal-iteration-128-source-pair-split-v1` are validation;
the other53 are training. Thus new splits are159/42 pairs, combined with160/40
old pairs. Pool all three teachers' examples for all fits. Never use opposing
roots as same-state counterfactual labels or select the best source fit.

Check every source game/root against the sealed parent/audit linkage. Reconstruct
the new actual game, legal actions, offered-group labels from all128 recorded
outcomes, first-maximum ties, lawful model tensors and hidden-information
invariance independently. No new simulated continuations are needed for export.
Retain roots from all outcomes, supports1--4, phases and private-reserve cases.
Seal per-game tensors, export receipts, source hashes, split identities and a
combined data audit before fitting. The 23,936 rows represent401 source pairs.

Warm-start the three frozen161 checkpoints, selected updates563/488/563, with
the exact hashes already bound by the parent. For each source fit161001/2/3,
train `treatment` and `replay` with corresponding new batch seed207001/2/3.
Treatment visits all combined training rows once each epoch. Control shares
every old-example slot and replaces each new-example slot with a uniformly drawn
old training example, using the frozen label-independent replacement tag.
Batch-slot order, updates, batch sizes and row-exposure counts match exactly.

Use four complete passes, batch64 including a final short batch, lr1e-5,
weight decay1e-4, gradient clipping1, fresh AdamW, deterministic float32CPU and
one Torch thread per worker. Three fitting workers run six fixed jobs. Retain
the existing architecture and max-concrete-logit offered-canonical-group cross
entropy; unoffered legal actions are not labeled bad. Freeze value-head
parameters; shared-trunk training does not imply unchanged value predictions.

Both arms use the same82-cluster validation panel (40old/42new), with equal
weight per source pair after averaging within pair. Checkpoint0 is eligible,
then updates1/2/4/8/16/32/64 when in range and quarter-epoch endpoints, deduplicated.
Select the earliest strict minimum pair-mean validation loss. Complete all four
passes regardless of early validation. Independently reconstruct all six fits,
every checkpoint tensor and validation report, batch/slot sequences, selected
checkpoint and frozen head. Seal all nine models (six updated and three anchors)
only after these checks and matched-dose verification.

If all three selected treatment state dictionaries equal their initial anchors
bit for bit, record `no_selected_model_update` and stop without identical-player
arenas. File hashes or agreement on a few positions are insufficient for that
test. An individually unchanged fit remains in its preassigned arena stratum.

## Fixed roles, stages and analysis

Only the model ranking the fourth candidate changes. Both sides retain the
original035 top two candidates and first random alternative, then use the
assigned model's highest remaining canonical group as the fourth candidate.
For support below4 retain the entire original set. Preserve concrete action
representatives, canonical/group-max ordering and ties, original observer-legal
world construction,128 coupled worlds, greedy035 continuations and terminalWDL
choice with original-order ties. Use the proven serial backend and mature rules.
No source module registries are overwritten to swap models; each worker loads
an isolated sealed registry and records model identity at every root.

- `probe_a`:12 games, six pairs, environment seeds2079000000+pair.
- `stage_a`:402 games,201 pairs, environment seeds2070000000+pair.
- `probe_b`:12 games, six pairs, environment seeds2079100000+pair.
- `stage_b`:402 games,201 pairs, environment seeds2071000000+pair.
- Isolated fixtures only: six games, seeds2079800000+pair; never production output.

Pair assignment ranks hashes of `["proposal-iteration-128-fit-v1", stage, pair]`
and cycles all three source fits. Search streams hash
`["proposal-iteration-128-stream-v1", stage, pair]`. Both seats share each initial
seed/stream. Each main stage has67 pairs/134 games per fit. Stages are disjoint.
No fitting/checkpoint decision uses probe or arena outcomes.

Stage A compares treatment against the unchanged corresponding161 proposer,
with both inside128-world planning. Pass only if observed score>=55% and the
paired95% interval lower endpoint>50%. After all prescribed checks and independent
statistics pass, and only then, proceed automatically to probe B and Stage B.
Stage B compares treatment against its matched old-data replay update. Its gate
is paired95% lower endpoint>50%. Overall positive controlled iteration requires
both gates. Both schedules and the conditional policy are frozen before fitting.

Score wins1/draws.5/losses0. Resample67 whole seat pairs independently within
each of the three fixed-fit strata, combining equal-size strata. Use50,000
bootstrap samples, chunks1000, NumPy default_rng seeds2073000000(A) and
2073100000(B), linear2.5/97.5percentiles. An independent implementation verifies
pair assignments, integer half-point totals, interval interpolation and gates.
The interval conditions on these three models. This is fixed-sequence testing:
report all A results and report B as conditional on A passing. No interim stopping
from scores, fit picking, extra games, extensions or outcome-based repair.

If A fails, stop with `practical_progress_not_established`; do not infer that all
iterative learning is impossible. If A passes and B fails, report actual progress
against the anchor with new-corpus value unestablished. A two-gate pass supports
this generation, not indefinite improvement, external strength or superhuman play.
No automatic model promotion or further generation follows.

## Verification and freezing

Follow `docs/engineering/verification_policy.md`: independently reconstruct
every actual game and decision, its lawful proposals/model identity/hidden-input
invariance, all declared world seeds and observer-legal states, record budgets,
all outcome-field validity, aggregates, choice and final score. Replay every
candidate in8 of128 worlds at each root, selected by smallest identity hashes
with audit seed2074000000 and `proposal-iteration-128-audit-world-v1`.

Add full-root stress replay using `proposal-iteration-128-audit-stress-v1`:
minimum-priority root per deployed role/assigned-fit x phase x acting seat x
private-reserve category, plus per role/fit for offered supports1/2/3 when present.
Phases0--19,20--39,40+; privacy neither/own/opponent/both. Deduplicate overlaps;
maximum162 full stress roots per stage. Missing cells are reported. The sample
selection does not inspect outcomes, timing, choice or outcome-derived hashes.
Seal inventory, sample and coverage before replay; reconcile every replay receipt
and actual root hash afterward. Shared rules/encoders/neural arithmetic remain
trusted dependencies; unreplayed continuation trajectories are not certified.

Preflight covers dynamic model roles, private reserves, legal supports, tie rules,
sample corruption/selected-branch mismatches, all-game reconstruction, hidden
Windows workers, fit reproduction and failures, and conditional/operational gates.
Inherited mature simulation tests remain unchanged. New fixture outputs use
isolated temporary folders and cannot target any production path.

The manifest binds code/config/runtime, both main/probe schedules, the full
current check plan with passing logs, parent frozen dependencies/closed results,
source export locks and all three initializers. Stage launch receipts bind data,
audited models and prerequisites. Before either B stage, check the result through
the finalized A audit's hash and reclose A evidence. Before every arena, verify
the learning seal and selected model hashes. Final closure rechecks every
completed probe/main game's roots, samples, audit receipts and statistics links,
including A evidence after B completes; it adds no rollouts or statistical tests.

## Runtime, resources and independent execution

Use a detached `pythonw` supervisor, three CPU workers for fits/reproduction,
eight for collection/audit, one deterministic Torch thread each. Use the matching
GUI-subsystem interpreter for every spawn pool; assert zero console attachment
and retain per-worker logs/health receipts. The pipeline works independently of
an active chat. Supervisor-scoped system-sleep inhibition leaves display and
persistent power settings unchanged and releases on exit. No monitoring is added.

Each12-game probe and its sampled audit precede its main stage. Forecast one
402-game stage from summed worker times divided by8: ordinary work scales402/12;
stress replay scales to162 roots. Apply25% time and50% storage allowances plus
512MiB per projected main stage. Before A, reserve BOTH potential main stages and
one further probe in disk/time; before B, recheck its own forecast against the
remaining budget. Gates use operational measurements only, never wins or choices.
Require per-stage forecast<=120h, total elapsed+remaining forecast<=216h, and
projected remaining disk usage leaving at least5GiB free. Fitting/export costs
already incurred count in elapsed time/free space. Typical expectation remains
roughly3 days for A or6 days for A+B, plus preparation.

Enforce the216h total runtime and5GiB reserve at completed-job boundaries.
On failure cancel jobs that can still be cancelled; already running single-game
or single-fit jobs finish and preserve their evidence. This is not an instantaneous
hard process kill. An unfavorable resource forecast writes an operational hold.
An integrity or worker failure writes a failure receipt and stops the pipeline.
No automatic retry, continuation from partial games, deletion, overwrite,
unbounded diagnosis or amended success criterion is permitted. Preserve all
partial evidence for an explicit later recovery decision.

The user has authorized this full staged run. A passing A needs no further
confirmation for B. It authorizes no unrelated run, further generation or monitor.
