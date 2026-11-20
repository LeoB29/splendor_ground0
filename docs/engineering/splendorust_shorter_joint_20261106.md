# One-pass joint-training diagnostic

6 November 2026. User authorization: "Do this test you suggest", referring to
one bounded test of shorter joint training using existing data and a schedule
fixed before running. This authorizes qualification and execution of this study.
It launches no new collection, games, search or playing-strength assessment.

## Question and fixed design

The previous accepted saved-corpus diagnostic found falling training losses but
worsening held-out joint losses by update668. Broader game coverage reduced that
deterioration without improving on original EF8A. The question is whether one
complete pass through the same selected positions improves held-out predictions.
One pass is an interpretable prospective dose, not a claimed optimum.

Run exactly two fresh joint fits, one for each original broad repetition:192
training games with16 positions per game,3,072 positions and96 full batch32
updates. Use the exact first96 batches, root selections and RNG seeds from the
previous broad fits. This isolates update exposure; do not reshuffle or substitute
examples. Start both fits independently from pristine historical EF8A checkpoint
ef8a4521cd6c03c7075f15efee23f4cde6ec94d1b5398765cf0c24a09ad745cb.

Bind the previously verified first96 update records for each broad fit through
the accepted saved audit. The new saved audit must compare every new update's
mathematical fields, batch/source identity and parameter-group hashes exactly
with that prefix. Exclude only elapsed time and the new selection manifest hash.
This adds no model calls and verifies the same learning path was stopped earlier.

Network architecture/size, joint policy/value objective and weights, AdamW recipe
(learning rate1e-5, weight decay1e-4, clipping1), float32 CPU execution and lawful
information remain unchanged. Update shared layers and both heads. Save0/95/96;
only0 and96 are evaluation endpoints. Reproduce95-to96 exactly for each fit,
including model/optimizer/RNG, outputs/losses/gradients. There is no intermediate
checkpoint selection, early stopping based on results or automatic promotion.

Retain two discarded qualification updates from the previously qualified scope
fixture. Scientific optimizer calls192; qualification2; exact reproductions2;
total196. No separate gradient-alignment probes are needed for this duration
question; finite per-update gradients and exact replay remain mandatory.

## Data, correctness and provenance

Use only the existing accepted corrected corpus:192 training games and the
already-opened32 validation plus32 offline games. Exclude engineering collection
benchmark trajectories. Authenticate the previous materialized selection SHA
757199b51e66cde02b56128e07cbc3fb31fc741e76073adc6425eec25f3d8e56 and derive the
two broad96-batch prefixes without changing original evidence. Bind source
games, target/row identities, legal support, complete corrected catalogue and
ruleset splendor_base_qualified_rules_20261018_v1 in the freeze/checkpoints.

Every new trainer/evaluator calls verify_runtime_sources once at startup. The
new actual loader/collator/trainer entry point must pass the corrected six-card,
noble/non-purchase/final-round, gold/return, private-reserve, value-sign and bad
identity/hash counterexamples required by future_training_rules_20261021.md.
Reuse accepted original collection evidence; no new game is required for this.
Never relabel historical targets. Historical EF8A remains an explicit warm start.

All new sources are in tools/splendorust_shorter_joint_20261106; outputs only in
outputs/splendorust_shorter_joint_20261106_01. Original sources, failures, players,
results and recovered acceptance stay immutable. Model-free tests and two
independent source-bound reviews precede an immutable execution freeze.

## Assessment and interpretation

Finish both fits before either fixed diagnostic evaluation is opened. Evaluate
the selected training positions and all3,557 held-out positions at0 and96. The
primary measurements are policy cross-entropy and value mean-squared error at96
minus original EF8A, averaged equally over source games. Report both repetitions,
validation/offline separately, pooled64 games and fixed mean of the repetitions.
Repetitions are not additional independent games.

Use20,000 paired whole-game bootstrap draws, seed20261106, common game indices
across fits/endpoints. Intervals are nominal95% and exploratory: all these held-out
games were previously opened, and the duration question followed prior results.
Historical668 results may be quoted as context, never as a new randomized arm.

Both pooled losses below original in both repetitions constitute a consistent
promising offline direction. Either pooled loss above original is a tradeoff or
deterioration, even when less harmful than668. Exact zero is unchanged. An interval
spanning zero indicates uncertainty, never equivalence. Show all estimates and
intervals; this descriptive classification is not a promotion or strength gate.
Complete the fixed schedule regardless of interim signs. No result establishes
stronger play, superhuman strength or the optimal number of training passes.

## Bounds, automatic sequence and acceptance

Use one retained controller clock, one exact Windows-owned worker at a time,
CPU0 model workers and CPU2 supervisor; saved-only workers inherit CPU2.
Automatic sequence: actual qualification, two
fits and exact replays, two fixed evaluators, saved-evidence audit, then numerical
analysis/report in a fresh process that never imports Torch. This separation
prevents the plotting/runtime conflict observed during the previous finalization;
never set KMP_DUPLICATE_LIB_OK to bypass it.

Generous actual ceilings:12h total,8h model cutoff,1h qualification,1h per fit,
2h for the full evaluator sequence,4h saved audit and1h isolated analysis, all
clipped by the same total clock. Aggregate owned RSS4GiB, new evidence4GiB and
actual free-disk floor1GiB. No available-RAM reserve, stability wait or forecast
admission stop. Forecasts are advisory only. No scheduled monitor or implicit
retry. Preserve failures and partial evidence; valid negative results complete.

The saved audit verifies exact selections and targets, all update/checkpoint/
optimizer/RNG/replay evidence, both fixed evaluator inventories and source
predictions, original clocks, actual exits and cleanup. It runs no networks or
optimizer updates. The separate analysis runs no models and delivers one or two
clear charts. Final completion requires independent evidence/interpretation/
visual acceptance plus actual retained owner exit and exact process absence.
Original EF8A remains reference. Larger-data, external-opponent, six-world and
speculative GPU work remain deferred.
