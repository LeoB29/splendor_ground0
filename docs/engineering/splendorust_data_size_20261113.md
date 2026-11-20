# Separate-core EF8A data-size diagnostic — 13 November 2026

User authorization: prepare and qualify the data-size study on a separate CPU
core, then run the four fits alongside the calibration if checks pass. Keep the
current experiment unchanged. This is a new version and a new bounded attempt.
No new games, live comparison, model promotion or later collection is launched.

## Scientific question and fixed comparisons

The completed duration study found improving training fit but deteriorating
held-out errors by four passes. That does not uniquely identify a data shortage.
This study asks whether more distinct corrected self-play games help under the
current fixed learning recipe, separating one-pass results from matched updates.

Reuse the accepted1,024-game training corpus and128 held-out games. Select one
outcome-blind nested whole-game sample256⊂512⊂1024 using a prospective study hash
of training game identity only. Both optimization seeds use the same subsets;
the study does not estimate uncertainty across arbitrary subset selections.
Retain every eligible position in each selected game. No scattered-position
subsampling, target-dependent selection or new self-play.

Four fresh fits start from the historical EF8A checkpoint:256 and512 games,
each with the two already fixed optimization seeds202611071 and202611072. Save
one-pass and1,782-update endpoints from each trajectory. Reuse the two accepted
1,024-game one-pass checkpoints and their accepted predictions. First-pass
ordering filters each seed's original full-corpus order to the selected games;
later passes use fresh prospective study/seed/pass/identity hashes. No model RNG
is consumed by schedule construction, and no best seed or endpoint is selected.

| Games | Distinct positions | One-pass updates | Final updates | Presentations at final |
|---|---:|---:|---:|---:|
|256|14,218|445|1,782|56,936|
|512|28,426|889|1,782|56,980|
|1,024, reused|57,022|1,782|1,782|57,022|

Use batch32 and each genuine final partial batch:10 for both new subset passes,
30 for the reused full corpus. Do not pad or drop a partial batch. After complete
subset passes, stop exactly at1,782 updates, possibly within another pass. This
is four complete passes plus two batches for256 games and two passes plus four
batches for512. The small presentation-count differences are disclosed; this
matches optimizer updates, not exactly every arithmetic operation or example.

Keep the accepted architecture, all trainable shared layers and both heads,
float32 operations, actual-batch mean reduction, AdamW LR1e-5, weight decay1e-4,
clip1, and policy/value coefficients1 unchanged. Policy targets are saved search
visit distributions. Mixed value target t=(z+0.837*q)/1.837, q=2*r-1, using the
original512-search root win credit r and official actor outcome z. No new value
target, regularizer, learning-rate sweep or teacher is introduced.

## CPU placement and protection of the running calibration

The existing calibration remains on CPU0 with its supervisor on CPU2. Use CPU4
for this study's model operations and CPU6 for its owners/model-free analysis.
Windows physical-core topology must confirm these are four distinct physical
performance cores; logical siblings are not a separate training core. Only the
new processes' affinity is changed. Do not pause, reprioritize, restart or modify
the calibration or any other user's process. Shared memory/power can still affect
elapsed times; neither study's scientific settings or existing clocks change.

Bind the active calibration freeze
51340bd0cae293d84fbc181b8985195f76f1db61d4a30cc09fbaa0a012a18940 and verify its
frozen source/data inputs before launch and at acceptance. Do not inspect its
partial scientific scores. Its mutable progress/evidence files are not inputs
to this study and are not rewritten. No scheduled monitor is created.

## Actual correctness qualification

Every new worker calls verify_runtime_sources once at startup and saves the
receipt. Bind ruleset splendor_base_qualified_rules_20261018_v1, authoritative
catalogue db4776918e612eee5a6f2617719c85ec5f413d592bbab0a64058babc83a70312,
turn contract atomic_prereveal_completed_turn_v1 and generated-under-declared-
rules provenance. EF8A remains an explicitly historical warm start/reference.

Use the accepted memory-mapped corrected corpus cache read-only. Exercise the
actual chosen loader/cache/collator/evaluator path on all six corrected card
faces; noble-after-nonpurchase/multiple/final-round cases; payments/returns;
private reserves/all-card reconstruction; explicit corrected legal lists; and
legacy, identity, data-hash and reconstruction rejection cases. Merely declaring
the ruleset or passing an outer helper self-check is insufficient.

On CPU4 reproduce both original CPU0 mixed fits' saved final30-example update
exactly from their accepted pre-last states, including predictions, gradients,
model tensors, AdamW state and RNG. Qualify both new subset partial-batch routes
by direct versus save/reload execution. These six updates are disposable and
never become a scientific fit. Four additional actual evaluator forwards must
match saved CPU0 structural/genuine outputs and fixed saved prediction batches.
Do not run fresh qualification work on CPU0. Separate-source equivalence checks
must establish unchanged optimizer/loss mathematics except the explicit supported
partial-batch-size admission. A real numerical mismatch blocks the new fits.

Each scientific endpoint retains its pre-last state and reproduces its last
update exactly. Four fits×two endpoints give eight replays. Fixed counts:
7,128 scientific updates +6 qualification +8 replays =7,142 optimizer calls.
All eight fixed new endpoint checkpoints are committed before new assessment.
Fresh assessment requires2,552 batched forwards; original outputs are reused.
Save all original and rejected evidence. No silent retry or checkpoint replacement.

## Assessment and prospective reading rules

Use the same3,072-position full-training diagnostic panel and all3,533 validation
and3,553 offline positions from the128 held-out games. Those held-out games have
already been inspected repeatedly: these are development diagnostics, not fresh
independent confirmation. For the training panel, separately identify positions
inside each selected training subset and outside it; compare every model and
parent on the identical membership masks. An outside-subset training position is
not retrospectively renamed a pristine held-out example.

Report both seeds and their fixed mean, policy CE, teacher entropy, policy KL,
teacher-gap closure, top-choice agreement, outcome/search/mixed-value MSE and
value drift. Weight positions within each game and games equally. Use a fixed
20,000-draw paired whole-game bootstrap stratified over the64/64 held-out splits.
Report raw differences and descriptive intervals, keeping policy/value tradeoffs
visible. Two optimization seeds do not quantify arbitrary seed/subset uncertainty.
Do not adapt training duration or select the most flattering metric.

- The one-pass curve describes this recipe with increasing data and updates.
  A rise alone does not isolate a data effect.
- Better held-out performance from more distinct games at1,782 updates supports
  a data-diversity benefit within this corpus and recipe. It does not predict a
  reliable number of additional games or prove stronger live play.
- Flat or uncertain differences weaken the case for immediate larger collection
  under this recipe; they do not prove more data can never help.
- Policy improvement with value deterioration is a tradeoff, not automatic
  success. Preserve the original model as reference in every outcome.

The running calibration independently tests a screening measure against known
search-budget gains. Its result is not needed to launch these fits. No partial
calibration result affects this design. Teacher-repeatability analysis and any
later live study remain separate recommendations, not automatic stages here.

## Bounds, execution and acceptance

One training/evaluation worker at a time beside the existing calibration. Fixed
24h global cap and18h model cutoff from this launch. Stage caps: qualification2h,
each fit2h, commitment1h, evaluation3h, analysis3h, acceptance4h, all clipped to
the original global/model deadlines. No available-RAM admission or forecast
gate. Actual model-tree RSS4GiB, audit8GiB, new evidence16GiB, actual free disk
reserve5GiB. All evidence and failed prefixes remain preserved.

Require source-bound model-free tests, independent reviews and a detached
launch/cleanup fixture before freezing. Durable owners use exact PID plus birth
identity and Windows job containment, original wall/monotonic clocks, and actual
exit receipts. Cleanup can only target the new owned tree. Stage failure prevents
later stages; valid negative findings end the study without another training run.

After four fits: commitment, fixed evaluation, analysis, charts and independent
saved-evidence acceptance run automatically. Recompute arithmetic from saved
predictions, verify exact endpoint replays and all consumed source/cache/runtime
hashes, and prove all worker exits/cleanup. Completion requires actual final
owner exit, not a completion file alone. Inspect and present at most two charts.
No automatic model promotion, new collection, live games or follow-up launch.
