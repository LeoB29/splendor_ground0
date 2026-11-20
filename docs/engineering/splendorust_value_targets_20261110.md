# Saved-corpus value-target study

10 November 2026. The user authorized implementing and running the recommended
next study after the accepted independent-data comparison. The original EF8A
checkpoint remains the reference. This is a separate version and freeze; all
previous source, data, receipts and failed attempts remain unchanged.

## Question and fixed intervention

Does replacing noisy terminal-only value targets with a fixed mixture of game
outcomes and the teacher's search evaluations improve this modest training update?
The preceding 400-game result was 48.625% [44.00, 53.25]; it did not establish a
useful strength gain. That result is historical context, not a concurrent live arm.

Reuse the corrected corpus of 1,024 training games and the exact two broad-arm
selections of 3,072 positions, three per game. Reuse the same seeds, batches,
single pass and 96 updates per fit. Both new fits start from pristine historical
EF8A, not from either trained checkpoint. Keep architecture, float32, policy
visit targets, policy/value loss weights, joint shared-layer training, AdamW
LR 1e-5, weight decay 1e-4 and gradient clipping 1 unchanged.

For each canonical acting-player position, let z be its official terminal
outcome in [-1,1]. Authenticate the saved native search capture and its
visit-weighted root value r in [0,1]. Set q=2r-1 and train the value output on

    t = (z + 0.837*q) / 1.837

with opposite-player output -t. Do not use the selected action's Q, unweighted
action average, or the raw network value. The fixed coefficient follows the
pinned upstream value formula, not a choice made from these assessment results.
This reproduces one ingredient of that recipe; upstream policy targets and
optimization settings differ, so this is not a full upstream reproduction.

Upstream provenance: commit 1828507b8d2f0e02bc03d102b9c6d29a54fa8d57,
research/training_strategy/strategy_train.py SHA256
06caf53b28a58fb74fe0ab69378cabfd01b15dcdd9485c8cbbb0853cb5947c94,
data.py SHA256 2465fc6bd44cfc1e0792ea82b64d0c7cf4f82ab479b940c1d930afd06dd0d5f8.

The two accepted terminal-only broad fits are authenticated controls, with no
retraining. The control and treatment must match selected rows, policy targets,
ordering, seeds, initialization and optimizer recipe. All selected root-Q values
bind their original captures, move rows, source games and corrected identities.
Save checkpoints 0/95/96 and exactly reproduce the final update including model,
optimizer, RNG, predictions, gradients and losses. Maximum optimizer calls:
192 scientific, two discarded qualification calls and two final-step replays.
No sweep, epoch selection, best-seed selection or automatic promotion.

## Qualification and reused diagnostics

Before fitting, exercise the actual corrected loader, encoded batches and new
target path on all six corrected card faces, non-purchase/multiple/final-round
nobles, gold payments and returns, and private reserves. Reject missing or wrong
rules/catalogue/source hashes, malformed captures, wrong actor orientation,
nonfinite/out-of-range values and unauthenticated replacement targets. Reuse
genuine accepted engineering rows; generate no new self-play collection games.

Every new trainer, evaluator and verifier calls verify_runtime_sources() exactly
once on startup. Bind ruleset splendor_base_qualified_rules_20261018_v1, catalogue
db4776918e612eee5a6f2617719c85ec5f413d592bbab0a64058babc83a70312 and turn contract
atomic_prereveal_completed_turn_v1 in corpus/checkpoint/runtime metadata.
All hidden-world card faces, explicit legal lists and transitions stay corrected.
The historical parent is identified as a warm start, not relabelled corrected
training data. Existing pre-reveal and internal forced-pass conventions remain.

Commit both new final checkpoints before evaluating them. The 64 validation and
64 test games have already been inspected in prior studies: explicitly treat
them as reused diagnostics, not fresh independent confirmation. Reuse accepted
terminal-model checkpoints and evaluate all five fixed predictors (parent, two
terminal controls and two mixed models) on the same positions, with fresh
forward passes but no extra fitting. Compare mixed minus terminal
within seed and game, then average the two fixed seeds. Report policy loss,
terminal-value error, search-value error and value drift separately. Group
uncertainty by whole game, using fixed 20,000 paired bootstrap draws. Do not
interpret lower mixed-target loss by itself as success. No offline performance
threshold admits, rejects or selects a model for the fresh strength assessment.

## Fresh independent playing-strength assessment

Both prospectively fixed mixed-target fits face the unchanged original EF8A at
128 simulations, three worlds, cpuct 0.4, depth 64. Qualify the actual imported
checkpoint/adapter and complete six excluded fresh operational games per fit.
Then run 200 fresh games per fit: 400 total, 200 disjoint swapped-seat pairs,
alternating ten-game blocks between fits. Freeze all seeds and exact corrected
initial states before any model work, checking known historical reservations.

The fixed equal average of the two candidates is primary. Retain 50,000
stratified paired bootstrap draws. A useful gain requires official AND
conservative score >=55%, two-sided 95% lower bound >50%, pooled score in each
seat >=50%, and the existing degenerate-interval safeguard. Report full effect
size and uncertainty even when these adoption criteria fail. A smaller gain may
remain plausible. Individual seeds are descriptive; do not promote the better one.

Keep actual 9-second internal and 10-second full-response research guards.
One second and 700 ms remain descriptive, with late completed moves accepted.
Classified hard-guard expiry loses the responsible player's game; infrastructure
or correctness failure stops incomplete. Preserve every failure, with no failed
decision retry, replacement game or active-clock reset. Retain every independent
game audit, all native graph/entry-point checks and at most 96 selected exact
replays per fit (192 total), followed by independent final acceptance.

Fresh mixed-versus-parent games establish strength relative to the parent; they
do not isolate live mixed-versus-terminal effects. The controlled target contrast
comes from matched reused diagnostics. No external-opponent, superhuman or strict
one-second deployment claim follows from this experiment alone.

## Bounds, ownership and automatic sequence

One retained root owns exclusive original wall/monotonic clocks and Windows job
trees. Overall cap 72 hours; model cutoff 48 hours. Stage ceilings, clipped to
those originals: qualification 4h, fits 4h, offline 6h, strength 32h, replays 8h,
final verification 24h including a 12h independent reader. Actual per-worker
aggregate RSS caps remain 4 GiB fit/offline, 2 GiB strength inference, 8 GiB saved
audit. Actual new evidence cap 64 GiB with 8 GiB reserved for finalization; actual
free disk floor 5 GiB. No available-RAM reserve, stability wait or forecast gate.
Duration and storage forecasts are advisory. No cloud rental or scheduled chat
monitor is created. No external testing, six-world work or new data collection.

Model-free regression tests and three independent source-bound reviews precede
freeze. The root then runs qualification, both fits and commitment, reused
diagnostics, live adapter qualification, 12 operational games, 400 main games,
selected exact replays, saved training/arithmetic audits, independent strength
readers and final acceptance automatically. Any technical repair requires a
separate qualified version/freeze and cannot change science or reset used clocks.

Reused evidence remains at its immutable paths. The accepted predecessor's full
preservation receipt is authenticated; every actually consumed input is hashed
again by its reader. Do not repeatedly rebuild the entire historical corpus
inventory as a substitute for source-specific validation. New outputs receive
their own full preservation inventory. Completion requires actual owner exit,
absence of recorded owned processes and final independent acceptance, not merely
a completion file. Report one or two result charts and update continuity.

## Interpretation and subsequent decision

The study tests this one value-target change at fixed modest training exposure.
A null does not rule out larger updates, alternative targets or more data. A
mixed-versus-terminal diagnostic difference alone is not evidence of stronger
play. A fresh strength gain motivates confirmation before model promotion; a
null motivates examining measured target/decision changes before another data
collection. No follow-up study starts automatically after this one completes.
