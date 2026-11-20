# Same-corpus mixed-target training duration diagnostic

13 November 2026. The user authorized the recommended next step after the accepted
full-corpus study. Implement and run this bounded offline diagnostic; do not launch
collection, search games, a live comparison, an external opponent or a new teacher.
The original EF8A model remains the reference. Earlier frozen evidence is read-only.

## Fixed scientific design

Continue the two accepted one-pass mixed-target checkpoints from the full-corpus
study, including exact AdamW moments, step counters and Python/NumPy/Torch RNG.
The common historical warm start is EF8A, SHA256
ef8a4521cd6c03c7075f15efee23f4cde6ec94d1b5398765cf0c24a09ad745cb.
The accepted one-pass endpoints supply the one-pass comparison. Each trajectory
continues through passes two, three and four. Assessment checkpoints are fixed at
one, two and four passes; neither seed nor checkpoint is selected during this study.

Every pass uses all 57,022 eligible corrected roots from the same 1,024 training
games, once each: 1,781 batches of 32 and one genuine batch of 30. No padding,
dropping, held-out training examples or copied data collection. Passes two to four
use a prospective deterministic hash ordering by study, seed, pass and root identity;
ordering consumes no model RNG. Do not reseed the model or reset the optimizer.
Save the schedule hashes before launch and retain full schedules in fit evidence.

Keep architecture, float32 CPU numerical operations, all trainable parameters,
policy/value loss coefficients (one each), AdamW LR 1e-5, weight decay 1e-4,
gradient clip one and every other accepted optimizer option unchanged. The mixed
value target is t=(z+0.837*q)/1.837, q=2*r-1, with original authenticated 512-search
visit-weighted root win credit r and official actor outcome z. Policy targets are
the same saved root visit distributions. Save both value outputs [t,-t].

New scientific updates: 5,346 per seed, 10,692 total. At most six disposable
qualification updates and six final-update reproductions make a maximum of 10,704
new optimizer calls. Keep pre-last and last checkpoints of each additional pass
for exact partial-batch reproduction. All four new assessment checkpoints must
be committed before fresh assessment predictions. No adaptive early stopping.

## Provenance and correctness

Use the independently accepted memory-mapped full-corpus cache and exact selection,
without copying the corpus or rebuilding historical games. Bind consumed arrays,
index, source acceptance, checkpoint manifests, saved predictions and transitive
runtime sources. Recheck those same consumed files at final acceptance. The prior
completed preservation audit covers unrelated historical evidence; do not repeat
an inventory of hundreds of thousands of unused game files.

Every new worker calls verify_runtime_sources exactly once on startup and records
the result. Bind ruleset splendor_base_qualified_rules_20261018_v1, catalogue
db4776918e612eee5a6f2617719c85ec5f413d592bbab0a64058babc83a70312,
turn contract atomic_prereveal_completed_turn_v1, and corrected generation origin.
Historical EF8A initialization is not labelled corrected training data.

Before continuation, run the actual corrected loader/cache/collator path on the
six corrected cards, noble-after-nonpurchase, multiple nobles, final round,
payment/return and private reserve fixtures and identity/capture rejection cases.
Exercise the fresh evaluator helper on these inputs. Reproduce each old final
30-example update exactly from step 1,781, matching original outputs, gradients,
losses, model tensors, AdamW and RNG at step 1,782. Then compare the next genuine
32-example update from uninterrupted restored state and separately saved/reloaded
state. Four additional forward passes qualify structural and genuine evaluator
inputs without inventing outcomes for structural fixtures. All qualification
updates are discarded. A separate new checkpoint schema
must authenticate continuation without changing old metadata validation or sources.
Run model-free failure/correctness tests and three source-bound independent reviews
before freezing the attempt. Failures remain recorded and prevent later stages.

## Assessment and reading rules

Reuse accepted parent and one-pass predictions after validating their exact input
and checkpoint bindings. Predict only the four new fixed two/four-pass endpoints.
Use the identical 3,072-position training panel (975 training games), all 3,533
validation roots and 3,553 test roots (64 games per split). Those 128 held-out games
have been inspected before: they are reused development diagnostics, not pristine
independent confirmation. The training panel is descriptive, not the full corpus.

Report each seed and the fixed mean of both seeds. Average within games, then
equally across games; for pooled held-out results stratify the 64/64 splits.
Report policy cross-entropy, teacher entropy and KL (CE minus teacher entropy),
policy top-choice agreement, official-outcome MSE, search-value MSE, mixed-target
MSE and absolute value drift from the parent. Report raw parent-to-model changes
and fixed adjacent changes one-to-two and two-to-four. Use 20,000 fixed whole-game
paired bootstrap draws for descriptive held-out uncertainty, preserving pairing
across models and seeds. These are not repeated significance tests for promotion.

Teacher-gap closure is (CE_parent - CE_model)/(CE_parent - H_teacher), computed
on identical roots with identical game weighting. Report denominator and raw KL;
mark near-zero-denominator percentages undefined rather than manufacture a gain.
This measures imitation of a fallible search teacher, not optimal play.

Prospective interpretation:
- Improving held-out errors through four passes supports further training under
  this recipe, followed by a decision-quality assessment; it does not prove strength.
- Falling training errors with stalled/worsening held-out errors indicates diminishing
  generalization. Game count, targets, regularization and loss balance remain possible
  causes; do not uniquely diagnose too few games or proven memorization.
- If neither improves, inspect optimization and target fitting before new collection.
- If policy and value disagree, report the tradeoff. Do not select the most flattering
  metric or seed. Evidence from two seeds does not quantify arbitrary-seed uncertainty.

The next candidate study is calibration of a cheaper decision-quality diagnostic
against known search-budget gains, then a separately designed fresh live assessment
if warranted. No automatic live launch, promotion, self-play or lower-value-weight
arm. Existing useful-gain criteria are unchanged; inconclusive is not noninferiority.

## Bounds, ownership and acceptance

New overall dual-clock cap 12 hours; all model work cutoff 8 hours. Qualification
1 hour; each continuation fit 3 hours; commitment 1 hour; fresh predictions 2 hours;
saved analysis 2 hours; final acceptance 2 hours. Each stage is clipped to the same
original global/model deadlines. No available-RAM reserve, forecast refusal or
stability wait. Actual worker-tree RSS limits are 4 GiB for model jobs, 8 GiB for
saved audits; actual new-output cap 16 GiB and free-disk floor 5 GiB. At most one
worker runs at a time. Earlier measured passes took roughly 11-12 minutes; fitting
should take about 68 minutes plus qualification, predictions and audits. This is
advisory, never an admission gate.

A hidden durable launcher owns the root, which owns each worker with Windows jobs,
exact PID/creation-time identities, original wall/monotonic limits and durable
observed exit receipts. Parent handshakes precede worker model work. Preserve all
nonzero exits, exceptions, partial updates and output. No automatic retry inside
this frozen attempt; diagnose technical faults separately without changing frozen
sources, discarding evidence or extending its clock.

After fits: saved exact-update audits, all-checkpoint commitment, predictions,
saved metric analysis and separate acceptance run automatically. Acceptance checks
every fixed checkpoint, update count, exact reproduction, prediction binding,
independent arithmetic, source preservation, prior actual exit and cleanup.
The root's completion file is insufficient: confirm durable actual root exit zero
and exact launcher/root/worker absence. Produce at most two learning-curve charts,
inspect them, update continuity and report findings and limitations. No further
experiment launches automatically.
