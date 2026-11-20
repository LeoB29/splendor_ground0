# More independent games at matched training effort

20 November 2026. **Prepared design only; no training or evaluation is launched.**
The user requested a bounded comparison using the accepted 4,096 training games,
with original EF8A retained as the reference and fresh assessment sealed until
checkpoints are fixed. This implements the follow-on design specified before
collection in `splendorust_collection4096_preparation_20261115.md`.

## Question and comparisons

Does more independent experience improve learning when the amount of training is
held fixed? Repeating the smaller corpus more often is the control. This is an
offline learning study, not a playing-strength test or model promotion.

| Arm | Independent training games | Positions presented per fit | Updates per fit | Fits |
|---|---:|---:|---:|---:|
| Larger corpus, one pass | 4,096 | 227,203 | 7,101 | Two seeds |
| Original corpus, repeated | 1,024 | 227,203 | 7,101 | The same two seeds |

Use seeds **202611071 and 202611072**. All four fits start independently from
pristine historical EF8A and a fresh AdamW optimizer. Both heads and the shared
layers are trained jointly. Keep EF8A unchanged throughout. The two previously
accepted 1,024-game one-pass mixed-target checkpoints are additional practical
baselines, not substitutes for the new matched-effort controls. Evaluate these
seven fixed models: EF8A, two historical one-pass models and four new endpoints.

The duration study found held-out policy gap closure of 4.81%, 4.86% and 2.76%
after one, two and four passes while training fit kept improving. The data-size
study found 1.05%, 4.11% and 4.81% closure at matched updates on 256, 512 and
1,024 games. These reused-development results motivate the design; they do not
predict the size of the new gain. Do not add duration, learning-rate, value-weight,
teacher or architecture arms to this comparison.

## Accepted sources and exact exposure

The accepted training union is
`outputs/splendorust_collection4096_resume_20261118_02/training_union.json`, SHA256
`f3ee04d027eb767de56ae1cc503accbaa42c57f8c84a0358a06108c5c83c8f18`.
Its 1,024 prior games supply **57,022** eligible positions and 3,072 new training
games supply **170,181**, giving **227,203**. Counts and schedules are obtained
only from accepted catalogue metadata; no fresh assessment target or game record
is needed for preparation. The union's amended recovery, eight disclosed gameplay
retries and one inherited verification retry remain visible in provenance.

Namespace every game and row by `(source, game_index, ply)`. The two sources reuse
game numbers; joining them by `g0000` or numeric game index alone is forbidden.
The planned schedule artifacts bind the accepted root-input/row hashes.

Each fit has **7,100 batches of 32 and one genuine batch of 3**. Mean losses use
the actual batch length. No padding, dropping or extra endpoint is permitted.
Total scientific exposure is 908,812 record presentations and 28,404 updates.

For each seed, order all union rows by a prospective SHA256 of study domain, seed
and namespaced row identity, with canonical identity as a collision tie-break.
The control's first pass filters this order to prior-source training rows. Later
control passes use independent prospective domain/seed/pass/identity hashes.
Concatenate the control passes, truncate to 227,203 rows, and **only then** slice
the identical batch-length vector. This gives three complete old-data passes and
56,137 rows of the fourth. Old pass boundaries cross batches; do not insert their
usual 30-row partial batches. The historical four-pass endpoint (7,128 updates,
228,088 presentations) is not this control.

Schedule construction consumes no Python, NumPy or Torch model RNG. Preserve all
four orderings and their hashes before fitting. Fixed serial order balances arm
order across seeds: larger r0, control r0, control r1, larger r1. No intermediate
prediction or scientific result may change this order or any fit.

## Unchanged numerical recipe

Historical warm start: `outputs/external_opponent_search_20261018_02/splendorust_model.pt`,
SHA256 `ef8a4521cd6c03c7075f15efee23f4cde6ec94d1b5398765cf0c24a09ad745cb`.
Keep the accepted 4,780,883-parameter model, CPU float32 graph, batch collation,
corrected legal policy mask and all trainable parameters. AdamW: learning rate
1e-5, weight decay 1e-4, betas (0.9, 0.999), epsilon 1e-8, amsgrad/foreach/fused
false; gradient norm cap 1 with foreach false. Policy and value losses have equal
weight; value loss averages the two signed outputs. No optimizer restart or
epoch reseeding occurs within a fit.

Policy targets are the original authenticated search-visit distributions. Use
the accepted mixed value target `t=(z + 0.837*q)/1.837`, `q=2*r-1`, outputs
`[t,-t]`. Here `z` is the official actor-oriented terminal outcome and `r` is
the saved actor-oriented, visit-weighted 512-search root win credit. Never replace
it with the chosen action value, an unweighted action average or a new prediction.
The data teacher stays EF8A, 512 simulations, three worlds, cpuct 0.4 and depth 64.

Retain steps 0, 7,100 and 7,101 of each fit, including full optimizer and RNG state.
Reproduce each final three-example update exactly from its pre-last checkpoint:
outputs, gradients, losses, model tensors, AdamW and RNG must match. Four replay
updates are additional to science. Allow at most 16 discarded qualification
updates, for **at most 28,424 optimizer calls** overall. No checkpoint selection,
early stopping, best-seed selection, silent restart or replacement fit.

## Sealed assessment and corrected data route

The 256 new assessment games comprise 128 labelled validation and 128 labelled
offline. **Pool them prospectively for one primary offline assessment. Neither
partition selects a duration, recipe, checkpoint or seed.** The old 128 development
games are already reused evidence and are not part of this fresh assessment.

Before commitment, only assessment membership/count/hash metadata may be accessed.
No fresh assessment game, position, outcome, target, encoded cache, model output
or aggregate statistic may be loaded. This applies to builders, qualification,
auditors, evaluators and diagnostic code, not merely the prediction function.

The old full-corpus builder eagerly encodes held-out examples. **Do not use it
unchanged.** Build a separately versioned training-only adapter and cache, with
an explicit train allowlist. Reuse the authenticated prior cache's training rows
through a qualified namespaced view; decode and independently check every new
training row against its original corrected source. A required full rebuild is
permitted within the same caps and must produce identical rows, but must remain
training-only. Use memory-mapped shards, not a corpus-sized Python feature list.

Require one durable commitment containing all four new endpoint hashes, their
passed exact-update replays and actual clean fit exits, both historical baseline
hashes, EF8A's hash, the complete evaluation recipe and the assessment membership
binding. Only after independent commitment validation may a separate worker open
and encode the two new assessment partitions. Missing, partial, stale, swapped,
changed or incomplete commitment must fail before a protected file is read.
After opening assessment, no refit, endpoint replacement or recipe revision is
allowed in this study, including a technical recovery that would change a model.

Future package: `tools/splendorust_training4096_20261120`; intended attempt:
`outputs/splendorust_training4096_20261120_01`. Both remain unimplemented/unlaunched
at preparation. Adapt the qualified corrected 7 November loader, 10 November
mmap/numerical route and 13 November numerical trainer/evaluator in new files.
Requalification is required; importing old code or copying metadata is insufficient.

Every eventual worker calls `verify_runtime_sources()` exactly once at startup.
Bind ruleset `splendor_base_qualified_rules_20261018_v1`, catalogue
`db4776918e612eee5a6f2617719c85ec5f413d592bbab0a64058babc83a70312`, and turn contract
`atomic_prereveal_completed_turn_v1` throughout source, cache and checkpoints.
EF8A is explicitly a historical warm start, not certified corrected training data.

Qualification must exercise the actual decoder/cache/mmap/collator/trainer and
evaluator on all six corrected faces, non-purchase/multiple/final-round nobles,
gold payment/returns, private reserve reconstruction, complete card faces and
explicit corrected legality. Reject legacy identity, changed hashes, malformed
captures, wrong actor orientation, nonfinite targets, split contamination and
duplicate/missing namespaced rows. Use existing engineering/training fixtures,
never the fresh assessment games. Preserve the existing turn and internal search
conventions; unofficial cutoff/deadlock outcomes are not training labels.

Also qualify exact reuse of the prior cache, original saved-update parity,
full-32/final-3 reduction, control pass-crossing batches, direct versus saved-state
execution, checkpoint import and evaluator parity. Model-free gate tests must
prove protected reads do not occur before all endpoints are committed. Before
implementation freeze, independent reviews cover scientific matching, data/seal
correctness and ownership. A helper-only rules selfcheck is insufficient.

## Measurements and interpretation fixed before opening assessment

Evaluate all seven fixed models on every eligible root of all 256 fresh games.
Primary contrast: policy cross-entropy of the larger-corpus models minus that of
the matched-effort controls. Average the two seed-specific **losses**, not their
predictions; this is not an ensemble. Average roots within each game, then games
equally. Report each seed and the fixed two-seed mean. Use 20,000 paired whole-game
bootstrap draws with fixed seed 202611201, stratified by the two 128-game partitions.
Use the same sampled game IDs for every model and contrast; seeds are not treated
as additional independent games. The primary contrast has a 95% interval.
"Clearly improves" means that this entire interval lies below zero. Seed-specific
directions are reported separately, not introduced afterwards as an extra test or
used to select one model. An interval spanning zero is unresolved.

Secondary contrasts compare the larger arm with EF8A and with the historical
one-pass baseline (same optimization seed where applicable). Report descriptive
95% intervals for each. If making inferential claims about either secondary
policy contrast, also report Bonferroni 97.5% intervals for that two-contrast family,
explicitly separate from the prespecified primary test. Other metrics are
descriptive; do not turn the most favourable one into a replacement primary result.

Report policy CE, teacher entropy, KL, top-choice agreement, official-outcome MSE,
search-value MSE, mixed-target MSE and absolute value drift from EF8A. Teacher-gap
closure is `(CE_EF8A-CE_model)/(CE_EF8A-H_teacher)` on the same game weighting.
Report its denominator and mark it undefined when the parent KL is at most 1e-8.
Do not treat teacher entropy or search repeatability as a certified performance
ceiling; the teacher is fallible. No numerical offline gain proves stronger play.

For descriptive training errors, use a fixed outcome-blind panel of up to four
roots per training game, selected by metadata hash before fitting, common to all
models. Report the prior 1,024 and new 3,072 game panels separately, plus their
game-weighted union. New-source training rows are unseen by the small control but
are not renamed independent assessment data. Do not use this panel for fitting
decisions. No additional assessment of old development games is required.

| Outcome | Reading and recommendation |
|---|---|
| Larger corpus clearly improves the primary contrast | Supports an independent-data benefit at this training effort and recipe; report seed consistency and value tradeoffs. |
| Better only than the repeated control, not the historical one-pass baseline | Evidence of mitigating excess repetition; useful practical progress remains unestablished. |
| Better than EF8A and the practical baseline, with favourable primary evidence | Supports a separately designed live strength test retaining both fixed larger-corpus seeds. It does not authorize one automatically. |
| Primary difference unresolved, but practical comparisons look favourable | Data attribution remains unresolved; the practical candidate can still merit consideration. Do not equate uncertainty with equality. |
| Policy improves while value worsens | Report the tradeoff; do not claim overall success or discard the unfavourable metric. |
| No useful improvement | Keep EF8A; reconsider the recipe/targets before another expensive collection. This is a valid result, not a technical failure to retry. |

Two fixed seeds and one nested corpus do not estimate arbitrary-seed or
arbitrary-corpus uncertainty. Selection of new checkpoints based on these games
would spend their independence for subsequent assessment. This study does not
change any existing live-promotion criterion, prove noninferiority or certify
superhuman strength. The previous search-based proxy was not sufficiently
sensitive to replace live games for modest learned-model gains.

## Proposed bounds and automatic sequence after separate launch authorization

Expected machine time: **12–18 hours**, principally data preparation/checking
(about 7.3–9.8h), four fits (3.1–4.8h), then assessment and final audit. These are
extrapolations from measured accepted workers, not an admission forecast.

Overall cap **48h**; all model work cutoff **36h** from the same original launch.
Stage ceilings: qualification 2h; training-cache build 12h and independent row
audit 12h; schedules/commitment 1h each; each fit including its replay 3h;
post-commit assessment-cache build/audit 2h each; predictions 3h; analysis 3h;
final independent acceptance 6h. All are clipped to the original dual UTC and
monotonic bounds. Stage ceilings do not add to or extend the overall clock.

Use one worker at a time, the accepted single-thread CPU numerical route, model
on logical CPU0 and supervision on CPU2 after verifying distinct physical cores.
No GPU change, concurrent fitting or throughput pilot. Actual model/evaluation
RSS cap 4GiB; cache/audit 8GiB. Expected additional artifacts about 3–6GiB;
cap 16GiB, including a 2GiB finalization reserve, and retain a 5GiB actual free-disk
floor. There is **no available-RAM admission threshold, stability wait or forecast
refusal**. Stream data and re-use authenticated caches within these actual bounds.

The eventual owner runs qualification, training-only cache acceptance, all four
fits/replays, commitment, assessment-cache acceptance, fixed predictions, saved
analysis/charts and independent final acceptance automatically. Preserve small
stage/progress/failure receipts. Use the corrected native Windows Job ownership
route from the accepted 18 November v2 recovery, exact process identities and an
external observer of actual launcher exit; no recursive ancestry-based ownership.
Do not create a scheduled chat monitor. Source-bound model-free tests, independent
reviews and actual detached success/failure cleanup fixtures precede freeze.

Record all failures and partial artifacts; stop incomplete on a technical failure.
No automatic fit retry, assessment reopening with changed checkpoints, source edit,
clock reset or budget extension. A separately qualified technical correction may
reuse completed verified fits only under a separately authorized recovery. A valid
negative result completes the study. Recheck all consumed source/cache/checkpoint/
prediction hashes at final acceptance and preserve original artifacts. Reuse the
already accepted corpus preservation receipt for unrelated collection evidence;
do not repeatedly inventory all unconsumed historical game files.

Completion requires independent arithmetic and provenance checks, all exact-update
replays, actual worker/root/launcher exits and clean ownership. Produce at most two
charts: policy-gap/primary-contrast results, and value or generalization tradeoffs
if informative. Update continuity and retain EF8A as reference. No new collection,
live strength match, promotion or further study launches automatically.

## Preparation deliverables and readiness

`outputs/splendorust_training4096_preparation_20261120_01` holds a metadata-only
proposal, source bindings, deterministic training-order files and consistency
checks. It is not an execution freeze or runtime qualification. The prospective
training package must still be implemented, tested and independently reviewed;
actual-model qualification begins only after a separately authorized launch.
No fresh assessment game/target/prediction was opened in this preparation.
