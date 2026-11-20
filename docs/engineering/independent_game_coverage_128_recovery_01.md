# Controlled independent-game coverage experiment

The owner authorized this study on 2026-10-09 after the completed finite
soft-target study. Implementation lives only in tools/independent_game_coverage_128_recovery_01,
entered through run_independent_game_coverage_128_recovery_01.py, with a new output tree.
Preserve all dirty work and frozen predecessors. No automatic successor follows.

## Question and interpretation

Does distributing a fixed amount of practice across more independent games
improve learned proposals and planning on fresh positions? The existing 9,461
training positions come from 159 source pairs. Substantial training acquisition,
the higher-rate checkpoint curve, and finite soft targets did not establish
useful transfer. Independent reassessment still found useful teacher guidance.
This motivates a data intervention; it does not establish diversity as the cause.

The narrow and broad arms receive the same old-data examples, total exposures,
optimizer, source-population mixture and game-phase exposure. The added component
contains 180 versus 720 positions, each from a distinct game. Fourfold breadth
applies to this added component, not the whole training set. Each of three matched
model pairs uses a different prespecified narrow subset. Results are conditional
on these fixed subsets and initializers; three fits do not triple the test sample.

## Authenticated data and retirement of old evaluation panels

Reuse the closed full-corpus old TRAIN tensors and 128-world selection targets.
The 2,489 old validation rows remain outside optimization. Add the 360 main roots
from proposal_breadth_128 and 360 from equal_budget_breadth_128. Both studies used
fast161 source games, the same eight-candidate teacher recipe, 128 selection
worlds per candidate and greedy035 continuations. Authenticate their closed
collection/audit records, source games, model identities and exact target export.
Do not replay old simulation trajectories or repeat already-closed training.

Verify that the 720 selected source games are distinct across both panels and
disjoint from the inherited training/validation game ancestry. Independently
reconstruct lawful neural inputs, canonical grouping, the actual first-three
exclusion and every exported target. All added selection outcomes must be exact
terminal values; no censored score is imputed into a target. Preserve original
candidate order and concrete-action alias pooling. Full legal-action inputs are
used even when the objective covers only recorded outsiders.

Permanently retire all 720 consumed evaluation roots from future evaluation once
used for this study. A retirement record lists identities and provenance. Their
old published measurements stay valid as historical results, but these positions
cannot provide independent future evidence for descendants trained on them.

Each panel has 18 source-fit by acting-seat by phase cells, 20 rows per cell.
Order each cell by the fixed partition-domain identity hash and split it into
four disjoint five-row blocks. Assign blocks zero, one and two to original
initializers 161001, 161002 and 161003 respectively. Each narrow fit sees 180
added rows; each broad fit sees all 720. All fits pool all source fits in their
training data. No outcome, teacher gap, win rate or training fit selects a block.

## Six fixed fits

Start each matched pair from its original161 initializer. Use finite soft targets
over recorded actions outside the actual first three:

    q_j = (128 + old_sums_wdl[j]) / 256
    t = softmax(q / .05)
    loss = -sum(t * log_softmax(canonical_max_logits))

The temperature is a fixed score scale, not calibrated uncertainty. Empty and
singleton outsider sets contribute differentiable zero while retaining their
batch denominator and optimizer step. No filtering, temperature search, model
change, checkpoint selection or early stopping by predictive results is allowed.

Use deterministic CPU float32, one Torch thread per worker, fresh AdamW lr1e-4,
weight decay1e-4, norm clipping1, frozen value-head parameters and trainable
policy/shared representation. Retain exactly 1,784 updates and 114,044 row
exposures per model, using the old batch-length schedule: 1,772 batches of64
and12 of53. Allocate exactly57,022 exposures to old rows and57,022 to added rows.
Alternate the extra old/new slot in the odd batches. Both arms receive identical
old-row IDs and added source-cell IDs in every batch.

Old rows and added cell pools use deterministic shuffled cycles. The36 added
cells receive a balanced common exposure stream, with at most one exposure
difference between cell totals. Within a cell, narrow cycles over its five rows
and broad over all20. Pair corresponding added draws, bucket by the maximum of
their lawful input lengths, and bucket old draws by their lengths; a shared
shuffled batch-index order avoids a monotone length curriculum. This controls
attention padding without using outcomes or changing the matched exposure totals.
The independently reconstructed learning configuration records exact tags and
arithmetic. The old SOFT fit is a practical context control, not an identical-batch
causal control for this mixed-data schedule.

Always assess the final update. Initial/final diagnostics report target KL,
old-score regret, tied-best agreement, confidence, clipping, policy movement,
old/added exposure counts and the narrow exposed/unexposed added slices. Those
slices are development diagnostics, not fresh test evidence. Independently
reproduce all six new optimizer trajectories and confirm frozen value heads.
Seal all six models, original anchors, saved LOW-rate and saved SOFT controls
before any new evaluation game is collected. No result chooses a fit or subset.

## Fresh source population and evaluation

Generate 1,080 main and54 probe fast161 restricted self-play source games from
the three unchanged anchors. Source schedules are fixed, with60 main and3 probe
games per fit/seat/phase cell. Seeds are2255000000+index main and2256000000+index
probe; fixture seeds2258000000+index never enter production. Prove source-seed
disjointness against all inherited and repurposed source games. Replay and audit
every actual source decision, including private-information invariance.

Select by identity hashes only:20 main and1 probe games per cell, then one
eligible turn per selected game in its assigned acting seat and phase. Phases
are turns0-19,20-39,40+. This yields360 main positions from360 distinct games,
plus18 probe positions. Do not replace ineligible games or extend a short pool;
record a bounded operational hold. Fresh sources represent the declared fast161
population, not every state distribution visited by a deployed planner.

Six decision arms are anchor, narrow, broad, LOW-rate, original SOFT and teacher.
Each learned arm proposes its fourth action from the full lawful legal set,
preserving the common first three. The teacher begins with anchor's four and
appends two ranked and two identity-uniform additional actions, up to eight.
Use the fixed new teacher randomness domain in common.py. There are at most12
distinct actions in the union. All neural arms use the source-matched fit.

Evaluate every union action on128 common selection worlds. Preserve each arm's
candidate order for ties. Durably seal the choices and selection evidence before
generating128 independent assessment worlds for the at-most-six chosen actions.
World seeds are2250000000 + stage offset (0 probe,1000000 main) + index*512 +
block offset (0 selection,256 assessment) + world index. Selection and assessment
namespaces and all fresh source seeds must be disjoint.

Reuse the mature planner-state-transfer bounded-outcome and exact repetition
equivalence machinery. An unresolved repetition has unknown score in[0,1]; it is
not a draw. Certify an original-order selected action only if it remains selected
for every admissible unresolved value, including independently checked exact
equivalence where applicable. Otherwise preserve evidence and hold. Assessment
keeps conservative arm/difference bounds; identical chosen actions cancel
exactly. Replay every unresolved continuation in addition to the fixed sample.
This does not change the game rules, continuation policy or termination contract.

## Prospective analysis and investment decisions

Use paired, stratified resampling of whole source games:18 fit/seat/phase cells,
20 games per cell,50,000 bootstrap draws, seed2253000000, chunk1,000. Retain all
arms jointly in each resample. Report all seven contrasts: broad-anchor,
broad-narrow, broad-LOW, broad-SOFT, narrow-anchor, narrow-SOFT and teacher-anchor.
Independently reconstruct point estimates/bounds, confidence endpoints and gates.
For unresolved outcomes, the95% interval is the2.5th percentile of lower-bound
bootstrap means through the97.5th percentile of upper-bound means; no imputation.

The primary practical screen passes only if BOTH:

- Broad-minus-anchor mean lower bound is at least+.005 score (+.5 percentage
  points), and its paired95% lower confidence endpoint is strictly positive.
- Broad-minus-LOW-rate paired95% lower endpoint is strictly positive.

Attribute benefit to greater added-game coverage only if broad-minus-narrow also
has a strictly positive paired95% lower endpoint. This is an additional mandatory
condition for that causal claim, separate from the practical-package classification.
Shared gains by both new arms without this contrast support the new-data/batching/
reweighting package only. Other contrasts are descriptive; no subgroup or alternate
arm rescues the primary gate. A miss does not establish equivalence or rule out
all data scaling. These are conditional planning scores, not game win rates,
fresh full-game strength, promotion or proof of superhuman play.

## Verification and bounded execution

Follow docs/engineering/verification_policy.md. Check all new source games,
positions, full-legal proposals, hidden-input invariance, seeds, candidate budgets,
terminal/censored fields, sums, original-order selection and durable seals.
Inherit authenticated completed evidence; do not repeat old trajectories.

The18-root probe receives exhaustive continuation replay. Main roots receive8
identity-selected whole worlds per128-world block, retaining every offered action
in each selected world. Add full replay at metadata-selected stress roots using
the established39-cell marginal scheme: per fit, six seat/phase cells, four private
reserve categories and three short-support cells. This prospective scheme uses
separate privacy and seat/phase margins, not their full Cartesian product. Every
root jointly covers all six decision arms. Report missing cells and deduplicate
stress roots, ordinary worlds and extra unresolved cells. Main replay is sampled,
not exhaustive. New targeted fixtures cover ties, short supports, private reserves,
refills, bounds, same-choice coupling, malformed seals and actual hidden workers.

After model fitting, source collection and the fully checked probe, forecast main
collection/audit cost using measured per-continuation times, frozen main support
upper bounds, actual stress plan, structural cost, a1.35 allowance and300-second
finalization reserve. The replay projection covers the prescribed sample/stress
work; extra unresolved branches outside that sample are additional overhead
covered by the allowance and live ceiling, not a claimed total replay upper bound.
Probe choices/scores cannot change scientific decisions.
Maximum main continuations829,440; maximum probe41,472, before verification.
The forecast and the entire live process must fit an eight-hour execution ceiling
and retain5GiB free disk. Expected execution3-6 hours plus preparation is an estimate.

Run one hidden detached supervisor, three fitting/audit workers and at most eight
source/simulation workers, all CPU. Inhibit system sleep while allowing the display
to sleep. No visible terminals, GPU use, recurring monitor or active chat is needed.
Use bounded task dispatch and cooperative aborts so a failed task cannot drain an
unbounded pending queue. Preserve partial evidence on any hold/failure; no automatic
retry, altered endpoint, extra games, arena, successor or promotion is authorized.
Freeze code, protocol, runtime identities, parent inputs and passing preflight
before production. Report provisional completed-position scores on user request,
clearly marked before prescribed verification has finished.

## Operational recovery 01

The first frozen launch on2026-10-09 stopped before any production fitting,
evaluation source game or simulation. A deliberately failing preflight fixture
monkeypatched the runner output path, but an abort helper's definition-time
default still referenced the production folder. Its saved stop signal correctly
blocked the supervisor. Preserve that complete frozen version and all evidence.

This separate version resolves the abort destination at call time, passes the
active output path explicitly in orchestration, and checks before and after
preflight and before preparation/launch that no runtime-control evidence exists.
Regression tests exercise the formerly leaking path. Scientific settings, source
identities, splits, batches, original model initializers, RNG domain tags/seeds,
world allocation, endpoint, gates and compute ceilings are unchanged. No new
training/evaluation outcomes were available to motivate or select this recovery.
A new complete passing preflight, source seal and immutable manifest are required.
