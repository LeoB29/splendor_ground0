# Bounded optimization of the largest independent-game corpus

Owner authorized on2026-10-15 after the completed independent_game_scaling_128
study. One experiment is authorized: continue its three largest-data models to
a fixed endpoint, establish TRAIN acquisition, and conditionally assess fresh
planning decisions. Implementation and outputs use large_corpus_optimization_128.
Preserve all frozen predecessors and the intentionally dirty worktree. No
automatic dose extension, arena, promotion, successor or monitor follows.

## Question and limits of the preceding result

The scaling study used720,2880 and11520 added positions with the same7136-update
dose. The largest arm gained+.114475 percentage points versus the anchor on720
fresh positions, paired95% interval[-.269111,+.533312], and+.359701pp versus the
small arm, interval[-.005425,+.764431]. Its prespecified joint gate failed. The
teacher retained+1.426324pp[+1.023220,+1.874457] of independently assessed gain.
These are conditional continuation scores, not complete-game win rates.

The largest models captured83.90%,80.12%,82.75% of added TRAIN regret and about
82-83% of old TRAIN regret. All were below the former90% added-acquisition guide.
Actual top-choice regret and target KL were still improving late in training
on both old and added identity-selected probes. This motivates a bounded dose
test; underacquisition alone would not establish that more optimization helps.
The medium arm acquired about97% without establishing fresh improvement, and
the small arm acquired about99% but had a negative point estimate. Earlier
well-acquired smaller-corpus and512-world-label experiments also failed transfer.

The present question is whether additional optimization of this existing
large-data package produces useful fresh proposals. It is not a clean causal
test of data coverage. Matching the medium arm's roughly79 added-row exposures
also gives the old pool substantially more exposure. A fixed high-dose endpoint
may overfit; a null cannot exclude every intermediate dose or representation.
Nevertheless, this bounded experiment has no automatic checkpoint or dose rescue.

## Inherited data and initialization

Authenticate the completed scaling study, binding
85bf68ca676a60610ab7c140afa6793573f3727df6a0d2ada273c35126891ead, its closed data,
model, fit, independent-reproduction, source and prescribed simulation receipts.
Reuse the exact9461 original breadth TRAIN rows and11520 added TRAIN rows. The
added pool contains the original720 independent-game core and10800 new roots,
one from each selected game. Do not add, remove, outcome-weight or relabel rows.
Retain all short supports and explicit zero-loss rows. All2489 original validation
rows remain outside optimization and checkpoint selection.

Training tensors contain only the unchanged lawful encodings and ORIGINAL
128-world targets. The later512-world targets, independent assessment outcomes,
and every earlier evaluation panel stay outside optimization. Retire the scaling
study's720 main positions and18 probe positions from confirmatory evaluation.
Saved evaluation evidence may provide explicitly labeled historical context only;
it cannot select the new endpoint, a fit, a subgroup, or whether a passing
acquisition result receives its prospective fresh evaluation.

For each original fit161001/161002/161003, start from its authenticated LARGE
model and complete AdamW optimizer state at update7136. Retain parameter ordering,
optimizer steps and moments, weight-decay settings, and frozen value-head tensors.
Do not reset the optimizer or restart from original161 weights. Authenticate the
saved optimizer checkpoint against the independently reproduced final7136 model.
Use original161 initializers only as controls and as the acquisition denominator.
Inherit completed historical audits after authentication; do not rerun old
learning trajectories, source games or continuation simulations.

The historical audit reproduced model tensors, diagnostics and gradient norms;
it did not independently compare the persisted AdamW moments. Bind the recorded
optimizer bytes in this new study and check their model/trajectory identity,
complete hyperparameters, parameter-ID/order membership, step counts, tensor
shapes/dtypes, finiteness and nonnegative second moments. The unchanged mature
checkpoint writer remains a trusted primitive for those historical moments.
Do not call them independently replayed. New continuation checkpoints, including
their optimizer states, are independently checked; bounded uninterrupted-versus-
resumed fixtures test the save/load and parameter-order contract.

## Exact fixed continuation

Finish at total update28544: exactly21408 new updates for each of three models.
Repeat the authenticated entire7136-step LARGE batch plan four times overall.
The first block is the already-completed trajectory; the next three blocks
repeat its exact ordered batches while retaining the evolving AdamW state.
Preserve each batch's old/added row identities,18-cell mixture, canonical
memberships, padding/bucketing convention and tie rules. Reconstruct the parent
plan independently and prove equality of every repeated block before fitting.
This is deliberate replay of the frozen course, not a claim to continue unseen
shuffle/cycling draws from an arbitrarily extended parent generator. The
1784-step batch-length pattern repeats16 times overall as a consequence.

The parent saved no RNG state. Its dropout-free evaluation-mode network and
fully precomputed batches introduce no stochastic fitting draws, so continuation
rests on exact model/optimizer tensors and the declared deterministic plan.
Do not silently add stochastic augmentation, dropout, fresh shuffling or an
optimizer reset in place of this contract.

There are1824704 row exposures over the complete28544-update schedule:
912352 old and912352 added. The new continuation adds1368528 exposures,684264
from each pool. Added rows receive about79.2 exposures overall; this is a dose
rationale, not a claim that equal exposures imply equal optimization or compute.
Report exact per-row, per-cell and total exposure accounting.

Preserve the original network, all policy/shared trainable parameters, frozen
value head, deterministic CPU float32 and one Torch thread per worker. Keep
AdamW learning rate1e-4, weight decay1e-4 and gradient clipping1. Keep the
50:50 old/added mixture and existing outsider soft-target objective:
softmax(original128-world expected terminal score/.05) over recorded candidates
outside the actual first three, with maximum concrete-action logits per canonical
group. Do not invent labels for unoffered actions. Empty or singleton outsider
sets keep their differentiable zero loss, batch denominator and optimizer steps.

Always select update28544. Save durable model/optimizer/schedule checkpoints and
fixed TRAIN-only diagnostic curves. Intermediate checkpoints are descriptive and
cannot select, shorten, extend or replace the final trajectory. Independently
reproduce all21408 new updates for all three models from the authenticated7136
optimizer states, checking schedule, objective, gradients/clipping diagnostics,
checkpoints, metrics and frozen tensors. Previously completed7136-step audits
are inherited rather than repeated. Authenticate all three final artifacts
before acquisition is decided. Create the evaluation model registry only after
the acquisition gate passes, before any fresh evaluation source generation.

## Acquisition gate before fresh simulation

Recompute full TRAIN metrics for each final model separately on all9461 old rows
and all11520 added rows. Use the corresponding ORIGINAL161 initializer's full
TRAIN regret as denominator, not the7136 continuation starting point. Capture is
one minus final regret divided by original-initializer regret, under the unchanged
original128-world outsider targets and the established zero-baseline convention.
Authenticate baseline reports and independently verify the acquisition arithmetic.
Keep full-corpus scope explicit; the small TRAIN probes cannot satisfy this gate.

All six conditions must hold without rounding:

- each of the three models captures at least95% of added TRAIN regret;
- each of the three models captures at least90% of old TRAIN regret.

Save the independent gate in acquisition_gate.json and bind its artifact from
learning_complete.json. All prescribed identity and fit-reproduction checks
must also pass. These
thresholds are prospective operational guides for whether to fund a fresh test;
they are not proven sufficient conditions for transfer. An acquisition miss,
including a narrow miss, stops this experiment before new source games or
evaluation continuations. Preserve and report all three models and diagnostics.
No extension, model selection, threshold relaxation, substituted panel or
automatic alternative study follows. If the gate passes, run the fixed fresh
test regardless of descriptive historical comparisons.

## Conditional prospective planning assessment

After the passing acquisition receipt and all final models are sealed, generate
2160 fresh main source games and54 separate probe games under the unchanged
original fast161 source population. Balance18 original-fit by acting-seat by
phase cells. Identity-select720 main positions from720 distinct games,40 per
cell, and18 separate probe positions, one per cell. Use new source seeds and
world namespaces; prove disjointness from the inherited training ancestry and
all previously consumed source panels. No outcome, runtime, disagreement or
teacher margin influences inclusion. A fixed-pool shortage holds; no replacement
or adaptive enlargement. Retain private reservations and short action supports.

Compare four fixed arms: the continued large model, saved large7136 control,
original161 anchor, and unchanged eight-candidate teacher. Each position uses
its assigned source-fit model; three fits do not triple the statistical sample.
Each learned family proposes its highest-scored full-legal canonical action
outside the unchanged common first three. Preserve concrete representatives,
canonical maximum pooling and ordered ties. The teacher retains the anchor's
four candidates plus two ranked and two identity-uniform extras, up to eight.

Evaluate the deduplicated union, at most10 actions, on128 common observer-legal
selection worlds using unchanged greedy035 terminal continuations. Durably seal
all choices before128 disjoint common assessment worlds for the at most four
distinct selected actions. Independent assessment outcomes must never choose
a proposal, action, checkpoint, fit or sample size. Retain mature conservative
bounds for unresolved repetition outcomes; never impute them as draws. Identical
selected actions cancel exactly in paired differences.

Use50000 paired bootstrap draws within the18 fixed cells, resampling whole
source games and keeping all arms together. Independently reconstruct means,
bounds, confidence intervals and the final decision. Require jointly:

- continued minus anchor mean lower bound at least+.005 score(+.5pp), with
  paired95% lower confidence endpoint strictly above zero;
- continued minus saved large7136 paired95% lower endpoint strictly above zero;
- all prescribed integrity and operational checks pass.

Teacher comparisons and per-fit results are mandatory descriptive controls,
not rescue endpoints. Do not select the best fit or subgroup. A joint pass
supports this fixed optimization continuation as a useful proposal improvement;
it does not isolate coverage as the cause or establish full-game strength,
superhuman play, or generalization beyond this source population and continuation
policy. Any gameplay comparison requires a separately proposed study.

## Verification, resources and recovery

Apply docs/engineering/verification_policy.md. Authenticate historical evidence
without rerunning exhaustive predecessor work. Independently check every new
training trajectory, schedule transformation, acquisition denominator/gate,
model/source link, lawful input, actual source decision, proposal and statistic.
Use full18-root probe replay, ordinary8/128 whole-world main replay, prespecified
metadata-selected stress roots bounded39, and every unresolved branch. Main
continuation verification is sampled/stress, not exhaustive. All output files,
source identities and stage-completion receipts remain immutable.

Run one hidden supervisor with one one-thread fitting worker or at most four
one-thread source/simulation workers. CPU only. Inhibit system sleep while
allowing display sleep; no visible shell or recurring monitor is needed. Freeze
a48-hour cumulative active-execution ceiling for THIS new experiment and a5-GiB
free-storage floor. The predecessor's60.36 hours remain historical evidence;
they are not charged again as new execution. A later explicit recovery carries
all active time already spent in this new study and excludes genuinely stopped
time. It must never reset the active-time budget or overwrite old receipts.

All normal/audit reports, source generation, evaluation and imported helpers
must use the current48-hour guard. Exercise hidden spawned workers beyond older
short limits and prove current48h/abort/storage enforcement. Price remaining
fresh evaluation from the fully audited18-root probe, including prescribed
replays, structural work and storage, with the frozen outcome-blind allowance.
Do not use measured strength to change resource or population decisions.

Save optimizer progress and completed game/root/audit receipts durably. On a
new operational or integrity failure, stop cooperatively and preserve completed
and partial evidence. Diagnose before an explicitly recorded recovery; no
automatic retry, checkpoint rescue, budget extension, arena, promotion or
successor. Report provisional completed-position comparisons when requested,
clearly distinguished from complete-game win rates and the final fixed result.
