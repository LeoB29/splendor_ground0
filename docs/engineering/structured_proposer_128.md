# Fully trainable action-card relations on the existing large corpus

Authorized by the owner on 2026-10-16 after the completed
large_corpus_optimization_128 study. New implementation and output namespace:
structured_proposer_128. Preserve every frozen predecessor and the intentionally
dirty workspace. The authorized bounded study includes six fixed fits, independent
learning reconstruction, an acquisition gate and, only on a pass, fresh planning
assessment. No automatic dose extension, arena, promotion, successor or monitor.

## Scientific question and preceding evidence

Extra optimization of the existing large corpus reached 97.7-98.1% added TRAIN
regret capture, yet its fresh conditional gain was -.047743 percentage points
against the original anchor, paired 95% interval [-.423177,+.324978]. The stronger
teacher retained +1.207682pp [+.866428,+1.566840]. This weakens inadequate fitting
as the sole explanation for that recipe's failure to transfer; it does not
identify the cause or prove a universal data, optimization or learning limit.

The present question is whether making action-to-card resource relationships
explicit helps generalization when the existing policy representation and added
branch can both adapt. Test one relations treatment against a paired raw control
with identical architecture, capacity, lawful information, initialization,
training examples, objective, batch stream and optimizer settings. The arms
differ only in twelve deterministic feature channels. Both contain shared card
processing: this is a narrow test of explicit relations at full trainability,
not a first test of shared-card architectures or a repair of missing information.

Earlier shared-card models failed in smaller, different settings. The recent
action_card_relations_128 add-on froze the entire base network, trained only
9,441 new parameters and acquired about 10-15% of its added training signal.
Those results are real negative evidence but leave this specific combination
unresolved. A well-acquired fresh null here should lower priority of this
teacher-compression route, rather than license another nearby automatic sweep.

## Matched architecture and initialization

For each original fit 161001, 161002 and 161003, authenticate and load the original
161 policy initializer. Attach the validated action-card branch in both arms.
Its last scalar layer has exactly zero weight and bias, so initial policy logits
match the original network. Pair every initial parameter tensor and seed across
raw and relations. Both arms train all base policy/shared parameters and all
branch parameters; only base value-head parameters stay frozen. Frozen value-head
tensors do not imply constant value predictions after shared features change.
The proposal experiment does not train a value objective or use changed values.

Reuse the original observer-legal 295-state/79-action information. There are
18 known-card slots: twelve market, three own reserved and three opponent public
reserved cards. Mask absent and unobserved cards. Preserve before-action card
identity, ownership/location before and after, availability and action target.
No opposing private identity, hidden deck order, unknown refill or blind-reserved
identity enters the features. Purchased-card identities remain omitted, equally
in both arms; this experiment cannot diagnose that separate information question.

Each action/card pair has 55 channels. The 43 common channels retain the thirteen
card features, three ownership indicators before and after, target and after-
presence flags, actor colored/gold tokens before and after, and colored bonuses
before and after. The twelve contrasted channels are:

- Raw: five colored token-plus-discount capacities before and after, and gold
  before and after.
- Relations: five nonnegative colored deficits before and after, and total
  remaining shortfall after subtracting the shared gold pool once, before and
  after.

Both arms retain all raw constituents in their common channels. Deterministic
effects include exact payments, takes, returns, new discounts, purchased-card
removal and reservation movements. Shared pair network 55->32->32 with ReLU,
masked mean/max pooling into 64 channels, concatenated with the full 79 action
and 61 public context channels, then 204->32->1. The branch adds exactly 9,441
parameters in each arm. Bind the exact base and total parameter inventories,
feature schema and model configuration before production fitting.

The independent scalar feature builder and rules-based fixtures check payments,
gold, reservation movement, removal, bonuses and private-information invariance.
Additional learning fixtures must prove gradients and optimizer updates reach
base policy/shared tensors and the branch while value-head tensors stay intact.

## Fixed data, objective and training trajectory

Authenticate completed independent_game_scaling_128 evidence, binding
85bf68ca676a60610ab7c140afa6793573f3727df6a0d2ada273c35126891ead, together with its
closed data, initializer and prescribed audit ancestry. Reuse exactly 9,461 old
TRAIN rows and 11,520 added TRAIN rows (720 original core plus 10,800 additional
independent-game roots). No new labels, relabeling, row removal or outcome-based
weights. Retain short supports and explicit zero-loss rows. Keep all 2,489 old
validation rows and every prior evaluation panel outside optimization and model,
checkpoint, dose or feature selection. All reused evaluation data are development
evidence; none is confirmatory here.

Start each of the six new fits from its original initializer plus paired zero-
output branch and a fresh AdamW optimizer. This is not continuation of the old
7136 or 28544 model or its moments. Reconstruct the authenticated LARGE 7136-step
batch plan and repeat it exactly four times: 28,544 updates per fit. Reconstruct
the plan independently and prove all four block identities before fitting.
Preserve row ordering, lengths, bucketing, canonical membership and tie rules.
This is deliberate replay of a fixed course, not newly shuffled extra epochs.

Each fit receives 1,824,704 row exposures, 912,352 per pool; added rows receive
about 79.2 exposures on average. Both arms use the same 50:50 old/added mixture,
original train seeds, deterministic CPU float32, one Torch thread, evaluation-
mode network computation without dropout, fresh AdamW learning rate 1e-4, weight
decay 1e-4 and gradient clipping at norm 1. Evaluation mode does not freeze
policy gradients. Report exact total, row and cell exposure accounting.

Keep softmax(original 128-world expected terminal score / .05) over recorded
canonical candidates outside the actual first three. Canonical scores remain
maximum concrete-action logits. No target is invented for unoffered actions.
Empty/singleton outsider sets retain differentiable zero losses, batch denominator
and optimizer steps. A memory-bounded implementation may split an exact batch
into fixed-order pieces only if it preserves this full-batch objective and one
optimizer update, records that execution contract and independently reproduces
it; it may not silently alter the scientific batch schedule.

Always select update 28,544. Save durable model, optimizer, schedule and RNG
checkpoints every 256 updates and at prescribed diagnostic endpoints. TRAIN
curves at 0, 7136, 14272, 21408 and 28544 are descriptive, never selection rules.
Full old/added diagnostics at the start and final endpoint determine acquisition;
identity-selected intermediate probes cannot substitute for full-corpus metrics.
Independently reproduce each entire new trajectory, including model and optimizer
checkpoints, parameter membership/order, objective, gradients/clipping, RNG state,
diagnostics and frozen value-head tensors. Test exact interrupted/resumed equality
on bounded fixtures. Authenticate all six final models before deciding acquisition.

## Full TRAIN acquisition gate

For each arm and each fit, compute full old and full added regret capture as one
minus final regret divided by the corresponding original 161 initializer regret,
using original 128-world targets and the established zero-baseline convention.
An exactly zero output branch preserves this denominator. Independently verify
every numerator, denominator, corpus identity and arithmetic result.

All twelve requirements must hold without rounding: each of the six models must
capture at least 95% of added TRAIN regret and at least 90% of old TRAIN regret.
All training and identity audits must also pass. Save acquisition_gate.json and
bind it from learning_complete.json. These are prospective investment criteria,
not evidence of fresh strength or proven sufficient conditions for transfer.

If either arm misses any requirement, finish acquisition_not_established before
generating fresh source games or continuations. Preserve all six models and
diagnostics. Do not select a fit, extend the dose, lower the threshold or substitute
another endpoint. If every requirement passes, seal the nine-model evaluation
registry (three anchors, three raw and three relations) and automatically proceed
to the fixed fresh assessment regardless of prior descriptive comparisons.

## Fresh conditional planning assessment

After the passing acquisition receipt and sealed models, generate 54 probe and
2,160 main source games under unchanged original fast-161 restricted self-play.
Balance the eighteen original-fit by acting-seat by phase cells; phases are
turns 0-19, 20-39 and 40+. Identity-select one probe and forty main roots per cell,
giving 18 separate probe roots and 720 main roots from distinct games. Audit
every actual source move. Inclusion cannot depend on outcomes, runtime, teacher
margins or disagreements. Retain private-reserve and short-support cases. A fixed
pool shortage holds; no replacement games or adaptive enlargement.

Fresh source seeds are 2706000000+index for probe, 2707000000+index for main and
2708000000+index for fixtures. Evaluation seed base is 2700000000 plus stage offset
(0 probe, 1000000 main, 2000000 fixture), index*512 and block offset (0 selection,
256 assessment), plus world index. Bootstrap seed 2703000000 and audit seed
2704000000. Bind separate assignment, source-stream, panel-selection, teacher-
random, world and audit domain tags. Prove disjointness against all inherited
training ancestry and consumed source panels, including the just-completed
large-corpus optimization study and both newly declared source/world domains.

Four arms: anchor, raw, relations and teacher. Preserve the common original first
three candidate moves. Each learned family proposes its highest full-legal
canonical action outside those three. Preserve concrete representatives, max
grouping and ordered ties. Teacher retains the anchor's four plus two ranked
and two identity-uniform alternatives, up to eight. Each root uses its assigned
source-fit models; three fits do not triple the statistical sample. The union
has at most ten distinct actions.

Evaluate every union action on 128 common observer-legal selection worlds with
unchanged greedy-035 terminal continuations. Durably seal all arm choices before
128 disjoint common assessment worlds for the at most four distinct chosen
actions. Independent assessment outcomes never choose a proposal, checkpoint,
action, fit or sample size. Retain mature conservative bounds for unresolved
repetition outcomes; never impute them as draws. Identical selected actions cancel
exactly in paired differences. Ambiguous selection uses only the established
equivalence procedure or produces an operational hold.

## Prespecified statistics and interpretation

Use whole source games as independent units, jointly resampling all arms within
the eighteen fixed cells, forty games per cell, for 50,000 paired bootstrap
replicates. Independently reconstruct exact score arithmetic, means, bounds,
intervals and final decision. Required contrasts are relations-anchor,
relations-raw, raw-anchor, teacher-anchor and teacher-relations. Require jointly:

1. Relations-anchor mean lower bound at least +.005 expected score (+.5pp).
2. Relations-anchor paired 95% lower confidence endpoint strictly above zero.
3. Relations-raw paired 95% lower confidence endpoint strictly above zero.
4. All prescribed integrity and operational checks pass.

Teacher and per-fit comparisons are mandatory descriptive controls, not rescue
endpoints. Never choose the best fit or subgroup. Score is win=1, draw=.5,
loss=0 from conditional continuations; differences are percentage points, not
Splendor prestige, Elo or full-game win rates.

A joint pass supports the fixed explicit-relations recipe over matched raw
features on this source population and planning procedure. It does not establish
the first benefit from shared card processing, identify missing information as
the earlier failure, prove full-game strength or imply superhuman play. Shared
raw/relations gains would support the combined trainable-branch recipe without
isolating the twelve relation channels. A well-acquired null weighs against this
particular local structural remedy; a failed acquisition gate limits conclusions
about transfer. No automatic architecture, target, budget or continuation rescue.

## Verification, resources and interruption handling

Apply docs/engineering/verification_policy.md. Authenticate historical work
without rerunning predecessor learning or simulations. Independently check all
new learning trajectories, features/lawfulness, acquisition arithmetic, models,
source decisions, proposals, choice seals, assessment links and statistics.
Fully replay all eighteen probe roots. At every main root replay eight of 128
whole worlds per block, plus the prespecified metadata-selected full stress
subset (at most 39 roots) and every unresolved branch. Check all recorded entries
structurally. Main continuation replay is sampled/stress, not exhaustive.

Run one hidden detached supervisor, one one-thread fitting/reproduction worker
or at most two one-thread source/simulation workers. CPU only, no visible shells.
Inhibit system sleep while allowing display sleep. Freeze preflight, protocol,
code/dependency inventory, data/model identities, seeds and execution settings
before production. The cumulative active ceiling is 96 hours and free-storage
reserve is 5 GiB. The ceiling is an operational bound, not an ETA; historical
predecessor runtime is not charged again. Explicit recovery must carry all active
time already spent here and exclude genuinely stopped time, without resetting
the budget or rewriting old receipts.

All normal/audit learning reports, imported helpers, source generation and
evaluation paths must use the current guard. Actual hidden-worker fixtures at
24 and 48 active hours exercise those paths and prove that stale ancestor guards
are unreachable; separate checks enforce the current 96-hour limit, storage
floor and cooperative abort. Fixture failures must never write production aborts.
Before production, time both arms on 32 fixed, evenly spaced batch indices from
0 through 7135, after two warm-up updates, using the first original fit and the
unchanged TRAIN tensors. In addition, select the first batch maximizing batch
length times maximum legal-action length over the 7136-step prefix, using only
shape metadata. Run one extra optimization step on that fixed padding-stress
batch after the 32 timed steps in each arm; exclude it from mean-time projections.
Record process resident memory, peak resident memory and available system memory
after loading the full corpus and after each padding-stress step. Discard both
temporary models and optimizer states. Record timing and serialized checkpoint
size, without predictive metrics or production-training credit; no learning
criterion is inspected. Project six full fits plus six independent replays
using the slower arm's mean update time, a 50% allowance, two hours for feature
checks/diagnostics/I/O and 24 hours reserved for subsequent fresh evaluation.
Project all saved optimizer checkpoints with a 25% storage allowance plus 5 GiB
for other new artifacts, leaving the separate 5-GiB free-space reserve. Freeze a
passing resource receipt bound to identical code/data identity before launch.
The first outcome-blind preparation probe projected 54.399 hours for learning,
including independent reproduction and allowances, plus the 24-hour evaluation
reserve: 78.399 hours total. This exceeded the tentative 72-hour ceiling, so the
operational bound was raised to 96 hours before freezing, with dose, architecture
and success criteria unchanged. Preserve the initial resource receipt and require
a passing receipt for the final frozen identity. Once frozen, a forecast miss
holds rather than silently extending the budget.

After the fully audited evaluation probe, price remaining collection,
prescribed replay, structural work, finalization and storage with the frozen 35%
allowance. Strength outcomes cannot change resources or population decisions.

Save model/optimizer progress and completed game, root, audit and stage receipts
durably. On new operational or integrity failure, stop cooperatively and preserve
all partial/completed evidence. The runner is one-shot: no automatic retry,
extension or replacement. Diagnose and authenticate before a separately recorded
explicit recovery. Detached execution does not require an active chat but cannot
survive a machine shutdown; recovery support does not mean a silent restart.

Finish with readable result/report and final status, no successor. On requested
status, report available completed-position differences as provisional conditional
scores. During fitting, acquisition is a training diagnostic rather than win rate.

