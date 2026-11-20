# Matched action-card relations and fresh planning assessment

Authorized by the owner on 2026-10-11 after
docs/strategy/next_after_control_variate_128.md. New code/output namespace:
tools/action_card_relations_128 and outputs/action_card_relations_128; entry
run_action_card_relations_128.py. Preserve every frozen predecessor and dirty
workspace file. This study includes preparation, six fits, independent training
reproduction, fresh source games, a timing probe and fresh decision assessment.
No arena, promotion, automatic successor or recurring monitor is authorized.

## Question and controlled intervention

Does explicitly expressing the resource consequences of an action for individual
known cards help a useful proposer generalize? The existing inputs already
contain card costs, resources, target affordability and aggregate affordability
changes. This is an accessibility hypothesis, not a missing-information claim.

Keep each original161 policy/value network entirely frozen and in evaluation
mode. Append a small shared action-by-card residual scorer. Its final scalar
linear output starts at exact zero, preserving all initial legal-action logits
and values. Train three relations fits and three paired raw controls; pair the
seed, parameter initialization, architecture, optimizer and exact examples.

The branch uses the original lawful295-state/79-action encoding. There are18
card slots:12market,3own-reserved and3opponent-public-reserved. Mask absent or
unobserved cards. Retain all original concrete actions, target identities and
location distinctions through common inputs; canonical grouping remains the
original max-logit rule. No future refill or private opposing card enters a
feature. Unknown cards newly obtained by blind reservation are not invented.

Each pair input has55channels. The43common channels retain the13card features,
three ownership/location indicators before and after, target and after-presence
flags, actor tokens before/after including gold, and bonuses before/after.
The12contrasted channels are:

- Raw: five colored token-plus-discount capacities before and after, plus gold
  before and after.
- Relations: five nonnegative colored deficits before and after, plus total
  remaining shortfall after subtracting shared gold once, before and after.

Both arms retain the same raw information in their common channels. Therefore
the comparison supplies computed relations versus raw redundant channels at
equal capacity; it does not deprive the control of card costs or resources.
Deterministic after-effects include exact payment, takes, returns, newly earned
discounts, purchased-card removal and reservation movement/availability.

Shared pair network55->32->32 with ReLU; masked mean and max pool64features.
Concatenate the full79action and61public context channels, then204->32->1.
Only these9,441parameters train. The exact layout and arithmetic are serialized
in MODEL_CONFIG and frozen with all code. A scalar independent feature builder
and rules-based fixtures check payments, gold, reservations, private information,
card removal and added bonuses.

## Fixed training and acquisition diagnostics

Authenticate the completed independent_game_coverage_128_recovery_01 result,
data, model registry, target conversion and optimizer audits. Reuse exactly its
9,461old TRAIN rows and720added rows; all720 stay retired from evaluation.
Do not alter inherited files or replay historical training/rollouts.

For each of three original161 initializers, both new arms receive the exact
R1 BROAD stream:1,784updates,114,044exposures,57,022from each pool;1,772batches
of64 and12of53. Retain the old narrow partition only to reconstruct identical
length bucketing and batch order. Use recorded outsiders excluding the actual
first three and unchanged softmax(q/.05) targets. Empty/singleton sets preserve
batch slots and differentiable zero losses. No weights or new target rule.

CPU float32, deterministic operations, oneTorch thread, original paired train
seeds, fresh AdamW lr1e-4/weight_decay1e-4, gradient norm clip1. Always select
the final1,784update. Save initial/final acquisition metrics and recoverable
optimizer checkpoints. Check all base tensors remain identical. Independently
reconstruct batch streams, objectives, optimizer trajectories, final tensors and
diagnostics. Both arms have the same trainable parameter membership and initial
parameter tensors. All six final models must be sealed before source generation.

Acquisition is explanatory, not a feature/checkpoint selection criterion. A valid
completed recipe proceeds to its fresh test regardless of acquisition or old
development scores. Old2,489validation rows and336conditional roots are not the
primary endpoint and need not be evaluated. No retry, extra dose or architecture
rescue is allowed on predictive results.

## Prospective fresh population and controls

Use the original fast161 restricted self-play population. Generate1,080main and
54probe source games, respectively60and3 per original fit/acting-seat/phase cell.
Phases are turns0-19,20-39,40+. Select20main and1probe game per cell and one
eligible position per game by fixed identity hashes only:360main independent
games and18separate probe games. Do not replace ineligible games or enlarge the
pool. Insufficient coverage is an operational hold. Audit every source decision.

Main source seeds2295000000+index; probe2296000000+index; fixtures2298000000+index.
Authenticate disjointness against inherited training/validation, added720 and
previous coverage sources. Source assignment, source search streams, panel
selection and teacher random choices each have new fixed domain tags.

Four decision arms: anchor, relations, raw and teacher. Retain the original first
three proposals. Anchor uses its unchanged fourth; relations/raw propose their
fourth from the complete lawful action set. Teacher starts with anchor's four
and adds two ranked and two identity-uniform alternatives, up to eight. Each
position uses its assigned source-fit models. The union contains at most10actions.
All fits, phases, seats, private-reserve states and short supports remain included.

Evaluate union actions in128common selection worlds. Use original candidate order
for ties. Durably seal all choices and selection evidence before128independent
assessment worlds for the at-most-four distinct selected actions.
World seeds2290000000 + stage offset(0probe,1000000main) + index*512 + block
offset(0selection,256assessment) + worldindex. Validate disjoint source/world
streams; no assessment result selects an action or model.

Keep mature observer-legal determinization and greedy035 terminal continuations.
Keep the established bounded repetition/equivalence contract; unresolved outcomes
remain unknown in[0,1], never draws. Ambiguous selections hold unless independently
certified equivalent under the existing procedure. Identical selected actions
cancel exactly in paired contrasts. This declared fast-source/continuation
population does not establish strength on all deployed search distributions.

## Prespecified statistics and interpretation

Use whole source games as independent units. Each main game has one root. Jointly
resample all arms within18fit/seat/phase cells,20games per cell,50,000replicates,
seed2293000000, chunks1,000. Compute point bounds and paired95% intervals with
conservative treatment of unknowns. Three fits do not multiply the sample count.
Two implementations independently reconstruct exact integer score arithmetic,
bootstrap draws/reductions, all contrasts and gate decisions. Report per-fit
descriptive means without fit selection or extra subgroup gates.

Contrasts: relations-anchor, relations-raw, raw-anchor, teacher-anchor and
teacher-relations. The joint investment gate requires all of:

1. Relations-anchor mean lower bound >=+.005expected score (+.5percentage points).
2. Relations-anchor paired95% lower endpoint strictly greater than zero.
3. Relations-raw paired95% lower endpoint strictly greater than zero.

The first threshold is a practical investment judgment, not a power guarantee.
A joint pass supports this particular explicit-relations correction and a
separately proposed full-game test. Shared residual gains do not establish the
arithmetic mechanism. Good acquisition without transfer weighs against this
local remedy; poor acquisition limits that interpretation. Small uncertain
differences remain inconclusive. No positive control or secondary endpoint can
rescue the joint gate. These are conditional scores, not full-game win rates.

## Verification, execution and interruption recovery

Follow docs/engineering/verification_policy.md. Independently check all new
features, lawfulness/private invariance, models, source decisions, proposals,
selection/assessment links and final statistics. Fully replay the18probe roots.
For main, replay8of128worlds per block at every root plus a metadata-selected
full stress subset (at most39roots) and every unresolved branch. Main replay
is sampled; historical audits are inherited by authenticated identity.

Use one hidden detached pythonw supervisor, at most3one-thread fitting workers
and8one-thread source/evaluation workers. No GPU or visible consoles. Inhibit
system sleep during execution; display sleep remains allowed. Freeze preflight,
protocol, code/dependency inventory, exact data/model identities and seeds before
production work. Reject stale production runtime evidence before preparation
or launch; fixture failures must never write production abort markers.

Before main assessment, use the completed outcome-blind18-root probe to project
collection, prescribed replay, structural checks, finalization and storage. Add
35%forecast allowance. Bound total active execution at24hours with5GiB reserve.
This ceiling deliberately allows substantial headroom over earlier experiments;
it is not an ETA. A forecast miss produces a documented hold, preserving evidence.

Completed fits, optimizer checkpoints, source games, source audits, root selection
seals, root outcomes, root audits and job completion references are saved as work
progresses. The runner is one-shot and never silently retries. After external
interruption, inspect process/lock state and authenticate saved completion and
optimizer receipts before an explicitly requested recovery. Reuse completed
audited work; preserve any partial or failing record in its original namespace.
Never mark an incomplete stage complete, overwrite frozen evidence or infer
scientific failure from process disappearance. Forced app/service updates are
outside the detached process's survival guarantee.

Finish with model/source/assessment audits, result, readable report and final
status. No recurring monitor, automatic arena, promotion or successor follows.
