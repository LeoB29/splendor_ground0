# Controlled proposal-ranking pilot at128 selection worlds

2026-09-27. Owner authorized implementing and running the bounded pilot in
docs/strategy/next_after_proposal_diagnostic_128.md. New implementation:
tools/proposal_ranking_128 and run_proposal_ranking_128.py; output:
outputs/proposal_ranking_128. All previous dirty work and frozen dependencies
are preserved. This protocol authorizes no automatic full-game arena or successor.

## Hypothesis and fixed comparisons

Does explicitly learning the ordering of the two evaluated alternatives outside
original035's top two improve the actual fourth-proposal role under unchanged
128-world planning? Compare ranking training, matched winner-label CE training,
and the untouched selected161 anchor, retaining all three fits161001/161002/161003.

This is a fresh-position conditional decision test, not full-game or superhuman
strength. The exact deployed first three include a random alternative: training
outside the top two is useful available supervision, not an exact marginal-value
objective outside all three. Never label unobserved actions bad.

## Frozen parent and learning data

Parent outputs/proposal_iteration_128 binding
1a3c53142516a5d8f08376d0a7227ae9199952a9e077e1309d13fdcdf6d40ace
must remain closed and audited. Bind its code/runtime inventory, parent data and
export/audit/game receipts, source dependencies, initializers, all three saved CE
trajectories including every checkpoint/validation record, and closed final result.
Source-root hashes are inherited through sealed audited source receipts and each
actual root is hash-checked when exporting the new target. No old simulation is
rerun merely to reconstruct targets.

Use exactly23,936 parent rows:11,985 old160 plus11,951 learned128 rows;
19,047 training and4,889 validation. Retain whole-source-pair splits, source order,
lawful input tensors, offered group membership, canonical maximum scoring, model
architecture, all three161 initializers and the previous treatment's batch slots.
Independent source/export checks and target arithmetic rebuild every new target.
The latest201 diagnostic roots and latest arena are not added to this corpus.

For support4 compute d=Q(slot2)-Q(slot3), zero-based, where each Q is mean128
recorded terminal scores, score=(WDL+1)/2, WDL in{-1,0,1}. For support<4, d=0.
Let M be the mean abs(d) over ALL training rows, including those zero rows;
require M>0. Freeze the resulting data lock/M before fitting. Validation never
determines M. Loss is abs(d)/M times softplus(-sign(d)*(z2-z3)), with zj the maximum
concrete-action logit in its offered canonical group. Exact ties/short supports
give zero ranking loss but remain in batch/exposure accounting. No threshold,
temperature, inverse-variance weight, anchoring term or loss sweep. Empirical
gaps remain noisy and are not calibrated regret or confidence estimates.

Three ranking fits: four passes, batch64, AdamW lr1e-5/weight_decay1e-4,
clip_norm1, fresh optimizer, float32 CPU, one deterministic Torch thread. Reuse
parent207001/207002/207003 training streams and exact shared-treatment batch
schedule, all1,192 optimizer steps, with frozen value-head parameters. Zero-loss
rows do not cause skipped optimizer steps. learning.CONFIG is part of identity.

Reuse all saved parent treatment/CE checkpoints rather than regenerate those
matching trajectories. Prove equal data/config/initialization/exposure/batch
identities. Select BOTH training families on the same validation mean of whole-
pair mean ranking losses, with M from training, the existing24-checkpoint schedule
and earliest strict minimum including update0. The prior selected CE checkpoint
need not remain selected. No test outcome participates in selection.

Independently reproduce all new ranking optimizer steps and compare every saved
checkpoint exactly, including selected/final tensors and frozen value head.
Independently recompute validation for both ranking and every reused CE checkpoint
and the common selection rule. The nine-model seal records anchor/ranking/ce for
each fit, selected updates and unchanged-from-anchor status. No best-fit choice.

Prospective efficiency exception: if ALL THREE selected ranking models are exactly
identical to their anchors, independently reload and compare model kind/config,
all tensor keys and every tensor. Identical parameters and unchanged deterministic
proposal/selection rules imply exact zero ranking-anchor effect on every coupled
position, so the positive progress gate cannot pass. Close after the full fit/
selection audits without source collection or fresh evaluation; report skipped
stages and no empirical interval/CE effect. This is an identity proof, not an
outcome-based early stop. If any model differs, run the full fixed panel unchanged.

## Fresh source positions

Generate fixed main pool1,080 games and separate probe pool54 games. In each
pool, hash-rank game indices using source_assignment_tag, then cycle cells ordered
fit ascending, acting-seat0/1, phase0/1/2. This assigns60main/3probe games to every
one of18 cells. The cell is a sampling assignment, not a restriction on gameplay.
Both source actors use the same cell's unchanged frozen161 restricted student.
Use mature lawful fast_selector decisions and original035 candidate construction.

Environment seeds:2095000000+index main;2096000000+index probe. Source streams
hash [proposal-ranking-128-source-stream-v1,stage,index]. Test-only fixtures use
2098000000+index and cannot target production or override production snapshots.
Every source game must terminate naturally or by established forced-pass deadlock;
turn400, fourfold repetition or60 no-progress turns are integrity cutoffs, not
accepted outcomes or permission to replace games. Check every actual source move,
lawful model input/hidden-information invariance, choice and final state/outcome.

Within each assigned cell, eligible games contain a pre-decision state for that
seat and phase (turn0-19/20-39/40+). Hash-rank by
[proposal-ranking-128-panel-v1,game,stage,index], choose20main or1probe game, then
hash-pick an eligible turn using the corresponding turn tag and turn index.
Sort selected states by source-game index for dense panel indices. One state per
distinct source game;360main,120perfit,20percell;18probe,onepercell.

Do not use winners, logits, gaps, disagreements or rollout outcomes to select
states. Retain private reserves and all legal supports. Fast source games may
end before the assigned phase: if fixed-pool eligibility is insufficient, write
an operational hold with evidence, without extending or replacing the pool.
Independently reconstruct the whole selection and eligible counts from audited
source records. Seal selection and audit sampling BEFORE new rollout collection.

## Exact planner and independent assessment

At each selected state retain original035 top two and its first random alternative
under the source search stream, including candidate nonce, concrete representatives
and ordering. Each sealed model proposes its highest-scored remaining canonical
group. Use an isolated registry; never repoint any frozen predecessor global.
For support<4 keep the complete original support. Deduplicate the union ordered
common first three, anchor fourth, ranking fourth, CE fourth: at most6 actions.

Selection:128 fresh observer-legal worlds, all union actions, shared per-world
randomness, mature serial greedy035 continuations for both actors, terminal WDL.
Each conceptual arm may choose only from its own ordered up-to-four support,
using first-maximum ties. Save the complete selection evidence, identity and
locked choices before any assessment simulation.

Assessment:128 DISJOINT fresh worlds, evaluating only the sorted unique union
indices of the three locked choices (at most3). Explicit candidate-index mappings
prevent confusing sparse assessment columns with full-union indices. Assessment
does not select another move, checkpoint, fit, sample size or subgroup.

World seed =2090000000 + stage_offset + panel_index*512 + block_offset +world_index.
Offsets:probe0/main1000000/test2000000; selection0/assessment256; index0-127.
Check pairwise disjoint integer ranges and separation from fresh source seeds.
They are distinct from ancestor world seeds/namespaces. Reconstruct observer-
legal hidden worlds; never retain the discovery deck order for assessment.

Main maximum360*128*(6+3)=414,720 recorded continuations. Probe maximum18*128*9
=20,736. Deduplication reduces actual counts. Candidate comparison shares research
computation; each deployed conceptual arm still uses its own4*128 budget.

## Statistics and fixed decision

Primary progress: ranking-minus-anchor independently assessed terminal score.
Required attribution: ranking-minus-CE. Average360 source games equally. Keep
all arms paired within game. Bootstrap50,000 replicates in50chunks of1,000,
NumPy default_rng seed2093000000. Within each chunk iterate18 cells ordered
fit/seat/phase, draw integers[0,20) in shape(1000,20), resample whole source games.
Sort rows by dense panel index within cells. Same draws for both contrasts.
Linear2.5/97.5percentiles. Reports condition on the three fits and student-source
population. Per-fit/phase/seat/privacy, proposal overlap and decisions descriptive.

Score=(WDL+1)/2, so128-world means lie on integer/256 grid. Independent statistics
use raw recorded outcome reconstruction plus direct indexed summation; verifier
uses independently reconstructed receipts and histogram-count accumulation.
Require means/intervals within absolute1e-12, all gates/classifications identical.

Pass only if ranking-anchor mean>=.005, its95% lower endpoint>0, AND
ranking-CE95% lower endpoint>0. Practical threshold uses exact integer totals:
anchor_difference_sum*200>=360*256. Classification is
positive_conditional_ranking_contribution if all pass, otherwise
practical_progress_not_established. Report all point estimates/intervals, including
uncertainty. The0.5pp hurdle allocates compute; it is not a55% arena prediction,
equivalence margin, guaranteed-power claim or proof smaller effects lack value.

No scientific extension or automatic arena follows any result. A pass supports
subsequently proposing fresh full-game confirmation; a failure closes this fixed
recipe without selecting another checkpoint, seed, dose, loss or extra states.

## Verification and frozen integrity

Apply docs/engineering/verification_policy.md. Every new source game/actual move,
model identity, lawful model input, proposal/support/representative, sampled-world
construction, declared budget, outcome field, choice, sparse index and score is
checked. Every selected actual source state is replayed from its source game.
Source/data/model artifacts, sampling manifests and stage dependencies are sealed.

Probe: full continuation replay at all18 states. Main: identity-hash sample8 of128
worlds in EACH block at EVERY state, replaying all candidates actually evaluated
there. Add full replay at hash-minimum state per fit x seat x phase (18), fit x
privacy(neither/own/opponent/both,12), and fit x canonical-support1/2/3 (9).
Deduplicate overlapping stress roots; missing cells are reported, never replaced.
Maximum39 full stress roots. Seed2094000000; hash tags are fixed in common.CONFIG.
Sampling depends only on frozen IDs/lawful metadata, not chosen moves or outcomes.

Persist sampling manifests and seals before any collection. Rebuild selections
independently and reconcile exact identities, every receipt, recorded/replayed
worlds and separate selection/assessment continuation counts. Check selection
artifact content and creation precede assessment. Final reclosure checks hashes,
links, model selections and independent statistics without extra simulation.

Preflight uses isolated fixtures for exact target sign/scale/zero support,
training/control selection/reproduction, source cells and natural terminations,
private reserves, sparse assessment, tampering, statistical gates, resource holds,
finalization and hidden Windows workers. No production data prep/fit/source game
or evaluation precedes the passing checked freeze. Changes after freeze require
preserving the old version and evidence, not patching a running study.

Mature rules, encoder and neural primitives remain shared trusted dependencies.
Main replay is sampled, not exhaustive. Any unexplained mismatch stops and keeps
partial evidence; there is no automatic retry or unbounded follow-up audit.

## Autonomous resources and stopping

One detached pythonw supervisor;3 CPU workers for fitting/reproduction and8 for
source/evaluation/audits,1 deterministic Torch thread each. All child pools use
the matching GUI-subsystem Python and assert no attached console. Retain logs,
health receipts, launch acknowledgement and model/stage provenance. Execution
does not require an active chat. Scoped system-sleep inhibition permits display
sleep and releases on exit; computer must remain on. No monitoring is scheduled.

Order: target export/audit; three ranking fits and CE reselection; three independent
fit/selection audits; model seal; fixed source pools and complete source audits;
selection and audit manifests;18-state probe/full replay; outcome-blind runtime
gate;360-state main collection/prescribed audit; independent statistics/report.

Eight-hour execution limit starts at launch.json and includes all these stages,
including finalization. Implementation/preflight is additional. Check limits at
target/export, optimizer/validation, source-move and rollout-world boundaries,
job completions and finalization; an active trajectory may finish before detection.
Five-GiB disk reserve. Queued jobs cancel on error; partial evidence stays saved.

The probe forecasts separate selection/assessment time per actual continuation
and structural collection time per root, scaled to max276,480selection and
138,240assessment main branches. Audit uses separate measured selection/assessment
replay costs, exact sealed main replay ceilings (max6/3 supports), and structural
audit time per root. Divide worker-seconds by8, add35% time allowance and300sec
finalization; allow50% storage overhead plus512MiB. All already-incurred elapsed
time/data/fit/source costs count. Proceed only if elapsed+forecast<=8h and projected
storage preserves5GiB. This operational gate never reads scientific effects.

Resource expiry/source shortage writes operational_hold.json; integrity/worker
failure writes failure.json. Both preserve partial evidence without retry.
Complete runs always write final result, audit, report and final_status, including
an inconclusive outcome. No promotion, extra generation or full-game tournament.
