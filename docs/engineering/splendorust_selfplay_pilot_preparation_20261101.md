# Preparation: one small self-play learning cycle

1 November 2026. **Proposal only: not frozen, implemented or authorized to run.**
Prepared while the separately frozen 512-versus-256 comparison continues. No
current player, source, deadline, evidence or reference is changed. Preparation
uses static source inspection; it creates no games, model calls or training data.

## Purpose and place in the roadmap

Ask whether one small update to the inherited network improves complete play at
the same search budget. Keep the architecture fixed. Better fitting of search
targets is a diagnostic; winning fresh games is the strength test.

The sequence is: finish and independently accept the current 512 comparison;
qualify the corrected-rules collection/training/export pipeline; run one separately
frozen learning pilot if approved and affordable. There is no automatic 1024
comparison, requirement to exhaust search gains, or repeated training cycle.
Larger-data work and external-opponent testing remain deferred.

This is a first generation of search-guided self-play learning: a frozen parent
plays both sides, its search supplies policy targets, completed games supply
value targets, and one successor is fitted afterward. The successor does not
generate its own training data during this pilot. A successful first generation
would not establish that repeated generations will keep improving.

## Fixed parent and one prospective choice

Use the existing 4,780,883-parameter Entity Transformer, original full network
execution, float32, and the exact inherited checkpoint as an explicitly historical
warm start. Checkpoint SHA256:
`ef8a4521cd6c03c7075f15efee23f4cde6ec94d1b5398765cf0c24a09ad745cb`.
Record that original checkpoint identity separately from runtime graph/weight
digests and the newly trained checkpoint. Do not relabel its historical data.

Keep three hidden worlds, depth64, cpuct0.4, dynamic FPU and auxiliary decisions.
Do not mix in the final-token execution optimization, enlarge the network, add
history, change precision, or tune search during this pilot.

Recommended teacher-budget rule, to be finalized before collection:

- Use512 only if the completed, independently accepted current comparison meets
  its already frozen useful-gain criteria and the measured generation cost fits
  this pilot's separately frozen resources.
- Otherwise use the already verified256 teacher. A technical interruption of the
  comparison would not establish that512 is weak; any256 choice is the affordable
  verified fallback, disclosed before collection.

No choice uses a partial score. Whichever teacher is selected plays both sides
with frozen weights and deterministic search action selection on fresh starts.
No new temperature or root-noise rule is introduced in this first pilot. This
limits exploration diversity; it is a limitation to report, not proof that the
result generalizes to every opponent or part of the state space.

## Proposed data and single update

Recommend **256 independent fresh self-play games**, assigned before collection
to192 training,32 validation and32 untouched offline assessment games. Every
position from a game stays in the same split; any duplicate setup, replay or
derived version stays with its source. Split by independent starting seed, not
by position. Check against existing seed registries. Current comparison games,
historical corpora and qualification fixtures are not training data.

For each eligible Main decision, save the lawful root observation, exact root
input tensor, legal/action support, complete root visit counts, chosen move,
search/seed identities and independently checked transition. After natural game
completion, attach official signed outcomes for both canonical player slots:
win+1, loss-1, draw0. Check seat mapping, tied games and terminal turn handling.

The policy target is the normalized root visit distribution on the qualified
Main support, not a one-hot label reconstructed from the chosen move. Define the
policy-loss softmax mask over all qualified legal Main channels before search
pruning; legal channels absent from the retained search subset have zero target
mass. Do not silently condition the loss only on retained moves. Prove the exact
mapping and mask in qualification before freezing this choice. The value
target is the actual official outcome, not root Q, an imagined-world result,
an administrative loss or a cutoff/deadlock heuristic. Retain all failures. A
collection timeout or non-natural termination stops the attempt incomplete;
do not replace the game or silently exclude it and train on the remainder.

Auxiliary Payment/Return/Noble decisions have no Main-policy target. Some forced,
single-choice or late fallback paths have no root callback/visit target. Declare
eligibility mechanically before collection, count every excluded decision by
reason, and never fabricate visits or a root tensor. Every scheduled game still
needs a verified complete trajectory. Abort on an unexpected missing target.

Recommend one small all-weight fine-tune, with no architecture changes and no
hyperparameter sweep. A provisional recipe is AdamW, learning rate1e-5,
weight decay1e-4, batch32, gradient norm cap1, and exactly two training epochs;
masked policy cross-entropy plus mean squared error over the two signed value
outputs, with equal aggregate loss weights. Average positions within each game
and then games, so long games do not dominate. The implementation must specify
the exact batching, reduction, shuffle seed and final partial-batch behavior.

These numeric defaults are a proposal, not an upstream-proven recipe. Qualify
numerics and measure bounded batch cost before freezing them. No changes after
seeing training/validation performance. Keep one final candidate; do not select
the best of several epochs, seeds or losses. Validation reports overfitting and
calibration without selecting a checkpoint. The32 offline assessment games are
opened once after the candidate hash is fixed. Neither offline split proves
playing strength. A failed finite-gradient/update check stops fitting and keeps
the parent; it is not grounds for silently trying another recipe.

## Required implementation and correctness qualification

There is **no currently qualified general training entry point**. The mandatory
[corrected-rules policy](future_training_rules_20261021.md) applies to collection,
the actual dataset loader, fitting, export and fresh evaluation. Every new worker,
including a fit-only worker, calls `verify_runtime_sources()` at startup and saves
its receipt. Bind source, data and checkpoint hashes, ruleset
`splendor_base_qualified_rules_20261018_v1`, catalogue SHA256
`db4776918e612eee5a6f2617719c85ec5f413d592bbab0a64058babc83a70312`, and turn
contract `atomic_prereveal_completed_turn_v1`.

Build separate versioned files, never patch frozen players or historical loaders.
The minimum implementation is:

1. **Root-target export.** Native search has a target-capture facility, but our
   live bridge does not currently expose its visit targets. Add a separate bridge
   and prove capture leaves the parent search outputs, decisions, RNG use and
   transitions unchanged. Map its67-index search target to the81 network logits
   by the actual native mapping; only60 Main channels are used and there are
   unused search indices. Record both authoritative legality and the search's
   retained subset. Test mass, ordering, masks and holes explicitly; never assign
   a root target to a deeper callback merely because it has the same tensor shape.
2. **Lawful inputs and corrected batches.** The model consumes31 by48 root tokens
   and emits81 logits plus two signed values. Capture the actual public-belief
   root input; do not substitute a sampled opponent reserve or hidden deck.
   Vary inaccessible true cards while holding lawful information and seeds fixed
   and require identical inputs/targets. Audit tensors back to the observer and
   corrected legal list, not merely to declared metadata. The trainable wrapper
   must reproduce the direct runtime tensor computation: upstream `forward()`
   expects525 packed features and tokenizes them itself, so passing31 by48 tokens
   to that entry point would be incorrect.
3. **Actual collector and loader counterexamples.** Exercise each of the six
   corrected faces, nobles after nonpurchase, multiple nobles, noble-triggered
   final-round termination, gold/payment/returns and private reserves. Verify
   actual encoded costs, legal support and simulated transitions. Reject legacy
   snapshots, old/missing identities and mismatched hashes. Helper self-checks
   or a corrected outer referee alone are insufficient.
4. **Train, save and load round trip.** Verify the unchanged parent through the
   proposed trainable definition and deployment route before any update. Test
   loss/target direction, seat inversion, masks, finite gradients, one bounded
   optimizer update, serialization and loading into a separate candidate player.
   Keep original hash restrictions intact; authorize the new checkpoint by its
   own frozen manifest. Keep the current Python host inference route rather than
   introducing a native network export. Evaluate the saved-and-reloaded candidate
   on actual-model fixtures against its reference outputs, including its own
   eager-versus-compiled inference parity.
   Different outputs from the parent after learning are expected.
5. **Independent preservation and evaluation.** Retain lossless evidence and
   all failed attempts, enforce real clocks and process ownership, independently
   replay all trajectories and reproduce a predetermined outcome-blind sample
   of model/search decisions. Freeze the exact fixture/replay counts and receipt
   inventory in the execution protocol before any pilot data is generated.

Existing pre-reveal auxiliary and internal deadlock conventions stay disclosed
and fixed. A sequential interface redesign is a separate research question, not
an automatic prerequisite for this bounded corrected-rules pilot.

## Independent playing-strength assessment

Recommend **400 fresh games /200 swapped-seat pairs** between the one successor
and its unchanged parent, both at128 simulations with identical search settings,
CPU allocation, network execution and clocks. Choosing128 prospectively tests
whether learning helps at our established practical research reference budget;
it does not promote or compare the teacher's high-budget player by itself.

Run correctness qualification and six excluded operational games first.
Freeze all assessment seeds separately from every training/validation/offline
seed and from existing benchmarks. No changes to the candidate after evaluation
starts. Preserve the established useful-gain rule: score at least55%, paired
two-sided95% lower bound above50%, at least50% in each seat, the existing
degenerate-interval safeguard and complete independent verification. Retain the
conservative sensitivity assigning every administrative game a candidate loss.
Both the official and this conservative analysis must pass every useful-gain
criterion. Specify the bootstrap seed and50,000 pair draws before launch.

Keep symmetric9-second internal/10-second complete-response research guards.
Accept completed moves over one second, report timings separately, and count
properly classified main-assessment hard guards as losses. Never retry a failed
decision or replace its game. Infrastructure, rule, privacy or verification
failures stop incomplete. Keep700ms descriptive. The matched quota isolates the
weight update as the intervention; altered weights can still change search work
and latency, so equal architecture/quota does not guarantee identical runtime.

The400-game assessment costs more than a token smoke test, but reuses the
established strength criteria and can support a meaningful local conclusion.
A smaller comparison could be an engineering screen only; do not silently turn
it into the same-strength evidence. No sample extension after seeing results.
The192 training games and two epochs are intentionally modest; the assessment
size does not imply that this amount of training will produce a detectable gain.

## Provisional resource envelope and decisions still needed

Preparation now is documents and static review only. Defer model fixtures,
timing qualification, data collection and fitting until the current owners have
exited and the512 study's independent acceptance or failure review is complete.

For a later approved pilot, propose one72-hour total clock,60-hour model cutoff,
and at most12hours saved-only finalization. Indicative nested stage caps are
4hours qualification,24hours data collection,4hours fitting,1hour operational
games,16hours main assessment,4hours reproductions and12hours final audit, all
clipped to the same original clock. At most662 games:256 source,6 operational,
400 assessment. Technical fixture work must be separately enumerated and capped.

Propose32GiB total new artifacts including8GiB reserved for final verification,
with at least37GiB free before launch and5GiB maintained. Freeze measured RAM
and batch bounds after a small implementation fixture; retain4GiB initial free
RAM stable60seconds,2GiB between decisions and at most two resident inference
workers. Run fitting and game workers sequentially. Retain the existing2GiB
combined inference-worker bound; propose a separate4GiB fitter RSS ceiling,
qualified before admission. Charge at most3hours RAM waiting to the original
clock; never suspend an active decision. No cloud/GPU purchase is assumed.

These are ceilings, not duration/storage forecasts or launch readiness. At
preparation time only about27GiB disk was free while the current study continued
writing. Storage will need reassessment and possibly separately authorized
archiving; do not delete frozen evidence. Before final freeze, measure source
game, batch, raw-evidence and verifier throughput on fixed qualification material,
reserve the entire downstream assessment/audit, and reject an unaffordable plan
before collection. Reduce scope only prospectively in a separately reviewed
plan; never shrink an active sample or extend a running clock.

Outstanding decisions are the post-acceptance teacher choice, exact training
recipe confirmation, qualified hardware/batch choice, fixture and replay counts,
measured storage/compute forecast, fresh seeds and independent launch review.
No open item is permission to start a partial pilot.

## How to interpret the eventual result

- Qualified pipeline only: establishes that a corrected learning experiment can
  run and its output can be deployed; no strength gain yet.
- Better offline fit without a useful match gain: no demonstrated stronger
  player. Diagnose target quality, coverage or update size before scaling data.
- Useful matched-search gain: evidence for one successful local learning cycle
  at this budget. Consider one independent replication or second generation;
  no automatic scaling, reference promotion or superhuman claim.
- Inconclusive match: retain the parent and interval. A small pilot cannot show
  that the architecture has exhausted its capacity or that self-play cannot work.
- Technical/resource failure: preserve it, repair separately, and do not describe
  it as evidence against learning.

The broader objective remains demonstrably superhuman play. Search can produce
better targets; learning may put some of that improvement into the same-sized
network. This pilot tests the second link. Independent strong opponents and
expert-human assessment will still be needed for the eventual broader claim.

## Local evidence used for preparation

- [Current512 protocol](splendorust_search512_20261101.md), especially its final
  roadmap paragraph; no current partial scores were used.
- [Verified256 acceptance](../../outputs/splendorust_search256_recovery_20261030_02_closeout/independent_acceptance_01.json).
- [Earlier learning roadmap](../strategy/splendorust_revised_roadmap_2026-10-24.md).
- [Corrected training helpers](../../tools/future_training_rules_20261021.py).
- [Pinned model definitions](../../outputs/external_opponent_search_20261018_02/splendorust/research/architecture_pivots/models.py),
  [current adapter](../../tools/external_strength_screen_20261018_02/splendorust_adapter.py),
  [native search](../../outputs/external_opponent_search_20261018_02/splendorust/crates/splendor-agents/src/neural_search.rs)
  and [native transfer](../../outputs/external_opponent_search_20261018_02/splendorust/crates/splendor-agents/src/transfer.rs).
