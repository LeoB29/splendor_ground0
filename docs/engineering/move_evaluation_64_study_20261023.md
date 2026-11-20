# Prepared 64-world move-evaluation comparison

23 October 2026. **Design and model-free preparation only. No new evaluator is
implemented, qualified or launched by this preparation.** A separately reviewed
implementation and immutable execution freeze are required before model work.
This is a new A64/B64 study; every previous 128-world result, timeout and frozen
artifact remains unchanged. Larger-data collection, training and strength games
remain deferred. The detailed policies and assessment contract in
[the original design](move_evaluation_study_20261023.md) apply except for the
explicit selection, timing and preparation changes below.

## Fixed comparison and world schedule

Keep the same 18 development roots, six each for fits 161001/161002/161003,
balanced by seat and phase, from 18 games and 16 source seed pairs. These known
positions are not fresh test games or a representative generalization sample.
Preserve every concrete candidate, ordering, payment/return/noble representative
and first-maximum tie rule. There are 71 candidates: 17 roots have four and game
28 / ply25 has three. Do not invent a fourth action or replace a difficult root.

Both arms use exactly the **first 64 seeds, indices 0--63, of each original
frozen 128-world schedule**, in that order. This prefix is fixed before opening
any new assessment outcome; it is not selected using scores or runtimes. First
reproduce the assigned model's original proposal and full128 schedule exactly,
then validate the separately derived64 schedule and evaluate only that prefix.
New versioned code must not mutate shared configuration or frozen selectors.

**A64:** raw035 plays both seats after the forced candidate move.
**B64:** raw035 plays the root actor's subsequent turns; the fixed greedy Entity
policy plays the opposing seat. Both evaluate all64 worlds for every candidate
and reduce terminal W/D/L using the original arithmetic and first-max rule.
Use identical initial sampled worlds across candidates and arms. No truncation,
adaptive stopping, candidate change, horizon change, batching or quota sweep.

B uses accepted per-decision public preparation reuse,
`entity_per_decision_public_preparation_reuse_20261023_v1`, with original model
inference. It does **not** use the later prebound private-kernel experiment.
Retain the same historical Entity checkpoint, CPU float32, one numerical thread,
batch one, canonical logits, qualified safe-choice filtering, observer-derived
seed, first legal maximum, pre-reveal auxiliary choices, single/PASS handling
and turn124 heuristic fallback. The value head is ignored. No policy state or
prediction cache survives a decision. Each actor receives only its lawful
observation; hidden referee worlds and assessment seeds remain parent-only.

## Correctness, clocks and ordered qualification

Before the launch freeze, review and test the actual new A/B entry points,
strict payloads, prefix derivation, clocks, interruption and cleanup. Every new
model-bearing evaluator, qualification and replay worker must call
`verify_runtime_sources()` once at startup and save the receipt. Bind ruleset
`splendor_base_qualified_rules_20261018_v1`, catalogue
`db4776918e612eee5a6f2617719c85ec5f413d592bbab0a64058babc83a70312`,
and `atomic_prereveal_completed_turn_v1`. Use approved corrected observers,
samplers and snapshot decoders; no historical default environment or loader.

Actual-model admission must cover both arm paths and all three proposal fits:
all six corrected faces in encoded inputs; non-purchase/multiple nobles;
noble-triggered final rounds; gold payments/returns; private reserves; Main,
auxiliary, forced/single actions; turn123/124; masks, action-index parity and
first-max ties. Reject missing/old identities, bad hashes and malformed native
responses. Independently verify transitions and a naturally terminal mixed-policy
fixture. The existing15 cases and18 fit/face checks are mandatory; add bounded
64-schedule rejection/completeness checks. Helper-only tests are insufficient.
Qualify the unchanged raw035/PUCT16 assessor on its15 fixtures under the actual
one-second complete-move clocks before assessment admission.

All timed selectors are fresh ordinary complete requests: proposal reproduction,
all continuations, evidence writing and response delivery count. Setup is a
separate60-second limit. Every timed A/B request must finish in **at most96s**,
providing a prospective24s margin below the **120s hard deadline**. A completed
request above96s fails the engineering gate; at120s the supervisor terminates it.
Do not average away a slow case, reset clocks, take a partial choice or substitute
a faster repeat. Record complete elapsed/CPU time, RSS and margin; this finite
qualification does not guarantee deadlines under arbitrary host conditions.

The ordered pilot is roots0/6/12, games25/57/58, fits161001/161002/161003. Pass0
visits those roots in that order, running A then B for fits161001/161003 and B
then A for161002. Pass1 repeats the same root order with each arm order reversed.
Use a fresh worker for every request. The six pass0 outputs are immutable core
results; all six repeats must match their corresponding scientific output and
compressed trace exactly, except the documented output path. Fully replay the
six unique core outputs: 1,536 continuations. The pilot, admission and all its
checks must pass within launch+1h before proceeding; no timing instrumentation
variant or additional candidate is allowed.

Then visit the remaining15 roots in ascending original lock order. Run A then B
when `(fit_index + seat) mod2 == 0`, otherwise B then A, with fit_index0/1/2.
Fully replay each original arm/root result. Finally repeat the six originally
fixed hidden-state privacy variants for both arms, in original lock order and
with the core arm order reversed. Change only actual hidden state before lawful
redaction; require unchanged observer payloads and exact scientific/gzip outputs.
The variants are checks, not new positions. They may not replace core results.

There are **54 timed selections**: 36 core arm/root requests, six pilot timing
repeats and12 privacy requests. This is six more than the48 core/privacy requests.
Core work is4,544 continuations per arm, **9,088 total**, all fully replayed with
original reference policies and independent dictionary transitions. Timing
repeats add1,536 and privacy repeats3,072: **13,696 selection continuations**.
Exact duplicates need no additional replay. Authentication, every scheduled
world/candidate, policy choice, legal observation, transition, ending, aggregate
and selected move remain mandatory checks. All qualification ends by launch+3h.

## Sealing, independent assessment and decision rule

Seal all36 core choices and all prescribed qualification receipts before any
assessment outcome is executed or opened. Require B64 to change at least six
A64 choices, with at least one in each fit. Otherwise stop for insufficient
intervention breadth, without replacing roots or asserting no useful improvement.

Use32 independent paired worlds per root. Derive seeds as integer SHA256 of
canonical JSON `["move_evaluation_64_20261023_assessment_v1", root_id, j]`,
j0--31. Keep the schedule in a parent-only artifact and reject collisions with
all original128 selection seeds. Selector payloads contain only lawful observer,
assigned fit, locked candidates and the64 selection schedule; reject extras.
Do not pass whole parent lock rows or assessment seeds to selector workers.

Force the sealed A64/B64 action in isolated copies of the same sampled world.
Thereafter both branches use deterministic raw035 for the root actor and the
unchanged qualified PUCT16 Entity opponent, with its original parameters,
observer-derived search seeds and fresh search state per move. Keep0.9s internal
and1s external move clocks,120s branch limits and turn_index<400. Models may stay
loaded; no setup or admission waits occur inside a branch. Require official
natural endings and scores win1/draw0.5/loss0. Deadlocks, cutoffs, resource errors
or timeouts mean incomplete assessment, never imputed outcomes or replacement.
Internal selector deadlock conventions remain separately disclosed search
outcomes and are not official terminal targets. Identical actions contribute
structural zero for all32 worlds and stay in the18-root denominator; skip their
redundant branches. Maximum workload is1,152 terminal branches, not independent
positions. Fully check every actual branch move, observer, transition, terminal
score and authenticated PUCT quota/branch receipt; preserve the original trust
boundary rather than claiming replay of every internal PUCT edge.

For each root average the32 paired differences B minus A, then weight all18
roots equally, each fit one third. Retain the original exact50,000-replicate
within-root paired bootstrap, PCG64 seed2026102301, root/index order, integer
half-point arithmetic and linear2.5/97.5 percentiles. Overall integer sums divide
by1152; six-root fit sums divide by384. Bind numerical runtime versions and use
an independent analyzer, including bootstrap-stream agreement. Report roots,
fits, phase/seat descriptions, changes, natural completion, costs and failures.

Call the result promising on this development panel only if all gates pass,
overall gain is at least0.02, the conditional95% lower endpoint is strictly
positive, and every fit mean is nonnegative. A saturated all-loss endpoint is
uninformative, not certain equality. No extra worlds, evaluator or revised
threshold follows a nonpass. A pass warrants preparing fresh whole-game
confirmation only. Shared Entity checkpoint/dependencies remain a disclosed
limitation: this is conditional one-move assessment, not independent-model
validation, repeated-use playing strength or population generalization.

## Resources, cost and interpretation

One windowless owning supervisor enforces wall and monotonic clocks from launch,
before admission: pilot1h, qualification3h, collection6h, absolute8h. Reserve the
last2h for verification/analysis. Clamp every request/audit to remaining phase
and global deadlines; reference replay ceiling300s/root, fixture ceiling300s.
Use one worker tree on logical CPU0 after rechecking its performance-core
assignment, one numerical
thread, tree RSS<=2GiB. Require2GiB available RAM stable60s initially and after
shortage, healthy fast paths at boundaries, at most30min cumulative admission
waiting. Never pause a live clock. Require7GiB free disk initially, maintain5GiB,
and cap new evidence/builds at2GiB. No automatic retry, extension or second quota.
Preserve partial evidence and failures; resource trouble stops dependent work.

The54 requests alone could consume86.4min at96s, plus initialization;36 audits
could consume180min at their ceilings. The3h qualification cap is therefore a
hard stop, not a promise to finish. Historical B128 times suggest only rough
half-work scenarios: reuse62.7--82.8s, later dispatch61.5--101.8s. Fixed work and
host delays need not halve; these are not B64 measurements or speedup claims.
Maximum assessment scenarios remain1.17/2.35/3.52h for10/20/30 remaining moves
per side, excluding some setup, I/O and verification. Use world0 at every changed
root as the original outcome-blind throughput wave; proceed only if remaining
fixed work with35% contingency fits the6h cutoff and2GiB cap. Never inspect wins
to decide continuation. Preserve the2h verification reserve regardless.

A64/B64 isolates opponent-policy substitution at64 worlds. It does not directly
estimate the quality effect of reducing128 to64. After choices are sealed,
historical A128/A64 comparisons at all18 roots and B128/B64 comparisons at the
three available diagnostic roots may describe sampling stability only. Separately,
exact equivalence to the first64 recorded world outcomes/state/trace identities
is a mandatory correctness gate for A on all18 roots and B on those three roots.
Version metadata differs explicitly; never compare an entire128-world gzip
hash to a64-world file. Historical records cannot supply fresh A64 timing or
replace full replay. These secondary comparisons cannot change choices or gates.

Preparation artifacts live in
`outputs/move_evaluation_64_20261023_preparation_01/`: `design.json`,
`selection_lock.json`, `assessment_schedule_parent_only.json`, cost scenarios
and a preparation receipt. They bind inherited evidence, fixed schedules,
coverage/count arithmetic and this protocol. They are not a runnable execution
freeze. Implementing and running the bounded study is a separate next action;
no model job, assessment or larger-data collection is launched here.
