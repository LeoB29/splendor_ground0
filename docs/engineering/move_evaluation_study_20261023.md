# Bounded move-evaluation study: opponent-policy substitution

23 October 2026. **Prepared design only. No evaluator implementation, model
inference, qualification, new continuation, training or game has been launched.**
The execution code and new native binary still need implementation, review and
an immutable execution manifest before the model-bearing qualification below.

The question is whether a more credible simulated opponent improves selection
among the existing candidates. This is one fixed alternative, not a policy,
checkpoint, quota or sample-size sweep. The completed benchmark/postmortem remain
development evidence. All historical success criteria and conclusions stand.

## Fixed inputs and comparison

Use exactly the 18 roots in the postmortem's fixed panel: six for each proposal
fit 161001, 161002 and 161003, balanced across seats and early/middle/late phases.
They span 18 games and 16 source seed pairs. Do not select the best fit or replace
a difficult root. These are known development positions, not fresh test games.

The preparation directory is
`outputs/move_evaluation_study_20261023_preparation_01/`. Its `candidate_lock.json`
binds each lawful root observation, original context, ordered concrete candidate
actions, canonical keys, 128 selection-world seeds, recorded control choice and
source receipts. There are **71 offered actions total and 9,088 candidate/world
continuations per complete evaluator pass**. Seventeen roots have four actions;
game 28 / ply 25 has only three legal canonical actions. Retain those three;
never invent a fourth or change payment/return/noble representatives.

At each root, reproduce the original candidate generator using its assigned
unchanged learned model, raw035 scores and original public schedule. Require
exact equality with the locked candidates and world seeds before proceeding.
The alternative may not propose, remove or reorder root actions. Both selectors
use the same 128 observer-legal sampled worlds and first-candidate tie breaking.

**Control A:** the accepted four-candidate evaluator: raw035 plays both seats
after the forced root move; use its already verified full128-world choice and
records. Do not rerun a different 64-world baseline or redefine A's clock from
the cost of loading cached evidence.

**Alternative B:** after the same forced root move, raw035 still plays the root
actor's future turns. Only the simulated opposing seat changes to the fixed
greedy Entity policy defined below. Reduce terminal W/D/L over all128 worlds in
the same way as A. No learned leaf value, horizon discount, extra search layer,
new candidate, fine-tuning or additional selection worlds are introduced.

This isolates the complete simulated opponent-policy substitution, including
that policy's inherited legal filters/auxiliary decisions. It does not isolate
the neural logits from those inherited heuristics. In future real play the
opponent is still external; B changes only our internal prediction of it.

## Exact alternative opponent policy

Use the authenticated SplendoRust Entity checkpoint:
`ef8a4521cd6c03c7075f15efee23f4cde6ec94d1b5398765cf0c24a09ad745cb`, from pinned
source commit `1828507b8d2f0e02bc03d102b9c6d29a54fa8d57`. The checkpoint's original
training provenance remains historical; corrected inference does not relabel
its training data. Its standalone policy has **not** been strength-qualified.

Create a separately versioned observer-only native operation and build output.
Reuse authenticated upstream encoder, model definitions and action mappings;
do not patch the frozen source tree, existing binary or qualified profiles.

For each simulated opponent turn:

1. Construct only that actor's lawful observation from the corrected sampled
   world. Never pass the actual referee's hidden deck, the other player's
   private reserve identities, or the assessment-world seed to the policy.
2. Enumerate authoritative corrected atomic actions and the existing ordered
   native Main choices through `bridge.main_actions`. Retain the qualified
   `safe_choices` filter exposed by `PolicyValue<Canonical>::choices`; if it
   returns `None`, retain the original ordered list. An empty filtered list is
   an error, not permission for an invented move.
3. For a nontrivial Main decision below turn124, call
   `PolicyValue<Canonical>::evaluate` once and select the **first maximum of the
   canonical policy logits** in the retained legal order. The returned value
   head is ignored. Use the canonical action-index mapping, not the raw81-head
   index on canonical actions. One choice and forced PASS need no neural call.
4. Keep the existing pre-reveal compound-action resolution, upstream non-Main
   heuristic for payment/return/noble decisions, and declared turn>=124 heuristic
   fallback. Auxiliary projections must never enter neural encoding or search.
5. Require the completed action to belong to the authoritative atomic legal
   list. Record observation/action hashes, policy-input/output digests, filtered
   support, selected index and which branch was used.

Do not substitute `PolicyValue<Canonical>::rollout_action`: that method calls a
heuristic rather than the neural policy. Zero PUCT iterations are also not a
defined implementation of this design. No tree, visit distribution, history,
outcome-dependent cache or RNG state persists between decisions. Initialize each
decision's encoder/determinization RNG with exactly `int(digest(observer)[:16],
16)`, using the qualified adapter's canonical digest. Model weights
may stay loaded inside a root's worker. CPU float32, one numerical thread,
batch one; no batching/GPU optimization is part of this study.

Selection trajectories retain the bound corrected planner's termination and
internal deadlock conventions, explicitly marked as internal search outcomes.
They must not be exported as official completed-game targets. Any unresolved
cutoff, illegal move, missing branch or nonfinite score blocks B's acceptance.
Report internal deadlock incidence separately; the assessment below accepts only
official naturally completed terminal outcomes.

## Independent terminal-outcome assessment

After A and B's actions at all18 roots are sealed and their prescribed checks
pass, assess those fixed choices using **32 new paired worlds per root**.
Do not assess using B's own value estimates or its greedy opposing policy.

For each world, construct two isolated corrected referee states with identical
initial sampled hidden cards/deck order. Force A's selected action in one branch
and B's in the other. Thereafter **both branches use the same fixed policies**:

- Root actor: unsearched, deterministic raw035, the same authenticated checkpoint
  used by the current planner, with complete corrected legal input.
- Opposing actor: the unchanged qualified **SplendoRust PUCT16** profile
  `entity_onehot_puct16_canonical_atomic_prereveal_cpu_v1`,16 iterations per
  searched Main phase, three internal worlds, depth64, cpuct0.4, dynamic FPU
  reduction0.02965, no noise or uniform prior, existing safe/auxiliary/late-turn
  branches,0.9-second internal and1-second external complete-move deadline.

PUCT16 is chosen because it has a corrected-rules 58–2 screen against raw035 and
recorded low cost. A higher quota is not assumed stronger merely from timing
qualification. This assessor is distinct from the PUCT6400 benchmark opponent.

Models remain loaded across a root's assessment branches to amortize setup.
Search agents/trees and decision state reset on every move, as in the qualified
profile. Use the existing observer-derived search seed unchanged. Referee worlds
and draw order remain parent-only; each actor receives its own observation at
every turn. Paired branches share initial random worlds, not later observations
or private information. Their trajectories may diverge naturally.

Score actual official terminal results from the root actor's perspective:
win1, draw0.5, loss0. No neural or score-margin label replaces the winner.
An identical sealed A/B action contributes exactly zero difference for all32
paired worlds and remains in the18-root denominator; no redundant branches need
run for that root. Otherwise run both branches. Maximum workload is
**18 × 32 × 2 = 1,152 terminal branches**, not1,152 independent positions.

This is **independent outcome assessment under a fixed reference policy**, not
independent-model validation. B and PUCT16 share the Entity checkpoint and some
adapter dependencies. Search and terminal outcomes remove direct self-grading
with B's scores, but shared model-family and simulator biases remain. The
estimand is a one-move intervention followed by raw035 versus PUCT16, not the
strength of repeatedly deploying B or its win rate against PUCT6400.

## Randomness, sealing and analysis

The selection-world schedule is exactly the bound original128 seeds. New
assessment seeds use the distinct namespace
`move_evaluation_20261023_assessment_v1` and canonical JSON
`[namespace, root_id, world_index]`, world_index0..31, SHA256 interpreted as an
integer seed. Reject equality with any selection seed and keep the assessment
seeds outside selector inputs. The selector payload allowlist is the lawful
observer, assigned fit, ordered candidates and original selection schedule;
reject extra fields. Never send a whole candidate-lock row containing parent-only
assessment seeds to a selector. Do not use actual saved referee decks as the new
assessment worlds. Freeze code and seed derivation before model-bearing work;
seal all B choices before executing or opening any assessment outcome.

Let d[r,j] = score_B - score_A on paired world j at root r. Compute each root's
mean over32 worlds and the overall equally weighted mean over18 roots. Each fit
therefore has weight one third. Always retain identical-choice roots as zero.
Report all three fit means, each root, phase/seat descriptions, selection changes,
natural termination rates, costs and all failures.

For Monte Carlo uncertainty, use50,000 bootstrap replicates, independently
resampling the32 **paired differences within each root**, then averaging the18
root means. Use NumPy `Generator(PCG64(2026102301))`; for each replicate visit
roots in candidate-lock order, including structural-zero roots, and draw32
int64 indices in[0,32) per root with replacement. World order is0..31. Represent
scores as integer half-points0/1/2, sum paired integer differences, then divide
by1152 for the overall mean or384 for each six-root fit mean. Use
`numpy.percentile(..., [2.5, 97.5], method="linear")`; bind Python and NumPy
versions in the execution freeze. This interval conditions on these known18
roots and the frozen reference
policies. It is not a population/generalization confidence interval. Do not
bootstrap inside the18 singleton model/seat/phase cells or treat the two branch
outcomes as independent. The16 source seed pairs remain a disclosed dependence
among selected roots; no claim about unseen roots follows from this calculation.

Prospective development investment rule, unrelated to historical study gates:

1. Before assessment, require B to change at least6 roots, with at leastone in
   each fit. Otherwise stop as **insufficient intervention breadth for this
   planned follow-up**, not proof of no useful improvement. Do not replace roots.
2. Require all prescribed correctness, timing, budget and assessment checks.
3. Call the result **promising on this development panel** only if overall gain
   is at least0.02 expected-score units (2 percentage points), the conditional
   95% lower endpoint exceeds0, and every fit's mean gain is nonnegative.

The2-point threshold is a prospective investment threshold for this new
conditional assay, not a conversion to game win rate or reuse of an old success
criterion. A pass warrants preparing fresh whole-game confirmation only. It does
not promote B, launch an arena or authorize larger-data collection. A valid
nonpass does not establish that all evaluation improvements are exhausted.
If both arms lose everywhere and the empirical interval collapses to[0,0],
label the endpoint saturated/uninformative, not certain equality. There is no
32-to64 extension, second evaluator or threshold revision after outcomes.

## Qualification and verification before assessment

The actual new entry points must integrate
`tools/future_training_rules_20261021.py`. Every collector/evaluator/replay worker
calls `verify_runtime_sources()` once at startup and saves the receipt. Bind
ruleset `splendor_base_qualified_rules_20261018_v1`, catalogue
`db4776918e612eee5a6f2617719c85ec5f413d592bbab0a64058babc83a70312`, and the current
atomic pre-reveal turn contract. No historical dataset decoder or default
SplendorEnv may reconstruct these positions/worlds. No training loader is used
or certified here.

Before any panel collection, demonstrate the policy's end-to-end counterexamples
through the chosen observer/snapshot decoder, actual legal encoder, hidden-world
sampler, both policy paths, compound resolver, transitions, audit and terminal
scorer: all six corrected faces; non-purchase nobles; multiple eligible nobles;
noble-triggered final round; gold payment/returns; private reserves. Reject
missing/old identities and bad hashes. Check exposed card costs in actual model
inputs, not just metadata. Verify policy-logit/action-index parity, masks and
ties against a separately written reference on bounded fixtures. Include Main,
auxiliary, single-choice, forced-pass and turn123/124 cases, along with malformed
native responses and interruption/clock tests. Helper self-check alone is not
entry-point qualification.

Then run all18 B decisions at the full120-second complete-decision limit,
including candidate reproduction, all4-or3 ×128 continuations, evidence writing
and response delivery. Setup has a separate60-second limit. Repeat six fixed
privacy variants: one root per fit/seat, phase index=(fit_index+seat) mod3.
Alter only actual hidden state before lawful redaction; require identical
observer payloads and byte-equivalent scientific outputs. These six repeats add
no scientific observations. All24 requests must complete correctly within120s;
no truncated-world choice, quota reduction or fallback is allowed. Report margin
to120 and latency separately; finite qualification is not a deadline guarantee.

Because the new native policy operation changes a shared action path, **fully
replay all9,088 original B panel trajectories**, including policy choices,
outcomes and hashes, with independent dictionary transition checks. Privacy
repeats must match their original scientific trace identities. Use a300-second
verification request limit per original root, separate from the120-second player
clock. Reusing mature engine code does not waive this bounded full verification.

For assessment, independently check every recorded branch move, legal action,
lawful observation, transition, ending and terminal score. Check every PUCT16
decision's authenticated profile, receipt and delivered quota/branch diagnostics;
do not claim to replay every internal PUCT edge or independently validate the
shared checkpoint. Retain that qualified source/parity trust boundary. Recompute
all paired differences, fixed-root statistics and bootstrap with an independent
analyzer. Authenticate consumed inputs before/after and bind final artifacts.
Apply [the verification policy](verification_policy.md); no failure-driven case
substitution or reduction in verification scope is permitted.

## Resource envelope and failure handling

| Limit | Fixed rule |
|---|---|
| Absolute job duration | **8 hours** from first model-bearing qualification launch; includes waits, audits and analysis |
| Qualification/selection | At most3 hours; includes preflight,24 timed B requests and full18-root replay |
| Assessment collection cutoff | Stop collection at launch+6h, preserving the final2h for verification/analysis |
| Compute | One active computational worker tree, one assigned performance-core logical CPU and one numerical thread; CPU only |
| Memory | At most2 GiB summed worker-tree RSS, sampled with process identity checks |
| Memory admission | At least2 GiB available for60 consecutive seconds initially; recheck at worker/root boundaries with a healthy fast path, at most30min cumulative waiting inside8h |
| New evidence | At most2 GiB; initial free disk at least7 GiB and maintain5 GiB free |
| B decision / initialization |120s complete response /60s model setup |
| Assessment decision |1s external complete response per actor; PUCT16 retains0.9s internal limit |
| Assessment branch | At most120s elapsed and absolute state turn_index<400 until natural termination |
| Retries / extensions | None automatically; preserve all partial evidence and errors |

The frozen supervisor must enforce real clocks and descendant cleanup. Clamp
every request/branch deadline to the remaining phase and global deadlines; no
120-second request grace may cross the6h collection cutoff, and no300-second
audit grace may cross the3h qualification limit. At worker/root boundaries,
proceed immediately if memory remains healthy; after a shortage require60 stable
seconds before admission. Never insert admission waits inside a live assessment
branch or move. No active
move clock pauses or resets because RAM becomes scarce. Resource trouble or
interruption during a request ends it as a failure; preserve the request and
stop dependent work. A later restart requires a separately disclosed plan.
In this diagnostic, assessment timeouts, unresolved deadlocks or cutoffs are
**incomplete assessment**, not invented losses/draws. The prior matched
benchmark's timeout-as-loss rule and result remain unchanged. Future strength
games will need their own frozen operational-loss policy.

Require official natural completion for every required assessment branch. A
forced-deadlock, repetition/no-progress or turn ceiling is recorded unresolved
and blocks the positive classification; never drop that world or score by
current prestige. The B internal selection convention is separately reported,
not carried into official assessment labels.

Use the first balanced assessment wave—one paired world at every changed root,
world0—as an outcome-blind throughput check. These branches are part of the
fixed32, not extra data. Inspect only time, size and completion status. Continue
only if projecting the remaining fixed workload from that wave with35% cost
contingency fits before launch+6h, and projected evidence fits2 GiB. If it does
not, retain evidence and hold; do not shrink32, remove roots, change quota or
inspect wins to decide. Final verification retains its2-hour reserve regardless.
The cap is not a completion promise: using all3h for qualification leaves3h for
collection, less than the central cost scenario with35% contingency.

## Cost basis and remaining uncertainty

PUCT16's corrected screen measured mean complete requests0.226394s for the
opponent and0.140769s for raw035. With all18 roots changed,1,152 branches and
20 remaining decisions per side, the scenario is
`1152 * 20 * (0.226394 + 0.140769) / 3600 = 2.35 hours`
of supervised move time. At10/30 decisions per side it is1.17/3.52h. These figures
exclude some initialization, state construction, I/O and verification. Actual
remaining games need not have those lengths. Model setup is amortized per root;
fresh model processes on every simulated turn would invalidate this estimate.

B's latency has not been measured. The existing6400 qualification used252,506
neural calls over46 decisions averaging30.102s, about5.5ms per call including
other search/transport work. For5,000–14,000 opponent calls that suggests roughly
28–77s before retained raw035 work and new conversions. This is a rough
engineering scenario, not a guaranteed neural timing measurement or a120s pass.
The fixed full-work qualification, with no adaptive quota, resolves feasibility.

## Preparation artifacts and next permission boundary

`design.json` records this exact mechanism,32-world assessment, decision rule and
resource limits. `candidate_lock.json` binds the18 roots and existing candidates.
`preparation_receipt.json` authenticates sources and verifies coverage/count/cost
arithmetic. Preparation is not a runnable or qualified evaluator. The next
authorized action would be **implement this fixed design and run its bounded
qualification and assessment**; it is not launched by this preparation.

Larger-data collection remains deferred. No historical targets are relabelled,
no new fit is proposed, and no new games against PUCT6400 are part of this study.
