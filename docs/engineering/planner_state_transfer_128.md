# Frozen learner transfer on planner-reached states

Owner authorized implementation and hidden execution on2026-10-06. This is a new
fixed experiment. No new training, model selection, promotion, arena, sample
extension, automatic retry, monitoring or successor is authorized by this protocol.
All earlier code/results, including dirty work, remain preserved.

## Question and interpretation

The completed equal-budget arena scored198W/4D/200L over402games:49.7512% score,
paired95% interval44.9005–54.6020%. U8x64 did not establish a material gain over
A4x128. Earlier T8x128 scored56.2189% against A4x128 at approximately1.94 times
decision cost. These findings do not establish that small conditional decision
gains must convert to large whole-game gains.

The frozen broad learner was trained on richer planner labels but failed its
practical screen on fast161-generated states. This experiment asks whether that
same frozen learner improves conditional planning utility on states actually
visited by the completed U/A planners. It explicitly revisits a failed learner
on a new population and fresh-world endpoint. Its original failure remains valid.
It does not identify a causal population interaction without a concurrent
cross-population test; it does not establish whole-game improvement or superhuman
strength. No checkpoint or recipe is changed after seeing this panel.

## Models and arms

Reuse the original nine selected checkpoints from the closed
proposal_breadth_learning_128 registry, binding
bf916c18fc25e5be4bb2b7ff13317157d155da82668ae4d8a86308544b5ca72a.
Three fits161001/161002/161003 each supply original anchor, BROAD and matched-dose
NARROW. Do not substitute specialist-reselected models. The new private registry
pins all nine SHA256s and the unchanged035 continuation checkpoint. Closed training
receipts are inherited; historical fitting and inner simulations are not rerun.

All arms retain the same035 top-two proposals and first random proposal:

1. Anchor: original161 fourth proposal, at most4actions x128 selection worlds.
2. Broad: frozen broad-trained fourth proposal, same budget.
3. Narrow: original matched-control fourth proposal, same budget.
4. Teacher: original T expansion of anchor4: next two anchor-ranked remaining
   groups plus two identity-uniform remaining groups, at most8actions x128worlds.

Teacher is diagnostic context, neither truth nor an upper bound: Broad may offer
an action outside its support. The common union contains at most10 distinct
canonical successor groups. Short supports and identical proposals are retained.
Each arm reads only its own offered support in its original order. Equal means
select the first offered action. All three learned proposers see only legal public
observations and their own private information.

## Source population and fixed panel

Reuse completed402games from equal_budget_breadth_arena_128, authenticated through
R3 binding904b6780bceedca3d620f7a548ba4e903b8558e0a3841f065dfb4eed8d5530c8.
ResultSHA2dde6af8330cb6ed3c5afa0369cca3a48c7646087cf903d753f07d93ddc73d03;
auditSHA22f36a08694dd9160c14c2cc0110c07f9909bbb0bf5462ddd21d89000d5c9572.
The final audit pins all402 game receipts. Selected roots link to their original,
R1 or R2 originating artifacts and their passed root receipts. No new source games.

Each fit has67 reversed-seat source pairs. Sort pairs by the new frozen identity
hash. The first3 are probe pairs assigned respectively to phases0–19,20–39,40+;
their source arm is assigned by a separate identity hash. The next60 are main
pairs, cycling across six phase/source-arm cells,10pairs per cell. The last4pairs
per fit remain unused. Main therefore has180pairs/360roots; probe9pairs/18roots;
the two panels share no whole pair or game.

Both legs of each selected pair contribute one position from the assigned source
arm and phase. The exact turn is the minimum new identity hash among eligible
turns. Sampling uses game/pair/turn/actor metadata only, never winners, scores,
disagreement, margins or proposal preferences. Source selection does not discard
losses, private reserves, short supports or agreements. Missing fixed cell means
hold, with no replacement. All67pairs per fit were checked before this protocol to
have both games extend beyond turn41, without inspecting outcomes for selection.
Authenticated source game environment seeds are disjoint from all1204 historical
training/validation corpus game references. This excludes literal source-game
reuse; it does not assert that no semantically identical board can ever recur.

## Fresh evidence and unresolved outcomes

Every root receives128 common selection worlds. Exact original-order maximum
choices for all arms are sealed durably before128 disjoint assessment worlds
evaluate at most four distinct chosen actions. Root/stage/block namespaces are
disjoint; world seed base2180000000, offsets probe0/main1000000/test2000000,
root index x512 and assessment+256. Greedy035 continuations and original natural
termination,400turn, repetition4 and no-progress60 guards remain unchanged.

Unresolved repetition outcomes remain null, bounded by loss/win. A selection may
continue only if exact worst-case pairwise bounds certify the same first maximum
for every arm. Where independent unknown bounds are insufficient, the previously
validated R2 equivalence proof may tie unknown variables whose full effective
future contexts are identical. Only purchased-card order may be normalized; guard
history, counters, all other state,035 identity and scoring actor remain exact.
Proof traces are saved and independently reconstructed. Proof replays are extra
charged work, not additional allocated selection evidence. If choices remain
ambiguous, preserve the root and hold; no fallback, extra worlds or invented draw.

Unknown assessment outcomes produce score and paired-difference intervals. Arms
choosing the identical action have exactly zero paired difference, even if that
action has unresolved worlds. Other unknown differences use conservative bounds.
No midpoint or draw imputation. All unresolved cells receive independent replay.

## Fixed statistics and decisions

Scores are win1/draw0.5/loss0 under the frozen continuation. Average the two roots
inside each whole source pair. The independent units are180pairs, not360roots or
simulation worlds. Use50,000 paired bootstrap samples (seed2183000000), resampling
ten whole pairs within each of18 fixed fit/phase/source-arm strata. All contrasts
share draws. Strata ordered by fit, phase, candidate then reference; pairs ordered
by source identity. NumPy linear2.5/97.5 percentiles, chunks1000.

For unknown outcomes, bootstrap lower and upper paired bounds on identical draws;
the interval envelope uses the2.5th percentile of lower values and97.5th of upper.
The mean is an interval unless exact. Both gates are required:

- Broad minus anchor: mean lower bound >=+0.005 and95% lower endpoint >0.
- Broad minus narrow:95% lower endpoint >0.

Teacher-minus-anchor and narrow-minus-anchor are descriptive. Teacher cannot
rescue or retroactively invalidate either primary gate. No switching to Narrow if
it looks better. Fit/phase/arm breakdowns cannot replace the pooled gates. A pass
earns consideration of a separately authorized whole-game confirmation, not
promotion. A miss leaves transfer unestablished; another clear miss closes this
frozen-model transfer hypothesis. No third population rescue or automatic sweep.
An inconclusive interval remains inconclusive, without extending the sample.

## Verification and resource contract

Before inference: focused meaningful preflight, immutable code/runtime/protocol
identity, model/source/selection seals, source-game disjointness, complete audit
plan. Actual hidden-worker fixture loads the private models without production
play. Tests include JSON snapshot serialization, corruption rejection, sparse
union/accounting, disjoint worlds, original-order ties, unknown bounds/coupling,
paired bootstrap, cost gate and bounded cooperative dispatch.

Every new root gets independent proposal/privacy reconstruction, complete saved
world identity/accounting, selection/assessment arithmetic and provenance checks.
Replay8/128 identity-selected whole worlds in each block for ordinary main roots.
Full replay of18probe roots and up to39 main stress roots: one per fit/seat/phase
(18), fit/private-reserve category(12), fit/short-support1/2/3(9), deduplicated;
missing categories are disclosed. Every unresolved cell is additionally replayed.
Final collector and independent receipt statistics must agree. Mature environment,
encoding, neural arithmetic and proven continuation primitives remain trusted.
Full replay of every main inner continuation or historical training is not claimed.

At most645120 main plus32256 probe allocated continuations before independent
replay. Eight hidden CPU workers, one deterministic Torch thread each. No GPU.
Bounded work dispatch and cooperative abort preserve all completed/failed artifacts.
The detached pythonw supervisor inhibits system sleep but permits display sleep;
closing the conversation does not terminate it. A machine shutdown still interrupts
it. The one-shot runner never silently overwrites or automatically restarts an
attempt; any required repair must retain the old evidence and explain the change.

The fully replayed18root probe gates main only on measured time, storage and audit
validity, never strength outcomes. Project remaining collection and sampled replay
from separate per-block measured rates and actual main support upper bounds, plus
structural costs,35% allowance and300seconds finalization. Require total elapsed
plus forecast <=8hours from this new launch and at least5GiB free after projected
storage. Forecast includes a1.5 storage factor and512MiB allowance. Extra unknown
or proof replay counts are reported; unpredicted work remains subject to the fixed
hard ceiling. Hold on unresolved provenance, audit, resource or choice ambiguity.
