# Learned proposals at the original teacher's 128-world budget

2026-09-21. The owner explicitly authorized the prepared batching benchmark and
then this experiment, including ordinary operational adjustments without another
approval round. The benchmark is handled in the current conversation; this
experiment runs in a hidden, detached Python supervisor after preparation and
checks. Preserve all existing, intentionally dirty work and frozen evidence.
No training, promotion, further experiment or recurring monitor is included.

This implements the scientific recommendation in
[next_after_proposal_attribution.md](../strategy/next_after_proposal_attribution.md).
The attribution experiment has now completed all prescribed checks: 229 wins,
173 losses, no draws, 56.9652%, paired 95% interval 52.4876–61.4428%.

## Question and players

Does the successful learned fourth-proposal rule improve the exact original
128-world teacher from Experiment 160 when both receive the same simulation
allowance? This tests the combination of better proposals and more worlds.

The reference is exactly the original 160 planner at 128 worlds, delegating to
its unchanged chooser. Its offered actions are the original 035 top two distinct
canonical groups plus two random alternatives. It is not the 035-ranked fourth
proposal used in the attribution experiment.

The candidate preserves the successful learned-proposal rule: the same first
three actions from the original list, followed by the assigned frozen 161 model's
highest group-max score outside those three. Fits are 161001/update563,
161002/update488 and 161003/update563, with inherited checkpoint hashes and
training/selection provenance. All three remain included, fixed within each seed
pair. No best-fit selection, ensemble or phase filter.

Preserve concrete representatives, canonical group order, complete lawful legal
inputs and original-order ties. For supports below four, retain the whole
original list unchanged. Keep coincident proposals and ordinary private-reserve
positions. Greedy 035 plays both sides of every simulated continuation. Terminal
simulation values are win 1, draw 0, loss -1; actual-game statistical scores are
win 1, draw 0.5, loss 0. These scales must not be confused.

Both players evaluate every offered action in 128 coupled observer-legal worlds.
Preserve the original 32-world prefix and the exact 160 continuation of the
schedule: candidate nonce is the digest of
`["155-candidate-v1", stream, belief]`; world seed strings are
`155-world-v1|stream|belief|world_index:02d`, indices 0 through 127.
The belief digest uses the acting player's lawful information. No mutable change
to old global configuration is permitted. Hidden deck order and opposing private
card identities must not influence inputs, proposals, schedules or choices.

No adaptive allocation or stronger continuation policy is introduced. The initial
implementation uses the existing serial trajectory path. Batching may be adopted
only before this study freezes, following the engineering decision below and
bounded validation at 128 worlds. Backend identity must be explicit in the frozen
configuration; it cannot change partway through the study.

## Fixed fresh schedule and endpoint

Main: 402 games, 201 distinct starting seeds played in both candidate seats,
67 pairs per fixed fit. Probe: 12 separate games, two pairs per fit, excluded from
the endpoint. Test: six isolated synthetic games, one pair per fit, which cannot
write into the production output or supply production evidence.

Main environment seeds are 2060000000 + pair; probe 2069000000 + pair; test
2069800000 + pair. Sort pair indices by the digest of
`["learned-proposal-128-fit-v1", stage, pair]`, then assign ascending fits by
rank modulo three. Game index is 2*pair + candidate_seat. The root stream is the
digest of `["learned-proposal-128-stream-v1", stage, pair]`, shared across a pair.
These namespaces were checked against existing scripts, tools, tests and docs.

The endpoint gives equal weight to the three fixed fits and retains both seats
within each seed pair. Use 50,000 paired bootstrap samples within fit with
numpy.default_rng seed 2063000000, 50 chunks of 1,000 samples and 67 pair indices
per fit in ascending fit order. Sum integer pair half-points and divide by 804;
use linear 2.5% and 97.5% quantiles. Reconstruct statistics independently.

Superiority requires the 95% lower bound above 50%. Practical progress also
requires observed score at least 55%. This is not proof of a true score above
55%. Preserve modest-positive, below-target and inconclusive classifications.
No optional stopping, extra games, subset/model selection, promotion or automatic
extension follows a near miss. Subgroup and latency summaries are descriptive.
Do not use probe strength or interim main strength to alter the design.

## All-game checks and sampled simulation replay

Apply the owner-approved [verification policy](verification_policy.md). Check
every actual move, legality, state transition, final state, result, fixed seed
pair and model assignment. At every decision independently reconstruct lawful
model inputs, proposals, concrete representatives, hidden-information invariance,
all 128 seed/world states, admissible continuation records, means and the chosen
action. Preserve the full recorded outcome array and provenance for later
learning development; opposing players' logs are not matched same-state labels.

At every root replay all offered actions in eight distinct worlds selected
without replacement by the smallest hashes of the fixed domain
`learned-proposal-128-audit-world-v1`, audit seed 2064000000, stage, game index,
turn and world index. Hash the compact, sorted-key JSON array
`[world_tag, 2064000000, stage, index, turn, world_index]`; rank by hash then
world index, retain the first eight and store them in ascending world order.
This audit randomness never enters the player's policy or game schedule.

Add complete 128-world replay for deterministic stress roots. Strata are arm
(candidate/reference) x assigned fixed fit x phase (turns 0–19, 20–39, 40+)
x acting seat x privacy category (neither, own only, opponent only, both).
Select the minimum identity hash in each eligible cell using domain
`learned-proposal-128-audit-stress-v1` and the same audit seed. Hash
`[stress_tag, 2064000000, stage, index, turn]`, with the stable zero-padded
`index:turn` root identity breaking hash ties. Also select one
root per arm/assigned fit for each offered support 1, 2 or 3 when available.
The reference remains the single unchanged 035-based planner; stratifying by
assigned fit covers the different opponent cohorts and does not create models.
At most 162 distinct stress roots exist per stage; overlapping selections count
once. Report missing cells instead of generating replacement games.

Seal each stage's complete root inventory, eligible cells and exact sampled/stress
identities after collection, before replay. Selection uses identity and lawful
metadata only, never choices, wins, values, execution times or pass/fail outcomes.
Validate the manifest against every game/root before audit and at finalization.
Ordinary eight-world replay remains distinguished from the additional 120 worlds
of a stress root, including timing and exact branch counts.

Any mismatch blocks scientific acceptance. Preserve partial and failed evidence;
do not replace games or audit samples. Diagnose and broaden replay of the affected
cases if necessary. A behavior-changing correction requires a new frozen version
and fresh evidence. No unlimited retry or silent audit reduction is permitted.
The final report states actual games, roots, recorded/replayed worlds and branches,
stress coverage, costs and failures. Set `full_inner_replay` false. Shared rules,
encoding and neural primitives remain trusted dependencies; the verifier is not
an independent implementation of Splendor's rules.

## Preflight and engineering decision

Before production, check exact 160 reference delegation, unchanged learned
proposal semantics, the 32-world prefix and full 128-world bounded fixtures.
Include both seats, all fixed fits, private reserves, ties, short/forced supports,
refills and terminal boundaries. Check changed/missing records, sample uniqueness,
outcome-independent stress selection, sampled-branch mismatch detection, coverage
reporting and detached Windows workers. Tests must not overwrite evidence.

One historical 160 test assumed its production output did not yet exist and
expected snapshot rejection before the overwrite guard. With completed evidence
present, the unchanged code correctly rejects the existing file first. The new
preflight explicitly replaces that single case with a regression checking the
existing-file refusal and unchanged hash, snapshot rejection in an isolated empty
output, both production test-path guards and the original auditor's rejection.
The old test/code/data remain untouched; the exact deselection and replacement
are bound in the check receipt and identity. All other selected legacy cases run.

The original prepared batching benchmark and its first launch error are retained.
Its isolated version 2 fixes only recycled-process-ID detection and package/output
paths. Original positions, timing schedule, numerical code and checks remain the
same. It compares exact traces before measuring serial versus batch16 speed.

Version 2 stopped on the first serial reference before attempting batching:
an in-memory snapshot used integer deck-count keys while its JSON form used
string keys. Every scientific field of the saved new serial root matched the
original saved root exactly. Isolated version 3 fixes the adapter to compare
persisted JSON against persisted JSON with strict equality. It adds regression
checks without changing simulations, trace comparisons or the independent audit.
Both earlier attempts and frozen preparations remain intact. The final version 3
evidence and backend decision will be bound through
`outputs/batching_benchmark_v2/adoption_decision.json`, whose location was agreed
before this repair; its versioned artifact references identify the actual run.

Before inspecting timings, use the following engineering rule: consider batch16
integration only if every equivalence check passes, the summed collection speed
ratio is at least 1.25 and the original full-collection-plus-audit ratio is at
least 1.10. These are adoption screens, not statistical strength claims or a
guarantee of eight-worker/128-world speed. Any integration must additionally pass
bounded exact 128-world comparisons and hidden-worker integration checks before
freezing this experiment. If equivalence fails, measured benefit is insufficient,
or integration cannot be completed and checked within a bounded four-hour effort,
retain serial execution and record the decision. Do not let optimization become
an indefinite prerequisite. No scientific outcomes select the implementation.

## Hidden execution and operational limits

Entry is `run_learned_proposal_128.py`; implementation and tests live under
`tools/learned_proposal_128`, outside historical frozen Python inventories.
Output is `outputs/learned_proposal_128`. Bind all new implementation/tests,
this protocol, inherited code/checkpoints, completed parent evidence, exact
schedules, runtime and benchmark decision before collecting probe outcomes.

Use a durable detached pythonw supervisor and eight hidden pythonw CPU workers,
one deterministic Torch thread each. Keep write-once artifacts, stage locks,
separate durable logs, worker identity receipts and a matching launch
acknowledgement. The parent chat may become inactive after that acknowledgement.
The supervisor requests that Windows remain awake while running without changing
permanent power settings or keeping the display on; it releases the request on
exit. Manual sleep, shutdown and power failure can still interrupt execution.
No GPU dependency, visible terminal, recurring automation or watcher is needed.

The separate 12-game probe and its prescribed audit must pass before main. Use
an outcome-blind time/storage gate. Forecast ordinary collection and verification
from summed probe worker-wall time, scaled by 402/12 and divided by eight workers.
Subtract measured additional stress replay from the ordinary audit component;
add the cost of at most 162 full-stress supplements at their observed mean cost,
then apply a 25% time allowance. Missing/invalid timing evidence fails the gate.
This avoids treating short-probe scheduling gaps or its high stress fraction as
costs repeated throughout main. Freeze the exact executable formula before probe.

Main forecast must be at most 120 hours. Project storage with a 50% allowance
and retain at least 5 GiB free disk. The expected serial runtime remains roughly
70–90 hours including probe and audit, provisional until probe timing. A
conservative ceiling is not the expected duration. A 168-hour total runtime
threshold and the 5 GiB disk reserve are checked at completed-game/audit job
boundaries. A failure cancels queued jobs; already-running jobs may finish and
preserve evidence. These are not hard mid-game termination guarantees. An
operational hold preserves evidence and stops unbounded work.

After main, finish every prescribed check, reverify identities/links, compute the
fixed statistics, save the final result and exit. A positive result supports a
stronger teacher in this matchup; it does not establish broad dominance, equal
wall-time strength, a repeated improvement loop or superhuman performance.
Subsequent controlled learning remains a separate scientific step.
