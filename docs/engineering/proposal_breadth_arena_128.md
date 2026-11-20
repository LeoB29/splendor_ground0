# Full-game confirmation of broader proposals

Owner authorized on 2026-09-28 after the completed positive breadth screen.
This is one fixed arena, implemented only under
`tools/proposal_breadth_arena_128/` with entry
`run_proposal_breadth_arena_128.py`. All predecessor code and evidence remain
unchanged. The scientific rationale is in
`docs/strategy/next_after_proposal_breadth_128.md`.

## Question and fixed intervention

Does the broader planner beat the current learned128 anchor in fresh complete
games? The candidate retains the exact anchor four proposals, appends up to two
highest-ranked remaining canonical groups and then up to two uniformly ordered
remaining groups. It evaluates every offered candidate in 128 worlds. The
reference retains the exact anchor four and its 128-world selection. When fewer
groups exist, use all available groups and retain those decisions in the study.

Both actors in a game, and both seat games in a pair, use the same assigned
original161 fit. The three original checkpoints are 161001/update563,
161002/update488 and 161003/update563. Their existing hashes and the frozen035
continuation checkpoint are checked and sealed separately. No model is trained,
selected, updated or distilled. The learned fourth-slot rule, concrete original
representatives, group enumeration, original candidate order and first-candidate
ties remain unchanged. Added groups use canonical first representatives.

Each actual decision runs only its own 128-world selection. There is no separate
assessment block, precision extension, shadow player, or extra proposal screen.
The mature observer-lawful belief-keyed world schedule and greedy035 terminal
continuations are unchanged. All candidates share the world's hidden-state draw.
Matched lawful states use the established common draws within a pair stream.

The injective public root identity is `[stage, game_index, turn_index]`, where
turn indices are below 400. Extra uniform proposals are ordered by the hash of
`["proposal-breadth-arena-128-random-v1", stage, game_index, turn_index,
canonical_key]`, breaking hash ties by canonical enumeration index. Hidden
state, outcomes and worker order never enter this identity. This fixes the same
ranked-plus-uniform recipe in a fresh gameplay namespace.

## Prospective allocation and analysis

Main: 402 games, 201 environment-seed pairs with reversed candidate seats,
67 pairs per original fit. Probe: 12 separate games, six pairs, two pairs per fit.
The probe is excluded from every strength estimate. Six isolated synthetic test
games are restricted to temporary directories and cannot write production data.

| Namespace | Probe | Main | Synthetic test |
|---|---:|---:|---:|
| Environment seed base | 2119000000 | 2110000000 | 2119800000 |
| Pair count | 6 | 201 | 3 |

Pair assignment ranks the digest of
`["proposal-breadth-arena-128-fit-v1", stage, pair_index]` and rotates the three
fits in that order. The environment seed is the relevant base plus pair index.
Search streams are the digest of
`["proposal-breadth-arena-128-stream-v1", stage, pair_index]`.

The sole strength endpoint is the candidate's complete-game score: win 1,
draw 0.5, loss 0. All pairs have equal weight and both seat games stay together.
Report wins, draws, losses, overall score and per-fit descriptive scores. The
paired 95% percentile interval uses 50,000 resamples within each fixed fit,
67 pairs with replacement per fit, seed 2113000000, 50 chunks of 1,000, integer
half-point totals and linear quantiles. A separately implemented accumulator and
quantile calculation must reproduce the result, and raw saved-game scores must
agree with audited scores. The interval conditions on these three fixed fits.

Practical progress requires observed score at least 55% and interval lower bound
strictly above 50%. This does not prove the true score exceeds 55%. Separately
identify established superiority below the observed target, demonstrated harm
(upper bound below 50%), exclusion of the 55% material target, and inconclusive
evidence. Classification priority is practical progress, modest superiority,
demonstrated harm, below material target, then inconclusive.

No interim strength stop, added games, subgroup rescue, mixture change,
automatic retry, promotion, second arena or successor is authorized. Invalid
cutoffs, illegal actions, integrity failures and resource limits stop the run
and retain partial evidence. Mature guards remain 400 turns, four repetitions,
and 60 turns without progress; artificial cutoffs are not scored as results.

## Independent verification

Apply `docs/engineering/verification_policy.md`. Reconstruct every actual game
and actual decision, including lawful observations, private-state handling,
model identity, legal actions, canonical groups and representatives, raw and
learned proposal scores, added rank/hash order, world seeds and determinizations,
all stored outcome fields, WDL aggregates, chosen moves, terminal state and score.
The auditor rebuilds proposal grouping and selection without calling the new
collection decision helper. Mature engine, encoding, neural and continuation
primitives are shared trusted dependencies; this is not a second game engine.

For both probe and main, replay every offered action in eight identity-selected
worlds at every root, plus all 128 worlds of deterministic full-stress roots.
All unreplayed world/outcome records still receive structural checks. This is
sampled inner replay, not an exhaustive rerun. Small changed-adapter fixtures
are fully replayed before launch.

Audit seed 2114000000; tags
`proposal-breadth-arena-128-audit-world-v1` and
`proposal-breadth-arena-128-audit-stress-v1`. Seal the inventory and deterministic
selection after collection and before audit, using identities and lawful
metadata only. Ordinary selections contain exactly eight distinct indices from
0 through 127. Stress cells are:

- Two roles, three fits, three phases, two acting seats and four privacy
  categories: 144 cells.
- Two roles, three fits and canonical supports 1, 2, 3: 18 cells.
- Candidate role, three fits and canonical supports 5, 6, 7: nine cells.

Select at most one root per cell, deduplicate overlaps, retain and report missing
cells without replacement. At most 171 distinct stress roots per stage:
90 candidate and 81 reference. Structural coverage, delivered ordinary/stress
indices, actual continuations and replay counts must match the sealed inventory.
Final closure rechecks artifact links and every actual root hash without
replaying historical experiments or repeating the new trajectory audit.

## Execution, resource gate and failure handling

Use a hidden detached supervisor and eight hidden CPU workers, each with one
deterministic Torch thread. Use the serial continuation backend, batch size one.
No GPU or visible terminal is needed; the chat can close. Inhibit system sleep
only while the supervisor runs and allow display sleep. Shutdown/restart still
interrupts the job. No recurring monitor is created.

Expectation is approximately four to five days, extrapolated from prior arenas.
The hard main forecast ceiling is 120 hours with a 25% allowance, and total
execution is capped at 144 hours from this launch. Maintain at least 5 GiB of
free space. These are new budgets; no expired predecessor budget is reused.

Run and audit all 12 probe games before the main arena. The gate reads timing,
coverage and storage only, never scores. Project collection worker wall time
and ordinary audit worker wall time by 402/12 and divide by eight. Subtract
measured extra full-stress replay from ordinary audit timing, then add a bounded
stress allowance using each role's measured extra time per stress root times
90 candidate or 81 reference roots, divided by eight. Apply the 25% allowance to
their sum. Require positive measured stress work from both roles. Add 600 seconds
for finalization when comparing remaining work plus elapsed time to 144 hours.
Project main storage as 1.5 times probe stage bytes times 402/12, plus 512 MiB;
require projected free space to retain the 5 GiB reserve. A failed cost gate
writes an operational hold and stops before main.

Budget checks run at stage boundaries, between actual roots, before each inner
world and audit world, during inventory/final closure, and before completion.
An in-flight world can finish before the next check. Partial strict records and
failure/hold receipts remain; no automatic restart or extension is attempted.
Scientific files use strict one-shot writes and reject overwrite. Status events
and the advisory progress display use the mature resilient reporter; an advisory
file lock must not turn a scientific success into a failure.

## Freeze and interpretation

Before production, run the fixed preflight suite over new models, arena, verifier,
orchestration, end-to-end fixtures and hidden-worker initialization plus the
mature windowless-worker tests. Freeze current passing test receipts, code,
configuration, this protocol, strategy recommendation, verification policy,
runtime identity, original models, bounded closed-parent evidence and fresh
schedule. Validate those identities at launch and stage/final boundaries.
Preserve every existing dirty/frozen file and all failed preflight evidence.

A pass demonstrates better full-game performance in this particular fresh
matchup at additional compute. It does not establish equal-compute or equal-time
efficiency, broad-opponent dominance or superhuman play. Preserve trajectories
for potential separately authorized later analysis or learning. No training or
follow-up launches automatically after either success or failure.
