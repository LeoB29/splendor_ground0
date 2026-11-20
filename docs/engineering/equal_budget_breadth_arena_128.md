# Uniform breadth gameplay at equal simulation budget

Owner authorized2026-10-03. One fixed U-versus-A arena with separate probe,
prescribed audits and no successor. Implementation and output use
equal_budget_breadth_arena_128. Rationale and scope are in
docs/strategy/equal_budget_breadth_arena_128.md. Preserve every predecessor.

## Players and exact allocation

Retain all three original161 checkpoints:161001/update563,161002/update488,
161003/update563, and the frozen035 greedy terminal continuation model. Both
players in a game and both games of a seat pair use the same assigned fit.
No checkpoint selection, fitting, adaptive elimination or model changes.

The reference A retains the original common first three proposals and learned
fourth, original concrete representatives and candidate-order ties. U retains
that prefix, appends up to two highest-ranked remaining canonical groups and
then up to two identity-uniform remaining groups. Additional groups use canonical
first representatives. Candidate generation and observer-lawful belief-keyed
world schedules retain the128-world convention, including the candidate nonce.

For canonical support G, m=min(G,4), n=min(G,8), B=128m. A executes128 worlds
for each of m candidates. U allocates q=floor(B/n) worlds to every candidate,
with one extra for r=B mod n candidates. Choose remainders by sorting the digest
of [remainder_tag,stage,game_index,turn_index,canonical_key], with original
candidate-index ties. Thus full support is eight-by64;5/6/7 candidates use
two103/three102, two86/four85, or one74/six73. G<=4 retains the same full support
and128 outcomes per candidate. At every lawful state both rules have the same
continuation quota B. Different reached states and lengths may change totals.

Actually execute only allocated cells: the first max(counts) schedule worlds,
and only candidates whose count exceeds the world index. Every recorded world
has explicit original candidate_indices and aligned outcomes. The final world
can be sparse. Store counts, allocation priority, delivered totals, raw WDL sums
and exact selected index. Compare rational means by integer cross-products and
break ties by original candidate order. No full eight-by128 shared matrix,
shadow teacher, assessment block or precision extension is run during games.

## Schedule and statistics

Main402 games:201 fresh initial-seed pairs, reversed candidate seats,67 pairs
per fit. Probe12 separate games:six pairs,two per fit, excluded from strength
statistics. Six isolated synthetic test games cannot write production evidence.
Environment bases:main2170000000,probe2179000000,test2179800000,plus pair index.
All namespaces and tags are distinct from earlier studies.

Tags use equal-budget-breadth-arena-128 with suffixes fit-v1,stream-v1,
random-v1,remainder-v1,audit-world-v1,audit-stress-v1. Fit assignment orders
pair indices by [assignment_tag,stage,pair] digest and rotates the three fits.
Search stream is the digest of [stream_tag,stage,pair]. Public root identity is
[stage,game_index,turn_index], with turn below400. Extra uniform candidates use
[random_tag,stage,game_index,turn_index,canonical_key], with canonical-index
ties. Hidden state, outcomes and worker order never determine these identities.

Primary endpoint is U complete-game score,win1/draw.5/loss0. Use all pairs and
fits equally. Bootstrap50,000 whole seat pairs within each fit,67 per fit per
replicate,seed2173000000,50 chunks of1000, integer half-point totals and linear
percentile interval. Independently reconstruct statistics and compare raw saved
game endpoints with audit endpoints. The interval conditions on these three fits.

Practical progress requires observed score>=55% and paired95% lower>50%.
Retain classification priority: practical progress; established superiority
below the observed target; demonstrated harm (upper<50%); material target
excluded (upper<55%); otherwise inconclusive. This does not prove a true55%
score or guarantee adequate power for smaller improvements. No strength-based
interim stopping, extension, subgroup rescue or automatic promotion.

Natural termination contracts remain unchanged:400-turn guard,four repetitions,
60 turns without progress. Artificial cutoffs and illegal actions are failures,
not draws or scored games. Preserve partial evidence.

## Verification

Follow docs/engineering/verification_policy.md. Independently reconstruct every
actual game/decision, legal inputs, private-state invariance, models, original
and expanded proposals, representatives, allocation priorities/counts, all
stored outcome fields, world seeds/determinizations, exact sums/means/choice,
actual moves, final state and outcome. Shared mature rules, encoding, neural and
continuation primitives remain trusted dependencies.

For N actual worlds at a root, select min(N,max(2,ceil(N/16))) distinct whole
worlds by the frozen audit hash law, seed2174000000. Replay every DELIVERED
candidate in each selected world. This is four of64 for full-support U and
eight of128 for A. Structural checks cover all un-replayed entries. Stress
roots replay all actual N worlds and their delivered candidates.

Stress strata are two roles x three fits x three phases x two seats x four
private-reserve categories (144 cells); two roles x three fits x canonical
supports1/2/3 (18 cells); U x three fits x supports5/6/7 (nine cells). Select the
minimum identity-only priority per cell, deduplicate overlaps, report absent
cells without replacement. At most171 distinct roots,90 U and81 A, per stage.
Seal root inventories and sample selections before audit. Actual recorded and
replayed continuations are sums of sparse cell counts, not N times support.

Fully replay bounded preflight adapter fixtures for sparse remainders, rational
ties, private reserves, refills and near-terminal games. Check original anchor
delegation, model-role isolation and windowless workers. Run the focused new
suite plus mature windowless-worker checks; retain failed attempts. Do not
repeat historical training, old experiments or exhaustive main continuations.

## Execution and resource gate

Freeze passing checks, code/tests, runtime, configuration, this protocol,
recommendation, verification policy, model identities, closed fresh-parent
evidence, closed adaptive detour and fresh schedule before production. Scientific
writes are one-shot and immutable. Advisory progress uses resilient reporting.

Hidden detached supervisor, eight hidden CPU workers, one deterministic Torch
thread each, serial continuation backend, batch size one. No GPU, terminal
windows, recurring monitor or follow-on task. System sleep is inhibited during
execution; display sleep is allowed. The chat can close.

Expected total60–75h is provisional. Run and audit all12 probe games first.
Project main collection and ordinary audit worker time by402/12 and divide by
eight. Remove measured full-stress extra replay time before scaling ordinary
audit; add each role's measured extra time per stress root times90 or81 maxima,
divided by eight. Require positive measured stress work from both roles. Apply
25% time allowance, then600 seconds for finalization. Gate forecast<=96h and
elapsed+remaining<=120h from original launch. Budget checks at stage boundaries,
each actual root and inner/audit world, inventory closure and final completion;
an in-flight world can finish before the next cooperative check.

Project main storage as1.5 times probe-stage bytes times402/12 plus512MiB and
require5GiB free afterward. A failed timing/storage gate writes a hold and stops
before main. A resource limit or integrity failure retains evidence and stops
without resetting budgets or changing the schedule. No automatic retry/resume.
Unexpected shutdown preserves completed files; later authorized recovery must
authenticate them, preserve incomplete attempts and use the same frozen player,
fixed seeds and original resource accounting. No unannounced extensions.

Final acceptance requires completed probe/main schedules, every actual game and
decision audit, prescribed sampled/stress replay, accurate coverage, frozen
identities, independent paired statistics and final hash linkage. Report actual
decision timing and per-role continuation totals separately from the equal
per-state quota. No claim of equal time, broad opposition or superhuman strength.
Save final result/audit/report and stop, without training or another experiment.
