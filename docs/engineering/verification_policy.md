# Verification policy for future Splendor experiments

Adopted by the owner on 2026-09-20: finish the current full audit, then use
all-game checks plus sampled continuation replay for ordinary experiments.
Reserve exhaustive continuation replay for substantial changes to simulation
machinery or investigating a discrepancy. This is the prospective default;
it does not revise completed or already frozen experiment protocols.

## Current commitments

- The running learned-proposal attribution experiment completes all402 games'
  full prescribed replays and final statistics. Its provisional229W/0D/173L
  readout does not change verification, sample size or stopping rules.
- The prepared batching benchmark retains its frozen full-trace equivalence
  checks and full-root audits. Batching changes continuation execution and
  arithmetic, so exhaustive verification of that bounded workload is warranted.
  Its separate timing stage also retains the existing verification cost it was
  designed to measure. No benchmark stage starts automatically.
- Existing files, checkpoints, evidence and frozen identities stay intact.
  A later experiment implements this policy in its own versioned verifier.
  No current executable or frozen benchmark plan is changed by this document.

## Always check every actual game and decision

Independently replay every recorded actual move from the declared initial state.
Check legal actions, state transitions, termination conditions, final states,
winners, scores, seat pairs, model assignments and the complete fixed schedule.
Verify code/model identities and the linkage of games, decisions and artifacts.
Reconstruct final statistics independently with the correct paired/cluster unit.

At every actual decision, reconstruct the intended proposal list, concrete
representatives, legal-input encoding, tie rules and choice from recorded
continuation scores. Verify information restrictions for the active models and
schedules, including invariance to opposing private cards and hidden deck order.
Check all declared world seeds, observer-legal reconstructed worlds, candidate
counts and promised budgets; do not infer delivered compute from a configuration
label. Check every recorded continuation entry for completeness, admissible
termination/value fields, and consistency with aggregates and the selected move.

These checks still require work. Checking an un-replayed continuation's record
for structural consistency does not verify that its recorded trajectory or
outcome is correct. Reports must distinguish these two assurance levels.

## Default continuation sample for mature simulation code

Freeze the sampling rule, seed, strata, failure response and cost allowance in
the new protocol before collecting outcomes. Selection must not depend on wins,
value estimates, which move won the search, observed execution time or whether a
case has already passed an audit. Apply the same rule to all experimental arms.

A concrete default for world-based planners:

1. At **every actual decision**, sample whole worlds uniformly without replacement
   and replay every offered candidate in each sampled world. For N worlds, take
   `min(N, max(2, ceil(N / 16)))`. This means2 of32 worlds, or8 of128. Retaining all
   candidates within a sampled world checks their shared-world coupling and
   avoids checking only the move eventually selected.
2. Select indices by the smallest hashes of a fixed domain tag, precommitted audit
   seed, study/stage/game/turn identity and world index. Do not use a root's outcome
   hash, timestamps or scores as the selection seed. The audit seed must not enter
   the player's decision rule. Specify unambiguous serialization and stable ties.
3. Add full-world replay of a small, deterministically selected stress subset.
   Default: one root per deployed arm/model x game phase x acting seat x private-
   reserve category, plus one per arm/model for offered supports1/2/3 if present.
   Phases are turns0--19,20--39,40+; privacy categories are neither, own only,
   opponent only and both. Choose minimum audit-priority hash within each cell,
   using only identity and lawful metadata. Missing cells are reported, not filled
   by extra games. Count overlapping selections once.
4. Retain targeted fully checked preflight fixtures for ties, private reserves,
   stochastic refills, forced moves, termination boundaries and mixed legal-action
   counts. These cover deliberately difficult cases independently of natural
   sample frequency. A new action family or known failure mode may justify an
   additional prespecified stress stratum.

For32-world roots, the ordinary sample replays6.25% of continuations before stress
supplements. This is a pragmatic engineering allocation, not a guarantee of a
particular bug-detection probability or a measured runtime reduction. The all-game
checks, shorter/longer branches, I/O and stress cases also cost time. Price the
actual design before launch. A different planner may need a different sampling
unit; document it prospectively rather than copying the fraction mechanically.

Sample selection may be materialized after the artifact inventory is complete,
provided its rule and seed were already fixed and selection does not inspect
outcomes. Save the full sample manifest, eligible cell counts, selected identities
and hashes before replay. Never replace selected cases because they fail or take
longer. A clean sample leaves some rare bugs undetected; it is not exhaustive
verification and does not create extra statistical observations about strength.

## When exhaustive verification is appropriate

Use full continuation replay on a bounded validation workload for changes to
batching or parallel scheduling, device/numerical execution, game rules,
determinization/hidden-information handling, continuation termination, RNG
coupling, action mapping or other shared simulation machinery. Include exact
action/state-trace comparisons where equivalence is claimed. The size and
coverage of that workload are specified in advance; a clean finite workload
does not prove universal equivalence.

New root-ranking/model comparisons that reuse validated simulation machinery
normally use the sampled policy, while checking every changed proposal and
actual decision. A new architecture or computation-budget change is assessed by
which execution paths it changes, not automatically assigned exhaustive replay
for an entire arena. Training reconstruction, model-selection integrity and
data-split checks remain separate questions; this policy does not silently waive
them.

An unexplained mismatch blocks acceptance of the study's result. Preserve failed
and partial evidence, diagnose the discrepancy, and broaden or fully replay the
affected implementation/cases as warranted. Do not delete a failing sample,
substitute a passing run or keep sampling until the problem disappears. Freeze
the escalation procedure and operational ceiling before launch; an over-budget
diagnosis is a documented hold, not an unbounded automatic compute job. If a fix
changes player behavior, it is a new version requiring appropriate fresh evidence.

## Reporting and scientific limits

Every report states actual games checked, actual decisions checked, worlds and
continuations replayed versus recorded, sample/stress selection, failures,
verification wall/CPU time and shared trusted dependencies. Use wording such as
"all games checked; prespecified sampled continuation replay passed". Never label
that as "all continuations verified" or retain a misleading `full_inner_replay`
flag. Final acceptance requires all checks actually prescribed for that study.

Repeat execution checks implementation consistency. It does not add independent
games or narrow the win-rate confidence interval. Verifiers that share the rules
engine cannot independently establish that engine's conformance to Splendor;
rules conformance needs separate fixtures or comparison. Report strength and
implementation assurance separately, and clearly label any authorized provisional
readout before its prescribed checks finish.

The engineering objective is to spend less time repeating mature computation
while retaining checks most likely to expose invalid evidence. Measure savings;
do not assume that a6.25% continuation sample makes the entire audit16times faster.

## Applying this policy

Future protocol authors should cite this policy and specify their selected audit
mode and exact sample manifest construction before outcomes. Implement it in a
new verifier outside frozen inventories; test missing/tampered records, biased
or duplicate sample selection, sampled-branch mismatches, all-game reconstruction
and accurate coverage reporting. Readiness alone neither launches an experiment
nor changes an existing pipeline.

References: [current attribution protocol](learned_proposal_attribution.md),
[prepared batching benchmark](batching_benchmark_readiness.md),
[previous arena costs and verification](learned_proposal_arena_result.md).
