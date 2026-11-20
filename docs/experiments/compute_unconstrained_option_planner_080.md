# Compute-Unconstrained Persistent Option Planner (`080`)

Frozen after the owner explicitly changed the prospective project objective to
permit substantially more compute, and before implementing this runner or
generating any `080` game. This does not alter `079`: that experiment remains an
operational invalidity under its frozen deployability contract.

## New question and champion class

`079` supplied no gameplay-strength or mechanics result. Its exact planner was
terminated solely because the 40-pair panel exceeded a frozen 45-minute limit.
The owner has now stated that compute cost is not a constraint. `080` therefore
asks a different prospective question:

> Is the already-frozen persistent card-option planner strong enough to become
> the compute-unconstrained research champion?

A pass creates a separate **compute-unconstrained champion**. Raw checkpoint
`035` remains the fast/deployable champion. `080` is never pooled with `079` and
cannot retroactively make `079` pass.

## Frozen operator

The candidate is byte-for-byte the `079` operator in
`src/splendor_ai/bots/persistent_option_bot.py`:

- persistent concrete card target;
- qualifying threshold `points / effective_cost >= 0.40`;
- own reserved targets before visible targets, ordered by effective value,
  prestige, cost-colour count, and card ID;
- literal lexicographic value: terminal result, purchased/reserved/visible/lost
  target status, funded prestige, negative deficit, then unchanged Greedy value;
- depth `3`, root branching `8`, internal branching `4`;
- four public-observation determinizations; and
- exact no-bonus full-action fallback when no target qualifies.

No code, target rule, tuple order, depth, breadth, determinization count, deck
behavior, or heuristic coefficient may change. No human replay, training, BC,
GPU, or parameter sweep is used.

Opponent and same-state dose comparator are raw legal `035`, SHA-256
`4422a82b593340eecd02af3a0941f24f0318cc7453e6a0739fd9f524539af2cd`.

## Stage A: mechanics with expanded compute allowance

Run 40 fresh paired seeds `775000--775039`, candidate once from each seat: 80
games. These seeds are disjoint from `079`; no partial `079` state or outcome is
recovered or reused. Stage-A outcomes are never summarized or gated.

All mechanics/integrity conditions remain exactly those of `079`:

- exact seed/seat coverage and byte-identical first-cluster replay;
- every game `completed` or `forced_pass_deadlock`, with zero stalls, timeouts,
  illegal/null actions, or loop fallbacks;
- frozen checkpoint hash and exact planner constants;
- movement from raw `035` at least 15%;
- an active option on at least 50% of candidate decisions;
- at least half of selected options complete successfully; and
- at least 5.0 purchases per candidate game.

The new prospective operational allowance is Stage-A panel wall time at most
**12 hours**, with its linear projection for the 200-pair strength panel at most
**48 hours**. Only these timing limits differ from `079`. If either fails, stop
invalid; do not alter the operator.

## Stage B: fresh strength panel

Only if Stage A passes, run 200 fresh paired seeds `776000--776199`, candidate
once from each seat against raw `035`: 400 games. Limits remain
`max_turns=400`, `repetition_limit=4`, and `no_progress_limit=60`. No game may be
replaced or excluded.

Primary endpoint: candidate paired score share. Use a 20,000-resample paired-seed
bootstrap with seed `776080`. With paired-cluster SD about 0.32, 200 clusters
have about 82% power for the declared 56.5% expected share against the conjunction
of observed share at least 55% and CI95 lower bound above 50%.

Report the same mandatory dose and option telemetry as `079`: overall/phase
movement from `035`, option selection/decision/completion/loss rates, action mix,
reserve and purchase tiers, deck reserves, gold spent, purchases per candidate
game, turns, terminations, and total CPU/wall cost. Diagnostics never rescue a
failed primary conjunction.

## Decisions

- Integrity or expanded operational failure:
  `compute_unconstrained_option_planner_080_invalid_no_strength_decision`.
- Stage-A mechanics failure:
  `compute_unconstrained_option_planner_080_mechanics_failed_close_architecture`.
- Stage-B share at least 55% and CI95 lower bound strictly above 50%:
  `compute_unconstrained_option_planner_080_stronger_promote_research_champion`.
- Valid Stage B with CI95 upper bound below 50%:
  `compute_unconstrained_option_planner_080_weaker_close_architecture`.
- Any other valid Stage B:
  `compute_unconstrained_option_planner_080_not_decisively_stronger_close_architecture`.

A pass promotes only the fixed operator as compute-unconstrained research
champion. It does not replace `035` as fast champion and authorizes no tuning or
training. Every non-pass closes this exact architecture under both champion
classes. Do not add seeds, pool panels, or change the new timing limits after
observing progress.

## Evidence

Single artifact: `outputs/compute_unconstrained_option_planner_080/result.json`.
Record its SHA-256 in `AGENT_CURRENT.md` after the single frozen run.

## Result

The single frozen run completed Stage A on 2026-08-05 and stopped before
strength evaluation with decision
`compute_unconstrained_option_planner_080_invalid_no_strength_decision`.

- All 40 paired seeds and 80 games were present with exact seed/seat coverage,
  deterministic replay, the frozen contract and checkpoint, zero illegal/null
  actions, zero loop fallbacks, and zero stalls.
- Seventy-nine games completed normally. One game, seed `775004` with the
  candidate in seat 1, reached the repetition cutoff. It was therefore marked
  timed out, so the frozen valid-terminal and zero-timeout integrity gates
  failed.
- The candidate moved from raw `035` on 87.95% of decisions, had an active
  option on 98.83% of decisions, and completed 80.41% of selected options. All
  three corresponding mechanics gates passed.
- It made 365 purchases across 80 candidate games, or 4.5625 per game, below
  the frozen minimum of 5.0. The purchase mechanics gate failed.
- Stage A took 11,140 seconds (3.09 hours), within 12 hours. Its linear Stage-B
  projection was 55,701 seconds (15.47 hours), within 48 hours. Compute was not
  the blocker.
- Stage B was not run, no gameplay-strength decision was made, and neither
  champion changed. Under the preregistered non-pass rule, this exact persistent
  option architecture is closed under both champion classes.

Artifact SHA-256:
`ba056955db5387a3834b7e05bd5b05354bf0f46ad53f844a5c263a743adeb94e`.
