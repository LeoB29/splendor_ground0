# Recommendation after155: challenge the frozen084 research champion

2026-09-11. Recommendation only. The owner asked for the next best step.
No156 implementation, test runner, inference, game, rollout or training was
launched. Preserve the intentionally dirty worktree and all frozen predecessors.

## Decision

Recommend one experiment, provisionally156: **does the unchanged155 rollout
selector beat exact084 in complete games?** Use400 fresh games from200 starting
seeds, with each player occupying each seat once per seed.

155 scored95.5% against035, paired95% CI[93.5%,97.25%], with complete replay.
This establishes a strong deployed improvement over035. However, the selector
also uses035 for both players inside its imagined futures. The next important
uncertainty is whether its choices succeed against a stronger, different planner.

084 remains the strongest confirmed research champion. Its historical score
against035 was75.875%, CI[71.75%,79.875%]. Those two scores are not a direct
155-versus084 comparison. Matchup-dependent strengths can make opponent-based
rankings misleading; see the general evaluation motivation in the primary
[AlphaRank paper](https://www.nature.com/articles/s41598-019-45619-9).
This is a reason for the direct match, not evidence that these particular bots
have a nontransitive relationship.

134 previously found that expanded086 lost badly to084 despite promising
conditional diagnostics. The new155 evidence is stronger than that old local
signal because it includes actual gameplay; the need to face084 still remains.

## Why this question comes first

- A pass establishes progress beyond the programme's strongest confirmed
  opponent and makes this selector a better-supported candidate for later
  efficiency or learning work. A failure reveals limited transfer from035
  before we invest in training from this controller or enlarging its search.
- The two randomly proposed candidates supplied3,550 of12,185 actual choices.
  That descriptive observation motivates a possible later ablation; it does
  not identify their causal contribution to the95.5% result. Preserve them in
  the direct match so that the tested player remains155.
- Earlier149-153 failures leave learned approximations unresolved. The success
  of expensive search does not by itself select a successful training recipe.
  A new learning run would introduce more uncertainty than this direct test.
- Extra worlds, stronger simulated continuations, fewer candidates and a
  late-only policy would each change the player. None belongs in this test.

## Fixed players and fresh match

Candidate: reuse155's exact candidate construction, concrete representatives,
ordering,32 observer-legal worlds shared across candidates, terminal WDL and
baseline-first ties on every own turn. Continue exact greedy035 for both players
inside all rollouts. Keep one-through-four candidate states, own private cards,
unknown opposing blind reserves and uniform legal-pool sampling. No changes to
policy weights, value heads, search budget or continuation opponent model.

Comparator: preserve exact serial084, backed by the same035 checkpoint, including
three determinizations, four root successors, four opponent successors and four
stochastic035 continuations per leaf:192 continuations per completed search.
Keep epsilon0.1, temperature0.5, top-k3, score-margin scale20, its canonical
selection/ties, original structural fallbacks and stateful decision accounting.
Create a fresh084 bot for each game. Its fallbacks are part of the frozen player;
do not replace them with155 logic or treat them as new policy errors.

Only randomness namespaces and necessary observation/logging adapters are new.
Before any preflight outcome, freeze the environment panel, independent player
streams, per-decision world/continuation seeds and bootstrap seed. Do not reuse
155 or any earlier arena, probe or protected position. Starting decks/nobles
are paired, engine starting-player default remains0, and each seat game has
separate search streams. Pass084 an independent logical random-stream identifier
through its historical scheduling interface; no policy may consult the actual
environment seed, hidden deck order or unknowable reserved-card identity.

Instrumented084 must match the original serial player for fixed states and
randomness before any main game. Adapters must not silently change actions,
branch support, sample ordering, utility, fallbacks or termination handling.
Freeze exact source/checkpoint identities through explicit file inventories;
do not modify frozen155/084 or expand149's historical dynamic manifest.

## Endpoints and fixed decision

Primary endpoint: candidate score share, win1/draw0.5/loss0. Bootstrap exactly
50,000 resamples of the200 complete starting-seed pairs using a new frozen seed
and independently reproduced arithmetic. Do not bootstrap moves or rollouts as
independent games. No outcome-driven extension, optional stopping or pooling
with155's035 comparison.

The proposed success rule preserves the research-champion comparison standard:

1. Candidate score share>=55% and paired95% CI lower bound strictly>50%.
2. Mean final-score margin>0 and candidate share>=50% from each seat.
3. All prospective integrity checks and complete independent replay pass.

The margin and seat conditions are explicit historical safeguards in addition
to the primary score-share endpoint. Report every gate and the entire interval;
a failed gate does not necessarily establish inferiority or equivalence.400
games are a bounded first direct comparison, not a guarantee of high power
for a small advantage. Both outcomes stop after audit and interpretation.
No automatic promotion, training, parameter change or successor is included.

Record both players' W/D/L, margins, turns, decision wall/CPU mean/p95/max,
candidate selection/support, private-reserve coverage, phase/seat summaries,
084 completed searches/fallbacks, and every continuation/termination count.
This is a comparison of the two fixed systems with their existing budgets;
it does not establish performance at equal compute or superhuman strength.

## Integrity and preparation

Full repository tests before freeze. Then eight off-panel preflight games
(four seat pairs) on eight CPU game workers, one Torch thread each, with full
replay and fixed-budget timing. Inspect timing/integrity, not preflight score.
No nested process pools or new NumPy BLAS/Torch combinations.

Outer games and155 inner paths retain155's valid ordinary/equal-turn and
forced-two-PASS endings. Fourfold repetition,60 no-progress turns, turn400
cutoff, missing/illegal actions and exceptions are failures, never replacements
or favorable outcomes.

084's internal simulation contract is different and must remain explicit:
preserve its existing loop-safe rollout behavior and bounded loop adjudication,
which helped define its original policy. Require every completed search's192
continuations to resolve, zero hard caps and at most0.1% loop adjudications,
as in084/134. Report those separately from155's valid forced-PASS paths; do
not relabel them all as natural terminal games. Its known structural fallback
reasons are logged, while unexpected errors fail the experiment. This preserves
the opponent rather than introducing a new084 variant through the auditor.

Check legally equivalent hidden allocations/orderings with fixed player streams,
including both sides' already-held private cards and exhausted decks. For084,
compare instrumented decisions to the untouched reference throughout preflight;
any incompatibility blocks main and requires review, not silent policy repair.

Save complete outer trajectories and both players' root evidence: states,
legally available scores/actions/keys, RNG identity, sampled-world hashes,
every simulated branch's trace/final hash/outcome and selection reconstruction.
Independently replay all400 games, every155 root/path and every084 decision/
continuation, including084 fallback decisions. Independently rebuild worlds,
candidate/response sets, payoff summaries and final statistics. Shared engine,
scorer and any unavoidable shared resolver primitive must be explicitly stated;
this is not an independent implementation of Splendor rules.

Immutable artifacts and an exclusive lock. Outcome-blind stage/count/time
telemetry. Full audit even after failed strength gates. Only an explicitly
resumed, externally interrupted, identity-matching run may reuse completed
records; recorded validity failures require review and are not retried.

## Expected cost

155 took13.54 hours including preflight and full replay on eight workers.084's
historical400-game confirmation took3.45 hours on ten workers, without155's
full audit protocol. Scaling that084 wall time to eight workers and adding an
equal-cost replay gives about8.62 extra hours, or **roughly22 hours total**.

Plan for **about20-30 hours**, subject to the new full-game preflight. This is
an extrapolation across different opponents, positions, runs and audit code;
game length, fallbacks and contention can change it substantially. It is not
a time ceiling. Do not alter game count or budgets to fit the estimate.

Typical main work is roughly155's1.56 million paths plus about2.2 million084
paths, with all paths replayed. The formal400-turn outer ceiling permits up to
25.6 million main inner paths across both players before counting replay or
preflight, far above typical lengths. Resolve operational blockers in preflight
before the protected match rather than improvising a cheaper experiment.

## Evidence and status

- [155 result](C:/Users/leoba/Desktop/splendor_ground0/docs/engineering/rollout_selector_gameplay_155_result.md)
- [084 protocol](C:/Users/leoba/Desktop/splendor_ground0/docs/experiments/resolver_research_champion_confirmation_084.md)
- [084 result](C:/Users/leoba/Desktop/splendor_ground0/outputs/resolver_research_champion_confirmation_084/result.json)
- [Programme review](C:/Users/leoba/Desktop/splendor_ground0/docs/strategy/independent_superhuman_review_after_134_137.md)

Verified084 result SHA:
`1ed98024072b92baf3c30bc896b9dae2484f7d80b727959307dfd83bbfe2cd0f`.
155 binding:
`b287ea26424623b3825e49404467329162fcefb3e114b0064c49bc11b867e8a0`.
155 full-audit SHA:
`c4e943e32678501e4952233db8a480155182104e1c42343bc87d5198d3469f7d`.

Recommendation only.155 stays closed, champions084/035 unchanged. This turn
read existing evidence/code and primary evaluation context, wrote this proposal
and updated continuity. No156 implementation, simulation, testing or launch.
