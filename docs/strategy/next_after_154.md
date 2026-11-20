# Recommendation after 154: validate the rollout selector in complete games

2026-09-10. Recommendation only. The owner asked what to do next; this is not
authorization to implement or launch 155. No new game, rollout, inference,
training, test runner or arena was run for this recommendation. Preserve the
intentionally dirty worktree and every frozen predecessor.

## Decision

Recommend one experiment, provisionally 155: **does using the fixed 32-world
rollout selector on every turn produce a stronger complete-game player than
raw 035?** Run 400 fresh games from 200 starting seeds, with the candidate playing
each seat once per seed. The comparator remains exact greedy 035.

154 established useful conditional move improvements: mean WDL gain
+0.048868815, 95% CI [+0.036814372,+0.061350759]. It changed one move and followed
greedy 035 thereafter. Repeated selection changes subsequent decisions, the
positions reached and the opponent's responses. Those changes are precisely
what the full-game test needs to measure.

The broader method has precedent in Monte-Carlo policy improvement: simulate
a base policy after candidate actions and select by estimated return.
[Tesauro and Galperin's original research](https://research.ibm.com/publications/on-line-policy-improvement-using-monte-carlo-search)
supports that motivation; it is not a finite-sample guarantee for this Splendor
player, candidate set, private information or complete-game performance.

## Why this is the highest-value next question

- 154 supplies a fresh positive signal for this particular improvement operator.
  Its independent gain was only 34.4% of the apparent discovery gain, so sampled
  maxima still contain substantial selection optimism. More training on those
  maxima is premature as the default next step.
- 149-153 failed to recover material move benefits with the tested learners.
  A positive simulator result does not reverse those results or identify a new
  learning recipe. Establish this deployed operator's gameplay value first.
- Earlier evidence cautions against substituting local or predictive gains for
  wins. Expanded 086's conditional promise preceded 134's clear loss to 084.
  144's public-value planner scored 52.75% versus 035, 95% CI [48.5%,57.0%], leaving
  gameplay benefit unestablished despite earlier predictive improvements.
- 154's larger observed late-game gain does not justify selecting a late-only
  variant from the same results. Test the fixed selector throughout the game.

Raw 035 is the right first comparator because it is both the baseline move
policy and the continuation policy behind 154. This is an initial gameplay
competence test, not a claim to replace 084. A later direct 084 comparison and
cost/strength assessment would still be necessary; historical win rates against
035 cannot rank the new player against 084.

## Proposed fixed player

On every candidate turn, retain 154's candidate construction: exact greedy 035
concrete action, highest-ranked distinct canonical alternative, and two uniformly
sampled remaining canonical groups. Preserve original representative actions,
ordering, fewer-candidate states and baseline-first exact ties.

For each real candidate, use 32 observer-legal world draws shared across the
actions, apply the root action, and continue exact greedy 035 for both players
to valid terminal WDL. Select the highest mean. Repeat this procedure when the
candidate next acts in the real game. Simulated continuations remain greedy 035;
there is no recursive rollout-selector call inside a rollout.

The real game is the independent evaluation. Do not add 154's 128-world comparison
block at every move. No learned evaluator, extra candidate, confidence filter,
phase filter, additional worlds or search-budget sweep is included.

154's source-index random schedule needs a deployment schedule. Before any new
outcome, freeze independent environment, candidate-sampling, search-world,
preflight and bootstrap streams. Tie per-root sampling to the candidate's own
random stream and legal observer information/turn context. Neither candidate
construction nor root sampling may depend on the actual environment seed,
hidden deck order, unknown private-card identity or eventual game outcome.
The live scheduling adaptation is explicit; this is not byte-for-byte reuse of
154's offline source-index seed string.

## Cover the full legal game

154 excluded source snapshots with already-held private blind reserves. Full
games necessarily allow such positions. Retain all legal blind reservations,
including at the root, and handle both the player's known private cards and the
opponent's unknown reserved identities throughout gameplay and simulations.
Do not ban blind reserves, skip these turns or silently fall back to a different
policy when they occur. This broader population is an explicit deployment change.

Before the protected match, test crafted legal positions with own and opponent
blind reserves, exhausted decks, terminal root actions, payment/return variants,
nobles and equal-turn endings. With fixed search randomness, changing only the
actual hidden order or reassigning unknown opponent cards within the same legal
information set must leave root scores, candidates, sampled worlds and choices
unchanged. Known own private cards must remain available to the acting policy.

The existing observer-relative sampler draws from the legal unknown pool; it
does not infer a strategically optimal posterior from opponent history. Preserve
and state that assumption. Verify its implementation independently against the
catalog-minus-known-cards construction. This is not a new belief-learning study.

## Fresh arena, decision and interpretation

Freeze 400 games from 200 entirely fresh starting seeds, two seat-swapped games
per seed. Pair the initial decks and nobles; keep environment randomness
independent of search randomness. Use distinct frozen search streams for the
two games. Do not consume earlier protected panels or reuse 154's evaluation
positions as an arena or tuning set.

Primary endpoint: candidate score share, with wins=1, draws=0.5 and losses=0.
Use 50,000 bootstrap resamples of complete starting-seed pairs. Keep both games
from each pair together; moves and inner simulations are not independent match
replicates. Freeze the seed and exact arithmetic before execution.

Proposed pass requires:

- score share at least 55%;
- paired 95% confidence interval lower bound strictly above 50%;
- all operational, provenance, rules and replay checks passed.

400 games is a bounded first gameplay test, consistent with earlier arena
sizes, not a guarantee of high power for a small improvement. Report the entire
interval. A failed gate with an interval including 55% is unresolved at the
practical target; it does not establish equivalence. Do not add games or choose
an alternative policy after observing the result.

Report W/D/L, seat shares, score margins, turns, candidate counts, change rates,
private-reserve coverage, valid deadlocks, all continuation counts and mean/p 95/
maximum decision latency and CPU time. Phase/seat/privacy summaries are
descriptive, not policy-selection endpoints. Score margin is not a substitute
for the primary outcome.

This compares the complete selector, using more computation, against a fast
baseline. It is not an equal-time efficiency test. A pass would establish this
matchup's gameplay improvement, not superiority to 084, optimal play, learnable
labels, an improving self-play loop or superhuman performance. Both outcomes
stop after full audit; no promotion or automatic next stage is included.

## Integrity and reproducibility

Use the existing terminal rules, including valid forced-two-PASS score/card-count
adjudication. Fourfold repetition,60 no-progress turns, the 400-turn ceiling,
illegal actions, missing actions and other invalid endings must be recorded as
failures rather than excluded, replaced or converted to favorable outcomes.
The same rule applies to inner simulations and complete games.

Freeze isolated 155 implementation, protocol, tests, independent verifier and
explicit 154 parent identities. Do not edit 154 or expand 149's historical dynamic
identity. Require full repository tests and a separate outcome-blind preflight:
eight off-panel games (four seat pairs), using all eight workers, with complete
replay. Inspect correctness, resource use and timing, not matchup scores for
policy selection. Freeze the main match design before these probes.

Use eight CPU processes, one Torch thread per process. Parallelize games;
avoid nested worker pools and new NumPy BLAS/Torch combinations. Save each
complete game, every real move/root, candidate ordering, world seeds/hashes,
branch trace/final-state hashes, WDL outcomes and measured time.

Independently reconstruct all 400 full game trajectories and raw 035 actions,
every candidate selection and every simulated continuation. Rebuild worlds
from legally known cards, verify information invariance, and reproduce paired
statistics and gates separately. Finish the full audit even if statistical
gates fail. The auditor can share the frozen engine, with that limitation
disclosed; it is not an independent Splendor rules implementation.

No partial arena win rates, margins or root gains should be inspected during
collection. Monitor stage/count/timing/failure telemetry only. Immutable files
and an exclusive process lock prevent duplicate launches. An externally
interrupted run may resume only with identical frozen identities and preserved
completed games; a recorded validity failure requires review rather than an
automatic retry. No outcome permits changing settings or replacing difficult
games.

## Expected cost

At about 64 total turns per game (consistent with 144/148), the candidate makes
about 32 decisions/game: 12,800 roots and at most 1,638,400 rollout continuations
at four candidates and 32 worlds. The realized number changes with game lengths
and real candidate support. Full independent replay adds the same simulation
count, alongside real-game trajectory checks and off-panel preflight.

154's discovery collection used 14,623.94 worker CPU seconds for 768 roots; its
discovery replay used 12,971.42. Scaling both to 12,800 roots across eight workers
gives about 16 hours. Scaling the measured stage wall times gives about 17 hours.
Plan for an **overnight run, roughly 16-24 hours including full audit**, pending
the new outcome-blind preflight. This is a measured-data extrapolation, not a
runtime guarantee or a fixed time ceiling: repeated play, private-reserve
positions, game length, contention and artifact writing can change the cost.
154's small preflight had underestimated its own main runtime, so its 90-minute
probe estimate is not the basis for this projection.

The outer 400-turn failure ceiling permits at most 200 candidate roots/game;
the formal main simulation ceiling is 400*200*4*32=10,240,000 continuations before
outer termination checks, much larger than the typical-length estimate.
Do not silently reduce the 400 games or 32-world budget to fit a duration.
If preflight reveals an operational blocker, preserve it and revise the proposal
before any protected match; do not improvise a different main experiment.

## Evidence and status

- [154 result and full audit](C:/Users/leoba/Desktop/splendor_ground0/docs/engineering/rollout_selector_154_result.md)
- [144 gameplay result](C:/Users/leoba/Desktop/splendor_ground0/docs/engineering/public_value_search_144_result.md)
- [148 gameplay and timing result](C:/Users/leoba/Desktop/splendor_ground0/docs/engineering/search_teacher_headroom_148_result.md)
- [Programme review, including 134 and prior learning failures](C:/Users/leoba/Desktop/splendor_ground0/docs/strategy/independent_superhuman_review_after_134_137.md)
- [Legal-information representation](C:/Users/leoba/Desktop/splendor_ground0/src/splendor_ai/encoding/public_belief.py)
- [Existing observer-relative world sampler](C:/Users/leoba/Desktop/splendor_ground0/src/splendor_ai/bots/hidden_information.py)

154 remains closed; champions 084/035 are unchanged. This turn adds a proposal
and continuity updates only. No 155 implementation, output directory, simulation,
training, test execution, arena or experiment process was created.
