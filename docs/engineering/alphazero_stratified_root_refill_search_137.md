# AlphaZero Stratified Root-Refill Search Pilot `137`

Frozen before state-manifest generation or any experimental search result on
2026-09-04.

## Purpose

Test whether information-legal, action-conditioned stratification of public
refill cards lets a 256-simulation AlphaZero root search reproduce an ordinary
1,024-simulation reference more accurately than the current 256-simulation
root-determinization search.

This is a search-mechanism test, not training and not gameplay. The preparation
stage may reconstruct and serialize protected states, implement the search,
run unit tests, and write a preflight manifest. It must not run any protected
state search until the owner separately authorizes launch.

The causal contrast is narrow. Both 256-simulation arms use the exact same
generation-11 checkpoint, PUCT parameters, legal actions, tree representation,
leaf evaluator, simulation count, and paired random seed. The treatment changes
only which information-legal card is placed first in the relevant root tier
deck after PUCT selects a visible buy/reserve action.

## Frozen inputs

Checkpoint:

- `outputs/alphazero_continued_scaling_104/checkpoints/generation_011.pt`
- SHA-256:
  `c97e7e06910639c0edf61fdd878fd665e390298790830e0e792061d9107354d0`

State source: all four exact-generation-11 experiment-114 sanitized self-play
shards. They are used only to reconstruct public states and selected actions;
winner, final scores, terminal value targets, and other outcomes are forbidden
selection inputs.

| shard | rows | bytes | SHA-256 |
|---:|---:|---:|---|
| 0 | 3,956 | 74,919,121 | `f22b29a3ce5dbfc2e18331dc10a9f30928fe8ee6cf93d65ad48b524a9835af8b` |
| 1 | 4,004 | 76,570,863 | `c3d333830f3d9e92557bcd762b5605580aa0af0b0f3b0d5fcddaaab77cde3e60` |
| 2 | 3,956 | 76,478,700 | `00e816d7c11c56e1dfc55a4c82198ffd6fceff7e57dfdf780239092b93ca00d8` |
| 3 | 4,000 | 76,509,765 | `7bd6b03227aa6e4e5ce1bb902adda6021d438b5e927d1da2b9f56793424dd95e` |

Experiment `134` is not an input. No `134` artifact or outcome may be read.

## Protected state schedule

Select 36 states: 12 each early, middle, and late, with six root-player-0 and
six root-player-1 states in each phase. The no-blind-reserve filter leaves only
23 eligible source games overall and only nine that reach the late/middle cells,
so require at least 20
distinct games, never reuse a game within the same phase/player cell, and use
at most three states from any source game globally.

Phase windows and target turns:

- early: turns `8--19`, target `14`;
- middle: turns `20--49`, target `34`;
- late: turns `50+`, target `58`.

A candidate state must:

1. reconstruct exactly from its source seed and stored selected legal actions;
2. have no blind deck-reserved card belonging to either player;
3. have at least one legal `BUY_VISIBLE` or `RESERVE_VISIBLE` action in a tier
   with at least eight remaining possible refill identities;
4. not be terminal or pending round end;
5. have exact stored and freshly reconstructed public entity encodings.

For each source game and phase/player cell, retain only its candidate closest
to the target turn. Solve the resulting deterministic capacitated assignment:
each phase/player cell demands six states, each game-to-cell edge has capacity
one, and each source game has capacity three. Lexicographically minimize (1)
second and then third uses of a source game, then (2) candidate rank by absolute
target-turn distance, source seed, turn index, and shard index. Implement this
as an integer min-cost flow with a diversity penalty larger than the sum of all
candidate-rank costs. Store the resulting schedule in phase/player order. This
selection uses no policy, value, visit, Q, score, winner, or terminal target.

The preparation-only preflight writes exact serialized snapshots and identities
to `outputs/alphazero_stratified_root_refill_search_137/state_manifest.json`.
The manifest SHA-256 becomes part of the launch identity and may not change
after preparation.

## Search arms

All searches use generation 11, real information-set mode, `c_puct=2.25`,
`max_depth=96`, no Dirichlet root noise, no temperature, and `self_play=False`.

For every state run two paired independent replicas of:

1. `ordinary_256`: current root determinization, 256 simulations;
2. `stratified_256`: action-conditioned root refill stratification, 256
   simulations;
3. `ordinary_1024_reference`: current root determinization, 1,024 simulations.

The two 256 arms use seed `137100000 + 100 * state_index + replica`. Reference
seed is `137200000 + 100 * state_index + replica`.

Total protected work if launched is 216 root searches and 110,592 simulations.
Use one process and one Torch thread while experiment `134` is active; the
owner should normally wait for `134` to finish before launch. Parallelism may
be separately reviewed after `134` completes.

## Stratified root-refill mechanism

The ordinary real-mode determinization remains the starting point for every
simulation. Only at search depth zero, after PUCT chooses an exact root action:

- if the action is not `BUY_VISIBLE` or `RESERVE_VISIBLE`, change nothing;
- if its tier deck is empty, change nothing;
- otherwise, use a schedule specific to that exact canonical root action;
- the schedule consists of independently seeded permutations of the sorted
  information-legal possible card identities, concatenated cycle by cycle;
- on the action's `k`th visit, place schedule item `k` first in the determinized
  tier deck, preserving a random order for all remaining hidden cards;
- then call the unmodified simulator transition.

Thus, for each exact refill action separately, realized root refill counts must
differ by at most one among possible identities. The schedule seed is a stable
SHA-256 derivation of the root search seed, canonical action key, and cycle, so
it cannot depend on Python hash randomization or hidden true deck order.

The protected states contain no blind reserves, so the possible set is exactly
the tier's hidden-deck identity set and every forced card is information-legal.
Deeper refills retain ordinary sampled deck order. The treatment adds no chance
outcome to the network input and does not expose a future card before the root
action is chosen.

Both arms record realized root refill identity counts by exact canonical action
for integrity and coverage measurement. Telemetry must not affect PUCT.

## Integrity gates

Preparation must establish:

- exact protocol, checkpoint, and replay hashes;
- exact 36-state phase/seat schedule with 36 distinct games;
- no blind reserves and at least one qualifying refill action per state;
- exact serialize/deserialize, public encoding, legal-action, and possible-card
  set round trips;
- production `AlphaZeroSearch` behavior is unchanged by its default transition
  hook;
- stratified schedules are deterministic, hidden-order invariant, legal, and
  balanced to a maximum identity-count difference of one;
- non-refill actions and all depth-greater-than-zero transitions are unchanged;
- both 256 arms account exactly 256 root visits and reference accounts 1,024;
- no optimizer, training, continuation, gameplay, checkpoint write, or `134`
  read occurs.

Any integrity failure makes the experiment invalid.

## Frozen outcome metrics

Create the reference distribution by averaging the two ordinary-1,024 visit
distributions on aligned exact root actions. A state is reference-stable when
the two reference replicas choose the same exact visit-argmax action. Require
at least 24 reference-stable states for a strength interpretation.

For each 256 arm report overall and by phase:

- exact and core-intent top-action agreement with the pooled reference;
- top-action agreement on reference-stable states;
- total-variation distance from the pooled reference visit distribution;
- reference-Q regret of the arm's chosen action;
- exact and core-intent agreement between the arm's two replicas;
- visit-distribution TV between replicas;
- realized refill identity coverage for all visited, priority (at least 5% of
  root visits), visit-winning, and selected refill edges;
- simulations, expansions, inference calls, maximum depth, and wall time.

Use a deterministic 50,000-sample source-game-cluster paired bootstrap with
seed `137300000` for treatment-minus-control changes in exact reference
agreement, core reference agreement, TV distance, Q regret, and replica
agreement. Repeated states from one source game always remain in one cluster.

## Frozen interpretation gates

First require the mechanism sanity gate:

- every stratified root refill is legal and balanced;
- stratified priority-edge mean realized coverage exceeds ordinary coverage by
  at least 10 percentage points;
- median stratified wall time is no more than 1.25 times ordinary wall time.

Define a material ranking gain when all hold:

- at least 24 reference-stable states;
- exact or core reference agreement improves by at least 10 percentage points,
  with its paired-bootstrap CI95 lower bound above zero;
- mean TV distance or mean reference-Q regret improves by at least 20%, with
  its paired-bootstrap CI95 upper bound below zero for treatment minus control;
- treatment replica agreement does not fall by more than two percentage points.

Decisions:

1. Integrity or mechanism sanity fails:
   `stratified_refill_mechanism_invalid_no_search_conclusion`.
2. Sanity and material ranking gain pass:
   `stratified_refill_improves_search_efficiency_design_training_pilot`.
3. Coverage improves but material ranking gain fails:
   `stratified_refill_coverage_not_ranking_bottleneck_keep_ordinary_search`.

Even a positive result authorizes only design of a separately frozen training
pilot. It does not authorize training, gameplay, installation, promotion, a
change to experiment `134`, or automatic continuation.
