# Experiment 155: repeated terminal-rollout selection in complete games

2026-09-10. Owner approved `docs/strategy/next_after_154.md` with "ok do it".
This isolated implementation preserves every dirty/frozen predecessor. No new
training, promotion, 084 comparison, policy variant or automatic successor.

## Frozen scientific design

400 games, 200 fresh initial seeds, two candidate seats per seed. Engine start
player stays 0; decks/nobles are identical within each pair. The candidate uses
exact greedy035, the highest-ranked distinct canonical action, then two uniformly
sampled remaining canonical groups. Preserve the 149/154 representative actions
and order; retain supports one through four and baseline-first ties.

At every candidate turn, 32 independent observer-legal worlds are shared across
all actual candidates. Apply each root action then exact greedy035 for both
players until terminal WDL (-1/0/+1). Maximize mean WDL, ties to original index.
No recursive selector in continuations, 128-world evaluation, phase filter,
extra world/candidate, learner or fallback. Comparator is exact original035.

The live random schedule replaces154's offline source index. Environment seeds
are 1550000000+pair for main,1559000000+pair for preflight. The bot never receives
that seed. Independent search streams are SHA256 of canonical JSON
`["155-independent-search-stream-v1", stage, game_index]`. Each legal information
key is SHA256 of `canonical_public_belief_key(state, acting_player)`. Candidate
nonce hashes `["155-candidate-v1", stream, information_hash]`; unchanged149
candidate logic receives split `155-live-v1` and this nonce. World seed strings
are `155-world-v1|stream|information_hash|NN`, NN=00..31. New tagged strings keep
world streams separate from historical integer schedules, candidate RNG and
environment randomness. Main/probe streams and environment panels are disjoint.
Both seat games have different search streams. No hidden environment order,
unknown opposing card identity, source seed or outcome enters bot randomness.

Own private reserve identities stay known; opposing blind reserve tier is known
but identity is sampled with unseen deck cards. The uniform legal unknown pool
does not model strategic posterior inference from opponent action history.
All legal blind reservations and already-held private reserves are retained.

Primary endpoint: win1/draw0.5/loss0 score share. Exactly50,000 NumPy PCG64
bootstrap resamples of all200 complete pairs, seed1553000000, generated as50
chunks of1000x200 indices. Mean the two seat scores within each pair, then mean
the sampled pair scores. Linear2.5/97.5 percentiles. Pass requires score>=.55,
lowerCI>.50 and full integrity. No precision gate, outcome-driven extension or
policy tuning. An interval spanning55% after a failed gate remains ambiguous
at that practical target; do not interpret it as equivalence.

## Terminal and implementation contracts

Reuse frozen149 termination: ordinary game completion/equal-turn endings;
two consecutive sole-legal-PASS turns adjudicated by score, then fewest purchased
cards. Fourfold repetition,60 no-progress turns, global turn>=400, missing or
illegal actions, exceptions and other endings are preserved failures. No game
or world replacement, skipping, favorable assignment or automatic retry.
These checks apply to outer games and each independent inner continuation.

Eight game processes, each with one Torch thread and deterministic algorithms.
No nested pools or NumPy BLAS. Upper bound400*200*4*32=10,240,000 main paths;
typical64-turn games imply1,638,400 paths. Full replay doubles this work.

All five new155 files and explicit154 parent artifacts/code form the manifest;
parent154 verifies its explicit frozen dependency chain. Never rerun149's
dynamic identity enumeration. Full-repository tests and their current identity,
exit code and log hash are required before freeze or preflight.

Immutable schedule, freeze, launch, root, complete-game and replay records.
Root records include full reconstructible snapshot, legal scores, candidate
actions/keys/order, legal information RNG context, every world seed/state hash,
every branch trace/final-state hash and WDL, choice and measured time. Outer
records contain every action, state hash, linked root hash and complete final
state. Hashes are deterministic trace hashes of all actions, replay reconstructs
all intermediate states. Failure records preserve available evidence.

Exclusive kernel lock covers the pipeline. `start` freezes/preflights then runs
main and full audit automatically under the owner's authorization. `prepare`
stops after preflight; `run` requires passed preflight. Only explicit `resume`
may recover an external interruption with identical identity. It reuses sealed
roots/games and hash-bound complete-game replay certificates. Any recorded
failure blocks automatic resume; in-flight incomplete records cannot be used
as complete. No statistical pass/fail affects replay completion.

## Tests, preflight and independent audit

Crafted tests cover both actors' known/unknown blind reserves, hidden deck
reversal and unknown opponent-card reassignment, own reserved purchases,
terminal root actions, equal turns, nobles, payment/return variants and exhausted
decks. Synthetic boundary fixtures target legal transitions; they are not drawn
as evaluation positions. Catalog-complete fixtures test independent worlds.
Full private-root tests use all32 worlds and compare every continuation.
Statistical tests use complete pairs, exact ties and each failed gate. Failure
preservation and unconditional audit-on-statistical-failure are tested.

Eight off-panel preflight games (four complete seat pairs) run at the exact main
budget on all eight workers after freeze. All games, roots and continuations
receive full independent replay. Only timing/integrity are exposed to operator;
preflight scores are neither summarized nor used to choose settings. An
operational failure stops before main; no automatic workaround changes design.

The independent verifier separately reconstructs the live schedule, full outer
trajectories, exact035 argmax actions, candidate sets and choices. It uses frozen
154's catalog-minus-legally-known world reconstruction and frozen149's separate
action-composition/greedy branch replay, sharing the engine and scorer. Each
root also verifies unchanged scores/candidates/randomness/worlds under hidden
order reversal and opposing-private-card reassignment. Every continuation is
replayed; no sampled-root audit. Independent pair bootstrap uses occurrence
counts and independent sorted linear quantiles; operational counts/latencies
are separately recomputed. This is not an independent game-rules implementation.

Report W/D/L, entireCI, seats, margins, turns, support, change fraction, own and
opponent private-root coverage, phase/seat/privacy descriptive summaries,
deadlocks, all continuation/move counts and mean/p95/max/sum CPU/wall times.
Partial scores/margins/root gains stay uninspected during collection. Telemetry
exposes stage, completed games, current turn counts, elapsed time and failures.

## Cost and interpretation

Prior154 extrapolation:16-24 hours including complete replay; no hard duration
guarantee. Preflight projects its measured collection+audit wall time by50.
Eight games can misestimate later game lengths/compute contention. Neither
400 games nor32 worlds may be reduced to fit this estimate.

This tests complete-system gameplay against greedy035 with additional compute.
It establishes neither equal-time efficiency, superiority to084, learnability,
an improving self-play loop nor superhuman performance. Both outcomes stop
after immutable result, full audit and final report. Champions084/035 unchanged.
