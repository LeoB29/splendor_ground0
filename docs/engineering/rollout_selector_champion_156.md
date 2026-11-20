# Experiment 156: unchanged155 versus the exact084 research champion

2026-09-11. Owner approved `docs/strategy/next_after_155.md` with "ok, do it".
This isolated experiment preserves the intentionally dirty worktree and every
frozen predecessor. The question is whether155's repeated terminal-rollout
selector transfers beyond the035 opponent used in its simulated futures.

## Fixed players

Candidate: direct reuse of155's move operator, without edits or global overrides.
At every own real turn, retain greedy035's concrete action, the highest-ranked
distinct canonical successor, and two uniformly sampled remaining canonical
groups. Keep the original representative rules, order, supports one through four,
32 observer-legal worlds shared across candidates, and exact greedy035 for both
players in every continuation. Maximize mean terminal WDL (-1/0/+1), with ties
to the earliest candidate. No recursive search in continuations, learned value
head, opponent-model substitution, extra worlds, phase gate or training.

Comparator: unchanged `FrozenPublicBeliefResolverBot` from the original050/084
implementation. Three worlds, four canonical root actions, four opponent actions,
four stochastic035 continuations per leaf:192 paths per completed search. Epsilon
0.1, temperature0.5, top-k3, score-margin scale20, continuation400 additional
turns, maximum64 loop-fallback triggers. Preserve canonical ordering, concrete
mapping, pooled minimizing response per opponent information set, root ties,
all structural fallbacks and stateful decision accounting. New bot each game;
its continuation worker count is exactly1. The outer pool contains eight games.

The original configuration validator also binds a ten-game-worker deployment.
156 checks every original config field after normalizing only `workers`,
`smoke_seed`, `belief_root_index_start` and `continuation_seed_start`; all other
fields must exactly equal the original configuration and satisfy its049 check.
These four approved changes affect outer deployment or random namespaces only.

Both players use035 checkpoint
`outputs/legal_action_scorer_035_legal_observation/legal_action_attention_scorer_best.pt`,
SHA256 `4422a82b593340eecd02af3a0941f24f0318cc7453e6a0739fd9f524539af2cd`.
155 binding `b287ea26424623b3825e49404467329162fcefb3e114b0064c49bc11b867e8a0`;
155 audit `c4e943e32678501e4952233db8a480155182104e1c42343bc87d5198d3469f7d`.
084 result SHA256
`1ed98024072b92baf3c30bc896b9dae2484f7d80b727959307dfd83bbfe2cd0f`.
The explicit parent chain verifies original source/checkpoint hashes, including
the084 resolver, its scorers, stochastic rollout and canonical-key primitives.
Do not invoke149's historical dynamic source enumeration.

## Panel and complete random schedule

400 fresh main games from200 initial seeds, candidate in each seat once per seed.
Eight off-panel preflight games from four separate initial seeds. Engine starting
player remains0; initial decks/nobles match exactly within each pair. Game index
`i`, pair `i//2`, candidate seat `i%2`, champion seat `1-i%2`. Main environment
seed1560000000+pair; probe seed1569000000+pair. No player receives that seed.

Candidate stream is SHA256 of canonical JSON
`["156-independent-search-stream-v1", stage, i]`, with stage `main` or `probe`.
Keep unchanged155's information hash: SHA256 of
`canonical_public_belief_key(state, acting_player)`. Candidate nonce hashes
`["155-candidate-v1", stream, information_hash]`; original candidate constructor
receives split `155-live-v1`. World seed strings are
`155-world-v1|stream|information_hash|NN`, NN=00..31. The155 tags preserve the
same operator while the new stream makes all156 randomness fresh.

084 receives logical stream index `i` through its historical `engine_seed`
interface, with `smoke_seed=0`; this index is independent of the actual engine
seed. For champion seat `s`, zero-based own decision `d`, world `w=0..2`,
continuation replicate `j=0..3`:

- Belief root index = B + i*1000 + s*400 + d, where B=1564000000 (main) or
  1569400000 (probe). Actual world seed=49000000+root_index*10003+w.
- Continuation seed = C + i*1000000000 + s*400000000 + d*1000000 + w*10000+j,
  where C=1560000000000000000 (main) or1569000000000000000 (probe).
- Preserve084's common random numbers across root/response alternatives: the
  continuation seed deliberately excludes the root and response indices.

With the400-turn outer limit each player has at most200 decisions. Namespaces
are disjoint across stages/games/decisions/worlds/replicates as applicable;
all integer continuation seeds fit signed64-bit. Initial environment seeds,
player random streams and bootstrap randomness remain separate. All schedules
are frozen before any preflight outcome; tests use synthetic boundary fixtures
from seed156, outside either panel.

Observer worlds preserve own private reserved identities and both sides' public
cards. Unknown opponent blind reservations retain their known tiers but sample
identities together with unseen decks from the uniform legal pool. No actual
hidden ordering, unknown identity or source seed enters either player's choice
or RNG. This is not a learned posterior over the opponent's strategy.

## Endpoint and fixed decision

Primary outcome: candidate win1/draw0.5/loss0 score share. Exactly50,000 NumPy
PCG64 complete-pair bootstrap resamples, seed1563000000;50 chunks of1000x200
integer indices. First average the two seat scores per pair, then mean the200
sampled pair scores. Linear2.5/97.5 percentiles. No move-level resampling,
pooling with155, optional stopping, additional seeds or outcome-driven tuning.

All statistical gates must pass: overall share>=0.55; paired95% interval lower
strictly>0.50; mean final-score margin>0; each seat's score share>=0.50. Full
integrity and replay are also required. Report all gates, full interval, both
players' W/D/L and margins, and seat results. A failed gate alone does not prove
inferiority or equivalence. This fixed400-game comparison can miss small gains.

## Distinct termination contracts

Outer games and every155 inner path retain the149/155 rules: ordinary completion
including equal-turn endings, or two consecutive sole-legal-PASS turns adjudicated
by score then fewest purchased cards. Fourfold repetition,60 no-progress turns,
global turn400 cutoff, missing/illegal actions and exceptions are preserved
failures. No replacement, skipping, favorable assignment or automatic retry.

084 internal paths retain their original loop-safe stochastic continuation.
After applying the forced opponent response, allow400 additional turns. On
fourfold repetition or60 idle turns, use the original canonical loop fallback:
best scored purchase if one exists, otherwise least-used canonical successor,
score then lowest action-index ties. Ordinary stochastic selection otherwise
uses frozen035 and the parameters above. At64 fallback triggers, preserve its
bounded score-margin loop adjudication. Require all192 paths per completed
search resolved, zero hard caps, and aggregate loop-adjudicated fraction<=0.001
separately in preflight and main. Report terminal and loop counts distinctly.

Known084 structural fallback reasons remain part of its policy:
`fewer_than_four_root_actions`, `root_mapping_changed`, `root_action_terminal`,
`fewer_than_four_opponent_actions`, `opponent_action_set_unstable`. These return
its original parent action, log the reason, and evaluate no continuation paths.
Other exceptions are failures. Do not apply155's forced-PASS semantics to084's
internal paths or repair the opponent's choices during the experiment.

## Records, execution and audit

Full repository tests must pass before freeze; record exact tested identity,
exit code and log SHA. Freeze six156 files: runner, root entry point,084
instrumentation helper, independent auditor, focused tests, and this protocol.
Hash the explicit155 dependency chain and084 provenance. Preparation wrappers
have their hashes recorded in test/process evidence. Save immutable manifest,
schedule, freeze and launch records; refuse mismatching identity throughout.

Eight CPU game workers, one Torch thread and deterministic algorithms each.
No nested pools or NumPy BLAS operations. Root records contain snapshots, all
legal scores, candidate/root support and representative actions, random context,
sampled world hashes, each branch's seed/trace hash/final hash/outcome, final
selection and CPU/wall timing.084 records also retain before/after telemetry,
structural fallbacks and complete information-set resolution payloads. Trace
hashes cover every simulated action; replay reconstructs all intermediate states.
Outer records contain all actual actions/state hashes, linked root hashes and
the final state. Available evidence is saved on exceptions.

The instrumentation only observes the existing serial084 implementation via
process-local wrappers and existing decision hooks. It restores original module
references even on failure. Preflight independently replays all eight games,
every155 path, and every084 decision/path; it additionally invokes the untouched
original084 bot at every084 preflight turn with the identical stream and requires
exact chosen actions and cumulative telemetry. Any mismatch blocks main.

Main audit independently reconstructs all400 outer games, candidate sets,
observer worlds,084 root/response trees, continuation RNG and stochastic
trajectories, canonical loop fallback, information-set minimization, selected
actions and cumulative counters. It verifies each root's information invariance
under hidden deck reversal and legal reassignment of unknown opposing reserves.
Catalog-complete synthetic tests also cover both actors' held private cards,
exhausted decks, terminal-root fallback, loop adjudication, hash-valid tampering,
crash reuse and audit execution after either statistical outcome. Existing155
tests retain supports1-4, noble/equal-turn/payment/return and termination cases.

The independent verifier shares the frozen engine, neural scorer, canonical
keys, serialization and progress predicates. It reuses149's separately written
candidate branch replay and154's independent world/candidate reconstruction.
084's scheduling, planning, stochastic loop, fallback and two-ply resolution
are separately implemented in156's auditor. Statistics use independent pair
occurrence counts and sorted linear quantiles. This is a full deterministic
reconstruction, not an independent Splendor rules implementation.

Exclusive kernel pipeline lock; immutable per-game/root/replay certificates.
`start` freezes, runs preflight and full replay, then only on integrity pass
runs the fixed main match and full audit. `prepare` stops after preflight;
`run` requires passed preflight. Only explicitly requested `resume` recovers an
external interruption under identical identity, reusing sealed records and
hash-bound complete-game replay certificates. Recorded failures block retry.
Both statistical outcomes receive the entire main audit before closure.

Outcome-blind operational telemetry exposes stage/count/turn/time/failure only;
do not inspect preflight scores or partial main strength. Report all selection
indices/support, changes, own/opponent private-root coverage, continuation/move
counts, endings, CPU/wall mean/p95/max/sum, and seat/phase descriptions. Common
phase summaries use155 bins0-19,20-39,40+; preserved084 internal telemetry keeps
its historical early0-19,mid20-49,late50+ bins. Selection indices remain saved
for later descriptive analysis and are not evidence of an ablation effect.

## Cost, stop and interpretation

Formal main path ceiling400*200*(4*32+3*4*4*4)=25,600,000 across both players,
before full replay. Typical total nearer3.8million paths, each fully audited.
Prior timing suggests roughly22 hours; plan20-30 hours, subject to preflight.
Preflight reports measured collection/audit wall times and their50x main
projection. The preflight audit includes an extra untouched084 reference pass
that main does not need, so this extrapolation includes extra work. Eight games,
game length and contention can otherwise change timing substantially. There is
no duration cap and no reduction in worlds/games to fit an estimate.

Write immutable observations/result, full-audit certificate, report and final
status, then stop. This compares two fixed systems at their existing budgets;
it does not establish equal-compute efficiency, learnability, a self-improving
training loop or superhuman play. No automatic promotion, training, budget
change, successor or change to champions084/035 is authorized by this run.
