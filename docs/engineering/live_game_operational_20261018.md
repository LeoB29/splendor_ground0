# Live-game adapter validation and six operational games

This prospective protocol implements the owner's authorization to validate a
separately versioned live-game adapter, then run three operational seed pairs:
six games, one pair for each fixed proposal model, at 120 seconds per complete
move. Passing validation authorizes the prescribed games without another
permission step. It does not authorize the remaining nine operational pairs,
a strength screen, training, larger-data collection, or edits to frozen files.

The operational results remain excluded from strength analysis. Matchup totals
stay closed throughout progress reporting and the final operational report.
The referee and verifier must still compute and check every individual winner,
score and forfeit internally; closed reporting does not weaken verification.

## Entrants and preserved identities

Use the corrected four-candidate, 128-world planner with all three fixed fits
161001, 161002 and 161003, unchanged raw035 continuations, ordered ties and
serial numerical execution. Keep the corrected proposal, world construction,
continuation, termination and reduction primitives. Change only the explicitly
versioned live context, evidence destination and corresponding validation.
Do not monkeypatch historical modules, reuse old `main` labels for new games,
substitute breadth, reduce worlds, return partial searches or select a fit by
its results.

The opponent is the authenticated profile
`entity_onehot_puct6400_canonical_atomic_prereveal_cpu_120s_v1`, SHA256
`6f74c3b6d77707c42619801d1b1bef3fbf203d1ee6ff548afbe935169c257599`.
Reuse its unchanged worker, model, native binary and atomic pre-reveal adapter.
Retain depth 64, world pool 3, scalar host inference and all other frozen search
settings, including the 108-second internal fail-closed allowance. Do not
rebuild the binary or repeat native qualification when its authenticated
dependencies are unchanged. Record actual work and declared heuristic branches;
6,400 iterations do not mean 6,400 independently sampled worlds.

Both entrants use `splendor_base_qualified_rules_20261018_v1`, catalogue SHA256
`db4776918e612eee5a6f2617719c85ec5f413d592bbab0a64058babc83a70312`.
Bind the prior accepted local and opponent qualifications, every preserved
input, all new code/tests, this protocol, profiles, fixture/variant inventory,
seed-collision receipt and schedule in a new freeze before timed validation.
Recheck their sizes and hashes at final acceptance. Preserve all earlier
results, failed attempts, checkpoints and preparations.

## Live context and randomness

Freeze these literal domains:

| Purpose | Value |
|---|---|
| Live namespace | `splendor_corrected_live_operational_20261018_v1` |
| Public search stream tag | `splendor_live_operational_search_v1` |
| Parent-only referee seed tag | `splendor_live_operational_referee_v1` |
| Audit domain | `splendor_live_operational_world_audit_v1` |
| Audit seed | `202610186401` |

Use canonical JSON with sorted object keys, compact separators, finite values
only and UTF-8 encoding for every hash input; SHA256 outputs are lowercase
hexadecimal. Context integers must be exact integers, not booleans. Reject
unknown fields, invalid namespace, mismatched fit/seat/turn, malformed public
stream, legacy-context misuse and context/observation disagreement.

An operational context contains mode `operational`, the live namespace,
pair, game, turn, fit, stream and identity. Game is 0 through 5, pair is
`game // 2`, fit is `[161001,161002,161003][pair]`, and identity is
`[LIVE_NAMESPACE, game, turn]`. Turn must equal the lawful observer's public
turn index and lie in 0 through 399. The public stream is
`SHA256(canonical([SEARCH_TAG, LIVE_NAMESPACE, game]))`. It depends on public
experiment identity, not the true initial seed, hidden state, audit sampling
or an outcome. Retain the previously validated belief-key/world-seed
derivation using this newly identified stream.

Derive each initial referee seed as the full integer represented by
`SHA256(canonical([SEED_TAG, LIVE_NAMESPACE, pair]))`. Keep the initial seed,
deck order and noble allocation in parent-only artifacts. The two games in
each pair use exactly the same initial state, with entrant seats swapped and
fresh player processes. Different game identities provide distinct local
public search streams. SplendoRust retains its observer-hash-only seed rule.

Before freezing, compare all three new referee seeds with the two completed
external screens and the discoverable historical experiment seed manifests,
registries and schedules. Freeze the searched paths, exact compared seeds,
source hashes, size/search exclusions and collision result. Do not claim
universal noncollision from a partial inventory. A collision blocks this
prepared schedule; changing its domain requires an explicitly revised
prospective preparation, not silent resampling during execution. Neither
referee seeds nor the audit seed enter a player's decision inputs.

## Twelve-request adapter validation

Use the three already selected corrected fixtures below. Do not select new
positions by timing, scores, actions or outcomes.

| Prior fixture index | Root | Fit | New operational game context |
|---:|---|---:|---:|
| 0 | `0204:010` | 161001 | 1 |
| 6 | `0250:000` | 161002 | 2 |
| 12 | `0292:002` | 161003 | 5 |

All three historical actors are seat 0. These game contexts follow the seat
assignment below and therefore also assign the local player to seat 0.
Their observations are saved counterfactual positions, not new live-game
states or actual progress in the six-game schedule.

For each fixture in table order, run its legacy-transfer original, legacy-
transfer hidden variant, new-operational-context original, and new-operational-
context hidden variant, giving 12 timed local requests. Legacy-transfer context
contains only mode `legacy_transfer`, fixture index, fit, stream and identity;
fit/stream/identity must equal the frozen prior observation, and the lawful
observation must equal that source. Retain the old random streams only for
this named transfer check. The new-context cases use the new public stream
and the fixture's actual public turn.

Construct hidden variants using the predetermined deck-order/opponent-hidden-
reserve permutation already qualified for these fixtures. Require actual
hidden state to change and the lawful observer to remain identical. Both
members of each pair must produce exactly equal scientific decisions and
compressed trace contents. For each legacy-transfer response, also require
exact equality with the previously accepted decision after removing only the
declared record profile-version label; actions, candidates, schedules, scores,
continuation outcomes and choices must match, and the compressed trace hash
must match. Wrapper/output paths and observer-envelope identity are checked
against their new contract, not silently stripped from scientific records.
New streams intentionally change samples; no old action equality is required
for the new-context cases.

Every completed validation response receives a separate exhaustive 128-world
audit covering every offered candidate, raw035 continuation choice and
dictionary-ledger transition, before starting the next validation request.
The audit reconstructs proposal scores, concrete representatives, candidate
order, all world seeds/states, outcomes, aggregates and selected action.
Give each audit a separate 600-second request allowance and 60-second
initialization cap under the same shared study deadline. Audit time is not
player thinking time. No timeout, incomplete response, privacy difference,
transfer mismatch or failed audit is accepted. Any validation failure blocks
the games, preserves the attempt and stops this execution; no retry,
replacement or scope reduction is allowed.

## Frozen six-game schedule

The game identities and entrant assignments are:

| Execution order | Game | Pair | Local fit | Local seat | Opponent seat |
|---:|---:|---:|---:|---:|---:|
| 1 | 0 | 0 | 161001 | 1 | 0 |
| 2 | 2 | 1 | 161002 | 0 | 1 |
| 3 | 4 | 2 | 161003 | 1 | 0 |
| 4 | 1 | 0 | 161001 | 0 | 1 |
| 5 | 3 | 1 | 161002 | 1 | 0 |
| 6 | 5 | 2 | 161003 | 0 | 1 |

This is execution order `[0,2,4,1,3,5]`, with
`opponent_seat = (pair + game % 2) % 2`. The fit is fixed within each pair.
Never reorder or replace a scheduled game based on an observed result or
duration. All six belong to the first three operational pairs of the broader
12-pair roadmap; they are not strength observations and authorize no automatic
continuation to the remaining nine pairs.

Use fresh initialization for each player decision, keeping the already
qualified no-cache/no-pondering behavior. Both players have the same separate
60-second initialization limit and externally enforced 120-second complete-
move allowance on the same assigned performance-core logical processor,
normal priority and one numerical thread. The parent stays model-free. Include
input serialization, transport, inference, search and complete output receipt
inside the move clock; for the local player include full compressed trace
writing and flushing. Use the trusted receipt timestamp, not child timing or
later validation time. No auxiliary payment/return/noble clock is added.

Only corrected natural terminal conditions determine an ordinary winner or
draw. Search continuation cutoffs are not referee rules. A nonterminal game
at 400 plies is an unresolved resource hold, never an invented draw. The frozen
clock statuses `timeout`, `worker_exit`, `malformed`, `wrong_request`, `illegal`,
`unsolicited` and `write_error`, plus the opponent's recognized internal budget
exhaustion response, are recorded player forfeits with no replacement; continue
the remaining games. Preserve the original status and its receipt.

Treat `worker_error` conservatively as a hard adapter hold. Any initialization
failure, timely result-wrapper/work/observer mismatch, unsupported/prerequisite
mapping, referee inconsistency, audit mismatch, authentication failure or
resource boundary stops execution. A timely action or scientific payload that
fails parent validation is a hard hold, not silently reclassified as a clock
forfeit. Keep detailed failure traces in preserved artifacts, while user-facing
progress gives failure categories only. Never turn an unresolved scientific
or adapter defect into a scored game.

## Every-game checks and sampled continuation replay

Apply `docs/engineering/verification_policy.md`. Independently check every
actual move, legal support, observer restriction, dictionary successor,
initial-state linkage, terminal result/forfeit, seat/model assignment and
schedule. Internally verify individual winners and scores without publishing
a matchup aggregate. At every completed local decision, reconstruct proposals,
legal input, tie order, all 128 world identities, every continuation record's
structural validity and completeness, aggregates and selected action. Retain
all complete compressed traces for the later outcome-blind stress selection.

For each completed live-game local root with identity `I`, select the eight smallest
`(hash, world_index)` values among indices 0 through 127, where hash is
`SHA256(canonical([AUDIT_DOMAIN, 'world', AUDIT_SEED, I, world_index]))`.
Replay every offered candidate in each selected world. The sampling unit is
a whole shared world, not an independently selected candidate or the chosen
action. The audit seed never enters player RNG or selection.

Also select the minimum `(hash, game, ply)` root in each nonempty cell of
fit Ã— game phase Ã— acting seat Ã— private-reserve category, with hash
`SHA256(canonical([AUDIT_DOMAIN, 'stress', AUDIT_SEED, I]))`. Phases are
turns0â€“19,20â€“39 and40+; privacy categories are neither, own only, opponent
only and both, derived from lawful reserve provenance/counts. Add one minimum-
priority root per fit and offered candidate support 1, 2 or 3 when present.
Use the same root-priority hash with numeric game and ply tie order; deduplicate all
overlapping selections. With three fits this is at most 72 ordinary cells
plus 9 short-support cells, not a commitment to invent missing cells.
Replay all 128 worlds and all candidates for each selected stress root.

Materialize and save the complete sample/stress manifest, eligible cell counts,
missing cells, selected identities and hashes before continuation replay.
Selection depends only on frozen identities and lawful metadata, never
outcomes, times, values, audit results or which candidate won. Do not replace
a slow or failing selected root. Report unique audited worlds/continuations
versus recorded totals, including sample/stress overlap. The ordinary 8/128
sample does not imply an audit 16 times faster than exhaustive verification.

For SplendoRust, check every delivered legal action, observer/profile binding,
phase mapping and exact reported fixed work using the unchanged qualified
diagnostics. Preserve its existing native-source/model parity trust boundary:
this does not reexecute every internal Rust search edge, PUCT backup or neural
inference. The new local verifier likewise shares frozen model/feature kernels
and corrected legal/serialization dependencies. Distinguish dictionary
transition independence from a wholly independent implementation.

## Finite execution envelope and failure preservation

The single serial execution has a 12-hour absolute ceiling, 43,200 seconds,
starting before the first validation initialization and including all
validation, games, audits, cleanup, source checks and final analysis. Reserve
four hours for post-game continuation replay and finalization; do not start
or continue game collection when that reserve would be consumed. Preserve
600 seconds for final checks and receipts. Before each validation case require
at least `60 + 120 + 60 + 600 + 14,400 = 15,240` seconds remaining, covering
its initialization, decision, audit initialization, full audit and post-game
reserve. Validation workers use the global deadline minus 600 seconds.

Before each live-game move require at least
`60 + 120 + 14,400 = 14,580` seconds remaining, and constrain its worker to
the global deadline minus 14,400 seconds. Before each audit require at least
`60 + 600 + 600 = 1,260` seconds remaining and retain the final 600-second
reserve. These are admission checks, not promises that every later case fits.
Apply live deadline/resource checks during initialization, requests, replay
and cleanup. The ceiling may yield an incomplete operational study; it does
not authorize extra time.

New evidence is bounded to 2 GiB, with at least 5 GiB free disk and at most 2 GiB
combined live worker/descendant RSS. Stop ordinary work at the evidence cap
minus 32 MiB, preserving that space for in-flight and final receipts. Before
worker launch require 1 GiB
available physical memory and 3 GiB commit headroom. Authenticate permitted
Python/Rust executables and OS helpers, pin them to the assigned core, enforce
the existing descendant/affinity rules and terminate the entire owned job on
failure. Limit complete response messages to 8 MiB, player requests to 1 MiB,
and audit requests to 16 MiB; the local/audit worker reader also accepts at
most 16 MiB. Limit each stdout, stderr and event log to 2 MiB. Include the
complete result plus observer and trace reference within the audit request
limit, so evidence accepted from a player can be delivered to the verifier.
Check disk/evidence/time live, write
artifacts exclusively, and verify closed process identities at finalization.

The cost rationale is a scenario, not a forecast: historical 51.933 plies/game
combined with fixture means 51.428 seconds local and 30.102 seconds external
implies about 3.53 hours of thinking for six games. At 120 seconds for every
move, the same historical lengths imply 10.39 hours before any audit. The
earlier 26 local decisions averaged 2.034 MiB per receipt plus compressed trace,
suggesting 317 MiB for the historical-length six-game scenario, before logs,
audits, game records, validation and fixed inventories. Changed game phases
or search paths can increase either cost. Twelve hours and 2 GiB provide an
explicit operational allowance, not a completion guarantee; never discard
evidence or weaken verification to stay within it.

On an unexplained mismatch, preserve partial/failure evidence, stop acceptance,
and diagnose only within the remaining frozen allowance. Any expansion,
behavior-changing repair, rerun or changed sample needs a separately reviewed
prospective plan; no automatic escalation launches additional games. A
resource-stopped game remains unresolved. Finished forfeits stay in the
operational record and are not removed to improve apparent reliability.

## Closed operational reporting and acceptance

Progress may report stage, attempted/completed games, fit coverage, decisions,
timing failures, legality/privacy/replay status, resource use, audit coverage
and remaining operational work. Do not publish win/loss/draw totals, per-fit
matchup scores, pair scores, a win rate, confidence interval or a superiority
claim in progress or the final operational report. Store necessary raw results
for authenticatable internal replay; they are closed analytical outcomes, not
uncomputed or deleted evidence.

Final acceptance requires all prescribed checks actually completed and all
bound inputs unchanged. Separately distinguish completed operational evidence,
timing reliability and readiness to continue: a validly recorded forfeit can
show operational weakness without being a referee defect. Report every
failure and unresolved game. A finite successful pilot does not guarantee
future deadlines or establish playing strength.

Any decision about the remaining nine operational pairs uses correctness,
privacy, timing, storage and measured cost only, while matchup totals remain
closed. The later fixed strength sample is selected prospectively before
strength outcomes are opened. All 12 roadmap operational pairs remain
excluded from that sample. This execution ends after these authorized six
games and their prescribed checks; the larger-data study stays deferred.
