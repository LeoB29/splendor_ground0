# Corrected-rules four-candidate planner qualification

18 October 2026. This prospective protocol covers the authorized corrected-rules
qualification of the four-candidate, 128-world planner across all three frozen
proposal fits, followed by a recommendation for a matched thinking-time allowance
against SplendoRust. It authorizes no games, new training, larger data collection,
external-model inference, or automatic expansion of this study.

## Player identity and immutable boundary

Retain proposal fits 161001, 161002 and 161003, the frozen raw035 continuation
checkpoint, the original candidate construction and ordered ties, all 128
worlds, the original public stream/RNG schedule and serial numerical backend.
Never choose a best fit, reduce work, accept an incomplete search, use a faster
backend, or substitute the historically stronger eight-candidate player.

The new profile uses `splendor_base_qualified_rules_20261018_v1`, catalogue
SHA256 `db4776918e612eee5a6f2617719c85ec5f413d592bbab0a64058babc83a70312`.
It incorporates the separately versioned six card-cost corrections and mandatory
noble visit after every action. A sole PASS that awards a noble resets the
forced-pass deadlock counter; only a sole PASS without a noble increments it.
This is a necessary, explicit consequence of the corrected rules. Other
termination ordering, repetition limits and no-progress limits remain fixed.

Implement explicit adapted orchestration in
`tools/corrected_planner_qualification_20261018/`; do not patch historical module
globals or overwrite frozen code, checkpoints, panels, outcomes or receipts.
Bind all new code, protocol, source inputs and models before launch, and verify
them again before acceptance. The original results remain results under their
original rules. Saved positions are labelled corrected-rules counterfactual
fixtures, not historical games replayed under an equivalent engine.

Every legality, grouping, forced-action and continuation transition must use the
corrected rule boundary. Audit reused function imports as well as call sites:
an imported historical `SplendorEnv`, catalogue dictionary, default argument or
cached afterstate can survive a caller's change of engine. In particular,
`encode_position` receives the complete corrected legal list explicitly;
determinization preserves supplied corrected Card objects; raw035 action
features and canonical successor/group keys must not reconstruct legacy faces
or use legacy transition semantics. Tests cover these transitive boundaries,
all six corrected cards, non-buy noble awards, noble-awarding PASS, terminal
rounds, ordered ties and short support.

## Fixed panel, schedule and observation boundary

Reuse the exact authenticated selection in
`outputs/local_player_qualification_20261018_recovery_01/selection.json`:
18 core positions, one per fit, acting seat and phase cell, plus its five
distinct metadata-selected edge positions. Their identities were selected
without outcomes or timing. Do not replace positions using this study's results.
The original root context and fit assignment remain fixed. Short support means
all available candidates, up to four; do not manufacture four distinct actions.

Freeze exactly 49 timed attempts in this order:

1. The 23 positions at an externally enforced one-second allowance.
2. The same 23 positions at an externally enforced 120-second allowance.
3. The first core position for each fit at 120 seconds, with its predetermined
   hidden-state permutation and unchanged lawful observation/RNG context.

Each attempt starts a fresh worker and initializes all three authenticated fits
and raw035 under a separate 60-second limit. The two declared allowances are
different scheduled cases; a successful 120-second result never replaces or
reclassifies a one-second timeout. No automatic retries or replacement cases.

Only public information, the acting player's private reserve faces, opponent
private-reserve counts/tiers, public deck counts and declared player randomness
enter the worker. Actual deck order, opponent blind-reserve identities and referee
game seeds remain outside it. The envelope binds the corrected ruleset and
catalogue. Canonical dummy hidden assignments are inputs to observer
determinization, not the true referee world. All 23 prepared original/permuted
lawful inputs must agree byte-for-byte. The three completed repeat pairs must
also have identical scientific records, excluding timing and file-path metadata.

## Exhaustive correctness audit

After every timely completion, before starting the next timed attempt, use a
fresh audit worker on the same assigned core. Its initialization allowance is
60 seconds and its verification allowance is at most 600 seconds. These are
verification costs, never additional thinking time for the player.

Audit every candidate continuation in all 128 worlds. Reconstruct the lawful
worlds independently, verify their ordered seeds and state identities, and replay
each recorded action through the independent dictionary referee. Check payment,
returns, refills, reservation visibility, nobles, score/bonus/token conservation,
end-round termination and winner/tie semantics. Verify the forced first action
and recompute raw035's complete ordered legal-action scores and first maximum
on every subsequent step. Check recorded action index, legal count, score
digest, state digest, trace hash, length and terminal outcome. Recompute all
candidate means and the final ordered choice independently. Never compare new
outcomes or choices with the legacy outcome as an acceptance requirement.

Recorded digests, proposal scores, means and discrete quantities require exact
agreement under the fixed serial backend and frozen feature/model dependencies.
A numerical discrepancy is a failed check, not a reason to introduce a tolerance
after launch. Scientific acceptance requires the exhaustive audit, not just a
legal returned action or a matching aggregate mean.

A scientific mismatch stops production qualification immediately and preserves
the failed evidence. An ordinary player move timeout or player initialization
timeout remains a recorded operational failure; other predetermined cases may
continue within the envelope. Every non-timeout player error, crash, malformed
response or initialization error stops conservatively. Every unsuccessful or
unfinished exhaustive audit, including an audit timeout or audit initialization
timeout, also stops; it cannot be counted as a verified decision. Global
resource/time exhaustion preserves partial results and yields incomplete
qualification, not a passing subset. No failed case is erased or replaced.

## Clock, evidence and resource envelope

Use one numerical thread and one logical thread of a verified performance core,
normal priority, and one active player or audit worker at a time under the shared
execution lease. Keep a lightweight supervisor without Torch. Record topology,
initialization, trusted wall/CPU timing, sampled process-tree RSS and work.
The desktop is shared; sampling and ordinary host load limit timing inference.

The new `PlannerWorker` reuses the frozen external02 process-tree sampler with
only the authenticated Python executable and its exact Windows system console
helper permitted; no Rust or other computation child is allowed. Account for
combined process-tree RSS/CPU and enforce affinity for both processes. An
authenticated console helper can be pinned once during initialization, with its
before/after affinity recorded. Any later affinity drift, unexpected descendant,
process replacement or unobservable process tree is fatal. Supervision tests
cover trusted receipt deadlines and rejection of forbidden children.

Start the player clock before input serialization/delivery. Acceptance ends at
the supervisor's receipt of the complete response; reject any response stamped
after its deadline, regardless of child timestamps or later validation. Include
search, full trace recording, compression, sealing/flushing and response delivery
inside this clock. Terminate overdue workers and descendants without pondering,
retained trees or a fallback decision. Keep the scientific audit outside the
move clock and inside the global clock.

Store complete action/digest traces once, as bounded gzip records in exclusive
case-owned files. The timed response contains their sealed identity and summary,
not a duplicate of the full traces. Stream per continuation; do not retain full
state snapshots or score vectors for every ply. The audit reconstructs states
and scores. Bind compressed bytes and the declared uncompressed record stream;
reject missing, extra, reordered, truncated or modified records. Incomplete
files from interrupted cases remain visibly incomplete evidence.

The shared execution ceiling is four hours from the first launch, including all
initialization, timed requests, trace handling, exhaustive audits and final
checks. Before any timed case starts, reserve its full 60-second initialization,
its move allowance, audit initialization of 60 seconds, audit allowance of
600 seconds, and a 600-second finalization reserve. The 600-second audit limit
does not authorize a new run window or an increased player allowance.

Use a 256 MiB total new-evidence cap, including partial traces, logs, observations,
audit receipts and final reports. Reserve eight MiB for in-flight/control/final
receipts; ordinary writes must respect the remaining soft cap. Require at least
5 GiB free disk, 1 GiB available physical memory and 3 GiB commit headroom before
worker launch; enforce a 2 GiB live worker/process-tree RSS limit. Check disk,
artifact size and global time during work as well as between cases. A resource
stop never silently changes trace coverage, world count, models or deadline.

Historical sizing informs feasibility but is not a guarantee. The completed old
four-candidate panel had 26 decisions, 13,056 continuations and 454,708
continuation turns: mean 34.83 turns, maximum 78. Its response receipts occupied
7,727,766 bytes and stored outcome/trace hashes, not full action sequences. The
entire old two-player 98-attempt recovery evidence occupied 27,598,004 bytes.
Therefore that small total does not demonstrate that full traces fit the cap.

Compact serialized legal actions in those four-candidate records averaged
192.83 bytes, range 161–213. Applying that size to the saved turn count gives
about 84 MiB of actions alone; two 64-character per-ply hashes add roughly
56 MiB before field names, indices, continuation headers and other evidence.
Gzip and one-copy storage make the 256 MiB envelope plausible, but corrected
trajectories and compression ratios are unmeasured. Full state/score-vector
storage per ply or duplicate trace payloads are unsuitable for this envelope.

The old 26 completed four-candidate requests averaged 43.61 seconds and peaked
at 87.28 seconds; all 23 one-second attempts timed out. Initialization median
was 2.686 seconds, maximum 3.152 seconds across the old 98 workers. Those values
do not include the newly required exhaustive policy/transition replay. Even
before audits, the maximum declared initialization plus move time for this
49-case schedule is 6,083 seconds (101.4 minutes). Audit throughput is not yet
validated; the four-hour cap can stop an otherwise correct unfinished study.

## Acceptance and matched-budget recommendation

Report all 49 prescribed cases, each fit and stratum, actual enforced allowance,
completion status, audit status, initialization and timely completion times.
Do not impute a timeout as an observed completion at its censoring threshold.
Report the three repeat comparisons and all 23 privacy-input comparisons.

Corrected-rules research qualification at 120 seconds requires all 26 scheduled
research requests to finish within their enforced deadline and pass exhaustive
audits, every prescribed fit/cell to be represented, all privacy/repeat checks
to pass, and final source/resource checks to pass. A failed research request or
incomplete audit prevents that acceptance. The one-second panel is assessed
separately on its 23 actually enforced requests.

If the complete 120-second gate passes and the one-second gate does not, retain
the tested 120-second allowance as the recommended research comparison ceiling.
Call it **supported with 20% observed margin** only when every 120-second
receipt is at most 96 seconds. If all pass but any exceeds 96 seconds, label the
recommendation **conditional at 120 seconds with thin observed margin**. Do not
increase the allowance automatically or recommend an untested lower allowance
by rounding an observed maximum. Passing finite fixtures never guarantees every
future turn will meet the deadline. If the complete one-second gate passes,
report that separately rather than pretending 120 seconds was necessary; any
subsequent one-second matchup profile must retain its own declared qualification.

This turn recommends an allowance only. Before a future matchup, separately
freeze and qualify SplendoRust at the same whole-turn ceiling, selecting its
work quota by a prospective timing-only rule rather than match outcomes. The
completed external02 profile was qualified at one second with PUCT16; it is not
automatically a matched-compute opponent for a 120-second planner. Equal wall
allowances also do not imply equal actual CPU time, simulations or inference
work. Both sides must use the same trusted clock scope, core/thread allocation,
initialization policy and fail-closed deadline semantics. If evidence delivery
or other instrumentation changes for a later match, qualify that exact timed
interface first.

No corrected-rules strength claim follows from this fixture study. Four
candidates do not inherit the stronger historical eight-candidate result, and
the completed external02 screen does not determine this planner's strength or
authorize a new matchup/sample size.

## Sources for this prospective design

- `docs/engineering/local_player_qualification_20261018.md` and its recovery
  protocol and result document.
- `outputs/local_player_qualification_20261018_recovery_01/selection.json`,
  `schedule.json`, `rows/` and the 26 completed four-candidate request receipts
  under `attempts/`: read-only timing, work-count and storage calculations.
- `docs/engineering/external_strength_screen_20261018_02.md` and
  `docs/engineering/external_strength_screen_20261018_02_result.md`.
- The separately versioned corrected rules and independent dictionary referee
  in `tools/external_strength_screen_20261018/`.

All numerical feasibility figures above come from completed saved evidence or
explicit arithmetic on it. No new inference, simulation or game was run to
prepare this protocol.
