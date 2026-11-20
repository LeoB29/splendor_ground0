# Local planner timeouts and the matched benchmark deadline policy

20 October 2026. Use the unchanged verified four-candidate, 128-world planner
against the unchanged qualified SplendoRust profile with a hard 120-second
complete-response deadline. Count genuine player timeouts as losses. This first
benchmark should measure the players as delivered on the declared hardware,
including their reliability. It should not be described as a comparison with
unlimited thinking time or as failure-free qualification.

This is a recommendation for the next protocol. No benchmark games, player
changes, model inference or timeout retries were performed for this review.
Matchup totals remain unopened and the larger-data study remains deferred.

## What the four records establish

Each worker initialized successfully, ran substantial search, and failed to
deliver a complete response within 120 seconds. Every receipt says `timeout`,
`accepted=false`, `received_before_deadline=false`, with no result or complete
response timestamp. Each output log contains only its initialization handshake;
error logs contain progress events and no recorded model exception. All 28
timeout artifacts match their frozen identities.

| Saved request | Model fit | Supervisor elapsed | Sampled request CPU | Readable completed continuations |
|---|---:|---:|---:|---:|
| Game 6, ply 0 | 161001 | 120.078 s | 80.109 s | 472 / 512 |
| Game 7, ply 7 | 161001 | 120.101 s | 90.406 s | 450 / 512 |
| Game 17, ply 6 | 161003 | 120.133 s | 104.500 s | 507 / 512 |
| Game 19, ply 7 | 161001 | 120.140 s | 95.500 s | 418 / 512 |

Ply numbering starts at zero, so every failure occurred within the first eight
actual moves. The extra 0.078–0.140 seconds is supervisor enforcement and
bookkeeping latency, not permitted extra thinking time or evidence of a reply
that narrowly missed acceptance.

The incomplete compressed traces retain 118, 112, 126 and 104 complete shared
worlds respectively, with additional candidate continuations in three cases.
These are lower bounds on completed work because compression and file buffers
can hide further work. The 507-of-512 case does not establish that a small grace
period would have sufficed: remaining computation, finalization and delivery
times are unknown. No partial action was delivered or independently qualified.

Sampled peak worker-tree memory was 240.5–280.3 MiB, well below the 2 GiB limit.
The separate initialization times were 2.719–4.715 seconds, within their
60-second allowance. These were decision timeouts, not initialization failures
or recorded worker-memory-limit breaches. Host memory pressure can still affect
a small worker, so these measurements do not rule out paging.

Request CPU counters show approximately 80–104 CPU seconds during each
120-second request. They are sampled process-tree deltas, include the console
helper and can include a little post-deadline activity. Their difference from
wall time does not identify scheduler contention, paging, I/O, instrumentation
cost or the amount of useful computation left. There is insufficient recorded
telemetry to assign a causal RAM diagnosis.

## Practical significance

Four of 365 attempted local moves timed out, but those failures ended four
different games out of the eighteen-game operational stage. The game-level cost
is material. Neither count is an estimate of a future independent failure
probability: moves and games are correlated, the sample is small, and a forfeit
removes later moves from observation.

Among the 361 completed moves, fifteen took more than 100 seconds and six took
more than 110 seconds. The slowest completed response took 118.652 seconds.
The 65 completed decisions in plies 0–7 averaged 90.592 seconds, compared with
50.579 seconds across all completed decisions. This is a post hoc description,
not a controlled causal comparison, but it shows that the early-game workload
already consumes much of the allowance.

Fit 161001 had three timeouts in 80 attempts, fit 161002 zero in 150, and fit
161003 one in 135. Do not select a different model from these small, unequal,
correlated counts; all three remain in the intended balanced benchmark. The
second recovery's 47 local moves had no timeout, but involved different
positions and cannot demonstrate that RAM waiting eliminated the problem.

## Recommended deadline and failure rules

1. **Keep the entrants fixed.** Retain the verified local four-candidate,
   128-world policy, all three model fits, and the qualified SplendoRust
   6,400-iteration profile with its existing 108-second internal allowance.
   The common external allowance remains 120 seconds. Equal elapsed allowance
   does not imply equal search work.

2. **Keep the qualified clock boundary.** Separately allow 60 seconds for
   initialization. Start the real monotonic move clock before request
   serialization and delivery; stop it at the trusted reader's timestamp for
   the complete response. Include observation reconstruction, search, required
   turn subactions, local trace compression/flushing/hashing, response
   serialization and transport. Parent legality and transition verification,
   and later replay audits, remain outside this move clock. Do not quietly
   remove trace overhead or substitute CPU time for elapsed time.

3. **Make a genuine timeout final.** No complete eligible response by the
   deadline means a timeout forfeit and game loss. Close the owned worker tree,
   retain its partial evidence, and continue the fixed schedule. Do not grant
   grace, restart the request, salvage a partial trace, replace the seed, or
   discard the game. Play the other scheduled seat-swapped leg as well.

4. **Wait for RAM before starting a worker.** Apply the same admission policy
   to both players: at least 2 GiB physical and 4 GiB commit headroom; require
   the existing two-minute stable window initially and after a shortage. Keep
   waiting episodes, total waiting and the full study under prospective finite
   limits. Once a request starts, never suspend, refund or reset its clock.
   Keep one game at a time, fresh workers, the same performance core, normal
   priority and one numerical thread. Avoid competing compute during the run.

5. **Keep infrastructure failures distinct.** Preserve the existing taxonomy:
   timeout, worker exit, malformed/wrong/illegal/unsolicited response,
   write error and recognized internal-budget exhaustion are scored forfeits.
   Initialization failure, worker error, adapter/work/observer mismatch,
   referee inconsistency, failed authentication/audit or a resource guard
   produce a hard hold. A pre-worker hold leaves an unstarted request available
   for the predeclared bounded recovery; an unresolved active-request fault
   blocks acceptance. Do not invent a draw or replacement game. Low sampled CPU
   utilization alone is not evidence for reclassifying a timeout.

6. **Include reliability in the primary result.** Count timeout losses in the
   paired game-score analysis. Report player failures, natural completions and
   thinking times separately. A comparison restricted to naturally completed
   games is selected after observing failures and must not replace the primary
   result. Keep the pair, not an individual move, as the relevant pairing unit.

The next protocol must freeze fresh seeds, balanced model and seat assignment,
sample size, paired analysis, failure/recovery rules, verification, resource
ceilings and stopping rules before opening matchup totals or starting strength
games. All 24 earlier operational games remain excluded from the fresh sample.
The existing operational contract already treats these player failures as
forfeits; retained timeouts prevent a failure-free claim, not an honest
as-delivered strength measurement.

## A separate future player for reliable deadlines

The current planner chooses its move only after all 128 worlds finish and emits
one response after trace finalization. Its progress callback does not publish
a usable incumbent. A deadline-aware version could establish a legal fallback
early and update its choice only after completing a shared world across every
candidate, leaving time for delivery. It would need explicit rules for a
partially completed world, ties, zero completed worlds and trace finalization.

That changes the player. A nominal 108- or 110-second internal cutoff is not
justified by these censored records alone. Version and qualify that policy on
the four development cases plus independent positions before testing it as a
separate entrant. Do not silently insert it into the first benchmark or erase
the existing baseline's reliability cost.

## Evidence

- Accepted operational analysis and report:
  `outputs/live_game_operational_recovery_20261020_02_validation_02/` and
  `docs/engineering/live_game_operational_final_validation_repair_20261020_result.md`.
- Authenticated timing aggregation:
  `outputs/local_timeout_policy_review_20261020_01/aggregate_completed.json`;
  1,094 inspected source artifacts were rechecked without modification.
- Authenticated timeout logs and readable trace prefixes:
  `outputs/local_timeout_policy_review_20261020_01/partial_trace_review.json`;
  all 28 source artifacts matched the frozen index.
- Clock and delivery semantics:
  `tools/local_player_qualification_128/clock.py`, especially lines 260, 339,
  390, 432 and 497; fixed-work selection in
  `tools/live_game_operational_20261019/local.py`; trace finalization in
  `tools/live_game_operational_20261019/worker.py`.
- Governing roadmap and failure rules:
  `docs/strategy/external_strength_calibration_2026-10-18.md`,
  `docs/engineering/live_game_operational_20261019.md`,
  `docs/engineering/live_game_operational_recovery_20261020_02.md`, and
  `docs/engineering/verification_policy.md`.
