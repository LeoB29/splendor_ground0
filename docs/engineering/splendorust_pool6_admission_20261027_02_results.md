# Fresh unchanged-player admission: verified negative margin result

27 October 2026. One separately frozen attempt completed qualification and all
six fresh operational games. Every live move met the one-second deadline, but
nine exceeded the required 700 ms admission margin. The candidate is therefore
not admitted to the strength comparison. No criterion was relaxed and no retry,
replacement game or 200-game benchmark was launched.

The runner's scientific verification passed. Independent review accepted the
verified negative outcome, with all 219 recorded owned process identities
absent. No experiment remains running.

## Fixed scope and qualification

The players were unchanged: optimized64 with three sampled worlds, and the
six-world candidate, both using the same pretrained network and 64 total search
simulations. We retained the original weights, precision, search, player
identities, response evidence, timing observer, CPU placement, 0.900-second
internal guard and 1.000-second complete-response deadline. Only the separately
versioned admission machinery and fresh inputs changed.

All 88 model-free implementation tests passed. The frozen qualification then
passed its 194 scientific decisions: 104 corrected-entrypoint counterexamples,
48 saved exact reproductions, six edge cases and 36 requests covering both
players on 18 fresh positions. All 90 ordinary timed qualification requests
finished below 700 ms; the maximum was 569.187 ms. This authorized the six games.

Corrected rules and catalogue identities remained bound at worker startup.
The checkpoint remains explicitly historical; no data collection for training
or fitting occurred.

## Six fresh operational games

All three swapped-seat pairs finished naturally: 338 moves in total, 169 by
each player. No timeout, illegal move, scientific mismatch or forfeit occurred.

| Player | Moves | Median receipt | Maximum receipt | At or above 700 ms | One-second timeouts |
| --- | ---: | ---: | ---: | ---: | ---: |
| Unchanged optimized64, three worlds | 169 | 461.569 ms | 952.880 ms | 2 | 0 |
| Unchanged six-world candidate | 169 | 519.204 ms | 790.287 ms | 7 | 0 |

The frozen rule permits completing the six-game schedule after successful
margin misses, while making admission negative. A margin miss is not a timeout
and was not converted into a loss. The operational games are not a sufficient
sample for a playing-strength claim.

`admission_gate.json` records `timing_margin_passed: false`,
`resource_passed: true`, and `passed: false`; `completion.json` records
`not_admitted_verified`. Inherited descriptive timing summaries contain wording
that calls 700 ms descriptive. That wording does not govern admission: the new
frozen protocol and admission gate explicitly enforce it, and it failed here.

## Verification, resources and preservation

All six games passed independent model-free ledger replay. The fixed selection
rule yielded 22 model reproductions from the available occupied strata, below
its ceiling of 48; all 22 matched their saved scientific outputs exactly.
Selection used neither timings nor game outcomes. These are selected model
reproductions in addition to full ledger replay, not model replay of every move.

The runner verified 36 workers, 396 graph cases, 488 parent requests, raw
request/response records, placement, compressed logs, clocks, source identity
and closure. The 488 requests comprise 450 ordinary decisions, 36 graph bundles
and two startup counterexample bundles. There were 554 scientific decisions
including the 104 decisions inside those two startup bundles.

The actual owned runner, watchdog and guardian processes all exited with code
zero. Run time was approximately 15 minutes 51 seconds, within the two-hour cap.
Peak combined worker memory was 616,726,528 bytes, approximately 588 MiB. The
completion seal covers 3,604 artifacts and 107,015,101 bytes, approximately
102 MiB, excluding later independent-review and reporting artifacts.

The RAM safeguard paused between moves for 235.281 seconds, including recovery
stability. Including the initial stability wait, cumulative waiting was 295.390
seconds, within the 600-second cap and charged to the unchanged overall clock.
The saved minimum available RAM during that wait was 54,968,320 bytes. Play
resumed automatically after the existing reserve recovered and remained stable;
no active move clock was reset.

All 8,064 frozen inputs and the earlier 982-file repeat-timing and 667-file
original admission inventories are preserved. The original negative admission,
the successful repeat diagnostic and this negative admission remain distinct.

Two read-only reporting/review mistakes were preserved separately under
`postrun`: the root reader indexed a row count as an array, and the independent
reviewer's lightweight helper envelope initially omitted `placement_sequence`.
Neither changed the experiment, players or frozen evidence, and neither caused
another game or model run.

## What the saved slow moves show

Indices below are zero-based, matching the saved evidence. All nine moves used
65 neural callbacks and 64 main-search simulations. Three had an additional
payment phase with zero search; there was no repeated full neural search.

| Game / ply | Player | Receipt | Handler wall time | Sampled worker CPU | Response write/flush |
| --- | --- | ---: | ---: | ---: | ---: |
| 0 / 4 | Six worlds | 740.821 ms | 591.006 ms | 281.250 ms | 67.322 ms |
| 0 / 6 | Six worlds | 714.387 ms | 573.825 ms | 437.500 ms | 50.266 ms |
| 0 / 20 | Six worlds | 717.886 ms | 600.283 ms | 312.500 ms | 51.181 ms |
| 1 / 41 | Six worlds | 714.584 ms | 617.694 ms | 359.375 ms | 30.944 ms |
| 2 / 36 | Three worlds | 725.962 ms | 643.387 ms | 375.000 ms | 14.551 ms |
| 2 / 39 | Six worlds | 725.141 ms | 600.558 ms | 484.375 ms | 50.385 ms |
| 2 / 47 | Six worlds | 702.573 ms | 567.562 ms | 390.625 ms | 45.328 ms |
| 3 / 26 | Six worlds | 790.287 ms | 634.928 ms | 453.125 ms | 80.544 ms |
| 4 / 19 | Three worlds | 952.880 ms | 881.367 ms | 328.125 ms | 5.477 ms |

All nine crossings occurred among the 246 moves before the RAM pause; the 92
subsequent moves had none. The 952.880 ms control move immediately preceded the
pause. This is evidence consistent with runtime interference, not proof that
memory pressure caused every slow move. Available RAM was not measured inside
each move, and the first eight misses occurred roughly one to four minutes
before the pause. The wait's RAM minimum cannot be assigned to those moves.

One slow candidate move, game 2 / ply 39, was selected by the fixed reproduction
rule and reproduced exactly at 523.999 ms versus its original 725.141 ms.
The later time and new worker lifecycle prevent treating this as a controlled
causal comparison.

Sending the candidate's larger response remains a real cost: median write/flush
was 48.264 ms versus 1.400 ms for the control. It cannot explain the entire
failure. The control's worst delay was mostly inside the handler, and even
subtracting the candidate's entire 80.544 ms write/flush interval from its
790.287 ms move would leave it above 700 ms. These boundary intervals are wall
time, not a direct measurement of inference time, scheduling delay or paging.

## Recommended next step

Prepare one bounded runtime-interference diagnostic using these nine saved
positions and fixed controls. Interleave the unchanged players, keep the same
deadlines and exact-output checks, and predeclare stable RAM headroom. First
qualify the overhead of lightweight host memory/page-fault and CPU scheduling
observations; use them only if their overhead passes. Run no fresh games in
that diagnostic. The question is whether the delayed handlers are spending
extra time computing or waiting on the host, and which mechanism explains it.

This is a recommendation, not a launched study. Another unchanged admission
would add less information than resolving the shared timing variation. A
candidate-only response optimization would leave the worst control delay
unexplained. Keep the 700 ms margin and the one-second deadline unchanged.

## Broader research implication

The project is testing whether spreading the existing 64 simulations across six
hidden-card worlds improves decisions relative to optimized64's three worlds,
without increasing the network or exceeding one second per move. The earlier
saved-position study supported more stable choices, and repeat timing showed
that the original three slow requests did not consistently recur. This fresh
admission adds the missing live-game evidence: correctness held, but rare slow
moves appeared for both players and failed the agreed safety margin.

We therefore still do not know whether six worlds is stronger in full games.
Resolve the timing variation before launching that strength comparison, so
machine delays do not confound the change in search allocation. Optimized64
remains frozen, external-opponent testing stays paused, and larger-data and
training work remain deferred.

## Evidence references

- Protocol: `docs/engineering/splendorust_pool6_admission_20261027_02.md`.
- Source: `tools/splendorust_pool6_admission_20261027_02` (frozen).
- Evidence: `outputs/splendorust_pool6_admission_20261027_02`.
- Freeze SHA256: `09551c1f9e8782e0d717576aa51f96bff47403637b56cb087fd843ff8c58c0e8`.
- Primary results: `qualification_summary.json`, `admission_gate.json`,
  `performance.json`, `boundary_analysis.json`, `coverage.json`,
  `verification_summary.json`, `completed_evidence_seal.json`,
  `replay_sample_manifest.json`, `completion.json`.
- Actual exit receipts: `execution.json`, `watchdog_execution.json`,
  `guardian_os_exit.json`.
- Independent acceptance: `postrun/independent_completion_review.json`,
  SHA256 `e68f6dc7b34c49147a933303ebfc9bdcb97671951e15e143a71a0b4f990f615e`;
  accepted verified negative, 11,699 consumed inputs, 2,347,813 bytes.
- Disclosure supplement: `postrun/independent_completion_disclosure_supplement.json`,
  SHA256 `34c2c17c08a2bb800946ffb1c79aa3e005d895628aa99ae50466ae82267d004e`;
  binds the separate root reader error and the descriptive-summary wording
  clarification without changing the acceptance receipt or frozen evidence.
