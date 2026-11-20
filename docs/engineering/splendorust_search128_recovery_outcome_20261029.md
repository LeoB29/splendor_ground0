# Fixed 128 versus 64: recovery outcome and qualified checker correction

29 October 2026, 00:29 Paris. **The benchmark is stopped, not running.**

## What ran

The user authorized reproduction and tests of the saved infrastructure failure,
then an attempt to resume. A separate recovery implementation passed 267 combined
model-free tests and independent review. Its exact freeze authenticated 40,447
inputs, including the original 17,931-file evidence tree. The original failed
attempt and its rejected move receipt were preserved.

The recovery authenticated the response already emitted for game 39 (index 38),
ply 39, at an original arrival of 1.0749384 seconds. It did not retry that decision,
reset its clock, or rewrite its rejected receipt. A separate disclosed bridge
applied that saved action once.

The live recovery passed 60 seconds of RAM stability (minimum 6,503,153,664 bytes),
119 actual entrypoint decisions and the remaining-resource admission. The same
two qualified players continued from ply 40. All 14 new moves completed successfully;
the game ended naturally at 54 plies. The maximum new response was 0.8024491 seconds
for 128 and 0.4550449 seconds for 64. These 14 observations do not establish general
timing reliability or deployment qualification.

## Why it stopped again

Closed-worker verification rejected the admitted startup suites because an
inherited checker allowed them only in folders named `qualification_*`. A second
check also expected only one startup request in ordinary gameplay folders. Both
assumptions conflicted with the explicit recovery plan to qualify two workers and
then reuse those same workers for the interrupted game.

This integration case was missed by the earlier tests and review. It is a checker
defect, not a player timeout, game loss, or failure of the numerical model.
The controller stopped before writing its normal online replay receipt. Its
actual owner exit was 1; all owned process identities are absent and cleanup
reported no errors. The recovery attempt remains incomplete and unaccepted.

## Separate saved-evidence qualification

An additive checker permits only the two exact frozen retained workers. It
requires graph request 1, one authenticated entrypoint suite at request 2, and
ordinary requests from 3 onward. It retains all other raw transport, scientific,
graph, placement, compression, cleanup and corrected-rules ledger checks.

Seventeen tests passed, including the actual saved suite-to-play-to-closure path
and rejection of extra, repeated, misplaced or incorrectly bound startup work.
Independent review approved the exact implementation. A bounded saved-only run
then passed in 12.531 seconds, peaking at 96,473,088 bytes of RAM:

- Both original checker rejections reproduced.
- Two complete worker histories: 18 raw requests,119 actual startup decisions,
  and 22 graph cases verified.
- All 14 new saved ordinary moves verified.
- Full 54-event ledger verified: 39 earlier accepted moves, one explicit recovered
  response, and 14 new accepted moves.
- 277 evidence inputs and 171 imported source/runtime pins bound and preserved.
- No new model calls, games, decision retries, receipt rewrites or relaunch.

Game 39 is therefore verified by this separate recovered-evidence receipt. It is
not retroactively accepted by either failed frozen attempt. The 400-game strength
benchmark has no accepted final result. There are 38 originally verified main
games plus this separately verified 39th game; 361 scheduled games remain.

## Recommended next continuation

Prepare a separate bounded continuation using the qualified lifecycle checker
and all three evidence segments. Start at game 40 (index 39), preserving the fixed
400-game schedule and all failures. Reuse existing passed startup and saved-game
receipts wherever exact worker entrypoint/runtime identity permits; do not rerun
the failed decision or the completed 39th game. Reconcile final reader, worker
inventory and resource accounting for these segments before freezing a launch.

Keep the original clocks: collection cutoff 29 October, 17:21:52 Paris; all independent
acceptance and owner work by 21:21:52 Paris. Charge elapsed finalization/verification,
RAM waiting and worker lifecycles from both attempts, and recompute remaining
capacity before launch. No clock extension is recommended. Preserve aggregate
16 GiB evidence, 4 GiB final reserve, 4 GiB available RAM, 2 GiB worker RSS, and all
scientific/replay checks. Do not edit or relaunch either failed frozen attempt.

No further continuation was launched during this task.

## Research meaning

The concurrency-safe accounting completed a short live exercise without its old
failure recurring. That does not prove the race was encountered, nor establish
long-run reliability. The checker correction now passes the exact lifecycle that
the earlier tests missed. Neither finding answers whether 128 simulations improve
playing strength over 64; that still requires the complete paired comparison.

Optimized64 remains the baseline. Both players still use three worlds and the
same network, precision and search settings apart from simulation count. The
9-second internal/10-second external research guards remain; 700 ms and 1 second are
descriptive. External testing stays paused; larger-data work and training remain
deferred.

## Evidence references

- Recovery freeze: `outputs/splendorust_search128_recovery_20261028_01/freeze.json`,
  SHA 524daba942dd6478ccbfc2a9293f391b1502ffca94163228aef8d1d3dba718f4.
- Actual failed owner exit:
  `outputs/splendorust_search128_recovery_20261028_continuity/root_owner_exit_observation.json`.
- Separate checker receipt:
  `outputs/splendorust_search128_closure_check_20261029_01/closure_check.json`,
  SHA 647f06bdca298e4b1fcc4b4bdff205906c422707e4a1684f23e2fe409dc17dad.
- Separate checker actual exit:
  `outputs/splendorust_search128_closure_check_20261029_01/checker_os_exit.json`,
  SHA f42b36d184b4e38f09954add6db291cf5ef40caa5b7932c35240c43b0a38f162.
