# Optimized64 versus unchanged32 completed comparison

25 October 2026. The frozen 200-game comparison completed successfully at
13:54:15 Europe/Paris, including verification and owning-process cleanup.
Optimized64 scored **64.25%** against unchanged32, meeting every predefined
strength criterion. Two deadline-related forfeits by optimized64 remain in its
71 losses. The result supports more search with these fixed weights, while
identifying a remaining timing-reliability problem.

## Playing strength

The assessment used 100 fresh initial worlds, each played twice with swapped
seats. Six additional operational games admitted the run and are excluded from
the strength estimate. Both players retained the one-second complete-response
limit and unchanged internal 0.900-second limit.

| Measure for optimized64 | Result |
| --- | --- |
| Wins / draws / losses | 128 / 1 / 71 |
| Score, counting a draw as half a win | 64.25% |
| Paired 95% bootstrap interval | 58.25% to 70.25% |
| Score from seat 0 / seat 1 | 65.0% / 63.5% |
| Natural games / forfeited games | 198 / 2 |
| Predefined success criteria | All passed |

The criteria required at least 55% score, a paired 95% lower bound above 50%,
at least 50% in each seat, and all integrity checks. The frozen analysis used
50,000 resamples of the 100 seat-swapped pairs, with independently verified
arithmetic. Both forfeits are included throughout. This establishes a useful
advantage in this particular matchup and allowance; it does not establish
human strength or a universal advantage against other opponents.

## Response time and retained failures

| Main-game completed decisions | Optimized64 | Unchanged32 |
| --- | --- | --- |
| Count | 5,575 | 5,576 |
| Mean arrival | 0.3851 s | 0.2386 s |
| 95th percentile | 0.5098 s | 0.3055 s |
| Maximum accepted arrival | 0.8683 s | 0.7269 s |
| Accepted arrivals above the 0.700 s engineering margin | 18 | 1 |
| Failed decisions | 2 | 0 |

These summaries of completed decisions exclude the two failed optimized64
requests. The 0.700-second operational admission margin was descriptive during
main play, so exceeding it did not create additional losses.

In game index 12, ply 56, optimized64 returned an internal-deadline error after
its unchanged 0.900-second allowance. The error arrived at 0.9752681 seconds,
before the external deadline, but supplied no completed action. The frozen
classifier recorded a player failure and a loss. In game index 66, ply 29,
the trusted response arrival was 1.0466894 seconds, beyond the one-second
deadline, and the player lost by timeout. No successor was invented for either
failed move, and neither game was replayed or replaced.

Both failures were deadline-related; the first remains labelled `worker_error`
in the original evidence and the second `timeout`. Sampled request CPU costs
were 0.359375 and 0.34375 seconds. The gap between CPU and wall time motivates
inspection of scheduling, waiting and communication, but does not by itself
identify the cause. The larger run shows that the earlier small qualification
did not eliminate rare timing failures.

## Verification and resources

Both actual corrected-entrypoint 52-decision suites passed, as did the six
fresh operational games. Final verification passed for all 206 game ledgers,
48 fixed-sample decisions reproduced in their original execution modes,
231 graph-instance proofs of exact outputs and float32 bits, and 462 worker
log/cleanup inventories. The evidence seal contains 40,636 artifacts. Frozen
source/runtime checks and independent statistical verification also passed.

The job took 8,746.625 charged seconds, about 2 hours 26 minutes, within its
six-hour cap. Final verification took about 13 minutes 27 seconds, within its
30-minute cap. Saved evidence was about 525 MiB before this additive report
and post-run audit, below the 2 GiB cap. The owning watchdog exited successfully
with no cleanup errors. No experiment remains running.

A supplementary model-free audit during the later status review rehashed all
2,691 frozen inputs successfully, then stopped at its 280-second cooperative
time boundary while rereading the 40,636-artifact seal. It observed no mismatch
in completed checks, but its status is `incomplete_bounded_extra_audit`; its
remaining checks must not be reported as passed. This separate attempt does not
replace the completed scheduled verification above. It preserved its partial
receipt and was not restarted.

## Recommended next step

Use optimized64 as the leading candidate for the next development comparison,
preserving unchanged32 as a control and both recorded failures. Before increasing
search again, prepare a bounded postmortem of the two failures and slower accepted
moves using saved evidence. It should identify the next specific reliability fix
or the smallest diagnostic needed to distinguish causes. Any implementation
change needs its own version, exactness qualification and fresh operational
assessment. This result changes no reference pointer automatically.

The preceding experiments established that small search improved the network
alone and that 32 simulations improved 16. The immediate preceding qualification
established that the exact-output inference optimization made 64 simulations
practical on the tested positions and six games. This comparison supplies the
missing playing-strength evidence: the extra search produces a substantial gain
with the network and training fixed. The programme can now work from a stronger
candidate while resolving rare deadline failures, then test whether gains survive
diverse independent opponents and expert humans. Larger-data collection and
training remain deferred.

## Evidence

- [Frozen protocol](splendorust_graph64_strength_20261025.md)
- [Completion](../../outputs/splendorust_graph64_strength_20261025_01/completion.json)
- [Execution and cleanup](../../outputs/splendorust_graph64_strength_20261025_01/execution.json)
- [Paired results](../../outputs/splendorust_graph64_strength_20261025_01/analysis.json)
- [Timing and forfeits](../../outputs/splendorust_graph64_strength_20261025_01/performance.json)
- [Independent statistical check](../../outputs/splendorust_graph64_strength_20261025_01/independent_statistics.json)
- [Supplementary audit and its time limit](../../outputs/splendorust_graph64_strength_20261025_01/independent_postrun_audit.json)
- [Revised project roadmap](../strategy/splendorust_revised_roadmap_2026-10-24.md)

Freeze SHA256: `556480a2501f4fb4cf829fe518aa4ba1ccbf10f12ee088fccf627408dfd69560`.
All frozen players, settings, source, failures and original evidence are preserved.
