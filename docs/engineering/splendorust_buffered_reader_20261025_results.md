# Buffered response reading qualification results

25 October 2026. The separate buffered reader preserved correctness but failed
the frozen timing qualification. The six conditional operational games were
therefore not launched. The attempt completed with verified evidence and clean
process shutdown in 372.638 seconds, from 00:42:20 to 00:48:32 Paris time.

## Correctness and timing

All 119 combined prelaunch tests passed. Both parent implementations at 32 and
64 simulations passed their actual corrected-rules entrypoint counterexamples.
All 288 saved-position requests completed within the one-second deadline and
passed independent transition checks. The 72 position-and-quota groups matched
exactly across parent implementations and repeats, including complete actions
and search traces. There were 164 comparisons against matching historical
records: 64 at quota 32 and 100 at quota 64.

Each quota and panel independently required at least 5% improvement in mean
trusted arrival time, no increase in the 95th percentile, and every candidate
arrival within 0.700 seconds.

| Panel | Simulations | Baseline mean | Buffered mean | Mean improvement | Buffered maximum |
| --- | ---: | ---: | ---: | ---: | ---: |
| Development | 32 | 0.325656 s | 0.323495 s | 0.66% | 0.451542 s |
| Reserved | 32 | 0.268199 s | 0.266192 s | 0.75% | 0.318057 s |
| Development | 64 | 0.545241 s | 0.529355 s | 2.91% | 0.742179 s |
| Reserved | 64 | 0.529231 s | 0.524322 s | 0.93% | 0.691463 s |

All four 95th-percentile comparisons improved, but none met the mean-improvement
gate. Two buffered 64-simulation development decisions exceeded the admission
margin: request index 46 at 0.7192573 seconds and index 65 at 0.7421787 seconds.
Their saved roots were respectively the earlier successful operational game
`p002_g5`, ply 21, and `p000_g0`, ply 8. Both were valid completed responses,
not timeouts. All baseline observations were retained as well.

The original three slow moves passed the margin on both buffered-reader passes:

| Original recorded ply | First pass | Second pass |
| --- | ---: | ---: |
| 42 | 0.492159 s | 0.553000 s |
| 48 | 0.549103 s | 0.594904 s |
| 50 | 0.489999 s | 0.586659 s |

Passing those three positions did not establish robust timing across controls.
The small average gains show that buffering alone was insufficient in this
attempt. They do not identify whether inference, supervision, scheduling or
hardware-frequency variation caused the remaining delays.

## Verification and next step

Final verification passed for all 288 saved decisions, four worker log sets,
frozen sources and runtimes, and preserved predecessor evidence. Peak combined
sampled worker and helper RSS was 603,017,216 bytes. The watchdog exited with
code zero and no cleanup errors. There were no retries, operational games or
strength games. `completed_verified` describes the evidence; the separate
`buffered_reader_qualified` result is explicitly false.

Keep the verified baseline and both timing thresholds unchanged. The next
recommended work is a bounded component-timing diagnostic, first qualifying
the instrumentation overhead, then separating model execution, initial and
periodic supervision, pipe reception and host scheduling effects. Use the slow
positions and controls for diagnosis; do not tune repeatedly against this
attempt's reserved panel or restart the same operational games. Larger-data
work remains deferred.

## Evidence

Frozen source: `tools/splendorust_buffered_reader_20261025`.
Protocol: `docs/engineering/splendorust_buffered_reader_20261025.md`.
Output: `outputs/splendorust_buffered_reader_20261025_01`.
Freeze SHA256:
`a038ce1f6f8158917509927d450af9ce4dcc2fadce29f044f63971ba6e69e1c6`.

Primary receipts are `qualification_summary.json`, `qualification_outcome.json`,
`verification_summary.json`, `all_worker_log_integrity.json`, `completion.json`
and `execution.json` in that output directory. The prior failed comparison and
every newly recorded timing miss remain unchanged.
