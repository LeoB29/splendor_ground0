# Observer/checker repair: overhead passed, diagnostic stopped with verified evidence

27 October 2026. A separate version corrected the saved context-counter failure
and the final checker's handling of a terminal invalid measurement. All 122
model-free tests passed. The replacement was frozen, independently approved and
launched once. Both actual workers passed correctness qualification and the
72-request observer-overhead gate passed. The full diagnostic then stopped after
122 of 144 requests on a different host-wide CPU counter inconsistency.

The player response at the stop was valid. The repaired finalizer completed its
preservation checks and reported `incomplete_measurement_failure_verified`,
with a nonzero exit. The full diagnostic is not complete. No retry, new game,
player change, deadline change or training job occurred. The earlier failed
attempt remains unchanged.

## Changes and qualification

Only the separately versioned diagnostic machinery changed. The observer now
preserves raw live-thread context aggregates and signed changes. It does not
call any such change an exact interval context-switch total, even when positive.
The old saved failure remains an invalid historical measurement; no missing
placement data was invented. Independent native measurements can be retained
when scheduling-count data is missing, while actual API and identity failures
remain fatal.

The checker separates live measurement acceptance from forensic preservation.
Live collection still stops on an invalid observer window. Final verification
can authenticate a single terminal invalid window, its valid player response,
available evidence and missing fields, without accepting the measurement or
continuing collection. Actual failure exits remain nonzero.

The regression suite includes the exact previous failure, positive/zero/negative
context changes, missing data, corrupt bindings and identities, the actual new
observer result shape, immediate stop, final evidence sealing and nonzero failure
status. Initial prelaunch test failures and their corrections remain saved.

All 122 aggregate tests passed in 15.06 seconds with source hashes unchanged.
Actual-model startup then passed 104 corrected-rule scientific decisions and
22 exact graph cases. The players retained their pretrained weights, numerical
precision, 64 simulations, respective three/six-world settings, CPU placement,
one-second complete-response deadline and 0.900-second internal guard.

## Observer overhead

Both players completed 18 OFF and 18 ON requests, in the frozen interleaved
order. All measurements in this stage were valid.

| Player | OFF mean | ON mean | OFF p95 | ON p95 | Gate |
| --- | ---: | ---: | ---: | ---: | --- |
| Optimized64, three worlds | 447.489 ms | 444.785 ms | 492.570 ms | 487.063 ms | Passed |
| Six-world candidate | 505.847 ms | 485.446 ms | 592.739 ms | 593.463 ms | Passed |

The fixed limits were a maximum 5% mean increase, 10 ms added mean and 25 ms
added nearest-rank p95. There was no trimming or replacement. Lower observed ON
means do not establish that instrumentation makes a player faster; this is a
finite overhead qualification under the observed machine conditions.

## Collected diagnostic prefix

The full stage completed 61 requests per player before the measurement stop.
All 18 roots had at least three full-stage requests per player; the fourth round
was incomplete. The frozen schedule was not extended or repaired.

| Player | Full-stage responses | Full-stage maximum | All-stage responses | All-stage maximum |
| --- | ---: | ---: | ---: | ---: |
| Optimized64, three worlds | 61 | 503.316 ms | 97 | 503.316 ms |
| Six-world candidate | 61 | 546.105 ms | 97 | 593.463 ms |

All 194 ordinary responses were valid, with zero one-second timeouts and zero
700 ms crossings. All prior slow moves and negative admissions remain recorded;
these selected repeats do not estimate live-game reliability or overturn them.

The corrected context handling was exercised in the real run: window 0037
recorded a signed aggregate change of -74 for the candidate's Python process,
and window 0090 recorded -73 for the incumbent's Python process. Both windows
remained valid, with exact interval switch totals explicitly unavailable.

The run admitted work after a 60.094-second stable-RAM wait; no recovery wait
was needed. Minimum pre-request available RAM was 5.588 GiB. The 157 valid
enabled windows contained 1,406 native samples, with minimum available RAM
5.553 GiB and commit headroom 12.840 GiB. There were also 36 valid disabled
windows and one terminal invalid enabled window.

## New measurement failure

The terminal record is zero-based sequence 193, the incumbent on control root
`g2_p37`. Its response was accepted in 467.0084 ms. The observer preserved nine
native snapshots and reported `System counter reset`.

Between the last two snapshots, raw host-wide idle time decreased by 4,062,500
100 ns units and kernel time by 3,593,750, while user time increased by 156,250.
The recorded sample starts were 14.2901 ms apart. Both samples were after the
trusted response arrival, so this was not a player deadline failure. Per-process
identities and native creation times stayed consistent, and their CPU, page-fault
and I/O sequences remained monotonic.

These raw values establish an inconsistent host-wide counter sequence. They do
not establish that the machine restarted, that a process identity changed, or
why the Windows reading moved backwards. Do not clamp the values, infer a reset
cause, or use the invalid window as an accepted measurement.

Microsoft documents 100 ns units, kernel time including idle time, and summed
processor times on machines with at most 64 logical processors. The documented
processor-group limitation for larger machines does not fit this machine's 20
logical processors in group 0. Its API documentation does not specify an atomic
or monotonic snapshot guarantee. The saved wrapper uses separate buffers and
integer reconstruction; no conversion or buffer-aliasing defect was identified.
CPU migration or inconsistent internal accounting remain unproven explanations.
See [Microsoft's GetSystemTimes documentation](https://learn.microsoft.com/en-us/windows/win32/api/processthreadsapi/nf-processthreadsapi-getsystemtimes).

Unlike the previous attempt, the end-placement record is present and the
checker completed the failed-prefix audit. The sampler joined and closed its
handles; all three actual owner exits were 1, with no cleanup errors.

## Verification scope

The frozen finalizer verified the 194-response prefix, including 97 exact
historical comparisons and 158 exact first-observation repeats, together with
104 startup decisions, 22 graph cases and 198 parent requests. It sealed 1,439
evidence files and completed source/runtime/preservation checks. Charged execution
was 334.536 seconds, about 5 minutes 35 seconds, within the two-hour cap.

The saved `overhead_gate.json` separately records a passed 72-request stage.
The later coverage record conservatively reports `overhead_qualified=false`
after the invalid diagnostic measurement, alongside
`overhead_gate_passed=true`. Neither field should be silently rewritten: the
stage passed, but the full study remains incomplete and unaccepted.

Independent postrun readback passed for the preserved incomplete attempt. It
authenticated all 12,087 frozen inputs, 96 runtime files, 325 prior-attempt files
and the 1,439-file evidence seal, independently reconciled saved science,
transport/logs, graphs, entrypoint checks, overhead arithmetic and descriptive
analysis, and confirmed all 15 owned process identities were absent. It made
no model calls and reran no games or tests. Its disposition is
`verified_incomplete_measurement_failure`: failure verified, original status
unchanged, full completion and admission not accepted. No experiment remains
running.

## What this result establishes

The original observer repair works on the saved case and on real negative
aggregate changes. The checker repair also works: a measurement failure no
longer prevents preservation and verification of the valid player responses.
The corrected observer passed the planned overhead comparison.

The original slow decisions did not recur in this collected prefix with ample
RAM. That is compatible with a host-condition effect, but it does not establish
memory pressure or CPU scheduling as the cause. The schedule is incomplete,
conditions were not randomized, and selected saved roots are not live games.

## The next step

Use a separate version to mark an inconsistent host-wide CPU window unavailable,
retaining every raw sample and the offending pair. Preserve separately validated
process CPU, memory, faults, I/O and context-aggregate changes. Keep API failures,
process identity failures and invalid process-counter evidence fatal. Test this
exact failure and the downstream availability/reporting rules. Do not clamp or
discard the values, invent a corrected CPU percentage, reset the counters, or
retroactively accept this attempt. Freeze and qualify any replacement before
another full diagnostic.

No replacement is implemented or launched by this result report. A strength
benchmark is still premature while the diagnostic remains incomplete.

## The broader picture

The research aim is to improve SplendoRust-based playing strength within one
second per move. The current candidate distributes the same 64 search simulations
over six hidden-card worlds instead of optimized64's three. Earlier saved-game
work suggested more stable decisions; fresh admission games then exposed rare
slow responses in both players. This diagnostic is intended to distinguish the
players' computational cost from interference on this machine before committing
to a paired strength comparison.

The present result improves confidence in the measurement and failure-preservation
tools and shows ample timing margin in this saved-position prefix. It does not
yet show that six worlds are stronger or sufficiently reliable in live play.
Optimized64 remains frozen, external-candidate testing remains paused, and
larger-data collection/training remains deferred.

## Evidence

- Protocol: `docs/engineering/splendorust_pool6_hostdiag_20261027_02.md`.
- Source: `tools/splendorust_pool6_hostdiag_20261027_02`.
- Output: `outputs/splendorust_pool6_hostdiag_20261027_02`.
- Freeze SHA256: `49677386c480a0b5fdcd4d37b824065cf819f1559d7178471331a5ecca28cff1`.
- Implementation review SHA256: `3beb7ffacf29a92828ffaa27274bcc584f209d954b5bf7323aaf799bc13e9359`.
- Prelaunch review SHA256: `9ae33827c055c7704ff3352090fe71e24fbcdd4d357b435cc12eeddc1c61fcad`.
- Terminal raw evidence: `records/0193.json`, `host_observations/0193.json`,
  `measurement_stop.json`, `failure.json`.
- Verification: `host_observer_forensic_audit.json`, `scientific_audit.json`,
  `verification_summary.json`, `completed_evidence_seal.json`, `completion.json`.
- Actual exits: `execution.json`, `watchdog_execution.json`, `guardian_os_exit.json`.
- Independent readback: `postrun/independent_failure_review.json` and
  `postrun/attempt_preservation_inventory.json`.
- Independent review SHA256: `c9f2d5d0a4709a41a1dca8435c17f70ae59f399f1c3d204ab00671d525073b8c`.
- Previous attempt: `outputs/splendorust_pool6_hostdiag_20261027_01`, unchanged.
