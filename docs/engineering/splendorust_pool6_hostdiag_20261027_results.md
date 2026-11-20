# Host-interference diagnostic: incomplete observer qualification

27 October 2026. The separately frozen attempt stopped at request 29 of the
72-request overhead stage because the new observer rejected a context-switch
aggregate decrease. The 144-request full diagnostic did not start. This is an
incomplete measurement attempt, not a passed overhead gate, an overhead-speed
rejection, a player admission or a strength result. No retry was launched.

The original completion record remains `incomplete`. Independent saved-evidence
review verified the failed prefix without accepting experiment completion,
observer overhead or player admission. No experiment remains running.

## Planned and completed work

The two unchanged players were optimized64 with three worlds and the six-world
candidate, each using the same pretrained network, precision and 64 simulations.
The original one-second complete-response deadline and 0.900-second internal
guard remained unchanged. No fresh game or training job ran.

All 92 aggregate model-free checks passed before the freeze. Both actual workers
then passed their corrected-entrypoint suites: 104 scientific startup decisions
and 22 exact graph cases. The study admitted work after 60.172 seconds of stable
RAM; the recorded minimum during that initial wait was 7,708,938,240 bytes,
above the predeclared 4 GiB headroom. No recovery wait was needed.

The overhead stage collected 29 model responses on the fixed controls: 18 with
the new observer off and 11 with it on. The first 28 observer lifecycle records
passed; the final enabled record contained the preserved measurement failure.
All 29 model responses were accepted by the unchanged clock, with no timeout
or 700 ms crossing:

| Player | Collected control responses | Maximum receipt time | Timeouts |
| --- | ---: | ---: | ---: |
| Optimized64, three worlds | 15 | 447.539 ms | 0 |
| Six-world candidate | 14 | 461.746 ms | 0 |

The counterbalanced 72-request schedule was not completed, so no overhead
comparison can be accepted. The nine selected slow roots were reserved for
the subsequent full diagnostic and were not reached. These partial control
timings do not explain the original slow moves.

## Failure and cause

The terminal request was zero-based sequence 28, case `g2_p34`, using optimized64
with the observer enabled. The model's accepted response arrived in 374.3033 ms.
Its observer window retained eight native snapshots, but context-switch
validation raised `Context-switch counter reset`.

The decrease belonged to the idle `frozen:reference` helper, PID 23768, declared
creation time 1793119540.4607868: the aggregate changed from 1,388 to 1,387.
The process creation FILETIME, 134375931404607869, remained identical throughout
the eight snapshots. This was not evidence of PID replacement.

The observer assumed that psutil's Windows process context-switch aggregate was
a monotonic lifetime counter. The implementation instead sums counts over the
threads present in each snapshot. An exiting thread can remove its accumulated
count from the next aggregate. The installed wrapper calls that native
implementation; see [psutil 5.9.0's source](https://github.com/giampaolo/psutil/blob/release-5.9.0/psutil/arch/windows/process_info.c#L697-L698).

Saved worker telemetry also reports two threads for that helper on request 16
and one on request 17. That supports thread disappearance, but the evidence
contains no per-thread identities or counters identifying the exact departing
thread and contribution. The established defect is the monotonic-aggregate
assumption; do not claim a fully reconstructed thread history.

The stop obeyed the frozen failure policy. The failed observer receipt retains
the original raw counts and error. Its derived summary/context delta is null,
and `placement_end` is null because the counter check raised before that step.
No missing measurement has been filled in or relabelled valid.

## Verification and cleanup limits

The runner's finalization reached scientific-prefix and closed-worker checks,
then rejected the same invalid observer window using its strict live-acceptance
validator. It therefore did not produce a completed evidence seal or successful
final verification summary. This is a second manifestation of the audit-policy
coupling, not another model failure.

The separate failure audit verified all 11,741 frozen inputs, 96 external runtime
files and 3,652 prior-attempt artifacts. It verified the 104 startup decisions
and all 29 saved model decisions (133 scientific decisions total), 33 parent
requests, 22 graph cases, 15 exact historical comparisons and 11 exact repeat
comparisons. It accepted 28 observer lifecycle records as valid (18 off and ten
on), and verified preservation of the terminal invalid record without treating
it as a usable measurement. All 15 recorded owned process identities are absent.

The sampler stopped and joined, with its process handles closed and the original
counter error retained. The actual runner, watchdog and guardian all exited
with code 1, as expected for the incomplete attempt. The owning PowerShell
session closed. Charged run time was 133.704 seconds, approximately 2 minutes
14 seconds, within the two-hour cap. The separate failure review preserves
these nonzero exits and reports `verified_incomplete_measurement_failure`.

The original admission02 and all its predecessors remain preserved. Initial
runner test-fixture failures are also retained in the new attempt's prelaunch
logs; the final combined 92-test run passed. No frozen source was repaired after
launch, and no new model requests were issued after the stop.

The review also records one optional missing-path metadata read in
`postrun/reviewer_reader_note_01.json`; this read-only note had no effect on the
experiment. The review executed no model, game or test rerun.

## What this result establishes

The collected player responses met their deadlines, but the observer did not
qualify. We found a concrete measurement-semantics error before using its
results to explain the rare live-game delays. The incomplete OFF/ON schedule
cannot establish observer overhead, and no conclusion about the cause of the
original nine slow moves follows from this attempt.

## Recommended next step

Make a separate observer/checker version with two narrow corrections:

1. Preserve the raw live-thread aggregates and their signed net change, but
   mark the true interval context-switch total unavailable. Do not clamp a
   negative change to zero or treat a positive change as an exact event total.
   Keep independently valid native CPU, memory, page-fault and I/O measurements
   usable, while process identity mismatches and unreadable/API failures remain
   fatal. Exact interval switch totals would require a larger thread-aware
   collection design, which is not necessary for this minimal correction.
2. Separate measurement acceptance from forensic verification. Live collection
   still stops on an observer failure. A final audit can verify one terminal
   invalid window's available raw data, bindings and error without accepting its
   measurements or declaring overhead qualified. Explicitly retain absent
   placement evidence and use the separate sampler-close proof for cleanup.

Use this exact saved failure for regression tests, including counter increases,
decreases, unchanged aggregates, corrupted identities and incomplete placement
evidence. Then freeze a new attempt and repeat the overhead qualification before
the full diagnostic. This is a recommendation only; no replacement attempt has
been implemented or launched.

## The broader picture

The research question remains whether six hidden-card worlds make better use of
the same 64 search simulations than optimized64's three worlds, within one
second per move. Earlier saved-position evidence suggested more stable choices,
but fresh live games failed the 700 ms safety margin for both players. The host
diagnostic is intended to separate genuine computational cost from interference
on this machine before committing to a strength comparison.

This attempt exposed a flaw in that measurement tool. Correcting it preserves
the value of the next diagnostic; it does not justify changing either player,
loosening the deadline or launching the strength benchmark. Optimized64 remains
frozen, external testing stays paused, and larger-data/training work stays
deferred.

## Evidence

- Protocol: `docs/engineering/splendorust_pool6_hostdiag_20261027.md`.
- Frozen source: `tools/splendorust_pool6_hostdiag_20261027`.
- Evidence: `outputs/splendorust_pool6_hostdiag_20261027_01`.
- Freeze SHA256: `424fab8fcfc079a01410ec093d76a4d7f2478144fdd32e95a6503a37d29d8c47`.
- Exact prelaunch review SHA256: `71b42d2538cd9d94a13b840f2a47ecda891526e2b8e8f5037ef9eb504fb8b70d`.
- Failure: `records/0028.json`, `host_observations/0028.json`, `failure.json`,
  `finalization_failure.json`, `completion.json`.
- Cleanup/actual exits: `host_observer_close.json`, `execution.json`,
  `watchdog_execution.json`, `guardian_os_exit.json`.
- Independent failure review: `postrun/independent_failure_review.json`, SHA256
  `c4d832791aaeb061bf1b2938bf0d078cd637408fc7a16487c184bf1fcf2eff61`.
- Separate preservation inventory: `postrun/attempt_preservation_inventory.json`.
