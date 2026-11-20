# Experiment161 operational recovery01

Owner explicitly authorized "repair and restart" after the01:11Paris status on
2026-09-17. Scientific binding remains
`f8f1fda4884cdcde518ad079430528b95abaf19e5b47946630dd5c4e5f22eeeb`.
Preserve all frozen sources, protocol, tests, checkpoints,48pilot games, original
failure and original logs. No training or pilot game collection is repeated.

## Fault and limited repair

All six fits and their full independent bitwise replays passed. All48pilot games
were saved. The supervisor stopped before sealing probe_games_lock.json because
the stage's upstream and selected references both name the same selected_lock.json.
The strict file verifier rejected that valid duplicate reference. No game or model
integrity error was involved; full pilot replay was still pending.

A separately hashed recovery module installs a supervisor-only adapter around
the frozen runner's file verifier for the duration of recovery. It accepts ONLY
two exactly equal `{path,sha256}` references to the exact selected_lock.json path,
still verifying the supplied hash. Every other input retains the original strict
duplicate rejection and hashing. The adapter is restored on exit; no frozen file
or worker scientific function is edited. This exception also applies to the
original verify_stage call during preflight and final closure, which would
otherwise encounter the same duplicate. There is no global duplicate suppression.

## Recovery evidence and boundary

Validate unchanged frozen identity, all completed export/fit/audit locks, every
checkpoint/validation file, all three matched batch streams and the nine selected
models. Bind the48existing game files to the hashes recorded before recovery in
manual_reviews/status_20260917_failure_01.json. Require exact frozen schedules,
valid headers/natural endings and expected creation order. Do not inspect aggregate
outcomes. No downstream audit, main run or results may already exist. Acquire the
original run.lock so another supervisor cannot run concurrently.

Freeze this amendment, recovery code/tests and passing focused test evidence in
recovery_01/manifest.json. Inventory and pin all completed scientific artifacts,
worker logs and original supervisor logs; exclude ephemeral OS lockfiles and
preparation-only artifacts. Verify this retained inventory before launch and again
after final closure. Recovery has its own immutable launch/completion/failure
records and append logs. The old failure.json remains as historical evidence.

Reconstruct only the missing48-game collection lock using the pinned original
game hashes and original launch reference. Its elapsed_seconds is a documented
upper bound from the original stage launch timestamp to original failure time,
15.974423seconds, excluding all interruption/recovery downtime. The monotonic
timer was lost on supervisor exit. This conservative wall-time estimate is used
only for the operational runtime forecast; no statistical gate changes.

Continue with full independent replay of all48saved pilot games, unchanged
integrity/storage preflight, all4,800fixed main games, all4,800full independent
replays and unchanged paired statistics/final report. All seeds, selected models,
comparisons, sample sizes, outcomes, gates and stopping rules remain frozen.

## Verification and process behavior

New regression tests cover the original run_jobs post-collection failure and
verify_stage/preflight dependency paths, strict rejection of unrelated/conflicting/
tampered duplicates, restoration of the scoped adapter, saved-game hash/boundary
rejection and continuation order/failure stopping. Run these with the existing161
scientific and actual Windows launcher/worker tests. The original1,403-test suite
passed under the unchanged scientific identity; this operational addition gets
its own targeted test evidence before sealing. No tests or assertions are skipped.

Reuse the tested held launcher, pythonw supervisor and workers, separate append
logs and no-console checks. Launch acknowledgement includes both scientific and
recovery identities. No automatic retry or second recovery; any failure preserves
evidence and stops. No monitoring, model promotion, sample extension or successor.
