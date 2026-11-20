# Experiment157 resumed: worker registration correction

The owner explicitly authorized resuming and asked to be told about external
blockers. The prepared I/O recovery ran at14:08:17 Paris,13September2026, PID35548.
It preserved the37 existing shards and exported the other763 successfully in
44.12s. All800games/24,342roots are sealed in export_lock.json. The progress-file
Windows error did not recur.

The export audit then stopped before any worker result: Python could not pickle
source_game because the dynamically imported module recovery157_audit was not
registered in sys.modules. This is an internal recovery-wrapper bug, not an
external problem for the owner to fix. The original preparation test dispatched
a wrapper function, so it did not exercise this exact callable/import path.
Both failure records and every export remain preserved; no fit/game was started.

The correction is isolated in
scripts/analysis/recover_fast_selector_transfer_157_audit_import.py. It imports
the original I/O recovery unchanged and registers the exact frozen verifier
module in sys.modules in the parent and Windows-spawned children. Independent
audit subprocesses use this same entry point. No scientific code, optimizer,
source, split, schedule, selection or gate changes. No export is rerun. Original
I/O recovery, frozen seven-file implementation and all parent files stay intact.

Validation used actual spawned processes with the exact audit.source_game
callable that failed. Two complete source games (155game0 and156game0) passed,
and all800 export hashes were verified. Pickle roundtrips also verify callable
identity for source_game, replay_fit and replay_game. Frozen experiment identity
was checked before and after this probe. The original1,272 full tests and17
focused/recovery checks remain preserved; the added validation specifically
exercises the missing multiprocessing path rather than changing the science.

Current harness SHA:
eb59720509fba396382fc1a7a6dd8791de0e249af3f62cd4f6c5f095112d99e1.
Current plan SHA:
c1ea532af9fc730f86524bd96cc2a895031ba74366686972ef1c08d052df5e48.
Frozen experiment binding remains:
3db2246c23aa6e9692a604dd95aa2dc40934e6c024b7eb4aa7fd60474d1c9122.

Under the owner's existing resume authorization, the corrected harness started
hidden at14:12:25 Paris, PID44764. Audit subprocess49840 and eight workers were
confirmed active, with45/800 full source audits completed at14:12:43 and no new
failure/stderr. Current records are under
outputs/fast_selector_transfer_157/recovery_audit_import: launch.json, plan.json,
stdout.txt/stderr.txt, then complete.json or failure.json. Earlier failure files
under the output root and recovery_export_io are historical.

The remaining fixed pipeline is full export audit, six fits/full bitwise fit
replay, sealed selection,48off-panel games/full audit,4,800main games/full audit
and report, then stop. No promotion, automatic successor or scientific retry.

The owner-requested blocker watch is active every15minutes as the thread
heartbeat splendor-157-blocker-watch. It stays quiet while healthy; notifies on
new blockers/unexpected exits/actionable stalls or fully verified completion;
then pauses on failure requiring owner action or completion. It distinguishes
internal code failures from external factors and must not restart work itself.
Read append-only logs/immutable locks for status. If reading mutable progress is
necessary, use FileShare.ReadWrite | FileShare.Delete and close immediately.
