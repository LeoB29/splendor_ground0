# Experiment157 export I/O stop and prepared recovery

At13:58:36 Paris on13September2026 the frozen pipeline stopped before fitting
or any experiment game. Windows denied atomic replacement of progress.json:
PermissionError WinError5 at old.atomic_write -> pathlib.replace. This happened
during a status read and is consistent with a transient file-sharing conflict;
the precise handle owner was not captured. No numerical/model/data failure was
reported. Launcher45924/pipeline50700 exited1 and are stopped.

The seven scientific files remain unchanged, binding
3db2246c23aa6e9692a604dd95aa2dc40934e6c024b7eb4aa7fd60474d1c9122.
All1,272 full repository tests passed in418.57s before freeze, evidence SHA
96800d5f09619c9f2656769873bed2e5fe7a663803519167f3c1d514b674566b.
The immutable failure/log/exit/freeze remain in place. Both original progress
files are additionally copied under recovery_export_io before any resume.

37 per-game Torch export shards survived. Each has now passed the frozen
independent source/feature/label/private-input audit; their hashes and the
original failure evidence are sealed in
outputs/fast_selector_transfer_157/recovery_export_io/plan.json.763 games remain
to export. There is no export_lock, fit, selected model, preflight or main arena.

Prepared recovery: scripts/analysis/recover_fast_selector_transfer_157.py,
SHA4352300d3a5c441ee37e67e55b46ff027e5f33f765f11880cb8b72f8c43e4cfd.
It verifies the exact frozen identity/failure/37shards, exports only the missing
games, then executes the same complete export audit, six fits/complete fit audit,
selection,48preflight games/full audit and4,800main games/full audit/report.
The original37 files are preserved, and the full export audit checks them again.

Only157 mutable progress.json/audit_progress.json writes receive bounded
0.1-second retries for Windows errors5/32, up to5seconds. Scientific/immutable
record writes, algorithms, optimizer, source labels, splits, schedules, model
selection and gates are unchanged. A further failure stops; no fit/game is
retried or replaced. The external recovery harness is hash-bound separately
and never changes the original manifest or code. Recovery-only tests plus all
14focused157 tests passed together:17passed in6.68s. They exercise retry scope,
exhaustion, non-retriable errors and the authorization guard, plus full small
fit/source/game verification. Preparation independently audited all37shards.

No resume has been launched. The approved recommendation requires an
"identical explicitly authorized crash resume"; the frozen protocol similarly
requires explicit review/authorization and a separate recovery implementation.
That implementation and review are ready. After explicit owner authorization,
launch Python hidden with:

    python -u scripts/analysis/recover_fast_selector_transfer_157.py resume --owner-authorized

Use new exclusive stdout/stderr paths under recovery_export_io. The harness
records its PID and plan binding in launch.json, then complete.json or failure.json.
Do not rerun the original start/launcher or delete its failure. No automatic
promotion, experiment replacement, parameter change or successor is authorized.
