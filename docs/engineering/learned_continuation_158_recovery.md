# Experiment158 main replay recovery

Owner authorization on2026-09-14: "yes please recover". All402 games finished,
but the wrapper, pipeline and auditors disappeared with19 immutable replay
certificates, no failure record and no exit record. The cause is unestablished.
The main result is provisional until the full independent audit completes.

Use the isolated `scripts/analysis/recover_learned_continuation_158.py` harness.
Do not edit any of the seven frozen158 files or any ancestor/checkpoint, game,
root, result, original log, or existing replay. No game collection, training,
new scientific choice or successor is involved. Scheduled monitoring remains
disabled by owner; recovery does not re-enable it.

Preparation takes all four original launcher/run/collection/audit kernel locks
to exclude live work. Validate the frozen identity and original preflight/launch
chain, all402 game hashes and24,496 root hashes. Bind the already observed
result and game lock to fixed SHA256 constants. Bind the19 existing replay
hashes (indices0..17 and22) to their independently recorded digest. Compare
their reconstructions with the saved observations and initial-state hashes.
Seal the recovery script, this document, tests and test evidence in a new plan.

Resume repeats these checks, preserves the19 certificates, and submits only
the383 missing whole-game replays to the unchanged frozen `audit_game` callable.
Each incomplete game restarts its audit from the beginning because no partial
replay certificate was saved. No real game or completed audit is repeated.
Eight spawned workers use the unchanged model initializer and one Torch thread.
Top-level module registration runs in the parent and Windows children. Test the
actual recovery CLI with one preserved and one missing tiny fixture replay.

Each newly completed certificate uses the original immutable format/path.
Progress and failure records are separate under `recovery_main_replay`; original
logs and mutable experiment progress files are untouched. A new failure stops
this attempt, preserves available evidence and requires review; no auto-retry.
Launch hidden with a detached process group, redirecting stdout/stderr into the
recovery folder. Record launch identity and actual recovery PID. Preserve all
original records of the interrupted run.

Closure requires exactly402 certificates, paired initial-state equality, exact
root coverage, comparison with both freshly reconstructed observations and the
saved observations, independent fixed bootstrap/gates and operational summaries.
Recheck every game/root hash, all prior replay hashes, the complete frozen and
recovery identities, and each new replay payload before sealing `main_audit.json`,
`report.md`, `final_status.json` and a recovery completion record. Do this on
either strength outcome. Never turn a failed statistical gate into a pass.

Audit provenance distinguishes19 reused certificates from383 new replays. Wall
time sums the previous logged audit segment and recovery, excluding downtime;
worker CPU sums complete certificates and excludes work lost at interruption.
The final experiment elapsed time includes downtime. These operational caveats
do not alter any gameplay statistic, scientific threshold or inference method.
