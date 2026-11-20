# Saved-evidence collector verification recovery

6 November 2026. This is a technical correction to finish the already authorized
collector comparison. It creates no games, model calls, targets or training.

The original frozen study is
`outputs/splendorust_gpu_collector_20261105_01`, plan SHA
`45f5aaf9bd7d9a5dc485e1eaaa39c5f908edda887b332e83cc71811dd60b3563`.
Its controller actually exited zero (retained session79670, chunk72c5a5).
All64games,192selected replays and four saved batch audits passed. The original
final auditor actually exited one (session42090, chunk8916e2); its failure remains
preserved in the original report directory. Original final acceptance is absent.

## Reproduced defect and exact correction

One bounded re-audit of the first CPU cell reproduced precisely one difference:
the occupancy histogram has integer keys in memory and string keys after JSON
serialization. The example is `{1:412077}` versus `{"1":412077}`. Every count and
other proof field matches exactly. The full diagnostic and failed audit remain
preserved and are bound into the new freeze.

The separate verifier privately loads the exact frozen original auditor. Change
only the derived `occupied_rows_histogram` keys produced by `JournalIndex.metrics`
to their JSON string representation, after validating integer occupancies1..8
and positive integer counts. Reject booleans, invalid keys or collisions. Retain
all original proof equality, source, journal, rule, target, replay, ownership,
clock, sample and threshold checks. No numeric tolerance, ignored field or broad
JSON normalization is introduced. Original audit code and public module objects
remain unchanged. Original saved cost receipts retain their original auditor
binding; the new verifier identity is separately bound in recovered acceptance.

## Separate freeze and bounded execution

Source: `tools/splendorust_gpu_collector_audit_20261106`.
Evidence: `outputs/splendorust_gpu_collector_audit_20261106_01`.
Bind the original plan, checker, actual controller exit, failed final-audit
receipt, minimal-difference diagnosis, correction source, passed tests and
independent review. Test the saved discrepancy, malformed keys/counts, exact
rejection of changes to counts and other fields, private-module isolation and
both clocks. Review and freeze before one saved-only recovery launch.

The new audit has a60-minute actual dual-clock cap, within the original study's
12-hour absolute cap. Neither origin is reset or extended. Check corrected runtime
sources once at startup, before any evidence interpretation. No models or new games
may be invoked, no frozen files rewritten, and no failed attempt removed. There
is no RAM-reserve or forecast admission gate. Preserve any new failure and stop.

The independent audit rechecks all original cells and qualification, source
preservation and exact owned-process absence. Save the actual new auditor launch,
exit and cleanup receipts. Root must observe actual retained process exit and
review the final report and chart before closeout. A completion file alone is
insufficient. Report successful recovery explicitly as recovered verification,
without claiming that the original final auditor passed. Include recovery work
in the total study account; the original per-cell throughput clocks stay fixed.

## Scientific decision remains unchanged

Keep the predeclared requirement of at least5% higher verified GPU games/hour in
each of the two blocks. The completed comparison did not meet that consistency
criterion; the checker correction cannot change that result. Original weights,
CPU8/GPU16 search settings, corrected rules, all64games and every recorded time
remain unchanged. Larger-data work, training, external testing and six-world work
remain deferred. No automatic next experiment is authorized by this recovery.
