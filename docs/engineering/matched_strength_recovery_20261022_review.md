# Review of the authorized amended recovery

22 October 2026. Scope: the single disclosed retry of game 59, ply 43,
completion of that game, all prescribed combined-game replay checks, and a
separately labelled amended analysis. The owner selected this option explicitly.
No original timeout, completed request or failed audit may be retried.

## Evidence and interpretation

The read-only interruption review passed all 31,853 original frozen inputs and
inventoried 28,702 original output files, totaling 3,597,091,574 bytes. The
interrupted worker has only its ready packet, no accepted response, no durable
request-start clock, and an unfinished compressed continuation trace. Its
absence of a complete response does not establish a 120-second timeout. The
old attempt remains incomplete; no old clock or cleanup receipt is invented.

The genuine game 39, ply 7 timeout is different: the trusted complete-line
receipt is 121.6558109 seconds for a 120-second deadline. The late packet and
trace are preserved. Approximately 96.06 sampled CPU seconds do not negate the
wall-clock miss or establish its cause. Retaining the loss implements the
prospectively frozen delivered-player comparison; it is not evidence that
search computation alone exceeded 120 seconds or that the chosen move was weak.

## Implementation review

A separate reviewer inspected the new common, memory, worker, opponent wrapper,
runner, analyzer and administrative launcher. The scientific and provenance
review found no remaining material issue after the corrections below.

- Exact referee reconstruction from the original seed checks all accepted
  actions and successors, including the final-game prefix through ply 42.
  Metadata imports are byte-identical and bound to their original locations.
  Original workers and traces remain referenced in place.
- Only `games/g059/t0043` can use the explicit extra attempt. Its old partial
  search is never read as decision guidance. Later new requests retain the same
  players, streams, CPU settings and 120-second trusted receipt deadline.
- Original and recovery wall and monotonic clocks all apply. New execution is
  capped at twelve hours including startup checks. Playing admission separately
  protects eighteen original hours and six recovery hours. Original RAM charges
  carry forward. Source and recovery evidence share the eight-GiB ceiling.
- Every new supervisor, numerical and final-analysis entry point authenticates corrected
  runtime sources. The frozen local numerical functions and qualified opponent
  initialization/handling remain unchanged. This is not a general migration of
  historical training entry points or data.
- The analyzer tracks the origin of every old/new receipt, the interrupted
  worker and its RAM gate, the retained timeout, combined decision and replay
  inventories, both execution segments, and inherited memory accounting.
  Statistics open only after the combined scientific checks. Analysis retains
  the original paired bootstrap and all three fits.
- Successful acceptance is `accepted_amended`, never original primary
  acceptance. The unresolved original final-game point-score range has width
  at most 1/60 and is explicitly not a confidence interval or an imputed result.

## Resolved operational findings

The launcher now requires a job-isolated WMI start and checks the actual
qualified executable, command, full affinity and Windows job membership. A
harmless final probe (03) passed both self and independent external identity
checks, remained outside any job for fifteen seconds after the invoking shell
returned, and exited. The receipt binds the tested launcher hash. Earlier
probes remain preserved. This supports caller-job isolation; it does not
establish the cause of the earlier disappearance or protect against reboot.

Emergency cleanup authenticates the parent creation time before PID-based
suspension or descendant discovery, and every captured descendant before
termination. Unknown or reused parent identities use only the owned process
handle. Cleanup errors and process-creation failure are preserved in the outer
execution receipt. There is no automatic relaunch.

Finalization now checks all four clocks between file hashes and inventory
entries. Before analysis it protects the existing final reserve; afterward it
may spend that reserve while retaining both hard caps. The separate supervisor
also enforces the execution ceilings and closes its authenticated owned tree.

## Verification and launch gate

Focused model-free checks passed: 27 recovery tests, 23 completion tests and
six launcher tests. The analyzer's new and predecessor checks are recorded by
their own test receipt. Preparation must additionally run the complete selected
preflight suite, reconstruct all saved games, rehash every original input and
source artifact, and write the new freeze before any numerical request starts.
The authoritative combined test outcome is the new output's
`preflight_tests.json`; a failed preparation is preserved and cannot silently
admit playing work.

The frozen preparation also binds the user authorization, exact completion
contract, source inventory, launch probe and this review. Production readiness
and launch receipts are written separately after the freeze. Original evidence
must never receive new logs or receipts. No matchup totals were opened in this
review. The larger-data study remains deferred.
