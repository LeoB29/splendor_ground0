# Review of the remaining games and memory waiting

20 October 2026. This records the review required before launching the second
authorized recovery under `live_game_operational_recovery_20261020_02.md`.
It does not record completion of the operational games or a strength result.

The primary agent reviewed the new parent orchestration, resource guard,
worker interface, independent analyzer and completion helper. The adapter,
analyzer and coordinator authors cross-checked their schemas, provenance
boundaries and admission interfaces. The exact reviewed files are recorded
in the preparation directory before the production freeze; the combined
preflight must pass before that freeze is written.

The adapter preserves the first sixteen completed game records and all four
timeout forfeits. Its continuation reconstructs game 21 through ply 10,
verifies that the old ply-11 request never initialized, and schedules only
that game's suffix followed by game 23. Original receipts and traces retain
their actual execution origins. Nested import mappings authenticate older
records copied through the first recovery. The tests reject changed source
identities, altered prefixes, initialized requests, overwritten copies,
outside trace paths and unintended schedule expansion.

Syntax-tree comparisons preserve the original worker's numerical decision
implementation and the common observation, stream, sampling and proof
functions. The new worker only changes output binding and authenticated
audit trace admission from the two historical evidence trees. Local writes
remain restricted to the new output. A Windows symlink fixture is skipped
when the host lacks the required privilege; ordinary resolved-path containment
and traversal rejection are still tested.

Memory admission happens before worker creation. Initial admission and recovery
after a low reading require thirteen good samples over two minutes. Ordinary
healthy boundaries use one reading. The helper preserves a JSON receipt and
reading log for every gate, independently limits each episode and cumulative
charge, and keeps disk, evidence, both global clocks and work reserves active.
Fast checks, stability observation and low-RAM waits all contribute to the
recorded charge. An active move or replay receives its original uninterrupted
deadline and cannot be retried through this mechanism.

Review corrected a final-flush edge before freeze: a passed memory receipt is
now assigned only after the reading log is flushed, the resource reserve is
rechecked, and a fresh charge remains inside both waiting limits. An exception
always leaves a held receipt. Tests cover delayed/failed flushes, low physical
RAM or commit, stability resets, episode/cumulative exhaustion, disk failures,
clock adjustments, nonrefunded guard overhead, distinct audit/analysis gates,
initial-admission ordering and rejection of duplicate attempts.

Review also corrected inherited metadata that still described the first
recovery's 24-hour cap. Both runtime and the new freeze explicitly use 43,200
seconds. The original source freezes retain their original limits. Preparation
and production require the original full supervisor CPU eligibility; numerical
workers still use CPU 0. Preparation tests and the completion helper use CPU 2.

The independent analyzer uses three explicit receipt/process/clock contexts.
It authenticates both historical holds and source inventories, both exact
continuation boundaries, byte-identical final games, all actual transitions
and endings internally, and the one combined audit manifest across all
eighteen games. It checks new memory receipts and logs independently, including
serial cumulative charges and each worker's admission. The completion helper
creates the analyzer's own admission before starting it.

The completion helper binds its source, analyzer, memory helper, prospective
plan, new freeze and exact runner identity. It waits through legitimate
between-worker memory pauses without treating them as a new failure. After
successful collection and replay it starts one independently checked analysis
within the same new cap, preserves any failure, authenticates owned process
cleanup and validates final artifact bindings. The report retains all four
earlier timeouts, all three segment costs, memory-wait charges and the actual
replay coverage. It does not call verification completion failure-free timing.

The preliminary two-minute RAM observation measured 4.189–4.543 GiB available
physical memory and at least 35.490 GiB available commit. This is a saved
preparation observation, not a guarantee of future headroom; initial in-run
admission is still mandatory. A real-source control check confirmed both old
supervisors/coordinators closed, no prior game-audit or final-analysis attempt,
and the new checker's agreement with the immediate source hold and unstarted
request. It loaded no player model.

Synthetic test collection initially rejected a reserved fixture parameter name.
The failed source copies and diagnostic remain in the preparation directory;
the corrected test suite subsequently passed. Final combined test results are
recorded in `preflight_tests.json`. No failed player request, completed game,
historical source or prior evidence file is changed by this preparation.
