# Saved-decision diagnostic: checker repair and second bounded attempt

27 October 2026. The user authorizes fixing both checker defects in a separate
version, adding tests from the saved failure, freezing and running one new bounded
attempt. The failed first attempt stays incomplete and immutable. This document
supplements the [prepared design](splendorust_decision_quality_20261027.md) and
[first execution protocol](splendorust_decision_quality_execution_20261027.md).
Their scientific panel, conditions, ordering and limits remain unchanged.

## Separate implementation and preservation

New source: `tools/splendorust_decision_quality_20261027_v2`.
New evidence: `outputs/splendorust_decision_quality_20261027_02`.
Reuse the exact native build from
`outputs/splendorust_decision_quality_build_20261027_01`; do not recompile it.
Optimized64, unchanged32, checkpoint, precision, search parameters, three worlds,
seeds, action support, response clocks and supervision placement remain unchanged.
The same research profile contents are required byte-for-byte after canonical
serialization. Source-module and output-directory identities change separately.

The first freeze SHA256 is
`a07c1d54dbbf9a67eb6a202e5c6d08717949ef50de8bcfee5413581e8d4d20b0`.
Its 1,776 frozen inputs and all 145 attempt files, including the three saved
postrun supplements, are authenticated before repair and bound into the new
freeze. No old receipt is upgraded, replaced or accepted retroactively. The new
attempt uses fresh owners and clocks; it is not a continuation of an old move.

## Exact checker changes

The native `SearchTargets.network_value` field remains unchanged in raw evidence.
Under the frozen dynamic-FPU search, it contains the root node's updated running
value after search. It is not the initial neural-network prediction. The new
analysis calls it `post_search_root_node_value` and records its upstream meaning
as `post_search_dynamic_fpu_running_mean`.

Derive the separately named `initial_network_win_credit` from the first recorded
root callback using `(float64(output83[81]) + 1) / 2`, matching the native initial
conversion for the acting-player root. Keep this initial estimate distinct from
the visit-weighted root mean and the selected-edge mean.

For N completed root visits, check the updated node value against
`(initial_network_win_credit + N * reported_root_value) / (N + 1)`.
The declared absolute rounding bound is `2^-23 + gamma_(6*N+16)`, where
`gamma_k = (k * 2^-53) / (1 - k * 2^-53)` and N is at most 1,024.
The first term conservatively covers the two separately rounded float32 reports
on [0,1]; the second covers positive binary64 recurrence, summation and
reconstruction. Retain the independent weighted-Q/root-mean consistency check,
legal support, visit counts, missing-Q rules and exact callback/bit checks.
No tolerance is introduced for decisions or scientific trajectory equality.

The original receiver may retain a rejected result with `status=illegal` and
`accepted=false`. Preserve and authenticate that payload as failure evidence.
Do not require its deletion, reclassify it as successful, expose it as an accepted
action/trace, or use it in scientific comparisons. Other failures must obey the
original receiver's status-specific contract. Deadlines, raw-byte bindings and
first-failure stopping remain unchanged. A non-deadline checker or infrastructure
failure leaves the new attempt incomplete even if its saved prefix can be audited.

Regression tests must use the actual saved frozen/OFF/rejected-ON responses,
including their original raw transport and timing records. Test incorrect
initial/post-search substitutions, corrupt values and malformed rejection
evidence as well as successful preservation. A regression accepting the saved
response under the corrected scientific checker does not change its historical
rejected status or qualify the new actual runtime by itself.

## Unchanged execution bounds

| Item | Fixed allowance or inventory |
|---|---|
| Global clock |7,200 seconds, non-resettable |
| Stages |Qualification 900s; diagnostic 4,500s; finalization 1,800s |
| Collection cutoff |Elapsed 5,400s |
| Frozen reproduction |1.000s external, 0.900s internal |
| Research requests |30.000s external, 29.000s internal |
| Panel |Same 24 roots; same 156 requests and exact seeds/order |
| Qualification |52 frozen + 52 research + 19 parity decisions |
| Worker lifecycles |Four serial trees, at most one resident tree |
| Graph fixtures |11 per tree, 44 total |
| Available RAM |2GiB, initially stable 60s; at most 300s total admission waiting |
| Worker-tree RSS |2GiB aggregate |
| Free disk |7GiB initial, 5GiB maintained |
| New attempt evidence |1GiB total; 512MiB reserved for finalization |
| Logical streams |Frozen 16MiB; research 128MiB per stream |
| Responses |Frozen child 256KiB; research 4MiB; parent requests 1MiB |
| Placement |Normal priority; player/helpers CPU0; supervisors CPUs2–19 |

Startup, waiting, qualification, shutdown and verification consume these clocks.
The six predetermined pilot roots remain a cost-only gate: measured remaining
time and evidence projections, including closure/verification, must fit with the
unchanged 1.5 factor before expansion. Decisions and values cannot influence that
gate. Stop at the first failure; no automatic retry, replacement or extension.

## Launch and acceptance

Model-free tests and independent implementation review precede the new execution
freeze. Independent readback must confirm preserved prior evidence, identical
players/runtime/native/profile contents, unchanged qualification and request
inventories, and every bound. Every new evaluator worker retains the corrected
rules startup guard and actual end-to-end counterexamples. Actual model work
starts only inside the newly frozen guardian clock.

Re-run the complete qualification on the actual new entry point, including exact
frozen/OFF/ON outputs at the prescribed quotas and seeds, before any panel request.
Audit all completed decisions, reproductions, raw logs, clocks, placement, source
preservation and owned exits. Completion still requires the original three
observed clean owner exits and independent finite receipt review. A preserved
failure cannot become completed qualification or an accepted full diagnostic.

Analysis retains the fixed original64 versus1024-A action pair and B/C support
checks. These remain same-network search estimates and random-seed stability
checks, not independent quality targets or observed game returns. No population
blunder rate, Elo, playing-strength or promotion claim follows from this panel.
Recommend at most one next hypothesis; any successor needs a separately frozen
fresh whole-game test at one second. External-candidate testing stays paused,
larger-data work stays deferred, and no games, training or model changes are part
of this attempt.
