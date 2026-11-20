# Exact inference graph result

25 October 2026. The separate inference graph preserved exact numerical outputs
and complete decisions. It reduced mean whole-decision time by 5.6-6.3% at 64
simulations and 2.6-3.5% at 32. The frozen joint admission rule required at least 5%
in every quota/panel group, so this attempt did **not qualify** and launched no
operational games. Its evidence and final verification completed successfully.

## Execution and correctness

The run took 279.986 seconds (4m40s), from 01:28:52 to 01:33:32 Paris, within the
60-minute cap. Outer exit 0 and owned cleanup passed. Watchdog 17348 and runner 21152
were absent after completion. No restart, retry, replacement or alternative
candidate was used.

The implementation is isolated in `tools/splendorust_inference_opt_20261025`.
It traces the existing tensor forward once and freezes it with
`optimize_numerics=False`. Input construction, validation and output conversion
remain outside the graph. Weights, float 32, one numerical thread, batches1/2/8,
32/64 search quotas, three worlds, depth64, native binaries, corrected rules,
raw reader, optimized sampler and supervision remain unchanged. No model was
trained or quantized. The inherited checkpoint retains its historical origin.

- All 113 prelaunch model-free tests and independent reviews passed.
- All four actual corrected-rule entrypoint suites passed, 52 decisions each,
  before any paired timing.
- All four candidate processes independently passed 11 numerical cases each,
  including the six recorded corrected-face inputs, batch2/8 compositions and
  zero controls. Full output values and float 32 bits matched; the six archived
  eager outputs also matched. Before/after weights and graph identities agreed
  across all candidate instances.
- All 288 paired decisions passed exact scientific checks across 72 position/quota
  groups. Full callback/output hashes, actions and independent corrected
  transitions agreed. All 644 available historical comparisons passed
  (304 at 32,340 at 64).
- Every timed decision met both the 1.000-second deadline and the .700-second
  admission margin. The maximum was .4335077s.
- All eight worker logs, raw receipts, full graph parity evidence, frozen
  sources/runtimes and sealed evidence passed final reread verification.

Peak combined sampled worker/helper RSS was 607,649,792 bytes (579.5MiB).
Completed attempt evidence was 42,004,263 bytes (40.1MiB) at inspection.
The scientific reference and all earlier failed attempts remain preserved.

## Timing gate

The panel contained 24 development and 12 newly reserved saved positions. The
reserved positions excluded 83 previously used main-game pairs and 373 observer
hashes; selection ranked identifiers, not timings or outcomes. Both modes and
quotas ran two reversed paired passes, yielding 288 timed requests.

| Panel | Quota | Eager mean | Graph mean | Mean reduction | Graph p95 | Graph maximum | Admission |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | --- |
| Development | 32 | 212.26ms | 206.77ms | 2.59% | 229.11ms | 269.88ms | Failed 5% gate |
| Reserved | 32 | 207.79ms | 200.50ms | 3.51% | 213.70ms | 217.52ms | Failed 5% gate |
| Development | 64 | 377.69ms | 356.61ms | 5.58% | 390.34ms | 433.51ms | Passed |
| Reserved | 64 | 381.91ms | 357.80ms | 6.31% | 384.31ms | 422.45ms | Passed |

Every group's candidate p95 improved and all margin/correctness conditions
passed. The only negative gates were the 32-simulation mean reductions. They
must not be rounded up, pooled with 64 or waived after observing the results.
The six-game operational branch was correctly skipped. No fresh game, strength
comparison or training was launched; larger-data work remains deferred.

`completed_verified` describes the integrity/completeness of this evidence.
It does not mean the implementation was admitted:
`inference_graph_qualified=false`, `saved_panel_qualified=false`.

## Interpretation and next decision

The result supports a modest, exact-behavior speed improvement at 64. It does
not support the prospectively required joint 32/64 promotion. Exact agreement
on this bounded assessment is strong implementation evidence, not a mathematical
proof over every possible input.

The earlier occasional >.700s delays were not reproduced. This run therefore
does not establish their cause or show that graph execution eliminates them.
Operational admission remains necessary before a new strength comparison.

A reasonable next study is a separately frozen qualification of **graph 64
against the unchanged eager 32 reference**, with new reserved timing positions
and fresh operational pairs. This would deliberately narrow the candidate to
the quota where the present development evidence supports a useful gain. It
must be described as a new decision informed by these results, not a retroactive
pass of this attempt. Do not further tune against the used reserved panel.
That next study has not been prepared, implemented or launched here.

## Evidence

- [Frozen protocol](splendorust_inference_opt_20261025.md)
- [Freeze](../../outputs/splendorust_inference_opt_20261025_01/freeze.json),
  SHA256 `2339f755ce171811d9474a53d53881e4d8e1476baa60f91517e778b903e7264a`
- [Exact graph parity](../../outputs/splendorust_inference_opt_20261025_01/all_graph_parity.json)
- [Actual corrected entrypoints](../../outputs/splendorust_inference_opt_20261025_01/all_entrypoint_checks.json)
- [Paired timing and scientific checks](../../outputs/splendorust_inference_opt_20261025_01/qualification_summary.json)
- [Qualification outcome](../../outputs/splendorust_inference_opt_20261025_01/qualification_outcome.json)
- [Final verification](../../outputs/splendorust_inference_opt_20261025_01/verification_summary.json)
- [Completion](../../outputs/splendorust_inference_opt_20261025_01/completion.json)
  and [owned cleanup](../../outputs/splendorust_inference_opt_20261025_01/execution.json)
