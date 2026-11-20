# Optimized64 timing diagnostic: instrumentation gate rejected

25 October 2026. The separately versioned run completed and verified its
overhead assessment. Optimized64 did **not** pass the instrumentation overhead
gate, so the planned 416-request diagnostic panel was not run. This is a valid
negative measurement qualification, not a correctness failure, a timeout, or
a new strength result.

## What ran and what passed

- 110 model-free implementation tests passed before freezing.
- All four actual-player correctness suites passed: baseline and profiled
  graph64 and eager32, with 208 completed correctness decisions in total.
- All four graph instances passed their eleven exact-value and float32-bit
  cases, preserving the previously verified graph and weight identities.
- All 48 overhead requests completed within the original one-second ceiling.
  Full results, legal actions, callback traces and resulting states matched
  across modes/repeats and their accepted historical records.
- Final verification passed for all eight workers: original receipts, actual
  request bytes, raw response bytes, timing sidecars, startup suites, graph
  proofs, logs, cleanup, frozen source/runtime hashes and 304 sealed artifacts.

The owning watchdog exited successfully with no cleanup errors. Elapsed time
was **209.12 seconds (3 minutes 29 seconds)**, from 19:32:53 to 19:36:22 Paris,
within the original 30-minute cap. Evidence occupied approximately 34.51 MiB
within the 128 MiB cap. Peak sampled combined worker/helper RSS was about
534.44 MiB, below the 2 GiB cap. Both supervisor processes were confirmed absent
after completion. No retries, replacement requests, new games or training ran.

## Frozen overhead result

Each row below uses twelve baseline and twelve profiled decisions: six fixed
positions, repeated with reversed mode/order. The p95 calculation is the frozen
linear-interpolation percentile; it is descriptive on this small sample.

| Measure | Optimized64 | Unchanged32 |
| --- | ---: | ---: |
| Baseline mean | 572.18 ms | 370.11 ms |
| Profiled mean | 610.30 ms | 388.12 ms |
| Observed added mean | **38.12 ms — fails 25 ms limit** | 18.01 ms — passes |
| Profiled/baseline mean | 1.0666 — passes 1.10 limit | 1.0487 — passes |
| Baseline p95 | 724.80 ms | 476.92 ms |
| Profiled p95 | 840.28 ms | 470.31 ms |
| Observed added p95 | **115.47 ms — fails 50 ms limit** | −6.61 ms — passes |
| Largest profiled response | 871.83 ms | 472.39 ms |
| Instrumentation qualified | **No** | Yes |

Both quotas were required to pass. The runner therefore skipped the full
104-position, 416-request panel and completed the scheduled verification of
the negative overhead result. The 0.700-second engineering margin remained
descriptive; it was not silently made a new deadline or rejection criterion.

## Interpretation

Correctness is preserved, but these measurements do not qualify the detailed
profiler as sufficiently unobtrusive for the intended optimized64 diagnostic.
The observed differences are not a clean estimate of intrinsic instrumentation
cost: paired timings varied substantially. For optimized64, the first pass's
profiled mean was 19.61 ms faster than baseline; the second was 95.84 ms slower.
Individual paired differences ranged from −74.75 to +186.04 ms. These descriptive
differences were inspected after the frozen gate and do not change its verdict.

That variation is compatible with intermittent execution conditions and
measurement overhead together. It does not prove a particular operating-system,
RAM, reader or inference cause. The profiled control measurements cannot explain
the original two benchmark failures, and no completed full-panel attribution is
available. All original game results and failures remain unchanged.

## Recommended next step

Prepare a **lighter, boundary-only timing observer**, in a separate version,
before attempting another player optimization. Retain a small fixed set of
timestamps around parent dispatch, child input/handler entry and exit, child
pre-write/post-flush, and trusted parent arrival. Remove the detailed per-call
CPU/native/hash instrumentation from this first localization pass. Preserve the
actual raw transport bindings and all original scientific and clock checks.

Qualify that observer against the same uninstrumented players under unchanged
overhead thresholds before using it on the saved case/control panel. Its first
question should be whether a reproduced delay occurs before the child receives
the request, within its decision handler, or during response delivery. Only a
qualified observation locating a delay should motivate a narrower second probe
or a player/runtime change. If the lightweight gate also fails or no stall
appears, report that limitation without reruns or relaxed thresholds inside the
same attempt. This is a recommendation, not a newly frozen or launched study.

## The broader picture

The one-second programme has shown useful strength gains from adding search to
the inherited network, then increasing from 16 to 32 simulations. Exact inference
optimization made the 64-simulation candidate practical enough for the next
comparison. That candidate then scored **64.25% against unchanged32** in the
verified 200-game benchmark, including its two retained deadline losses.

We now have evidence that more search can improve this player, but the remaining
task is to deliver that strength reliably within one second. This overhead gate
protects that investigation from a measuring tool that materially changes its
timing sample. It does not undo the strength result or justify a larger search
quota. Keep graph64 as the leading strength candidate, eager32 as the unchanged
control, and larger-data work deferred. Diverse external opponents and expert
humans remain necessary later to support the project's wider exceptional-play
goal; a chain of wins against our own preceding versions is insufficient.

## Evidence and preservation

- [Frozen protocol](splendorust_graph_timing_20261025.md).
- [Freeze](../../outputs/splendorust_graph_timing_20261025_01/freeze.json):
  `a3892ae7083f5a73db490843c594338d959c350d2792f84c9c78c612f3d840e7`.
- [Overhead gate](../../outputs/splendorust_graph_timing_20261025_01/overhead_summary.json).
- [Diagnostic outcome](../../outputs/splendorust_graph_timing_20261025_01/diagnostic_outcome.json).
- [Final verification](../../outputs/splendorust_graph_timing_20261025_01/verification_summary.json).
- [Independent 48-request arithmetic and receipt review](../../outputs/splendorust_graph_timing_20261025_01/independent_overhead_review.json).
- [Owning execution and cleanup](../../outputs/splendorust_graph_timing_20261025_01/execution.json).

New source is confined to `tools/splendorust_graph_timing_20261025` and its tests.
Players, checkpoints, precision, search quotas, original reader, corrected rules,
deadlines and prior frozen evidence were not changed. The old benchmark's
separately incomplete supplementary audit remains incomplete; its scheduled
verification remains accepted. No reference pointer was promoted.
