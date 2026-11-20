# Lightweight timing observer: qualified; full saved-position diagnostic completed

25 October 2026. Status: **completed and verified**, including clean owning-process exit.

The separate lightweight observer passed its overhead gates for both unchanged players. The full 104-position diagnostic then completed: all 416 requests were correct and within one second, with a maximum of **0.578 seconds**. Neither historical failure recurred. This qualifies the observer for further investigation; it does not establish that the original rare stalls have been fixed or identify their cause.

## Scope and preservation

Source: `tools/splendorust_boundary_timing_20261025`. Evidence: `outputs/splendorust_boundary_timing_20261025_01`. The [protocol](splendorust_boundary_timing_20261025.md) and 2,041 inputs were frozen under SHA-256 **`919544571d367be1a3ecfc463063206302c4dbebe048ee6fae232d73c103702e`**.

The players remain optimized graph64 and unchanged eager32, with the same weights, precision, search, corrected rules, one-second trusted-arrival deadline and 0.900-second internal guard. No time is subtracted for instrumentation. The original reader, resource sampling, guards and cleanup remain in use. The observer records a few request/handler/response boundary timestamps; it does not instrument individual inference calls, native search, CPU time or guard spans.

The prior detailed-observer attempt remains preserved, including its negative overhead result and skipped full diagnostic. Its 1,686 frozen inputs and 304 sealed artifacts were checked unchanged during preparation and final verification. The earlier strength benchmark, both timeout losses, and its separately recorded incomplete optional audit remain unchanged. No new games, training, retries or player promotion occurred.

## Overhead qualification

There were 48 timed requests: six saved control positions per player, two modes and two reversed passes. Each row below compares 12 baseline requests with 12 observed requests. All were scientifically exact and completed within one second.

| Player | Baseline mean | Observed mean | Added mean | Mean ratio | Added p95 | Gate |
|---|---:|---:|---:|---:|---:|---|
| Optimized64 | 419.87 ms | 432.82 ms | +12.94 ms | 1.031 | +16.22 ms | Passed |
| Unchanged32 | 277.43 ms | 271.02 ms | -6.41 ms | 0.977 | -9.22 ms | Passed |

Both meet the frozen limits: mean ratio at most 1.10, added mean at most 25 ms, and added p95 at most 50 ms. The negative difference for 32 is execution variability, not evidence that observing the program makes it faster. These small-sample differences do not precisely isolate intrinsic instrumentation cost. Both gates passed, permitting the complete diagnostic.

## Full saved-position diagnostic

The unchanged panel contains 68 positions in their original graph64 role and 36 in their original eager32 role. Every position was evaluated in both modes and both passes, producing 416 requests and 104 four-way scientific comparisons.

| Player and mode | Requests | Mean | p95 | Maximum |
|---|---:|---:|---:|---:|
| Optimized64, baseline | 136 | 433.91 ms | 551.01 ms | 578.08 ms |
| Optimized64, observed | 136 | 430.38 ms | 528.09 ms | 559.32 ms |
| Unchanged32, baseline | 72 | 256.90 ms | 298.67 ms | 302.56 ms |
| Unchanged32, observed | 72 | 257.40 ms | 302.35 ms | 313.97 ms |

No request crossed the descriptive 700 ms margin; the admission deadline remained one second. Full scientific outputs, chosen actions, callback traces and resulting states agreed exactly across modes and passes. There were 408 comparisons with historically accepted results, plus four science-only comparisons with the historical late response. The aborted game-12 move has no invented historical completed target; its four newly completed results agree with one another.

| Historical failure position | Baseline replay range | Observed replay range |
|---|---:|---:|
| Game 12, ply 56 | 420.08-425.40 ms | 410.03-413.19 ms |
| Game 66, ply 29 | 401.61-423.60 ms | 409.37-413.21 ms |

The two positions therefore did not deterministically reproduce their original failures under these replay conditions. Their successful replays do **not** rescind the old losses or establish reliable timing under a longer live workload.

## What the timing boundaries show

For the 136 observed optimized64 requests, mean elapsed time partitions as follows:

| Interval | Mean |
|---|---:|
| Trusted parent start to complete input read by child | 47.12 ms |
| Input read to decision handler start | 0.07 ms |
| Inside the decision handler | 372.07 ms |
| Handler end to response write start | 0.03 ms |
| Response write start to trusted parent arrival | 11.08 ms |
| **Total** | **430.38 ms** |

Almost all ordinary pre-input time precedes parent dispatch: start to dispatch start averaged 47.02 ms. This is not a 47 ms pipe-transfer measurement. Delivery from response write start to trusted arrival peaked at 45.82 ms. The diagnostic did not reproduce the large historical delay outside the child's guarded interval.

These are wall-clock intervals, not CPU measurements. Time inside the handler includes its complete work and cannot be equated with neural inference alone. Parent dispatch completion can overlap child processing, so overlapping intervals must not be added twice. The data do not prove a RAM, host-load, scheduling, inference or communication cause for the old failures. Differences from the earlier, heavier observer attempt also cannot be attributed entirely to observer overhead because execution conditions varied.

## Verification and resource use

- 130 model-free tests passed before freezing, including failure, clock and raw-transport paths.
- Four actual entrypoint qualification suites completed 208 correctness decisions, including the corrected-rules counterexamples and original reference checks.
- All **464 timed requests** passed correctness, receipt and timing checks: 48 overhead plus 416 diagnostic.
- Six graph workers passed 11 exact value/float32-bit cases each; all 12 workers passed startup, raw-transport, log-integrity and cleanup verification.
- Final checks verified frozen source/runtime identities and read back 2,014 sealed artifacts. The previous attempt remained unchanged.
- Execution took **500.689 seconds (8 minutes 21 seconds)**, from 19:59:37 to 20:07:58 Paris, within the unchanged 30-minute cap. The owning process exited zero with no cleanup errors; watchdog and runner were confirmed absent.
- Evidence occupied about **50.65 MiB** before this postrun review/report, under the 128 MiB cap. Peak sampled combined worker RSS was about **534.4 MiB**, under 2 GiB. No resource-limit extension was used.

A separate independent scoped review recalculated timing distributions and gates, checked all 464 row/receipt identities, scientific groups, historical comparisons and boundary summaries. It passed. This additional review made no model calls and did not repeat the old benchmark's optional full-seal audit. Its receipt is supplementary postrun evidence, not a retroactive addition to the frozen collection seal.

## Recommended next step

Prepare a **longer, bounded operational reliability workload using this qualified minimal observer**, preserving the original live-game and worker lifecycle, players, deadlines and verification. Its purpose should be to capture a slow or failed move in the context where rare stalls can occur, then use the recorded boundaries to select a narrow fix. Freeze the workload, stopping rules and resource limits before running it. No such follow-up has been launched.

This selected, short replay panel is useful for correctness and reproducibility, but it cannot establish the population timeout probability. A component-specific optimization, increased simulation quota or new strength contest is not yet justified by these measurements. Preserve both historical losses and every attempt, including this successful observer qualification.

## The broader picture

The project now uses the verified SplendoRust policy/value network and search as its main starting point. Earlier comparisons showed gains from adding search and increasing its budget; the exact inference optimization made 64 simulations practical, and optimized64 scored **64.25%** against unchanged32 in the 200-game comparison, including its two timeout losses.

The current obstacle is making that demonstrated search gain dependable under a strict one-second allowance. The first detailed observer failed its overhead gate; this lighter version now gives us a qualified way to locate timing delays without changing decisions or clocks. The next investigation should target rare operational failures before committing to more search or training. Once timing is dependable, broader external opponents and expert-human calibration remain necessary to judge progress toward exceptional playing strength. Larger-data work stays deferred until storage and evidence justify the investment.

## Evidence links

- [Freeze](../../outputs/splendorust_boundary_timing_20261025_01/freeze.json), [overhead results](../../outputs/splendorust_boundary_timing_20261025_01/overhead_summary.json), [full diagnostic](../../outputs/splendorust_boundary_timing_20261025_01/diagnostic_summary.json)
- [Boundary analysis](../../outputs/splendorust_boundary_timing_20261025_01/boundary_analysis.json), [verification](../../outputs/splendorust_boundary_timing_20261025_01/verification_summary.json), [owning execution](../../outputs/splendorust_boundary_timing_20261025_01/execution.json)
- [Independent scoped review](../../outputs/splendorust_boundary_timing_20261025_01/independent_boundary_review.json)
- [Preserved detailed-observer result](splendorust_graph_timing_20261025_results.md), [earlier strength comparison](splendorust_graph64_strength_20261025_results.md)
