# CPU0 versus CPU2: verified negative overhead result

27 October 2026. Implemented separately, frozen once and launched once. The procedure finished in **6 minutes 58 seconds**, within its two-hour cap. Its final status is `overhead_not_qualified_verified`. The independent saved-evidence review accepts this negative gate outcome, **not a completed CPU-placement comparison or player admission**.

## What ran

The unchanged optimized64 player (three hidden worlds) and six-world candidate each had a fixed worker on CPU0 and CPU2. All four workers retained their original network, float32 precision, graph, 64 simulations, seed, one-second external deadline and 0.9-second internal allowance. Supervisors and the owning PowerShell launcher used CPUs 4–19. The two target roots were g2_p36 and g2_p45; the structural controls were g2_p34 and g2_p37.

- All 167 model-free implementation tests passed before freezing.
- All 208 actual-entrypoint startup decisions and 44 graph cases passed across the four worker trees.
- All 96 scheduled overhead decisions on the two controls completed correctly. Every result matched its authenticated earlier same-player result exactly; 92 comparisons also matched the first new observation across cores/modes.
- There were **zero 700 ms margin crossings and zero one-second timeouts**. The slowest overhead decision overall was 649.3045 ms.
- The observer gate failed for one of the four configurations. As frozen, the 192 full diagnostic decisions were **not run**. In particular, this attempt did not perform the planned ordinary timing comparison on the two slow target roots.

## Observer-overhead result

Each configuration had 12 OFF and 12 ON measurements. All three criteria had to pass: ON mean at most 1.05 times OFF mean, mean increase at most 10 ms, and nearest-rank p95 increase at most 25 ms. With 12 values, that p95 is the maximum.

| Player and core | Mean OFF (ms) | Mean ON (ms) | p95 OFF (ms) | p95 ON (ms) | Gate |
|---|---:|---:|---:|---:|---|
| optimized64, CPU0 |438.981|407.834|548.241|441.063|Pass|
| optimized64, CPU2 |435.267|412.522|597.196|560.999|Pass|
| Six worlds, CPU0 |479.021|478.907|649.305|565.745|Pass|
| Six worlds, CPU2 |457.151|459.219|524.013|580.051|**Fail: p95**|

The failing arm's mean increased only 2.068 ms, but its p95 increased 56.038 ms. The ON maximum was sequence 40, g2_p37, round 5: 580.051 ms total, 402.0881 ms in the handler and 81.1132 ms in response emission. The OFF maximum was sequence 90 on the same root: 524.0133 ms. Neither was a missed deadline or margin crossing.

These are different observations, not a direct measurement of 56 ms of observer work. This small tail comparison does not prove that observation caused the difference, that CPU2 is worse, or that the difference is merely noise. It failed the prospective acceptance rule, which was neither relaxed nor retried.

## What this result establishes

Both fixed-core implementations passed the actual correctness qualification. The control-position outputs remained exactly unchanged across cores and observation modes, and the run preserved the original clocks, failures and verification checks.

The added 12-process host sampler did not qualify under the frozen overhead rule. Therefore the intended full placement comparison has no result: CPU2 eligibility and advantage remain **not evaluable**. Earlier negative admissions remain negative. This is no evidence of stronger play or reliable live-game deployment.

## Verification and resources

Independent readback checked 15,268 frozen inputs, 93 unique external runtime/Python files, the complete 1,646-file prior v3 attempt, 817 sealed evidence artifacts totaling 66,643,464 bytes, raw compressed responses, scientific outputs/transitions, 44 graph cases, placement/ownership, all 96 valid observer windows (48 ON/48 OFF), and preservation. Actual workload was 304 scientific decisions and 104 parent requests.

Peak combined worker/helper RSS was 1,195,003,904 bytes (about 1.11 GiB). Minimum available RAM before an ordinary request was 8.227 GiB. All three actual owner exit codes were 0, cleanup errors were empty, all 27 experiment identities were absent, and the owning PowerShell identity was absent. A clean process exit here means the negative-gate procedure finished and preserved its evidence; it does not turn the failed overhead gate into a pass.

All earlier attempts and failures remain unchanged. The initial model-free preparation path-reader failure is disclosed in prelaunch/preparation_failure_01.txt. An ad-hoc postrun display initially looked for nonexistent handler field names and printed a placeholder zero; this was discarded. The handler figures above come from the validated boundary-timing extractor. No raw result or frozen artifact was edited. No new games, models, tests or retries were run during the independent postrun audit.

## The next step

Recommend a **separately frozen lean CPU-placement comparison**, with the extra host sampler disabled. Retain four fixed workers, the same panel and counterbalancing, a fresh CPU0 control, supervisors on 4–19, one-second deadline, 700 ms margin, exact historical/cross-core outputs, and all trusted boundary timing, graph, corrected-rules, raw-response, placement, ownership, source, resource, preservation and cleanup checks. The frozen engineering criteria should remain unchanged.

The extra host-wide scheduling counters are not needed to compare complete-response latency. Disabling that additional measurement prospectively avoids repeatedly attempting the same small-sample overhead gate until it happens to pass. This changes the next study's measurement context; it does not amend or rehabilitate this attempt. No follow-up is implemented or launched by this report. A favorable lean comparison would justify fresh admission and operational checks, not immediate deployment.

## The broader picture

The project is trying to improve a strong SplendoRust-based player while reliably staying below one second per move. The six-world candidate was promising because distributing the same 64 simulations across more hidden-card samples might make decisions more stable. Its live timing margin has been the obstacle to a clean strength test.

The preceding saved-position study reproduced slow decisions even with ample memory, so insufficient RAM alone did not explain them. This follow-up was meant to test whether core placement supplies more headroom without changing either player's decisions. Its correctness work succeeded, but the added observer prevented the placement test from proceeding. The lean follow-up keeps attention on that practical question. Stronger play still requires a later paired game comparison; optimized64 remains frozen, external-opponent testing stays paused, and larger-data collection/training remains deferred.

## Evidence identities

- Source: tools/splendorust_pool6_core_compare_20261027
- Evidence: outputs/splendorust_pool6_core_compare_20261027_01
- Protocol: docs/engineering/splendorust_pool6_core_compare_20261027.md
- Freeze SHA256: e095d6e8e04100925968d95a39a0f4e49aa1781988cfef4f7fe617f0eaf1a43c
- Independent review: postrun/independent_overhead_review.json
- Independent review SHA256: ec45ea0f2817fbea36eedfe86f5c8c9df3c80833e7bf557dc5d96e81d733d0a9
