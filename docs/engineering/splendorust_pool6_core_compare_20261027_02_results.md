# Lean CPU comparison: startup deadline failure, independently reviewed

27 October 2026. The separate sampler-OFF version was implemented, frozen once and launched once. It stopped after **1 minute 46 seconds** during startup qualification. The original status remains **incomplete**. No CPU-placement comparison or player admission is accepted.

## What ran

The intended study retained both players, four fixed player/core configurations, CPUs 4–19 for supervisors, the original one-second external and 0.9-second internal limits, the 700 ms engineering margin, and all scientific and verification requirements. Its 192-decision matrix was the earlier planned full comparison, reindexed with additional host sampling disabled throughout. The two target roots and two fixed controls were unchanged.

All **141 model-free implementation tests passed** before freezing. The freeze and independent prelaunch readback authenticated 16,156 source/evidence pins, 93 runtime files and the complete 872-file prior failed attempt, including its unchanged negative overhead result.

| Stage | Actual outcome |
|---|---|
| optimized64, CPU0 startup suite | All 52 scientific decisions passed |
| optimized64, CPU2 graph qualification | All 11 graph cases passed; CPU0 also passed 11 |
| optimized64, CPU2 startup suite | 27 scientific decisions completed, then one handler was rejected by the internal deadline |
| Six-world player startup | Neither core configuration started |
| Paired timing comparison | **0 of 192 decisions attempted** |
| Additional host sampler | Never created; zero windows or counter samples |

The CPU2 prefix consists of 18 successful ordinary handlers and nine direct reference decisions. Including CPU0, 79 successful scientific decisions were separately verified. This does not constitute completion of CPU2's 52-decision suite. Both started workers' 22 graph cases passed. All subsequent startup and comparison work remained unperformed.

## Exact failure and clock scope

The failed subcase was **private_reserves**, search arm, optimized64 with 64 simulations and three hidden worlds, on CPU2. It was the nineteenth ordinary handler inside that startup suite. The saved action and complete 65-callback trace exactly match CPU0's successful execution; the first 150 scientific forensic events match across the cores. That saved result remains rejected and is not retroactively counted as a successful decision.

The original internal guard raised `ONE_SECOND_INTERNAL_DEADLINE: complete response exceeded 0.900s`. The enclosing administrative request returned a `worker_error` after 7.1041547 seconds within its separate 300-second suite allowance. This was not an ordinary one-second comparison timeout and not a game loss.

The failed boundary wrapper lasted 1,655.2953 ms; CPU0's same-case wrapper lasted 327.6697 ms. **These do not isolate search time.** The startup qualification synchronously JSON-encodes, writes, flushes and fsyncs its complete prevalidation forensic record inside the 900 ms handler budget. It also logs the error before the failed wrapper ends. There is no separate saved timestamp that attributes the delay to search, scheduling, serialization or disk synchronization. The guard failure is real under the unchanged instrumented startup path; ordinary CPU2 move latency cannot be inferred from the 1.655-second wrapper.

The additional host sampler was never started, so it was not involved in this failure. This does not prove that forensic logging caused the failure, or that CPU2 is slower than CPU0.

## Finalization and independent failure review

The normal finalizer has no forensic-only acceptance path for an incomplete startup suite. Its strict raw startup validator requires a successful admission response. It therefore rejected the failed request during closure auditing and again during final verification. The expected full-suite summary and success artifacts were also absent. Both the original qualification failure and the finalization failure are preserved.

The attempt has **no normal completed-evidence seal**, coverage or comparison analysis. A separate, finite saved-only review authenticated the failure without changing these facts or bypassing the successful-startup requirement. It verified:

- All 16,156 frozen source/evidence pins, 93 runtime files and 872 prior-attempt files.
- CPU0's complete startup science, both workers' 22 graph cases, CPU2's completed scientific prefix and the exact failed raw error packet/forensic chain.
- Placement and ownership evidence, all three actual exit codes of 1, absence of all 15 experiment identities and the owning PowerShell process, and empty cleanup-error lists.
- A separate preservation inventory of all 112 original attempt files. This is a failure-evidence inventory, not a replacement successful-completion seal.

Charged execution time was 105.812390 seconds, within the two-hour cap. There were no retries, new games, model executions or tests during the postrun audits. A read-only reviewer initially used the wrong elapsed-time field name; that reader exception is retained in postrun/independent_reader_error_01.json. No experimental source, clock, result or frozen artifact was changed to resolve it.

## What this result establishes

The sampler-OFF implementation was frozen and exercised, but qualification stopped before it could measure the placement question. CPU0 passed its suite; CPU2's saved failed-case action and callback trace were scientifically identical, while its instrumented startup handler missed the internal deadline.

We cannot choose a core, estimate ordinary one-second reliability, claim stronger play, or attribute the failure to CPU2 from this attempt. We have identified two concrete limitations to address: the absence of a proper failed-startup forensic finalization path, and synchronous qualification logging inside the timing budget.

## The next step

Recommend a separate version that:

1. Adds bounded forensic finalization for failed startup requests, tested against this exact saved failure and corrupted clock, raw-packet, chain and placement counterexamples. It must preserve failed qualification, nonzero exits, partial counts and zero ordinary collection; it must never synthesize a full-suite pass.
2. Runs a small bounded check of qualification logging overhead on the exact private-reserve fixture before another full CPU comparison. Measure the logging contribution before choosing a buffering or deferred-write change. Keep both players, all scientific evidence, the original deadlines and acceptance criteria unchanged.

Neither follow-up is implemented or launched by this report. The current attempt and the previous failed overhead attempt remain unchanged; no result is reused as a fresh timing observation.

## The broader picture

The project is trying to improve playing strength while reliably staying below one second per move. The six-world candidate may improve hidden-card sampling without increasing its 64-simulation budget, but it still needs a valid timing qualification and then a paired strength test against optimized64.

The earlier ample-memory study motivated testing core placement. The first placement attempt was blocked by the additional observer's overhead gate; this lean attempt stopped earlier in an instrumented startup test. The immediate priority is therefore to make that qualification and failure-audit path trustworthy and economical, so the eventual comparison measures the intended player behavior. This result says nothing about the candidate's playing strength. Optimized64 remains frozen, external-opponent testing paused, and larger-data collection/training deferred.

## Evidence identities

- Source: tools/splendorust_pool6_core_compare_20261027_02
- Evidence: outputs/splendorust_pool6_core_compare_20261027_02
- Protocol: docs/engineering/splendorust_pool6_core_compare_20261027_02.md
- Freeze SHA256: 92e35152b24276baaa676572b8891b723ae64eec4c4320be8389147eb5915745
- Independent failure review: postrun/independent_failure_review.json
- Review SHA256: 4a5cdd7aa91e92f10bebf70621b990bead9ad2afb31d524ff268c933e5131cb0
- Failure inventory SHA256: 192c8d834bda33f00e76ab3b6f03f86787b3037460a75cb22b092fa86bc8273e
- Supplemental diagnosis: postrun/startup_failure_diagnosis.json
- Diagnosis SHA256: 8e50962e387f1ab1a56fef2249312d291cad5e9ea072eeb526c1af3b8bf42faf
