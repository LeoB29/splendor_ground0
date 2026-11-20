# CPU0 versus CPU2: bounded saved-position comparison

Status: prospective separate attempt, 27 October 2026. User authorized implementation, freeze and one launch. All earlier attempts and failures stay immutable.

## Question and unchanged players

Does moving computation from logical CPU0 to CPU2 improve response-time headroom for optimized64 (three hidden worlds) and the six-world candidate? Both retain 64 simulations, checkpoint, precision, graph, scientific outputs, seeds, one-second external deadline and 0.9-second internal allowance. The 700 ms margin remains the engineering target. This is a saved-position placement diagnostic, not fresh games, strength testing or player admission.

CPU0 and CPU2 are on different physical cores of the same highest reported efficiency class. Their siblings are 1 and 3. Every supervisor uses CPUs4–19 in both conditions, at normal priority. Each player's Python and native helpers stay on their assigned single CPU. No exclusive reservation against other applications is claimed. Revalidate native topology and all identities/affinities/threads before accepting measurements.

## Fixed inputs and schedule

The targets are accepted v3 slow decisions g2_p36 (727.0924 ms, optimized64) and g2_p45 (704.6184 ms, six worlds). The controls are g2_p34 and g2_p37: nearest earlier same-game, same-source-role positions among the existing v3 fixed controls, excluding both targets and prior selections. Their timings do not enter selection. Both unchanged players use all four roots. Every response must match the accepted v3 full scientific result for that player/root exactly, including across cores and observer modes.

Use four fixed, resident worker trees, one per player and CPU, with only one active request at a time. This avoids migration and repeated initialization; separate process instances remain a limitation. Each tree receives its original 52-case actual-entrypoint suite and 11 graph cases:208 startup decisions and44 graph cases. Verify corrected rules before model loading, through the actual entrypoints and frozen source identities. The corrected ruleset/catalogue/turn contract are mandatory; historical training is not launched.

First run96 observer-overhead requests:two controls × four arms ×12 rounds. Each cell uses OFF,ON,ON,OFF repeated three times; each player/core has12 OFF and12 ON observations. For each of all four arms, require all correctness and clock checks, ON mean no more than1.05× OFF mean, added mean no more than10 ms, and added nearest-rank p95 no more than25 ms. With12 observations the nearest-rank p95 is the maximum. Any arm failure skips the entire full stage. Keep every observation; no trimming or warm-up removal.

If all four overhead gates pass, run192 full diagnostic requests:four roots × four arms ×12 rounds, observer ON. Each root/player gets adjacent CPU0/CPU2 requests. Core lead alternates each round; player lead alternates by floor(round/2)+root index. Controls reverse order every round. Full root order follows the frozen four-row Williams square repeated three times. This balances core/player lead combinations and root order. The generated request matrix is authoritative and immutable after freeze.

## Checks, failures and interpretation

Keep original scientific, raw-response, compressed-log, callback, successor, graph, clock, process ownership, source, runtime, preservation and cleanup checks. The plan's `existing_observer_unchanged` field refers to the trusted boundary observer and in-clock sampler. The separate external host observer changes to12 compute targets and supervisor CPUs4–19; requalify its overhead rather than assuming the earlier six-target cost. Raw host CPU inconsistencies retain the v3 optional-availability treatment; process counters, identity and API failures remain fatal. Context aggregates remain signed net changes, not true interval counts. No active clock can be reset.

A response at or above700 ms is recorded and contributes to the engineering verdict; it does not stop this diagnostic. Stop at the first actual timeout, mismatch, unexpected rejection or resource/integrity failure. Preserve the terminal failure and ordered prefix. No retries, replacements, selective continuations or player changes. Incomplete or overhead-negative runs have no full-stage engineering verdict.

CPU2 is eligible for recommending a fresh admission study only after full verified completion, all96 CPU2 full responses strictly below700 ms, and neither player's mean regressing more than5% nor p95 more than25 ms against its fresh CPU0 arm (48 versus48 per player). A demonstrated panel advantage additionally requires either a CPU0 margin crossing absent on CPU2, or at least25 ms p95 improvement for either player, while both non-regression checks hold. Thresholds are engineering criteria, not significance tests. Report per-root paired differences, means, p95, maxima and crossings; repeats of selected roots do not estimate live timeout risk. No stronger-play or specific scheduler-cause claim follows from this study.

## Hard limits

Two-hour whole-job cap from one monotonic and wall-clock origin; collection stops by one hour and finalization has the remainder. Qualification, overhead and full diagnostic each have a900-second stage cap. Maximum four workers/graph builds,208 startup decisions,288 ordinary requests,496 scientific decisions,296 parent requests,44 graph cases. No replacements.

Keep aggregate worker/helper RSS at or below2 GiB; require4 GiB available RAM before requests with60-second initial/recovery stability and600 seconds total RAM wait. Do not reset an active decision's clock for memory. Evidence cap2 GiB with512 MiB reserved for finalization; host observer buffer32 MiB. Initial free disk12 GiB and maintained free disk5 GiB. A limit failure is preserved, without increasing the cap or rerunning.

Freeze source, protocol, matrix, inputs, runtime, rules, all prior v3 evidence and its independent acceptance and supplementary diagnosis before a single launch. Verify all original failures through the inherited preservation chain. Finish with actual owner exit receipts and independent saved-evidence readback. External opponents remain paused and larger-data collection/training deferred.
