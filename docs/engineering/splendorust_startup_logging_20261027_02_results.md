# Qualification logging diagnostic: verified negative overhead result

28 October 2026 (Europe/Paris). Both players, weights, search, corrected rules,
one-second external deadline, 900 ms internal guard and 700 ms margin were held
unchanged. Extra host sampling remained disabled. No games or training ran.

## Outcome

The corrected attempt finished at its prospectively frozen overhead gate.
Its status is `overhead_not_qualified_verified`. The completed prefix and negative
gate are independently verified; the timing observer is not admitted, and the
32 conditional detailed requests were correctly withheld.

- 109 final model-free engineering tests passed.
- 64 fresh saved-record replays passed their instrumentation gate.
- All four actual startup suites passed: 208 scientific decisions and 44 graph
  cases, covering both players on CPU0 and CPU2.
- All 64 overhead requests completed correctly: 64 exact historical comparisons
  and 60 repeat comparisons, with independent successor and legal-support checks.
- Zero one-second timeouts and zero 700 ms crossings in those 64 requests.
  The maximum was 696.3909 ms (six-world player, CPU2, timers OFF).
- No detailed requests, new games, retries or player changes.

Each arm had eight timer-OFF and eight timer-ON requests, equally divided between
the saved private-reserve failure fixture and the fixed gold-return control.
OFF retains the original synchronous evidence logging; it does not disable logs.

| Player / core | OFF mean, ms | ON mean, ms | ON minus OFF mean, ms | ON minus OFF p95, ms | Gate |
|---|---:|---:|---:|---:|---|
| optimized64 / CPU0 | 491.481 | 486.732 | -4.748 | +0.428 | Pass |
| optimized64 / CPU2 | 477.800 | 472.832 | -4.969 | -28.022 | Pass |
| six-world / CPU0 | 514.918 | 528.600 | +13.682 | +22.935 | Fail |
| six-world / CPU2 | 512.852 | 503.682 | -9.170 | -92.507 | Pass |

Every arm had to satisfy all three frozen requirements: mean ratio at most 1.05,
added mean at most 10 ms, and added p95 at most 25 ms. Only the six-world CPU0
added-mean requirement failed. Its ratio was 1.02657; its p95 requirement passed.
The threshold was not relaxed and the failed gate was not retried.

With eight observations per condition, nearest-rank p95 is the maximum. These
small, temporally dependent observations do not isolate timer cost from operating
system scheduling, persistence waits or other temporal variation. Negative
differences in the other arms do not imply that instrumentation makes play faster.

## Preserved first attempt and separate correction

Logging attempt 1 stopped before any model or worker started. Its 64 saved-record
replays passed, but 1,845,824 raw replay-log bytes bypassed the existing evidence
accountant. The cap itself was not exceeded: original output was about 11.2 MB.
Its original completion remains incomplete, its three owner exit codes remain 1,
and its independent failure review and diagnosis are preserved.

Attempt 2 uses separate source and output directories. It registers each durable
raw replay file once after close, outside the measured interval, including partial
write and close-error paths. Regression tests exercise the complete stage
transition and preserve rejection of missing, unaccounted or excessive evidence.
All preparation failures and superseded test receipts are retained. The final
test receipt is `prelaunch/model_free_tests_final_02.json`.

## What this result establishes

The corrected harness can cross the saved-replay boundary, execute both unchanged
players on both selected cores, and preserve exact scientific results. The old
private-reserve startup failure did not recur. This is a successful correctness
check on the fixed workload, not a rare-failure reliability estimate.

The six-stage logger did not qualify as sufficiently unobtrusive in every arm.
Its detailed observations are exploratory leads only. They are not promoted to
the withheld detailed diagnostic, used to subtract time from decisions, or used
to rehabilitate earlier failures. The run neither explains the original slow
startup causally nor establishes a preferable core.

For context only, the 32 instrumented ordinary records averaged about 7.38 ms of
measured logging, with a 13.16 ms maximum; fsync accounted for about 86% on average.
Two of 148 startup prevalidation spans were roughly 121 and 166 ms, almost entirely
fsync, on different fixtures from the original failure. These observations suggest
a possible persistence tail but do not justify removing or deferring required
durable evidence. The observer failed admission and scheduling is included in
these elapsed spans.

## The next step

Recommend one separately versioned, prospectively bounded qualification of a
minimal observer: two timestamps around the unchanged original recorder, measuring
only total logging time. Defer timing-entry construction and match post-response
bookkeeping between OFF and ON so that the comparison changes as little as possible.
Keep exact log bytes, event order, flush/fsync, all checks, both players and all
deadlines unchanged. Apply the same frozen overhead requirements once; preserve
a negative result rather than retrying or relaxing the gate.

If that observer qualifies, use it only for the bounded saved-position question
needed to understand startup logging, then return to the separate CPU-placement
decision. If it does not, pause this instrumentation route and reassess the method.
Do not buffer mandatory evidence, choose CPU2, launch a full comparison or start
games on the strength of the current result. This is a recommendation, not a new
authorized or launched experiment.

## The broader picture

The research question remains whether spreading the same 64 simulations across
six hidden-card worlds improves decision quality over the frozen three-world
optimized64 player while remaining reliably below one second per move. Earlier
sampling work motivated that candidate; slow operational positions then made
timing admission and host scheduling the immediate obstacles.

The CPU comparison is intended to resolve a deployment choice before another
strength comparison. This logging detour arose because a startup correctness
check itself failed its internal time budget. Resolving that measurement problem
helps distinguish player work from verification overhead. It does not constitute
evidence of stronger play. Optimized64 stays frozen, external-opponent testing is
paused, and larger-data collection/training stays deferred.

## Verification and resource accounting

Independent saved-only review authenticated 16,533 frozen inputs, 93 runtime files,
205 prior-attempt files, the 803-file completed evidence seal, all four startup
suites, raw responses, compression, logging sidecars, placement and cleanup.
All three actual owner exits were 0; all 28 distinct owned identities were absent.
No reviewer errors, model calls, tests or games occurred in that independent audit.

Attempt 2 charged 349.453 seconds at the maximum owner clock. Combined with the
prior attempt's 70.878584 seconds, cumulative charged execution was 420.331584
seconds (about seven minutes), below the one-hour limit. Attempt 2's cap was
3,529 seconds. Peak combined worker/helper RSS was 1,215,291,392 bytes. Original
attempt-2 artifacts totalled 68,642,418 bytes before supplemental review outputs.

Source: `tools/splendorust_startup_logging_20261027_02`.
Output: `outputs/splendorust_startup_logging_20261027_02`.

- Freeze SHA256: `9d649f47e3e3ee0e9545495e0e4d71ed804bb44a51977a153ad86cd593a1481b`
- Independent review: `postrun/independent_review.json`
- Review SHA256: `e9c2b22b19ae1d105a8f172050eec091022c17601b7ce0006bd7e1522940944f`
- Independent acceptance: `accepted_verified_negative_instrumentation_result`
- Full diagnostic completion, timing admission, player admission and strength
  acceptance remain false.
