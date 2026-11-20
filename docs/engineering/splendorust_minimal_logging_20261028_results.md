# Minimal logging observer: verified negative qualification

28 October 2026 (Europe/Paris).

The single authorized attempt ended at its frozen overhead gate with status
`overhead_not_qualified_verified`. Independent review accepted the completed
evidence and the negative result. The observer is not admitted, the 32 conditional
requests did not run, and this logging-measurement route is now paused.

## Implementation and scope

The separate source version delegates the exact original synchronous forensic
recorder in both conditions. ON adds two timestamps around its total callback;
OFF reads no timing clock. Both use common preallocated raw capture and deferred
log reconstruction, validation and sidecar construction. Successful metadata is
published after response flush; failed-decision metadata precedes the error packet
so the original failure remains diagnosable. No mandatory evidence write, flush,
fsync or chain operation was removed or deferred.

Players, weights, precision, search, seeds, corrected rules, source checks,
verification, process placement and clocks stayed unchanged. The external deadline
was one second, the internal guard 900 ms, and the 700 ms margin remained descriptive
in this diagnostic. Extra host sampling stayed disabled. There were no new games,
training, replacement decisions or retries.

## Results

- 128 final model-free tests passed, with all 27 held inputs unchanged.
- All 64 fresh saved-record replays passed their overhead gate and byte checks.
- All four actual startup suites passed: 208 scientific decisions and 44 graph cases.
- All 64 overhead requests completed correctly, including 64 exact historical
  comparisons and 60 repeat comparisons, legal support and independent successors.
- No one-second timeout occurred in those 64 requests.
- Two requests exceeded 700 ms: 736.283 ms and 707.9327 ms. Both were the six-world
  player on CPU0, gold_returns control, timers ON (sequences 10 and 37).
- No conditional detailed request ran. Both optimized64 arms passed the overhead
  gate; both six-world arms failed at least one requirement.

| Player / core | OFF mean, ms | ON mean, ms | Added mean, ms | Added p95, ms | Gate |
|---|---:|---:|---:|---:|---|
| optimized64 / CPU0 | 589.694 | 599.569 | +9.875 | +15.826 | Pass |
| optimized64 / CPU2 | 563.759 | 571.079 | +7.319 | -27.234 | Pass |
| six-world / CPU0 | 634.399 | 639.045 | +4.646 | +49.528 | Fail: p95 |
| six-world / CPU2 | 575.285 | 593.953 | +18.669 | +29.432 | Fail: mean and p95 |

The unchanged requirements were mean ratio <=1.05, added mean <=10 ms, and added
nearest-rank p95 <=25 ms, for every arm. All mean-ratio checks passed. Each condition
contained eight observations per arm, four per fixture; p95 is therefore the maximum.
The two 700 ms crossings were retained and did not trigger replacements or early
stopping, consistent with the frozen diagnostic contract.

## What this result establishes

Both unchanged players passed correctness on this fixed workload, and all 64
ordinary diagnostic decisions arrived within one second. This does not establish
rare-timeout reliability or admit the six-world live player. Startup scientific
checks and ordinary response timing are different counts and should not be conflated.

The minimal observer still did not satisfy its overhead requirements. This is a
negative engineering admission result, not proof that two timestamp reads caused
18-50 ms of additional delay. These small repeated observations are temporally
dependent; shared deferred work, request delivery and host variability remain part
of the comparison. A single slow ON response can fail the declared p95 requirement.

The earlier private-reserve startup failure did not recur. Neither a cause for
that failure nor a preferable CPU core has been established. Comparisons with the
previous observer run are exploratory because they were separate executions, not
a controlled causal comparison of observer designs.

The recorded total-logging spans for the two slow requests were approximately
3.3 ms and 11.9 ms, versus response times of 736.3 ms and 707.9 ms. These are only
exploratory observations from a rejected observer. They do not support attributing
the full delay to logging writes, and cannot justify removing durable evidence or
subtracting time from the original deadline. No per-component timing was measured.

## The next step

Pause this profiling route, keep CPU choice unresolved and leave six-world admission
on hold. Do not run another observer version, repeat the negative gate or relax its
criteria under the current authorization.

Recommend a short saved-evidence review of the execution and timing protocol,
without new models or games. Its concrete output should identify which work is
deployment-critical, which synchronous forensic work exists only for qualification,
what the current clocks include, and whether the already-verified playing path can
be reused unchanged for direct six-world admission.

If that established path suffices, prepare a bounded admission plan using its trusted
outer clock and existing checks. If a genuinely different harness is necessary,
specify one justified change with exact parity and forensic requirements first;
a later small unchanged-player A/A check may then qualify that harness. Do not
repeat the baseline's accepted 200-game reliability study merely to gather more
timings. Do not move work outside an existing clock retrospectively. If the review
finds no defensible simplification, park the candidate rather than search for a
passing timing run. These are recommendations, not newly launched studies.

## The broader picture

The research question remains whether allocating the same 64 simulations across
six hidden-card worlds improves on optimized64's three-world approach while keeping
reliable one-second play. The earlier sampling-stability study motivated the candidate;
it did not establish a live playing-strength gain. Operational timing then became
the admission obstacle, and a slow startup check led to this logging investigation.

This result closes that instrumentation detour without weakening its criteria.
The next useful task is to simplify the experimental question using trusted clocks
and an appropriate execution path, before any new operational or strength test.
Optimized64 remains frozen. External-opponent testing stays paused, and larger-data
collection and training remain deferred.

## Preservation, verification and bounds

All prior attempts and negative results remain unchanged. Preparation preserved a
formatting-only placement-file copy failure before restoring exact original bytes,
and reviewer-only descriptor/path and process-check errors. The prelaunch review
needed corrected read-only continuations and took 346.69 s overall, exceeding its
initial 300 s review target; this is disclosed in its receipt. The final corrected
readback passed before any experiment clock or model started. No failed scientific
decision was retried. The independent postrun review had no errors.

Independent saved-only verification checked 17,422 frozen inputs, 93 runtime files,
863 predecessor files, the 803-file completed evidence seal, all 272 scientific
decisions, 72 parent receipts, raw/compressed responses, timing sidecars, graphs,
placement and preservation. All three actual owner exits were 0, all 28 distinct
owned identities were absent, and cleanup errors were empty. The reviewer ran no
tests, models or games and completed in 36.89 s.

Maximum charged experiment time was 387.844 s (about 6 minutes 28 seconds), below
the frozen 3,529 s cap. Peak combined worker/helper RSS was 1,249,054,720 bytes,
below the 2 GiB ceiling. Original evidence and disk reserve checks passed.

Source: `tools/splendorust_minimal_logging_20261028`.
Output: `outputs/splendorust_minimal_logging_20261028_01`.

- Freeze SHA256: `a5f5e547a9d27cb4b81a521ca23d60ccc26d8110fb6a0a09bbd105700691a18b`
- Independent review: `postrun/independent_review.json`
- Review SHA256: `67aea43edb26f933f932215d2bd66939767a82ad368d52ce57375011611042c3`
- Independent acceptance: `accepted_verified_negative_instrumentation_result`
- Original completion: `overhead_not_qualified_verified`
- Observer admission, full diagnostic completion, player admission, CPU selection
  and strength claims remain unaccepted. No experiment is running.
