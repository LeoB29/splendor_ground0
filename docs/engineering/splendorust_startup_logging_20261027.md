# Bounded startup logging diagnostic and failed-startup finalization

27 October 2026. Separate authorized implementation and one frozen execution.
Preserve the incomplete sampler-OFF CPU comparison and all earlier attempts.

## Question

How much time does synchronous qualification evidence logging consume on the
saved private-reserve fixture, and can a failed startup be finalized as verified
failure evidence without being accepted as a qualification pass? This study
does not select a core, qualify a live player or compare playing strength.

Keep optimized64 (64 simulations, three hidden worlds) and the six-world player
unchanged: weights, precision, graph, search, seeds, corrected rules, support and
responses. Keep the original external one-second clock and internal 900 ms guard.
Retain the 700 ms margin as a descriptive engineering threshold. No retries,
warmup exclusions, trimming, worker replacements, extra host sampling or games.

## Prospective sequence

1. Qualify the separate failed-startup forensic validator using the exact saved
   failure and corrupted clocks, raw packets, chains and placement. It must
   preserve failed qualification, partial counts and nonzero actual exits.
2. Within the frozen execution, replay two saved full prevalidation records,
   private_reserves and gold_returns. The second is a fixed rule-edge control,
   not selected for its latency. Use 64 writes: 16 timer-OFF and 16 timer-ON per
   fixture, counterbalanced ABBA/BAAB. Require identical log bytes, chain and
   unchanged write/flush/fsync order. The new replay gate allows at most 5 ms
   added mean and 25 ms added nearest-rank p95 per fixture. These are frozen-route
   payloads; the gate does not guarantee overhead for every startup event size.
   No models yet.
3. If replay qualifies, start four fixed worker trees: both players on CPU0
   and CPU2, with all supervisors on CPUs4–19 at normal priority. Verify topology
   and placement. Each tree runs the unchanged 52-decision actual-entrypoint
   counterexamples and 11 graph cases. New timers are ON during these suites so
   a repeated failure has attribution evidence. Each worker verifies corrected
   runtime sources once before loading its model. No helper-only qualification.
4. If all suites pass, run 64 actual fixture requests, ABBA then BAAB timer modes
   for both fixtures and all four arms. Each arm has eight ON and eight OFF.
   Require mean ratio <=1.05, added mean <=10 ms and added p95 <=25 ms for every
   arm; these are the existing observer-overhead thresholds. All requests must
   be valid completed one-second responses. A failed gate stops detailed work.
5. Only if that gate passes, run 32 additional timed requests: four rounds of
   both fixtures in each arm. Report measured logging spans and full response
   time; never subtract logging to manufacture a passing move.

Maximum actual workload: 208 startup scientific decisions, 44 graph cases,
96 ordinary fixture decisions, 304 scientific decisions total, 104 parent
requests, four resident trees. Preserve exact science against authenticated
same-player startup records and every repeat; check legal support and independent
successors. The ordinary payload remains exactly player plus observer.

## Measurement and failure semantics

OFF uses the exact original synchronous recorder. ON preserves its event
dictionaries, chain bytes, JSON serialization, write, flush, fsync and digest
order, adding only monotonic timing reads and in-memory timing metadata. Record
serialization, write, flush, fsync and chain-digest spans. No evidence buffering
optimization is introduced. Only timing metadata is published after success
response flush, or before emitting a caught error so immediate supervisor cleanup
cannot erase the failure diagnosis. Error timing is explicitly labelled as such.
Persistent ordinary log
streams open before READY; startup retains its original stream lifecycle.

Authenticate sidecar/log-prefix bindings and distinguish events inside a handler
from outside-handler suite work. Failure logging after the guard must not be
mistaken for time which caused that guard. Timing spans include OS scheduling
and can identify elapsed contribution, not the physical cause of a slow fsync.
The small sample cannot estimate rare timeout rates or establish core causality;
with eight observations per condition, nearest-rank p95 is the maximum.

Stop at the first real timeout, mismatch, unexpected rejection or integrity/
resource failure. A 700 ms crossing is recorded without stopping. Failed-startup
finalization is a distinct forensic disposition: retain original failure and
partial counts, never issue a full-suite pass or proceed to ordinary work.
Unsupported failure shapes remain incomplete, with all available evidence saved.

## Bounds and completion

One-hour job cap, 30-minute collection cutoff. Stages: saved replay 300 s,
qualification 900 s, overhead 300 s, detailed diagnostic 300 s, final audit at
most the remaining 1,800 s. Original monotonic and UTC origins cannot reset.
Four workers maximum; combined worker/helper RSS <=2 GiB; require4 GiB available
before requests,60 s initial/recovery stability and600 s cumulative RAM wait.
Keep original in-request resource guards. Evidence <=2 GiB with512 MiB reserved
for finalization. Disk12 GiB initially and5 GiB maintained.

Freeze sources, protocol, inputs, matrix, tests, review, scientific/runtime
identities and every prior failed-attempt file. Require independent prelaunch
readback. Launch once under actual owner handles, finish audits and confirm
actual exits. Independent postrun review is saved-only, with no new models,
games or tests. Optimized64 remains frozen, external testing paused and larger
data collection/training deferred. Any follow-up optimization or full CPU
comparison requires its own separate design and authorization.
