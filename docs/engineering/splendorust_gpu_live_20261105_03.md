# Immediate GPU dispatch: separately authorized bounded attempt

5 November 2026. The user explicitly authorized implementing immediate GPU dispatch
in a separate version, qualifying it, then running the concurrent-game comparison
if it passes. Preserve both failed attempts in full. This is a new study with its
own prospective freeze and the same two-hour total and 100-minute worker duration;
it does not extend, resume or replace either closed attempt or any active move.

## Single change

After receiving the first GPU request, collect only requests already in the queue
using nonblocking reads, up to the eight fixed slots, then dispatch immediately.
Do not intentionally wait for more arrivals. The requested gather interval changes
from 0.001 seconds to zero. Blocking while the queue is empty is permitted: that
wait ends when work arrives and is not a wait to fill a batch. The observed time
between submission and dispatch can still include host scheduling delays.

Retain the exact fixed eight-row physical input, permanent client slots, zero
padding, one outstanding request per client, dedicated GPU thread, full float32
network and original weights. No batching inside a search tree, extra simulations,
changed precision, cache, fallback, search order or player setting. Retain the
qualified Windows status publication/reader behavior from attempt02, with the
prelaunch API-binding change disclosed below.

Everything else follows the preserved original protocol
`docs/engineering/splendorust_gpu_live_20261105.md`: original512, three worlds,
cpuct0.4, depth64, corrected catalogue/nobles and native implementation; 9-second
internal and 10-second outer whole-decision clocks. One second and 700ms are
descriptive. Reference16 remains qualification-only. Same pre-reveal auxiliary-choice and
forced-pass conventions, same samples, numerical thresholds and replay checks.

## Qualification and comparison

Before model execution, pass source-bound model-free checks and independent review.
Include single-request immediate dispatch, coalescing an already waiting backlog,
requests arriving during an active batch, routing, expiry and failure propagation.
Retain all existing tests, including the actual Windows status-I/O stress suite.

Run fresh complete qualification; no prior partial result substitutes for a pass:
35 historical CPU exact roots, 840 GPU slot/companion cases, 80 concurrent service
requests, the actual corrected-rules 67-decision entrypoint suite per route and
35 full512 roots per route. The prior failed face_T1_05 request occurs first in
each full512 entrypoint suite; its saved failure is bound as predecessor evidence.
CPU/GPU actions remain descriptive; numerical tolerances and same-route exact
reproduction remain gates. Call verify_runtime_sources once at evaluator startup.

If every qualification gate passes, automatically run the originally reserved
eight concurrent CPU games, then eight concurrent GPU games, on the same eight
matched initial states. These seeds remain unplayed because neither prior attempt
started a game. No CPU/GPU wave overlap. Retain every move/terminal transition
check and the selected same-backend exact model replays (up to24 per route).
Stop at the first functional, numerical, deadline, replay or verification failure;
preserve it. No retries, replacement games, promotion or training labels.

Measure actual complete decision timing, queue waiting, batch occupancy and games
per hour. Removing deliberate waiting may reduce batch occupancy under concurrency;
neither faster collection nor a particular speedup is assumed. The CPU reference
is the unchanged compiled network in the existing shared-process, eight-thread
harness; this is not a production CPU-farm comparison or playing-strength test.

## Bounds, preservation and acceptance

Fresh actual two-hour total clock, 100-minute worker, actual worker-tree RSS8GiB,
new output2GiB, free disk1GiB. No available-RAM reserve or resource-forecast gate.
Both wall and monotonic clocks apply. No active move clock is restarted. This
authorization covers one frozen immediate-dispatch attempt, not an automatic
successor if a valid failure occurs.

Bind all source/input hashes and every file in both failed output/report trees;
verify their exact owners have exited before launch. Keep original failures and
prior clocks intact. Require final independent saved-evidence acceptance, actual
worker/controller exit, owned-child cleanup and frozen-input preservation. A
completion file alone is insufficient. Save an intuitive result chart when useful.
Training, larger-data collection, external testing and six-world work stay deferred.

## Preserved prelaunch status failure and narrow implementation change

The first aggregate check passed123 tests but failed one existing Windows stress
case: one of two readers exceeded the unchanged250ms mutex acquisition bound during
1000 publications. No model was loaded. Preserve tests01.json/.log, raw stress
receipts and the original source snapshot under prelaunch/status_stress_failure01.

One bounded instrumented reproduction completed1000 publications and more than
27000 reads per reader. It did not reproduce the250ms failure; its cause remains
unresolved. It measured repeated Windows API-wrapper setup and garbage collection
inside the reader's critical section. The maximum observed lock hold was17.64ms,
mainly file opening; garbage-collection pauses reached9.48ms while the lock was held.
Instrumentation itself can affect these timings, so they are diagnostic evidence.

Before freezing, initialize immutable typed Windows API function bindings once per
process. This removes recurrent wrapper setup; it does not cache status data,
mutex handles or file handles. Every operation still creates/opens its own mutex
handle, reads the current file through a fresh handle, releases/closes handles and
propagates errors. Retain the250ms acquisition/elapsed check, abandoned-owner
rejection,64KiB bound, publication durability, exact response clocks and all original
tests. Preserve the previous implementation and diagnostic. Require one full final
qualification of this substantive change plus independent review. Do not claim that
it explains the earlier timeout or proves general reliability. A pass admits only
the bounded study, whose real deadlines and fail-closed behavior remain enforced.
