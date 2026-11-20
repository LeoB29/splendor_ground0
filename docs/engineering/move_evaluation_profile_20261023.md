# Bounded diagnosis of the fixed move evaluator

23 October 2026. Owner authorized the recommended performance diagnosis. This
is a separate run, not a repair or retry of the closed120-second qualification.
Its failure and all historical evidence remain unchanged. No optimization,
assessment, strength game, promotion, training or larger-data collection occurs.

Use the unchanged evaluator, raw035 policy, pinned greedy Entity opponent, three
proposal checkpoints and corrected rules from the failed execution freeze.
Preserve four ordered candidates,128 worlds, seeds, termination and score rules.
All old frozen inputs are authenticated before and after execution.

Predetermine roots0,6,12 from the existing candidate lock: game25/ply8/fit161001,
game57/ply14/fit161002 and game58/ply4/fit161003. These are the first existing
root for each fit, all early positions and seat0. They cover the fits, not the
18-position timing distribution or worst case. No outcome-based root selection.

The new entry point first runs the existing complete correctness qualification
under profiling:15 corrected fixtures,18 actual proposal-encoder checks, native
main/auxiliary mapping, privacy and mixed terminal trajectory. Its scientific
output must equal the old accepted qualification. Each new model worker invokes
verify_runtime_sources once at startup and saves corrected identity receipts.
Only corrected decoders/legality/continuations are used; no training loader is
invoked or claimed qualified.

For each root, serially run a fresh ordinary worker, a fresh profiled worker,
then a fresh independent replay worker. The first two must return exactly the
same scientific decision and compressed trace bytes (excluding only trace path).
Replay every one of the512 continuations in one identical trace independently;
this is one full audit per matching pair, not two independent audits. Compare
root0 with all296 complete recoverable records in the old interrupted trace.
Never treat its incomplete tail as a decision or verified outcome.

Use cProfile's default wall timer around the unchanged complete scientific
selection body, including trace serialization/compression/close/fsync/hash.
Initialization and profile export are separate measurements. The trusted parent
clock includes complete response serialization/delivery and profile export.
Native pipe waiting includes child work and scheduling; report Python/Rust CPU
deltas from supervised identities separately. Exclusive self time categories
are additive; cumulative hotspot rows overlap and must not be added. Count
transitions, raw035 and Entity turns and trace bytes. One pair per root gives
diagnostic attribution, not a robust optimization speedup estimate.

Hard resources:60 minutes from the outer owner's runner launch, including
Python startup/source admission, memory waits, model
initialization, fixtures, all requests, replays and final analysis. Reserve the
last five minutes for final authentication/analysis. Each selection or replay
gets at most600 seconds, initialization60 seconds, correctness fixture request
300 seconds; clamp all to the remaining stage and global clocks. Before starting
a new root, reserve660 seconds for its audit (initialization plus full ceiling)
before the55-minute work cutoff. Both selections are clamped to that earlier
deadline. If the fixed trio cannot finish, report incomplete coverage and stop.
The600-second ceiling is for diagnosis only; it does not change the120-second
player qualification criterion. These maxima cannot guarantee completion.

One CPU0 worker tree at a time, one numerical thread, CPU only,2GiB combined
worker-tree RSS,1GiB new evidence, at least5GiB free disk (6GiB at preparation).
Require2GiB available RAM stable60 seconds initially, a healthy fast path between
requests, at most five minutes cumulative memory waits within the global cap.
An external hidden watchdog owns the process tree and enforces the absolute cap.

No automatic retry, adaptive work reduction, deadline extension or late-result
acceptance. Any mismatch, timeout, exhausted reserve or resource breach retains
all evidence and stops dependent work. A hard kill may leave no pstats file;
preserve the partial scientific trace and failure receipt instead. Completed
partial-root measurements may be described only with their missing audit status.

Recommend the smallest promising exact-equivalence optimization only after
measuring costs. Do not implement it in this diagnosis. A later version needs
full correctness/equivalence and120-second panel qualification before the study
can proceed. Larger-data work remains deferred.
