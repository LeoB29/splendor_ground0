# Batched GPU integration and small throughput study — 5 November 2026

Authorization: user accepted the proposed next GPU integration test. This separate
study qualifies the real original512 search path, then runs eight concurrent games
per backend if correctness passes. No training, larger-data collection, strength
benchmark, deployment or checkpoint promotion is authorized by this study.

## Fixed player and implementation

Original pretrained EF8A checkpoint, unchanged full float32 network and weights;
512 simulations, three sampled worlds, cpuct0.4, depth64, existing native search
binary and corrected two-player rules. Reference16 is used only by the established
entrypoint qualification suite. Internal research allowance9s and outer response
allowance10s remain; one second and700ms are descriptive. No additional search when
time remains. Internal forced-pass/deadlock conventions and pre-reveal auxiliary
choices remain those of the original player; no altered game labels.

GPU uses the previously qualified full-transformer primitive implementation on the
installed DirectML backend. The live native search requests one neural row at a
time. Eight independent clients have permanent positions in a physical GPU batch
of eight. Missing rows are zeros; wait at most1ms to gather ready requests, never
wait for all clients. One outstanding request per client; no batching within a
single search tree and no reordering of that game's simulations.

CPU control uses the original compiled full forward, one Torch thread per call,
directly in concurrent game threads sharing read-only weights. It is not forced
through a serial GPU-style queue. Both routes use the same evaluator process CPU
allowance0..18, supervisor19, original native processes and game harness. The GPU
model has a dedicated inference thread; no special thread affinity tuning.
Each thread enters inference mode. No overlap between CPU and GPU measurement stages.

## Qualification and admission

Before models: source-bound model-free tests, independent review and immutable freeze
of sources, profiles, checkpoint, inputs, runtime, protocol and samples. Every new
evaluator/model worker calls verify_runtime_sources once at startup. The worker
uses the actual corrected snapshot decoder, legal-action encoder, native hidden
world reconstruction, teacher path and transition checks. Run the established
67-decision entrypoint suite independently for each route: corrected six card faces,
nobles including non-buy/multiple/final-round cases, gold/returns, private reserves,
malformed identities, native catalogue, late-game branches and same-route repeats.
No dataset or trained checkpoint is produced; no general training migration claim.

Before live play, the actual GPU batch-eight endpoint must produce identical
float32 outputs for fixed inputs under repeat, changed companions, unused slots
and movement among all eight slots. Concurrent CPU responses must match the
original single-call outputs exactly. Compare GPU outputs with CPU using the
accepted probe's elementwise atol/rtol1e-4, absolute value1e-4, legal softmax1e-5 and
robust argmax gap1e-3 gates. Preserve all discrepancies; any fallback invalidates.

Run full512 decisions on all35 distinct corrected saved roots on both routes.
Require legal complete actions, unchanged quotas and strict root-output numerical
agreement when a root neural request exists. Report CPU/GPU Main and complete-action
differences descriptively. No arbitrary cross-backend action-agreement percentage
gate; this is not a lossless player replacement or a strength result. Same-backend
deterministic replay remains required. Save same-state full decision latencies.

## Small operational comparison

After qualification, eight fixed fresh seeds, with starting roles balanced, each
played once as CPU self-play and once as GPU self-play. Eight independent games
run concurrently per route, CPU stage followed by GPU stage, without overlap.
Both seats use the same original512 weights/search. Barrier-start after readiness;
report startup and operational wall time separately. Preserve all complete moves,
legal support, search diagnostics, timing, observations and before/after snapshots.
Independently validate every transition and completed terminal outcome. Replay
the first learned decision in each fixed ply band0..19,20..39,40+ per game on the
same backend, in a shifted slot with the other game clients inactive (at most24
replays per route).

Record completed games, turns, inference calls, batch occupancy and queue waiting,
full decision mean/p50/p95/max and counts above one second, stage wall time and
games/hour. Pairing seeds does not guarantee identical trajectories across backends;
the small sequential game throughput comparison is descriptive. Same-state decision
timing supplies a complementary controlled comparison. No extrapolated large-data
cost or playing-strength conclusion is accepted from this pilot alone.
The CPU control is the original numerical implementation in this new shared-process,
eight-thread harness, not a measured eight-process production CPU farm. Python
coordination overhead can differ between those arrangements; any speed ratio is
limited to the tested integration harness.

Stop at any functional, numerical, deadline, replay or verification failure and
preserve all partial evidence. No failed decision or game is retried/replaced.
A timeout is an operational failure, not an official terminal training outcome.

## Bounds and completion

Fresh actual2h total clock, worker100min, actual worker-tree RSS<=8GiB,
new evidence<=2GiB, free disk>=1GiB. No available-RAM reserve or forecast admission.
Each move keeps its original absolute clock through queuing, GPU work, native
search and response validation; a supervisor watches active deadlines and owns
cleanup. No active move clock reset, automatic restart or second implementation
inside this freeze. The source and evidence inventories are immutable.

Completion requires actual evaluator/supervisor exit and owned-child cleanup,
all frozen inputs preserved, independent saved-evidence acceptance, and a chart
if meaningful. External testing, six-world work and larger-data training remain
deferred. Recommendations after this study do not automatically launch a successor.
