# Parallel-throughput pilot: brief for implementation

15 November 2026. Brief only. It authorizes preparation (code, model-free tests,
review). It does NOT authorize a launch. Do not run anything while
`outputs/splendorust_collection4096_20261115_01` or its corpus acceptance is
active. Launch needs the user's explicit go-ahead once that collection has exited.

## Why

The machine spends most of its time waiting, not computing. Evidence (verify each):

- Collection: CPU8 made 51.15 games/h while collecting, 36.81 games/h including
  per-cell checks. Cells took 19.22/18.31 min of collection, 27.54/24.62 min with
  checks, so checks take about 26–30% of wall time.
  `outputs/splendorust_gpu_collector_report_20261105_01/results.md`
- Shared bottleneck: one serial live game (six-world block 1) took 54,837.6 s for
  400 games, about 137 s/game or about 200 network calls/s. CPU8 runs 8 games
  together at about 400 calls/s (51.15 games/h × ~55.4 roots × ~513 callbacks).
  That makes each concurrent game about 4× slower than a lone one.
  `outputs/splendorust_sixworld512_20261114_02/jobs/block_01/worker/arena/progress.json`
- Idle CPU: during collection4096 (15 Nov, about 13:50 UTC) the collector used
  about 4.7 of 20 logical CPUs, and the whole machine used 38–48%.
- Scaling of separate searches on saved openings: CPU1/2/4 gave 9.23/17.28/33.42
  decisions/min, so 4 searches ran 3.6× faster than 1.
  `outputs/splendorust_gpu_concurrency_report_20261105_01/results.md`
- GPU is not the lever for now. Per batch of 8, the service spent 10.5 ms on the
  network round trip and about 9 ms on input preparation, receipts/hashes and
  journal writes. The full collector gained only +8.25% (+5.75% with checks).
  `outputs/splendorust_gpu_costs_report_20261105_01/results_v2.md`

Hypothesis, not established cause: per-call Python handling in one shared process,
plus serial checks, limits throughput. Separate processes and overlapped checks
should raise verified throughput without changing any decision.

## What changes, and what does not

Only orchestration changes: the number of independent processes, CPU pinning, and
when per-cell checks run. Everything else is unchanged: EF8A weights and qualified
float32 CPU graph, 512 total simulations, three worlds, cpuct 0.4, depth 64,
dynamic FPU, seeds, capture/recording, replays, audits and deadlines.

Build a separate versioned package, copying the accepted entry points:
- Collector route: `tools/splendorust_collection4096_20261115`.
- Live-match route: `tools/splendorust_sixworld512_20261114_v2` (or the accepted
  arena it derives from), using the control player in both seats.

Never edit a frozen package or evidence. Each model process gets its own inference
service and its own CPU pin. Verify the core map at startup: typically logical
CPUs 0–11 are six hyperthreaded P-cores and 12–19 are E-cores on this i5-13600KF.
Use one physical P-core per live model process, never two on hyperthread siblings.
Keep the controller off those cores and record the mapping.

The six-world attempt01 failed at CPU admission. Generalize its affinity
correction (`docs/engineering/splendorust_sixworld512_affinity_correction_20261114.md`)
to N pinned services, including fresh replay services.

## Arms and workload (fixed in advance)

All setups are fresh engineering setups. Keep them disjoint from every frozen
schedule and reservation, including collection4096. They are excluded from every
corpus, never become training labels, and support no strength claim.

**A. Live play** (16 setups, each played in both arms; order A1 on 1–8, A4 on 1–8,
A4 on 9–16, A1 on 9–16):
- A1, control: one process, one game at a time (current behaviour).
- A4, candidate: four independent pinned processes, each playing one game at a
  time from a shared fixed queue.

**B. Collection** (two 16-setup cells X and Y, each collected by both arms, in
order control X, candidate X, candidate Y, control Y; 64 games in total):
- B-control: current CPU8, with checks after each cell.
- B-candidate: four independent processes × two clients. Each cell's selected
  replays and audit run while the next cell plays. If a check fails, stop and
  quarantine the overlapping cell.

Deadlines stay those of the reused entry points: live 119 s internal / 120 s
complete; collection 60/75 s. Never run parallel configurations under 9/10 s
guards: CPU8 under those guards completed only 2 of 8 decisions in the
concurrency diagnostic.

## Correctness (must all pass)

- Every worker calls `verify_runtime_sources()` once at startup and binds ruleset
  `splendor_base_qualified_rules_20261018_v1`, catalogue
  `db4776918e612eee5a6f2617719c85ec5f413d592bbab0a64058babc83a70312` and
  `atomic_prereveal_completed_turn_v1`.
- Reused entry points re-run their existing startup qualification: live
  counterexamples; collector fixture decisions and loader checks.
- Identity: the same setup must give identical results in both arms. That means
  every action, root visit vector, root value bytes, terminal outcome and, for
  collection, every target record. Decisions are seeded from the position, and
  same-route replays already reproduce bit-exactly across physical slots (5–6 Nov
  collector study), so exact equality is expected. Any mismatch stops the pilot
  and is reported. Do not substitute a numerical tolerance without a separate user
  decision. (CPU and GPU did diverge in that study, which is why both arms here
  stay on the same CPU route.)
- Before freezing, confirm how the CPU8 control packs network requests. If it
  forms multi-row batches, changing from 8 to 2 clients per process could
  legitimately change float32 bits. In that case keep the packing identical, or
  redesign the identity check, before launch.
- Existing replays, audits, ownership, exact process exit and cleanup are retained.

## Measures and adoption bar (engineering, not statistical)

Report, per arm:
- Games/h and decisions/h end to end, from first dispatch to the last verified
  check, plus steady-state cell cadence for collection.
- Per-decision median, p95 and maximum response time.
- CPU utilisation, peak RSS, and technical failures.

Adopt for future studies only if the candidate meets its bar:
- Live A4: at least 2.0× A1 games/h, 100% identity, zero failures, and maximum
  decision response under 60 s (half the internal guard).
- Collection: at least 1.3× control verified games/h including all checks, 100%
  identity, zero failures, and all audits passed.

A failed bar keeps the current configuration for that job type. No retuning or
extra configurations inside this pilot.

## Bounds and process

- Machine time: about 3 h expected (qualification about 30 min, live about 50 min,
  collection about 1.5 h).
- Caps: 6 h overall and 4.5 h model work. RSS: 2 GiB per process and 12 GiB
  aggregate. New evidence at most 10 GiB, with a 5 GiB free-disk floor.
- Ask the user to keep the machine plugged in and quiet: close the browser and
  other apps during timing.
- Preparation: model-free tests of process allocation, pinning, overlapped checks
  and stop-on-failure, plus a detached launch fixture.
- Review: one independent review is proportionate because players, rules and
  settings are unchanged.

Out of scope: GPU work, any search or algorithm change (including cheaper searches
on unrecorded moves), deadline changes in scientific studies, training,
collection for the corpus, and strength conclusions.

## Deliverable

A short report containing:
- The throughput table.
- Identity and audit receipts.
- Pass/fail against each bar.
- The recommended configuration for:
  1. The fresh live strength test after the 4,096-game training study (first use).
  2. The next collection.
