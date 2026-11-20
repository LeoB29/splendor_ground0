# Batching benchmark launch repair, version 2

2026-09-21. Owner explicitly authorizes running benchmark then preparing and
launching the learned128-versus-original128 experiment without another approval
round. The original prepared benchmark remains intact. Its first attempted
launch exited before model import, simulations or stage output because Windows
reused two completed-study worker PIDs for Discord and Firefox. Original launch
stdout/stderr are retained under outputs/batching_benchmark.

This version copies the original workload and execution code, changing only the
process-identity guard and isolated package/output/protocol paths. It recreates
and verifies the exact original selected positions and timing schedule. Numerical
code, models, sample selection, stage ceilings, trace checks and timing comparison
are unchanged. A recycled PID is cleared only when its live process has a
readable executable path with a matching basename that is definitely not Python
and not the original worker executable. Unknown/Python identities remain a hold.
The current benchmark process itself cannot be an earlier completed worker.
No unrelated process is terminated. The guard runs again at each stage.

The bounded stages may proceed sequentially within current owner authorization.
The original description below records workload details; prior no-launch wording
describes the preparation phase and is superseded by this explicit authorization.
A discrepancy retains evidence and blocks accepting batching. The following
scientific study may still use the unchanged serial implementation.

---

# Batching benchmark prepared during the proposal-attribution experiment

2026-09-19. Owner approved preparing the benchmark now and leaving performance
measurements until the current workers are free. This preparation does not queue
a run. No training, arena, monitor, production integration or automatic successor.

## Question and existing work

Can the existing CPU035 batching prototype preserve the current teacher's moves
while reducing collection plus verification cost? Its implementation already
exists in `src/splendor_ai/alphazero/teacher_batching.py`. The original tiny probe
has at most nine snapshots, no forced root action and serial-first timings; it
cannot establish representative throughput or exercise a full batch of16.

The successful completed proposal arena spent15.699h on main collection and
16.753h on independent replay. Speeding up collection alone cannot eliminate
verification cost. The new benchmark keeps the existing independent verifier
serial and intact, so its cost remains visible rather than being assumed away.

All new preparation/runtime/check code lives under `tools/batching_benchmark_v2`,
outside the running experiment's frozen src, analysis and experiments inventories.
The existing prototype, chooser, models and verifier are unchanged.

## Fixed workload

Use only completed, fully audited `outputs/learned_proposal_arena` evidence,
binding2154ba0b27284ae6077300376e6f4694254ab84f8826be8ce992d284add59a81.
Verify its final audit, main completion, both stage locks and all402 audit
receipts. Selection reads an explicit metadata allowlist: game/turn, actor,
planner role, offered support, private-reserve counts and root hash. Outcomes,
selected actions, continuation lengths and old timings do not influence sampling.
This is engineering development evidence from an already consumed dataset.

Select24 primary roots, one per phase x actor x planner x any-private-reserve
cell. Phases are turns0--19,20--39,40+. Within each cell choose the minimum SHA256
of compact sorted JSON `["batching-readiness-v1", game_index, turn]`.
Missing cells stop preparation; there is no replacement or design change.
Add one natural root for each offered support1/2/3 not already represented, if
available; otherwise explicitly report absence. Maximum27 roots. Retain all32
original observer-legal worlds and all supplied root actions, giving32--128
simultaneous trajectory jobs per root. Never pool unrelated roots to inflate a
batch. Root legal-action widths, privacy subtypes and action families are recorded
as coverage, not tuned selection criteria. This finite sample is not exhaustive.

The preparation binds selected roots through their audited hashes to the original
games and snapshots. It binds code, protocol, checks, source evidence and current
frozen dependencies, including inherited models. Preparation and verification
use only the standard library; they run no game engine or neural inference.

## Deferred equivalence stage

Use one CPU worker and one deterministic Torch thread. On every root, reconstruct
the exact proposal list and32world schedule, then compare original serial
continuations with batching at sizes1/4/16. Preserve every action and state-hash
trace, final state, action-trace digest, termination reason, winner and turn count.
Compare every trace exactly with serial, not just the winning root action.
Reconstruct root WDLs/means/choice and match the saved scalar record. Each variant
also passes the unchanged independent full-root verifier: lawful inputs, hidden
information invariance, proposal reconstruction and every continuation replay.

Record actual batch widths and full-batch counts. A requested size is not evidence
that it was exercised. Require at least one full batch of every prescribed size.
Any discrepancy fails equivalence, preserves artifacts and prevents timing.
No tolerance on action/state equality, fallback to scalar, removal of difficult
cases, replacement of failed work or promotion. Numeric score equality between
batch widths is not required if all resulting actions/states agree. A finite
pass does not prove universal equivalence; existing synthetic tie/termination
tests remain separate evidence and are not rerun during this preparation.

## Deferred timing stage

Timing requires a completed, verified equivalence result from this exact plan.
Primary comparison is fixed at batch16 versus the original serial production
chooser; sizes1/4 are diagnostic correctness arms, not candidates from which to
pick a favorable speed result.

Select12 timing roots from the24 primary roots: lowest selection hash per
phase/actor/planner, ignoring private-reserve category. Short-support supplements
are excluded from the throughput aggregate but retained in correctness evidence.
Warm up both arms on the first timing root, including independent verification;
exclude warmups. Four repetitions per root alternate ABBA/BAAB first-arm order,
with adjacent roots starting in opposite orders. Keep all96 measured root runs.
Each run retains32worlds, original choices/ties and full serial independent audit.

Measure collection wall/CPU time including input decoding, proposal work, worlds,
continuations and root-record I/O, followed by audit wall/CPU including its saved
receipt. Root equality checks and diagnostic receipt/metrics writes sit outside
these two clocks and are identical in purpose for both arms. Batch instrumentation
overhead remains part of batch collection. Record batch encoding/forward timings,
padding and actual widths as diagnostics; these do not profile serial cost shares.

Report summed serial/batch collection ratio and summed collection-plus-audit
ratio, plus all paired per-root ratios and their medians. These are descriptive
measurements of a fixed balanced workload on one CPU worker. No win-rate test,
confidence claim, universal speedup, equal-strength claim or GPU claim. This is
full decision-root collection and verification, not a new full-game arena or an
eight-worker end-to-end throughput measurement. Loading/startup, outer-game steps,
parallel contention and final global statistics need a later integration check.
The approximately50% historical audit share is motivation, not a timing forecast.

## Execution boundaries and commands

Preparation commands, safe while the current study runs:

```
python -B -m tools.batching_benchmark_v2.benchmark check --output outputs/batching_benchmark_v2/checks_01
python -B -m tools.batching_benchmark_v2.benchmark prepare --checks outputs/batching_benchmark_v2/checks_01/checks.json --output outputs/batching_benchmark_v2/preparation_01
python -B -m tools.batching_benchmark_v2.benchmark verify --plan outputs/batching_benchmark_v2/preparation_01
```

Deferred commands, not launched or scheduled by preparation:

```
python -B -m tools.batching_benchmark_v2.benchmark equivalence --plan outputs/batching_benchmark_v2/preparation_01 --output outputs/batching_benchmark_v2/equivalence_01
python -B -m tools.batching_benchmark_v2.benchmark timing --plan outputs/batching_benchmark_v2/preparation_01 --equivalence outputs/batching_benchmark_v2/equivalence_01 --output outputs/batching_benchmark_v2/timing_01
```

Before importing models or creating run outputs, both execution commands require
the current attribution experiment's matching final status with every audit
passed, and absence of its recorded supervisor/worker processes. Never kill a process to clear this check. Check for other
heavy host workloads before later launch; the PID guard is not a whole-machine
idle detector. Start any later background execution hidden with durable logs.
No command waits for completion or automatically wakes up. Timing is a separate
explicit invocation; equivalence success does not start it.

Each stage has a six-hour ceiling checked between complete root jobs. An overrun
or failure retains partial evidence and exits; it does not replace work or grant
permission for an extension. Output folders and files are write-once. Preparation
tests validate selection, outcome independence, timing balance, exact comparison,
artifact protection and the idle guard with standard-library fixtures. Full model
execution and runtime measurements remain intentionally unperformed until later.
