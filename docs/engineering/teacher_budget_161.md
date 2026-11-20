# Experiment161: matched stronger-teacher incremental distillation

Owner authorized implementation and launch on2026-09-17. One fixed experiment;
no automatic resume, replacement, extension, promotion, successor or monitoring.
Preserve all dirty work and frozen predecessors. This file, implementation,
independent verifier and tests are hashed before any production export or fit.

## Hypothesis and data

Does128-world supervision improve the existing157 fast controller more than
32-world supervision on identical positions, and make useful progress over157?
Use all11,985 candidate128 roots from all400 completed160 main games. Both arms
share exact observation tensors, all concrete legal action features, canonical
groups and original ordered proposal representatives. Treatment selects the first
maximum summed WDL over all128 saved worlds; control uses the first32 of those
same worlds. All outcomes are -1/0/+1, natural endings only. No new teacher
rollouts. Keep unchanged choices, support1, private reserves, every phase and all
game outcomes. Expected matched-label changes4,081 is an integrity count, not a
quality filter. All source games/audits and selected157 initializer evidence are
hash-bound to the frozen completed predecessor chains.

Rank the200 whole160 initial seed pairs by SHA256 canonical JSON of
["161-source-pair-split-v1",160,pair]. First40 are validation; remaining160 train.
Both seats and all turns remain together. This reused development validation is
not pristine after the prior saved-data reviews; only fresh arenas confirm strength.

## Six fixed CPU fits

Initializers are157 treatment seeds157001/update1220,157002/update1220,
157003/update1144, with hashes in teacher_budget_161.INITIALIZERS. Corresponding
new order streams161001/161002/161003. Each produces world128 and world32; no
selection among initializers. Start fresh AdamW state for each run. Same157
architecture and offered-group cross entropy in both arms: evaluate every legal
action together, maximum logit within each offered canonical group, and never
penalize unoffered actions. Retain original representative and first-max ties.
Freeze value_head.*; every saved tensor in that head must stay bitwise unchanged.

Four complete passes, batch64 including final short batch, lr1e-5, weight
decay1e-4, gradient norm clip1, float32 CPU, deterministic Torch, one Torch thread.
Three simultaneous workers. Retain157 batching and validation ordering:
random.Random("157-batches-v1|SEED|EPOCH"), shuffle row indices, stable length
sort, chunk64, shuffle chunks. Validation uses seed157000/epoch-1. No dropout
randomness is present in this frozen architecture. Identical paired batch IDs.
Initial0 and updates1/2/4/8/16/32/64 within total, plus each epoch's quarters
(ceil(q*steps/4)), are checkpointed. Select lowest mean validation cross entropy
after averaging within whole source pairs and equally across pairs. Earlier
update wins an exact tie; initial is eligible. Finish every pass regardless.

## Fixed fast gameplay and analysis

Freeze all six independently verified selected checkpoints and three anchors
before the first arena. At every decision retain exact157 controller structure:
one frozen035 full-legal-set proposal pass, one learned full-legal-set pass, group
max scores and original representative. No planner or learned continuation swap.

For each of400 new initial seeds, both seats, all three paired fits:
(1) world128 versus matched world32 (transfer);
(2) world128 versus unchanged corresponding157 (progress).
4,800 main games. Index order: pair, comparison(transfer/progress), fit, seat.
Environment seeds1610000000+pair. Stream SHA256 canonical JSON
["161-proposer-stream-v1","main",pair], shared across every fit, matchup and
seat in the pair.155's frozen proposal nonce combines this exogenous stream
with canonical observable belief. No hidden-state seed or mutable RNG coupling.
Fixed48-game off-panel probe uses four pairs,1619000000+pair and "probe" tag.
No probe score-based decision. Its complete replay must pass; disk free must be
at least max(10GiB,1.5*100*probe_artifact_bytes+5GiB). Main runtime projection is
100 times probe collection plus replay duration; timing has no strength gate.

Win1/draw0.5/loss0 from world128's seat. Each seed cluster includes all12 games.
Average both seats and three fits separately for each comparison, then across400
seeds.50,000 joint cluster bootstrap resamples with numpy.default_rng(1613000000),
chunks1000, each shape(1000,400). Integer halfpoint sums avoid floating rounding
at55%. Percentile95% intervals, same sampled seed indices in both comparisons.
Intervals condition on these fixed initializers/order streams.

Clear success requires BOTH: transfer interval lower>50% (no extra five-point
point gate); progress observed score>=55% and interval lower>50%; and all audits.
Per-fit, seat, score margin and latency are descriptive, without subgroup gates.
No automatic sample extension. If either gate fails, clear success is not
established; report intervals and comparison-specific gates without treating
absence of significance as equivalence or rejecting distillation generally.

## Integrity and operations

Independently reconstruct all source trajectories, public tensors, memberships,
proposals, privacy invariance and both labels. Independently replay all six
complete fits, every checkpoint tensor and checkpoint selection bitwise.
Independently replay every actual probe and main move using frozen157's separate
decision implementation, including hidden-information invariance at every move.
Separately implement schedule, split, label arithmetic and final statistics.
Shared dependencies: frozen engine/encoders/canonical keys, Torch model/AdamW,
and frozen157 independent batching/loss/decision primitives. No claim of a
separately reimplemented game engine.160's original sampled-inner audit is
inherited, not repeated or described as full inner replay.

Natural terminal games or two consecutive sole-legal-PASS deadlocks are valid;
adjudicate deadlocks by final scores. Repetition>=4, turnindex400, idle60,
missing/illegal actions or nonfinite values fail and preserve evidence. No
replacement or recovery is automatic. Immutable per-file/stage locks, explicit
append logs, pythonw supervisor and workers. Reconfigure spawn before every
pool; verify absence of console attachment and save worker health. The direct
launcher waits up to60s for the immutable supervisor acknowledgement containing
its PID and binding; both parent-executable diagnostics finish before that seal.
Never relaunch after an acknowledgement timeout without explicit investigation.

Before freeze: full repository test suite under the same identity. The original
160 guard test assumes its designated production directory is empty; after160
completion, its no-overwrite guard raises FileExistsError before the expected
snapshot ValueError. The first full161 check preserved this one legacy failure
(1,401 passed). A hashed test-only pytest_environment_161 fixture isolates only
that original test's output globals to an empty temporary designated-production
directory. All its original assertions run unchanged; no existence checks are
masked, no tests skipped and no saved data/frozen files changed. An additional161
test explicitly checks both existing-output FileExistsError with unchanged game
hash and empty-output snapshot rejection. Full-suite invocation loads this fixture.
Tests include
actual Windows successive pools and a launcher which exits after acknowledgement
while its child continues; small isolated fits/games verify replay and tamper
rejection. Fixtures never write production outputs. No mutable progress files.

Pipeline: export400; independent export audit400; six fits/three workers; six
complete fit replays/three workers; seal nine models;48 probe games/replays;
operational preflight;4,800 main games/replays; independent statistics and report.
All scientific files stay frozen after prepare. Monitoring remains off.
