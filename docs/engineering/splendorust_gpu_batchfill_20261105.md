# Focused GPU scheduling and batch-filling diagnostic

5 November 2026. User authorized the proposed focused test of scheduling and
batch filling before considering another GPU software stack. This is a separate
version and freeze. No previous player, source, evidence or failed attempt changes.

## Question and intervention

The accepted preceding GPU8 workload averaged 3.7038 occupied rows per physical
eight-row inference batch. Test whether more independent searches feeding the
same physical batch improve correctly completed decision throughput.

Keep original EF8A weights, all float32 operations, physical batch eight, original
native search, 512 simulations, three worlds, cpuct0.4, depth64, corrected rules
and the immediate-dispatch DirectML implementation. No deliberate gathering wait.
No precision, network operator, driver or software-stack change.

CPU8 and original GPU8 are fresh controls. Candidate GPU16 and GPU32 use a new
queue scheduler supporting 32 logical clients. Each client has permanent row
index modulo8. At dispatch select the oldest ready request for each distinct row,
up to eight. Skipped same-row requests retain their order ahead of later arrivals.
One outstanding callback per client; no batching inside a search tree. Record
logical ownership, immutable row, enqueue order and dispatch membership. Physical
padding, neural-network math and all native search decisions remain unchanged.

## Qualification

Model-free tests cover colliding rows, FIFO ordering, sustained arrivals, duplicate
requests, output ownership, expiry, close, failure propagation and control behavior.
Authenticate unchanged physical B8 mathematical/packing qualification using a
narrow source-equivalence proof. Do not infer scheduler correctness from that proof.

Each actual new worker calls verify_runtime_sources once before loading a model.
Run the complete fresh67-decision corrected-entrypoint suite for CPU and the new
GPU route. Run actual16/32-client collision tests, two callbacks per client, with
known inputs and exact same-route float32 outputs. Test new scheduling while
preserving all original corrected face/noble/private-state/identity counterexamples.
Neither a manifest nor a helper self-check substitutes for the actual entrypoint.

## Fixed workload and order

Use 32 predetermined states: plies0,12,24,40 from each of the preceding eight
accepted matched games. Order by ply, then game0..7; permanent row equals game
index. All32 are distinct nonterminal neural-decision states. Authenticate full
prior CPU/GPU requests, states, responses and source clocks. Each fresh completed
decision must reproduce its prior same-route complete scientific result and root
float32 bytes exactly, with independent afterstate and search-work checks.

Two fixed blocks:

1. CPU8, original GPU8, candidate GPU16, candidate GPU32.
2. Candidate GPU32, candidate GPU16, original GPU8, CPU8.

Every configuration processes all32 requests once per block: maximum256 measured
decisions. Five service warmups per cell are outside timing. Each cell has a fresh
worker/model and prepares its8/16/32 native adapters before the first release.
Adapter i is reused sequentially for later cases mapping to i modulo concurrency.
No search has two simultaneous requests; every decision has its own original clock.

The primary panel timer runs from first barrier release through the final durable
case writes, including intergroup gaps, validation and evidence recording. Exclude
model/native startup, warmups, final summary serialization and cleanup. This avoids
charging repeated process startup only to configurations with more waves.

Report complete decisions per second, both block rates and total64 divided by
summed block time; batch occupancy/padding, queue time, forward/readback time and
complete move latency. A useful candidate must complete both blocks correctly and
improve throughput over fresh GPU8 by at least10% in each block. Compare against
CPU8 separately. These are descriptive diagnostic criteria, not a significance
claim or a guarantee of full-game speed. Fuller batches alone are not success.

## Clocks, failures and limits

Retain60-second internal and75-second independent outer safeguards from one
original completed-decision origin, including queueing and response validation.
Accept completed responses beyond1/9/10seconds and report those thresholds only
descriptively. No failed-decision retry, replacement, fallback or clock reset.

Only authenticated60/75-second decision expiry permits the next independent cell
after exact cleanup. Preserve failed prefixes, peer cancellation and unstarted
cases; an incomplete cell receives no complete-workload rate. Any source, routing,
numerical, replay, status, setup, resource-cap or cleanup failure stops the study.
A negative timing result is a study outcome, not a reason to modify the candidate.

Fixed total4hours from controller start, model cutoff3hours from that same origin;
qualification worker45minutes, each panel worker15minutes. Actual worker-tree
RSS8GiB, new evidence4GiB and minimum free disk1GiB. No available-RAM reserve or
forecast admission gate. No concurrent CPU/GPU study workers or scheduled monitor.

Require independent source review before freezing, all relevant model-free tests,
saved evidence audits after stages, final independent acceptance, actual controller
exit and exact owned-process cleanup. Preserve every failure. No new full games,
training corpus, fitting, external opponents or larger-data work is launched.

## Interpretation

If filling batches gives a useful gain, recommend a separately scoped full-game
confirmation before collection integration. If batches fill but throughput does
not improve, assess measured forward/communication costs before recommending a
software-stack test. Keep CPU8 as the verified practical reference. The outcome
concerns data-generation engineering, not playing strength or one-second play.
