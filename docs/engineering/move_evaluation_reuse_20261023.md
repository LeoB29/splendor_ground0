# Per-decision preparation reuse: bounded verification

23 October 2026. Owner authorized implementing the per-decision reuse
optimization as a separate version, then bounded correctness, timing and
replay checks across all three models. Implementation:
`tools/move_evaluation_reuse_20261023/`; evidence:
`outputs/move_evaluation_reuse_20261023_01/`.

## Scope and exact behavior

Create one validated public context inside each Entity opponent decision.
Copy the lawful input before validation, decode the corrected observer once,
and enumerate the complete ordered atomic legal set once. Construct the native
Main observation and ordered Main actions once. Reuse that context through
selection, compound resolution and final action membership validation.
Compound resolution receives a detached copy of the native observation because
it mutates auxiliary phases. Store public snapshots/caches as immutable
serialized data and the legal set as frozen actions. No referee hidden state,
cross-decision/world/turn cache, or hidden future draw may enter preparation.

Preserve every legal action, ordering, tie, policy input/output, phase/hash,
random seed, transition, cutoff convention and selection result. The frozen
evaluator still uses four candidates and 128 worlds, the same three proposal
fits, raw035 own moves and the same greedy Entity opponent policy. Reuse the
unchanged native binary and model kernels/checkpoints. No inference batching,
weight changes, pruning, quota reduction, relaxed deadline or unrelated
optimization is included. All prior sources, players and evidence stay frozen.

Implementation identity:
`entity_per_decision_public_preparation_reuse_20261023_v1`.
Record this separately in worker initialization and request metadata. Scientific
policy IDs and trace contents remain unchanged to permit exact comparisons;
the new implementation must not be mistaken for the original code version.

## Corrected-rule and small actual-model admission

Every new model-bearing worker calls `verify_runtime_sources()` once at startup
and saves the receipt, including reference replay workers. Preserve explicit
corrected rule/catalogue metadata and strict hidden-world decoding. Existing
checkpoint provenance remains historical; no new training or dataset loader
is admitted by this job.

First run model-free tests of the changed adapter: equality of the complete
ordered support/native observation, compound choices for corrected-rule
fixtures, mutation isolation, privacy and per-decision call counts. Cover all
six corrected faces, non-purchase/multiple nobles, terminal noble rounds,
gold payments/returns, private reserves, forced PASS and single-choice paths.
Test worker dispatch so the independent audit cannot silently use the optimized
adapter. Bind the new sources/tests/protocol and unchanged dependency inventory.

Under one frozen job clock, fresh ordinary and timed optimized workers each
run the unchanged 15-case actual-model qualification. Both results must exactly
equal the accepted original `entrypoint_qualification.json`: 18 proposal encoder
checks cover all three fits, native inputs/masks/outputs and auxiliary branches
are checked, and a three-transition terminal trajectory is independently
validated. Expect 17 optimized opponent choices and 17 calls to each new
prepare/encode/Main/resolve helper, zero original opponent choices; retain the
known raw34/proposal18/shared-forward52/world16 counts. Original bridge calls
inside the separate qualification reference checks are allowed and disclosed.

Use the already-qualified selected Python-region timer, with balanced returns
and exception unwinds, unique source/code identities, exact exclusive-time
partition, and root/body boundary gap <= max(5ms, 1%). No C-event profiler,
monkeypatching or scientific function replacement is used for measurement.
Any scientific, timing-count or provenance mismatch stops dependent work.

## Three fixed full-work cases

Only after the actual-model gate passes, evaluate the same candidate-lock roots:

| Index | Game/ply | Proposal fit |
| --- | --- | --- |
| 0 | 25/8 | 161001 |
| 6 | 57/14 | 161002 |
| 12 | 58/4 | 161003 |

Each case runs one fresh ordinary optimized selection, one fresh instrumented
optimized selection, and one full independent audit. The two selections must
exactly match each other AND the completed original selection from
`outputs/move_evaluation_timing_20261023_01/`, including compressed trajectory
bytes and every scientific field except output path. No new ordinary baseline
model runs are needed. Root0 also retains the original 296-record interrupted
prefix comparison; its incomplete tail remains excluded.

Reconcile the optimized counts against all 512 saved continuations immediately:
new prepare/encode/Main/resolve = Entity choices each; original opponent and
original bridge preparation/encoding/Main/resolution = zero; observer decodes
= Entity choices + 1; corrected legal calls = 2*all transitions + Entity choices
+ 129. Keep exact trajectory512/world128/proposal1/trace512/finalize1 counts,
raw calls = raw moves + 2, shared forward = raw moves + 3, and inference calls
derived from recorded native policy branches. The changed preparation census
is intentional; no scientific or model-call reduction is permitted.

The audit worker uses the ORIGINAL greedy opponent implementation, not the
optimized adapter. Replay all 512 continuations, reexecute both continuation
policies and proposal model, check every transition through the unchanged
independent dictionary-ledger oracle, and verify selection arithmetic. An
audit failure prevents the next root. This is three-position development
verification; the full 18-position qualification is not launched here.

## Resource limits and preserved failures

One owned outer clock begins before runner creation and includes startup,
source admission, RAM waits, qualification, selections, replay, analysis and
final authentication. Hard cap 60 minutes; work cutoff 55 minutes with the
last five minutes reserved for final authentication. Before each root's
selections reserve 660 seconds for reference audit initialization/request.
Initialization limit 60s; actual-model fixture request 300s; selection and audit
requests 600s, each clamped to the remaining monotonic stage clock. The live
resource guard also enforces the corresponding wall-clock stage deadline;
the independent owner enforces the global cap on both clocks.

The 600s request ceiling is for diagnosis only. Record whether each completed
ordinary optimized request is <=120s; exceeding120s does not authorize changing
the workload or prevent collecting the remaining bounded diagnostic cases.
The original failed120s qualification remains unchanged. Even three ordinary
successes below120s would not qualify the complete 18-position panel.

One CPU0 worker tree at a time, CPU-only models and one numerical thread.
Available RAM >=2GiB continuously for60s before first admission, healthy later
admissions may use the existing fast path; cumulative admission wait <=300s
inside the same hour. Active RAM shortage stops the request, never resets its
clock. Tree RSS <=2GiB; new run evidence <=1GiB; free disk >=5GiB (>=6GiB at
preparation). Use the existing hidden Windows job/watchdog process ownership.
No automatic retry, quota adaptation or extension; preserve all partial files,
failures and receipt/process identities. Authenticate frozen dependencies again
at completion and verify process closure.

## Interpretation and next decision

Report ordinary complete-response time, instrumented body partition, exact
work counts, replay outcome, CPU and memory. Comparisons to prior ordinary
timings are observational: host conditions were not paired. Instrumented
region shares include unselected descendants, waits and overhead; inclusive
times overlap, and measured shares do not promise removable cost or speedups.

If outputs differ, reject the implementation pending a separately identified
repair. If outputs agree but times exceed120s, use the remaining measured costs
to recommend the next bounded optimization. If all three ordinary cases meet
120s, recommend the full fixed-panel qualification with edge cases before any
fresh move-quality assessment. No strength games, larger-data collection or
training are launched by this plan; larger-data work remains deferred.
