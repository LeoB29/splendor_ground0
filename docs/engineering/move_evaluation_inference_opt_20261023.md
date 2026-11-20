# Stage B: one exact-behavior transformer dispatch optimization

23 October 2026. Implements the authorized follow-up to the closed Stage A
diagnosis. Read move_evaluation_inference_20261023.md, its result report and the
separate launcher repair amendment. This stage inherits the original immutable
study_plan.json, its three-hour dual-clock deadline of 18:18:54 Paris, and its
75-minute cumulative owned-job limit. It does not start another study clock.

## Selected implementation and invariants

Implementation entity_prebound_fused_transformer_cpu_float32_20261023_v1 lives
only in tools/move_evaluation_inference_opt_20261023. It inherits the accepted
per-decision public preparation and named inference scopes. The measured
transformer stack dominates inference, so this one candidate prebinds the six
original fused TransformerEncoderLayer calls. Their nineteen static arguments
refer to the same original Parameters, with the same CPU float32 arithmetic,
native kernels, order, batch, output shape, checks and 83 output values.

Construction validates the exact pinned Torch runtime, model architecture,
layer attributes, Parameter type/device/dtype, absence of masks, nesting and
hooks, and conditions required by the original fused fast path. Relevant
dynamic execution conditions are checked inside the measured transformer scope.
No inference occurs during prebinding. Save the eligibility receipt separately
from scientific output and bind it to every request response. Original reference
probes and replays explicitly retain the original adapter and a reference receipt.

Only Python dispatch and repeated preparation are changed. There is no prediction
cache, JIT, batching, reduced precision, weight or tensor-layout modification,
candidate/world reduction, different action policy, or second candidate. Preserve
all historical failures and frozen sources. A mismatch stops dependent work.

## Qualification and measurements

Run model-free tests before freezing. Every actual-model worker calls
verify_runtime_sources() exactly once before model loading and saves its receipt.
Use the corrected rules/catalogue, lawful public observations and unchanged
historically trained checkpoints. This is no training or dataset migration.

First run one ordinary and one instrumented actual 15-case qualification, requiring
exact equality to each other and the original reference, including 18 proposal
encoder checks across all three fits. Then run fresh original-reference,
optimized ordinary and optimized instrumented public panels: the same 18 fixed
observers repeated eight times. Require exact tokens, all 83 outputs, action and
trace equality; preserve privacy and input immutability. These are diagnostics,
not 144 independent positions or full planner qualification.

Only after those gates pass, run roots 0, 6 and 12 (fits 161001, 161002, 161003)
in fixed order. Each gets one fresh ordinary selection, one instrumented selection,
exact full scientific and compressed-trace comparison to each other and the
accepted reuse version, and one full original-policy independent replay. Retain
root 0's comparison to the 296 complete original interrupted records; exclude
the incomplete tail. Each selection has the same four candidates, 128 worlds
and 512 continuations, unchanged raw own-turn and Entity opponent policies.

Use the accepted selected-region timer and all prior invocation/partition
requirements. Each of the nine inference helpers must match the actual callback
count, preserving the two single-choice branches on root 12. Fine-scope shares
include native work, descendants and waits; they are not removable-time claims.
Record complete response time, initialization, body timing, CPU and memory.
Comparisons to earlier runs are observational because host conditions differ.

## Shared limits and closure

Stage A charged 644.969 seconds, leaving 3855.031 owned-job seconds. Stage B is
capped at 3600 seconds and whatever remains before the original study deadline;
all startup, requests, waits, validation and cleanup count. Reserve the last
300 seconds for final authentication. Initialization is capped at 60 seconds;
qualification and panel requests at 120 seconds; complete selection and replay
requests at 600 seconds, clamped by both monotonic and wall deadlines. The
600-second diagnostic allowance does not alter the 120-second practical target.

One CPU0 model-bearing worker tree and one numerical thread at a time. Tree RSS
at most 2 GiB. Require available RAM at least 2 GiB, stable for 60 seconds initially;
later healthy admissions may proceed immediately. Include Stage A's 60.234 seconds
in the shared 300-second RAM wait limit. Active RAM shortage stops the request
without suspending or resetting its clock. All three new run directories together
must remain below 1 GiB; free disk at least 6 GiB at preparation and 5 GiB running.
Use the qualified hidden Windows job/watchdog with authenticated owned identities.

The Stage B freeze binds all 2480 Stage A inputs, complete closed evidence,
the original failed launch, original plan, authenticated closeout, diagnosis
report and these separate new sources/tests. The disclosed boot-estimate repair
tolerance remains at most one second with continued uptime and both clock bounds.
No automatic retry, extension or substitution. On closure reauthenticate all
inputs, scientific checks and replay receipts, preserve failures and verify all
owned processes closed. Report whether each fixed case meets 120 seconds and
its margin. Even three passes only justify recommending the deferred full
18-position-plus-edge qualification. Larger-data work and new strength games
remain deferred.
