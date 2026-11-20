# GPU inference feasibility and timing — 5 November 2026

User authorization: “ok let's do the gpu test”. This separate bounded study tests
whether the installed AMD Radeon RX 7800 XT can accelerate our original SplendoRust
network. The accepted policy-only study has finished and its exact owners exited.
The original checkpoint remains the playing-strength reference.

## Fixed scope

Use original checkpoint EF8A4521… and its original 4,780,883-parameter transformer,
float32, inference/evaluation mode, with every weight unchanged. Existing native
Windows Torch 2.4.1+cpu and torch-directml 0.2.5.dev240914 are used without installs.
CPU reference is the accepted full frozen TorchScript forward, one thread on CPU0.
The separate supervisor uses CPU2. No new games, training, search changes, player
deployment or automatic successor. External testing and larger-data work remain deferred.

One GPU implementation is chosen before any model execution: the entire original
transformer expressed with explicit projections, batched matrix products, softmax,
layer normalization, residuals, GELU and original output heads. All 31 tokens pass
through all six layers. This avoids an unsupported fused CPU encoder route; it does
not remove computation, approximate attention or change precision. There is no
fallback or second implementation attempt inside this freeze.

## Inputs and correctness

The immutable panel binds 11 historical CPU graph cases plus 35 distinct corrected
saved roots and five timing composites. The source roots are qualification fixtures,
including counterfactual card/noble cases, not a representative self-play sample.
Batch 64 cycles 35 roots and repeats 29; batch 32 contains 32 unique roots. All
duplication and source provenance are recorded. No statistical independence claim.

The actual worker calls verify_runtime_sources once at startup and reconstructs the
panel through its authenticated corrected decoder/loader before loading Torch.
All 16 saved loader counterexamples and 13 negative mutations must pass. Ruleset
splendor_base_qualified_rules_20261018_v1, catalogue
db4776918e612eee5a6f2617719c85ec5f413d592bbab0a64058babc83a70312,
turn contract atomic_prereveal_completed_turn_v1 remain fixed.

First reproduce historical CPU outputs exactly; original eager and compiled outputs
must match bit-for-bit. Next compare the primitive implementation on CPU with that
reference; only then try it on the selected GPU. For every numerical and timed case:
finite full output; elementwise abs error <= 1e-4 + 1e-4*abs(reference); absolute
value error <= 1e-4; legal softmax probability error <= 1e-5; unchanged best legal
policy action where the CPU top-two logit gap exceeds 1e-3. Report every raw/legal
argmax disagreement, including near ties. This is numerical feasibility, explicitly
not exact search-decision or live-player qualification. Near-tie allowance was chosen
prospectively, before the freeze or any CPU/GPU model run.

Verify all model tensors reside on the selected DirectML GPU, input/output device
on every call, and original weights before/after readback. Any unsupported operator
or CPU fallback warning invalidates qualification. Capture Python warnings and inspect
retained native stderr; never report a CPU fallback path as GPU acceleration.

## Timing and interpretation

Only after all numerical cases pass, measure batches 1, 2, 8, 32 and 64 with five
warmups and 30 timed repetitions per route. Alternate CPU/GPU then GPU/CPU order.
CPU timing includes the original compiled forward and output readback. GPU timing
includes CPU tensor transfer to GPU, the full forward, transfer of the full output
back to CPU and blocking host read. Input-list conversion is excluded equally.
Validation and durable evidence writes occur outside the timed operation.
Report raw samples, mean/median/p95, positions per second and CPU/GPU ratio.

Batch 1 addresses immediate single-request latency. Larger batches address possible
parallel self-play throughput; they do not establish faster sequential search or
actual games per day. No game ETA extrapolation or promotion from this probe alone.

## Execution bounds and preservation

One worker, no retries: 30-minute actual worker cap within a fresh 45-minute overall
dual-clock probe cap; maximum actual worker-tree RSS 4 GiB; new output <=100 MiB;
actual free disk >=1 GiB. No available-RAM reserve, forecast admission or waiting
gate. Limits are safety bounds for this small inference probe, not a larger study.
Supervisor checks actual process identity, kills only its owned process on exit or
failure, and retains every output/error. Model-free tests, an independent review,
source/panel/runtime hashes and this protocol must be frozen before model execution.

A valid unsupported/numerical result ends the probe and is retained as a negative
feasibility result. A process failure is reported separately. Completion requires
source preservation, actual worker/supervisor exit and independent saved-evidence
review; a completion file alone does not establish acceptance.
