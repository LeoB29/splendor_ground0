# Component timing diagnostic: inference is the main measured cost

25 October 2026. The bounded diagnostic completed and verified successfully in
257.907 seconds (4m18s), from 01:10:33 to 01:14:51 Paris. Instrumentation passed
its overhead gate before the full panel ran. This identifies the main normal
execution cost; it does not reproduce or explain the earlier occasional stalls.

## Result and preservation

- 108 model-free prelaunch tests passed, including the original clock tests.
- Four actual corrected-rule entrypoint batches passed, each with 52 decisions.
- All 48 overhead and 192 diagnostic requests completed within the unchanged
  1.000-second trusted arrival deadline. The maximum across all 240 was 0.535040s.
- Exact scientific outputs agreed across modes/repeats: 12 overhead and 48 full
  panel position/quota groups. Historical comparisons passed: 24 overhead,
  116 diagnostic (16 at32, 100 at64). Every recorded transition was checked.
- Original receipts, separate parent timing records, causal child boundaries,
  all eight worker logs, frozen sources/runtimes and sealed evidence passed
  final reread verification. Outer exit0, no cleanup errors; watchdog42024 and
  runner23176 were absent after completion.
- Peak aggregate sampled worker/helper RSS was 604,446,720 bytes (576.4MiB).
  Attempt evidence was 37,782,714 bytes (36.0MiB) at completed inspection.

The baseline remains the verified raw reader with the optimized sampler. The
unsuccessful buffered reader was not promoted. Both modes kept the checkpoint,
32/64 search quotas, three worlds, depth64, float32 CPU, one numerical thread,
assigned core, corrected rules, action mappings and existing checks. The .900s
internal limit and .700s future admission margin also remain unchanged; the
latter was descriptive in this diagnostic. No player optimization, fresh game,
strength comparison, training or automatic retry occurred. Earlier failures
remain preserved. Larger-data work remains deferred.

## Instrumentation overhead first

Six predetermined development roots, two modes, two search quotas and two
reversed paired passes produced 48 requests. Each quota independently required
profiled mean <=110% baseline, added mean <=25ms and added p95 <=50ms.

| Search quota | Baseline mean | Profiled mean | Added mean | Added p95 | Result |
| --- | ---: | ---: | ---: | ---: | --- |
| 32 | 209.90ms | 210.60ms | +0.70ms (+0.33%) | -1.29ms | Passed |
| 64 | 369.87ms | 392.14ms | +22.28ms (+6.02%) | +49.10ms | Passed |

The64 gate passed narrowly. This is a small engineering point estimate, not
proof of zero instrumentation effect. The full panel later showed much smaller
mean differences (+3.19ms at32, +2.34ms at64), but that does not erase the
overhead observations or their variability. No measured duration was subtracted
from the trusted deadline.

## Whole-decision timing on the 24-position development panel

The conditional panel used 192 requests: 24 saved development positions, two
quotas, two modes and two passes. No reserved assessment roots were evaluated.

| Quota and mode | Mean | p95 | Maximum |
| --- | ---: | ---: | ---: |
| 32 baseline | 218.78ms | 251.94ms | 277.34ms |
| 32 profiled | 221.97ms | 256.89ms | 289.80ms |
| 64 baseline | 412.91ms | 505.19ms | 535.04ms |
| 64 profiled | 415.25ms | 508.08ms | 513.39ms |

Every request also fell below .700s here. That does not reverse the earlier
operational failure: selected saved roots on this run cannot establish a
general tail rate or qualify64 for the main strength comparison.

## What occupies the measured time

Means below are from profiled full-panel decisions. They describe different
scopes and are **not an additive breakdown**: parent supervision overlaps child
work, and several parent labels are nested.

| Measurement | 32 simulations | 64 simulations | Interpretation |
| --- | ---: | ---: | --- |
| Trusted whole decision | 221.97ms | 415.25ms | Original hard-clock receipt |
| Full child handler | 192.42ms | 385.70ms | Includes inference and remaining child work |
| Inference path, wall | 158.47ms | 329.66ms | Tensor creation, checks, network, output conversion |
| Inference path, Python process CPU | 127.60ms | 253.91ms | Excludes parent and native Rust CPU |
| Parent supervision wall union inside trusted interval | 121.89ms | 205.38ms | Overlaps inference; not removable latency |
| Initial supervision sample | 25.42ms | 25.09ms | Almost all of the pre-child dispatch gap |
| Native writes plus flushes, wall | 3.99ms | 8.07ms | Can include blocking/scheduling |
| Native read wait, wall | 2.23ms | 3.92ms | Includes native computation and scheduling |
| Parent response-reader thread CPU | 1.95ms | 2.60ms | Coarse CPU accounting; read can start before request |
| Child pre-emission to trusted receipt | 3.77ms | 4.05ms | Serialization, transfer, reading and scheduling combined |

At64, inference occupies about79% of mean trusted arrival and85% of child
handler wall time. There were about63.9 neural calls per decision at64 and32.8
at32; some roots use fewer neural calls under the unchanged search rules.
Inference costs roughly double with the larger quota. This makes inference
execution the main measured normal-work target.

Supervision remains substantial. Its64 union is205ms, including nested
aggregate/resource checks, but much of it occurs while the child is computing.
It is incorrect to predict a205ms speedup by removing it, or to add it to330ms
of inference. The approximately25ms initial sample is on the dispatch path.

The response path is comparatively small in this run. Raw read wall time mostly
waits for the response; it is not pure transport work. CPU timers on this system
show coarse increments, including15.625ms ticks, so small per-call CPU numbers
must not be treated as precise costs. No host-load/frequency evidence establishes
the cause of earlier slow moves. Faster normal execution could create margin,
but curing those stalls remains an unproven outcome.

## Recommended next fix

Prepare a single bounded **same-weight inference execution pilot**. A separate
frozen inference-graph candidate is a reasonable first hypothesis: try to reduce
the repeated cost of executing the existing network while retaining weights,
float32, one thread, input/output support, batching and search decisions.

The present timer does not isolate tensor preparation from neural operators or
Python dispatch. It therefore does not prove that graph execution will help or
predict its speedup. Compilation/fusion can change rounding despite identical
weights and dtype. Require exact numerical outputs, full callback hashes,
actions and transitions first; a parity failure rejects this behavior-preserving
candidate rather than weakening the checks. Include startup/compilation cost
and resource limits in its new plan.

If correctness passes, require a meaningful gain in **uninstrumented whole
decisions**, preserved tail margin, development and a fresh reserved assessment,
then fresh operational admission under unchanged supervision and clocks. Do not
combine inference, sampler, reader, search and training changes in that pilot.
Only an admitted candidate would justify a separately frozen64-versus32 strength
comparison. This recommendation has not been implemented or launched.

## Place in the broader research direction

Our objective remains exceptional two-player Splendor at roughly one second per
move, eventually validated against independent strong opponents and expert
humans. The working starting point is the verified pretrained SplendoRust
policy/value network plus adaptive search. Our earlier four-candidate,
128-world full-rollout planner remains a research control.

The completed200-game comparison gave32 simulations a64% score against16
(paired95% interval57.75%-70.25%). More search has therefore helped at that step;
whether64 beats32 sufficiently within the practical budget is still unanswered.
We are creating enough reliable timing margin to ask that strength question.
The current diagnostic advances engineering understanding, not playing strength.

After useful strength-per-second improvements, a bounded corrected-rules
learning or distillation cycle may be justified. Scaling data or training remains
deferred. The120-second player remains a separate research reference/possible
teacher. No internal comparison establishes superhuman strength.

## Evidence

- [Frozen contract](splendorust_component_timing_20261025.md)
- [Freeze](../../outputs/splendorust_component_timing_20261025_01/freeze.json),
  SHA256 `0867e9869b88c725a3afff34b22ce0c2c13349a979ced909e75610dd6f013c09`
- [Overhead gate](../../outputs/splendorust_component_timing_20261025_01/overhead_summary.json)
- [Diagnostic panel](../../outputs/splendorust_component_timing_20261025_01/diagnostic_summary.json)
- [Component analysis](../../outputs/splendorust_component_timing_20261025_01/component_analysis.json)
- [Verification](../../outputs/splendorust_component_timing_20261025_01/verification_summary.json)
- [Completion](../../outputs/splendorust_component_timing_20261025_01/completion.json)
  and [owned cleanup](../../outputs/splendorust_component_timing_20261025_01/execution.json)
- [32-versus16 strength evidence](../../outputs/splendorust_search_scale_20261024_01/analysis.json)
- [Broader roadmap](../strategy/splendorust_revised_roadmap_2026-10-24.md)
