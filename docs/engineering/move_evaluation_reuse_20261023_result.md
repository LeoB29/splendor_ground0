# Per-decision preparation reuse: completed results

23 October 2026. The separate reuse implementation passed correctness, timing
instrumentation and full replay checks across all three fixed development
positions. Ordinary responses were 137.125s, 125.361s and 165.686s. All remain
above the unchanged 120-second target. The observed times are 20.1–39.4% lower
than the previous run, with uncontrolled differences in host conditions.
This is successful exact-behavior optimization, not deployment qualification
or evidence of stronger play.

The owned job ran from 13:13:10 to 13:41:37 Paris, taking 28m26.860s within its
60-minute cap. It exited 0 with no cleanup errors. No retry, workload reduction,
new baseline selection, quality assessment, strength game or training was run.
All earlier failures remain preserved. Larger-data work remains deferred.

Protocol: `docs/engineering/move_evaluation_reuse_20261023.md`.
Implementation: `tools/move_evaluation_reuse_20261023/`.
Evidence: `outputs/move_evaluation_reuse_20261023_01/`.
Administrative inspection: `outputs/move_evaluation_reuse_20261023_status_01/`.
Freeze SHA256: `d1cca323f022ab0531217fe7bdf2175bd329e1f23c348dea951762ca62dd103d`.
Implementation ID: `entity_per_decision_public_preparation_reuse_20261023_v1`.

## Timings against the target

| Proposal fit | Saved game/ply | Previous ordinary | Reuse ordinary | Instrumented | Original-policy replay | Above 120s |
| --- | --- | ---: | ---: | ---: | ---: | ---: |
| 161001 | 25/8 | 226.438s | 137.125s | 140.574s | 229.538s | 17.125s |
| 161002 | 57/14 | 183.014s | 125.361s | 125.733s | 222.654s | 5.361s |
| 161003 | 58/4 | 207.472s | 165.686s | 166.691s | 269.893s | 45.686s |

The remaining reductions needed from these observed ordinary responses are
12.49%, 4.28% and 27.57%, respectively. These are complete request times;
worker/model initialization is separately timed. One ordinary/instrumented
pair per position does not measure a latency distribution, instrumentation
overhead or controlled causal speedup. The third position has more simulated
transitions, and its wall time exceeds sampled tree CPU time by about 33s;
these receipts do not establish a cause for that gap. Wall time remains the
qualification criterion.

## Correctness and work preserved

The 412 combined prelaunch checks passed, as did 13 focused worker tests.
Fresh ordinary and instrumented actual-model fixtures exactly reproduced the
original 15-case reference, including 18 proposal encoder checks across all
three fits, corrected-rule counterexamples, native/auxiliary/privacy checks
and the mixed three-transition terminal trajectory. Every model-bearing worker
performed its startup runtime-source check.

All three ordinary/instrumented outputs and compressed traces matched each
other and the prior verified original outputs exactly, excluding only output
paths. Four candidates, 128 worlds, policies, numerical routines, ordering,
ties, seeds and termination rules remained unchanged. Root0 also matched all
296 complete original interrupted records; the incomplete tail was excluded
and the original timeout was not reclassified.

The independent audits used the ORIGINAL opponent implementation, replayed
all 512 continuations per position, reexecuted both policies and the proposal
model, and verified transitions and selection arithmetic. Totals: 1,536
continuations and 67,978 transitions (22,362 / 21,554 / 24,062), including 33,989
Entity choices and 32,453 raw035 choices. The third position has two correctly
handled single-choice branches, giving 12,029 inferences for 12,031 choices.
The replay shares authenticated legal/model kernels but uses the independent
dictionary-ledger transition oracle.

The selected timing events balanced and their exclusive durations reconciled
with the complete root; body boundary gaps were 8.7 / 9.8 / 9.6 microseconds.
Each Entity decision prepared, encoded, built Main actions and resolved once.
Observer decodes fell from 203,937 to 33,992 across the three instrumented
selections; legal enumerations fell from 340,277 to 170,332. Both reductions
are exactly 169,945 calls, with unchanged scientific/model work. No old bridge
preparation calls or original opponent choices occurred in optimized selections.
The immutable public context stays within one decision; no cross-turn cache
or hidden state was introduced.

## Remaining measured costs

| Selected scope | Fit161001 | Fit161002 | Fit161003 |
| --- | ---: | ---: | ---: |
| Entity inference pipeline | 40.09% | 40.68% | 41.51% |
| Corrected legal enumeration | 12.96% | 12.03% | 12.52% |
| Native exchange excluding selected inference | 8.00% | 8.25% | 8.03% |
| Shared raw/proposal attention forward | 7.71% | 7.86% | 8.21% |
| Raw score preparation excluding selected forward | 6.85% | 7.43% | 6.52% |
| Corrected observer reconstruction | 4.81% | 4.93% | 4.88% |

These are exclusive selected-scope fractions of instrumented root time,
including unselected descendants, native work, waiting and instrumentation
overhead. They are not pure kernel times or promises of removable cost.
Inclusive regions overlap and are not added. Trace sink/finalization together
account for less than 0.28%; trace construction elsewhere is not included.

Ordinary tree CPU times were 123.063 / 113.844 / 132.484s; Python-process CPU
includes tensor kernels. Native child CPU was only 2.05–2.56s. Ordinary peak
tree memory was 275–280MiB. These observations do not support storage cleanup
or faster Rust execution as the principal next speed intervention.

## Recommended next step

Retain this verified reuse version. The next bounded investigation should
separate input conversion, tensor preparation, network arithmetic and output
conversion inside the Entity inference pipeline, now the largest measured
scope at about 40–42%. First validate any finer timing instrumentation on a
small actual-model fixture. Use the same fixed workload and exact reference
outputs to identify a concrete optimization; do not assume that changing
batching, precision or numerical kernels preserves behavior.

Any resulting optimization belongs in another separate version, with the same
corrected-rule checks, exact outputs/traces and original-policy replay. The
third case still needs a substantial reduction, so passing the first two alone
would be insufficient. Once all three have useful margin below 120s, qualify
the full 18-position panel and edge cases before the deferred independent
move-quality assessment. Stronger play remains unproven. No next experiment
is launched by this result or recommendation.
