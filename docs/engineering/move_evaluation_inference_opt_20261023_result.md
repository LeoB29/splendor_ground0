# Exact inference dispatch optimization: verified, still over 120 seconds

23 October 2026. The authorized two-stage investigation is complete. Finer
actual-model measurements identified transformer execution as about 92% of
Entity inference time. One separate implementation prebound the six existing
fused transformer calls. It preserves exact scientific results, but **none of
the three ordinary decisions completed within 120 seconds**. The first two
observed times fell slightly; the third rose. There is no demonstrated consistent
overall speed benefit and no basis to open the independent move-quality assessment.

## Complete decision timings

| Proposal fit / fixed root | Prior reuse ordinary | New ordinary | New instrumented | Original-policy replay | Above 120s |
| --- | ---: | ---: | ---: | ---: | ---: |
| 161001 / 0 | 137.1251101s | 133.4286568s | 139.5397158s | 231.3210251s | 13.4286568s |
| 161002 / 6 | 125.3612249s | 122.9659476s | 123.4130502s | 240.2701356s | 2.9659476s |
| 161003 / 12 | 165.6855068s | 203.6416851s | 208.8235260s | 261.7440391s | 83.6416851s |

Ordinary times include the complete request and delivery, not only model forward
time. Initialization is recorded separately. The new diagnostic allowed up to
600 seconds for a complete selection or replay; this did not alter the original
120-second qualification or reclassify any failure.

Observed changes relative to reuse were -2.70%, -1.91% and +22.91% in elapsed
time. These are unpaired runs under different host conditions, not controlled
causal estimates of the optimization. Ordinary tree CPU times were 120.375,
113.015625 and 137.65625 seconds. In the third case, CPU time rose about 3.9%
from 132.484375 seconds while elapsed time rose 22.9%. This discrepancy does
not identify its cause. Peak ordinary tree memory was 274–280 MiB. There were
no resource or scientific failures in this diagnostic, but all timing targets
were missed. Required reductions from the new ordinary times are 10.06%, 2.41%
and 41.07%, respectively.

## What was changed and verified

Implementation entity_prebound_fused_transformer_cpu_float32_20261023_v1 is
isolated in tools/move_evaluation_inference_opt_20261023. The same original
Parameters, nineteen static arguments per layer, six native operations, CPU
float32 arithmetic and surrounding inference operations are retained. Static
eligibility is checked at initialization, dynamic execution conditions on each
callback. The exact supported Torch sources/native runtime are pinned. This
introduces a private-runtime dependency; it does not justify promotion without
a useful, repeatable performance benefit. All frozen predecessors are unchanged.

All 179 combined prelaunch model-free tests passed. Fresh ordinary and timed
actual-model qualification exactly matched the original 15-case result, including
18 proposal checks across all three fits. Fresh original-reference, optimized
ordinary and optimized timed probes agreed exactly on the 18 fixed public
positions repeated eight times: all 144 decisions, all 11,952 captured neural
output values per route, public input tokens, actions and phase traces.
Repeated positions are not independent observations or full planner qualification.

For each complete planner case, ordinary and instrumented scientific outputs
and compressed traces matched each other and the previously verified reuse
implementation exactly. The same four candidates, 128 worlds, first-max rules,
seeds, legality/privacy checks, policies and simulation work were retained.
Root 0 also matched all 296 complete records from the original interrupted run;
its incomplete tail remains excluded and its original timeout remains a failure.

Independent replay used the ORIGINAL opponent implementation and the independent
dictionary transition ledger. It reexecuted all 1,536 continuations and 67,978
transitions: 22,362 / 21,554 / 24,062. It checked both policies, candidate
reproduction, observations, transitions, outcomes and selection arithmetic.
There were 33,989 Entity choices and 32,453 raw035 choices; 33,987 Entity
inferences reflect the unchanged two single-choice branches in the third case.
All selected-region event counts, code identities and timing partitions passed.

## What the measurements imply

| Instrumented scope | Fit161001 | Fit161002 | Fit161003 |
| --- | ---: | ---: | ---: |
| Entity inference, inclusive | 54.645s | 48.790s | 83.565s |
| Transformer scope within inference | 49.740s | 44.495s | 75.363s |
| Inference as fraction of instrumented body | 39.32% | 39.70% | 40.13% |
| Transformer as fraction of inference | 91.02% | 91.20% | 90.18% |

The regions include native computation, unselected descendants, waits and timing
overhead; these are not pure-kernel or removable-time measurements. Inclusive
parent and child regions must not be added. The small public panel showed about
8% less instrumented inference time than Stage A, but that observational result
does not establish a whole-planner speedup. The complete timings are decisive
for the 120-second gate. Another small Python inference cleanup is not supported
as a sufficient solution to the remaining gap.

## Recommended next decision

Keep the prior reuse implementation as the working reference and preserve this
candidate as a verified experiment. The original four-candidate/128-world
alternative remains unqualified for the 120-second research allowance; the
full balanced panel and independent quality assessment remain deferred.

Recommend preparing ONE separately frozen **A64/B64** study: reduce the fixed
selection-world count to 64 for both the raw035 control A and Entity-opponent
alternative B, keeping candidate support/order, policies, common-world pairing,
corrected rules and tie breaking fixed. A128 versus B64 would conflate opponent
policy and sample count, so cached A128 is not an equal-work control for that
question. This is a new scientific version, not an amendment to the failed
A128/B128 design or permission to reduce work inside the current implementation.

Precommit its schedule, resource bounds, one world count and a useful timing
margin before new quality outcomes. First qualify the same three development
cases and replay; only a pass should lead to the balanced 18-position panel
and six fixed privacy repeats. Retain the original development investment
criteria and the independently seeded 32-world terminal assessment once choices
are sealed and every prescribed gate passes. A timing pass alone says nothing
about stronger play. Fewer worlds may increase selection noise and change the
chosen move; this tradeoff needs the assessment.

64 is a finite candidate to propose, not a qualified solution: simply halving
203.64 seconds gives about 101.82 seconds, only 18.18 seconds of headroom, and
fixed costs or waiting need not halve. A successor must define its margin gate
prospectively and stop on failure, without adjusting 64 to another count within
the same plan. If retaining 128 worlds is instead essential, further exact
engineering needs a concrete measured opportunity. Batching would require a
separate numerical/scheduling qualification, not assumed bitwise equivalence.

No successor job, assessment, new game, training or larger-data study has been
launched. Larger-data work remains deferred.

## Frozen evidence and closure

Stage B output: outputs/move_evaluation_inference_opt_20261023_01.
Freeze SHA256: 84debfcaa72ec383a666d69713e5669023f6ae18bd1e3c008de63f1905882d5a.
Protocol: docs/engineering/move_evaluation_inference_opt_20261023.md.
Stage A diagnosis: docs/engineering/move_evaluation_inference_20261023_result.md.

Stage B ran 15:49:35–16:19:39 Paris, charged 1,803.191806 seconds, exit 0, no
cleanup errors. Together with Stage A's 644.969 seconds, cumulative owned time
was 2,448.160806 seconds, within the original 4,500-second bound. The original
three-hour wall/monotonic plan was retained; cumulative RAM wait was 120.437
seconds of 300. The initial pre-model Stage A launcher failure and its separate
same-clock repair remain disclosed and unchanged.

The independent model-free closeout authenticated all 2,567 frozen inputs,
external runtime hashes, 192 automatic inventory entries and 200 closed-run
files totaling 99,055,212 bytes. All 58 recorded process identities are closed.
Combined new study evidence is 154,676,121 bytes, below the 1 GiB cap. It
reproduced fixture/panel/count/scientific comparisons and authenticated every
original-policy replay receipt without new inference or numerical replay.

Closeout: outputs/move_evaluation_inference_opt_20261023_status_01/closeout.json.
SHA256: ab448a9ba5ffcc14b24b128933983d47be3eec30ba91a7777cdefe729caff5ee.
The sole administrative verification passed in 8.078 seconds; no model reruns.
