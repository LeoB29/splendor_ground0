# Simulation input preparation: bounded prototype result

Completed 2026-10-16. The prototype reduced elapsed time by **27.75%** across
48 fixed historical continuations, equivalent to **1.384x throughput**. All
saved move traces and final states matched. This supports further bounded
integration work; it does not establish production throughput or universal
equivalence. The running `structured_proposer_128` experiment is unchanged.

## What was tested

The prototype prepares the existing 79 action features with per-state/card reuse,
integer-array affordability calculations, and direct dense-array construction.
The 295 observation features, model, decisions, game rules and available
information are unchanged. It lives in a separate package and is not enabled in
production.

Only completed `large_corpus_optimization_128` evidence was used. The nine input
test roots were chosen by metadata: minimum, median and maximum canonical action
count within each of the three game phases. No current fresh results were read.

The continuation test used the three median-support roots, first and last saved
union candidate, and saved world indices 0, 17, 35, 53, 71, 89, 107 and 127. This
gives 48 continuations. Each had an untimed exactness check, then one timing of
each implementation. Timing order was balanced: reference first in 24 cases and
prototype first in 24. Three reference continuations were separately profiled.

## Results

| Continuation sample | Reference seconds | Prototype seconds | Throughput ratio | Elapsed time saved |
| --- | ---: | ---: | ---: | ---: |
| All 48 | 4.617874 | 3.336385 | 1.384095x | 27.7506% |
| Early phase | 2.520907 | 1.815576 | 1.388488x | 27.9792% |
| Middle phase | 1.659071 | 1.110756 | 1.493641x | 33.0495% |
| Late phase | 0.437896 | 0.410053 | 1.067901x | 6.3584% |

The sample contained 1,438 turns. All **1,390 intermediate input comparisons**
passed; the other 48 turns were the forced opening moves. Every checked and
timed continuation matched the historical termination reason, winner, turn
count, full trace hash and final-state hash. All sampled terminations were
ordinary `completed` outcomes.

The prototype was faster in 44 of 48 cases. Both timing-order groups improved:
about 1.41x when the reference ran first and 1.36x when the prototype ran first.
The aggregate CPU-time ratio was 1.352x, consistent with a useful gain. Total
probe duration, including loading, validation and separate profiles, was 16.01
seconds; that total is not the timing denominator in the table.

The nine-root input-only test also passed exact float32 tensor and same-shape
model-output comparisons. Aggregate input preparation was 4.545x faster, heavily
influenced by the largest, 558-action example. Median-support examples improved
by approximately 1.85x, 2.39x and 1.59x; tiny 1-3-action examples were slightly
slower, at 0.80-0.98x. **The 4.545x input-only result is not an overall simulation
speedup.** Its CPU ratio is especially sensitive to Windows timer granularity.

Separate reference profiling attributed approximately 44% of instrumented time
to action encoding. Observation encoding, compatibility copying and tensor
construction added about 10%, for roughly 54% input preparation in total.
Neural forward accounted for about 24%. These are attribution measurements on
three continuations, not universal production fractions.

## Resources, safeguards and provenance

The work ran with one extra CPU thread at below-normal priority, while the main
study was in an allowed fitting/replay phase. It neither signalled nor modified
the running study. All added probe processes have finished.

An original loading-only probe measured 196.8 MiB peak working set. Before any
timing, a separately frozen version 2 retained the same scientific cases but
used a measured 512 MiB own-memory ceiling, 512 MiB remaining physical-memory
floor, at least 768 MiB available before loading, 6 GiB commitment headroom and
5 GiB free disk. Version 1 and its receipts remain intact.

Two initial attempts stopped on resource guards before loading the model.
After the owner freed memory, bounded admission waits required five consecutive
seconds with at least 1.25 GiB available and sufficient commitment headroom.
The micro and continuation probes then passed their unchanged version 2 guards.
There is no continuing resource monitor or automatic retry process.

The input probe observed 224.1 MiB peak working set. The continuation probe's last
periodic snapshot, at 10.3 seconds, recorded a peak of 205.4 MiB; its final six
seconds do not have a separate peak-memory receipt. These measurements are not
a worst-case production memory bound.

- Frozen version 1: `tools/simulation_speed_probe_128/`,
  `outputs/simulation_speed_probe_128_01/`.
- Frozen version 2: `tools/simulation_speed_probe_128_v2/PROTOCOL.md` and
  `outputs/simulation_speed_probe_128_02/plan.json`.
- Version 2 binding:
  `87eb986685357c79c18c2b2f587350228f1fe964932be3b0279d323dcc40ee7b`.
- Results: `outputs/simulation_speed_probe_128_02/micro.json` and `rollout.json`;
  admission and hold receipts are preserved beside them.
- Both versions passed their 11 focused checks before use. A separate read-only
  audit reproduced the timing totals, 48-case inventory and 1,390 comparisons,
  and verified all 308 bound source hashes, nine historical root hashes and
  associated references. It ran no further simulations or fresh evaluation.

The alternative learner recipes and weights had already been sealed before
this investigation. Their separate result-readiness receipt is
`outputs/alternative_ranker_128/readiness.json`; their test remains unevaluated.
At 16:24 Paris, the original hidden supervisor (PID 6312) was live and the
alternative registration identity check again confirmed bound dependencies
unchanged and no fresh outcomes read.

## Interpretation and recommended next engineering step

Input preparation is a demonstrated optimization target. The observed whole
continuation gain is useful, with substantial variation by phase. This short
sample covers only three roots, one CPU worker and ordinary terminations;
concurrent host activity and cache effects can affect timing. It does not cover
complete 128-world roots, all candidates, unusual terminations, multiworker
throughput or full collection overhead. It is not evidence for a 5-10x speedup.

Keep the prototype isolated. The next engineering step should be a bounded
integration/equivalence trial in a separate future collector: complete
128-world roots, high-action-count cases, exceptional termination and
private-information checks, followed by multiworker memory and net-runtime
measurement. Preserve the original implementation as the independent reference.
That trial has not been launched, and this report does not authorize automatic
adoption or changes to the active experiment.

This improvement would make future simulation work cheaper. It does not speed
the current neural fitting stage or resolve the scientific question of learning
generalizable improvements from planning.
