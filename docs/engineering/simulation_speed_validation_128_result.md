# Completed saved-workload speed validation

Completed 2026-10-17 at 00:36:37 Europe/Paris. All prescribed blocks completed
under the frozen v2 resource amendment. Independent receipt verification passed.
The current main experiment was not changed and the prototype is not enabled in
production.

## Result

Aggregate ratios use the sums of both prescribed repeats within each worker
count. All successful blocks are included; no timing was selected or excluded.

| Workers | Reference wall seconds | Prototype wall seconds | Throughput ratio | Wall time saved | Worker-work CPU time saved |
| --- | ---: | ---: | ---: | ---: | ---: |
| 1 | 2465.701 | 1806.345 | 1.365x | 26.74% | 25.95% |
| 2 | 1253.016 | 974.661 | 1.286x | 22.21% | 24.18% |

One-worker repeats saved 24.17% and 29.24% wall time. Two-worker repeats saved
20.41% and 24.25%. All four matched comparisons favored the prototype. The
time-saving percentages and throughput increases describe different denominators:
26.74% less time corresponds to 36.50% more throughput.

## Exactness and workload

All 462 untimed correctness continuations passed, including 15,492 exact
intermediate model-input comparisons. Each of eight timed blocks completed the
same six historical roots, 7,040 continuations and 243,536 saved turns. All final
states, action traces, termination reasons, winners and turn counts matched the
saved references exactly. There were 56,320 timed continuation executions across
repeated passes; these are repeated measurements of one fixed workload, not
56,320 independently selected positions. No extra tensor audit ran inside the
timed blocks.

Independent verification authenticated the frozen plan, all nine stage receipts,
root and resource hashes, expected row/count inventories and exact final-stage
references. No complete successful block was omitted or duplicated.

## Scope and resource conditions

This supports the prototype's correctness and a material runtime reduction on
the prescribed saved selection workload with the unchanged 035 continuation
policy. It is not an overall training-speed, new-position, playing-strength or
superhuman-performance result. It does not measure independent-assessment timing
or production integration overhead outside the prescribed timing boundary.

Net wall measurements include process startup, model/root loading, dispatch,
simulation, result verification, output and worker exit. The main learning study
ran concurrently. Balanced order and consistent CPU savings strengthen the
engineering result but do not isolate intrinsic encoder speed from every host
effect; two repeats do not establish a precise general performance interval.

None of the eight timed blocks sampled available physical RAM below 512 MiB;
the smallest sampled availability was 3,541,037,056 bytes. Earlier interrupted
attempts remain preserved and charged, with no timing fragments reused. The
four-hour cumulative active limit was respected: 7,135.956 seconds total,
including 215.398 seconds carried from v1. Physical availability remained
telemetry-only; commitment, own-memory, disk, phase and time limits stayed fixed.

## Evidence

- Plan: `outputs/simulation_speed_validation_128_02/plan.json`.
- Binding: `2cfc98c4d26d4a208114fda89caa761ddec144a86549a2cc36d18c4408ab38e2`.
- Completion and all stage references: `outputs/simulation_speed_validation_128_02/final.json`.
- Frozen amendment: `tools/simulation_speed_validation_128_v2/PROTOCOL.md`.
- Last launch and startup evidence: `resume_02.json` and `resume_02_startup.json`
  in the same output directory. Supervisor39820 exited normally after completion.

No additional experiment, production adoption or main-study modification was
performed when reporting this result.
