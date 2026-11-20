# Batching benchmark result — 2026-09-21

The existing CPU batching prototype preserved every tested continuation but
narrowly missed the prospectively fixed adoption screen. Use the already-tested
serial backend for the authorized learned128 versus original128 study. Do not
round the result into a pass, change the cutoff, add timing repetitions, or start
the conditional 128-world batching integration on this evidence.

## Correctness

Version 3, binding
`5e468e2f3e4722cd6b3a96b979a7dcdee4df3720b516aa404f823485e097f66e`,
completed all26 fixed saved positions across serial and batch sizes1/4/16.
Each mode collected and independently replayed3,136 continuations. All104 root
executions matched the saved scientific fields; complete action/state traces and
final states were identical across modes. Independent closure verified520 unique
artifact hashes and104 receipts. Actual full batches numbered97,332 /24,172
/5,902 for sizes1/4/16. Runtime was57m29s.

Equivalence result:
`outputs/batching_benchmark_v3/equivalence_01/result.json`, SHA256
`1e7d96ba9477d498dd1b92a4a733dafb73e9cf37996889884ccb9e9cd49572de`.

## Fixed timing comparison

All96 measured root runs completed, with two separate excluded warmups. The
12 predetermined positions were repeated four times with balanced serial/batch16
order. Each collection was followed by the unchanged complete serial audit.

| Summed measured seconds | Serial | Batch16 |
| --- | ---: | ---: |
| Collection |770.2696245 |618.6218244 |
| Independent full audit |782.3665892 |782.6427467 |
| Collection plus audit |1552.6362137 |1401.2645711 |

The collection speed ratio is **1.2451381346706236**, below the predeclared
minimum1.25. The collection-plus-full-audit ratio is **1.108025026623804**,
above its minimum1.10. Both conditions were required, so the adoption screen
does not pass. These correspond to about19.7% less collection time and9.7% less
combined time on this fixed workload. The timing stage took51m18s including
warmups, verification and bookkeeping.

Timing result: `outputs/batching_benchmark_v3/timing_01/result.json`, SHA256
`8c7c89e53f656ee3321b5fbd959ca417f3c8c660615ec2b3ea1d1637fc3eb3a4`.

## Interpretation and next action

This near-cutoff engineering decision is not evidence that batching lacks value,
nor a statistically established difference between1.245 and1.25. It enforces the
previously chosen bounded-work rule. Retain the implementation, complete evidence
and conditional integration plan for possible later work. No numerical mismatch
or scientific-strength failure was observed in this benchmark.

The measurements cover one CPU worker, one Torch thread,32-world saved roots and
complete serial audits. They do not establish128-world/eight-worker production
speed, equal-wall-time playing strength, universal equivalence, or speed under
the new study's sampled-audit policy. Shared rules, encoders and neural primitives
remain trusted dependencies.

Versions1 and2 and their failures are preserved. Version1 stopped before model
loading because Windows reused completed-study PIDs for other applications.
Version2 stopped at a serial comparison of in-memory integer keys against saved
JSON string keys; all persisted scientific fields matched. Version3 made the
strict comparison between persisted records, retaining every original position,
world, proposal, trace comparison and full audit.

Record the completed engineering decision at the previously agreed
`outputs/batching_benchmark_v2/adoption_decision.json`, linking the actualv3
evidence and independent closure. Freeze and launch the serial learned128 study
under the owner's existing authorization. Its hidden detached supervisor runs
the separate12-game operational probe, then the fixed402-game comparison and
prescribed sampled verification if the operational gate passes, and exits.
No additional training, promotion, successor study or recurring monitoring.
