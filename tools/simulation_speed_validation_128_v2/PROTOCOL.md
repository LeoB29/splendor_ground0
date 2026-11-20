# Speed validation resource amendment: tolerate low available physical RAM

On 2026-10-16 the owner explicitly requested that validation keep running when
available RAM drops. This authorizes a new resource policy, not a silent edit
to the previous frozen study. Preserve all code, plans, attempts and results in
`tools/simulation_speed_validation_128` and
`outputs/simulation_speed_validation_128_01` unchanged.

The new code is `tools/simulation_speed_validation_128_v2`; its output is
`outputs/simulation_speed_validation_128_02`. Freeze the amendment, exact code,
prior plan and prior charged active-time receipts before launch. This protocol
supersedes only the old resource rules and associated timing interpretation.

## Unchanged scientific and engineering workload

Retain the original frozen validation's six metadata-selected historical roots,
model, reference encoder, prototype encoder, source/world seeds, candidates,
action order, all exactness requirements and counterbalanced timing schedule.
There are 440 untimed selection and 22 untimed assessment continuations, then
eight complete selection timing passes of 7,040 continuations and 243,536 saved
turns each. No extra input audit belongs inside a timed pass. No prior partial
correctness/timing block is reused. Never read current fresh scientific results.

All startup, root-process loading, dispatch, result verification and output costs
remain within the same prescribed timing boundary. Preserve complete-block
recovery, atomic immutable receipts, duplicate-run exclusion, controller leases,
hidden workers, one numerical thread per worker and below-normal priority.
No workload, hyperparameter, model, success criterion or production code changes.

## Amended resource policy

Available physical RAM is now **telemetry, not an admission or stop condition**.
A dip below 512 MiB or 128 MiB is recorded but does not pause or end the job.
The old five failures were triggered by physical availability despite roughly
8-12 GiB of commitment headroom and approximately 0.2-0.3 GiB of validation
working set. They do not establish that allocation was about to fail.

Keep at least 6 GiB of observed system commitment headroom while working.
Before starting a block, require that headroom plus its full planned allocation
budget: 256 MiB parent plus 512 MiB per worker (768 MiB for one worker, 1,280 MiB
for two). Require five healthy admission samples, with a 180-second maximum wait.
These are monitored headroom checks, not an OS memory reservation.

Continue checking both current and peak resident memory against 512 MiB per
worker and 256 MiB for the parent. Additionally enforce the same per-process
limits on current private committed memory, so paging cannot hide excessive
allocation behind a small resident footprint. Enforce the 1,280 MiB combined
process limit on both resident and private committed memory. Keep 5 GiB free disk.
Allocation errors or any other hard guard failure stop the affected block and
preserve its evidence; they never become successful timing observations.

The four-hour cumulative active budget includes charged time from every prior
version-1 attempt plus all new attempts. No allowance is silently reset. Keep
the original current-study phase firewall: coexist only during permitted
training/data-validation/replay phases or after completion. Do not modify,
suspend, restart or terminate the original experiment or any unrelated app.

## Interpretation and reporting

Record physical availability, system commitment headroom, own resident/private
memory and worker overlap. Summarize observed physical minima and sample/time
shares below 512 MiB and 128 MiB. These pressure indicators do not independently
prove page-file I/O; do not claim observed paging without direct evidence.

Report each prescribed repeat and the aggregate matched net-wall ratios, plus
summed worker CPU ratios. Include every completed prescribed block regardless
of how slow it became; no selective exclusions or reruns to obtain nicer timing.
Only genuinely interrupted/failed blocks are rerun whole, with all partial
evidence retained. Never combine timing fragments across attempts.

These are runtime measurements under the concurrent host load, potentially with
memory pressure. Balanced order does not guarantee equal external pressure for
the two implementations. Neither the wall nor CPU ratio alone isolates intrinsic
encoder speed. Exactness evidence remains meaningful even when comparative
timing is inconclusive. A slower desktop or main study is possible through
shared resources despite their code and settings being unchanged.

The amendment enables only this bounded validation. It does not enable the
prototype in production, start a successor, change Windows paging settings or
create an ongoing monitoring automation.

Background reference: Microsoft's [Introduction to page files](https://learn.microsoft.com/en-us/troubleshoot/windows-client/performance/introduction-to-the-page-file)
explains that physical availability and the system commitment limit are distinct.
The local failure receipts, not that general reference alone, motivate this
specific bounded amendment.
