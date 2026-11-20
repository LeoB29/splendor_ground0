# Move-evaluation study: stopped at the timing gate

23 October 2026. **Implemented and launched; full qualification did not pass.**
The first complete four-candidate decision reached its 120-second deadline
without delivering a result. The supervisor stopped the run and preserved the
partial evidence. The independent fresh-world assessment never started.

## What passed

- All 151 model-free prelaunch tests passed.
- Actual model-bearing correctness checks passed on all 15 fixed fixtures,
  including the six corrected card costs, noble awards, gold/returns, private
  cards, forced/single actions and the turn-123/124 boundary.
- All three proposal models passed the actual encoded-input checks: 18
  model/face combinations. Native policy mapping, masks, first-maximum ties,
  auxiliary phases and a three-move mixed-policy terminal trajectory passed.
- The assessment policies completed all 30 fixture requests within one second.
  The slowest complete response was 0.2760835 seconds.

These are finite fixture results. The required full panel decisions, privacy
repeats and complete trajectory replays remain incomplete.

## Exact stopping event

| Item | Recorded result |
|---|---|
| First panel root | Game 25, ply 8; proposal fit 161001 |
| Required work | Four unchanged candidates × 128 worlds = 512 trajectories |
| Complete-response deadline | 120.0000000 seconds |
| Accepted response | None |
| Supervisor elapsed time | 120.0998616 seconds, including timeout processing |
| Sampled worker-tree CPU | 105.921875 seconds |
| Sampled peak worker-tree memory | 279.07 MiB, below the 2 GiB cap |
| Preserved compressed trace | 3,646,517 bytes; incomplete gzip stream |
| Recoverable complete JSON trace records | 296, containing 12,888 recorded moves |

The recoverable records are an incomplete prefix, not accepted or independently
replayed trajectories. No partial action was selected. The remaining panel roots
were not attempted after the failure. In particular, full-decision timing for
fits 161002 and 161003 was not measured.

The recorded failure is a deadline timeout, not a RAM/disk hold. This does not
identify which implementation component dominates the cost. The entire launched
attempt lasted about 3 minutes 24 seconds, including memory admission and setup;
it did not exhaust the eight-hour outer allowance.

All authenticated runner, worker and native process identities were confirmed
closed. All 2,024 frozen input artifacts were rehashed unchanged after the stop.
The original benchmark, its timeout loss and the saved postmortem are unchanged.

## Research implication and next step

The alternative evaluator is **not qualified for the 120-second allowance**.
There is no new estimate of move quality or playing strength, and this timeout
is not an assessment loss or a game loss. The study recorded no assessment score.

The next useful step is a separately bounded performance diagnosis of this exact
evaluator. Measure the cost of neural inference, observer/native conversion and
evidence writing before choosing an optimization. Any revised implementation
should preserve the fixed candidates, all three models, the 128-world workload
and exact scientific traces, then repeat qualification under a new disclosed
plan. No repair, retry, larger allowance or larger-data collection was launched.

## Evidence

- Execution plan: [move_evaluation_execution_20261023.md](move_evaluation_execution_20261023.md).
- Run directory: `outputs/move_evaluation_20261023_01/`.
- Freeze SHA256: `d059b8b2dbf5ebebc0543a4df8cbbe94b1a92bc5526a5356d15c3dcf98e30d8e`.
- Actual fixture receipts: `entrypoint_qualification.json` and
  `assessor_qualification.json` in the run directory.
- Timeout: `workers/selection/r00/request-00000001.json`.
- Preserved failure and cleanup: `failure.json` and `execution.json`.
- Administrative read-only closeout and independent inspection:
  `outputs/move_evaluation_20261023_status_01/`.

The closed run directory remains untouched by this report and administrative
closeout. Larger-data collection remains deferred.
