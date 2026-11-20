# Saved game timing postmortem for optimized64

25 October 2026. The saved evidence points to intermittent execution delays as
the next investigation target. The slow completed moves did not make unusually
many network evaluations, and several delays affected both players in the same
part of the run. The records cannot yet separate inference stalls, supervision,
dispatch, response reading and operating-system scheduling sufficiently to
choose a corrective optimization.

The recommended next step is a small timing diagnostic on the current graph64
backend, with instrumentation overhead checked first. Preserve the 64-simulation
player, the unchanged32 control, both deadlines and the benchmark's two losses.
Increasing search or collecting training data would not answer this timing
question.

## Scope and preservation

The postmortem read all 11,153 main-game move ledgers and all 200 game results
from the completed optimized64-versus-unchanged32 benchmark. Every consumed
ledger/result matched its original seal. Counts and timing summaries reconciled
with the saved benchmark performance report. This was analysis of existing
records: no model calls, new games, fitting, player modifications or changed
failure classifications.

The plan selected all 20 graph64 cases: two failed decisions and 18 completed
responses above the descriptive 0.700-second engineering margin. The one eager32
margin miss was a secondary case. Controls comprised nearest preceding/following
decisions by each role in the same game and hash-selected completed decisions
from other games matched on role, seat, game phase and opponent private-reserve
presence. Hash selection allowed slow completed decisions and used no latency
ranking or game results. Twenty distinct graph64 hash controls were selected;
none happened to be a case. The combined panel contains 104 distinct positions:
68 graph64 and 36 eager32. Missing neighbors after forfeits remain missing.

The analysis script and selection plan were frozen before the population read.
Execution took 5.74 seconds under a 300-second internal and 420-second external
cap. New evidence was about 12.6 MiB before final reports, below 32 MiB. A separate
raw-log review checked 39 selected artifacts, and a source review checked ten
frozen files. This is a scoped postmortem, not a restart of the earlier incomplete
supplementary full audit.

## The two failures were different

| Saved event | What actually happened | What remains unknown |
| --- | --- | --- |
| Game index 12, ply 56 | The 0.900-second internal guard detected expiry immediately after a neural inference returned. A complete error packet arrived at 0.9752681 s, with no action or complete trace. | Which prior interval consumed the allowance; the failed callback's duration and partial callback count were not saved. |
| Game index 66, ply 29 | A complete, scientifically valid action with one search phase, 64 simulations and 65 neural callbacks arrived at 1.0466894 s. It remained rejected by the one-second clock. | How much delay occurred in initial supervision/dispatch, child work, final write/flush or parent reading/scheduling. |

The second response was a visible reservation of tier 2, market index 2. Its
5,710-byte raw packet was complete and passed the existing scientific validator.
The first packet was a 168-byte error, not a playable answer. Both workers had
intact compressed logs, empty stderr and clean recorded cleanup. There is no
truncation evidence in these records.

The internal and external clocks cover different intervals. The external clock
starts before initial parent sampling and request dispatch. The internal clock
starts when the child has read the complete request. Its last check follows JSON
serialization but precedes stdout write/flush; the parent stamps arrival after
reading the complete response line. Therefore game 66 spent **more than 146.7 ms
outside the child's internally limited interval**: 1.0466894 minus less than
0.900 seconds. This is a derived bound, not a measured breakdown of that delay.
Compression of the same response occurs after its arrival stamp and cannot
explain that recorded timestamp.

## Work counts do not explain the slowest responses

| Graph64 group | Moves | Mean arrival | Mean sampled worker CPU | Median neural calls |
| --- | ---: | ---: | ---: | ---: |
| All completed moves | 5,575 | 385.1 ms | 287.6 ms | 65 |
| Hash-selected matched controls | 20 | 384.1 ms | 289.1 ms | 65 |
| Slow completed cases | 18 | 782.1 ms | 392.4 ms | 65 |

The 18 slow completed cases used 63 to 65 neural calls. Their median native
legal-action count was 22, compared with 21 across all completed graph64 moves.
Their median canonical result size was 5,665 bytes, compared with 5,664 bytes
overall. Five had a second auxiliary phase; some corrected legal-action sets
were larger than usual, but these features do not explain all cases. Counts do
not measure the duration of individual neural calls or all other work.

Sampled CPU did increase in the slow group, but much less than elapsed time.
The failed requests recorded 0.359375 and 0.34375 seconds of worker-tree CPU.
That does **not** make the remaining elapsed time a measured waiting delay:
the sample excludes the supervising parent, uses different endpoints from the
arrival clock, is coarse, and can lose final CPU increments from exited child
processes. Game 12's final sample no longer contained its native descendants.

## The delays cluster in time and affect both players

Nine of the 18 slow completed graph64 moves occurred in game index 8. The
unchanged32 player's only response above 0.700 seconds occurred in that same
game, immediately before the first graph64 margin miss. In game 8, graph64
averaged 640.6 ms and eager32 averaged 404.5 ms, versus 385.3 and 238.6 ms over
all attempted main decisions. Their average neural-call counts in game 8 were
61.29 and 31.11, close to their overall 61.31 and 31.52.

The 20 graph64 cases occupied eleven games. Thirteen occurred in games 8 to 13,
six in games 65 to 75 and one in game 199. Both players had higher average
response times in the corresponding first and fourth fixed 20-game blocks.
This pattern supports investigating shared runtime conditions. It does not
prove a particular application, RAM shortage, CPU frequency change or scheduler
event caused them; historical host-load measurements are absent. These are
retrospective descriptive comparisons, not independent causal tests.

## Proposed next diagnostic

Use the saved 104-position panel in its original roles and quotas. A separate
version should add lightweight timestamps to distinguish:

1. Parent clock start, initial supervision, request dispatch and complete-line
   arrival.
2. Child complete-input arrival, handler start/end and final pre-write boundary.
3. Inference wall/CPU totals and the longest call, native read/write intervals,
   and parent supervision/reader work.
4. The last reached boundary and partial work counters when a request fails.

Keep overlapping spans separate. Preserve the one-second arrival deadline,
0.900-second internal guard, exact weights, float32, one numerical thread,
affinity, search, raw reader, corrected rules and all safety checks. Do not
subtract instrumentation time from a deadline. Validate cross-process timestamp
ordering and retain failure-side information without treating late actions as
accepted decisions.

A concrete small design is 48 baseline/profiled overhead requests on twelve
preselected control positions, followed conditionally by 416 diagnostic requests
(104 positions, two modes and two reversed paired passes). Check each quota's
overhead separately against the previous gates: at most 10% mean increase,
25 ms added mean and 50 ms added p95. Apply actual corrected-entrypoint suites,
per-process graph parity, exact scientific-output comparisons for completed
pairs, saved-input verification and cleanup. An unfinished historical move has
no historical complete trace to match; compare newly completed mode pairs while
retaining the original failure. Repetition does not resume the benchmark.

Proposed overall limits are 30 minutes including qualification and verification,
128 MiB of new evidence and the existing memory safeguards. The final resource
forecast and exact request/worker inventory must fit before freezing that new
diagnostic. This is a recommendation, not an implemented or launched experiment.
If it does not reproduce a stall, report that limit; a successful replay alone
does not establish that rare failures are cured.

The earlier component diagnostic measured eager inference and established the
normal inference cost. It led to the exact graph optimization. This follow-up
would measure the current backend and locate rare delays across the two clock
boundaries. The failed buffered-reader version remains unqualified; source
evidence alone does not justify promoting it or selecting another optimization.

## Place in the research programme

The 64-simulation player has already demonstrated a substantial direct advantage
over unchanged32: 64.25% score, including both deadline losses. The current
question is how to deliver that strength reliably within one second. A narrowly
targeted timing diagnosis preserves the strategic gain while identifying a
defensible reliability intervention. Larger-data learning remains deferred;
later strength claims must also survive independent opponents and expert-human
evaluation.

## Evidence

- [Completed strength result](splendorust_graph64_strength_20261025_results.md)
- [Postmortem plan](../../outputs/splendorust_graph64_tail_postmortem_20261025_01/plan.json)
- [Population and grouped analysis](../../outputs/splendorust_graph64_tail_postmortem_20261025_01/analysis.json)
- [Fixed case and control panel](../../outputs/splendorust_graph64_tail_postmortem_20261025_01/case_control_panel.json)
- [Consumed source hashes](../../outputs/splendorust_graph64_tail_postmortem_20261025_01/source_manifest.json)
- [Raw failure audit](../../outputs/splendorust_graph64_tail_postmortem_20261025_01/failures_raw_audit.json)
- [Clock semantics and checked source files](../../outputs/splendorust_graph64_tail_postmortem_20261025_01/source_semantics_review.json)
- [Analysis execution](../../outputs/splendorust_graph64_tail_postmortem_20261025_01/execution.json)
- [Earlier component diagnostic](splendorust_component_timing_20261025_results.md)
