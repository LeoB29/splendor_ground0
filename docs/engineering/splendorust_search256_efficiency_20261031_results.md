# Search256 saved-position efficiency diagnostic: verified results

31 October 2026. The separately frozen diagnostic completed successfully. It
locates most measured response time inside compiled neural-network evaluation.
It does not establish a particular kernel fix, one-second readiness, or a new
strength result. Unchanged128 remains the reference; verified256 remains the
stronger challenger.

## Execution and acceptance

The model run and final saved-evidence reader finished in about 7.6 minutes,
within the three-hour cap. There were no failed requests, automatic retries,
new games or training. Preparation failures remain preserved.

- All 123 model-free tests passed before freezing.
- Both worker routes passed 134 scientific qualification decisions and 22
  graph cases in total, including the corrected-rule entrypoint checks.
- All 192 ordinary measurements exactly reproduced the complete saved
  scientific outputs: 48 overhead measurements, followed by 144 panel
  measurements. Matching only the chosen move was not sufficient.
- The independent reader verified all 2,373 frozen source/input pins, coverage,
  raw evidence, scientific results, clocks, placement and timing records.
- Runner, watchdog, reader and guardian all exited successfully; final owner
  absence and cleanup are bound in the separate closeout acceptance receipt.

The observer qualified before the full panel: paired median overhead was
**0.885% / 10.86 ms**, with a one-sided 95% bootstrap upper ratio of **1.0220**.
All three preset overhead gates passed. No estimated overhead was subtracted
from any trusted response clock.

## What was measured

The unchanged accepted256 player used the same checkpoint, float32 graph,
single thread, batch size one, three worlds, exploration setting 0.4, native
search binary, placement and corrected rules. The one-second target and 700 ms
margin were descriptive. The unchanged 9-second internal and 10-second complete
response guards remained in force.

Eighteen distinct saved games supplied 18 positions selected from all 11,066
eligible decisions. Twelve systematic controls covered early/middle/late play
and both seats. Six additional positions were deliberately selected historical
slow cases. Each received four paired observer-OFF/ON repetitions in the main
diagnostic. These two groups must remain separate; neither their pooled mean
nor their repetitions are an independent sample of whole-game performance.

| Main diagnostic measurement | 12 systematic positions | 6 historical slow positions |
|---|---:|---:|
| Repetitions per observer mode | 48 | 24 |
| Mean response, extra observer OFF | 1.2005 s | 1.2522 s |
| Maximum response, extra observer OFF | 1.3130 s | 1.4035 s |
| Responses over one second, observer OFF | 44/48 | 24/24 |
| Mean response, extra observer ON | 1.2204 s | 1.2512 s |
| Compiled network evaluation, observer ON | 0.9519 s | 0.9738 s |
| Network evaluation share of complete ON response | 78.0% | 77.8% |

The small OFF/ON reversal for the slow group is measurement variation, not
evidence that instrumentation has negative physical cost. The separately
frozen overhead test is the admission result.

The observer-ON mean elapsed-time partition is:

| Non-overlapping component | Systematic positions | Historical slow positions |
|---|---:|---:|
| Compiled network evaluation | 951.86 ms | 973.77 ms |
| Tensor conversion and inference interface | 44.50 ms | 45.36 ms |
| Scientific callback bookkeeping outside inference | 59.30 ms | 60.52 ms |
| Native exchange bookkeeping outside callbacks and I/O | 50.43 ms | 51.66 ms |
| Native read/wait | 12.11 ms | 12.82 ms |
| Native write | 25.21 ms | 25.90 ms |
| Native flush | 0.42 ms | 0.33 ms |
| Other handler work | 7.69 ms | 7.54 ms |
| Response time outside the handler | 68.84 ms | 73.31 ms |
| Complete response | 1,220.38 ms | 1,251.20 ms |

Rounded rows may differ slightly from totals. Nested timers were converted to
exclusive spans before summing. Native process CPU averaged about 30 ms and
27 ms respectively; it is a separate measurement, not another row to add to
elapsed time.

Seventeen positions required 257 network evaluations; one required 184. Each
evaluation is small, but repeating it during search accounts for most of the
decision's elapsed time. A saved-input check found only two repeated complete
inputs among 4,553 evaluations across the 18 distinct decisions. Simple
within-decision input caching therefore has little opportunity on this panel
(at most 0.044% of calls). Cross-decision caching was not evaluated.

## Interpretation and limits

**Network execution is the main measured target.** Compiled forward evaluation
accounts for about 78% of the complete response and 83% of the handler. Even
eliminating the measured time outside the handler would leave mean handlers
above one second. Concentrating first on supervision alone would miss the
largest cost.

This is elapsed time inside the network stage, not proof that every millisecond
is arithmetic. Fine-grained Windows thread-CPU increments were too coarse for
reliable component attribution: for example, measured native-write thread CPU
exceeded its short wall span. Do not interpret component CPU shares, or wall
minus CPU, as a measurement of scheduling delay or paging. The independently
verified wall-time partition remains useful.

The historical roughly 3-7 second spikes did not recur here; all slow-position
OFF repetitions finished within 1.404 seconds. That does not establish their
original cause or guarantee that they will not recur. The panel also does not
replace the previous whole-game timing distribution or qualify deployment.

## Recommended next step

One bounded **exact-output optimization study of single-position network
execution** is justified. Begin with a small operator-level cost measurement
inside the existing fused transformer blocks, then choose and test one concrete
execution change only if those measurements support it. Require full numerical
outputs and decisions to remain exact, retaining weights, float32 precision,
single-thread policy, search, lawful inputs and guards. Measure genuine elapsed
gains with the extra observer disabled before considering operational games.

The saved graph already contains six fused transformer encoder operations.
Another generic compilation pass or removal of Python module overhead is not
an evidenced solution. This aggregate diagnostic cannot honestly identify a
specific kernel or executor rewrite yet. Simple input caching is also a poor
primary target given the observed lack of repeated inputs. Stop the feasibility
study if no meaningful exact-output improvement is found; do not turn it into
another indefinite timing-admission campaign.

No optimization, fresh game, 512-simulation sweep or reference promotion has
been launched. External testing, six-world work, training and larger-data work
remain deferred. Scheduled chat monitors remain cancelled.

## Evidence and continuity

- Protocol: `docs/engineering/splendorust_search256_efficiency_20261031.md`.
- Frozen evidence: `outputs/splendorust_search256_efficiency_20261031_01`.
- Freeze SHA256:
  `1bc4796d0578fa3082d2f5f940ce366b3f899876ee06551f6750c5e047c555f3`.
- Panel SHA256:
  `5a41ce0a8863f42c1550ae849a148f5a1659bf114476ff2debabeaae4d9caada`.
- Complete measurements: `diagnostic_analysis.json`; observer admission:
  `overhead_result.json`; saved-only full review: `independent_review.json`.
- Separate final acceptance, actual guardian exit and independent interpretation:
  `outputs/splendorust_search256_efficiency_20261031_01_closeout`.
- Previous strength result:
  `docs/engineering/splendorust_search256_recovery_20261030_02_results.md`.

The previous 400-game study established a 61.0% score for256 against128. This
diagnostic locates most of256's measured thinking time. Together
they justify attempting to preserve the demonstrated strength gain while making
execution cheaper; they do not yet establish that256 can reliably meet one second.
