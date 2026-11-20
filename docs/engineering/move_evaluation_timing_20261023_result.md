# Replacement timing diagnosis completed

23 October 2026. The actual-model instrumentation gate and all three fixed-root
diagnostics passed. The owned job ran from12:16:16 to12:50:22 Paris in34m06.359s,
within its60-minute cap. Execution exited0 with no cleanup errors. The original
120-second qualification remains failed; the earlier cProfile failure remains
preserved. This is a timing diagnosis, not evidence of improved playing strength.

Protocol: `docs/engineering/move_evaluation_timing_20261023.md`.
Evidence: `outputs/move_evaluation_timing_20261023_01/`.
Freeze SHA256:
`f59bba8cce763e7997d41016accdb1c38fff8107e0738c0586d618f47cd33aa0`.
Administrative verification: `outputs/move_evaluation_timing_20261023_status_01/`.

## Accepted checks and timing

The ordinary and instrumented actual-model fixtures exactly reproduced the
accepted15-case result, including18 proposal encoder checks across all three
fits and the three-transition terminal trajectory. Predetermined call counts
matched. The replacement uses selected Python events without replacing any
scientific function. All full requests also passed their trace-derived call
census and exact root-time partition checks; boundary gaps were10.0,8.7,8.7µs.

| Proposal fit | Saved game/ply | Ordinary response | Instrumented response | Full replay | Replayed transitions |
| --- | --- | ---: | ---: | ---: | ---: |
| 161001 | 25/8 | 226.438s | 211.833s | 228.434s | 22,362 |
| 161002 | 57/14 | 183.014s | 186.418s | 218.357s | 21,554 |
| 161003 | 58/4 | 207.472s | 234.805s | 242.659s | 24,062 |

Each root retained four candidates and128 worlds. Ordinary and instrumented
decisions and compressed traces matched exactly, excluding only their output
path. Each independent audit replayed all512 continuations, reexecuted both
policies and the proposal model, and checked transitions and selection arithmetic.
Total:1,536 continuations and67,978 transitions, including33,989 Entity choices
and32,453 raw035 choices. The third root made12,029 Entity inferences for12,031
choices; two single-choice branches correctly required no inference.

Root0 also exactly matched the prior completed result and all296 complete
original interrupted records/12,888 transitions. The incomplete original tail
was preserved and excluded. The corrected-rule and catalogue bindings are
unchanged. The audit shares authenticated legality/model kernels but expresses
transitions through the frozen independent dictionary-ledger oracle.

The ordinary requests still exceed120s: these three cases need roughly34–47%
less elapsed time to fit that allowance. These are three early seat0 development
positions, not qualification of the full18-position panel. One ordinary/timed
pair per position cannot estimate instrumentation overhead or host variability;
the lower instrumented time on the first case is not a speedup claim.

## Valid measured costs

Fractions below use the instrumented selected-root duration. Regions are
scope-exclusive: they include unselected descendants, native work, waiting and
instrumentation overhead. They are not pure kernel times or measurements of
uninstrumented shares. Inclusive times overlap and are not added.

| Selected region | Fit161001 | Fit161002 | Fit161003 |
| --- | ---: | ---: | ---: |
| Corrected observer reconstruction | 20.19% | 20.87% | 20.63% |
| Corrected legal-action enumeration | 20.09% | 18.47% | 19.68% |
| Entity inference pipeline | 28.02% | 28.96% | 28.93% |
| Shared raw/proposal attention forward | 5.63% | 5.80% | 5.72% |
| Native exchange residual, excluding selected Entity inference | 5.65% | 5.81% | 5.73% |

Observer reconstruction plus legal enumeration account for39.35–40.32% of
instrumented root time. Their call counts reconcile with the scientific trace:
each Entity choice causes five bridge state/legal preparations and six observer
reconstructions in total, including final legality verification. Root-level
decoding adds one more reconstruction. Full legal calls were111,939,107,899 and
120,439. Some legality work is essential; these shares cannot all be removed.

The ordinary workers used roughly162–183s of Python-process CPU and2.08–2.50s
of native-child CPU, measured from initialization to the final receipt sample.
Python-process CPU includes native tensor kernels. Ordinary peak tree memory
was274–277MiB. Trace writing/compression/finalization was below0.20% of selected
root time in every case. Storage or trace removal is not the leading speed target.

## Recommendation

Prepare a separate exact-behavior optimization of repeated observer and legal
preparation within one Entity decision. Reuse a validated public decision context
through encoding, Main-action construction, auxiliary resolution and final
legality checks. Preserve action order, tie handling, privacy, all corrected
rules, policies, four candidates and128 worlds; avoid stale state or cross-turn
caches. Do not edit any frozen adapter or player.

The next version should first pass the same actual-model counterexamples and
exact-output/trace checks, then full independent replay and the unchanged
120-second qualification across the18-position panel and all three models.
Whether this optimization alone is sufficient is unproved; further work on the
Entity inference pipeline may be needed. No optimization, new assessment,
strength game or training was launched. Larger-data work remains deferred.
