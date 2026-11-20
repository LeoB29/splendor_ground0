# Repeat-timing diagnostic: completed and independently verified

27 October 2026. The single frozen attempt completed all 144 ordinary requests
with exact results, zero timeouts, and zero 700 ms crossings. Both players and
the one-second complete-response deadline remained unchanged. No games, model
training, player optimization or larger-data work ran.

## Results

Six saved positions × both players × twelve rounds, interleaved with balanced
root and arm order. The three controls were selected by saved position ID within
early/middle/late strata, without consulting timings. All observations count.

| Player | Decisions | Median receipt | p90 receipt | Maximum receipt | ≥700 ms | Deadline failures |
|---|---:|---:|---:|---:|---:|---:|
| Unchanged optimized64, three worlds | 72 | 439.5 ms | 512.6 ms | 631.3 ms | 0 | 0 |
| Unchanged six-world candidate, 64 simulations | 72 | 495.8 ms | 562.2 ms | 646.4 ms | 0 | 0 |

The original three slow observations did not recur in twelve repeats each:

| Original slow player/position | Prior receipt | Repeat median | Repeat maximum |
|---|---:|---:|---:|
| Optimized64 / reserved_06 | 736.9 ms | 432.7 ms | 521.0 ms |
| Six-world / gold_returns | 772.3 ms | 528.1 ms | 562.2 ms |
| Six-world / reserved_00 | 714.3 ms | 485.7 ms | 532.1 ms |

Across the balanced panel, handler medians were 366.2 ms for optimized64 and
371.1 ms for the candidate. Mean paired candidate-minus-control handler duration
was 3.75 ms. The mean complete-response difference was 54.43 ms. Response emission
accounted for a measured mean difference of 44.60 ms; median emission was 1.47 ms
versus 50.07 ms. Median raw response sizes were 5,727 versus 40,606 bytes. These
are observed boundary durations, not a controlled causal attribution: emission
also includes possible scheduling and transport waiting. The candidate's extra
callback evidence was retained exactly, as required.

Every successful request matched its historical reference. Eleven player/root
cells used saved ordinary responses. The incumbent gold/returns cell used the
accepted actual-entrypoint result with only its extra startup inference-capture
arrays excluded; every common numerical value, action and callback hash remained
exact. All twelve new results in every cell matched the first new response in
full, including candidate float32 callback bits. Nothing was retried or omitted.

## What this result establishes

The three selected positions do not consistently require the previously observed
slow durations. This run provides direct evidence of timing variation while
preserving the exact decisions and outputs. It also confirms a recurring
secondary cost in the candidate's larger response transmission; the handler
costs are similar on this small balanced panel.

It does not identify the cause of the earlier delays. Both workers were resident
here, whereas the previous admission tested them serially. Repeats are temporally
dependent, and the targets were selected because they were slow. The panel does
not estimate general timeout risk, prove live reliability, or demonstrate
stronger play. No confidence interval treats the repeats as independent trials.

The previous admission remains **not_admitted_verified** under its original
700 ms criterion. This diagnostic's clean timings do not change that result or
constitute a new admission. All prior evidence remains intact.

## The next step

Recommend one separately frozen fresh admission attempt with both players
unchanged, the same 700 ms engineering margin and one-second hard deadline.
Use genuinely fresh reserved positions and run six fresh operational games only
if the qualification passes. Preserve any failure and stop according to the
prospective rules. The 200-game strength comparison remains conditional on full
admission, operational and verification success.

The present evidence does not justify a transport change before that attempt:
the measurable emission overhead left sufficient margin on this panel, and it
cannot explain the previous incumbent slowdown. A fresh admission attempt is a
recommendation, not something launched by this diagnostic. External-opponent
testing remains paused and larger-data work remains deferred.

## The broader picture

The project is improving the verified SplendoRust-based player within one second
per move. Earlier saved-position work found that sampling six hidden-card worlds
produced more stable choices than three while keeping 64 total search simulations.
That is a promising decision-stability signal, not proof of stronger play.

The earlier live admission was stopped by timing-margin misses before games
began. This diagnostic narrows that concern: the same decisions can be reproduced
with substantial headroom, and their recurring handler cost is similar between
players. The next fresh admission and live games test whether that headroom holds
on unseen positions. Only a verified paired whole-game comparison can establish
whether six worlds improve playing strength. Optimized64 remains the frozen
incumbent throughout.

## Verification and preservation

- 65 model-free tests passed before freezing; held source hashes unchanged.
- 104 actual corrected-entrypoint decisions, 22 graph cases, 144 ordinary
  decisions and 148 parent requests were verified.
- All 144 historical comparisons and 132 subsequent within-cell comparisons
  passed, with independent corrected observer/transition checks.
- Raw transport, trusted clocks, sidecars, placement, compressed evidence,
  source identities, resident-stage deadlines and resource accounting passed.
- All three actual owner exits were zero; all fifteen recorded owned process
  identities were absent after completion.
- Charged execution: 251.282 seconds (4 minutes 11 seconds), within the 3,600-second
  cap. Peak aggregate worker/helper RSS: 615,849,984 bytes (about 587 MiB).
- All 7,065 frozen inputs, 667 prior-attempt artifacts and 939 sealed artifacts
  totaling 38,066,879 bytes were authenticated. Earlier failures remain preserved.
- Independent saved-evidence review completed without errors or model calls.

Protocol: `docs/engineering/splendorust_pool6_repeat_timing_20261027.md`.
Evidence: `outputs/splendorust_pool6_repeat_timing_20261027_01`.
Full per-request and per-cell timing distributions: `analysis.json` in that
evidence directory. All twelve observations and same-round paired differences
are retained there.

Freeze SHA256:
`69c7fb88b503c8a4ea3138d107b90f7e189ddbfbffc45127f83167fc0e5ddbfa`.
Independent completion review SHA256:
`4860ab1b0cb876275b8a6a48b47f4a17f32d6725aba711c14fa1c0d26e3eea8d`.
The accepted review is `postrun/independent_completion_review.json`; it accepts
this diagnostic only and explicitly grants no game/admission authorization.
