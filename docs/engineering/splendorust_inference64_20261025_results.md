# 64-simulation inference qualification: passed

25 October 2026. The optimized **64-simulation player qualified**, with the
32-simulation reference unchanged. The frozen run completed all timing,
operational and verification stages in **7 minutes 53 seconds**, with no move
timeouts, margin misses, scientific mismatches or automatic retries.

This qualifies the tested implementation for a future strength benchmark. It
does not establish that 64 simulations play better than 32.

## What stayed fixed

The existing optimized execution graph was used only at 64 simulations. Its
weights, float32 precision, three hidden worlds, search, native binaries,
corrected rules, legal-information interface and runtime stayed unchanged.
The original 64-simulation execution served as the timing comparator; the
original 32-simulation player served as the operational opponent. The optimized
graph was forbidden at 32 simulations.

Ordinary decisions retained the **1.000-second complete-response deadline**,
0.900-second internal limit and stricter **0.700-second admission margin**.
Startup and verification remained charged to the common 60-minute allowance.

## Saved-position results

All **144 paired requests** completed, covering 24 development positions and
12 fresh reserved positions, with reversed execution order across two passes.
Actions and full scientific traces matched across execution modes and repeats:
36 exact groups and 724 checks against applicable historical 64-simulation
records. Historical 32-simulation answers remained provenance only.

| Timing panel | Original 64 mean | Optimized 64 mean | Reduction | Original / optimized p95 | Result |
|---|---:|---:|---:|---:|---|
| Development | 372.47 ms | 349.25 ms | 6.23% | 407.75 / 381.85 ms | Pass |
| Fresh reserved | 342.44 ms | 324.75 ms | 5.17% | 385.98 / 370.73 ms | Pass |

Both panels independently passed the frozen requirements: at least 5% lower
mean, no p95 regression, and every candidate request within 0.700 seconds. The
slowest optimized saved-position arrival was **0.385 seconds**. No deadline or
margin was relaxed after results were seen.

The 12 reserved observer states were absent from earlier selected panels and
reproduction fixtures, with one position per distinct namespaced source pair
and balance across stage and seat. Some source games had supplied other earlier
positions. These are fresh reserved positions, **not 12 wholly untouched source
game pairs**. Selection used identifiers, not timings or outcomes.

## Six fresh operational games

All three fresh initial-world pairs completed with seats swapped: **six natural
endings and 328 complete moves**, split equally between the players.

| Player | Moves | Mean arrival | Slowest arrival |
|---|---:|---:|---:|
| Optimized 64 simulations | 164 | 336.71 ms | 406.40 ms |
| Unchanged 32 simulations | 164 | 213.02 ms | 252.29 ms |

Every move remained below both the 0.700-second admission margin and the
1.000-second deadline. All six full games passed independent dictionary-ledger
replay. The first move by each exact player in each game was reproduced:
**12 exact reproductions**, retaining six optimized64 and six original32 modes.
Game outcomes were not used for qualification and are not a strength estimate.

## Correctness, evidence and resource checks

- All **125 model-free preparation tests** passed.
- Three actual corrected-entrypoint suites passed, each completing 52 decisions:
  original64, optimized64 and original32.
- All **14 optimized worker instances** passed the original 11 numerical cases:
  exact outputs and float32 bits, with six archived root outputs and supported
  batch controls. All shared the prior graph and weight identities.
- Final checks reread 472 saved-panel/game decisions, the 12 reproductions,
  29 worker logs and cleanup receipts, and a seal of 1,763 evidence artifacts.
  The frozen sources and runtime were verified again.
- Peak sampled aggregate worker/helper memory was about **580 MiB**. Evidence
  occupied about **50 MiB**, well within the fixed 512 MiB allowance.
- Started **01:58:04 Paris**, finished **02:05:58 Paris**. The owning watchdog
  exited successfully with clean cleanup; its PID and the runner PID were absent.

The initial preparation in output directory `_01` failed before freezing or
model calls: 269 imported artifact paths were absolute while the verifier
expected project-relative paths. That failure, its plans and all 12 matching
source snapshots are preserved. The correction changed only those metadata
paths and their derived hash. Independent review confirmed identical selected
positions, scientific data, timing requests, parity inputs and operational seeds.
The strictly verified never-launched schedule was explicitly exempted from the
seed scan. The actual frozen execution is in `_02`; it ran once.

An independent post-run audit also passed. It recomputed timing arithmetic from
all 144 raw receipts, rechecked the 328 transitions and 12 original-mode
reproductions, verified 154 exact output/bit pairs, 87 lossless log streams,
all 1,763 sealed artifacts and 3,799 frozen inputs, and confirmed both preserved
negative/preparation records. It made no model calls or new games. Its additive
receipt and source provenance are `independent_postrun_audit.json` and
`independent_postrun_audit.py` in the run directory; original evidence is intact.

The earlier joint32/64 qualification remains a verified **negative** result.
This narrower qualification does not rewrite it or alter the unchanged32 player.

## Interpretation and limitations

The speed gain repeated: approximately 23 ms per development decision and 18 ms
per reserved decision. The reserved result exceeds the 5% engineering threshold
by only about 0.17 percentage points. These are finite-sample timing measurements,
not a guarantee under every future machine load. The old large latency spikes
were not reproduced in these tests, so this does not prove their cause or cure.

The present evidence is sufficient to pass the prospectively fixed qualification
for this exact configuration. Further tuning against this reserved panel would
require a new independent assessment. Larger-data collection and training remain
deferred; no strength benchmark was launched by this run.

## The next step

Prepare and freeze a **fresh 200-game comparison: optimized64 versus unchanged32**,
using 100 initial-world pairs with seats swapped. Preserve the one-second limit,
existing verification and attributable-timeout-as-loss policy, and existing
success criteria. Use fresh seeds; keep these six operational games outside the
strength estimate. Set the bounded runtime/storage plan from measured costs and
report paired uncertainty. Launch requires the next user instruction.

## The broader picture

The goal is a much stronger Splendor player that can act reliably in roughly one
second. We already found that 32 simulations improved on 16. This qualification
makes it practical to test the next question: does 64 improve on 32 enough to be
worth the extra thinking? The network and weights stay fixed, so that comparison
can isolate the value of additional search before investing in training.

## Reproducible references

- Protocol: [frozen design](splendorust_inference64_20261025.md).
- Run: `outputs/splendorust_inference64_20261025_02`.
- Freeze SHA-256: `0f3ce62b9ed4cdf53ed540d3b02efb0c1d2a4d55dc87bf9324c0cb7fae714df2`.
- Graph identity: `c4d572908ddb3c216689c702bb00367dc263c9f1374be8bd397250e0b65e6ba2`.
- Weight-state identity: `32cdd9466befc527d265a356be65200c3666a4fa8a82cfefdc40b327b2f529db`.
- Primary receipts: `qualification_summary.json`, `operational_summary.json`,
  `all_entrypoint_checks.json`, `all_graph_parity.json`, `all_game_replay.json`,
  `repeated_decision_summary.json`, `verification_summary.json`,
  `qualification_outcome.json`, `completion.json`, `execution.json`.
