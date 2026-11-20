# Saved-position timing repeat: verified results

29 October 2026. Complete and independently verified. Both players are unchanged.

All 120 measured decisions completed below 0.9 seconds, with no hard-guard failure, illegal move or scientific mismatch. None of the ten formerly slow 128-simulation roots exceeded one second in any of its three repetitions. The actual owned run, including its independent reader, finished in 225.94 seconds; final external acceptance was recorded within the same one-hour allowance. No process remains running.

## What this result establishes

The original extreme delays are not an unavoidable computational cost of those particular positions. Identical 128-simulation results, including callback inputs/outputs, actions and traces, were reproduced much faster. Formerly slow roots were not consistently costlier than systematic controls in this run.

This does not identify a scheduler, RAM, inference or communication cause. Original timings and current timings were collected at different times and with different worker lifetimes. Selecting the original ten extremes also creates regression-to-the-mean risk. These twenty selected roots and three repetitions do not establish a future-game timeout rate or one-second deployment reliability.

## Measured complete-response timing

All times below use the original trusted complete-response clock, before parent decoding/validation. Thresholds are descriptive. The research guards remained 9 seconds internally and 10 seconds at complete-response arrival.

| Player and panel | Decisions | Median | Maximum | Above 1 s | Above 700 ms |
|---|---:|---:|---:|---:|---:|
| 128 — Formerly slow | 30 | 0.712 s | 0.752 s | 0 | 22 |
| 64 — Formerly slow | 30 | 0.380 s | 0.436 s | 0 | 0 |
| 128 — Systematic controls | 30 | 0.732 s | 0.816 s | 0 | 28 |
| 64 — Systematic controls | 30 | 0.397 s | 0.483 s | 0 | 0 |

Across all 60 decisions per player, 128 had a 0.725-second median and 0.816-second maximum; optimized64 had a 0.388-second median and 0.483-second maximum. All 50 crossings of 700 ms were retained normally. They are not failures and do not trigger another admission gate.

The formerly slow 128 roots had a 0.652-second median handler duration, compared with 0.668 seconds for controls. Handler duration includes search, inference, inner communication and bookkeeping; it is not an isolated inference or CPU measurement.

## The ten original slow positions

Each triple below is chronological repeat 1 / 2 / 3, in seconds. The complete source rows and all boundary intervals are retained in the frozen panel and analysis.

| Slow rank | Original 128 | New 128 repeats | New 64 repeats |
|---|---:|---|---|
| 1 | 2.012 | 0.703 / 0.734 / 0.716 | 0.368 / 0.373 / 0.385 |
| 2 | 1.752 | 0.716 / 0.712 / 0.733 | 0.366 / 0.387 / 0.373 |
| 3 | 1.622 | 0.748 / 0.700 / 0.699 | 0.367 / 0.366 / 0.374 |
| 4 | 1.600 | 0.752 / 0.691 / 0.713 | 0.367 / 0.366 / 0.394 |
| 5 | 1.585 | 0.749 / 0.723 / 0.732 | 0.370 / 0.379 / 0.382 |
| 6 | 1.563 | 0.707 / 0.709 / 0.749 | 0.399 / 0.384 / 0.392 |
| 7 | 1.557 | 0.698 / 0.726 / 0.732 | 0.374 / 0.436 / 0.385 |
| 8 | 1.533 | 0.684 / 0.689 / 0.719 | 0.399 / 0.370 / 0.379 |
| 9 | 1.448 | 0.697 / 0.703 / 0.712 | 0.416 / 0.379 / 0.400 |
| 10 | 1.433 | 0.698 / 0.706 / 0.719 | 0.385 / 0.392 / 0.390 |

The original worst move took 2.012 seconds; its three repeats took 0.703, 0.734 and 0.716 seconds. All ten old slow roots have a recurrence count of zero out of three for both complete response and handler durations above one second.

## Fixed systematic controls

| Original game | Original 128 | New 128 repeats | New 64 repeats |
|---|---:|---|---|
| 20 | 0.832 | 0.727 / 0.760 / 0.722 | 0.364 / 0.395 / 0.483 |
| 60 | 0.643 | 0.721 / 0.741 / 0.729 | 0.386 / 0.388 / 0.398 |
| 100 | 0.692 | 0.697 / 0.743 / 0.753 | 0.395 / 0.369 / 0.422 |
| 140 | 0.693 | 0.706 / 0.771 / 0.722 | 0.383 / 0.380 / 0.439 |
| 180 | 0.670 | 0.754 / 0.762 / 0.733 | 0.424 / 0.397 / 0.443 |
| 220 | 0.682 | 0.712 / 0.816 / 0.731 | 0.414 / 0.401 / 0.432 |
| 260 | 0.649 | 0.698 / 0.734 / 0.728 | 0.420 / 0.389 / 0.411 |
| 300 | 0.672 | 0.738 / 0.768 / 0.747 | 0.478 / 0.377 / 0.379 |
| 340 | 0.636 | 0.713 / 0.742 / 0.732 | 0.433 / 0.397 / 0.372 |
| 380 | 0.655 | 0.717 / 0.726 / 0.741 | 0.411 / 0.402 / 0.385 |

## Correctness and preservation

- 95 model-free implementation tests passed before freezing.
- The two actual workers passed a combined 119 startup scientific decisions and 22 graph-parity cases.
- All 120 ordinary decisions passed corrected snapshot, lawful observation/action and independently checked one-step transition validation.
- All 60 new 128 results exactly matched their historical full scientific results. All 80 later-repeat comparisons matched their own cell reference, covering 40 three-repeat cells. No historical same-root 64 target was assumed.
- Two retained worker trees accounted for all 124 parent requests. Raw requests/responses, lossless compressed logs, graph evidence, actual-entrypoint forensics, placement, clock boundaries, source/runtime pins and the finite evidence seal passed both primary and fresh-process readback.
- Runner, watchdog, independent reader and guardian actual exits were successful. Recorded process identities were absent at final acceptance.
- Aggregate worker/helper peak RSS was 620,658,688 bytes; the independent reader peaked at 113,074,176 bytes. Evidence occupied about 45 MiB. No new games, model fitting or training-data collection occurred.

An initial preparation-only failure incorrectly required an absolute external runtime file to be inside the workspace. Its exact source and receipts were preserved, the narrow verifier fix passed a new regression and the full 95-test set, and freezing occurred afterward. A later root receipt inspection used the wrong field name (`review_artifact` instead of the existing `review`); this failed inspection is also retained. It changed no experiment evidence, launched no model and wrote no acceptance. Corrected inspection authenticated the actual reader exit and its existing review before acceptance. There was one frozen experimental attempt and no scientific retry.

## The next step

Recommend one small, separately fixed comparison of newly initialized, graph-qualified workers against retained workers on the same saved roots, using unchanged 128 and the existing observer and research clocks. Interleave the conditions closely and counterbalance their order. Qualify the actual entrypoint separately; use the original game-style graph startup for measurement trees, rather than warming every fresh tree with the entire counterexample suite. Declare, record and validate the retained worker history, and count all setup/warmup work. Preserve exact scientific-output checks.

This tests a specific remaining lifecycle explanation before choosing an optimization. A newly initialized worker is not a recreation of an original midgame prefix; any such prefix would need to be explicitly included. Either outcome remains a result about the declared present conditions, not proof of the historical host mechanism. Do not add a new observer, change transport, launch a large game study or promote 128 automatically. This next experiment is recommended only; it is not prepared, frozen or launched here.

## The broader picture

The accepted 400-game comparison already showed a useful strength gain from 128 simulations: 223 wins, 3 draws and 174 losses against optimized64, for a 56.125% score and a paired 95% interval of 51.5–60.75%. The current question is whether that extra strength can be delivered consistently near the one-second deployment target. The saved analysis located the extreme delays mainly inside the handler; this repeat shows that identical work usually finishes much faster under the present execution conditions. That supports investigating how the player is run before assuming the search or network needs redesign.

Optimized64 remains the practical frozen baseline. The strength result is unchanged, and no new strength or reliability claim is made. The six-world branch, external-opponent testing and larger-data/training work remain deferred.

## Evidence

- Freeze: `223ecb1964b25630e27d07464f0ce71e695bd3ea09f05e780355049a6e158732`; 441 finite pinned inputs.
- Acceptance: `ff2dfa01f429d6b8d2117969506823f7a17778607794a9760bef470f2797aab8`.
- Independent reader: `2494873498b2ec0e28b9d0ef38d34d1a7b74dc47c675b357af925c410518947e`.
- Directory: `outputs/splendorust_search128_repeat_20261029_01`.
- Main artifacts: `analysis.json`, `coverage.json`, `independent_review.json`, `independent_review_os_exit.json`, `root_exit_observation.json`, `acceptance.json`.
- Root observed actual guardian exit 0 through retained session26917, final tool chunkf22453. Correct final external acceptance command exited0, chunk5120c4.
