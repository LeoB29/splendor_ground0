# Host diagnostic v3: completed and independently verified

27 October 2026. The separately versioned correction passed its saved-failure
tests, actual-model correctness qualification, overhead gate and full diagnostic.
All 216 ordinary responses were valid and arrived within one second. Two full-stage
responses crossed the descriptive 700 ms safety margin: one from each player.
The run and independent verification are complete; no experiment remains running.

The players, weights, precision, search, saved roots, request order, one-second
deadline and resource limits remained unchanged. Both earlier incomplete attempts
and their failures remain preserved. No game, training job, retry or replacement
request ran. This accepts a diagnostic, not a player admission or strength result.

## Correction and qualification

Version 3 makes an inconsistent host-wide CPU estimate explicitly unavailable
while retaining every raw reading, offending interval and independently valid
process/memory measurement. An inconsistent whole window cannot produce a trusted
CPU estimate by selecting apparently consistent endpoints or a shorter subspan.
It never clamps, discards or resets the raw counters. API, type, timestamp,
identity and process-counter failures remain fatal.

The exact v2 failure at sequence 193 was tested through a clearly labelled
model-free projection. Its original failed record was not changed or accepted.
All 157 combined tests passed in 10.15 s, including forged availability, endpoint
recovery, subspan reporting, valid process/RAM retention and the earlier signed
context-aggregate regressions. Sources were held unchanged through the tests,
independent review and freeze.

Both actual workers then passed their corrected-rule entrypoint suites:
104 scientific decisions and 22 exact graph cases in total. They retained the
original 1.000-second complete-response clock and 0.900-second internal guard.

## Overhead gate

Each player completed 18 observer-OFF and 18 observer-ON requests in the frozen
order. Both passed all three limits: at most 5% added mean, 10 ms added mean and
25 ms added nearest-rank p95. No observation was discarded or replaced.

| Player | OFF mean | ON mean | OFF p95 | ON p95 | Gate |
| --- | ---: | ---: | ---: | ---: | --- |
| Optimized64, three worlds | 369.378 ms | 360.103 ms | 432.017 ms | 381.579 ms | Passed |
| Six-world candidate | 392.608 ms | 389.816 ms | 444.341 ms | 419.811 ms | Passed |

Lower ON means in this sample do not show that the observer improves speed.
The gate establishes acceptable measured overhead under the recorded conditions.

## Full diagnostic

All 18 saved roots were evaluated four times by each player: 144 requests, following
the 72-request overhead stage. All 216 ordinary decisions completed correctly.

| Player | Full-stage requests | Mean | p95 | Maximum | At least 700 ms | One-second timeouts |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| Optimized64, three worlds | 72 | 463.426 ms | 614.604 ms | 727.092 ms | 1 | 0 |
| Six-world candidate | 72 | 509.577 ms | 692.460 ms | 704.618 ms | 1 | 0 |

The 700 ms threshold was explicitly descriptive in this diagnostic. Crossing it
does not make the diagnostic incomplete, and completing the diagnostic does not
reverse either earlier negative admission.

All 216 observer lifecycle records were valid: 180 enabled and 36 disabled. All 180
enabled windows had available, consistent host-wide CPU readings in this run,
including their recorded in-request spans. There were 1,573 native samples and
three valid negative live-thread aggregate changes. Thus the new host-CPU fallback
was validated against the saved failure and counterexamples, but no new native
CPU inconsistency required it during this run. Exact interval context-switch
totals remain unavailable, as designed.

## What the two slow decisions show

| Saved request | Root and player | Receipt | Handler | Emission | Minimum sampled RAM |
| --- | --- | ---: | ---: | ---: | ---: |
| 80 | g2_p36, optimized64; original slow target | 727.092 ms | 657.243 ms | 1.075 ms | 11.406 GiB |
| 170 | g2_p45, six-world candidate; fixed control | 704.618 ms | 565.716 ms | 75.339 ms | 11.100 GiB |

Both retained the fixed 64 simulations and 65 inference callbacks. Exact scientific
outputs also matched their saved and repeated references. The first root's other
three same-player full-stage responses were 462.385, 446.195 and 452.375 ms. The
second root's other three same-player full-stage responses were 472.582, 504.522
and 450.872 ms; its earlier overhead responses were 383.748–395.018 ms.

Compared with the median of the other three full-stage repetitions, the first
slow response added 274.717 ms, while its handler added 278.000 ms and emission was
lower. The second added 232.037 ms overall, including 204.831 ms more handler time
and 27.696 ms more emission. Most of the exceptional delay was therefore inside
evaluation. Transport alone cannot explain both events.

The completely in-request sample spans retained 375.000 ms of active-player CPU
over 697.129 ms for request 80, and 406.250 ms over 621.648 ms for request 170. Their
host-wide busy fractions were 54.68% and 29.26%. These are sampled intervals,
not exact handler CPU or scheduler traces. The saved observations show elapsed
time not accounted for by active-player CPU within those spans, but do not
identify runnable-queue delay, blocking, frequency changes or a specific outside
process. CPU consumption also varied across repetitions, so a pure scheduling
explanation would be too strong.

The slow events occurred amid nearby slower decisions from both players.
Available RAM was ample: minimum before any request 11.048 GiB and minimum in all
native samples 11.038 GiB. Initial stability took 60.078 s, with no recovery wait.
Low available RAM is therefore not necessary for a 700 ms crossing. It may still
have contributed to earlier runs; this study did not randomize memory pressure.
Page-fault counters combine soft and hard faults and cannot establish paging.

The six-world player also has a persistent emission cost: full-stage mean
48.575 ms versus 5.805 ms for optimized64. Its mean handler cost was 401.262 ms versus
397.449 ms. That is a separate, potentially useful optimization target, but does
not explain the incumbent's 727 ms event with only 1.075 ms emission.

## Verification and preservation

The independent audit accepted `completed_verified`. It authenticated all 13,596
frozen inputs, 96 runtime files, 1,495 immediate-predecessor artifacts and 1,592
sealed evidence files totaling 60,328,596 bytes. The inherited v1 evidence remained
pinned and preserved. It verified 320 scientific decisions (104 startup plus 216
ordinary), 220 parent requests, 22 graph cases, 108 exact historical comparisons
and 180 exact first-observation repeats. All saved transitions, raw transport,
lossless logs, source identities, placement receipts, optional-value calculations
and overhead arithmetic reconciled.

All three actual owner exits were zero, all 15 owned process identities were
absent and cleanup errors were empty. Charged execution was 328.102 s, about
5 minutes 28 seconds, within the two-hour cap. Peak combined worker/helper RSS was
615,821,312 bytes. Independent review used only saved evidence: no models, games
or test reruns.

Prelaunch reviewer-only identity/path assumptions and a postrun root reader
assumption about the location of the selection field are disclosed in their
separate notes. They did not change experimental source, evidence or results.
Supplementary saved-evidence diagnosis is separate from the original experiment
seal and does not change its accepted scope.

## What this result establishes

The measurement and verification repairs are qualified, and the entire bounded
diagnostic completed. Both unchanged players can reproduce every selected
decision within one second under these conditions. However, rare 700 ms crossings
recurred despite ample memory, including on a fixed control position. Memory
headroom alone is insufficient to guarantee the desired margin.

Large variations on identical decisions and slower nearby requests from both
players support testing execution conditions next. They do not prove a specific
CPU scheduling, thermal, frequency or background-process cause. This selected,
dependent saved-position panel is not a live-game reliability or strength sample.

## The next step

Prepare a bounded, balanced CPU-placement comparison on the two newly slow
positions plus fixed controls, using both unchanged players and exact outputs.
Compare player CPU 0 with CPU 2, on distinct physical cores of the same highest
efficiency class in the saved topology. Keep one logical CPU per player/helper,
normal priority, unchanged numerical settings, the one-second deadline and 700 ms
margin. Put supervisors on the same CPUs 4–19 in both prospective arms, excluding
both test cores and their siblings 0–1/2–3. Revalidate topology before freezing.

The common supervisor mask differs from this completed run's CPUs 2–19, so both
comparison arms must be measured afresh: this run is not its experimental
control. Counterbalance order to limit time/load drift and check measurement
overhead before any new observation. Placement does not reserve a core against
external applications; the result would test sensitivity to placement, not prove
the cause of operating-system interference.

Do not launch another admission or strength comparison first. If a placement
offers reproducible margin while preserving exactness, qualify that arrangement
on fresh admission positions and operational games before returning to the
paired strength benchmark. This is a recommendation only; no new placement
version, admission or experiment has been implemented or launched.

## The broader picture

The project is testing whether distributing the same 64 search simulations over
six hidden-card worlds improves decisions over optimized64's three worlds while
remaining practical within one second per move. Saved-game work suggested more
stable choices, but live admissions exposed occasional slow responses. We now
have a complete, verified diagnostic showing that these delays can recur even
with substantial RAM available and fixed scientific work.

The next investigation should improve the reliability of execution before
spending time on a playing-strength comparison. It preserves the learned network
and search policy so that any later strength result measures the six-world
change cleanly. Optimized64 stays frozen, external-opponent testing stays paused,
and larger-data collection/training remains deferred.

## Evidence

- Protocol: `docs/engineering/splendorust_pool6_hostdiag_20261027_03.md`.
- Source: `tools/splendorust_pool6_hostdiag_20261027_03`.
- Output: `outputs/splendorust_pool6_hostdiag_20261027_03`.
- Freeze SHA256: `d3c4231c8676822dfff40a59d7f5abc8b9d0eef8b6089582cc8cc2fce7f794d1`.
- Implementation review SHA256: `40646c3a9cf0eaa99c3c3ff8c70f316034ae27ae1d8ef215880fd73ac062d706`.
- Prelaunch review SHA256: `1fa0d3aa5886e3ab950d551816de5b27ae95cc384e4e733e59d8d39b4ba62fdf`.
- Acceptance: `postrun/independent_acceptance_review.json`, SHA256
  `a78c75e516bd1fe01148448befdfb6bb86f172de5d1b27bc3225f72210b65321`.
- Results: `overhead_gate.json`, `analysis.json`, `scientific_audit.json`,
  `verification_summary.json`, `completed_evidence_seal.json`, `completion.json`.
- Supplementary diagnosis: `postrun/slow_move_diagnosis.json`, SHA256
  `e48e4e7b578f354d7146de99f71c42911f39d65f9337e9099e75caffa3affd43`. Its reader-only
  path-type note is retained as `postrun/slow_move_reader_note_01.json`.
- Slow decisions: `records/0080.json`, `records/0170.json` and matching
  `host_observations` files. Counts and timing include both without trimming.
