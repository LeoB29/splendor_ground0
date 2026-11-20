# Both separated players qualified; six fresh operational games passed

26 October 2026. The new prospectively frozen qualification completed successfully. Both optimized64 and unchanged32 passed under separated supervision, followed by all six fresh games and the full verification sequence. The original one-second deadline, players, checks and earlier failed attempts remain unchanged. No further experiment is running.

## Result and scope

Freeze: `cc5f19859865e41844689deee48cf1e089db59fdbb6b46af14da3711ac34d2e4`, binding 4,885 inputs. Separate implementation: `tools/splendorust_supervisor_qualification_20261026`. [Frozen protocol](splendorust_supervisor_qualification_20261026.md). Evidence: [completion](../../outputs/splendorust_supervisor_qualification_20261026_01/completion.json), [owning execution](../../outputs/splendorust_supervisor_qualification_20261026_01/execution.json).

The run began at 00:25:17 Paris and finished at 00:36:58 Paris on 26 October, using 700.937 charged seconds (11 minutes 41 seconds) against the frozen one-hour cap. The owning watchdog exited successfully, cleanup reported no errors, and the watchdog, runner and owned worker identities were independently confirmed absent.

This version changed prospective qualification handling: a genuine original-placement control timeout would remain a failed observation, be preserved and audited, and allow only the next unattempted request in a new worker. It would not cancel the isolated tests. There were **no control timeouts in this attempt**, so no continuation worker was needed. The continuation path passed model-free regression checks but was not exercised by a real timeout here. The earlier failed qualification and its control timeout retain their original status.

## Correctness and saved-position timing

All four actual-entrypoint suites passed: both quotas under both placements, 52 decisions each, **208 correctness decisions**. Corrected-rule startup guards, privacy, legal support, card faces, noble, payment/return, end-round and historical-decoder rejection checks remained active. Every graph worker passed the original eleven exact numerical/float32-bit cases and graph/weight identity checks.

All **192 scheduled timing attempts** completed: 24 saved positions, two players, two placements, two passes. All 96 isolated decisions were below 700 ms. Full scientific outputs, actions, callback traces and independent resulting states matched in all 48 four-way state/quota cells; all 92 applicable historical comparisons passed, comprising 46 isolated and 46 control checks. Nothing was omitted, replaced or retried.

| Player and placement | Decisions | Mean | 95th percentile | Slowest |
| --- | ---: | ---: | ---: | ---: |
| optimized64, separated | 48 | 0.3320 s | 0.3689 s | 0.3835 s |
| optimized64, original control | 48 | 0.3332 s | 0.3757 s | 0.4207 s |
| unchanged32, separated | 48 | 0.1904 s | 0.2096 s | 0.2146 s |
| unchanged32, original control | 48 | 0.1915 s | 0.2099 s | 0.2212 s |

Both primary timing gates passed. The complete-control relative comparisons were also within their descriptive thresholds: separated/original mean ratios 0.99616 for 64 and 0.99403 for 32. These relative comparisons were not admission gates in this new freeze. The selected, failure-enriched saved panel is not an independent estimate of general reliability, and these small differences do not prove that CPU separation removed the earlier cause.

## Fresh operational games and replay checks

All **six fresh games**, three swapped-seat pairs, finished naturally. All **338 live decisions** completed, 169 per player. There were no timeouts, forfeits, missing timing receipts or moves at or above the 700 ms engineering margin.

| Separated player | Live decisions | Mean | 95th percentile | Slowest |
| --- | ---: | ---: | ---: | ---: |
| optimized64 | 169 | 0.3313 s | 0.3866 s | 0.4229 s |
| unchanged32 | 169 | 0.1970 s | 0.2212 s | 0.2682 s |

The clock remained a real **1.000-second complete-response deadline**, with the same 0.900-second internal guard. The 700 ms threshold is engineering headroom, not a timeout or loss rule. No time was subtracted from decisions. The fixed network, precision, quotas, three hidden-world samples, search behavior and corrected rules were preserved. Only supervisor/watchdog CPU placement differs from original controls; unrelated applications are not restricted.

The deterministic stratified replay rule selected **38 completed decisions**, all of which reproduced exactly under the original clock. The plan allowed up to 48: only 38 of its position categories were populated in these games. Earlier conversational progress updates used 48 as the denominator; the verified final count is 38 of 38 selected, with no omitted selected replay. All game ledgers, scientific outputs and transitions were independently checked. The operational score was 3-3; these six games are qualification evidence, not a playing-strength estimate or basis for promotion.

## Verification, resources and preservation

The final checks covered 62 workers, 31 graph workers (341 exact graph fixtures), 603 worker request receipts, all raw transport and compressed logs, placement/thread ownership, worker lifecycle, corrected runtime/source identities and source preservation. The exact completed seal contains **4,392 artifacts / 79,294,810 bytes**. Maximum sampled combined worker/helper memory was 554,577,920 bytes (about 529 MiB), below the 2 GiB ceiling; this is a sampled maximum, not a continuous peak guarantee. RAM/disk/evidence guards and the fixed time caps remained enforced.

The supplementary [independent saved-evidence review](../../outputs/splendorust_supervisor_qualification_20261026_01/independent_results_review.json) passed in 20.43 seconds. It verified the exact seal, recomputed timing/scientific checks, audited saved game transitions and reproduction selection/output equality, checked worker inventories and absent identities, and read back 4,467 consumed input hashes unchanged. It made no model calls, new decisions or evidence changes. Its raw/graph scope binds the original final-validator receipts and every sealed byte; it does not independently rerun graph computation or the full historical ancestry. Review SHA256: `df09400ed6a5b3313c951344bb6abe6bb783f14ef89af3494ea637009b572d82`. [Review execution receipt](../../outputs/splendorust_supervisor_qualification_20261026_01/independent_review_execution.json).

Before freezing, 284 combined model-free tests and the implementation review passed. An initial 283-pass/one-failure test attempt is retained with the exact failing test source and explanation: a source comparison incorrectly included an explanatory module docstring; the final check compares executable syntax while ignoring only that docstring. No experiment had launched when it was corrected. The previous stopped separation attempt, original live failure and all historical losses remain preserved; this new pass does not relabel them.

## Recommended next step

Prepare a fresh bounded **200-game live reliability trial**, 100 swapped-seat pairs, with both supervisors separated and the same players, one-second limit, timeout losses and verification. This exceeds the exposure of the earlier live failure, which appeared after roughly 79 games including the operational prelude. Retain the first-failure stop and verification of the played prefix; preserve slower valid moves without mislabelling them as timeouts. Define the 700 ms margin's prospective admission consequences explicitly in the new freeze.

Use a refreshed forecast and fixed resource limits before launch. The present six games used about 223 seconds of game collection, about 2.1 hours if scaled linearly to 200, before separate startup/verification/resource-wait allowances. This is a planning extrapolation, not a promised duration; a six-hour overall cap is a reasonable candidate to assess. No longer trial has been frozen or launched by this result report.

A clean longer reliability result would support proceeding to a separately frozen playing-strength comparison. A renewed failure would provide another preserved record to diagnose. Neither outcome licenses widening the deadline or discarding failures.

## The broader picture

The project is developing a strong, verified Splendor player that can deliver its moves within roughly one second. After the earlier rollout planner lost decisively to our verified SplendoRust adaptation, the roadmap shifted to building on that pretrained policy/value network and adaptive search. The current question is whether the extra search in optimized64 is useful and practical relative to unchanged32, while preserving exact behavior through runtime improvements.

The immediate obstacle has become rare timing failures, rather than ordinary average computation time. This qualification establishes that both separated players are correct and comfortably fast on this panel and short live run. It does not establish that the old stalls are cured or that 64 simulations are stronger. Sustained reliability is the bridge back to a meaningful strength comparison. Larger-data collection, training and network changes remain deferred.
