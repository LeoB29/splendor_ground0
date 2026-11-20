# Dom PPO512 screen: second attempt with expanded evidence allowance

27 October 2026. **Completed and independently accepted as a negative screen.**
The tested Dom policy is not admitted as a competitive opponent.

Evidence: `outputs/dom_screen_20261027_02`. Protocol:
`docs/engineering/dom_screen_20261027_v2.md`. Freeze SHA256:
`41bec2b2ea61e05aeb0b549d60858fa726dbae53839e3b37c56401ff4fe32492`.

## Authorized amendment

The user authorized another attempt with a much larger cap after the first
attempt passed qualification and six operational games but could not admit the
main screen. Its final-verification forecast exceeded its reserved allowance.
That attempt remains an accepted verified operational prefix, with zero main
games, and has not been relabelled as a completed screen or a strength result.

This separate version raises the evidence cap from 1 GiB to **4 GiB**, with
**2 GiB reserved for finalization** and 2 GiB available before finalization.
The one-second complete-decision deadline, four-hour job cap, three-hour
collection cutoff, stage caps, RAM safeguards, 1.5x forecast method, admission
threshold and verification checks remain unchanged. This changes disk-file
allowances, not the memory used by the players.

The exact Dom player profile and qualification panel match the first attempt.
Weights, arithmetic, lawful hidden-card completion, decisions and the unchanged32
reference are preserved. Scientific-source differences are limited to new module
paths and line endings. The independently ported source encoder and literal
browser forward remain the references; the upstream Rust engine was not run.

The fresh schedule contains six operational games and 60 main games in swapped
seats, using 33 new paired worlds. A new 256-bit namespace and separate schedule
and reproduction domains were frozen before play. The bounded registry checked
690 historical seeds from 13 schedules, including the prior Dom attempt, with
no overlap. It is not an exhaustive historical filesystem search.

The preservation manifest records the entire previous Dom source and evidence
trees, excluding bytecode: 4,108 files, 50,939,883 bytes. Finalization must confirm
their exact inventory and hashes and bind a separate preservation receipt.

## Prelaunch and actual-model qualification

All 122 combined model-free tests passed in 51.49 seconds. The 23 source, test
and protocol files stayed unchanged across that run. The initial Windows
file-identity test failure is retained with its source and output; using the
full file-stat API corrected the preservation check without changing a player.

Independent frozen readback authenticated 1,387 inputs, 105 absolute runtime
files plus Python, 37 upstream asset Git blobs, the unchanged profile/panel,
budgets, tests and schedule. Launch review SHA256:
`9666f73b32a8fc01e5cbd9afc48c5316c33176719d3544fe3340a3f3875b5f65`.

Actual startup checks passed 52 control decisions and 64 Dom handler/reference
turns with 72 policy forwards on each side. All 256 ordinary qualification
requests and independent transitions passed. Maximum complete-decision time:
**0.0996025 seconds**, below the unchanged 0.700-second qualification margin.

All six operational games completed naturally, covering 332 live decisions,
with no player faults. Their outcomes are excluded from the main score and were
not used by the resource forecast.

## Resource admission

The new operational resource gate passed. Its unchanged conservative method
forecast 148,040,736 bytes (about 141.18 MiB) of final evidence against the
2 GiB reserve, and 484,712,089 bytes (about 462.26 MiB) total against 4 GiB.
Collection was forecast at 2,664.41 seconds and finalization at 890.79 seconds,
both within their original limits. The main 60-game screen therefore started.

## Main result

**Dom scored 0 wins, 0 draws and 60 losses against unchanged32**, with 0 points
from 30 games in each seat. All 60 games ended naturally. Operational results
are excluded from this score. The frozen admission threshold was at least 15
natural points out of 60; the completed screen therefore rejects this policy
as competitive opposition for the planned confirmation study.

This is a completed negative screen, not a timing, correctness, storage or
infrastructure failure. There were zero player faults, timeouts, replacements
or retries across the six operational and 60 main games. No move reached the
descriptive 0.700-second marker. The full one-second limit remained in force.

| Player | Live decisions | Mean complete decision | 95th percentile | Maximum |
|---|---:|---:|---:|---:|
| Dom PPO512 | 1,775 | 0.061977 s | 0.089006 s | 0.321656 s |
| Unchanged32 | 1,775 | 0.199716 s | 0.227725 s | 0.538472 s |

These are finite observations on this machine, not guarantees of zero future
deadline risk. The result concerns this exact exported policy and corrected
adapter, not every upstream checkpoint or PPO architecture.

## Final verification, preservation and closure

The terminal status is `completed_verified`, with `full_workload_verified`,
all 66 planned games present, no failure and no unplayed games. All 3,550 live
decisions replayed. All 24 selected complete decisions reproduced exactly,
12 per role. The fixed outcome/time-blind selector used every occupied category;
24 empty categories are explicitly disclosed. Its maximum of 48 did not require
inventing extra cases or changing the selection rule.

The verifier reconciled 158 workers, 3,832 parent requests and 474 compressed
stream proof bindings. This includes 3,550 live moves, 256 qualification
requests, 24 reproductions and two bundled startup requests. Source/runtime,
raw communication, placement, lossless compression, deadline and evidence
checks passed. The seal covers 25,189 artifacts, totalling 234,782,652 bytes.
Top-level summary receipts are bound separately by the independent review.

The complete previous Dom evidence and source inventory was preserved:
4,108 files and 50,939,883 bytes, with no additions, removals or changed hashes.
The new verification summary binds `prior_preservation_after.json`, SHA256
`c0d507316b6e517faa45421e296150d634a1f31ba094b632d88cac5f7cc07b0c`.
No old model, game or scientific audit was rerun for that preservation check.

The output occupied about **233.5 MiB of the 4 GiB cap** after finalization;
the larger storage allocation did not stop the run. Peak sampled combined worker
RSS was 339,107,840 bytes. This is a sampled bound, not a continuously observed
memory maximum.

| Stage | Monotonic seconds |
|---|---:|
| Qualification | 186.313 |
| Operational games | 177.828 |
| Main screen | 1,636.188 |
| Final verification | 333.453 |

The owning guardian finished in approximately **2,334.50 seconds (38 min 54 s)**,
from 01:42:37 to 02:21:31 Paris time on 27 October, well before the original
05:42:37 hard deadline. Runner44716, watchdog27792 and guardian31564 have actual
observed OS exit0 and clean cleanup receipts. The retained PowerShell process
handle recorded the guardian's exit; tool session73503 also completed exit0.

Independent receipt-only reconciliation accepted the completed negative screen.
It authenticated 204 finite files, including 22 new source/test pins, recounted
the 60 main results and both seats, reconciled reproduction categories, worker
and request counts, bound the preservation receipt and confirmed all three
original owner identities absent. It reused the passed exhaustive science,
raw-log, seal, runtime and prior-preservation checks; no model, game or full audit
was rerun, and no review-harness failure occurred. Operational results were
1 win and 5 losses for Dom and remained excluded from the screen score.

Receipt: `outputs/dom_screen_20261027_02/independent_results_review.json`.
Status: `accepted_completed_negative_screen`; SHA256:
`a7cbcddb9e7d33dd5683d61ac7a1855368bfeea3c76d4a89779ce6a452b22aff`.

## What this result establishes

The enlarged allocation let the unchanged experiment complete with all checks.
Dom is usable and fast under the tested corrected interface, but it is far below
the required competitiveness threshold against unchanged32. It should not be
used to test optimized64's generalization: an opponent that loses every game to
the control would provide little discrimination between the two local players.

## The next step

Stop the prepared two-opponent sequence and reassess opponent selection. Both
predeclared candidates now have completed 0/60 negative screens. Do not tune the
admission threshold, extend either sample or launch the 400-game confirmation
against a rejected opponent. Recommend a bounded review seeking stronger
independent opposition with credible playing-strength evidence before investing
in another adapter. No third opponent, extra games or new study is launched here.

## The broader picture

Optimized64's earlier advantage over unchanged32 and its accepted recovered
reliability verification remain intact. This sequence was intended to select
independent opposition for a later test of whether that advantage carries to a
different playing style. It has ruled out two insufficiently strong exported
policies, but has not established optimized64's performance against strong
independent opposition. Optimized64 did not participate here, and no human
ranking or superhuman-play claim follows. The project remains focused on a
strong, reliable one-second player; larger-data collection, fitting and training
remain deferred.
