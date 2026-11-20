# Dom PPO512 adapter: qualified, screen not admitted by the resource gate

27 October 2026. Output: `outputs/dom_screen_20261027_01`.

The separate adapter passed actual-model correctness and timing qualification,
and all six fresh operational games against unchanged32 completed naturally.
The 60-game competitiveness screen did **not** start: its conservative final
verification forecast exceeded the frozen allowance for final audit files.
This is a verified operational prefix, not a completed negative strength screen.
No conclusion about Dom's competitiveness follows from this attempt.

## Frozen scope and preserved implementation

Protocol: `docs/engineering/dom_screen_20261027.md`; separate source package
`tools/dom_screen_20261027`. Freeze SHA256:
`529a8f724b5d8c5aa74287dea10522082f1a34ee1820b03d921896cc98ffbc61`.

The external policy is the pinned domdomegg/splendor browser PPO512 export at
commit `426fa50e06fe99b40620131ca962f418824bb73a`, weights SHA256
`e80e468a7e99ead87b43277c8c17219be669e55fac54604abbd55d68b1586191`.
It has 267 inputs, two 512-unit ReLU layers and 77 action outputs. It uses no
search, value head, ensemble, fitting, alternative weights or local fallback.
The historical origin of the weights remains explicitly identified.

Source browser arithmetic is preserved: float32 stored weights and hidden
activations, float64 accumulation and final logits. An independent literal
browser-loop reference checks the numerical outputs. The encoder reference is
an independent Python port of the pinned Rust source; the upstream Rust engine
was not executed. Numerical tolerances were fixed before learned execution,
with exact masks, chosen indices, completed actions and reproduction required.

The adapter uses corrected rules and the strict lawful observation interface.
Unknown opposing reserves receive one deterministic lawful completion, never
actual private identities. Token returns and noble selection use pre-reveal
shadow states. All policy phases belong to one completed action and its single
one-second clock. The maximum is five policy calls per complete decision.
Unchanged32's weights, three worlds, search, internal 0.900-second limit,
placement and external 1.000-second deadline remained unchanged.

The freeze binds 1,378 inputs, 105 absolute runtime files plus Python, both asset
manifests and 37 authenticated upstream assets. Independent prelaunch review SHA:
`b68c6617f6cb40a0f9d95f2f8d95df559faec912ff52c44fb23b6e798272b884`.
All 104 combined model-free tests passed. Earlier fixture failures and their
source/log receipts are preserved. No frozen input was repaired during execution.

## Qualification and operational evidence

The actual unchanged32 startup suite passed all 52 decisions. Dom passed 64
complete handler decisions and 64 independent reference decisions, with 72
actual policy forwards on each side. The tests also checked all 77 action-index
meanings, 13 malformed/legacy/privacy rejections and corrected-rule edge cases.

All 256 ordinary qualification requests passed, with 256 independently checked
transitions and no timing-margin failure. The slowest completed in 0.112392 s,
below the separate 0.700-second qualification margin. The panel contains 64
structural positions with hidden variants and repetitions, not 256 independent
natural positions; structural fixtures need not all be naturally reachable.

All six operational games (three swapped-seat pairs) ended naturally. There were
334 live decisions, 167 per player, with no timeout, player fault or replacement.
No completed move reached the descriptive 0.700-second marker. Operational
outcomes are excluded from the strength screen and were not used by the gate.

| Player | Mean complete decision | 95th percentile | Maximum |
|---|---:|---:|---:|
| Dom PPO512 | 0.076197 s | 0.107277 s | 0.120865 s |
| Unchanged32 | 0.251819 s | 0.302040 s | 0.400550 s |

These are observed timings on this finite run, not guarantees for all positions.

## Why the screen did not start

The frozen gate requires every resource forecast to fit before main games begin.
Its 1.5x conservative final-evidence estimate is:

`1.5 * (67,904,832 reproduction bytes + 35,318 artifacts * 333 bytes + 16 MiB fixed)`

This equals 144,664,413 bytes, or **137.963 MiB**, exceeding the reserved 128 MiB
by 9.963 MiB. That was the only failing condition.

| Forecast | Estimate | Frozen allowance |
|---|---:|---:|
| Main collection time | 3,313.65 s | 6,300 s, also within collection cutoff |
| Final verification time | 1,054.74 s | 3,600 s |
| Evidence before finalization | 323.38 MiB | 896 MiB |
| Final verification evidence | **137.96 MiB** | **128 MiB** |
| Total evidence | 461.34 MiB | 1,024 MiB |

The allowance is an allocation inside the attempt's evidence cap. This was not
a disk-full event, a RAM shortage, a player timeout or a source-parity failure.
The forecast conservatively charges an entire measured game worker's evidence
for each one-decision reproduction, as fixed before outcomes. It was not revised
after observing the run. There were **zero main games**, so the assessment's
zero accumulated main points must not be read as a 0/60 playing result.

## Verification and process closure

The recorded terminal status is `operational_not_admitted_verified` with
`complete_planned_workload: false` and `failure: null`. All six games and their
334 moves replayed. All 16 selected complete decisions reproduced exactly,
eight per role; all available selector cells were used and 32 empty cells are
explicitly disclosed. The up-to-48 selector was not replaced by extra sampling.

The final verifier reconciled 30 workers, 608 parent requests and 90 compressed
stream proof bindings, including startup and qualification. Source/runtime,
placement, raw evidence and lossless-log checks passed. The evidence seal covers
4,021 artifacts totalling 46,813,423 bytes; top-level summary receipts are
authenticated separately by the independent reconciliation.

Qualification took 207.109 s, operations 221.359 s and finalization 92.156 s.
The owning guardian completed in approximately 521.64 s (8 min 42 s), within
the four-hour cap. It ran from 01:09:31 to 01:18:12 Paris time on 27 October.
Peak sampled combined worker RSS was 338,436,096 bytes. Runner40420,
watchdog46508 and guardian18744 have actual observed OS exit0 and clean cleanup;
the PowerShell owner captured the guardian's exit through its retained handle.
No process should be restarted from this frozen attempt.

Independent receipt reconciliation accepted this verified operational prefix,
not the unplayed screen. It bound 129 finite files, reconciled the counts above,
confirmed all three owner identities absent and independently identified the
sole failed resource condition. Receipt:
`outputs/dom_screen_20261027_01/independent_results_review.json`, SHA256
`97ae7a9eed80d96e1ff14c088935980f9658ba362b3f96bcfd560ae2e1085b8b`.
It reused the passed exhaustive science/raw/seal/runtime checks rather than
rerunning them. A review-harness assertion initially confused a canonical JSON
digest with the newline-terminated file's hash; that correction is disclosed
in the review, and did not change or rerun the experiment.

## What this result establishes

The corrected-rules adapter is functional on the qualified panel and the six
operational games, and it has substantial measured headroom under one second.
The preserved verification checks also close this completed prefix. They do not
establish the unplayed 60-game screen or admit Dom as competitive opposition.
Optimized64 did not participate. The original prepared two-opponent sequence
has therefore not yet produced an admitted independent opponent.

## The next step

Recommend a separate bounded version reserving **256 MiB for finalization inside
the same 1 GiB total evidence cap**, leaving 768 MiB for earlier evidence. The
current forecasts fit both allocations comfortably. Preserve this attempt and
its failed gate. Keep the forecast method, 1.5x allowance, players, one-second
deadline, corrected rules, 15/60 admission threshold, four-hour cap and all
verification checks unchanged. Qualify the allocation change, freeze new inputs
and fresh operational/screen schedules, then rerun the conditional sequence.
Do not carry these operational games into a scored screen or silently relabel
this attempt as completed. No new attempt is launched by this recommendation.

## The broader picture

Optimized64 already scored 64.25% against unchanged32 in the earlier paired
comparison, and its later reliability evidence has separately accepted recovered
verification. We are now seeking competitive independent opposition to test
whether the improvement carries to a different playing style. Breck's completed
0/60 screen rejected that first opponent; Dom remains the predeclared reserve.
This engineering result makes Dom usable, but its strength remains unresolved.
A qualified competitive screen would justify proposing the separate optimized64
versus unchanged32 common-opponent comparison. Larger-data collection, fitting
and self-play training remain deferred.
