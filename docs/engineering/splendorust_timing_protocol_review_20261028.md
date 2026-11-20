# Saved-evidence timing review and conditional admission plan

28 October 2026. **Recommendation: retain the existing live execution path, but
park the next six-world admission until a specific improvement to live execution
has been qualified.** This review found no unused, already-qualified simplification
that would justify another unchanged admission attempt. The plan below is prepared,
not frozen or authorized to launch.

Only existing source and saved evidence were read. No models, games, positions,
scientific replays or tests were run. No player, deadline, verification requirement
or historical evidence was changed. The two recent logging-observer attempts stay
negative, and that measurement route stays paused.

## What the clocks actually measure

| Clock or check | Existing definition | Consequence |
| --- | --- | --- |
| One-second parent deadline | Starts before parent resource sampling, request encoding and dispatch. Ends at the trusted timestamp immediately after the reader receives the complete response line, before decoding or validation. | Includes request preparation, communication, waiting, player work and delivery of the entire scientific response. A late packet is rejected before validation. |
| 900 ms child guard | For ordinary requests, starts immediately after the complete input line is read, before envelope decoding. Checks search, result construction and serialization; the last check precedes stdout write/flush. | It is an internal guard, not a second complete-delivery clock. The parent deadline still governs response arrival. |
| 700 ms admission margin | Uses the original parent `receipt_elapsed_seconds`; every ordinary qualification and operational response must be strictly below 0.700 s. | A correct response between 700 ms and one second is valid play but negative admission. It is not a timeout or loss. |
| Startup qualification | The 52-decision suite has a bounded 300-second administrative request. Actual handler subcases retain their existing 900 ms guards. Graph parity has a separate 60-second administrative allowance. | A failed startup handler remains a qualification failure. The whole suite's elapsed time is not a one-second game move. |
| Subsequent verification | Parent decoding/validation, successful post-flush sidecars, full ledger and transport checks, selected reproductions, preservation and cleanup retain their existing placement and job accounting. | Passing the arrival clock never waives correctness or evidence checks. Later work still consumes the bounded job allowance. |

The parent receipt timestamp and the later `elapsed_seconds` are different
measurements. We must not select the faster one after seeing a result. Child
post-flush markers can also occur after the parent has received the complete line.
The existing boundary markers are explanatory evidence, not substitutes for the
trusted receipt. Any unfinished previous-request work that delays service of a
new timed request remains a real cost of that request.

Source anchors: `tools/local_player_qualification_128/clock.py:260,315,390`;
`tools/splendorust_boundary_timing_20261025/parent_profile.py:79`;
`tools/splendorust_pool6_admission_20261027_02/worker.py:175` and
`adapters.py:110`.

## The ordinary player and the logging diagnostic are different workloads

**Optimized64 ordinary play** uses the established boundary worker and trusted
response reader. It retains callback hashes/counts and scientific checks, but does
not enable the startup-only full root capture or install the startup forensic
recorder during ordinary moves.

**The six-world ordinary player** already retains additional root/callback
information and research fields in its scientific response. Those fields are part
of its present verified contract. It does not install the synchronous startup
forensic recorder during ordinary moves. Removing its full capture, shrinking its
response by dropping fields, or changing transport would require a separate
implementation and equivalence qualification; none is approved by this review.

**Startup qualification** installs a recorder that serializes, writes, flushes,
fsyncs and chains forensic events. The completed-result event is recorded inside
the handler before its final 900 ms check. The suite removes the callback afterward.
These durable records preserve scientific evidence even when validation fails.

**The recent logging diagnostics deliberately reinstall that recorder on every
measured ordinary fixture**, with full capture, to investigate the startup path.
In the latest diagnostic, timer-OFF still performs all the synchronous logging;
timer-ON adds the two timestamps. Its overhead gate tests the measurement addition,
not logging versus no logging. Both latest gates failed, so their component spans
cannot support a quantitative causal attribution or be subtracted from deadlines.

Static comparison confirms that the original six-world live wrapper and admission02
have identical parsed bodies for ordinary processing, handling, response encoding,
serving and cleanup. Initialization differs to bind the new output/freeze. The
existing normal path is therefore available, **but admission02 already used it**.
There is no newly discovered logging switch that would repair its live failures.

Source anchors: `tools/splendorust_search_scale_20261024/worker.py:33,88,120`;
`tools/splendorust_pool6_live_20261027/worker.py:88,132`;
`tools/splendorust_pool6_admission_20261027_02/worker.py:88,132`;
`tools/splendorust_minimal_logging_20261028/worker.py:187`.

## What the saved results establish

| Evidence | Verified observation | What it does not establish |
| --- | --- | --- |
| Accepted recovered optimized64/unchanged32 reliability | 200 natural games, 11,020 live decisions, zero player failures/timeouts. Optimized64 maximum 854.2834 ms. | The 700 ms margin was descriptive here; this is not a strict-margin pass or a guarantee about future failures. Original incomplete attempts retain their statuses. |
| Latest six-world admission | All 194 scientific checks passed; all 90 ordinary qualification requests were below 700 ms, maximum 569.187 ms. Six natural games then produced 338 correct moves, zero timeouts, but nine margin misses: two optimized64, seven six-world. | Qualification success did not transfer to the stricter live margin. The 952.8797/790.2872 ms maxima remain valid moves and negative admission evidence. |
| Qualified host diagnostic | Two later margin crossings, one per player, despite at least 11.038 GiB available in all recorded native samples. | More available RAM alone is not a demonstrated cure. The observations do not identify a particular scheduling, thermal, paging or outside-process cause. |
| CPU comparison with added host sampler disabled | Stopped during CPU2 startup qualification; zero of the planned 192 ordinary comparison requests ran. | No CPU ranking or qualified CPU2 arrangement. Its 1.655-second failed wrapper includes forensic/error work and is not pure search time. |
| Two logging-observer attempts | Both failed their prospective actual-model overhead gates; conditional diagnostics were withheld. | No observer admission, logging-cause proof, live-player admission or strength result. |

The baseline reliability and negative six-world admission are compatible: they
used different opponents/workloads and different roles for the 700 ms threshold.
We keep both conclusions. We neither demote the accepted baseline from these
diagnostics nor reinterpret its descriptive margin as permission to waive the
candidate's stricter admission rule.

The candidate's larger response is a real engineering cost: the last six games
recorded median emission of 48.264 ms versus 1.400 ms for optimized64. It is not a
complete explanation: the control also missed the margin, and even removing the
entire recorded emission span from the candidate's worst move would leave it above
700 ms. That subtraction is illustrative only; it is not a corrected timing result.

## The next admission plan — conditional, not launchable now

**Entry condition.** First require an identifiable, separately qualified change
that addresses a cost or operating condition of actual live execution, preserving
the scientific player, clocks and evidence contract. Record the hypothesis,
equivalence requirements and fixed failure rules before measuring it. A new
execution harness would need a small unchanged-player A/A qualification; no such
harness is justified by this review, so do not add that experiment automatically.
Removing the failed logging probe, waiting for a quieter run, increasing RAM alone,
or choosing CPU2 from incomplete evidence does not satisfy this condition.

If the condition is met in a later authorized study, prepare exactly one fresh
admission using the existing admission02 design:

1. **Freeze identities and inputs.** Keep optimized64/three worlds and the fixed
   six-world candidate, both at 64 simulations, the same historical weights,
   float32 graph inference, seed A, numerical settings, auxiliary choices and
   late-game behavior. Retain corrected rules/catalogue/turn contract, source guards
   at actual worker startup and all actual-entrypoint counterexamples. Bind the
   qualified execution arrangement explicitly. CPU0 players with CPUs2–19
   supervisors remain the established reference; CPU2 is not an approved substitution.
2. **Run the unchanged qualification workload:** 104 startup scientific checks,
   48 exact saved reproductions, six fixed candidate edges and 36 requests on
   18 fresh reserved positions: 194 scientific decisions, of which 90 are ordinary
   individually timed requests. Use new disjoint, model-free predetermined seeds,
   the same seat/phase strata and interleaving, and no replacement of awkward
   positions. Require all 90 correct responses strictly below 700 ms. A margin miss
   prevents every operational game; an actual failure stops immediately.
3. **Only after qualification passes, run six fresh operational games:** three
   swapped-seat pairs, fresh workers and graph checks as before. Every move must
   remain below 700 ms for positive admission. A successful margin miss is retained
   while the fixed six-game schedule completes; final admission is negative. An
   attributable player timeout/fault loses its game and stops collection. No move,
   seed or game retry/replacement is allowed. Infrastructure/scientific/resource
   failures remain incomplete evidence rather than invented game results.
4. **Finish every verification requirement.** Independent ledger replay for all
   six games; the unchanged deterministic selection of up to 48 completed decisions
   reproduced exactly under one second; full scientific responses and callback
   evidence; eleven exact graph cases for every worker; raw transport, lossless logs,
   trusted clocks, placement, source/runtime and profile checks; evidence sealing
   and readback; all three actual owner exit receipts and absence of owned processes.
   Never replay a failed decision to change its outcome. Admission requires all
   six natural games, the strict margin and independent acceptance.
5. **Retain the original bounded admission envelope.** Two hours overall, collection
   cutoff after one hour, 30-minute qualification and operational stage caps, and
   up to one hour for final verification, all clamped to the same clock. Evidence
   cap 4 GiB with 1 GiB final reserve; 12 GiB initial/5 GiB maintained disk; at most
   two concurrent workers and 62 total worker trees; 2 GiB combined worker/helper
   RSS. The original admission RAM contract is 2 GiB available, 60-second initial
   stability and at most 600 seconds of charged waiting, with its recovery checks.
   The recent diagnostics' stricter 4 GiB pre-request floor is a different contract;
   any stricter floor in a future admission must be explicit in its freeze and is
   not itself evidence of a timing remedy. Retain all per-route message/log bounds
   and outcome-blind finalization forecasts. Caps are not extensions or target runtimes.

This prospective template generates no new seeds or positions now and launches no
games. A positive result would justify considering a separately frozen 200-game
strength comparison; it would not launch one automatically. A negative result
would remain negative without threshold changes or another unchanged attempt.

**Current decision: park six worlds.** No new live-path remedy has been established,
so the entry condition is presently unmet. Preserve the candidate and its evidence
for later work. Do not repeat the accepted baseline 200-game reliability trial,
create a third logging observer or seek a passing admission by repeated sampling.

## The broader picture

The project seeks a stronger, dependable Splendor player within one second per move.
Increasing search from 32 to 64 simulations already produced an accepted strength
gain; the separated-supervisor reliability evidence supports the practicality of
optimized64. The six-world idea redistributes the same search budget over more
possible hidden-card arrangements. Saved-position results suggested steadier
choices, but that is not proof of stronger play.

Live admission exposed a timing-margin problem shared by both players. This review
ends an instrumentation detour without pretending to solve that problem. Keeping
the verified incumbent and parking the unadmitted branch is more informative than
another unchanged retry. Optimized64 remains frozen, external-opponent testing
stays paused, and larger-data collection and training remain deferred.

## Review scope and evidence

The finite saved-only review authenticated nine anchor files, 23 frozen source/
protocol files, three result files against the accepted admission receipt, and
inventoried seven explanatory reports: 42 distinct consumed files, re-read unchanged.
It reused previous independent acceptance; it did not re-audit every historical
file, run runtime validators or execute models. Five live/admission worker function
bodies were compared statically. Three independent read-only source/method reviews
agreed with the recommendation.

An initial reader omitted the original reliability freeze from its pin index and
stopped on a missing-key error. Its source/error are preserved separately. The
corrected reader authenticated that already-bound freeze and completed. Read-only
path/search mistakes are also disclosed in the receipt; no scientific evidence or
experimental status changed.

- [Finite review receipt](../../outputs/splendorust_timing_protocol_review_20261028_01/review_receipt.json)
- [Accepted recovered reliability](splendorust_reliability_finalization_20261026_results.md)
- [Latest admission protocol](splendorust_pool6_admission_20261027_02.md) and [negative result](splendorust_pool6_admission_20261027_02_results.md)
- [Host diagnostic](splendorust_pool6_hostdiag_20261027_03_results.md)
- [Incomplete CPU comparison](splendorust_pool6_core_compare_20261027_02_results.md)
- [Detailed logging result](splendorust_startup_logging_20261027_02_results.md) and [minimal logging result](splendorust_minimal_logging_20261028_results.md)

Primary acceptance identities remain unchanged: reliability
`54f0d7a4babb1970a2d47118988f66683f33f3dbc1bd8878c837f07ca4836c9a`;
admission02 `e68f6dc7b34c49147a933303ebfc9bdcb97671951e15e143a71a0b4f990f615e`;
minimal observer `67aea43edb26f933f932215d2bd66939767a82ad368d52ce57375011611042c3`.
