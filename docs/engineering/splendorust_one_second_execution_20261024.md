# One-second Entity comparison: execution contract, 24 October 2026

The user authorized implementation and execution of the prepared
[scientific protocol](splendorust_one_second_comparison_20261024.md). This document
specifies the new implementation; it changes no player, outcome criterion or
resource bound. Old evidence and failed attempts remain intact. Larger-data work
and the 10,240-simulation study remain deferred.

## Implementation and identity

New source lives in `tools/splendorust_one_second_20261024`; evidence lives in
`outputs/splendorust_one_second_20261024_01`. All consumed repository Python,
native build inputs, checkpoint/profile/model files, actual fixtures, runtime
bindings, tests and the materialized schedules are bound before execution.
Preparation runs no model inference. The fixed additional entrypoint-counterexample
manifest is saved in `qualification_plan.json` before freezing.

One Python worker loads one immutable Entity network, with the frozen PUCT16
constructor's zero-input architecture warm-up. The greedy and search transports
share those weights and use the exact frozen host arithmetic. Two authenticated
Rust child processes persist for that game, each constructing a fresh native
agent for every phase/request. No mutable search tree, observation cache, game
history or random stream crosses requests. Each game has a fresh worker tree.
One performance-core logical processor and one numerical thread are assigned.

The ordinary greedy response retains canonical logits, native support indices
and the original network output, sufficient to independently reconstruct the
first maximum. It drops unnecessary entity tokens from ordinary evidence.
Search keeps its original work diagnostics. Complete response construction and
delivery count within the trusted one-second parent clock; the internal guard
is 0.900 seconds. The 0.700-second margin applies only to qualification and the
six operational games. There is no fallback or quota change.

## Qualification

Before the 184 scheduled timed requests, the actual worker exercises fixed
corrected-rule, encoding, legality, privacy, auxiliary and error counterexamples
for both arms. It compares the greedy wrapper with the original explicit greedy
operation, and subsequently compares every search fixture with its saved
PUCT16 record. The original numerical parity receipt authenticates the model
and source; additional fixed parity probes verify the current runtime against
the same native network representation. Any changed historical profile input
or failed parity stops qualification. These checks are part of the initial
qualification attempt, not a repair or retry after failure.

Parent collection uses strict corrected snapshot decoding and an independent
dictionary ledger. Model workers call `verify_runtime_sources()` on startup;
the collecting/verifying parent calls it once at its startup. No historical
snapshot conversion or training loader is used. Nothing here qualifies a
general future training pipeline.

## Clocks, admission and ownership

The owning watchdog records one monotonic and UTC origin before launch
admission. The total cap is six hours. Stage caps from their own starts are
one hour, 30 minutes, four hours and 30 minutes; all collection also stops at
absolute hour 5.5. Both clocks enforce deadlines, including RAM waits and source
checks. Each initialization is clamped to 60 seconds and the remaining stage.

Admission requires 7 GiB free disk and 2 GiB available RAM stable for 60 seconds.
Later admissions require 2 GiB available RAM, with at most ten minutes cumulative
waiting, only between requests. Active moves are never suspended. Combined
worker and descendant RSS is capped at 2 GiB; samples cover the Python worker,
both native children and their authenticated Windows console helpers.
Evidence is capped at 512 MiB, with 64 MiB reserved for final checks. Five GiB
free disk must remain. Sampling is reported as sampled telemetry, not a proof
of continuous memory peaks. Windows jobs own descendants and close them on exit.

## Fixed outcome-blind cost gate

Six full operational games must finish, independently replay, and satisfy the
0.700-second margin. The gate never aggregates their winners. Observed complete
collection time is multiplied by `200 / 6 * 1.35`; variable evidence is projected
the same way and added to already written evidence. The projections must fit
the remaining four-hour/absolute collection window and 448 MiB allowance.

Final verification is also forecast prospectively: measured operational ledger
and replay time times `200 / 6 * 1.35`, plus fixed overhead. During qualification,
measure a full source/runtime rehash and the exact independent statistical
procedure on 200 synthetic draws. Add 48 times the measured qualification-worker
initialization plus one second per decision and the largest measured parent
validation time. Multiply this fixed sum by 1.35 and add 30 seconds for writes
and cleanup. The total final forecast must fit 30 minutes. Missing or failing
measurements stop the study; they never reduce its sample or audit.

## Evidence and acceptance

Keep every complete pre/post snapshot, lawful observation, result, work trace,
trusted receipt and failure. Independently replay all six operational and 200
main games; bind the exact frozen schedule, initial worlds and clock receipts.
Select the at-most-48 complete-decision audit by the predetermined lawful
identity hash rule, materialize its manifest, then reexecute those decisions.
Seal completed game evidence before computing and independently checking paired
statistics. Source/runtime hashes must remain unchanged. Acceptance additionally
requires the outer watchdog's cleanup receipt without errors.

Attributable player errors during main games are losses. Source, referee,
resource, initialization and supervisory faults stop incomplete. Deadlocks or
the 2,000-ply guard cannot create a game result. There is no automatic recovery,
replacement, deadline extension, repeated attempt or player promotion.
