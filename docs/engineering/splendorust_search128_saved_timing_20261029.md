# Bounded saved-record timing diagnostic for 128 simulations

29 October 2026. The user authorized the next recommended step after the accepted
128-versus-64 comparison: compare saved slow moves with ordinary moves, identify
where the extra time accumulated, and recommend one targeted next action.

This is a retrospective analysis of existing evidence. The ten longest responses
and aggregate timing results have already been inspected. Freezing this extraction
method does not make those observations a prospective experiment.

## Scope and unchanged reference

Use the separately accepted recovered verification of the 400 main games and six
operational games. Optimized64 remains the practical baseline. Preserve the 128
candidate, both scientific players, network, weights, precision, three root worlds,
search configuration, clocks, original failures and acceptance dispositions.
The completed research games used symmetric 9-second internal and 10-second
complete-response guards. One second is the deployment objective; 700 ms and one
second remain descriptive in this analysis. No timing observation changes a game
result, creates a loss, or admits a player.

Run no model, search, new game, transition replay, bootstrap or training. Read the
existing qualified parent/child boundary observer rather than introducing another
instrumentation layer. New output is additive in
`outputs/splendorust_search128_saved_timing_20261029_01`; no source evidence is edited.

## Fixed extraction panel

Rank accepted ordinary 128-player moves using `clock.receipt_elapsed_seconds` from
the 400 main-game replay receipts. Operational games are checked separately and
excluded from the main denominator. Keep the separately adopted response outside
ordinary-clock distributions, as in the accepted benchmark.

Select the following before examining their detailed timing components:

1. The ten slowest main-game 128 responses; ties use game index then ply.
2. For each, the same-game 128 response at or below one second nearest in ply,
   excluding the ten slow cases; ties choose the earlier ply. Record missing
   matches or reused controls explicitly, and deduplicate actual reads.
3. Both adjacent 64-player decisions where present. These show the neighboring
   execution context, not an identical position or equal-compute comparison.
4. One systematic 128 response from each one-based game 20, 60, 100, 140, 180,
   220, 260, 300, 340 and 380: the player's tenth decision, or its last decision
   if fewer exist. Selection ignores response time and game outcome.

This is at most 50 distinct detailed move records. Matched controls are selected
by response time and the slow cases are extremes; neither group represents the
ordinary population. Report all group membership and any overlap. Reuse the
accepted whole-population timing summaries for context rather than treating this
small panel as an estimate of population reliability or strength.

Using the same compact receipts, also describe main-game timing by the three
source-freeze identities, player and game. Game 39 spans the original and recovery
segments; do not assign the whole game to one source-freeze identity. Report
late-move clustering without attributing it to a machine condition: segments
contain different positions and were collected at different times. This adds no
detailed move records to the fixed panel.

## Measurements and interpretation

Bind every detailed row to its original clock and parent/child sidecars. Verify
the same request, role, quota, observer and result identity. Reconcile the ordered
wall-time partition exactly to the trusted response time:

- request start to child input read;
- input read to handler start;
- handler start to handler end;
- handler end to response write start;
- response write start to trusted complete-line arrival.

Report search work/callback counts, packet size, sampled request CPU and RSS where
available. Keep missing measurements missing. Parent validation completion is a
different clock. Signed cross-process flush gaps can legitimately be negative and
must not be clamped or added as a second cost.

The handler includes search, inference, internal communication and bookkeeping.
Boundary elapsed time is not pure inference time. Sampled tree CPU is neither
per-component CPU nor exact runnable/waiting time; the wall-minus-CPU difference
does not identify scheduling, paging, temperature or a particular external app.
Its final sample occurs after parent result validation, so its interval is not
aligned with either the handler or the trusted arrival interval.
RSS does not prove system-wide memory pressure was absent. Correlation and local
clustering guide the next experiment; they do not prove a cause.

Recommend an execution change only when the evidence identifies an addressable
cost. Otherwise name the smallest additional measurement needed. A pure speed
change would require a separate version, exact scientific outputs/decisions,
qualification and later assessment under the actual one-second deployment clock.
Do not launch that implementation or another timing campaign automatically.

## Authentication, bounds and completion

Anchor the accepted recovery observation (SHA256
`8571ce7f3ee6adc52a3aee4b7e18b9dba3587dae1878ab629613d30e9b098680`)
and its authenticated acceptance and preservation inventory. Authenticate consumed
game manifests, compact replay proofs, summary reports and selected row/clock/
sidecar files against those existing pins. Recheck consumed input hashes at the
end. Do not repeat the full historical evidence audit or modify old acceptance.

Bind this protocol, implementation and tests in a new extraction freeze before
the diagnostic run. Check negative cases for path escape, hash mismatch, missing
or inconsistent clock markers, player/quota mismatch and partition disagreement.
Call `verify_runtime_sources()` once at analyzer startup. No environment or dataset
is reconstructed, so this does not claim qualification of a fresh evaluator or
training entry point.

The run has a 15-minute total wall-time cap, 1 GiB process RSS cap and 64 MiB new
output cap; require 2 GiB available RAM and 1 GiB free disk at launch. These are
analysis resource bounds, not new player admission requirements. Stream selected
records and retain only compact summaries. Preserve any failure without retrying
the frozen attempt in place. Completion requires a successful actual process exit,
input preservation checks and an independent targeted readback of the findings.

External testing remains paused. The six-world branch and larger-data collection,
fitting, self-play training and network enlargement remain deferred.
