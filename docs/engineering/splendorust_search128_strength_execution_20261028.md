# Fixed 128 versus 64 simulations: authorized execution, 28 October 2026

The user authorized implementing and running the prepared comparison. This additive
protocol makes the prepared recommendation executable; it does not alter any older
freeze, player or result. Read together with
[the prepared design](splendorust_128_vs_64_strength_plan_20261028.md) and
[the updated roadmap](../strategy/splendorust_roadmap_update_20261028.md).

## Question and player identities

Does doubling search from 64 to 128 total simulations improve whole-game strength
enough to justify the extra computation? Both arms use three hidden-card worlds,
the same historical checkpoint, optimized graph inference, float32 precision,
one numerical thread, depth 64, PUCT, canonical observer seed A, auxiliary choices
and existing turn-124 heuristic. The original checkpoint remains explicitly a
historical pretrained model, not newly corrected-rules training. No fitting occurs.

Use the existing verified native bridge, SHA-256
`b15322f1c6aac1034638bdc52d91524639c63cbc2b61d361a9fef79dfdea728c`.
The reference16 and search binaries are the separately recorded original paths;
there is no native rebuild or pool-selectable six-world bridge. The checkpoint SHA
is `ef8a4521cd6c03c7075f15efee23f4cde6ec94d1b5398765cf0c24a09ad745cb`.

New research execution profiles are prospective: genuine 128 must be checked as
128, never relabelled as 64 to satisfy an older checker. The 64 route must reproduce
saved scientific outputs exactly. Both arms use the same full callback evidence
path. Original optimized64 files remain byte-identical and remain the baseline.
This is a comparison with increased computation, not equal computation redistributed.

## Correctness before games

Use the corrected two-player referee and the mandatory catalogue, noble and turn
contract checks in `future_training_rules_20261021.md`. Every new evaluator worker
calls `verify_runtime_sources()` at startup. Qualify its actual entry point using
the original 52-case suite on each route, including corrected-rules and interface
counterexamples. The candidate additionally performs 15 direct native/host parity
decisions at genuine quota128; each has the same internal nine-second guard.

The 84 ordinary decisions comprise 24 saved64 exact reproductions, both players on
18 predetermined fresh positions, both on six fixed edge cases, and both on six
fixed deterministic repeats. Thus there are 203 scientific decisions: 52 baseline
suite + 67 candidate suite + 84 ordinary decisions. Four administrative requests
contain two graph-equivalence bundles and two entry-point suites; there are 88
parent requests. Both startup graph bundles contain 11 cases. Record all failures.
Neither a failed position nor an interrupted decision can be replaced or retried.

Fresh positions are generated before any model calls using a fixed legal-index
policy and corrected state transitions. Their separate namespace is
`splendorust_search128_strength_reserved_20261028_v1`. Every scheduled seed is
checked for disjointness from registered prior schedules/manifests and the other
new stages; any collision stops preparation rather than selecting a new seed.

## Whole games and analysis

After correctness, play six operational games on three new setups with swapped
seats. All six must pass correctness, transport, ownership and replay checks.
They are excluded from the primary strength estimate. Use their observed cost to
test the conservative resource forecast before starting the main comparison.

The main sample is exactly 400 games on 200 independent setups, two swapped-seat
games per setup. The schedule namespace is
`splendorust_search128_strength_20261028_v1`. No optional extension, replacement,
quota sweep or result-dependent early success stop is allowed. Only the fixed128
candidate is considered. Retain every game and administrative outcome.

Primary score gives win=1, draw=0.5, loss=0, including attributable research-guard
forfeits. Resample complete pairs, not individual games: 50,000 PCG64 bootstrap
resamples, seed20261028128, linear quantiles and the established degenerate-sample
interval safeguard. A useful gain requires score at least55%, paired95% lower
bound strictly above50%, each seat score at least50%, and all verification passing.
Report a smaller supported gain separately from an inconclusive estimate. No
automatic promotion follows any outcome.

## Separate strength from timing

Accept every correct completed move arriving within the ten-second research guard,
even if it takes more than one second. Both players have the same nine-second
internal and ten-second external limits. Fixed simulation quotas do not expand
to fill remaining time. Measure trusted complete-line response arrival starting
before parent preparation/dispatch and ending before later decode/validation.
Retain raw clocks; do not substitute a wider postprocessing elapsed time.

Report 700ms, 900ms and one-second exceedances descriptively, plus distributions
by arm, stage and seat. None of these three thresholds gates admission, stops a
run, loses a game, selects a player or authorizes a retry. One second remains the
deployment objective, which this strength experiment does not qualify.

An attributable player failure at the actual nine/ten-second research guard stops
qualification or operational admission. During the main sample it causes an
administrative loss and the fixed schedule continues. Scientific, identity,
transport, infrastructure or resource corruption stops the study incomplete;
never turn those into an invented player loss. Preserve unfinished moves and
report their durations separately. Report full-sample bounds for administrative
games, not a counterfactual win rate obtained by truncating games at one second.

## One bounded attempt through acceptance

One output directory, one freeze and one original guardian clock cover every
stage: `outputs/splendorust_search128_strength_20261028_01`. No new clock starts
between qualification, operations, the main sample, finalization and acceptance.

- Total:24 hours; all collection must end by hour20.
- Qualification:at most one hour; operational games:at most one hour.
- Finalization:at most four hours, also clipped to the original24-hour deadline.
- Independent saved-evidence review and acceptance:at most3600 seconds together,
  contained inside the four-hour finalization and original24-hour caps.
- Evidence:16GiB total; reserve4GiB for finalization, so at most12GiB beforehand.
- Free disk:at least24GiB initially and5GiB thereafter.
- Available RAM:4GiB before decisions,60 seconds stable initially/after shortage;
  at most600 seconds cumulative waiting, charged to the same clock. The existing
  two-GiB worker-startup reserve is retained. Never pause an active decision clock.
- At most two resident worker trees, one active decision, twoGiB combined worker
  and helper RSS; at most862 worker trees and graph builds across the attempt.
- Worker CPU0; supervisors logical CPUs2–19 at normal priority, excluding CPU0's
  siblingCPU1. Verify actual topology and recorded process identities.

The worker ceiling is2 qualification +12 operational +800 main +48 selected model
reproductions. The resource forecast must include all406 independent ledger
replays,48 main reproductions, all862 graph/startup/closure checks, full capture
for both arms, hashing, statistics, source preservation and receipt publication.
Apply1.5 contingency to observed operations and reserve3600 seconds for the
independent review. A failed forecast stops admission; it does not relax a cap.

The owning launcher retains actual process handles through guardian, reader and
acceptance, closes only owned identities on failure, records actual exit codes,
and invalidates acceptance if any publication/exit deadline is missed. Eight
hashing threads with at most16 pending reads may accelerate the saved-only review;
each consumed file still receives its complete SHA-256 check. Reusing an unchanged
identity in that single review does not substitute a weaker hash or new model run.

All frozen inputs and newly sealed evidence must verify. The finite historical
preservation manifest pins inherited referenced dependencies, the original failed
audit and separately accepted recovery receipts; it does not launch another audit
of unrelated historical game trees. Historical failed acceptance remains absent.
Incomplete and failed attempts remain separately labelled and are never overwritten.

## Afterward

Interpret strength and observed cost together before deciding on speed work.
Retain optimized64 unless later evidence justifies a replacement. External-player
testing stays paused; new training and larger-data collection stay deferred. This
attempt tests a focused search question, not network training or general superiority
against all opponents.
