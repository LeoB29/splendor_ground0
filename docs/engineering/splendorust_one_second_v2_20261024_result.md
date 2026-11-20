# One-second comparison v2: checker qualified; storage forecast stopped collection

24 October 2026. The separately versioned checker was implemented, tested,
frozen and run once as authorized. The actual correctness and timing gates passed.
All six excluded operational games completed and independently replayed. The
predeclared evidence-size projection failed, so the attempt stopped before any
main comparison game. This is an incomplete strength study, not a strength result.

## Identity and preservation

- Source: `tools/splendorust_one_second_v2_20261024`.
- Evidence: `outputs/splendorust_one_second_20261024_02`.
- Freeze SHA256: `5b86734afec36093da0916cab4811c7abf076f273fd7c52ce42afc2dc4988f60`.
- The new freeze binds 1,313 inputs. All were reverified unchanged after stopping.
- All 1,253 original frozen inputs, the original nine source files and 42 failed
  attempt evidence files remain unchanged. See `prior_attempt_preservation.json`.
- Same scientific players, checkpoint, native binaries, settings, original
  prospective game schedule, one-second limit, sample and success criteria.
- Larger-data work remains deferred. No training, promotion or automatic retry.

## What the repair established

All 204 prelaunch regression tests passed, including the exact native search
binary with deterministic dummy outputs and a real owned child-process cleanup.
Three independent implementation reviews closed with no remaining blockers.
Preparation made no checkpoint inference and checked 791 independent transitions.

The actual worker then passed its fixed counterexamples: 52 complete decisions,
101 native checks, 26 malformed-request rejections, corrected-rule/encoding/privacy
tests and fresh numerical parity. The turn-123 regression produced exactly
16 simulations, 17 native leaf evaluations, one neural callback and 16 heuristic
leaf evaluations. Turn124 used the direct heuristic with zero simulations, zero
leaf calls and zero neural callbacks. These actual records confirm the distinction
that the old checker incorrectly rejected. Individual requests/results were saved
before assertions in the new forensic log.

All 184 timed panel requests passed, covering 23 roots, lawful hidden-state
variants, both players and repetitions. The 92 historical PUCT16 comparisons
and 184 independent transitions passed. Maximum parent arrival time was
0.2187278 seconds, below the unchanged 0.700-second qualification margin.

All six operational games completed naturally and passed independent replay.
Their 322 completed decisions had a maximum arrival time of 0.2456911 seconds.
Operational outcomes were not aggregated or used for the resource gate.

## Why the attempt stopped

The six operational games took 105.596 seconds and added 10,243,846 evidence
bytes. Applying the frozen `200 / 6 * 1.35` projection and adding already written
evidence produced the following gate results:

| Gate | Projection | Frozen allowance | Result |
| --- | ---: | ---: | --- |
| Main collection time | 79.20 minutes | 240 minutes remaining | Pass |
| Final verification | 8.36 minutes | 30 minutes | Pass |
| Evidence before final verification | 460.584 MiB | 448 MiB | Fail |

The evidence projection exceeded its allowance by 12.584 MiB (2.81%). The
total evidence limit is 512 MiB, with 64 MiB reserved for final verification;
the reserve was not borrowed or relaxed. This was a prospective storage gate,
not an observed disk-full failure or a player/checker error.

The run started at 08:31:06 Paris and stopped at 08:34:55, after 228.56 seconds.
No main game started. The planned six-hour deadline was 14:31:06 and was not
extended. All seven worker cleanups passed; the outer watchdog recorded no cleanup
errors. A subsequent creation-time/PID inspection found none of the 44 recorded
owned process identities still running. See `poststop_inspection.json`.

## Next step

Retain both stopped attempts and this qualified checker. Prepare a separate,
lossless reduction of evidence duplication, then qualify its readers, integrity
checks and timed write overhead before freezing any further attempt. Keep every
required observation, action, trace, clock receipt, failure and replay check;
retain the existing storage cap, final reserve, sample and statistical criteria.
The six operational worlds are now played and must be accounted for when planning
any fresh operational panel. Do not simply restart the current frozen attempt.

A read-only sizing check compressed the six raw transport stdout logs in memory
with gzip level1: 1,201,470 bytes became 398,819 bytes, and every decompressed byte
matched its original. Applying that reduction to the same prospective arithmetic
gives 425.37 MiB, below 448 MiB by 22.63 MiB. No evidence file was changed. This
size-only estimate makes lossless raw-log compression a concrete first option;
it is not a passed new timing/storage gate. Preserve request limits, bounded
decompression, trusted timestamps, raw contents, hashes and all replay evidence.

A separate read-only review rechecked all 506 completed timed decisions, all
434 chained counterexample forensic events and the exact 52/101/26 counterexample
quotas. They passed. This review did not aggregate operational outcomes or run
another model evaluation.

The present evidence supports practical one-second operation on this panel and
six operational games. It does not yet establish whether search improves playing
strength over the same network alone.
