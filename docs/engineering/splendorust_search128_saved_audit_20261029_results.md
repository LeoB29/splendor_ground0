# Verified result: 128 versus optimized64, with separate recovered verification

29 October 2026. **The saved-evidence recovery audit passed.** Its reader,
acceptance process and actual retained owner all exited 0. The audit completed in
779.9 seconds, before its shared final cutoff. No new models, searches, games,
scientific replay jobs or bootstrap computations ran.

This is separately recovered verification of the completed amended benchmark.
The original collection and recovery failures, failed independent reader, failed
owner and invalidation remain unchanged and unaccepted. The new audit does not
retroactively change those outcomes.

## Playing strength

The fixed assessment contains 400 main games on 200 paired starting setups with
swapped seats. Six operational games are excluded from the strength score.
Both players use the same trained network and three root worlds; the candidate
uses 128 simulations and the optimized64 reference uses 64.

| Measure | Verified result for 128 simulations |
| --- | ---: |
| Wins / draws / losses | 223 / 3 / 174 |
| Score, counting a draw as half a point | 56.125% |
| Paired bootstrap 95% interval | 51.5–60.75% |
| Score from seat 0 | 58.75% |
| Score from seat 1 | 53.5% |
| Administrative hard-guard forfeits | 0 |

All predetermined useful-gain criteria passed: score at least 55%, paired lower
bound above 50%, each-seat score at least 50%, and completed verification. The
nondegenerate interval safeguard also passed. This supports a useful advantage
in this exact matchup under the research execution policy. It does not establish
general opponent strength, a human ranking, or one-second deployment strength.
The independent audit reconciled the saved bootstrap verification receipt; it did
not recompute the bootstrap.

## Timing remains a separate practical problem

The research policy accepted completed moves beyond one second, using symmetric
9-second internal and 10-second complete-response guards. No game was forfeited.
The 700 ms and one-second thresholds stayed descriptive throughout.

| Main assessment moves only | 128 simulations | Optimized64 |
| --- | ---: | ---: |
| Completed moves with ordinary accepted clocks | 11,139 | 11,140 |
| Moves strictly above one second | 248 | 2 |
| Proportion above one second | 2.226% | 0.01795% |

Across all 406 main and operational games, median response times were 0.672 and
0.372 seconds, 95th percentiles were 0.831 and 0.481 seconds, and maxima were
2.012 and 1.039 seconds, respectively. In that broader timing denominator, 128
exceeded one second on 257/11,309 moves and optimized64 on 2/11,310 moves.
These are trusted complete-response arrival times, with no overhead subtracted.

The single adopted already-emitted move remains a separately authenticated event.
Its original rejected clock is unchanged and excluded from ordinary timing and
model-reproduction denominators. It remains part of the game result. Late research
moves cannot simply be relabelled as losses to infer a one-second match result:
stopping earlier would change the games that followed.

## What the recovery established

The original independent verifier failed because it passed absolute Windows
runtime paths to a project-relative path checker. The separate version permits
only the exact 95 runtime descriptors authorized by the authenticated closure
receipt. It retains hash, file-size, identity, parent-path, redirection, cache and
resource checks. The original full scientific verifier remains unchanged.

All 51 tests passed, including the actual saved failure through the unchanged
closure-verification function, all 449 nested descriptors, and malformed or
tampered-path rejection. The audit then preserved 159,151 files across all four
old evidence folders, totaling 5,103,576,352 bytes. Final identity verification
covered 181,773 authenticated files and 6,702 directories, including source and
runtime dependencies. Peak reader RSS was about 797 MiB, within the 2 GiB cap.

Recovered verification reconciled all 406 game ledgers, 200 assessment pairs,
22,620 live events (22,619 ordinary clocks plus one adopted response), 441
qualification decisions, 48 selected decision reproductions, 864 worker trees
and 9,504 graph cases. Existing passed scientific/raw/placement/replay receipts
were authenticated and reused; no new game transitions or inference were executed.

## Recommended next step

Keep optimized64 as the practical baseline. The strength gain now justifies a
small, bounded investigation of the saved slow 128-simulation moves and fixed
controls, using the existing timing records first. Identify whether the slow
tail is computation, scheduling or communication, then choose one targeted fix.
Require the same outputs and decisions for a pure speed change. Follow any
successful fix with a separately prepared one-second qualification and comparison.
Do not promote 128 automatically or launch another timing campaign without a
specific diagnosis. No follow-up experiment is launched by this result.

This follows the strength-first roadmap: demonstrate a useful whole-game benefit
before investing further in execution. The six-world branch remains parked.
External opponent testing stays paused; larger-data collection, fitting, self-play
training and network enlargement remain deferred.

## Evidence

- [Observed successful recovery acceptance](../../outputs/splendorust_search128_saved_audit_20261029_continuity/recovered_verification_accepted.json).
- [New acceptance receipt](../../outputs/splendorust_search128_saved_audit_20261029_01/acceptance.json).
- [Audit freeze](../../outputs/splendorust_search128_saved_audit_20261029_01/freeze.json), SHA `14fd968d476a3b69d554ef6ca99d357c29d86950bb82ebcd64ea183e66bb8305`.
- [Preserved source analysis](../../outputs/splendorust_search128_resume_20261029_01/analysis.json) and [timing](../../outputs/splendorust_search128_resume_20261029_01/performance.json).
- [Recovery audit protocol](splendorust_search128_saved_audit_20261029.md) and [strength-first roadmap](../strategy/splendorust_roadmap_update_20261028.md).

The observed acceptance receipt SHA is
`8571ce7f3ee6adc52a3aee4b7e18b9dba3587dae1878ab629613d30e9b098680`.
