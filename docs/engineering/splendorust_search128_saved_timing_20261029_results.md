# Saved timing diagnosis: 128 simulations

29 October 2026. The bounded saved-record extraction completed successfully. It
used the accepted 400-game comparison and six operational games, with no new
model calls, games, searches, transition replays, bootstrap or training.
Both players and every historical result/failure remain unchanged.

## Findings

**The slow tail is concentrated in an early part of collection, and the ten worst
delays occur predominantly inside the decision handler. The evidence does not
yet isolate a particular computation or waiting mechanism.**

| Main-game 128 responses | Ordinary decisions | Above one second | Proportion |
| --- | ---: | ---: | ---: |
| Games 1-38 | 1,054 | 221 | 20.97% |
| Recovered game 39 | 26 | 19 | 73.08% |
| Games 40-400 | 10,059 | 8 | 0.0795% |
| All 400 games | 11,139 | 248 | 2.226% |

Thus 240 of the 248 overruns (96.77%) occur in the first 39 games. These blocks
contain different positions and were collected at different times. The recovered
game spans two source segments; its adopted response remains excluded from
ordinary-clock counts. The report also groups events by their actual source-freeze
identity, rather than assigning the whole recovered game to one segment.

The 64-player's mean response time was also lower in the final block: 458 ms in
games 1-38, 605 ms in game 39, and 357 ms in games 40-400. This is compatible with
an execution-condition contribution affecting both players, but it does not prove
that the same positions would run faster, identify a host process, or establish
that the continuation code caused the difference.

## Where the ten longest responses spent their time

All ten made 128 search simulations and the normal 129 neural callbacks. The
systematic 128 controls also made 129 callbacks. No extra simulation quota or
extra neural-call count explains the extreme times. Equal counts do not prove
identical native-search work or identical machine conditions.

The following five nonoverlapping segments were reconstructed from saved markers
and checked against the original trusted complete-response arrival. The tiny
input-read-to-handler and handler-to-write segments are combined as "other" here.
Numbers are rounded to milliseconds; the machine-readable report retains exact
nanosecond partitions. Turn numbers count the 128-player's own decisions.

| Game / turn | Total | Request delivery | Decision handler | Response delivery | Other |
| --- | ---: | ---: | ---: | ---: | ---: |
| 34 / 9 | 2,012 ms | 63 ms | 1,887 ms | 62 ms | <1 ms |
| 39 / 1 | 1,752 ms | 64 ms | 1,622 ms | 66 ms | <1 ms |
| 131 / 5 | 1,622 ms | 64 ms | 1,492 ms | 65 ms | <1 ms |
| 352 / 10 | 1,600 ms | 52 ms | 1,500 ms | 48 ms | <1 ms |
| 39 / 15 | 1,585 ms | 64 ms | 1,438 ms | 82 ms | <1 ms |
| 34 / 8 | 1,563 ms | 64 ms | 1,444 ms | 54 ms | <1 ms |
| 18 / 19 | 1,557 ms | 61 ms | 1,419 ms | 76 ms | <1 ms |
| 35 / 3 | 1,533 ms | 55 ms | 1,426 ms | 52 ms | <1 ms |
| 34 / 12 | 1,448 ms | 57 ms | 1,318 ms | 72 ms | 1 ms |
| 34 / 10 | 1,433 ms | 57 ms | 1,323 ms | 52 ms | <1 ms |

The handler accounts for 90.7-93.8% of each response. Its median is 1.441 seconds
in the slow ten, compared with 0.566 seconds in the ten systematic 128 controls.
All ten slow handlers themselves exceed one second. Reducing only the outer
response-delivery interval would therefore not remove these overruns. This is an
arithmetic observation, not a claimed speedup from an implemented change.

The handler includes native search, neural inference, inner Rust/Python exchanges,
legal-action work and bookkeeping. It is not a pure neural-inference measurement.
Sampled worker-tree CPU for these ten moves is 0.578-0.828 seconds, but its final
sample occurs after parent validation. It is not aligned with the handler; the
difference must not be labelled a measured scheduling or waiting duration.

The fixed selection has 50 membership slots and 44 distinct records: ten slow
128 responses, six distinct nearest same-game controls, eighteen distinct adjacent
64 responses, and ten systematic 128 controls. Six slots reuse an existing
control; none is missing. One same-game control used 126 callbacks, while the
other five used 129. Same-game controls were explicitly selected at or below one
second, and adjacent decisions are different game positions. Their group medians
are descriptive rather than representative estimates or matched causal effects.

## What this result establishes

The accepted strength improvement remains intact: 128 scored 56.125% against64
under the research completion policy. This analysis localizes the extreme delays
to the handler and identifies pronounced temporal clustering. It does not certify
one-second deployment, explain all 248 overruns, or identify a proven optimization.
The late continuation responses remain failures to meet the descriptive target,
and the much slower early segment must not be discarded.

The existing graph implementation and separated supervisors are already in both
players. Ordinary play does not install the synchronous startup forensic logger.
Repeating a previous outer-reader change, choosing a different CPU without evidence,
or removing a logger that is absent from this path would not follow from this result.

## The next step

Recommend one small, separately frozen repeat diagnostic before changing the
player: the ten saved slow positions plus the ten fixed systematic controls,
three interleaved repetitions with each unchanged player (120 ordinary decisions),
using the already qualified boundary observer. Keep the same weights, precision,
three worlds, seeds, search quotas and scientific checks. Verify reproducibility
against the available saved outputs and across repeats. Use a bounded overall
allowance, preserve all failures, and keep 700 ms and one second descriptive under
the existing 9-second/10-second research guards. No new games are needed.

The question is whether the same roots repeatedly remain expensive or whether
the original extreme elapsed time mostly disappears. Neither outcome alone
identifies CPU scheduling. Persistent handler cost would justify measuring a
specific component; nonpersistent spikes would direct attention to execution
variability. Do not keep repeating admission until a favorable sample appears.

The inner callback bridge is a concrete later hypothesis: a normal128 decision
performs129 sequential Rust-to-Python inference exchanges, versus65 for64. Binary
framing could reduce JSON conversion while retaining serial search and process
ownership; its benefit has not been measured. Batching independent leaves would
change search order and is not an equivalent pure speed change. Do not implement
either route from these boundary timings alone.

The repeat diagnostic is a recommendation, not a launched follow-up. No new player
version, model process or game was started by this saved-evidence task.

## The broader picture

The project now has a demonstrated reason to spend limited engineering effort on
128 simulations: it improved whole-game strength with the existing network.
This follows the strength-first roadmap, unlike the earlier six-world timing
campaign whose whole-game benefit did not materialize. We are narrowing the
practical timing problem while retaining optimized64 as the baseline. Further
speed changes must preserve scientific behavior and earn their own one-second
qualification; broader opponent strength still remains a separate question.
External testing stays paused and larger-data/training work stays deferred.

## Verification and preservation

- Nineteen model-free tests passed, including real saved clocks, sidecars and
  compact replay inventories, deterministic selection, tamper rejection,
  path/identity failures, missing markers, resource probing and deadline expiry.
- An initial status-schema assertion and an access-time-sensitive preparation
  check failed before freezing. Their receipts and exact prior sources are kept
  under `prelaunch/`. Windows read-induced access-time changes are now allowed;
  device/inode, modification time, size and full SHA checks remain. A test rejects
  same-size content mutation even after modification-time restoration.
- Freeze SHA256: `3fa528958ca39f85fdb19b56f5dfce902e3abff91b8173ed29a9a62ab769ac9a`.
- Root observed successful preparation exit0 and actual diagnostic exit0. The
  diagnostic completed in 5.657 seconds of guarded work (8.157 seconds observed
  command duration), with about156 MiB peak observed RSS, within the15-minute,
  1 GiB RSS and64 MiB output bounds.
- Every consumed input was authenticated against finite pins descending from
  accepted recovery and rehashed before completion. No full historical audit
  was rerun. New analysis does not replace the original benchmark acceptance.
- Independent targeted readback passed: all 44 detailed partitions and all
  22,619 ordinary timing entries reconcile. The reader rehashed only 1,086
  consumed artifacts and exited0. Receipt SHA256:
  `41e4f751cf1f30280483be3d322d08ba8230b1d7eff84b9ad3fb3f00547c5605`.
  A separate interpretation review found no material numerical or scientific issue.

Sources: [frozen protocol](splendorust_search128_saved_timing_20261029.md),
[diagnostic report](../../outputs/splendorust_search128_saved_timing_20261029_01/report.json),
[completion receipt](../../outputs/splendorust_search128_saved_timing_20261029_01/completion.json),
[observed process exits](../../outputs/splendorust_search128_saved_timing_20261029_01/root_exit_observation.json),
[independent readback](../../outputs/splendorust_search128_saved_timing_20261029_01/independent_readback.json),
and [accepted benchmark result](splendorust_search128_saved_audit_20261029_results.md).
