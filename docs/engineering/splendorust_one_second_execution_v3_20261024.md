# One-second Entity comparison: lossless log compression v3

24 October 2026. The user authorizes qualifying lossless compression in a separate
version, then freezing and running one new bounded attempt if it passes, using
fresh operational games. Preserve both previous attempts and all scientific
players, limits and checks. Larger-data work remains deferred.

The [original scientific protocol](splendorust_one_second_comparison_20261024.md),
[execution contract](splendorust_one_second_execution_20261024.md) and
[v2 checker contract](splendorust_one_second_execution_v2_20261024.md) continue to
apply, with only the storage implementation, its qualification and operational
world namespace changed as specified here. No checkpoint, native executable,
player arithmetic, PUCT quota, referee rule, sample size or outcome criterion
changes. No actual model inference occurs before the execution freeze.

## Separate version and preservation

Source: `tools/splendorust_one_second_v3_20261024`.
Evidence: `outputs/splendorust_one_second_20261024_03`.
Both previous output trees and source packages are inventoried before edits.
Their frozen inputs and every existing evidence file are hash-verified and bound
into the new freeze. Original freezes:

- Attempt01: `617b5c4852a8eddc18156b3df98b377f93c1aa57a6dad9291ad4ed478efc3f34`.
- Attempt02: `5b86734afec36093da0916cab4811c7abf076f273fd7c52ce42afc2dc4988f60`.

No compression, deletion or rewriting of either historical attempt is permitted.
Their existing six operational logs are read only as a fixed byte workload for
compression qualification, without aggregating game outcomes.

## Exact raw bytes, different storage

The parent stores each worker's stdout, stderr and event stream as one gzip member
at level1, with a synchronous flush after every write. Original line/chunk order,
bytes, limits and truncation behavior are preserved. The inherited trusted clock
still stamps receipt immediately after reading a complete line, before logging,
decoding or validation. Queue delivery stays after logging; compression can cause
a conservative delay, never create an earlier arrival timestamp or extend a move.

Only newly created, verified-empty raw placeholders are replaced by `.gz` files.
An independently accumulated raw SHA256 and byte count accompany the compressed
SHA256 and count. Worker cleanup finishes and verifies all three streams within
its existing five-second clock budget. Compression failures are infrastructure
failures; neither a game result nor a proposed timeout forfeit may bypass failed
cleanup. Interrupted or invalid archives remain preserved and cannot pass.

The reader expands in at most64KiB chunks, enforces the unchanged16MiB logical
cap per stream, and rejects missing trailers, invalid checksums, extra members,
trailing bytes, expansion beyond the cap, and length/hash mismatches. Its physical
input bound covers worst-case one-byte sync flushes; the separate study evidence
cap continues to count all actual on-disk bytes without relaxation. Actual gzip,
compression support and loaded zlib implementation identities are frozen and
rechecked. No raw observation, action, diagnostic, clock, error or replay is omitted.

## Compression qualification before the execution freeze

Freeze the implementation and fixed workload in a qualification plan before its
first write. This is a model-free instrumentation check, bounded to300seconds and
64MiB retained evidence, all counted in the later study's fixed evidence baseline.
Its source workload is exactly stdout, stderr and events for all six previous
operational workers:18files, at most8MiB each and16MiB combined. Use three ordered
passes through the unchanged workload and actual production write/flush calls.

Require exact decoded bytes and SHA256 agreement both through the production
reader and an independent standard-library gzip reader. Record every append/flush
duration; require a maximum of0.100seconds and an aggregate encoded/raw ratio
of at most0.75. Exercise the actual16MiB logical cap and rejection at cap+1;
retain the intentional failing archive as a labelled counterexample. Regression
tests cover corruption, truncation, expansion, short writes, identity, real child
transport and trusted timestamp ordering. Save every failure and partial artifact.
Do not repeatedly rerun a failed frozen qualification with a fresh clock.

These are additional instrumentation gates. The actual184model requests and
six fresh operational games must still satisfy the original0.700second margin;
the actual main-game deadline remains1.000second with the0.900second internal guard.

## Fresh operational worlds; unopened main worlds retained

The six operational games from attempt02 have been played and are excluded from
this attempt's gate and primary result. Derive three fresh worlds using SHA256
of canonical compact JSON:

`["splendorust_policy_puct16_1s_20261024_v3_operational", "operational", pair_index]`

where pair_index is0,1,2, using the full unsigned big-endian digest as the referee
seed. Seats swap with the same stage-global indices0..5 and allocation rule.
The100main worlds, never played in either previous attempt, keep their exact
original seeds, stage-global indices0..199 and seat allocation. The schedule is
materialized and hash-bound before any fresh game.

Collision scanning still covers discovered manifests/schedules/seed registries
up to8MiB. Only references to verified-unplayed main seeds in the exact prepared
schedule and the two original frozen schedules are allowed. Other collisions
stop preparation. Verify both prior completion stages, absence of any main game
directory and exact prior schedule identity; explicitly check new operational
seeds are disjoint from every original world. Disclose oversized omissions.

## Same resource gates, including verification cost

Unchanged quotas:52complete counterexamples,101native checks,26malformed request
rejections,184timed panel decisions, six operational games, then200main games only
if the fixed outcome-blind forecast passes. All actual games replay independently;
the predetermined sample retains up to48complete-decision repeats. Keep v2's
branch-aware leaf/neural accounting and all original scientific comparisons.

Verify compressed logs after qualification and each completed game, and again
for every worker before final evidence sealing and paired statistics. Operational
independent-check timing includes decompression, so its `200/6*1.35` projection
covers the added game-log verification. Fixed final overhead also includes the
measured qualification-log verification time once and for each of the48maximum
sample workers, in addition to original initialization/decision/validation costs.
The storage forecast uses actual new compressed-file sizes, all retained fixed
qualification evidence and the unchanged1.35multiplier; a sizing estimate alone
cannot admit main games.

Overall cap: six hours from before initial admission; stage caps one hour,
30minutes, four hours,30minutes. Absolute collection cutoff: hour5.5. Evidence:
512MiB total,64MiB reserved for final checks,448MiB maximum before finalization.
Same RAM admission, ten-minute cumulative wait cap,2GiB worker-tree RSS,7GiB
launch disk and5GiB reserve. Attributable main-game timeouts remain losses;
infrastructure or evidence faults stop incomplete. No automatic retry, reduced
sample, replacement game, changed success criterion or deadline extension.
