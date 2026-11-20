# One process-sampling optimization, then conditional search qualification

24 October 2026. This is the next prospective freeze in the user's authorized
sequence: diagnose timing, make one measured behavior-preserving optimization,
validate it at 32 simulations, then qualify 64 and compare 64 against 32 if it
passes. The two completed diagnoses and all earlier attempts remain immutable.
Larger-data collection and training remain deferred.

## Evidence and the single change

The completed actual-model diagnostic reproduced all 220 decisions exactly.
The separate model-free sampler probe found mean exclusive costs of about
50.8 milliseconds for parent-PID queries and 45.6 milliseconds for thread-count
queries per supervision sample in its one-worker fixture. These are attribution
measurements, not guaranteed savings from the move clock. Samples can overlap
the trusted arrival interval or occur after it.

The candidate obtains these two fields for all processes from one fresh Windows
Toolhelp32 process snapshot at each original supervision poll. It retains the
original descendant discovery, executable resolution and hashes, process roles,
CPU, RSS, affinity, clock boundaries and polling frequency. Fresh creation-time
checks before and after bind the selected fields to the same owned process.
Missing live processes, changed identities and snapshot errors fail closed via
the original trusted failure-receipt path. There is no cross-poll metadata cache,
fallback or suppressed error. The snapshot is bounded at 32,768 entries; unrelated
system process records are not persisted. Microsoft's PROCESSENTRY32W documents
the parent PID and execution-thread-count fields used here.

This is the only optimization. Raw stdout reading, path resolution, neural
arithmetic, weights, native code, search logic, hidden-world pool, random seeds
and scientific settings remain unchanged. In particular, the old raw stdout
reader remains in use. The child entry point is the frozen
tools.splendorust_search_scale_20261024.worker, without timing instrumentation.
The candidate changes its parent supervisor in a separate source namespace.

## Qualification and assessment

Before any actual-model requests, model-free tests must cover snapshot failure,
malformed or duplicate entries, overflow, cleanup, changed process identity,
missing owned processes and inherited resource/ownership checks. An authenticated
idle helper then provides 20 comparisons of snapshot fields against public psutil
getters for stable owned processes. All evaluator workers call
verify_runtime_sources() at startup. The actual baseline32, candidate32 and
candidate64 entry points must each pass the preserved corrected-rule, encoded
cost, privacy, rejection and native-parity counterexamples before their panel.

The 32-simulation schedule contains 316 requests: baseline and candidate, two
interleaved passes with reversed order, on 54 development roots and 24 reserved
saved holdout roots, plus four requests on the exact prior failed64 hidden-state
variant. Both workers stay resident on the same assigned core; only one request
is active. The aggregate resource supervisor applies symmetrically. All outputs
must match exactly across implementations, repeats and available historical
records: complete action, full search trace, callback hashes and independent
corrected-state transition. The hidden variant must preserve lawful observations
and decisions. The unchanged constructor warmup is the only warmup.

Each 32 panel independently requires candidate mean trusted arrival at most 90%
of baseline mean, candidate p95 no greater than baseline p95, and every candidate
arrival at most 0.700 seconds. Failure on holdout cannot be rescued by pooling
the panels or choosing another implementation in this attempt. Baseline arrivals
above 0.700 seconds are retained diagnostically; any actual 1.000-second deadline
failure, incorrect result or infrastructure failure stops the attempt. Holdout
selection excludes development and earlier replay pairs and is a saved-position
assessment, not an unbiased tail-latency sample or fresh-game evidence.

After the 32 gates pass, six fresh operational games compare baseline32 and
candidate32 in three swapped-seat pairs. Candidate moves must all meet the
0.700-second admission margin; both arms retain the 1.000-second hard deadline.
Outcomes are not used for strength claims or admission. Every game must replay.

Only then qualify candidate64: the existing 184-request panel including reference16,
privacy variants and repeats, followed by 48 requests on the 24 reserved holdout
roots. Require the original exact historical64 outputs, all original correctness
checks, exact holdout repeats and the 0.700-second admission margin. Then play six
new operational games, candidate64 against candidate32, in three swapped-seat
pairs, with both arms using the same optimized supervisor. Every move must meet
the admission margin and every game must replay. No quota fallback or timing
retry is allowed. All historical failures remain part of the record.

Final verification checks all compressed logs, source inventories, trusted clocks,
saved-position transitions and operational game ledgers. Reproduce the first
decision of each player in each operational game: at most 24 complete decisions,
with exact actions/traces and the original one-second hard deadline. The
0.700-second margin is diagnostic for these final repeats; it does not replace
the prospective operational admission gate. Both worker and outer cleanup must
succeed before acceptance.

## Non-resettable limits

The attempt has a 90-minute combined monotonic and UTC cap, including admission,
startup, checks, waits and cleanup. No collection beyond minute 75. Stage maxima:
sampler 10 minutes, paired32 qualification 20, operational32 10, qualification64
15, operational64 10, final verification 15. These are maxima within the common
clock, not separately renewable allowances. A windowless watchdog owns the tree.
No automatic retry, clock reset, extension, game replacement or favorable rerun.

The original trusted whole-decision arrival limit is 1.000 seconds and internal
fail-closed allowance 0.900 seconds; no timing component is subtracted. Require
2 GiB available RAM stable for 60 seconds at initial admission, check between
requests, and allow at most 10 minutes cumulative admission waiting. Never pause
an active move clock. Aggregate sampled RSS across resident workers and helpers
is capped at 2 GiB, including initialization of the second worker. Require 7 GiB
free disk initially and maintain 5 GiB. Evidence ceiling 512 MiB with 64 MiB
reserved for final checks; preserve original logical stream limits, lossless
compression, bounded decompression and all failures.

## Conditional next comparison

Success supports this exact implementation and hardware measurement, not a
universal timing guarantee or playing-strength improvement. If 64 passes every
gate, prepare a separate prospective freeze for 200 fresh games (100 swapped-seat
pairs), optimized64 versus optimized32. Keep the network, worlds, other search
settings and verification checks fixed. Use fresh operational games and an
outcome-blind resource forecast before collection, one-second deadlines with
timeouts as losses, paired uncertainty and all replay/source/log checks. This
conditional bounded comparison is already authorized; this qualification runner
does not launch it implicitly. If qualification fails, preserve the attempt and
recommend the next step from its evidence rather than selecting a favorable retry.
