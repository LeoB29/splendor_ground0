# Buffered response reading qualification

25 October 2026. Qualify one change to the parent communication layer, beginning
with the three saved slow moves and controls. Run six fresh operational games
only if the saved-position timing and correctness gates pass. This attempt
contains no strength benchmark or training. Preserve every earlier attempt,
including the operational admission failure. Larger-data work remains deferred.

## Change and fixed players

The baseline is the verified process-sampling implementation from
`tools.splendorust_timing_opt_20261024.common.CandidateStudyWorker`. The candidate
inherits that implementation and replaces only its stdout reader. It reads raw
pipe data in bounded chunks of at most 64 KiB, retaining at most one bounded
incomplete message suffix. Every complete line receives the timestamp of the
read that delivered its newline, before logging or decoding. Multiple complete
lines in one chunk retain that original timestamp. A later request must not
receive a newly assigned arrival time for previously buffered data.

The original framing, exact raw bytes, maximum message length, malformed and
unterminated input handling, unsolicited-response detection, queue bounds,
lossless logs, sampling, resource checks and cleanup remain in force. The
whole-decision clock still starts before initial sampling and request encoding.
No communication, sampling or logging time is deducted from the clock.

Both versions use the unchanged scientific worker
`tools.splendorust_search_scale_20261024.worker`, historical checkpoint, corrected
two-player rules and catalogue, atomic pre-reveal turn contract, three hidden
worlds, depth 64, float32 CPU inference, one numerical thread, selected
performance-core logical processor and lawful-observation seed. Fixed search
quotas are 64 and 32. No model, native binary, arithmetic, search policy, cache,
auxiliary-action rule or random seed mechanism changes.

## Checks before execution

Model-free tests cover real pipes carrying the archived slow-move response
packets and adversarial chunk boundaries, multiple lines per read, exact and
exceeded size limits, partial EOF, empty EOF, read errors, queue overflow,
unsolicited messages and late-arrival rejection. Verify that chunk timestamps
are assigned before logging and retained across all lines in that chunk.
Complete tests and an independent code review before freezing the attempt.

The freeze binds sources, tests, this contract, position and game schedules,
source evidence, checkpoint, native binaries, Python, model libraries,
compression runtime and process-sampling API modules. Every evaluator worker
and the parent call `verify_runtime_sources()` at startup. The actual baseline
and candidate workers at both quotas must pass the established corrected-card,
encoded-cost, noble, payment, privacy, reconstruction rejection and native-parity
entry-point counterexamples before their corresponding timing panel.

## Saved position timing and correctness

The fixed development panel has 24 roots: the three failed admission positions
at plies 42, 48 and 50 first, 11 nearby or early controls from the same game,
and 10 controls from the earlier successful operational games. The reserved
assessment panel has 12 roots selected deterministically without looking at
their timing, from distinct pairs in the completed 32-versus-16 comparison.
Exclude development positions and positions already used in prior timing or
reproduction panels. Record any practical exclusion limitations in the plan.
These selected panels diagnose and qualify this implementation; they do not
estimate an unbiased probability of a deadline failure.

Run 64 simulations first, then 32. For each quota keep the baseline and candidate
workers resident together, with one active request at a time. Two paired passes
reverse order to reduce ordering bias, for 288 requests in total. Constructor
warmup and the mandatory actual-entrypoint counterexamples precede the timed
panel; no additional timing warmup or discarded requests are added. Preserve
every observation and compare complete
actions, full search and callback traces, repeated outputs, available historical
outputs at matching quotas, and independently reconstructed corrected-state
transitions. Both versions retain the one-second hard deadline.

Each quota and each panel must independently satisfy all three timing conditions:
candidate mean trusted arrival at most 95% of the corresponding baseline mean,
candidate 95th percentile no greater than baseline, and every candidate arrival
at most 0.700 seconds. A baseline observation above 0.700 seconds remains a
diagnostic observation; an actual one-second deadline failure stops the attempt.
Preserve timing summaries before stopping on a failed panel gate. Do not pool
panels to rescue a failure or tune another implementation using the reserved
panel in this attempt. No retries or replacement positions are permitted.

## Six conditional operational games

Only after all saved-position gates pass, play three fresh swapped-seat pairs
with buffered-reader 64 versus buffered-reader 32. Generate seeds from full
SHA256 over canonical JSON using the new domain
`splendorust_buffered_response_ops_20261025_v1`, stage and pair index. Check
registered prior manifest, schedule and seed-registry files up to 8 MiB for
collisions and disclose omissions. The six games are operational qualification,
with no strength estimate or outcome-based admission.

Every game must finish naturally, independently replay, and have every move
arrive within 0.700 seconds. Actual timeouts and invalid responses retain the
established failure classification and evidence; they cannot qualify a game.
Infrastructure, rules and resource failures stop incomplete. No replacement
games, clock resets, quota fallback or margin relaxation are permitted.

Final verification rechecks all frozen sources and runtimes, every worker log
and cleanup receipt, all saved-position checks and all six independent game
ledgers. Reproduce the first decision of each player in each game, at most 12
complete decisions, with exact actions and traces and the one-second hard
deadline. The 0.700-second margin on these final reproductions is diagnostic.
Successful outer watchdog cleanup is required for acceptance.

## Fixed resource limits

One 60-minute allowance uses both monotonic and UTC clocks and includes startup,
admission, collection, verification and cleanup. No collection beyond minute
45. Stage maxima are 30 minutes for qualification, 15 for operational games,
and 15 for final verification, all within that common allowance. A windowless
watchdog owns the runner and its descendants. No automatic relaunch, clock
reset, favorable retry or allowance extension is permitted.

Keep the 1.000-second trusted arrival deadline and 0.900-second internal limit.
Require at least 2 GiB available RAM stable for 60 seconds at initial admission,
check between requests, and cap cumulative admission waiting at 10 minutes.
Never pause a live move. Aggregate sampled worker and helper RSS is capped at
2 GiB, including the second worker's initialization. Require 7 GiB free disk at
launch and maintain 5 GiB. Evidence is capped at 512 MiB, including a 64 MiB
final verification reserve; retain the existing logical stream limits and
bounded decompression.

Success qualifies this exact implementation under the measured conditions.
It does not identify the cause of the earlier shared slowdown or establish a
universal timing guarantee. A subsequent strength comparison needs its own
prospective freeze and authorization; this runner cannot launch one.
