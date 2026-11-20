# Corrected host observer: separately frozen diagnostic attempt 02

27 October 2026. The user authorized a separate observer/checker correction, tests using the
saved failure, and a new overhead qualification followed by the full diagnostic
only if qualification passes. The nine slow roots, nine controls, exact request
order, player settings, overhead thresholds and resource limits remain unchanged. This separate attempt runs no fresh
games, performs no training and cannot admit a player or launch a benchmark.

Source: `tools/splendorust_pool6_hostdiag_20261027_02`.
Evidence: `outputs/splendorust_pool6_hostdiag_20261027_02`.
Preserve the entire incomplete first host diagnostic, its independent failure
review, the earlier negative admission02 and every frozen input/failure. Freeze source, selected roots, ordered requests, resource limits,
observer runtime and all gates before the first model runs. No retry, replacement,
sample extension or post-result adjustment.

The old execution freeze is
`424fab8fcfc079a01410ec093d76a4d7f2478144fdd32e95a6503a37d29d8c47`;
its independent failure review is
`c4d832791aaeb061bf1b2938bf0d078cd637408fc7a16487c184bf1fcf2eff61`.
Its original `incomplete` status and raw 1,388-to-1,387 aggregate decrease stay
unchanged. Before freeze, compare this attempt's entire prepared panel and
request matrix directly against the preserved attempt and require equality.

## Saved-failure regression and checker correction

Use the old sequence-28 record and observer window as model-free regression
fixtures. Exercise signed decreases, increases and unchanged aggregates; none
can identify an interval event total. Assert that independently valid native
counters remain available. Corrupted process identities, timestamps, bindings,
counter shapes or native monotonicity must still fail. Never edit the fixture,
invent its missing end-placement observation or call its old failed status a
new valid measurement. Any new-schema projection used in a unit test is a
labelled copy, not experimental evidence.

Separate strict live measurement acceptance from final forensic verification.
The collector stops immediately on any real observer error. The sampler attempts
end-placement capture in a finally path, preserving an error if that also fails.
The final checker may verify exactly one terminal invalid observer window as
preserved evidence: authenticate every available raw record, original clock,
request/process binding, counter and error, and explicitly disclose missing
fields. A clean sampler-close receipt proves cleanup, not a missing earlier
placement observation.

For such a failure, finish source/preservation/ownership and forensic-seal checks
where possible, but retain an explicit incomplete-measurement outcome, nonzero
exit, false overhead qualification and no eligibility for further requests.
No failure verifier may silently convert an invalid measurement into accepted
overhead evidence. The old failed attempt itself is never rerun or relabelled.

## Players, rules and exactness

Retain unchanged optimized64 with three worlds and the original six-world
candidate, both with 64 total simulations and observer-derived seed A. Preserve
weights, float32 inference, native binaries, search, profile IDs and bytes,
ordinary response payloads, the existing minimal observer and trusted reader.
The new worker facade changes only this study's output and freeze ownership.
Use the existing isolated supervisor placement: players on CPU0, supervisors
on CPUs2–19 excluding sibling CPU1, normal priority.

Use `splendor_base_qualified_rules_20261018_v1`, catalogue SHA256
`db4776918e612eee5a6f2617719c85ec5f413d592bbab0a64058babc83a70312`, and
`atomic_prereveal_completed_turn_v1`. Each actual worker calls the corrected
runtime source guard before loading the model and passes its 52 actual-entrypoint
counterexamples and eleven exact graph cases. Preserve the prereveal auxiliary
choice and internal forced-pass/deadlock conventions. No historical checkpoint
is relabelled as trained under corrected rules.

Keep two workers resident throughout, dispatching one request at a time. Every
ordinary request has the original 1.000-second complete-response deadline and
0.900-second internal guard. The 104 startup decisions and 22 graph cases retain
their bounded administrative bundle clocks. Never restart an active clock.

## Frozen panel and controls

Use all nine successful margin misses from admission02, indexed from zero:
g0/p4, g0/p6, g0/p20, g1/p41, g2/p36, g2/p39, g2/p47, g3/p26 and g4/p19.

For each in that order, choose the nearest earlier completed 65-callback move
from the same game and original player, excluding every target and previously
selected control. Do not inspect control latency for selection. This yields
g0/p2, g0/p0, g0/p18, g1/p39, g2/p34, g2/p37, g2/p45, g3/p24 and g4/p17.
Require 18 unique observer hashes and independently valid corrected snapshots.
No alternative or replacement control is allowed if an identity check fails.

Run both players on every selected root. The original moving player's complete
scientific output has an exact saved reference. The other player has no invented
historical reference: independently validate its action, science and successor,
then require exact complete-result equality across every repetition, including
observer-off versus observer-on. Preserve raw evidence before comparisons.
Controls are structurally selected, not identical states or guaranteed identical
auxiliary phase counts. Do not describe cross-state timing differences as causal.

## External observer and overhead qualification

Use a persistent, normally parked sampler thread in the supervisor process.
Do not alter either player's inference or existing reader. During enabled
windows, native read-only queries sample every 100 ms: available physical RAM,
system commit headroom, identity-bound process working set/private bytes,
combined soft/hard page-fault count, process CPU time and host CPU time. Process
I/O counters are retained as auxiliary observations, not identified as paging.
Windows live-thread context-switch aggregates are sampled before and after the
request. Retain each raw aggregate and its signed net change, including decreases.
The true interval switch total is unavailable for positive, zero and negative
changes alike; label this explicitly. Do not clamp changes to zero, present them
as interval totals/rates, or invent a voluntary/involuntary split. This correction
does not relax identity binding, API/read failure checks or monotonic native
CPU, page-fault and I/O counter checks; those failures still stop collection.

Bind actual API DLLs and the psutil implementation/runtime before launch.
Hold samples in a bounded 32 MiB memory buffer. Write observer records only
outside active decisions, and verify the sampler thread stops and joins.
Timestamp the observation window: endpoint measurements may include time
outside the move/handler and do not measure direct runnable-queue delay. CPU,
memory, page-fault and I/O summaries remain independently usable when the
context-switch interval total is unavailable by design. Page faults
are not identified as hard faults. No memory pressure or competing load is
deliberately induced.

First run exactly 72 ordinary requests: nine controls × two players × four
rounds. For each cell the observer order is OFF, ON, ON, OFF. Rotate position
order by round, reverse odd rounds and alternate the leading player according
to `(round + position_index) % 2`. The saved matrix is authoritative. Retain all
requests, with no discarded warmup, trimmed outlier, retry or replacement.

For each player independently, require all 36 requests to be correct and within
one second, and all of the following with 18 observations in each mode:

- Observer-on mean receipt time is at most 1.05 times observer-off mean.
- Observer-on minus observer-off mean is at most 10 ms.
- Observer-on p95 is at most observer-off p95 plus 25 ms, using nearest rank.

Require every enabled window to have valid counter evidence. Report all cell
paired differences and maxima. This engineering gate bounds detectable
perturbation in this sample; it does not prove zero measurement overhead.
If any overhead condition fails, skip the full diagnostic and finish verification
of the preserved negative overhead result. Do not tune thresholds or retry.

## Conditional full diagnostic

Only after the full overhead gate passes, run exactly 144 further requests:
18 roots × two players × four interleaved rounds, with the observer enabled.
Use the same cyclic/reversed order and alternating lead-player rule. Keep one
exact-result cache across both stages. A successful receipt at or above 700 ms
is retained as descriptive recurrence; it does not stop this diagnostic or
reverse either prior negative admission. An actual timeout, unexpected rejection,
scientific mismatch, integrity fault or exhausted resource cap stops immediately.

Describe timing and sampled counters by root, player and stage. Separate elapsed
handler time, worker CPU time, response emission and observed host conditions.
Report missing temporal coverage explicitly. Repeated observations are dependent,
targets were chosen for prior slowness, and a non-recurrence under better RAM
headroom is not proof that RAM caused the earlier result. Do not estimate a live
timeout rate or playing strength from these requests.

## Bounds, verification and completion

Use one non-resettable two-hour cap, collection cutoff at one hour, stage caps
of 900 seconds each for startup qualification, overhead and diagnostic, and up
to 3,600 seconds for final verification. Cap evidence at 2 GiB with 512 MiB
reserved for final proofs; require 12 GiB initial and 5 GiB maintained disk.
At most two model workers/graph builds and no replacements; combined worker/
helper RSS stays below 2 GiB. The observer has its separate 32 MiB buffer cap.

For this diagnostic require 4 GiB available RAM before every request, with
60 seconds stability initially and after an observed shortage. This supersedes
the inherited 2 GiB pre-request reserve for this study. Available RAM is not a
continuously enforced stop inside an active request; do not add a poll or reset
to the unchanged timed path. Charge all waits to the
same overall clock and a cumulative 600-second wait cap. Waiting happens between
requests. The higher pre-request headroom is a disclosed diagnostic context,
not a modification to an active or earlier experiment. Observer samples may
reveal transient in-request drops and must report them truthfully.

Maximum workload: 104 startup scientific decisions plus 216 ordinary requests,
320 scientific decisions total, 22 graph cases and 220 parent requests. An
overhead-negative attempt has only the 72-request ordinary prefix. Final checks
must distinguish intentional gated non-execution from interrupted work.

Validate every saved root transition independently, all available historical
targets exactly, all within-cell outputs exactly, raw request/response and
clock bindings, graph bits, placement handshakes, observer windows/counters,
lossless compressed logs, source/runtime identities and evidence seals.
Recompute the overhead gate independently from saved receipts. Preserve every
failure and prefix. Successful study acceptance requires actual zero runner,
watchdog and guardian exits; failed-prefix verification requires preserved actual
nonzero exits and clean owned-process cleanup. Require
absence of their owned process identities, and sampler stop/join evidence.
Perform independent saved-evidence acceptance without new models or games.

Conclude with the evidence-supported timing interpretation and one recommended
next action. Optimized64 remains frozen, external-opponent testing remains
paused, and larger-data collection and training remain deferred.
