# Exact-evaluator diagnosis: verified timing, failed profiling attribution

23 October 2026. The separately bounded diagnostic was implemented and run.
**The first complete decision took 223.130 seconds and passed exact-output and
full independent replay checks. The profiler dropped calls, so the diagnostic
was stopped and its cost attribution must not be treated as reliable.**

The fixed three-model diagnostic is incomplete. No optimizer was implemented,
no original qualification was retried, and no fresh assessment, strength game
or larger-data collection was launched.

## Trustworthy result

Root 0 is game 25/ply 8, fit 161001: four unchanged ordered candidates, 128 unchanged
worlds and the same corrected rules, checkpoints and termination conventions.

| Request | Complete-response time | Result |
|---|---:|---|
| New profiled correctness fixtures | 2.834s | Exact match to the original accepted qualification |
| Ordinary fixed-work decision | 223.130s | Complete, 512 continuations |
| Profiled fixed-work repeat | 390.216s | Exact decision and compressed-trace equality |
| Independent full replay | 225.840s | All 512 continuations and 22,362 transitions passed |

The trace contains 11,181 Entity opponent choices and 10,669 raw035 choices, plus
512 forced first moves. The replay reexecuted the proposal model and both
continuation policies. All 296 complete records/12,888 moves recoverable from
the old interrupted trace match byte-for-byte; its incomplete tail was not used.
This is one full independent replay of the identical paired trace, not two
independent audits. The audit discloses shared corrected legality, model/feature
kernels and native transport dependencies.

The ordinary request took 1.859 times the 120-second target. That target and its
original failure remain unchanged. The longer diagnostic clock provides no
new 120-second qualification. One known early/seat 0 root is not a full-panel or
general latency estimate, and no move-quality or strength conclusion follows.

Ordinary request tree CPU was 184.344s and peak tree RSS 278.79 MiB. Sampled CPU
deltas by process role were Python 181.844s and native Rust 2.500s. Python's figure
includes its native neural kernels; it does not mean that all time is Python
interpreter overhead. The endpoints differ slightly from exact request timing.

## Why the run stopped

The authenticated raw pstats file reports **one GreedyOpponent.choose call**,
where the matching, independently replayed trace proves 11,181 calls. It also
omits the enclosing scientific_body, evaluator.choose and evaluator.trajectory
frames. This is not merely a display or JSON-summary error.

Recorded self time totals 351.102s against 389.744s measured around the profiled
scientific body, leaving 38.642s outside recorded self attribution. Neither the
cumulative call tree nor category percentages support a validated cost
breakdown. Complete-response latency was 74.9% higher with profiling in this pair.
This observed slowdown does not separate profiler overhead from host variation;
it is not an optimization speedup.

The frozen coarse category called rule_observer_and_component_conversion also
includes payment/token-return enumeration. Its full 32.74% must not be described
as observer conversion. All original profiler files and summaries are retained
unchanged, with these limitations overriding any optimistic interpretation.

The local cause is unproved. An upstream [pybind11 issue](https://github.com/pybind/pybind11/issues/5600)
reports a similar missing-calls symptom under Python 3.12, but it does not establish
the cause here. No interpreter, library, model or frozen source was changed.

Under the frozen mismatch policy, the authenticated owned runner was terminated
at 11:42:43 Paris. Its watchdog closed the owned process tree. Execution lasted
18m 13.531s, within the one-hour cap. This was an instrumentation failure, not
a 600-second decision timeout or RAM/disk failure.

Root 6 (game 57/ply 14, fit 161002) was in its ordinary request when stopped. Its
partial trace is preserved; there is no complete response receipt. This is an
administrative interruption, not a player loss or an observed completion time.
Root 12 (game 58/ply 4, fit 161003) was not started. The correctness fixtures exercised
all three proposal encoders, but full timing/replay coverage exists only for fit 161001.

## Recommendation

**First validate replacement timing instrumentation on a small, countable actual
model fixture; then prepare another separately bounded three-model diagnosis.**
Require known enclosing-function and opponent-call counts to agree with its
scientific trace before committing to full 128-world requests. Measure complete
response time independently and disclose any unassigned elapsed time.

The leading optimization hypothesis is reuse of the lawful observation and
legal-action preparation within one opponent turn. Static code reconstructs
these repeatedly, and selected profiler leaf counts agree with that repetition.
That makes it worth measuring; it is not a verified bottleneck share or a
demonstrated speedup. Any later implementation must be a separate version that
preserves ordering, checks, tensors, decisions and full replay equality, followed
by the unchanged 120-second qualification. No optimization or retry is authorized
by this result alone. Larger-data work remains deferred.

## Evidence and closure

- Frozen plan: `move_evaluation_profile_20261023.md`.
- Run: `outputs/move_evaluation_profile_20261023_01/`.
- Freeze SHA256: `6f3757aa2ac045a3263e683fe4ec675c616e434fb9351cc88cadf7f9e5d9fd68`.
- Administrative stop: `outputs/move_evaluation_profile_20261023_status_01/operator_stop.json`.
- Independent inspection: `outputs/move_evaluation_profile_20261023_status_01/independent_inspection.json`.

197 prelaunch tests passed, with one Windows symlink-privilege skip. After a
model-free profiler-runtime binding repair, 63 targeted tests passed with the
same skip. The actual 15-fixture qualification included 18 proposal encoder checks
across all three fits, native/auxiliary/privacy checks and the mixed trajectory.
These scientific checks did not validate profiler call-count fidelity; that
missing admission check is the reason for the next instrumentation step.

Independent closeout rehashed all 2,040 frozen inputs and four profiler runtime
bindings. All 2,024 original frozen inputs are unchanged members. All 22 recorded
PID/creation identities are closed. execution.json records runner_failed/exit 15;
the external operator_stop.json records the reason, because external termination
does not pass through the runner's Python exception handler. Preserve both.
