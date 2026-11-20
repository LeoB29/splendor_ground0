# Bounded validation of faster simulation input preparation

Authorized by the owner on 2026-10-16: validate the isolated speed prototype,
keeping the current experiment running unchanged. This is engineering validation
using completed historical records. It creates no new strength estimate, changes
no learner, and does not enable a production backend.

## Fixed workload and scope

Reuse the completed and audited `large_corpus_optimization_128` panel and the
original 035 continuation model. The frozen version 2 speed-probe plan determines
the six source roots: median and maximum canonical support within each of the
three phases, ordered by support then source index. Their indices are
515, 517, 465, 251, 277 and 382. Selection uses metadata, never outcomes.

Each measured pass processes all 128 saved selection worlds and every recorded
union candidate at each root. This is 7,040 continuations and 243,536 saved turns
per pass. The benchmark covers complete saved selection batches, including
root input, determinization, continuation, checking and output handling. It does
not time fresh source generation, proposal formation, independent assessment,
statistics or a complete planner decision. Those exclusions must accompany any
reported speed ratio.

Before timing, run an independent exact-input pass on world indices
0, 17, 35, 53, 71, 89, 107 and 127 for all candidates of all six roots. Compare
the complete float32 observation, action features and mask at every subsequent
policy decision. Separately exercise the assessment-record integration on saved
assessment worlds 0 and 127 and all officially assessed candidates. Neither
check contributes to the speed denominator. These subsets contain 440 selection
and 22 assessment continuations, respectively.

Every checked and timed continuation must exactly match its saved termination
reason, winner, turn count, full move-trace hash and final-state hash. Unknown
record semantics or malformed records fail; lawful cutoff results with an
unknown winner remain unknown. Mismatches must never be silently excluded.
Historical snapshots, determinized world identities, model and source code are
authenticated before use. Only saved historical outcomes may be read.

## Comparison and interpretation

Compare the unchanged reference encoder with the already-frozen version 2
prototype, at one and two CPU workers, each with one numerical thread. Run two
timing repetitions per backend and worker count. Fix their balanced order in
the plan before any measurement. An additional input-equivalence encoder call
must not occur inside a measured pass.

Report elapsed batch time including process startup, model loading, dispatch,
root work and output writing, with precisely identified timing boundaries.
Each root runs in a fresh process; this startup cost is included for both
backends. One/two workers denotes the concurrency limit across the six tasks.
Report aggregate worker/root time separately where available. Use the ratio of
total matched elapsed times across the two repetitions, and also show each
repetition. Do not treat continuations as independent timing replicates or claim
a statistical confidence interval from them. Report phase/support differences,
actual one/two-worker overlap and memory. A two-worker result is not transferable
to arbitrary worker counts or hardware.

Focused tests additionally cover observer-private information, public and own
reserve sensitivity, supplied action ordering, empty/terminal states, forced
pass and cutoff termination, action bounds, and exception restoration. Synthetic
edge fixtures are separate correctness evidence, never speed observations.
Retain the original implementation as the independent reference.

## Isolation, resources and durable execution

All new code and outputs stay under this validation's own directories. Frozen
probe versions 1 and 2, alternative learner artifacts, shared source, and current
`structured_proposer_128` files and process settings remain unchanged. The parent
controller must not load the neural model. Child workers are hidden and run at
below-normal priority, with a maximum of two concurrent validation workers.

Freeze exact resource limits, code, protocol, workload and order in `plan.json`
before launch. Enforce a four-hour cumulative active limit across resumed
attempts. Require at least 512 MiB physical memory remaining, 6 GiB commitment
headroom and 5 GiB free disk; cap each worker at 512 MiB and the model-free parent
at 256 MiB, with the corresponding aggregate process limit. Before loading,
require at least 768 MiB available for one worker and 1.25 GiB for two workers.
Admission may wait up to 180 seconds and requires five consecutive healthy
seconds. Record final as well as periodic process memory observations.
Only operational fields from the current study may be read. Allow concurrent
validation during its training/data-validation/replay phase or after completion;
stop the extra validation workers if it enters source collection or evaluation.
Never signal or restart the current experiment.

Resource admission waits are bounded. Preserve any hold, failure, partial
receipt or attempt; do not loosen frozen limits to obtain a run. A resumed run
may reuse authenticated complete checks and complete timing blocks. An interrupted
timing block must restart in a new attempt, with its partial timing excluded from
performance aggregation and retained on disk. No benchmark wall time can span
downtime or mix different concurrency conditions.

Each result must record the plan binding, exact workload counts, correctness
status, timing order/boundaries, memory observations and limitations. Completion
requires all prescribed blocks and checks; a partial run is not a successful
validation. This command is a bounded hidden job independent of an active chat,
not an ongoing monitoring automation. No automatic successor or deployment.

## Decision

Any exactness failure blocks adoption and must be diagnosed without editing
frozen evidence. If correctness passes, assess both repeated net-runtime results
and memory to decide whether the prototype is suitable for a separately
authorized future collector. A result on these six roots does not prove universal
equivalence or guarantee a particular gain in another workload. Even a successful
result does not authorize altering the currently running experiment.
