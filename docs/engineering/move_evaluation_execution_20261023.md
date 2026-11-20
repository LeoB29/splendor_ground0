# Execution of the bounded move-evaluation study

23 October 2026. The owner authorized implementation and execution of the saved
[study design](move_evaluation_study_20261023.md), starting with correctness
qualification. This execution specification preserves that design's candidates,
models, worlds, policies, decision rule, clocks and stopping rules.

New implementation: `tools/move_evaluation_20261023/`. New run evidence:
`outputs/move_evaluation_20261023_01/`. New native build and retained build
attempts: `outputs/move_evaluation_20261023_build_01/`. Existing benchmark,
postmortem, checkpoints and native binaries remain unchanged.

## Admission and sequence

Model-free tests and review precede the immutable execution freeze. That freeze
binds the prepared design, candidate lock, recovered saved root states, fixtures,
Python sources, native sources/binary, checkpoints and relevant numerical runtime
files. Static reconstruction of old saved positions does not simulate new play.

The eight-hour clock begins at launch, before memory admission and the first
model initialization. A separate windowless watchdog owns the runner and its
descendants and enforces the same absolute deadline. There is no automatic
restart. Every new Python collector/evaluator/auditor worker authenticates the
corrected runtime once at startup. Native operations are authenticated children
of that guarded worker and receive only lawful observer inputs.

The initial actual-entry-point counterexample request has a 300-second ceiling,
inside the three-hour qualification window, plus the prescribed 60-second model
initialization ceiling. It exercises all corrected faces and edge cases through
the actual models, encoders, sampler and mixed-policy trajectory, and verifies
independent transitions. The unchanged PUCT16 assessment path is then checked
on the same 15 fixtures, with each raw035 and opponent response subject to the
actual external one-second clock. Neither stage uses panel assessment outcomes.

Next come the 18 full B decisions, their full independent replays and the six
fixed privacy repeats. Each B decision gets 120 seconds; each original-root
replay gets 300 seconds. Requests are clamped to the remaining phase deadline.
The complete response includes compressed evidence writing and delivery.

## Assessment and final verification

One assessment worker keeps its immutable models loaded across all roots and
both waves. This avoids a second model setup at a root when the balanced first
wave ends, while retaining one active computational worker tree. Every PUCT
decision still creates fresh search state using the qualified observer seed.
No model setup or memory-admission wait occurs inside a live branch or move.

The first world at every changed root supplies only completion, time and size
to the frozen throughput gate. Remaining work proceeds only if it fits the
collection cutoff with the prescribed contingency. Terminal outcomes do not
enter that continuation decision. The fixed 32 worlds, all three fits and
unchanged structural-zero roots remain mandatory.

Every completed branch has a parent timing record, its immutable move receipts,
independently regenerated initial hidden world, full transition replay and
official terminal target. Receipts cannot be reused across branches. Primary
and independently implemented statistics must agree, including the exact
bootstrap integer-stream digest; numerical runtime identities are checked
before and after. Final acceptance binds all scientific artifacts and rechecks
consumed inputs inside the eight-hour clock.

The 2 GiB evidence cap conservatively includes the new native build directory.
All failures and partial evidence remain saved. A failed correctness, clock,
memory or disk check stops dependent work and produces no imputed score. A
subsequent repair/restart would require a separately disclosed plan, as specified
in the saved design. Larger-data collection remains deferred.
