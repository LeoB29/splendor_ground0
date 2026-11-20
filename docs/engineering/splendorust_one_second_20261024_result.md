# One-second comparison: stopped during qualification

The first frozen attempt ran on 24 October 2026 from 02:07:39 to 02:08:45
Paris time. It stopped during the additional actual-model correctness checks.
None of the 184 timed panel requests, six operational games or 200 main games
started. There is no playing-strength or one-second timing conclusion.

The implementation passed 159 model-free prelaunch tests, independent review,
and 791 independent rule-transition checks. Stable RAM admission and worker
initialization passed. These checks did not establish actual-model qualification.

## Established failure

The worker returned `ONE_SECOND_AUDIT: aggregate inference count`. Read-only
inspection identified an invalid assumption in both new verification routines:
they required native leaf evaluations to equal host neural-network calls.
The pinned native search increments its leaf counter before its turn-124
heuristic fallback, which returns without a host network evaluation. These
counters therefore need not agree.

The fixed turn-123 counterexample searches into turn124 and is the strongest
source-based explanation of the failure. The failed batch did not preserve its
individual partial decision traces, so the precise failing fixture and numeric
counts are inferred rather than recovered. This is a defect in the new checking
code; the evidence does not establish a change or defect in either player.

A separate immediate cleanup check reported the Python worker still visible
after shutdown and obscured the primary exception in the parent traceback.
The original worker error and cleanup record remain intact. The outer watchdog
completed cleanup without error, and a subsequent identity-based inspection
confirmed that all recorded owned processes were gone.

## Preservation and next step

All 1,253 frozen inputs and the bound runtime files still match. No frozen file
was patched, no allowance was reset, and no retry or comparison game was run.
The scientific protocol, exact players and larger-data deferral remain intact.

Prepare a separate checker version that distinguishes total leaf evaluations
from neural forwards using the native source semantics and targeted branch
fixtures. Save each counterexample before its assertions, and qualify the
Windows cleanup-completion check. Then freeze a new bounded attempt with the
same players, schedule, margins, sample size and success criteria. This stopped
attempt must remain preserved; passing a new checker must not relabel it.

Evidence: `outputs/splendorust_one_second_20261024_01`, particularly
`workers/qualification/request-00000001.json`, `workers/qualification/cleanup.json`,
`completion.json`, `execution.json` and `poststop_inspection.json`.
Freeze SHA256: `617b5c4852a8eddc18156b3df98b377f93c1aa57a6dad9291ad4ed478efc3f34`.
