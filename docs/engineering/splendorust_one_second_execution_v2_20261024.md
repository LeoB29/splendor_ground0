# One-second Entity comparison: separate checker v2, 24 October 2026

The user authorized a separate checker repair, preservation of the failed
attempt, and one newly frozen bounded attempt. The scientific protocol remains
[the prepared comparison](splendorust_one_second_comparison_20261024.md).
The implementation and resource contract in
[the original execution specification](splendorust_one_second_execution_20261024.md)
applies except for the explicit checker, evidence and cleanup changes below.
No player, checkpoint, native binary, search setting, sample, outcome criterion
or resource allowance changes. Larger-data work remains deferred.

## Separate source and evidence

New source: `tools/splendorust_one_second_v2_20261024`.
New evidence: `outputs/splendorust_one_second_20261024_02`.
The failed source and `outputs/splendorust_one_second_20261024_01` are preserved.
The latter freeze is SHA256
`617b5c4852a8eddc18156b3df98b377f93c1aa57a6dad9291ad4ed478efc3f34`.
The new freeze binds its original 1,253 inputs, all inventoried failed-attempt
evidence and source, and every new dependency, test and preparation receipt.
No actual checkpoint inference precedes the new freeze. Model-free regressions
may exercise the exact pinned native executable with deterministic dummy outputs.

The first attempt stopped during qualification, before any timed panel decision
or game. Its 103 prospective worlds are therefore still unplayed. This attempt
uses the exact prepared 206-game schedule and original stage-global game indices.
The collision check explicitly excludes only the old unplayed schedule in
addition to this study's preparation and new output. Prior completion status,
schedule equality and absence of any prior game directory are checked before
allowing that exclusion. This is a new disclosed attempt with a new clock origin;
it does not repair, resume or relabel the failed frozen attempt.

## Corrected accounting, unchanged decisions

The native `leaf_calls` counter includes late-game heuristic evaluations that
never invoke the network. Its equality with actual neural forwards was an
incorrect assertion. The new checker counts actual host callbacks independently
and checks native work against the exact pinned branch semantics:

- Auxiliary, single-choice and turn-124-or-later heuristic branches use zero
  search simulations, zero leaf evaluations and zero neural calls.
- A searched Main phase retains exactly 16 simulations and between 1 and 17
  total leaf evaluations. Actual neural calls are at least one and no greater
  than those evaluations. Each callback has exactly one input row.
- Before turn 108, a fresh 16-simulation tree cannot reach the turn-124 fallback,
  so leaf evaluations and neural calls must still agree. At turn 123, only the
  root invokes the network: exactly one callback, irrespective of heuristic
  leaves below it. Turns 108–122 may contain both kinds.

These bounds rely on the frozen scalar, fresh-agent, zero-rollout implementation:
one root evaluation and at most one new evaluated node per simulation, with
absolute turns in canonical keys and one turn increment per completed move.
They do not independently reconstruct every hypothetical native search path.
That internal path attribution remains an explicitly stated native trust boundary.

Every search phase records compact hashes of its actual callback inputs/outputs,
count and batch-one row count; aggregate host counters must match those receipts.
Diagnostic probes additionally retain the original first-input/output arrays.
Hashing and response construction occur within the same timed request. The new
wrapper identity distinguishes this instrumentation while the scientific player
identity, neural arithmetic and selected-action procedure remain unchanged.
Historical comparisons strip only declared instrumentation fields; original
action, simulation, leaf and legal-support diagnostics must still match exactly.
Repeated current decisions also compare the new callback receipts.

## Failure preservation and cleanup

Each counterexample records its identifier, input and returned result before
assertions; a partial failure leaves those records and the primary error intact.
Timed panel and actual-game records are also saved before independent audits.
The existing turn-123/124 cases explicitly check the repaired distinction within
the same request quotas. A fixed-output native regression reproduces the former
false assertion, and corruption tests ensure genuine accounting errors still fail.

Shutdown now waits for actual process exit, using PID plus creation-time identity,
rather than treating a still-enumerable Windows process object as a live process.
All cleanup shares a maximum five-second budget, clamped to the existing absolute
stage/global deadline. Primary and secondary cleanup errors are recorded separately.
A cleanup problem cannot establish acceptance or erase the original failure.
An owning Windows job and the independent outer watchdog still control descendants.
Tests cover a real owned child, delayed Windows visibility, PID reuse and deadline
clamping without loading a model.

## Unchanged gates and limits

One-second parent complete-move deadline; 0.900-second internal guard;
0.700-second qualification/operational margin. Exactly 52 additional complete
counterexample decisions, 101 native checks, 26 rejected malformed requests and
184 timed panel requests precede six excluded operational games. Their fixed,
outcome-blind time/storage/replay forecast must pass before 200 main games.
All actual moves receive independent replay; the predetermined sample includes
at most 48 complete-decision reproductions. Paired analysis, evidence sealing,
unchanged source/runtime hashes and successful cleanup remain required.

Overall cap: six hours from before initial admission. Stage caps: one hour for
qualification, 30 minutes operational, four hours main, 30 minutes final;
collection also stops at hour 5.5. Stable initial RAM admission and bounded
between-request RAM waiting remain unchanged. Combined worker-tree RSS: 2 GiB.
Evidence: 512 MiB including 64 MiB final reserve. Disk: 7 GiB initial, 5 GiB kept
free. The operational storage projection includes the new callback telemetry and
already written forensic evidence; neither quota nor verification may be reduced
to fit. Attributable main-game timeouts remain losses; infrastructure, rules,
resources or supervisory failures stop incomplete. No automatic retry, replacement
game, extended clock, training, or player promotion is authorized.
