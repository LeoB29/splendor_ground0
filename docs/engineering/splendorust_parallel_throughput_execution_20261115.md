# CPU parallel-throughput pilot — preparation only

This implements the 15 November brief in a separate package,
`tools/splendorust_parallel_throughput_20261115`. It authorizes no launch. The
current 3,328-game collection, corpus acceptance and final verification continue
unchanged. Future launch requires explicit user instruction, quiet-machine
confirmation, accepted independent collection verification, actual root and
launcher exits and complete recorded cleanup. There is no automatic follow-up.

## Question and evidence

Can independent CPU processes and overlapping the existing checks increase
**verified games per hour**, while reproducing exactly the same decisions?
The evidence review is in
`outputs/splendorust_parallel_throughput_preparation_20261115_01/evidence_review.md`.
Existing measurements justify testing this hypothesis. They do not establish a
Python bottleneck: the earlier CPU1/2/4 test shared its model, the live/collection
call-rate estimates concern different workloads, and the historical idle-CPU
claim has no saved measurement receipt. No strength conclusion is intended.

## Fixed experiment

EF8A weights, the accepted float32 CPU graph, batch-one inference, 512 total
simulations, three worlds, cpuct0.4, depth64, dynamic FPU, corrected catalogue and
atomic pre-reveal turn contract remain fixed. CPU8 dispatches each CPU request
immediately as physical batch1; eight clients do not imply an eight-row batch.

Live: sixteen fresh setups, each in both arms. A1 has one serial model process;
A4 has four independent model processes and a shared, fixed once-only queue.
Order: A1 setups0–7, A4 setups0–7, A4 setups8–15, A1 setups8–15. All seats use
the three-world control player. Identity is checked after the first matched eight
and again when all sixteen are available. A mismatch immediately stops work.

Collection: two fresh sixteen-setup cells X/Y, each in both arms. Order is
controlX/checks, candidateX, candidateY concurrently with candidateX checks,
candidateY checks, controlY/checks. The control retains current CPU8 (one service,
eight clients, allowed CPUs0–18, supervisor19). The candidate uses four separate
services with two clients each. This deliberate control-affinity exception
preserves the existing baseline; pinning eight control clients to one core would
handicap it. An X check or identity failure stops and quarantines overlapping Y.

Windows' physical-core and CPU-set APIs must agree at each startup. Observed
placement is model logical CPUs0,2,4,6 on four distinct performance cores,
audit/replay CPU8 on a fifth, controllers12–19 on efficiency cores (controlCPU8
supervisor19 as above). SMT siblings are never treated as independent cores.
Temporary restoration of the original allowed mask permits trusted admission of
fresh model/replay services; the controller mask is restored even on failure.

The96 main engineering games, and reused startup qualification's excluded games,
never enter any training corpus. Seeds are reserved prospectively and checked
against registered schedules/reservations, including all3,328 active collection
setups and its qualification starts. A collision fails, with no seed redraw.
The score-free registry scan is repeated before future launch. Coverage is every
discovered manifest/schedule/seed-registry/panel/freeze/selection/reservation JSON;
it cannot certify hypothetical unregistered seeds in arbitrary raw logs.

## Qualification and verification

Preparation runs no model or game. Tests cover allocation, physical placement,
exclusive queue claims, actual orchestration with fake services, overlap failure
and quarantine, exact-value mismatch rejection, preserved clocks and a real
detached model-free success/failure lifecycle. One independent review covers the
final code. The active collection's frozen sources and evidence are read-only.

After a later authorized launch, the existing actual live counterexamples and
collector116 fixture decisions, two excluded engineering games, exact replays
and actual corrected loader checks must pass first. Every collector/evaluator
worker, including saved-only audit workers, calls the corrected runtime-source
guard once at startup. A separate live model-service process also checks its
own sources. Qualification does not train or promote a scientific checkpoint.

Full actions, phase root visit vectors, root scalar bytes, complete scientific
result packets and terminal outcomes must agree for matched setups. Collection
also compares every scientific target field and float32 bytes. Each original
target's provenance digest is authenticated; only its clock/path-dependent
provenance is excluded from cross-arm identity, not any target or game content.
No tolerance is substituted. Original wire/transition/capture audits, selected
fresh-service replays, exact actual process exits and cleanup remain mandatory.
Live replay selection is fixed at up to two hash-selected moves per game,
covering early and late play when present; both arms pay the same checks.

Any technical failure is preserved and ends the pilot. There are no retries,
replacement games or in-pilot retuning. A valid failure to meet a speed bar is
an engineering negative result, not a reason to retry. Independent final
saved-evidence acceptance and observed actual owner exit are required; a
completion marker alone does not qualify a configuration.

## Measures, thresholds and limits

Report games/h and decisions/h through all route checks and actual exits,
response median/p95/maximum, CPU consumption, sampled peak RSS, and all failures.
Live phase time includes initialization; collection reports first game assignment
to final checks, with initialization also retained separately. Shared cross-arm
identity work is charged equally to arms to avoid an order penalty. Collection
reports its one observed two-cell completion interval; two cells do not establish
long-run steady-state throughput. Ownership polling supplies resource samples;
there is no extra high-frequency host sampler.

| Future configuration | Required engineering result |
|---|---|
| A4 live play | At least2.0× A1 verified throughput, every identity check exact, zero failures, maximum response strictly below60s |
| Four-process collection | At least1.3× currentCPU8 verified throughput including checks, every identity check exact, zero failures, all audits pass |

Live clocks stay119s internal/120s complete; collection stays60/75s. These are
unchanged entrypoint deadlines. Total cap6h; model cutoff4.5h from the original
launch clock, never reset. Limits:2GiB per process,12GiB aggregate,10GiB new
evidence and5GiB actual free-disk floor. Expected runtime about3h is a forecast,
not an admission gate. No available-RAM or forecast-based gate is added.

## Prepared deliverable and later use

The future report supplies the throughput table, identity/audit receipts, pass or
fail for each engineering bar and at most one summary chart. Until this pilot
actually runs, performance and adoption remain **unmeasured**. Its first intended
application is the fresh live strength test after the4,096-game training study;
the collection recommendation applies to a later corpus, never the running one.
External opponents, GPU work, search changes, new scientific training and strength
claims are outside this pilot.
