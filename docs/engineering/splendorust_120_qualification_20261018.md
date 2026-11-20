# SplendoRust qualification for a 120-second comparison

This prospective protocol selects a larger, fixed SplendoRust search quota using
completion times alone, then qualifies that exact profile on the existing
corrected-rules fixtures. It authorizes no benchmark games, training, larger
data collection, or changes to frozen evidence. A subsequent benchmark is a
recommendation requiring its own frozen schedule and execution authorization.

## Preserved player and evidence

Use the existing observer-only native binary, pinned source and Entity model
already qualified for the completed external screen:

- Source commit: `1828507b8d2f0e02bc03d102b9c6d29a54fa8d57`.
- Checkpoint SHA256:
  `ef8a4521cd6c03c7075f15efee23f4cde6ec94d1b5398765cf0c24a09ad745cb`.
- Native binary SHA256:
  `b15322f1c6aac1034638bdc52d91524639c63cbc2b61d361a9fef79dfdea728c`.
- Binary path:
  `outputs/external_opponent_search_20261018_02/native_build_v2/target/x86_64-pc-windows-gnu/release/splendor-observer-bridge.exe`.
- Prior source and timing qualification:
  `outputs/external_strength_screen_20261018_02/qualification/source_attempt_02.json`
  and `timing_attempt_03.json` in that same directory.
- Fixture source:
  `outputs/external_strength_screen_20261018_02/fixtures.json`.

Require the bound prior files, source qualification, model, executable and
dependencies to retain their recorded sizes and SHA256 hashes before and after
this execution. Bind every newly introduced runner, worker, profile validator,
supervisor, test, schedule and this protocol before the first calibration
request. Retain earlier failed attempts and completed evidence unchanged.
No binary rebuild or repeated native-versus-host model parity inference is
needed when the authenticated dependencies of the successful prior check are
unchanged. A failed preservation check stops this qualification.

The corrected referee is `splendor_base_qualified_rules_20261018_v1`, catalogue
SHA256 `db4776918e612eee5a6f2617719c85ec5f413d592bbab0a64058babc83a70312`.
These are labelled corrected-rules counterfactual positions, not historical
replays. Do not require old actions, search values or outcomes to match.

## Fixed profile and clock

Retain canonical PUCT, depth 64, world pool 3, chance universes 3, cpuct 0.4, dynamic
FPU, FPU reduction 0.02965, zero uniform prior, no noise, batch size 1 and the
same atomic pre-reveal adapter. The chance-universes setting is inactive in
this canonical path. Increase only the requested iteration quota and change
the internal fail-closed allowance to 108 seconds. Give each new profile a
distinct identity binding its quota and allowances.

The existing native transport accepts a checked unsigned32-bit iteration
count. Its canonical scalar search loops over `0..iterations`, with no native
elapsed-time cutoff configured. Thus all proposed quotas through 6,400 are
supported without changing the binary; their runtime and memory use are not
assumed to be validated. A completed non-heuristic Main phase must report
exactly its requested number of simulations. Root evaluation can add a leaf
inference beyond that count. Record actual simulation, leaf-inference and
host-inference counts rather than treating those quantities as interchangeable.

Increasing quota revisits the same three sampled worlds; it does not add
independent determinizations or increase depth. Retain the upstream heuristic
branches for auxiliary phases, a single Main choice, and turns at or beyond 124,
and the explicit forced-pass adapter. Record those branches separately from
neural search. This remains a local adaptation, not the published native
PUCT6400 competition entrant.

Every calibration and final request uses a fresh Python worker and fresh Rust
child. Initialize only the authenticated opponent and its fixed architecture
warm-up, with no fixture input, within a separate 60-second limit. No position,
tree, inference result or public-history cache persists between requests. A
new search agent is used for every native phase. The seed derives solely from
the lawful observation hash; never pass the referee seed, actual deck order,
or an opponent's hidden reserve face.

Use the same assigned physical performance core, normal priority and one
numerical thread as the corrected planner. The trusted parent starts the
120-second move clock before input serialization and accepts only a complete
response received within that deadline. Include serialization, IPC, model
inference, search, auxiliary choices and output delivery. Later parent
validation does not change a packet's trusted arrival time. Reject unsolicited,
malformed, wrong-request, late or illegal responses. Use trusted
`receipt_elapsed_seconds` for the 96-second selection margin; retain both receipt
time and total supervised elapsed time.

The 108-second internal check raises a failure rather than returning a
partially searched fallback action. It is not the authoritative clock and may
be reached only at particular adapter boundaries; the parent still enforces
the complete 120-second deadline. No adaptive quota reduction or early action
return is permitted. Preserve the frozen external02 validator, which limits
its old internal allowance to 0.9 seconds; the new preparation must introduce
an explicitly authenticated validator for this new profile.

## Predetermined calibration and selection

The quota ladder is `[256, 1024, 4096, 6400]`, in that order. For each quota,
use the following seven original fixtures in the recorded order. Indices are
zero-based in the preserved 23-fixture source.

| Index | Root | Predetermined reason |
|---:|---|---|
| 0 | `0204:010` | First core phase 0 position for acting seat 0 |
| 3 | `0385:017` | First core phase 0 position for acting seat 1 |
| 18 | `0084:048` | Own hidden reserve edge |
| 19 | `0342:006` | Opponent hidden reserve edge |
| 20 | `0323:031` | Both players' hidden reserves edge |
| 21 | `0171:025` | Small canonical support edge |
| 22 | `0252:040` | Intermediate canonical support edge |

This subset is fixed from existing fixture labels, not newly measured search
times or actions. SplendoRust has no local proposal-fit dependency. All three
local-fit fixture groups remain covered by the complete final gate.

A quota passes calibration only if all seven requests finish legally,
complete their declared work, pass observer and dictionary-referee checks,
and have trusted receipt time at most 96 seconds. On the first external move
timeout, the recognized internal fixed-work timeout, or a timely response
above 96 seconds, preserve that case, reject that quota, and stop the ladder.
Select the largest earlier quota whose full seven-case calibration passed.
If the first quota is rejected, no profile is selected. If 6,400 passes, select
it and stop at that predetermined ceiling; do not expand the ladder to fill
unused time.

Initialization failure, non-timing crash or error, malformed or illegal output,
wrong work count, observer/transition mismatch, failed input authentication,
or resource-limit failure stops the entire qualification. These are not
reasons to select a lower quota. Likewise, insufficient remaining global
reserve means incomplete qualification, not a timing rejection or fallback.
No action quality, evaluation score or match outcome enters selection.

## Final gate

Freeze the selected profile, then run exactly 46 requests: all 23 original
fixtures and their predetermined hidden-state variants, in source order,
original followed by variant for each fixture. Every request has its own
fresh worker and the same 60-second initialization, 108-second internal and
120-second external allowances. There is no reuse of calibration responses.

Verify that each variant changes hidden state while producing exactly the
same lawful observer input. Authenticate the variant construction and record
both observer hashes. Require the two complete scientific response payloads
to match exactly, including selected action, phase choices, work counts,
observer identity and profile identity. Clock, PID and memory telemetry are
external receipts and are not part of this deterministic equality test.

For each timely response, independently validate the lawful observer against
the full referee snapshot, the complete legal-support digest and the chosen
atomic action. Compute its transition with the dictionary ledger oracle and
compare the full resulting state with the corrected referee, including
payments, returns, reserves, reveals, compulsory noble choice, scores and
end-of-round/terminal semantics. The full hidden state remains confined to
the parent referee. Record legal and transition hashes and all receipt/profile
bindings. These are isolated one-step fixture checks, not played games.

All 46 requests must complete their fixed work within 120 seconds, be legal,
pass these checks and satisfy all 23 deterministic pairs. Preserve ordinary
external move timeouts and recognized internal budget exhaustion, then
continue the remaining fixed cases to measure the full predetermined panel.
Any such timing failure prevents qualification even if all other cases pass;
a completed negative gate is distinct from an interrupted execution. Compare
a deterministic pair only when both responses completed correctly, and require
all 23 matched pairs for acceptance.

Stop immediately on initialization failure, non-timing worker/adapter error,
malformed or illegal output, wrong work count, observer or referee mismatch,
deterministic pair mismatch, or a resource/input-authentication failure. Retain
every attempted case. Never downshift, retry, replace a fixture, change a seed,
or choose a new quota after a final-gate failure. A necessary implementation
repair requires a separate preserved attempt and a prospective review before
execution.

## Resource and time envelope

Run serially under one execution lease and one four-hour absolute deadline,
starting before the first calibration initialization. Include all calibration,
final initialization and requests, referee checks, cleanup, source checks and
final receipts in this shared 14,400-second envelope. Preparation and synthetic
tests preceding the execution are separately recorded.

There are at most 28 calibration requests and 46 final requests. At the hard
per-case allowances, `74 * (60 + 120) + 600 = 13,920` seconds, leaving 480
seconds for supervision and inter-case work. This is allowance arithmetic,
not a runtime forecast or a guarantee that overhead will fit. Retain 600
seconds for final checks and receipts. Before every calibration request,
require at least `46 * 180 + 600 + 180 = 9,060` seconds remaining; before
every final request require at least `180 + 600 = 780` seconds remaining.
Do not start a case when its reserve check fails. The live absolute deadline
and guards remain authoritative during initialization, requests and cleanup.

Bound new execution evidence to 256 MiB, preserving 8 MiB for stop/final receipts,
and require at least 5 GiB free disk. Before each fresh worker require 1 GiB
available physical memory and 3 GiB commit headroom. Enforce combined live
Python/Rust/descendant RSS at 2 GiB. Inventory evidence bytes without deleting
prior files. Use exclusive writes and bounded logs/messages.

The supervisor authenticates the Python and Rust executables and, on Windows,
only the exact OS console helpers belonging to them. Account for all allowed
descendants in sampled CPU and RSS. An authenticated console helper may be
pinned once during initialization with a recorded before/after affinity;
later affinity drift, process replacement, an unexpected executable or an
unobservable process tree fails supervision. Terminate the entire job on
failure and confirm cleanup. Sampled resource maxima are not continuous
measurements. Live disk/evidence/time guards stop execution while preserving
the reserved final receipt space.

## Acceptance, limitations and benchmark recommendation

Accept only after the selected quota passed calibration, all 46 final checks
passed, all prior and new frozen inputs retain their identities, and worker
cleanup and final receipts are complete. Retain every rejected quota and
attempted case. Report calibration and final timing separately, including
initialization, receipt latency, CPU time, RSS, work counts, heuristic branches
and evidence size. Timeouts are censored, not completed latency samples.

If accepted, recommend the selected fixed profile against the corrected
four-candidate/128-world planner at the same 120-second external ceiling,
same core and thread constraints, and prospectively fixed initialization and
evidence-delivery policies. The planner's completed qualification supports
120 seconds with thin observed margin. Classify the opponent as having at
least 20% observed margin only if every final receipt is at most 96 seconds;
otherwise report a conditional 120-second qualification with thin observed
margin. Never recommend an untested lower allowance or increase it after
seeing the results. Equal ceilings do not imply equal actual computation,
simulation count, world coverage or predictive strength.

The dictionary oracle checks the actual chosen referee transition, not every
internal Rust search transition, PUCT backup or neural calculation. Those
internals rely on authenticated upstream source, the existing compatibility
and numerical-parity evidence, exact work diagnostics, and deterministic
repetition. The host model, semantic catalogue and parts of observer/legal
plumbing are shared dependencies; do not describe this as wholly independent
search verification. A finite development fixture panel does not prove all
future turns will meet the clock or demonstrate a stronger player. No match
outcome is produced by this qualification, and no benchmark launches from its
acceptance alone.
