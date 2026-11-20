# Saved-position decision-quality diagnostic: prepared design

27 October 2026. **Preparation only; no evaluator or experiment has launched.**
External-candidate testing is paused at the user's request. Optimized64 remains
frozen and larger-data collection, fitting and training remain deferred.

## Question and limits

Where does the current one-second player make choices that are sensitive to
search budget or the sampled hidden cards? Do stable alternatives emerge with
more computation, and is the extra effort concentrated in a few decisions?
This can identify a concrete search-allocation hypothesis before changing a
player. It cannot establish that a different move wins more games.

Use a small, fixed selection of accepted saved positions. No fresh games,
full-game continuations, new opponent, training targets or parameter sweep are
part of this design. Higher-budget searches use the same imperfect network and
are diagnostic references, not an oracle. The two held-out seed families test
repeatability under different lawful samples; they are not an independent
evaluator, independent training lineage or held-out playing-strength test.

## Preserved starting point

The deployment reference is the exact graph-backed optimized64 player with
separated supervision, a 1.000-second complete-response deadline and its original
0.900-second internal guard. Its existing one-second evidence is unchanged.

| Identity | SHA256 or value |
| --- | --- |
| Optimized64 profile | `2b60b56419c938ecd5e96d5b9c9a5153f36b2796d4e4b35219b746dc7dfed6d9` |
| Historical pretrained checkpoint | `ef8a4521cd6c03c7075f15efee23f4cde6ec94d1b5398765cf0c24a09ad745cb` |
| Frozen native bridge | `b15322f1c6aac1034638bdc52d91524639c63cbc2b61d361a9fef79dfdea728c` |
| Upstream revision | `1828507b8d2f0e02bc03d102b9c6d29a54fa8d57` |
| Corrected ruleset | `splendor_base_qualified_rules_20261018_v1` |
| Catalogue | `db4776918e612eee5a6f2617719c85ec5f413d592bbab0a64058babc83a70312` |
| Turn contract | `atomic_prereveal_completed_turn_v1` |

Retain float32 weights/arithmetic and the qualified graph, one numerical thread,
three sampled root worlds, depth64, cpuct0.4, dynamic FPU0.02965, batch1, no added
noise/uniform prior, fresh per-phase native search and the same Main support,
ordering, safety filter, auxiliary routines and hidden-information boundaries.
The inherited checkpoint's historical provenance is not relabelled corrected
training. No sequential-turn or forced-pass/deadlock migration is authorized.

The earlier 200-game strength comparison scored optimized64 at64.25% against
unchanged32, including its retained deadline losses. Separately, the recovered
reliability audit accepted all200 natural games/11,020 live moves with zero
observed player faults. These are distinct studies and claims. The latter's
original attempt and failed recovery attempts remain incomplete/failed.

## Fixed saved panel

Source: `outputs/splendorust_isolated_reliability_20261026_01`, authenticated by
the accepted recovered verification in
`outputs/splendorust_reliability_finalization_20261026_01`.
Bind original freeze `a7be16152ff398eb5feca4f68419539c5817148d9fe31d74c7441ae6a5b63497`,
finalization freeze `fd33f23af6d88179eee1ee0f783b1205f253a8a413657d561f60e7bdfbc3c02b`
and independent review `54f0d7a4babb1970a2d47118988f66683f33f3dbc1bd8878c837f07ca4836c9a`.
Reuse accepted receipts rather than repeating the entire historical audit.

The source contains200 games: six operational and194 main games/97 paired
worlds. **Exclude the six operational games.** Select24 completed optimized64
Main decisions: four in each absolute-game-ply band (0-19,20-39,40+) crossed with
actor seat0/1. Select no more than one position from any paired world, hence no
more than one per game. Require at least two supported native Main actions;
atomic compound-action count is not a substitute for this requirement.

The preparation selector ranks identities by the fixed domain
`decision_quality_saved_panel_20261027_v1`, source paths and deterministic hashes.
Its receipt specifies exact ordering and tie handling. It ignores game outcome,
response time, chosen action, network values and whether deeper search is likely
to disagree. Phase and actor are eligibility variables, not performance filters.
Read at most512 candidate rows; stop rather than relaxing a deficient cell.
Bind selected source hashes to the preserved inventory and verify corrected
envelopes. Save only lawful observations in the panel; private referee snapshots
remain in the original evidence. The future player worker must never receive
them, deck order, game seed, score/result labels or the source game identity.

Panel and preparation receipts live under
`outputs/decision_quality_preparation_20261027_01`.
The design plan and its preparation seal live separately under
`outputs/decision_quality_design_20261027_01`.
The materialized panel contains24 roots from24 paired worlds, with four per cell;
the selector opened62 saved move rows. Its SHA256 is
`bf6de558236bc006295197036e318d681a1c4b6bb5e79009e55d90676f866153`.
The selection receipt SHA256 is
`016baa66aac06517cf8296bba2ef2ed6e886118bdc0454984145f9490347dfb4`.
This balanced purposive panel is not a random sample of all future decisions;
do not interpret its disagreement fraction as a population blunder rate.

## Fixed measurement matrix

| Per saved root | Seed | Budget and purpose |
| --- | --- | --- |
| Frozen optimized64 reproduction | A, original | 64 simulations, original1.000/0.900s clocks; exact recorded action and deterministic scientific result |
| Research64 | A | Research wrapper/telemetry control |
| Research256 | A | Intermediate search budget |
| Research1024 | A | Deeper discovery reference |
| Research1024 | B | Held-out sample stability check |
| Research1024 | C | Second held-out sample stability check |

There are24 original-clock reproductions and120 research requests. Add exactly
12 research reproductions selected before results: one fixed root per six cells,
each repeating Research64-A and Research1024-B. No failed request is retried.
Qualification fixtures are additional, finite, and must be enumerated in the
implementation freeze before model launch; their stage cap is separate below.

A is the original seed `int(digest(observer)[:16],16)` and retains its exact
per-phase use. For B/C use the first16 hex digits of SHA256 over UTF-8
`decision_quality_saved_panel_20261027_v1|B|` (or `|C|`) followed by the same
canonical observer serialization used by `digest`. Preserve per-phase reuse
within a complete turn. Require A/B/C to be distinct for every selected root;
collision is a preparation failure, not a reason to search for another seed.
Seed construction uses only information available to the acting player.
Changing seed changes stochastic search/world sampling together; this design
does not causally isolate hidden-world uncertainty from search randomness.

Freeze an order interleaving all six cells. Rotate the five research conditions
by panel index modulo5 to distribute ordering effects. Complete all A conditions
before examining B/C in analysis. Always collect B/C for every planned root,
including roots where A agrees with64; do not select only apparent mistakes.
For any alternative, the predeclared comparison is the original64 Main action
against the1024-A Main action, with B/C read only after that pair is fixed.
The prepared request matrix fixes all156 non-qualification requests and their
seeds. It uses phase-then-seat cell order, selection-hash order within each cell,
and four round-robin passes. Its first six roots form the pilot and reproduction
subset. Qualification fixtures must be added to the later execution inventory.

The research wrapper has a **30.000-second complete-response limit** and a
29.000-second internal guard, including all mandatory turn phases. This is
deliberately extra research computation, not a relaxation of the one-second
deployment target or a new deployable player. Work counts must show the requested
quota was actually completed under the existing quota semantics. Record genuine
single-action/forced-phase shortcuts separately, without inventing simulations.
In particular, verify that none of the selected Main decisions reaches the
existing turn-index124 heuristic shortcut; a shortcut is not a completed search.

## Additive implementation and qualification gates

Future implementation belongs in a new versioned package and build directory.
The frozen Python wrappers permit only32/64 and0.900s; do not modify them,
monkey-patch shared modules, or relabel a larger-quota run as optimized64.
The native engine already supports larger iteration quotas. A separate bridge
may expose its existing `capture_search_targets` root visits, optional Q values,
root search value and network value. Capture only the already-computed Main
root statistics: no extra forward calls, search iterations or RNG consumption.
Record final compound action separately from Main action.

Qualification must prove the new bridge's telemetry-on/off paths and Research64-A
produce the same action, exact callback inputs/outputs and float32 bits as the
frozen control, excluding only declared clock/identity/telemetry fields. Require
the existing11 graph-parity cases and52 actual-entrypoint corrected-rule cases
through each applicable frozen/new route, with a finite request inventory.
Telemetry must not change the search decision, legal support/order or native
trajectory. Do not silently fall back to an action-only study if telemetry fails.

Every new preparation/evaluator process calls `verify_runtime_sources()` once
at startup. The actual evaluator/loader chain must demonstrate the corrected
six card faces, noble after non-purchase, multiple noble choices, noble-triggered
final round, gold/payment/returns, private reserves, explicit corrected legal
support and all90 corrected card objects in hidden-world reconstruction.
Reject missing/legacy/wrong identities, hashes and prohibited hidden fields.
Apply the strict helpers and actual-entrypoint counterexamples in
`future_training_rules_20261021.md`; a manifest or helper self-check alone is
insufficient. Use independent corrected transition checks where a fixture steps
a state. Any such fixture transitions are qualification, not new games.

Use lawful-observation equality, hidden-state perturbation counterexamples and
seed bindings to prove that private referee state never reaches the player.
Preserve internal cutoff/deadlock conventions and do not turn them into official
terminal targets. No dataset or training checkpoint is created.

The first six selected roots (one per cell) form a predetermined throughput
pilot within the same120-request workload. Reuse them, with no rerun or discarded
warm-up. Admit the remaining18 only when a1.5x measured time/evidence projection,
including all startup, graph proofs, communication, telemetry and final audit,
fits every remaining bound. The gate uses costs and integrity only, never actions
or agreement. A failed gate preserves a partial diagnostic; it does not authorize
a smaller quota, replacement root, cap increase or restart.

## Measurements and interpretation

For all24 roots show original action, each research Main action, final compound
action, actual simulations, neural callbacks, complete elapsed time and failure
status. Report64-A to256-A,64-A to1024-A and256-A to1024-A action changes;
report B/C agreement with the fixed A alternative and with each other.

Values are acting-player win credits, not signed values or calibrated
probabilities. `SearchTargets.root_value` is the visit-weighted mean across root
edges; it must not be confused with `agent.root_value()`, which is the selected
edge mean. `SearchTargets.network_value` is the root network estimate. Unvisited
edges have missing Q, never zero. Compare the two fixed
actions' Q only if both edges were visited in that particular reference; a
cross-reference preference requires both actions to be observed in BOTH B and C.
Publish every missing comparison and visit count. Q gaps and root value changes
are internal model/search estimates; do not call them regret, independent
evaluation, proven mistakes or actual win-probability improvement. Existing
root64 callbacks supply raw policy logits, not final normalized priors. Report
them as logits unless the exact frozen legal/safety filtering and native prior
transformation is reproduced and qualified, without an extra network call.

Counts are descriptive, with denominators and per-cell results. No significance
test, Elo, achieved strength score or automatic success threshold is attached to
this deliberately small panel. Never use the eventual recorded game winner to
label one isolated move right/wrong, or consult hidden future draws to resolve
an uncertain decision. Saved games used for diagnosis are development evidence,
not fresh confirmation for a later tuned successor.

The report must recommend at most one subsequent hypothesis:

- Stable budget-dependent alternatives and concentrated callback costs justify
  considering an adaptive allocation study under one second. They do not prove
  those alternatives are stronger or authorize deploying1024.
- Persistent B/C instability suggests a separate study of world/search sampling
  or robust aggregation, without declaring three worlds inherently inadequate.
- Few stable changes provide little support for simply increasing search. They
  do not establish that the value head, model size or training data is the cause.
- Missing support/value evidence or qualification/resource failures remain
  unresolved. Do not force a candidate-coverage-versus-evaluation conclusion.

Any proposed player change needs a separately frozen fresh paired whole-game
comparison against optimized64 under the real one-second limit and usual
timeout-loss policy. Independent opponents remain useful for later generalization
and exceptional-strength claims but are paused during this work.

## Bounded execution plan for a later authorized run

One owning guardian provides non-resettable monotonic and UTC clocks. Total cap
**7,200 seconds/two hours**, including RAM waiting, worker startup and verification.
Qualification cap900s; diagnostic collection cap4,500s; finalization cap1,800s.
Stop collection by elapsed5,400s even if preceding stages used less time. All
stage clocks are clamped to the common hard deadline. No automatic retry or
extension. Known request ceilings alone are120x30 +12x30 +24x1 =3,984s; this
leaves516s within collection for startup and I/O, subject to the measured gate.
This is an upper bound on permitted work, not a predicted completion time.

Maintain the qualified CPU placement: active player/helpers on logical CPU0,
supervisors on2-19, normal priority, one serial player at a time. Check topology
and identities. Require2GiB available RAM stable60s before play; check between
requests and bound cumulative waiting at300s within the job. Keep the2GiB sampled
aggregate worker RSS cap,7GiB initial disk and5GiB maintained free disk.

Execution evidence cap1GiB, with512MiB reserved for finalization. Charge compressed
logical logs, physical files, late-written proofs and all worker sidecars using
the already qualified accounting/compression discipline. Never delete evidence
to fit. A model-free projection is required before launch, and the fixed six-root
pilot recomputes it before expansion. Compilation/build artifacts are inventoried
separately during implementation preparation, not hidden in model-run accounting.

Stop at the first correctness, identity, evidence, process, resource or request
deadline failure; preserve the whole completed prefix and the failed request.
A failed saved request is a diagnostic failure, not a newly played game loss.
There is no new game to score. Do not retry or replace the position. Existing
historical timeout losses remain unchanged. Distinguish failed/incomplete,
verified partial diagnostic and fully verified complete diagnostic.
A present-day frozen64 timeout is an operational reproduction failure, not
proof of historical nondeterminism or an incorrect saved action. An exact
scientific mismatch is a distinct integrity failure. Preserve both categories
without changing the original successful record.

Verify all request receipts, corrected reconstruction, deterministic replays,
callback/root-statistic bindings, log hashes, actual owner exits and cleanup.
Hash the selected originals and frozen dependencies before/after. Reuse accepted
historical receipts rather than reading/replaying all200 games again. Bind only
the finite required preservation inventory, seal new evidence, and have an
independent receipt review reconcile coverage and all failures.

## Preparation acceptance and launch boundary

This preparation is complete only when the24-root panel, selection receipts,
protocol, machine-readable plan and independent design review are bound together.
It is a **design seal, not an execution freeze**. No execution deadline has begun.
Implementation still needs its additive source, tests, binary/profile/runtime
pins, exact fixture count/order, resource projection and independent prelaunch
review, then a new execution freeze before any model run. The current user
authorized preparation, not that implementation/run.
