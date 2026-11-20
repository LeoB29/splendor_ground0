# Fresh strength comparison of both one pass models

6 November 2026. Prepared under the user's instruction to do the suggested next
step: prepare a bounded comparison before launching games. This document fixes
the scientific design and reserves its sample. It is not an execution freeze.
Implementation, qualification and a later launch remain the next action.

## Question and fixed models

Does the small policy-prediction improvement after one training pass produce
stronger play against the unchanged original EF8A model? The accepted one-pass
study found policy loss lower in both repetitions, while value changes remained
uncertain. Prediction loss is not a win rate, and no value-loss gate is added.

Use both original final step96 checkpoints, `r0_broad_joint` and `r1_broad_joint`,
from `outputs/splendorust_shorter_joint_20261106_01`. Their source manifests,
checkpoint bytes, complete accepted training closeout and weight identities are
bound in the preparation evidence. Never choose a repetition, intermediate
checkpoint, retrain, average weights or initialize from an unpromoted successor.
EF8A remains the reference; its historical training origin stays explicit.

Both candidates and the reference use **128 simulations, three hidden worlds,
cpuct0.4, depth64, float32, the original full network execution, unchanged dynamic
FPU and canonical observer-derived search randomness**. This is the matched
evaluation setting used in the previous joint and policy-only studies. The
512-simulation teacher and CPU8 collection throughput are different workloads.
Keep one active request, one thread per inference worker, player CPU0 and separated
supervision on CPUs2-19, with at most two resident inference trees. Weights are the
only scientific intervention; no new GPU or concurrent-game route is introduced.

## Sample and fixed order

| Stage | Repetition 1 | Repetition 2 | Total |
|---|---:|---:|---:|
| Operational qualification | 6 games / 3 pairs | 6 games / 3 pairs | 12 games |
| Main assessment | 200 games / 100 pairs | 200 games / 100 pairs | 400 games |
| Exact decision reproductions | Up to96 | Up to96 | Up to192 |

Every pair uses one complete initial deck/market/noble state twice, with seats
swapped. The repetitions have disjoint setups: 200 independent main setup seeds,
plus six separate operational seeds. Operational outcomes are excluded from the
strength estimate. The two repetitions are fixed models, not a sample large
enough to establish performance across arbitrary future training seeds.

Reserve the namespace `splendorust_onepass_strength_20261106_v1`; setup seeds
derive by SHA256 from study, stage, repetition and local pair. Save the complete
ordered schedule and corrected initial snapshots before any model execution.
Retain global and local identities, fixed seat order, and disjoint operational,
main and analysis seeds. Check registered prior manifests/schedules/seed registries
and explicitly the previous pilot corpus and both training assessments. Disclose
oversized files and the finite registry scope; this is not an exhaustive raw-log
search. Recheck new registry entries before an execution freeze without replacing
any reserved seed on a collision.

Operational order is repetition1's six games then repetition2's six. Both must
pass before main play. Main play alternates ten-game blocks: repetition1 pairs
0-4, repetition2 pairs0-4, repetition1 pairs5-9, and so on. Seats alternate within
each pair and predetermined first seats vary. This spreads host/time effects
without restarting models for every pair. No outcome-dependent ordering, stopping,
sample extension, replacement games or checkpoint selection is allowed.

## Main estimate and interpretation

Score wins as1, draws as0.5 and losses as0. The sole primary estimate is the
equally weighted average of the two fixed candidates' mean scores against EF8A.
This is also the score of the combined400-game sample because allocation is equal.
It describes those two candidates at matched128, not a promoted single player,
all training seeds, play at512 simulations or independent-opponent strength.

Use50,000 PCG64 stratified bootstrap replicates, seed20261106128. For each draw,
sample100 whole swapped-seat pairs with replacement within repetition1, then100
within repetition2; average the two scores. Use identical draws for official and
conservative scoring. Preserve draw-stream and integer-total hashes and NumPy
version. Use linear2.5/97.5 percentiles. A separate reader must reproduce integer
winner/seat arithmetic, draws and percentiles, rather than trust the primary report.

Retain the existing useful-gain requirements, under **both** official and
conservative scoring:

- Score at least55%, with the two-sided95% interval's lower bound strictly above50%.
- Score at least50% in each candidate seat across the fixed equal allocation.
- The existing degenerate-bootstrap safeguard: if the interval collapses, the
  lower bounded-pair Hoeffding95% bound must also exceed50%. Its radius is
  `sqrt(log(40)/(2*N))`, with N=200 independent main pairs for the pooled result.
- Complete trajectory, model-reproduction, preservation and independent acceptance.

Official administrative forfeits are losses for the player responsible for an
independently classified research hard-guard failure. The conservative analysis
scores every administrative game as a candidate loss, including reference failures.
Report the full-sample administrative sensitivity bounds; never drop these games.

Report each repetition's score, seats, draws, administrative events and descriptive
paired95% interval using seeds20261106129 and20261106130, respectively, and N=100
for its degenerate safeguard. Show the same threshold checklist descriptively.
There is one primary pooled decision, no separate confirmatory claims selected
from these two intervals, and no choosing the better repetition. A pooled positive
with divergent repetition results must say the improvement is not demonstrated
uniformly. No automatic promotion follows even a positive result.

Splitting400 games gives less precision for each individual checkpoint than the
previous400-game single-candidate test. Illustrative normal-approximation half-widths
at maximal pair variance are about6.9 percentage points pooled and9.8 per repetition;
these are planning examples, not promised intervals or guaranteed power. A small
real gain can remain inconclusive.800 games would double cost to restore400 per
candidate; the current modest offline signal does not justify that initial spend.

## Required implementation and qualification

Use a new versioned package, proposed `tools/splendorust_onepass_strength_20261106`,
and new runtime output. Preserve all earlier source and frozen evidence. The
accepted **joint** assessment route in
`tools/splendorust_selfplay_pilot_resume2_20261103/assessment` is the appropriate
base. Carry forward the later qualified clock, response, preservation and resource
handling after explicit source comparison. The policy-only adapter's requirement
that shared/value weights remain unchanged is inapplicable to these joint models;
do not weaken that historical adapter or label new weights with its schema.

The diagnostic checkpoints contain state_dict, optimizer and RNG in the
`learning_diagnostic_checkpoint_v1` format. They are not directly interchangeable
with the old live `selfplay_candidate_checkpoint_v1` manifest. A separate strict
reader/export must authenticate the accepted step96 manifest, full tensor tree,
group digests, corrected identity and original-parent provenance, then export
**only the exact state_dict tensors**, without arithmetic, conversion or updates.
Independently verify tensor names, shapes, float32 bytes, ordering and full weight
digest before and after import/export. Optimizer/RNG bytes remain source evidence,
not live player inputs. This qualification is pending, not implied by preparation.

Before each new evaluator starts, call `verify_runtime_sources()` once and save
its receipt. Use the corrected ruleset `splendor_base_qualified_rules_20261018_v1`,
catalogue SHA256 `db4776918e612eee5a6f2617719c85ec5f413d592bbab0a64058babc83a70312`,
and `atomic_prereveal_completed_turn_v1`. Use strict corrected decoding and explicit
legal lists throughout observations, encoders, sampled hidden worlds, native
search transitions, referee and replay. Historical default loaders are excluded.

Require the actual graph fixtures and full existing67-decision entrypoint suite
at each new live lifecycle, including all six corrected faces, nonpurchase and
multiple nobles, noble-triggered final rounds, gold/returns, private reserves,
legal encoding, corrected hidden worlds and bad/missing identity/hash rejection.
The parent must retain exact historical parity. Each new checkpoint must pass
its own eager/compiled graph parity and native/reference search checks with the
existing tolerances unchanged; do not demand equality between different weights.
Keep clock/late-response, legal/privacy, request accounting, process ownership,
checkpoint-mixup and wrong-repetition rejection tests. No source-helper self-check
or tagged manifest substitutes for actual-player qualification.

Every main game is independently checked against the corrected dictionary ledger,
including transitions, final round, terminal result and tie-breaks. Keep the
disclosed internal search forced-pass/deadlock convention; unresolved actual-game
deadlocks and2000-ply safeguards are incomplete, not fabricated official outcomes.

For **each** repetition retain the established outcome-blind replay cells:48 main
cells (variant x seat x ply phase x private-reserve category),12 operational cells
(game x variant), and36 temporal main cells (three local-pair thirds x variant x
seat x ply phase). The fixed local thirds are0-32,33-66,67-99. Hash-order minima,
deduplication and empty-cell disclosure remain unchanged. A reference move is
tagged by its candidate matchup. Fresh qualified workers reproduce each selected
move's full scientific result, action, trace and afterstate exactly within the
same checkpoint; up to192 reproductions overall. No failed move is replayed.

## Timing and failures

Retain nine-second internal and ten-second complete-response research guards.
Keep each request's QPC/perf_counter clock distinct from the owner's monotonic
and UTC clocks; never compare or convert unlike clock domains. Test inherited
deadlines and expiry at every actual boundary, without adding a tolerance.
One second and700ms remain descriptive: correctly completed moves exceeding those
targets are accepted. Record per-player latency distributions, maxima and counts
above each descriptive threshold. Live positions differ between players; this
does not measure same-input implementation speed.

Operational hard-guard events stop admission and preserve the attempt. In the
main sample, independently classified research hard-guard events finish that game
under the loss policy; close the affected pair of workers and qualify fresh workers
for the next scheduled game. Never retry a failed decision, reset an active clock,
replace a game or discard a result. Infrastructure, malformed outputs, legality,
privacy, source, replay, unresolved ending or actual resource failures stop as
incomplete and preserve all evidence. They are not automatically player losses.

All stages advance automatically after their required checks pass once execution
is authorized. On a technical stop, diagnose promptly and prepare/test a separate
correction; no automatic scientific replay or recovery launch is included here.
A valid negative or inconclusive result ends the fixed study. No scheduled chat
monitor, repeated status messages or partial-score interpretation is introduced.

## Cost and actual bounds

The closest accepted matched128 comparison played400 main games in11h31m, then
took1h45m through independent acceptance. Its complete evidence tree is about5.4GiB.
For this study allow **roughly12-20hours including verification**, and budget about
6-10GiB evidence as an advisory estimate. The new checkpoint conversion, doubled
operational/replay coverage and host load add uncertainty. These are not admission
or stop predicates and do not justify assuming CPU8 teacher collection rates.

The proposed execution envelope retains one real UTC and monotonic origin:

| Bound | Allowance |
|---|---:|
| Overall including independent acceptance | 72hours |
| All model activity | 60hours from the same origin |
| Initial qualification / operational stage | 4hours each |
| Main assessment | 40hours |
| Fresh-process reproductions | 8hours |
| Saved finalization including independent reader | 12hours; reader at most6hours |
| Combined inference tree RSS | 2GiB; at most two trees |
| New evidence | 48GiB, including8GiB reserved for finalization |
| Actual free disk floor | 5GiB |
| Live inference worker lifecycles | At most850, with exact accounting |
| Scientific games / plies / selected replays | 412 /2000 per game /192 |

Every stage is clipped to the original global/model deadlines. Normally use
separate two-player workers for each ten-game main block, each operational
repetition and each repetition's reproductions; qualify every lifecycle. Freeze
the exact normal and worst-case worker/request ledger in the implementation plan;
the lifecycle ceiling does not authorize extra decisions or retries. Keep existing
raw stream/request/response caps and lossless recording limits unless a separately
reviewed execution plan prospectively specifies a necessary change.

No available-RAM reserve, RAM stability wait, projected-storage reservation or
resource-forecast refusal is allowed. Actual allocation/execution failures and
the finite clocks, RSS, disk and evidence limits remain enforceable. Source corpus
and accepted checkpoint evidence are read-only inputs, not copied wholesale.

## Freeze and completion

The preparation receipt binds models, sources, this design, schedule, initial
states, registry scope and independent design/feasibility reviews. It must state
no models loaded, games played, fitting or live qualification performed. A later
execution freeze must add the new qualified source/tests, strict checkpoint
conversion, independently reviewed statistics and reader, exact ownership and
resource contracts, and a fresh check of the reserved seed registry. Do not launch
the existing policy-only controller or old RAM-gated owner by mistake.

The authorized runtime would progress qualification ->12 operational games ->400
main games ->selected model reproductions ->saved audit ->separate no-Torch
analysis/figures ->independent acceptance ->actual retained owner exit and cleanup.
A completion file alone is insufficient. Check the full preserved inventory,
current source bindings, game sample, statistics, all requests, original clocks
and exact process identities. Historical superseded prelaunch declarations remain
historical; they cannot be reapplied as current-file assertions.

Deliver one strength chart with both checkpoints and the fixed pooled result,
paired intervals,50% and55% markers. A second timing chart is optional. Explain
what is and is not established, the next step and the broader project connection.
Original EF8A remains reference. Larger-data collection, external opponents,
six-world work, additional training and GPU optimization remain deferred.
