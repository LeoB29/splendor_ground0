# Bounded one-second comparison: Entity policy versus Entity PUCT16

24 October 2026. **Prepared design only.** The user authorized preparation of
this comparison, not implementation or execution of an evaluator, model run,
training job or game study. This document is not an execution freeze. A later
implementation must pass review, bind its exact sources and schedules, and
receive execution authorization before the clocks below start.

Preparation artifacts are saved separately: the
[preparation manifest](../../outputs/splendorust_roadmap_20261024_preparation_01/preparation_manifest.json),
[materialized prospective schedule](../../outputs/splendorust_roadmap_20261024_preparation_01/prospective_schedule.json)
and [bounded historical seed check](../../outputs/splendorust_roadmap_20261024_preparation_01/seed_collision_check.json).
The check inspected 516 discovered JSON manifests/schedules, extracted 86,309
seed/master integer fields, and found no collision with the 103 proposed initial
worlds. It records 24 oversized or unavailable/unsupported omissions and is not
an exhaustive history scan. Repeat admission against subsequently registered
schedules before execution. These preparation artifacts contain no new games
or model outputs and are not an executable source freeze.

## Question and scope

Does the already qualified, inexpensive PUCT16 player gain useful playing
strength over choosing directly from the same imported Entity network, under
the same one-second whole-move ceiling?

The comparison holds the checkpoint, corrected rules, observation restrictions,
public action filtering and auxiliary decisions fixed. It compares the existing
greedy policy with the existing search policy. Search also uses the network's
value output; therefore this does not separately identify the effects of tree
search, leaf values or their interaction. It does not compare neural
architectures, training amounts, our three historical proposal fits or PUCT6400.
It does not establish strength against expert humans.

There is one imported checkpoint, with its historical training origin disclosed
and no retraining. The principal endpoint is PUCT16's score against the greedy
player in **200 fresh games, 100 swapped-seat pairs**. This is a useful comparison
for choosing the next research investment, not automatic promotion or proof of
an independently stronger general-purpose player.

## Exact players and preserved dependencies

| Item | Retained identity |
| --- | --- |
| Upstream source | `1828507b8d2f0e02bc03d102b9c6d29a54fa8d57` |
| Entity checkpoint SHA256 | `ef8a4521cd6c03c7075f15efee23f4cde6ec94d1b5398765cf0c24a09ad745cb` |
| Corrected ruleset | `splendor_base_qualified_rules_20261018_v1` |
| Catalogue SHA256 | `db4776918e612eee5a6f2617719c85ec5f413d592bbab0a64058babc83a70312` |
| Turn contract | `atomic_prereveal_completed_turn_v1` |
| Search player | `entity_onehot_puct16_canonical_atomic_prereveal_cpu_v1` |
| Search profile SHA256 | `cd915908df1a7c6f39e55cfc9df59d88002a2b5ce167ce9512803e7c63d64424` |
| Search native executable SHA256 | `b15322f1c6aac1034638bdc52d91524639c63cbc2b61d361a9fef79dfdea728c` |
| Greedy scientific policy | `entity_onehot_greedy_canonical_atomic_prereveal_cpu_20261023_v1` |
| Greedy native executable SHA256 | `843ef76a157537dfe93bcb215dedd0113995abf2127098b2847bceee8dc352c4` |
| Proposed greedy deployment wrapper | `entity_onehot_greedy_canonical_atomic_prereveal_cpu_1s_20261024_v1` |

The search profile and its dependencies are recorded in the
[existing profile](../../outputs/external_strength_screen_20261018_02/opponent_profile.json).
Retain 16 simulations, depth 64, world pool three, batch one, cpuct 0.4,
dynamic FPU reduction 0.02965, zero uniform prior and no noise. The configured
chance-universes setting remains inactive on the canonical search path. Retain
the qualified native binary and host float32 inference arithmetic.

Use the explicit
[greedy operation](../../tools/move_evaluation_20261023/native_bridge/src/main.rs),
[greedy adapter](../../tools/move_evaluation_20261023/greedy_adapter.py) and
[authenticated build](../../outputs/move_evaluation_20261023_build_01/build_receipt.json)
as the scientific reference for the new timed wrapper. On an ordinary Main
decision with more than one retained choice and turn below 124, it performs
exactly one neural evaluation, maps the 81 outputs into the canonical action
vocabulary, and selects the **first maximum** among the retained legal actions
in native order. The value output may be computed by the shared forward pass
but is ignored in choosing the move. It performs zero search simulations.

Do **not** implement this by calling ordinary PUCT selection with a zero quota.
That path has different tie and branch behavior and is not the qualified greedy
policy. A determinization used to validate lawful native legal support is not
a searched continuation and must not be misreported as one.

Both players retain the upstream public safe-choice filter and the same
qualified atomic bridge for payment, returns and nobles. They retain their
declared single-choice, forced-pass and turn-124-or-later heuristic branches;
these exceptions are recorded rather than described as neural decisions.
Greedy can recognize one retained choice without inference even where the
search profile still searches an originally larger Main list. The study keeps
these existing conventions explicit; it is not a pure change to one integer.

Future implementation belongs in new versioned files. Do not edit the existing
profiles, binaries, model definitions, helpers or frozen evidence to make this
comparison convenient. Bind every consumed dependency, the new wrapper,
supervisor, independent verifier, fixtures and this protocol in the execution
freeze. Check those bindings before and after execution. Preserve all previous
results, partial traces, timeouts and failed attempts. The stopped A64/B64 study
is not resumed or extended, and the PUCT6400 reference remains unchanged.

## Clocks, runtime state and resources

Each arm receives the same externally enforced **1.000-second complete-move
deadline** and **0.900-second internal fail-closed limit**. The trusted parent
starts before sending the lawful observation. Serialization, transport,
inference, Main/auxiliary decisions, required per-decision evidence generation
and complete response delivery are charged. Later parent validation does not
alter a packet's trusted arrival time. Reject unsolicited, malformed,
wrong-request, partial and late responses. No partial-search fallback is allowed.

Use a persistent model/transport worker for each game, as in the earlier
one-second screen. Initialize it within a separate 60-second limit, with only
the declared architecture warm-up and no game observation. Initialization is
outside a move's clock but inside all stage/resource clocks. Freeze and report
the exact shared-versus-separate model loading arrangement before execution.
Only one decision may run at a time, on one assigned performance-core logical
thread with one numerical thread; all descendants must be supervised.

Persistence saves model startup, not player search state. Construct a fresh
native agent for each phase/request, retain the existing observation-derived
seed, and carry no search tree, game history, cycle-escape memory, mutable RNG
or position-dependent inference cache across moves. The same input must produce
the same scientific response after intervening requests. Reusing immutable
weights and architecture constants is permitted. Close and recreate the owned
worker tree at each new game. Do not introduce cross-game warm state.

The proposed execution envelope is **six hours total** from one monotonic and
UTC origin established before startup admission. Both clocks enforce the
limits. Each stage also has its own cap measured from its recorded start;
early completion does not enlarge a later stage's allowance:

| Stage | Maximum wall time, including its checks |
| --- | ---: |
| Actual entry-point qualification and timing | 1 hour |
| Six operational games and their gate | 30 minutes |
| Fixed 200-game collection and online checks | 4 hours |
| Final replay, source verification and independent analysis | 30 minutes |

Collection must also end by absolute hour 5.5, leaving the final 30 minutes.
Every stage start/end, initialization and request is clamped to its stage and
global deadlines. Time spent waiting for RAM or writing evidence counts.
No stage retry, clock reset, hidden pause, deadline extension or transfer of
unused stage allowance is permitted.

Use at most **2 GiB combined worker/helper RSS**. Require **2 GiB available
physical RAM stable for 60 seconds** before initial admission and at least
2 GiB at later between-request admission checks. A shortage may wait only
between requests, for at most 10 minutes cumulatively across the study and
within the existing clocks. Never suspend an active move or reset its budget.
Keep an owning process-tree watchdog and preserve every resource failure.
Report sampled memory/CPU telemetry as sampled, not a guaranteed continuous peak.

Cap all new study evidence at **512 MiB**, including workers, diagnostics and
final analysis. Reserve **64 MiB** for final verification; collection cannot
consume that reserve. Require **7 GiB free disk at launch** and maintain **5 GiB**.
No cleanup of frozen evidence is authorized to satisfy these limits. Do not
retain every internal PUCT edge; retain the actual-game and diagnostic evidence
specified below. Disk/RSS/evidence breaches stop dependent work.

## Stage 1: actual correctness and timing qualification

Use the unchanged 23 roots in the
[corrected fixture panel](../../outputs/external_strength_screen_20261018_02/fixtures.json):
18 core positions and five edge cases. Their historical three-fit grouping is
fixture provenance only; neither Entity arm has three model fits. Include the
existing lawful hidden-state variant for each root. Both variants must produce
the same public/own-private observation while the hidden referee state differs.

Run two passes over both arms and both variants: **23 x 2 x 2 x 2 = 184 timed
requests**. In pass zero visit roots in manifest order, search then greedy at
even indices and greedy then search at odd indices, original before variant.
In pass one reverse root order, arm order and variant order. Retain all outputs
and clocks. This also tests repeated inputs after intervening positions in the
persistent runtime. The pass schedule is fixed before execution.

Every timed response must be complete within **0.700 seconds**, legal and
independently transition-correct. This is the prospective engineering margin;
it is stricter than the actual one-second game deadline. A margin failure stops
the comparison without selecting a new quota or replacing a fixture.

The new collector/evaluator and verifier workers must call
`verify_runtime_sources()` once at startup and save its receipt. Integrate the
[corrected-rules requirements](future_training_rules_20261021.md) through the
actual entry point: snapshot decoding, explicit legal support, native catalogue
and determinization, model encoding, returned action, corrected successor,
saved record decoding, replay and official termination. Qualify both arms using
the six corrected faces, non-purchase noble award, multiple eligible nobles,
noble-triggered final round, gold/payment/return and private-reserve cases.
Include forced/single choices, late heuristic behavior and final-round edges.
Reject old/missing identities, mismatched hashes and historical reconstruction.
The actual
[greedy entry-point fixtures](../../tools/move_evaluation_20261023/qualification.py)
provide reusable counterexamples, not blanket qualification of a new wrapper.
No training loader is certified because this study performs no fitting.

In addition to the 184 scheduled requests, bounded startup/model-free and
actual-entry-point counterexamples belong inside the same one-hour stage.
Their exact fixed manifest and counts must be specified in the implementation
freeze, not chosen after a failure. Check the full 90-card/10-noble identity,
all-action mappings, both starting seats, pre-reveal information restriction,
tie handling, malformed input and supervisor deadline/error classification.
Native-versus-host parity may reuse the existing authenticated receipt only
when every consumed model/inference dependency is byte-identical; otherwise
stop for a separate qualification design rather than silently accepting drift.

Require the following scientific checks:

- PUCT16 reproduces the existing saved fixture decisions and scientific work
  records. A new envelope may change only declared transport metadata and clocks.
- Greedy exactly matches the explicit frozen greedy operation; normal Main
  requests have zero simulations and one forward evaluation. Independently
  reconstruct its legal canonical indices and first maximum from saved logits.
- PUCT16 reports exactly 16 simulations for its searched Main branch; known
  heuristic/forced branches follow their own declared counts. Neural call count
  need not equal simulation count.
- Hidden-state variants and the repeated passes reproduce each arm's scientific
  outputs, including action, relevant work counts and greedy logits. No referee
  seed, actual deck order or opponent blind-reserve identity reaches a player.
- Auxiliary decisions match the authenticated helper on identical intermediate
  observations and retain the atomic pre-reveal contract in both arms.
- Every resulting complete turn passes the independent dictionary ledger,
  including payment, token totals, noble award, seat order and terminal state.

Existing native search internals remain a source/numerical-parity trust boundary;
these checks are not exhaustive verification of every hypothetical PUCT edge.
Any unexplained discrepancy blocks acceptance. Preserve it; do not patch and
retry within the same frozen attempt.

## Stage 2: six operational games and the cost gate

Only after qualification passes, run exactly **three fresh swapped-seat pairs**,
six games, with the same players, clocks and referee. These games are excluded
from the primary endpoint. Require all six to complete and replay successfully,
with no timeout, illegal response, source/privacy fault or infrastructure failure.
Require every response to remain within the 0.700-second engineering margin
before advancing. A slower but timely response is still a legal game move,
while failing this study's operational gate.

Keep aggregate operational win results closed for the gate. Use only timing,
bytes, counts and check status. Project 200-game collection wall time from the
six games' complete collection time, including initialization, referee work,
serialization and online checking:

`projected_collection_seconds = observed_six_game_collection_seconds * (200 / 6) * 1.35`.

Likewise project variable evidence bytes with the same multiplier and add all
already written fixed evidence. Both must fit the remaining four-hour collection
allowance, the absolute collection cutoff and the 448 MiB pre-finalization cap.
Use operational independent transition checks to estimate final all-game replay
cost with the same factor; add the fixed source/statistical/replay-sample work
and require it to fit the final 30 minutes. Freeze this arithmetic and its fixed
overhead measurement procedure before games. Do not assume replay is free.

A failed or undefined projection stops the study. Do not reduce 200 to 60, alter
the player, omit verification or extend time after observing the rehearsal.

## Stage 3: fixed fresh paired comparison

Run **100 predetermined initial worlds, each with seats swapped**, giving 200
games. PUCT16 is the scored arm. On pair index `i`, run its seat `i mod 2` first,
then the opposite seat. Each arm therefore plays exactly 100 games in each seat.
No operational, historical or qualification game enters the primary result.

Derive each referee seed as the unsigned big-endian integer represented by the
full SHA256 digest of UTF-8 canonical JSON:

`["splendorust_policy_puct16_1s_20261024_v1", stage, pair_index]`.

Use JSON separators `,` and `:`, no whitespace, ASCII string escaping, stage
`"operational"` for indices 0..2 and `"main"` for indices 0..99. The paired games
reuse the initial world's seed; only arm-to-seat assignment swaps. Freeze the
materialized schedule before outcome access. Referee seeds remain parent-only;
each player's ordinary seed stays the existing lawful-observation digest rule.

Before freezing, compare these seeds against the prior screens, operational and
matched benchmarks, and every discovered registered schedule/manifest up to
8 MiB. Record exact checked files, hashes and any oversized/unavailable omissions.
Do not claim an exhaustive historical-log scan. Any discovered collision stops
preparation for an explicitly revised namespace; no after-outcome resampling.

During the comparison, only **1.000 seconds** determines a deadline loss. The
earlier 0.700-second engineering margin does not turn a timely game response into
a loss. Retain all observed latencies and every failure. Apply the same frozen
classification to both arms:

- An attributable timeout, illegal action, malformed player response or player
  failure to return a complete move is a forfeit loss, with no replacement game.
  Retain its request, full/partial response, trusted arrival time and process
  evidence. The opponent receives the win without invented continuation moves.
- An initialization failure, referee defect, changed source, resource breach,
  supervisor fault or unresolved interruption makes the study incomplete and
  stops dependent work. Do not misclassify these as playing-strength losses.
- An unresolved blocked-turn/deadlock or the fixed 2,000-ply safety limit is not
  an official terminal result. Preserve it and stop incomplete; do not invent
  a score winner or draw. Internal native-search empty-action conventions remain
  disclosed and unchanged, and are not actual-game adjudication rules.

Natural draws receive half credit under the corrected end-of-round rules.
Never reset an interrupted move's allowance. There is no automatic recovery,
second attempt, replacement seed, partial result promotion or sample extension.

## Evidence and independent verification

Apply the [verification policy](verification_policy.md) to these exact players.
Save every actual observation digest, legal/action mapping, selected complete
move, phase branch, actual simulation/inference counts, trusted request/response
clock, corrected pre/post-state record and official result. Keep game seeds in
parent-only evidence. Greedy records include enough logits and retained-index
information to independently verify every ordinary Main argmax. Keep the exact
model/profile and executable identities with each worker.

Independently replay **every actual move in all six operational and 200 main
games**, including auxiliary choices, legal support, scores, nobles, final
round, terminal winner and each forfeit classification. Verify the full fixed
schedule and both seats. Check all greedy argmax records and PUCT branch/quota
receipts. Source hashes and native work counters alone do not prove every inner
search transition correct; retain that limitation in the result.

For main-game repeat execution, prospectively select one actual completed
decision per arm x phase x acting seat x private-reserve category, at most
**48 distinct decisions**. Phases are completed plies 0..19, 20..39 and 40+;
privacy categories are neither/own/opponent/both hidden reserves. Choose the
smallest SHA256 of canonical JSON
`["splendorust_policy_puct16_1s_20261024_audit_v1", pair_index, game_index, ply, arm]`
within each occupied cell, using identity and lawful metadata only. Report empty
cells without adding games. Materialize the sample manifest before model replay.
Reexecute those complete decisions in bounded initialized workers, requiring
scientific action/work/greedy-logit equality; repeat executions are audit work,
not extra games or independent strength observations. Do not describe these
root reproductions as a stored trace audit of every internal PUCT edge.

Replay mismatch stops acceptance. Keep the original evidence and diagnose only
within the remaining audit bound; no replacement sample, silent correction or
new player version is accepted in the same study. Independent analysis runs on
sealed completed evidence and recomputes pair arithmetic separately. Verify all
consumed frozen dependencies unchanged and record process-tree cleanup before
final acceptance.

## Primary statistic and interpretation

For PUCT16, score a win 1, draw 0.5 and loss 0. For each pair average its two game
scores. The primary estimate is the equal mean over all 100 pair scores, compared
with 0.50. An equivalent exact representation uses game halfpoints 2/1/0,
pair numerators 0..4 and overall denominator 400. There is no local-fit weighting.

Use 50,000 pair-bootstrap replicates with NumPy PCG64 seed `2026102401`.
For each replicate, in replicate-major order draw 100 int64 indices uniformly
from 0..99 with replacement, then average those original pair scores. Report the
2.5th and 97.5th percentiles using linear interpolation. Freeze NumPy/runtime
versions and verify the index stream and integer arithmetic independently.
Use pairs, not individual turns or unpaired games, as the resampling unit.

The predeclared positive decision is **useful evidence for cheap search** only
when all evidence gates pass, PUCT16's score is at least **55%**, the paired 95%
lower bound is above **50%**, and each seat's descriptive score is at least 50%.
Report both seat results, all forfeits, natural-game lengths and latency/CPU/
memory distributions. The conclusion concerns this exact paired matchup and
allowance; it is not an Elo estimate, promotion or proof of general strength.

The symmetric criterion for useful evidence favoring greedy is PUCT16 score
at most **45%**, paired 95% upper bound below **50%**, and each PUCT16 seat score
at most 50%. These are two interpretations of the same prespecified statistic
and interval. Otherwise the result can be inconclusive or small; it does not
prove equivalence or establish that search is useless. No outcomes select a
new quota, stopping time, sample size or model.

A degenerate empirical bootstrap is not zero population uncertainty. Report,
as a sensitivity bound for all results, the clipped bounded-pair Hoeffding
interval `mean +/- sqrt(log(40) / (2 * 100))`. If the empirical interval is
degenerate, require its conservative lower bound also to exceed 50% for search
evidence, or its conservative upper bound to be below 50% for greedy evidence.
This additional display is not a different population or extra observations.

The choice of 200 games improves on a 60-game screen at little model-thinking
cost: near 50%, an unpaired independent-game approximation has roughly a
7-percentage-point 95% half-width instead of 13 points. Actual pair dependence
can change that width; 200 games does not guarantee power for a five-point gain.
Report the realized uncertainty, and keep any later confirmation separately
designed and fresh.

## Cost rationale and next decision

The accepted earlier PUCT16-versus-raw035 screen completed 60 games/3,116 plies
in 33m29s, and qualification plus screen took 42m44s. It used a persistent worker
per game with fresh native decisions. PUCT16 averaged 226 ms per move, the other
player 141 ms; there were 61 worker initializations, averaging 2.878 seconds.
These are historical measurements of a different opponent, not a promise for
greedy Entity. See the [accepted screen report](external_strength_screen_20261018_02_result.md).

Scaling that complete screen time by 200/60 gives about **1h52** before allowing
for different game lengths, greedy cost and the new checks. A planning range of
roughly **2-4 hours for main collection** is reasonable; the operational forecast
decides feasibility before opening main outcomes. Six rehearsal games scale to
about 3m21s under the old conditions, leaving room within their 30-minute cap.
The 184-request panel has little ordinary inference cost; source/counterexample
checks are included in its one-hour limit. The hard overall bound remains six
hours, including final verification, and may stop the study incomplete.

If cheap search clearly helps, retain PUCT16 as the economical research/deployment
reference and investigate one justified search-efficiency or learning question.
If the result is unresolved or favors greedy, inspect the saved disagreements
and measured costs before choosing the next intervention. Do not automatically
launch the deferred PUCT10240 study, alter hidden-world treatment, enable tree
reuse, fine-tune the model, or collect a larger training corpus. PUCT6400 remains
the verified 120-second research reference; this study does not compare its
strength with PUCT16 or greedy. Any new learning route requires its own
corrected-rules end-to-end qualification and a fresh strength test; this
comparison qualifies no general training loader.

