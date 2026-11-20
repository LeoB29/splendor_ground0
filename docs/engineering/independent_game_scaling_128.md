# Independent-game training scale

Owner authorized on 2026-10-12 after the completed teacher-precision experiment.
One study is authorized: nested independent-game teaching sets, nine matched
fits, independent reconstruction, and prospective planning assessment. No arena,
promotion, automatic parameter rescue, successor or recurring monitor follows.
Implementation and outputs use independent_game_scaling_128 only. Preserve all
frozen predecessors, evidence and intentionally dirty work.

## Scientific question

Does substantially broader independent-game experience improve transfer of
learned proposals to new positions? The previous 180-versus720 added-position
test was negative. The 128-to512-world precision test was also negative despite
strong acquisition of the added targets. Saved-data checks found independently
reassessed improvement on seen positions, but no descriptive fresh-state rescue
from restricting proposals to recorded support. These motivate this hypothesis;
they do not establish data shortage or sufficient representation.

The old 9,461 TRAIN rows already have broad-teacher labels. They come from159
source pairs/318 games. This experiment increases independent-game coverage,
not merely the total row count or the first availability of broad comparisons.
The flat policy's omission of purchased-card identities remains an unmeasured
structural rival. Nothing here changes that representation.

## Corpus and nested intervention

Keep the authenticated original9,461 TRAIN rows, original three161 initializers
and completed short-dose R1 BROAD controls. Keep the2,489 old validation rows
outside fitting and selection. Reuse the720 previously retired added roots with
their ORIGINAL128-world selection labels, not their later512-world labels or
assessment outcomes. Inherit closed conversion and simulation audits after
authentication; do not replay historical trajectories or retrain old controls.

Generate32,400 new fast161 restricted-self-play source games, balanced across18
source-fit by assigned acting-seat by phase cells,1,800 games per cell. Use
exactly the declared source population and original anchors. Fully audit every
actual source decision, lawful model input, transition and terminal result.
Select600 games per cell by fixed identity priority among those with an eligible
turn, and one eligible turn per game by identity. Phases are0-19,20-39,40+.
Retain private reservations and small supports. A fixed-pool shortage holds;
do not extend or replace games based on outcomes, runtime or difficulty.

This creates10,800 new positions, each from a different game. Establish source
ancestry/seed disjointness from all inherited TRAIN/VAL and new evaluation
sources. The existing720 core has40 roots per18-cell. Nested added datasets are:

- small: the original720 core;
- medium: the core plus120 identity-selected new roots per cell,2,880 total;
- large: the core plus all600 new roots per cell,11,520 total.

Nested identities are fixed without reading outcomes. Source outcomes never
choose inclusion or training weights. Training pools all three source fits.

At every new root, use the original four-candidate anchor recipe, then two
ranked and two identity-uniform additional canonical actions, up to eight.
Keep concrete representatives, canonical maximum pooling and candidate tie
order. Record128 observer-legal common worlds per candidate with unchanged
greedy035 terminal continuations. Save the128-world block with a separate
completion hash receipt. An unresolved training outcome is independently
replayed and causes a hold; never impute it as a draw or omit that row.

The target is softmax(expected terminal score/.05) over recorded candidates
outside the actual first three. Empty/singleton outsider sets contribute a
differentiable zero with unchanged batch denominator and optimizer semantics.
Check every exported tensor, legal group, first-three exclusion and score sum
independently. Assessment outcomes never enter the training tensor.

## Nine fixed fits

Start small/medium/large from each original161001/161002/161003 model. Train all
policy/shared parameters and freeze value-head tensors. Deterministic CPU
float32, one Torch thread per process; fresh AdamW lr1e-4, weight decay1e-4,
gradient norm clipping1. Architecture, objective and teacher remain unchanged.

All nine models receive7,136 updates: repeat the inherited1,784 batch-length
schedule four times (7,088 batches of64 and48 of53). Exactly456,176 row exposures,
228,088 old and228,088 added, per model. Alternate odd-batch extra slots to retain
50/50 totals. Match old-row and added-cell streams across sizes. Cycle through
each arm's nested cell pools with deterministic shuffles; use a shared padded
length bucketing schedule based on the maximum of matched arm input lengths.
Never outcome-weight or skip a batch. Largest rows average about19.8 exposures;
this is not a guarantee of adequate acquisition.

Always select update7,136. Save durable optimizer/model/schedule progress every64
updates plus required initial/final/diagnostic endpoints. Record full exposed
TRAIN acquisition initially/finally, separately for old and added pools. Fixed
cheap diagnostic curves at0/1784/3568/5352/7136 use256 identity-selected old rows
and up to eight added rows per cell in each arm. No checkpoint, size, fit, phase
or target revision is selected using these reports. A low-acquisition null
limits the conclusion; it never triggers automatic extra training.

Independently reconstruct all nine complete optimizer trajectories, diagnostic
reports and saved endpoints from the same initial weights. Verify frozen value
heads and exact exposure/stream matching. Seal all nine new models, three
original anchors and three completed short-dose BROAD controls before generating
any evaluation sources. The earlier source-model seal contains only anchors and
authorizes TRAIN source generation, never evaluation before learning completes.

## Prospective assessment

Generate2,160 new main source games and54 separate probe games under the same
original fast161 population. Identity-select720 main positions from720 distinct
games,40 per18-cell, and18 probe positions. No protected historical panel serves
as the primary endpoint. All actual source games and model inputs are audited.

Six arms are anchor, small, medium, large, control (saved short-dose BROAD), and
teacher. Each learner supplies one full-legal fourth proposal while retaining
the common first three. Teacher retains its eight-candidate recipe. Each root
uses the source-matched fit; three initializers do not triple sample size.

Evaluate the deduplicated union, at most12 candidates, on128 common lawful
selection worlds. Preserve each arm's candidate order for ties. Durably seal
all choices before128 disjoint assessment worlds for at most six distinct
chosen actions. Unknown assessment outcomes stay bounded; identical choices
cancel exactly. No checkpoint or proposal selection may inspect assessment.

Use50,000 paired bootstrap draws within the18 fixed cells, resampling whole
source games and retaining all arms together. Independently reconstruct means,
confidence bounds and decisions. Joint primary investment gate:

- large minus anchor mean lower bound >=+.005 score (+.5 percentage points),
  with paired95% lower confidence endpoint strictly above zero;
- large minus small-long-dose paired95% lower confidence endpoint above zero;
- all prescribed integrity and operational checks pass.

The medium and saved short-dose control comparisons are descriptive. They cannot
replace a failed largest-size endpoint. The short-dose control helps expose
degradation caused by prolonged small-data training. Report all fits without
selecting a winner. Positive conditional planning evidence earns a separately
proposed gameplay comparison; it is not an arena winrate or superhuman claim.

## Namespaces and verification

Exact seeds/tags are frozen in common.CONFIG and the manifest. Source starts:
train2405000000, probe2406000000, main2407000000, fixture2408000000. Evaluation
world base2400000000 uses separate probe/main/fixture ranges and512-seed root
strides; assessment offset256. New teaching world base2420000000 reserves1024
seed IDs per root. Bootstrap2403000000; audit2404000000. Freeze an interval
disjointness proof and source-ancestry proof before launch.

Apply docs/engineering/verification_policy.md: full source/decision/target
checks, every recorded continuation structurally checked, full18-root probes,
ordinary8/128 whole-world replay plus metadata-selected stress roots (bounded39
per stage), every unresolved branch independently replayed. Freeze identity-
and metadata-only sampling, audit domain tags and failure response in code.
Main continuation replay is not exhaustive. New simulation semantics are not
introduced; use bounded full-trace fixtures for privacy, refills, repetition,
ties, short supports and real hidden spawned-worker execution.

## Execution, forecasts and preservation

One hidden supervisor, at most eight one-thread source/simulation workers or
three fitting workers. CPU only. Inhibit system sleep and allow display sleep.
The process continues independently of chat activity. No periodic monitor.

Before full TRAIN source generation, collect/audit a fixed54-game source probe
(three games per cell, part of the32,400). Use observed times and bytes,35%
allowance, a120-hour downstream reserve and32-GiB downstream storage allowance
to check the bounded source workload. The reserve is conservative planning,
not a finish-time estimate. All probe games are reused.

Next,18 identity-selected TRAIN label roots are fully collected/audited and
reused within the10,800. Price remaining labels and prescribed replay from
their actual timing/counts and all block/tensor bytes, add35% allowance,24h
and32GiB for fitting, durable optimizer checkpoints, fresh sources, evaluation
and finalization. Before main evaluation,
reprice its remaining work using the fully audited18-position evaluation probe.
No forecast examines measured strength or selects an experimental variant.

Freeze a168-hour cumulative ACTIVE execution ceiling and5-GiB free-space floor.
The3-4 day estimate is provisional; forecasts may document a hold when the
frozen budget is inadequate. Every phase uses the same current resource guard,
including imported diagnostic helpers and independent audit paths. Test actual
hidden workers after96h elapsed and separately prove168h/abort/storage guards
still stop correctly. Hash-seal launch timing before any work. Explicit recovery
must preserve prior active execution and exclude stopped time; no timestamp reset.

Save completed game/root/audit receipts, label block completion receipts and
optimizer checkpoints durably. Stop cooperatively on new failure, preserve all
partial evidence and diagnose before a versioned explicit recovery. No automatic
retry, overwrite, runtime extension, replacement sample or successor. Report
provisional completed-position results when the owner asks for status.
