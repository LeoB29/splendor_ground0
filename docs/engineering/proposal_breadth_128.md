# Fixed breadth versus matched precision

Owner authorized execution on 2026-09-27 after accepting
`docs/strategy/next_after_proposal_ranking_128.md`. This is a new experiment;
all predecessor freezes, dirty files and recovery evidence remain unchanged.
Entry: `run_proposal_breadth_128.py`; code: `tools/proposal_breadth_128/`;
output: `outputs/proposal_breadth_128/`.

## Question and arms

Does a fixed, broader proposal set improve decisions more than spending the
same additional continuation count evaluating the existing candidates?

- Anchor: unchanged original035 top two, original first random alternative,
  and the unchanged original161 learned fourth; up to four candidates at128 worlds.
- Breadth: preserve those candidates and representatives, append up to two
  highest anchor-ranked remaining canonical groups, then up to two uniformly
  sampled remaining groups without replacement. Evaluate all at128 worlds.
- Precision: original anchor candidates with exactly the same total continuation
  count as breadth. No new proposal or model training.

For G lawful canonical groups, m=min(G,4), n=min(G,8), precision worlds=128*n/m.
This is integral:128 for G<=4,160/192/224 for G5/6/7,256 for G>=8.
Breadth and precision each use128*n continuations. At G<=4 all arms coincide;
such positions remain in the analysis. Ranked additions precede random additions.
Stable original group order resolves rank ties; original incumbents precede
additions in selection ties. Canonical representatives preserve the original
four exactly. Random additions use hash order over fixed stage, root index and
canonical key with the new random tag. There is no outcome-dependent candidate
selection, alternate mixture, fit selection, subgroup gate or extension.

The corresponding original161 models are frozen at updates563/488/563 for
fits161001/161002/161003. Checkpoint hashes and completed ranking/recovery
provenance are verified by models.py. The unchanged035 continuation/proposal
checkpoint is separately checked against its historical hash and bound as an
execution dependency. Only these three anchor models are loaded as proposers;
neither the new ranking models nor matched CE models are used. Existing model
registries are not mutated; historical training is not replayed.

## Fresh source population

Both source actors greedily use the corresponding unchanged restricted161 model.
Main: fixed1,080 source games, prospectively assigned60 to each of18 fit x acting
seat x phase cells. Probe: separate54 games,3 per cell. Phase bins are turns0-19,
20-39,40+. Select20 main games per cell (360 positions) and one probe game per
cell (18 positions), with one identity-selected eligible turn per distinct game.
Assignments, eligible-game priority and turn priority use the frozen source tags.
No disagreement filtering, short-support exclusion or private-reserve exclusion.
Insufficient eligible source games causes a hold; never extend the fixed pool.
Every source game and actual action is independently checked; every retained
source position is replayed to its saved state. No prior inspected positions
are reused as fresh evidence.

## Worlds, locks and outcomes

The common selection block has128 worlds for all n breadth candidates. The
extension block has P-128 additional worlds for the original m candidates only.
All three choices are derived from those blocks and saved in an immutable
selection artifact before any assessment begins. Precision uses the correctly
weighted sum across both blocks, including ragged32/64/96 extensions.

Assessment uses128 fresh disjoint worlds for only the distinct locked choices
(one to three). Shared worlds allow paired comparison. No assessment maximum
is selected. Mature observer-legal determinization, serial greedy035 terminal
continuations, WDL scores and end-condition guards remain unchanged. Scores are
win1/draw0.5/loss0, on an exact1/256 grid. Artificial termination invalidates a
root. Full outcomes, candidates, inputs, model hashes and block timings are saved.

World base2100000000, stage offsets probe0/main1000000/test2000000, root stride1024.
Common offsets0..127, extension128..255 (required prefix), assessment512..639.
Source bases main2105000000/probe2106000000/test2108000000. Bootstrap2103000000;
audit2104000000. Distinct namespaces and fixed schedule are checked before freeze.
Test fixtures cannot target the production directory.

## Fixed analysis and investment gate

Primary contrast: breadth minus equal-budget precision. Required practical
contrast: breadth minus anchor. Secondary contrast: precision minus anchor.
All360 source games receive equal weight. Retain all three fits and all cells.
Use50,000 paired bootstrap resamples within the18 fixed cells,20 positions each;
seed2103000000,50 chunks of1,000, cell order fit/seat/phase and common draws for
all three contrasts. Percentile95% intervals use linear interpolation. Fits and
worlds are not additional independent games; inference conditions on these
three fits and the fixed fast-source population.

Breadth passes only if breadth-anchor point gain>=0.005 and its95% lower endpoint
is strictly positive, and breadth-precision95% lower endpoint is strictly positive.
Both independent statistical implementations must agree within1e-12, including
every gate. The root implementation uses indexed integer sums; the verifier uses
histogram-count multiplication and an independently implemented quantile.
Classify as positive_conditional_breadth_contribution only when all gates pass,
otherwise practical_progress_not_established. No post-hoc precision promotion.

Report all three contrasts, descriptive changed choices, chosen original fourth,
ranked/random additions, realized arm budgets and cell summaries. These are
conditional chosen-move scores under greedy035 continuations, not arena win
rates or established broader/superhuman strength. A pass warrants a separate
full-game design; it does not authorize that run or training automatically.

## Verification and trusted boundary

The accepted `verification_policy.md` applies. Separate collection and verification
logic reconstruct proposals, legal groups/representatives, observations/features,
private-input invariance, world law, block budgets, tie order, sparse assessment,
all recorded outcome aggregates and the persisted decision boundary. Mature
environment, encoding/neural arithmetic and terminal rollout primitives are shared.

Fully replay all18 probe roots. Main ordinary replay:8 of128 selection worlds for
every candidate, ceil(extension_length/16) extension worlds for the original
candidates, and8 assessment worlds for every actually assessed choice. Indices
are identity-hash sampled with the frozen audit tag/seed before collection.
Each selected world is replayed in full, across all its relevant actions.

Main full stress roots are selected without outcomes, by minimum identity hash in
18 fit/seat/phase cells,12 fit/privacy-presence cells,9 fit/support1/2/3 cells, and
9 fit/support5/6/7 cells. Deduplicate; maximum48 roots, report empty cells without
substitution. Thus every root has structural/input/aggregate checks plus sampled
replay, and stress roots have full replay. Sample manifests and seals precede
collection. Per-block worlds/continuations and exact coverage are reconciled.
Main is explicitly not a full replay of every mature continuation.

Focused preflight covers original-anchor preservation, stable mixtures/ties,
supports1 through8+, empty/ragged precision extensions, exact equal budgets,
all-fit privacy invariance, real source/engine/verifier integration, corrupt
receipts/blocks/seals, independent statistics, runtime/storage holds, finalization
and actual hidden worker initialization. Current passing check artifacts, all
new code, inherited dependencies, protocol, strategy and runtime versions are
bound into a one-shot freeze before production.

## Execution and resource limits

Eight hidden CPU workers, one deterministic Torch thread each. No GPU, fitting,
batching/backend changes, scheduled monitoring, automatic retry or successor.
The detached hidden supervisor survives closure of this conversation. Scoped
system sleep inhibition lasts only while it runs; display sleep remains allowed.
Use durable best-effort StatusReporter from the completed reporting recovery:
immutable status events plus advisory progress. Reporting I/O failures remain
nonfatal; scientific artifact writes/integrity failures remain fatal and preserved.

Main continuation ceiling691,200:368,640 common,184,320 extension,138,240
assessment. Probe ceiling34,560. Source games and verification are additional.
Equal continuation counts are not equal wall times. Save actual shared block
collection/replay timing and realized arm budgets; do not claim independently
measured arm latency from shared execution.

Eight-hour execution cap begins at the new launch (preflight is additional),
including model sealing, source generation/audit, probe, main, verification and
finalization. Five-GiB disk reserve. Boundaries check time/storage throughout
source work and between evaluation/replay worlds; an in-progress world can finish
before the next check. Partial artifacts and failures are never erased.

After the fully replayed18-root probe, before main, derive separate observed
seconds/continuation for selection, extension and assessment, for both collection
and replay. Use the already fixed main supports and replay plan to compute exact
selection/extension counts and worst-case distinct assessment choices (up to3).
Scale structural work by360/18, divide work by8 workers, add35% runtime allowance
and300 seconds for finalization. Include elapsed launch time. Storage estimate:
probe bytes per recorded continuation times planned main count times1.5 plus512MiB;
retain5GiB free afterward. Unmeasured required probe blocks cause a hold, not a
guessed rate. Forecast failure causes a hold without running main. This gate uses
timing/size/counts only, never scientific effects. Preliminary expectation4-6h is
unmeasured; the probe supplies the first design-specific estimate.

Finalization validates frozen inputs, linked stages, source/model provenance,
all prescribed receipts, and separate raw-outcome versus receipt statistics.
Result/report precede final_status; a late budget/integrity failure cannot produce
a false completion. No automatic arena, distillation, extension or subsequent
experiment follows either a positive or a null result.
