# More independent self play games

6 November 2026. **Prepared proposal only. No collection, fitting, evaluation,
capacity rental or cloud upload is authorized by this document.** The running
one-pass 400-game comparison continues under its existing freeze. Its partial
scores have not been used to choose this design.

## Recommendation and research question

Generate **1,152 fresh independent games: 1,024 for training and 128 held out**.
Use original EF8A at512 simulations as the fixed teacher. This increases training
game coverage5.33-fold over192 games without committing to a much larger campaign.
Expected local duration is **48-60 hours including collection, training, fresh
strength assessment and verification**, after implementation and qualification.

Test whether spreading a fixed number of training examples across more games
improves generalization. More data and more optimizer updates are different
changes: keeping both arms at3,072 examples and96 updates avoids interpreting a
longer training run as an effect of game diversity.

| Arm | Independent training games | Positions per game | Positions per fit | Updates per fit |
|---|---:|---:|---:|---:|
| Narrow control | 192 | 16 | 3,072 | 96 |
| Broad candidate | 1,024 | 3 | 3,072 | 96 |

The192-game control is a predetermined subset of the same new1,024-game pool, so
it needs no additional collection and avoids comparing an old collector/corpus
with a new one. Fit two fixed repetitions of each arm: four models in total.
This tests a breadth-versus-within-game-density tradeoff at fixed exposure. It
does not estimate the optimal dataset size or the effect of using every new root.
Keep the complete collected corpus for separately designed future work.

## Fixed player and training procedure

Original warm start and teacher checkpoint:
`outputs/external_opponent_search_20261018_02/splendorust_model.pt`, SHA256
`ef8a4521cd6c03c7075f15efee23f4cde6ec94d1b5398765cf0c24a09ad745cb`.
Keep the4,780,883-parameter architecture, float32, three hidden worlds, cpuct0.4,
depth64 and accepted search/target construction unchanged. Both self-play seats
use the same frozen512-simulation teacher; the teacher never learns during
collection. CPU8 with fixed16-game queues and refill is the local default.

All four fits start from pristine EF8A. Retain the accepted joint policy/value
loss and its coefficients, AdamW learning rate1e-5, weight decay1e-4, gradient
clipping1, batch32 and one96-update pass. Update the shared layers and both heads.
Bind the exact qualified trainer and loss implementation at execution freeze;
this is not an invitation to choose a new loss coefficient. There is no
hyperparameter sweep, early stopping or selection among intermediate checkpoints.
Save0/95/96 and reproduce95-to96 exactly for each fit, including optimizer/RNG,
outputs, losses and gradients. Two discarded qualification updates plus384
scientific and4 replay updates give a proposed maximum390 optimizer calls.

## Independent games and predetermined sampling

Reserve all seeds and schedules before collection in a new namespace, disjoint
from the running400-game study, prior training/held-out games and registered
engineering games. Record finite registry coverage and any exclusions. This
preparation fixes counts and allocation; concrete seeds and immutable source
hashes still require the future implementation freeze. Do not regenerate seeds
after seeing outcomes. The old64 engineering trajectories are not training data.

Use1,152 distinct setup/randomness seeds, one trajectory per seed. Balance which
seat starts within training and each held-out partition. Training and held-out
games have disjoint seeds and trajectories; swapped views of one setup do not
count as two independent training games. The128 held-out games comprise two
preassigned64-game partitions, validation and final assessment. Both remain
unopened until every final model is committed; neither selects a model or recipe.

Choose one192-game training subset before outcomes, shared across repetitions.
Each arm has exactly1,024 selected roots from each relative trajectory third.
The broad arm takes one root from each third per game. The narrow arm takes16
roots per game, rotating the6/5/5 allocation over64 games per extra-third group.
Choose roots by frozen hashes/seeds, without replacement within a fit. Repetition
seeds change the root selection and optimizer RNG only. Retain both repetitions;
they are not additional independent source games.

Divide the ordered usable roots into three contiguous thirds using a frozen
integer convention, then select within those thirds. Validate this convention on
the actual loader before collection. An insufficient root count, missing target,
non-natural ending or failed game is preserved and reported as incomplete; it
does not authorize dropping the game, duplicating rows or collecting a substitute.
All moves must still be played to obtain the official outcome, even though a fit
uses only three positions per broad game.

## Corrected rules and collection qualification

Every new collector, trainer and evaluator calls `verify_runtime_sources()` once
at startup and saves its receipt. Use the approved helpers, corrected decoder,
explicit legal lists, targets and independent transitions from
`docs/engineering/future_training_rules_20261021.md`.

Bind ruleset `splendor_base_qualified_rules_20261018_v1`, catalogue SHA256
`db4776918e612eee5a6f2617719c85ec5f413d592bbab0a64058babc83a70312`, and turn
contract `atomic_prereveal_completed_turn_v1` in source, corpus and checkpoint
metadata. Historical EF8A remains an explicitly historical warm start; old
targets must not be relabelled as corrected. Preserve the disclosed atomic
pre-reveal interface and internal deadlock conventions. Only natural official
outcomes are training value targets.

Implement a separate production collector using the qualified CPU8 mechanics;
the engineering benchmark is not automatically a production training entry
point. Qualify the actual collector, target capture, loader/collator and trainer
end to end on the six corrected faces, noble/non-purchase/final-round cases,
gold/returns, private reserves, value orientation and invalid identity/hash cases.
Retain the existing graph, native numerical, full-decision and legal-support
checks. Qualified production provenance, rather than an outer environment or a
manifest alone, is required before collection.

Run72 fixed cells of16 games. Retain full independently checked transitions,
genuine root-visit policy targets and actor-oriented official outcome labels.
Replay the first neural decision in each available ply band0-19,20-39,40+ for
each game with the prescribed shifted physical slot: at most3,456 collection
model reproductions. Independently verify every complete game and target, each
cell and the final corpus. No data is released to training before corpus acceptance.

Collection retains60-second internal/75-second complete-response safeguards.
One-second and old9/10-second observations are descriptive during concurrent
collection. Preserve a failure and stop subsequent releases; already released
requests keep their original clocks. No failed decision is retried, no game is
replaced, and an interrupted/failed game supplies no fabricated outcome.

## Independent prediction and playing-strength assessment

Commit all four fixed final checkpoints before opening held-out predictions.
Evaluate original EF8A and all four fits on every usable position in the128
held-out games. Evaluate selected training roots separately for diagnosis.
The primary offline contrast is the equal-repetition mean policy cross-entropy
of broad minus narrow, averaging positions within each game and then games.
Value mean-squared-error differences and each model versus EF8A are secondary.
Report both64-game partitions and the prospectively specified128-game pooled
contrast. Use20,000 paired whole-game bootstrap draws, stratified by partition,
with the same draws across models. Repetitions do not turn128 games into256.
These intervals concern the fixed fits and corpus, not arbitrary training seeds.

Prospectively nominate BOTH broad models for actual play, regardless of the sign
of offline loss changes. After correctness qualification and six operational
games per model, run **400 fresh paired games against original EF8A**,200 per
model, with128 simulations for both players. Keep the current main assessment
design:100 swapped-seat pairs per model, disjoint starting setups, alternating
ten-game blocks, fixed equal-average primary,50,000 stratified pair bootstrap,
at most96 exact selected reproductions per model and full independent audits.

Retain official AND conservative useful-gain criteria: mean score at least55%,
two-sided95% lower bound above50%, each pooled seat at least50%, and the existing
degenerate-interval safeguard. Conservative analysis treats every administrative
game as a candidate loss; official scoring assigns a classified hard-guard loss
to the responsible player. One-second/700ms are descriptive;9-second internal
and10-second full-response guards apply to this sequential strength assessment.
Infrastructure/correctness failures stop incomplete rather than invent a score.
No failed-game replacement, reattempt or automatic checkpoint promotion.

The two narrow fits are offline controls, not additional strength candidates.
Thus the study can establish broader-data prediction transfer and broad-versus-
original playing strength. It cannot directly establish that broad beats narrow
in live play. A direct broad-versus-narrow match would require a separate proposal.
The prepared main sample is not extended to chase a tiny or inconclusive effect.

## Time estimates on this computer

Measured host: Intel i5-13600KF,14 cores/20 logical processors,32GiB installed
RAM and Radeon RX7800XT, Windows. Estimates assume this study runs alone after
the current comparison finishes. Implementation and any new cloud-port effort
are outside these experimental runtimes.

| Stage | Central evidence | Advisory allocation |
|---|---|---:|
| 1,152 games, batch checks and independent corpus acceptance | 22.52h raw play;31.30h including cell checks; roughly2.62h further corpus audit | 34-40h |
| Four fits, exact final-update reproductions and qualification | Prior96-update fits316/329s; four fits about21.5min before overhead | 0.5-1.5h |
| Held-out and training-position predictions | About7,088 held-out roots, five models; prior evaluator timings | 20-60min |
| 12 operational plus400 strength games, replays and final verification | Prior400 main games11h31m; accepted tail1h45m; two cohorts here | 13-17h |
| Whole study | Rounded combined forecast | **48-60h** |

The measured CPU rate is51.15 collection games/hour, or36.81 including cell
startup, selected replays and saved checks. These came from only32 CPU games;
multi-day throughput is unmeasured. The compiled GPU16 alternative measured
38.93 verified games/hour, a5.75% pooled advantage, but failed the criterion for
a consistent advantage in both blocks. CPU8 remains the practical default.

For scale,512 training plus128 held-out games projects17.4h through cell checks
and32-42h for a comparable whole study;2,048 plus128 projects59.1h through cell
checks and80-100h overall. These are cost illustrations, not extra study arms or
approved fallback samples. The1,024 training-game recommendation gives a material
contrast while avoiding the approximately doubled collection cost of2,048.

## Storage and actual bounds

The current detailed recording format projects roughly68.5GiB for collection
before additional shared/replay/fit/assessment evidence. A practical whole-study
estimate is75-90GiB. This is a sampled estimate, not a measured future total:
four fixed game directories contributed43.9-47.0MB/game, with separate journals
averaging17.35MB/game and target records1.08MB/game. Full metadata and formulas
are in the preparation output's `runtime_estimates.json`.

Recommend a destination with160GiB working capacity. The observed local free
space was about39.7GiB, so identify another destination or first qualify a
separate lossless storage format. No historical data is deleted, and no assumed
compression ratio is used in this plan. An external SSD or rented persistent
storage is a practical option. This is preparation, not a new resource stop.

Proposed generous actual ceilings:120h overall,96h model cutoff,4h qualification,
60h collection including cell checks,8h independent corpus acceptance,4h total
fits,6h offline evaluation,40h strength qualification/operations/main,8h strength
replays,24h finalization including a12h independent reader. Each16-game cell has
a4h ceiling. All deadlines inherit and are clipped by one original wall and
monotonic clock; stage maxima are not additive extensions.

Actual owned RSS ceilings:8GiB collection/audit,4GiB fitting/offline evaluation,
2GiB strength inference. At most eight collection search clients, one model
service and one active phase; do not overlap this work with the current study.
New evidence cap128GiB, of which16GiB is reserved for finalization; actual free
disk floor5GiB. No available-RAM reserve, stability wait or forecast admission/
stop is introduced. These final proposal bounds supersede the cost note's
earlier96/8GiB draft. Failure receipts and incomplete prefixes remain preserved.

## Faster generation and rental options

The first scalable option is to divide predetermined independent games among
additional CPU workers or machines. The frozen teacher requires no synchronization
between games. Each machine should run the complete search/network/game loop
locally and return finished evidence; sending each neural call over the internet
would add latency to thousands of dependent calls. Distinct seed shards, exact
model/rules bindings and complete merged inventories remain mandatory.

A separately qualified Linux/CUDA service on rented NVIDIA hardware is another
candidate. More concurrent games can supply useful inference batches, but the
existing GPU batches were already nearly full. The measured service also spent
time preparing inputs, constructing receipts and writing journals. Faster GPU
arithmetic alone therefore has no established end-to-end speedup. Test complete
verified games per hour and per dollar, including startup, recording and audits.
Lossless binary/buffered recording and host preparation are further hypotheses;
they require preserved raw semantics and independent overhead/correctness checks.
Do not combine weaker teacher search or reduced precision with the data study.

Official public prices checked6November2026:

| Rental example | Published allocation | USD/hour | Compute for an assumed35 rented hours |
|---|---|---:|---:|
| DigitalOcean CPU-Optimized regular | 16 dedicated vCPU,32GiB RAM,200GiB SSD | 0.50 | 17.50 |
| Runpod RTX4090 Secure Cloud | 24GB VRAM,6vCPU,41GB RAM | 0.89 | 31.15 |
| Runpod L40S Secure Cloud | 48GB VRAM,16vCPU,94GB RAM | 1.09 | 38.15 |

Sources: [DigitalOcean pricing](https://www.digitalocean.com/pricing/droplets),
[Runpod pricing](https://www.runpod.io/pricing),
[Runpod L40S tiers](https://www.runpod.io/gpu-models/l40s).
These are illustrative compute costs, not cloud completion-time forecasts or
regional stock quotes. Taxes, GPU storage, setup, training and strength evaluation
are additional. We have not measured this workload on these machines. A vCPU
count is not a desktop-equivalent core-speed guarantee.

Runpod lists network storage below1TB at$0.07/GB/month;160GiB is about171.8GB,
approximately$12.03 for a full month, prorated. DigitalOcean's quoted disk is
included. Both offer usage billing, but stopped resources can retain charges:
settle evidence transfer and resource deletion as part of any future bounded
deployment. Avoid relying on interruptible spot capacity for an initial pilot.
[Runpod billing](https://docs.runpod.io/pods/pricing),
[DigitalOcean billing](https://docs.digitalocean.com/products/droplets/details/pricing/).

If renting is later authorized, first propose a **2-4 total instance-hour pilot
with a$10 compute ceiling**, plus at most$2 prorated persistent-storage allowance,
after preparing the portable build. Use predetermined CPU and/or GPU cells and
report verified games/dollar. No provider account, purchase or upload is needed
for the present preparation. Windows process ownership, native binaries and
DirectML are not portable by declaration; a Linux/CUDA route needs separate
end-to-end rules, numerical, decision, replay, clock and cleanup qualification.
Do not promise cross-backend bit equality or silently relax numerical checks.
Freeze the accepted collection backend before the scientific games; both data
arms must use that same corpus/backend. Cloud qualification games stay excluded.

## Decisions and deliverables

Finish and independently accept the current comparison before this study starts.
A useful current gain supports the short update; an inconclusive or modest result
still permits testing limited game diversity. A clear deterioration calls for a
saved-evidence postmortem before committing to this exact recipe. A significant
current win is not an invented admission requirement for collecting more data.

Before any execution: implement the separate production collector/loader/trainer
and assessment chain, choose the storage destination, reserve exact seeds, bind
sources, pass model-free and actual-entrypoint qualification, obtain independent
reviews and create an execution freeze. No new cloud optimization is a prerequisite
for a local CPU8 study. The running players, deadlines and verification stay fixed.

On completion report the fixed offline contrasts, both broad models' strength,
pooled and conservative uncertainty, time and storage including verification,
failures and exact cleanup. Produce one prediction-transfer chart and one
playing-strength chart. Original EF8A remains the reference until separately
justified promotion; external opponents and six-world work remain deferred.
