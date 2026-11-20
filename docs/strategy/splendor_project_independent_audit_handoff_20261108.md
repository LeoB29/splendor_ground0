# Splendor AI: handoff for an independent audit of research direction

Prepared 8 November 2026. Operational snapshot: **13:21 Paris / 12:21 UTC**.

This document is intended for a model or researcher with no access to the preceding conversation. It summarizes the evidence, current reasoning and unresolved choices; it is not a request to endorse the current approach. Please challenge the allocation of effort as well as the algorithms and experimental designs.

**Requested review:** Given the goal below, is our present research direction sensible? What are its strongest and weakest assumptions, what evidence are we missing, and which one or two experiments should follow the current study? Distinguish established facts, plausible interpretations and speculation. Please do not change or interrupt the running experiment. This is an advisory audit, not execution authorization.

The text is self-contained. The evidence index at the end is for a reviewer who also has the repository. Historical roadmaps contain superseded plans: the current execution protocol and subsequent user authorizations take precedence over their old “deferred” or “prepare only” language.

## 1. Executive overview

**Goal:** demonstrably superhuman play in two-player base-game Splendor, using only information lawfully available to the player. Roughly one second per move is an eventual practical ambition. The owner currently prioritizes understanding and improving strength over prematurely forcing every research candidate below that time.

The project has moved through locally trained agents, an expensive sampled-world rollout planner, and then a substantially stronger pretrained external player, **SplendoRust**. Our former planner lost all 60 games in an accepted, explicitly amended comparison against SplendoRust. We adopted the verified SplendoRust adaptation as the starting point, preserving earlier work and results.

With the inherited network unchanged, increasing search from small budgets through 128, 256 and 512 simulations has produced verified gains against the preceding search budget. **Improving the network itself has proved much harder.** A joint policy/value update, a policy-head-only update, and a shorter joint update have not established a useful playing-strength gain. Small improvements in prediction losses have not reliably translated into wins.

The running study collects **1,024 new training games and 128 held-out games**. It compares learning from many games with few positions per game against fewer games with more positions per game, keeping the number of examples and optimizer updates equal. It then tests both prospectively chosen broad-data models against the original network in 400 fresh games. It has no learning or strength result yet.

The current hypothesis is that limited independent experience contributes to weak learning. This is **a hypothesis, not a diagnosed explanation**. Alternatives include unsuitable targets, insufficient or excessive updating, policy/value interference, a strong pretrained model that is hard to improve with a small update, search masking prediction changes, and limitations of the teacher or state distribution.

No human benchmark establishes superhuman strength. Nor have we established that SplendoRust is the strongest publicly available player. Most recent comparisons are within one model/search family. This is a major gap between incremental progress and the final objective.

## 2. Vocabulary and what the player actually does

| Term | Meaning in this project |
|---|---|
| Network / checkpoint | Learned numerical weights. The original imported weights are called **EF8A**, from their file hash. |
| Raw035 | Our earlier fast local baseline. This is a historical player identifier, not a search-iteration count. |
| Complete player | Network plus search, hidden-information sampling, legal-action adapter, auxiliary choices, execution and clock policy. A stronger player need not contain a better network. |
| Policy output | Scores/preferences over available main actions, used to guide search or choose a move. It is not simply a separate win-probability estimate for each action. |
| Value output | An estimate of eventual game outcome from a position, used to evaluate an unfinished search leaf. It is not guaranteed to be calibrated. |
| World | One plausible completion of hidden information consistent with the player's observation. It is an approximation for reasoning under uncertainty, not an additional real game. |
| Search simulation | One tree-search iteration. It usually stops at a leaf evaluated by the network rather than playing an entire game. A normal 512-search decision uses **512 total iterations across three worlds**, not 512 per world: 171/171/170 starts feeding a shared tree. Forced moves and other declared branches can bypass search. |
| Self-play game / trajectory | One complete game played by the teacher against itself. Many training positions share that game's outcome and are correlated. |
| Training example / root | A recorded decision position, its search-derived policy target and its actor-oriented terminal outcome target. |
| Score | Win = 1, draw = 0.5, loss = 0. “56% score” need not mean exactly 56% wins. |

### Former planner versus current starting point

| | Former four-candidate planner | Verified SplendoRust adaptation |
|---|---|---|
| Main-action consideration | Four proposals: two Raw035 proposals, a seeded exploratory proposal and a learned proposal | Policy-guided adaptive PUCT tree search over the qualified main-action support |
| Future evaluation | Greedy Raw035 continuations played to termination across 128 shared sampled worlds | Search leaves evaluated by the pretrained value network; three sampled root worlds reused through search |
| Learned contribution | Learned proposal from one of three local fits; its learned value heads did not determine final move selection | Policy directs search allocation; value estimates support comparisons of unfinished positions |
| Main concern | Narrow proposals and potentially unrealistic continuation policies; expensive full-game rollouts | Imperfect leaf estimates, approximate hidden-world reasoning and search allocation; inherited design choices need not be optimal |

Both systems used attention-based neural models. The change was not “a neural player replacing a non-neural player.” It changed the starting model, search, evaluation and engineering together. Their individual contributions have not been causally isolated.

The imported SplendoRust network is an Entity Transformer with **4,780,883 parameters**, 31 semantic tokens, width 256, six attention layers and eight heads. Its output includes 81 main-action logits and two signed values. The policy and value heads share most of the network:

```text
Lawful observation -> shared representation -> policy scores -> guide search
                                          -> value estimates -> score search leaves

Search allocates work, compares actions, and selects the actual move.
```

Training shared layers can change both outputs even if one head's own weights are frozen. Our policy-only ablation therefore froze the shared representation and value head as well; it tested a much narrower intervention than unrestricted policy learning.

SplendoRust was imported from the MIT-licensed `paytonjjones/splendorust` repository, pinned at commit `1828507b8d2f0e02bc03d102b9c6d29a54fa8d57`. It is a developed pretrained system, not a newly written untrained baseline. We have not established a complete upstream lifetime training-compute accounting; the cached documentation explicitly does not establish superhuman status. Upstream results on other rules/profiles are not substitutes for our local tests.

## 3. History that should inform the audit

### Earlier programme: learning difficulties predate SplendoRust

The August roadmap, after experiments 102–132, recorded two blocks of searched self-improvement and deeper-search headroom, followed by repeated failures to obtain a reliable successor through that compact AlphaZero recipe. Tested directions included extra data, deeper teachers, capacity, replay diversity, representation and alternative value targets. That was a closure of a particular recipe, not evidence that AlphaZero or self-play is fundamentally unsuitable for Splendor.

Some specific precedents deserve attention:

| Historical experiments | Evidence and relevance |
|---|---|
| 102–103: learning across generations | Searched generation 5 beat generation 2 at 58.25%; generation 8 beat generation 5 at 59%, with paired intervals excluding 50%. Yet searched generation 8 scored only 26.5% against Raw035. Internal progress can coexist with a large external deficit. |
| 110: more fresh training games | A 256-versus-896-game training comparison did not establish a successor: the higher-dose model scored 50.5% against generation 11; policy loss improved but value generalization worsened. |
| 113–114: stronger search teacher | 1024-versus-256 search scored 57.375% over 400 games. A subsequent 256-game deeper-teacher update improved held-out policy/value losses, but scored 52.125% [47.5–56.75%] at ordinary 256 search. |
| 116: extra capacity | A residual tower increased total parameters from 356,450 to 991,828 while freezing the parent. Offline fit improved; live score was 51.625% [46.875–56.25%], with seat asymmetry. This was not an unrestricted larger-network comparison. |

These used earlier models, historical rules and different training schedules; they are not interchangeable replications of today's experiment. The new study starts from a much stronger 4.78-million-parameter model and a different search implementation, uses corrected collection, and holds training exposure fixed while varying game breadth. Nevertheless, the recurring pattern—stronger search, better prediction losses, but uncertain learned playing gains—must remain visible.

The older review motivated an information-set search-and-learning programme. It also identified narrow root coverage and unstable labels: independent shallow-resolver calls agreed on the exact best action only 45.83%, and an early visit-search experiment looked stable partly because its sharp prior dominated allocation. These numbers describe that historical resolver, not SplendoRust. Their warning is relevant: stable targets and better fit alone do not establish stronger play. Historical data must not be silently reused as corrected training data.

### Why the project adopted SplendoRust

The accepted **amended** 120-second benchmark ended **60–0 for SplendoRust PUCT6400**, against all three local planner fits across both seats: 59 natural wins and one retained local timeout loss. One interrupted move in the final game received an explicitly authorized retry; the original frozen attempt remains incomplete. The amendment is part of the result's provenance.

The four-candidate planner was the choice qualified at that allowance. An eight-candidate successor had a stronger historical local comparison, but missed **8 of 26** decisions in its 120-second timing panel. It did not enter this benchmark, so the 60–0 result is not a direct test against every retained planner under unlimited time.

A saved-game postmortem found that immediate maximum-prestige actions were offered, but greedy rollout futures generally lasted longer than games against the actual strong opponent. This suggests a continuation/evaluation mismatch. It does not establish that candidate coverage was sufficient, that every longer-term investment was wrong, or that any one architectural difference caused the defeat. More worlds can reduce sampling noise without correcting a poor continuation policy.

### Search gains with the same inherited weights

Intervals below are reported paired 95% intervals; draws count half. Each row is a distinct comparison, not a universal ranking or a transferable Elo estimate.

| Candidate versus reference | Main games | Candidate score | Paired 95% interval | Material qualification |
|---|---:|---:|---:|---|
| 32 versus 16 simulations | 200 | 64.0% | 57.75–70.25% | One-second games, no forfeits |
| Optimized64 versus unchanged32 | 200 | 64.25% | 58.25–70.25% | Includes two deadline-related losses |
| 128 versus optimized64 | 400 | 56.125% | 51.5–60.75% | Amended resumed collection; separately recovered verification |
| 256 versus 128 | 400 | 61.0% | 56.5–65.375% | Amended recovery with one disclosed retry |
| 512 versus 256 | 400 | 56.5% | 51.875–61.125% | All games natural, independently accepted |

The 128 amendment adopted one already-emitted move as a separately authenticated event; its original rejected clock remains preserved and excluded from ordinary timing/reproduction denominators. Assigning the retry-affected 256 game a loss conservatively still gave 60.75% and passed its useful-gain criteria. Later research comparisons accepted correct moves over one second under longer hard guards. They therefore do **not** demonstrate one-second deployment strength. In the 512 comparison, mean candidate response was **2.371 seconds**, with **92.46%** of moves over one second.

The owner retained **128 simulations with original EF8A as the practical research reference**. The stronger 512-search version supplies training targets. The older 6400-search player remains a high-budget historical reference, not the current teacher or a demonstrated fast configuration. References to 128, 512 or 6400 describe search settings, not different learned weights.

### Six worlds, timing and external opponents

Six versus three worlds at 64 simulations scored **50.5% [46–55%]** in 400 games. Better saved-position stability had not established better whole-game play. There was one move over one second for six worlds and three for three worlds, too few to establish a reliability ordering.

We spent substantial effort on a 700 ms admission margin before obtaining that strength comparison. The owner challenged this priority. The revised approach treats **700 ms as descriptive**, tests strength once correctness and basic feasibility are established, and then targets efficiency where useful. Six-world work is parked. A later exploration-coefficient screen at 128 selected neither 0.2 nor 0.8 over 0.4.

The qualified BreckEmert and domdomegg PPO512 exports each lost 0–60 against unchanged32 with **zero timeouts**; their policy responses were comfortably within one second. These tested exports were noncompetitive, not demonstrated casualties of the clock. External testing is paused because it had not produced useful strong opponents. That does not eliminate the need for eventual independent opponents and expert-human calibration.

## 4. Recent training evidence and its limitations

Our recent learning loop is:

```text
Frozen original EF8A + 512 search
    -> complete corrected-rules self-play games
    -> search visit distributions + official final outcomes
    -> modest updates starting again from original EF8A
    -> independent prediction assessment
    -> fresh complete games against original EF8A at matched 128 search
```

Policy targets contain information from search, rather than merely asking the network to reproduce its own unsearched favourite move. Value targets come from completed games. Playing collection games to the end supplies those targets; search itself still evaluates unfinished positions. This is one fixed-teacher generation and training update, **not yet an iterative, compounding AlphaZero loop**.

The earlier corrected corpus had **192 training games**, producing **10,660 training positions**, plus 32 validation and 32 test games. Ten thousand correlated positions are not ten thousand independent game outcomes.

| Study | Design | Accepted finding |
|---|---|---|
| Original joint pilot | 10,660 positions; 668 updates, approximately two passes; shared layers and both heads trained | 400 fresh games: **49.125% [44.50–53.75%]**. No useful strength gain. |
| Policy-head-only ablation | Same corpus and 668 updates; only 20,817 parameters trained, **0.4354%** of network; shared layers and value head frozen | 400 fresh games: **50.375% [46.00–54.75%]**. No useful gain from this restricted update. |
| Breadth × trainability diagnostic | 64 games × 48 positions versus 192 × 16; both 3,072 positions and 668 updates, about seven passes; joint/policy-only, two repetitions each | Joint held-out policy/value losses worsened; broader games reduced that deterioration. Policy-only gave about 0.21% policy-loss improvement and exactly unchanged values. |
| Shorter joint update | Two fixed fits, 192 × 16 positions, 96 updates: one pass | Mean policy loss improved **0.634%**, interval **0.347–0.927% improvement**. Value change unresolved. Previously opened assessment games make this exploratory. |
| Fresh one-pass strength comparison | Both fixed fits, 200 games each against original, matched 128 search | **51.75% [47.25–56.25%]** overall: 201 wins, 12 draws, 187 losses. No useful strength gain established. |

The last two fixed models scored 54.25% [48–60.5%] and 49.25% [42.75–55.75%]. Neither was selected or promoted. All 400 main and 12 operational games ended naturally; all required verification passed. These are not different candidates selected after inspecting scores.

Several distinctions are essential:

- **The overfitting explanation is incomplete.** The long diagnostic reused 3,072 examples about seven times, whereas the original pilot used 10,660 about twice. Divergence in the former is evidence about that recipe, not proof of what caused the original null playing result.
- **Better loss is not necessarily better decisions.** The original pilot improved policy cross-entropy by roughly 0.7% on held-out partitions while its agreement with the teacher's top action fell from 70.26% to 69.54%. Value-loss changes were uncertain.
- **Policy-only was heavily constrained.** Its result does not rule out useful policy learning when the representation can adapt. Conversely, training both shared representation and heads can introduce interference; we have not isolated that as the cause.
- **Historical scores do not establish a monotonic learning improvement.** The intervals around 49.125%, 50.375% and 51.75% overlap and came from different game sets. No direct comparison established that these recipes rank in that order.
- **No demonstrated gain is not proof of no gain.** The last interval permits modest harm, no change and a worthwhile benefit. It does not establish equivalence or that more data will work.

The breadth diagnostic's original charting/finalization failed through an OpenMP conflict; a separate saved-evidence finalizer reproduced the numerical analysis and accepted it. That technical failure and its recovered verification remain separate from the scientific result.

## 5. Exactly what is running now

Authoritative protocol: `docs/engineering/splendorust_independent_data_execution_20261107.md`.

Source: `tools/splendorust_independent_data_20261107`.

Evidence: `outputs/splendorust_independent_data_20261107_01`.

### Scientific design

| Component | Frozen choice |
|---|---|
| Collection | **1,152 fresh independent setup trajectories**: 1,024 training, 64 validation, 64 test. Original EF8A teacher; 512 simulations, three root worlds, cpuct 0.4, depth 64, float32. |
| Narrow control | Predetermined nested subset of 192 training games × 16 selected positions = **3,072 examples**. |
| Broad candidate | 1,024 training games × 3 selected positions = **3,072 examples**. |
| Position selection | Predetermined, outcome-blind selection; equal totals from relative early/middle/late thirds. Distinct examples within each fit; no failed-game substitution. |
| Fitting | Two fixed repetitions per arm: four fits, each starting from pristine EF8A. Joint shared-layer/policy/value training, 96 batches of 32. AdamW learning rate 1e-5, weight decay 1e-4, gradient clip 1. No sweep or early stopping. |
| Assessment separation | Commit all four final checkpoints before held-out model predictions. Raw held-out correctness checks may precede that; learning results may not guide fitting. |
| Primary offline question | Equal-repetition average **broad-minus-narrow policy cross-entropy**, averaging positions within games first. Lower is better. Value MSE and each model versus original are secondary. |
| Offline uncertainty | 20,000 paired whole-game bootstrap draws, stratified across the two 64-game partitions; report each partition and predefined pooled 128. |
| Fresh playing strength | **Both broad models**, nominated before results, receive six excluded operational games and 200 main games each against original EF8A: 400 main games, 200 disjoint seat-swapped pairs. Both sides use 128 search simulations, three worlds and unchanged remaining settings. |
| Strength analysis | Equal average across the two fixed models; 50,000 stratified paired bootstrap draws. Report each model and both seats. Narrow fits are offline controls only. |

The useful-strength-gain criteria require score **at least 55%**, a two-sided 95% lower bound **above 50%**, and pooled score **at least 50% from each seat**, plus frozen validity/degenerate-interval safeguards. Both official and conservative analyses must pass. Classified hard-guard failures lose for the responsible player; conservative analysis treats every administrative game as a candidate loss. Infrastructure or correctness failures stop the study incomplete.

Both broad fits proceed to fresh games regardless of offline loss signs. There is no post-hoc winner selection, sample extension, game replacement, automatic promotion or automatic subsequent study.

### What this design can and cannot answer

It tests **broader game coverage versus denser within-game sampling at fixed training exposure**, followed by whether the broad-trained models improve on the parent. It deliberately does not train on every collected position.

It does **not** directly compare broad versus narrow in live games, since narrow models receive no live assessment. Therefore, even broad offline superiority and a broad-versus-parent playing gain would not directly establish that breadth caused a live advantage over narrow training. It also does not establish an optimal dataset size, learning rate, training duration or teacher budget.

Two repetitions share the corpus and held-out games; they do not turn 128 held-out games into 256 independent games. The uncertainty is conditional on these fixed fitted models and corpus, not all possible training datasets/seeds. A gain at matched 128 search need not persist at 512 search or against different opponents.

### Status at the snapshot

At **13:21 Paris on 8 November**, **1,120 of 1,152 games had finished**; **69 complete cells / 1,104 games** had passed their independent cell checks, including **3,312 exact selected model replays**. The remaining finished cell was progressing through replay/audit work. The exact root and cell processes were alive, progress was fresh, and no recorded game/cell failure, bad exit or error output was found. **Scientific fitting had not started.**

This is an operational snapshot, not a provisional learning result. The normal remaining sequence is automatic: finish collection and corpus acceptance, four fits, held-out assessment, operational and main strength games, selected replays, independent final verification and actual process cleanup. Later status must be read from current receipts; this document does not assert completion.

## 6. Correctness, rules and reproducibility boundaries

External qualification exposed six incorrect historical card costs and missing compulsory noble awards after non-purchase turns. All new collection, training and assessment must use the corrected rules/decoder/loader chain. Merely attaching a corrected label to old data is prohibited. Original EF8A is explicitly a historical warm start, not a checkpoint whose original training data we have certified under our corrected rules.

The current entry point passed actual collector/loader/trainer counterexamples covering corrected cards, noble eligibility/final-round effects, payment and returns, private information, value orientation and invalid metadata. Each new worker verifies its rule sources at startup. Frozen inventories, independent transition/outcome checks, selected exact model replays and final independent arithmetic/acceptance checks bind the evidence to the intended players.

Remaining limitations are disclosed rather than silently repaired during experiments:

- **Atomic pre-reveal auxiliary choices:** payment/return/noble choices are precommitted before certain reserve/refill information appears. Final moves can be legal while the strategy space is restricted relative to a fully sequential interface.
- **Forced-pass/deadlock conventions:** internal search conventions are not authority to manufacture official training outcomes. Only natural official completions supply outcome labels; unresolved deadlocks/cutoffs are not relabelled wins.
- **Hidden-world approximation:** lawful inputs and privacy checks do not establish optimal information-set reasoning. Belief sampling, strategy fusion and allocation among hidden possibilities remain audit questions, not demonstrated defects from the present results.
- **Verification is extensive but not omniscient:** all saved moves/outcomes are checked and selected decisions replayed exactly; this is not independent reexecution of every internal neural search edge or a proof that every algorithmic assumption is appropriate.

An incomplete attempt is not a valid negative scientific result. A later separately accepted recovery must remain identified as such. A completion file alone is insufficient: independent acceptance, observed actual exits and cleanup are required.

## 7. Compute, storage and the cost of the research process

The measured machine is an **Intel Core i5-13600KF, 14 physical / 20 logical cores, about 32 GiB RAM, AMD Radeon RX 7800 XT, Windows**. Current collection uses **eight concurrent CPU search clients**, a shared inference service and a fixed 16-game refill queue. Fresh strength matches are sequential. Older README references to an intended NVIDIA/CUDA machine do not describe this measured setup.

Generating complete games and evaluating strength dominate the cost; the small training updates take minutes. Taking three rather than sixteen training positions from an already completed game barely changes collection cost. More independent games require more games to be played, even with equal training examples.

A qualified CPU8-versus-compiled-GPU16 comparison used 32 distinct setups on each backend, 64 trajectories total. CPU8 achieved **51.15 raw / 36.81 batch-verified games per hour**; GPU16 achieved **55.37 / 38.93**. GPU's verified gain was **5.75% pooled**, but its two blocks were +12.92% and -1.26%, failing the predeclared consistency criterion. CPU8 remains the default. This is evidence about this AMD/Windows implementation and workload, not a general conclusion that GPUs cannot help.

Current CPU collection throughput has varied substantially without changing the player; recent verified rates ranged from roughly 35 to 62 games/hour. The cause is unmeasured. Host variability complicates tiny speed comparisons and forecasts, and concurrent collection gains must not simply be applied to sequential strength assessment.

Detailed immutable records, repeated audits and many preserved versions consume substantial storage and time. The older recording format forecast 75–90 GiB for this study. A separately qualified lossless compression route now preserves the same logical evidence; a measured journal compressed to 24.82% of its raw size, not a guaranteed whole-study ratio. No historical evidence was deleted to make room.

Current actual bounds include **120 hours overall**, a **96-hour model cutoff**, 128 GiB new evidence with 16 GiB reserved for finalization, a 5 GiB actual free-disk floor and phase-specific actual process-memory ceilings. Stage caps share the original clocks. Collection decisions retain 60/75-second internal/complete-response guards; strength uses 9/10 seconds. **One second and 700 ms are descriptive in this research study.**

At the owner's request, available-RAM reserve waits and forecast-based stops have been removed from this study. Actual guards, preservation and correctness checks remain. There are no scheduled chat-monitor tasks. The owner wants automatic progression through authorized stages and prompt investigation of failures, without repeatedly waiting for a status request.

Cloud parallelism is a plausible future option, not a qualified faster route. No rental, paid capacity or upload is authorized. Porting, runtime qualification, storage/transfer and total verified throughput belong in any cost comparison. The current narrow study is authorized; further dataset scaling is not automatically authorized by it.

## 8. Questions the independent audit should answer

1. **What would establish the actual goal?** Propose an operational definition of superhuman two-player strength, credible opponents/humans, declared rules and compute, and a realistic progression toward that evidence. How much confidence can a chain of parent comparisons justify?

2. **Are we targeting the right bottleneck?** Rank data diversity, target quality, teacher coverage, representation/capacity, update size/duration, loss balance and policy/value interference as hypotheses. Explain which observations discriminate among them and which do not. Do not assume “overfitting” diagnoses the original pilot.

3. **Is the current fixed-effort study sufficiently informative?** Does spending to collect 1,024 games but training each fit on only 3,072 positions and 96 updates offer good value? What can a null result distinguish? Does the absence of narrow-model live games materially weaken the decision we hope to make?

4. **Why should these targets produce stronger play?** Consider search visits versus raw-policy imitation; noisy, policy-dependent terminal values; hidden information; game/position correlation; and whether better cross-entropy actually changes useful search decisions. Would better targets or weighting be more promising than simply more games?

5. **What should a first successful learning cycle look like?** Should the next intervention be another controlled update, training on more of the collected corpus, stronger/diverse teachers, an iterative self-play cycle, or distillation? What evidence would justify a second cycle and later scale? How does this differ from the older compact AlphaZero programme?

6. **Are we measuring the right strength/cost trade-off?** Should confirmation use matched simulations, matched real time, a fixed generous teacher budget, or several budgets? Could a gain at 128 vanish at 512? Is further search scaling informative enough to justify its cost before more learning?

7. **How should generalization be tested?** When should paused independent-opponent work or expert-human tests return? Could a historical-agent league expose non-transitivity even before a credible new external opponent is found?

8. **Which engineering work is worth its opportunity cost?** Assess evidence volume, repeated replay/audit work, exact-output requirements, process supervision and platform variability. Identify prospective simplifications that retain credible correctness and failure accounting. Do not weaken the running protocol retroactively.

9. **Are important game-model limitations being neglected?** Assess sequential auxiliary choices, legal action support, hidden-information search and internal deadlock handling. Distinguish a demonstrated correctness problem from an untested opportunity for strength.

10. **What should make us change course?** Give explicit evidence and cost thresholds for continuing the current approach, revising training, prioritizing search/efficiency, or revisiting architecture. Avoid both endless local tuning and an unsupported wholesale redesign.

## 9. Requested audit output and decision boundaries

Please return:

- A concise diagnosis of where the programme stands: strong evidence, weak assumptions and the three most consequential gaps.
- A recommendation to retain, adjust or replace the present direction, with reasons and alternatives considered.
- A conditional interpretation of the running study: broad offline improvement plus live gain; offline gain without live gain; inconclusive evidence; or deterioration. None should automatically trigger promotion or more samples.
- **At most two prioritized next experiments**, with a specific hypothesis, controlled contrast, independent success criterion, rough full cost, and a decision/stopping rule. Say what result would change your recommendation.
- Any missing information essential to that recommendation, separated from information that would merely be useful.

The current working view is that the frozen study should finish, then determine whether to investigate data, objectives/optimization or transfer into search. The reviewer is invited to disagree with that prioritization. Original EF8A stays the reference until a deliberate decision; a new negative result is valid evidence, not a technical failure to be repaired until positive.

External testing, six-world work, further GPU optimization, additional scaling and rentals remain parked unless separately chosen. Earlier work should not be discarded, but sunk engineering effort is not a reason to preserve an unproductive research direction.

## 10. Evidence map and exact identities

Repository root: `C:/Users/leoba/Desktop/splendor_ground0`.

Links below are relative to this document. They are supporting evidence, not required reading to understand the summary. Start with the active protocol and recent accepted report; inspect older sources for specific disputed claims.

| Topic | Primary local source |
|---|---|
| Live continuity; current authorizations | [AGENT_CURRENT.md](../../AGENT_CURRENT.md), [AGENT_HANDOFF.md](../../AGENT_HANDOFF.md) |
| Original objective and earlier failed learning recipe | [August programme review](superhuman_information_set_search_roadmap_after_132.md) |
| Earlier learning and external deficit | [102 result](../engineering/alphazero_endpoint_evaluation_102_result.md), [103 result](../engineering/alphazero_scaled_continuation_103_result.md) |
| Earlier data, teacher and capacity experiments | [110](../engineering/alphazero_fresh_dose_110_result.md), [113](../engineering/alphazero_search_headroom_113_result.md), [114](../engineering/alphazero_high_budget_distillation_114_result.md), [116](../engineering/alphazero_capacity_pilot_116_result.md) |
| Pivot and architecture comparison | [24 October revised roadmap](splendorust_revised_roadmap_2026-10-24.md), [23 October technical description](splendorust_revised_roadmap_2026-10-23.md) |
| Amended 60-game result | [Accepted finalization](../../outputs/matched_strength_recovery_20261022_01/finalization.json), [postmortem](saved_game_postmortem_2026-10-22.md) |
| Why four rather than eight candidates entered qualification | [Local player timing panel](../engineering/local_player_qualification_20261018_result.md) |
| Upstream provenance | [Cached README](../../outputs/external_opponent_search_20261018_02/splendorust/README.md), [cached champion report](../../outputs/external_opponent_search_20261018_02/splendorust/docs/CHAMPION.md) |
| 32 and 64 search results | [32 analysis](../../outputs/splendorust_search_scale_20261024_01/analysis.json), [64 report](../engineering/splendorust_graph64_strength_20261025_results.md) |
| 128 and 256 search results | [128 recovered audit](../engineering/splendorust_search128_saved_audit_20261029_results.md), [256 recovery result](../engineering/splendorust_search256_recovery_20261030_02_results.md) |
| 512 strength and timing | [Accepted report data](../../outputs/splendorust_search512_20261101_01_closeout/report_data.json), [final closeout](../../outputs/splendorust_search512_20261101_01_closeout/final_closeout.json) |
| Six worlds and strength-first correction | [28 October roadmap](splendorust_roadmap_update_20261028.md), [accepted separate review](../../outputs/splendorust_pool6_strengthfirst_20261028_continuity/assessment_accepted_separate_review_recovery.json) |
| Exploration allocation | [128 exploration result](../engineering/splendorust_exploration128_20261029_02_results.md) |
| Independent opponent screens | [BreckEmert](../engineering/breck_screen_20261026_results.md), [domdomegg](../engineering/dom_screen_20261027_v2_results.md) |
| Mandatory corrected training rules | [Rules policy](../engineering/future_training_rules_20261021.md) |
| Original joint pilot | [Accepted closeout](../../outputs/splendorust_selfplay_pilot_resume2_20261103_01/final_closeout.json), [postmortem](../../outputs/splendorust_selfplay_postmortem_20261104_01/report.json) |
| Policy-only pilot | [Accepted closeout](../../outputs/splendorust_policy_head_20261104_01/final_closeout.json), [strength analysis](../../outputs/splendorust_policy_head_20261104_01/assessment/main_analysis.json) |
| Breadth × trainability diagnostic | [Recovered final summary](../../outputs/splendorust_learning_finalization_20261106_01/report/final_summary.md) |
| Shorter joint update | [Accepted summary](../../outputs/splendorust_shorter_joint_20261106_01/report/final_summary.md) |
| Latest completed strength study | [One-pass final report](../../outputs/splendorust_onepass_strength_20261106_01/closeout/final_summary.md), [accepted closeout](../../outputs/splendorust_onepass_strength_20261106_01/accepted_closeout.json) |
| CPU/GPU collection comparison | [Accepted results](../../outputs/splendorust_gpu_collector_report_20261105_01/results.md) |
| Current study design, scientific and operational bounds | [7 November execution protocol](../engineering/splendorust_independent_data_execution_20261107.md), [earlier preparation](../engineering/splendorust_independent_data_preparation_20261106.md) |
| Active freeze and stage | [Freeze inventory](../../outputs/splendorust_independent_data_20261107_01/freeze.json), [stage receipt](../../outputs/splendorust_independent_data_20261107_01/stage.json) |
| Simulation/world accounting | [Collector-bound native search](../../outputs/splendorust_selfplay_qualification_20261101_01/native/crates/splendor-agents/src/neural_search.rs), loop at line 1473 and world selection at line 1491 |

Exact reference identities:

```text
Original checkpoint:
outputs/external_opponent_search_20261018_02/splendorust_model.pt
SHA256 ef8a4521cd6c03c7075f15efee23f4cde6ec94d1b5398765cf0c24a09ad745cb

Ruleset: splendor_base_qualified_rules_20261018_v1
Catalogue SHA256:
db4776918e612eee5a6f2617719c85ec5f413d592bbab0a64058babc83a70312
Turn contract: atomic_prereveal_completed_turn_v1

Current independent-data freeze SHA256:
9e442e2711107e33c78d35905db384f8d633f4e29c0c05d412643fa414a3c517
```

Prepared from saved protocols, accepted reports and small operational receipts. No active held-out prediction/loss results were opened for this handoff, and no running player, source, setting, clock or process was changed.
