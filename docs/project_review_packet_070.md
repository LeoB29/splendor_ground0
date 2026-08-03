# Independent Project Review Packet

**Project:** strong two-player Splendor AI  
**Review snapshot:** 2026-08-03, repository commit `2fa2944`  
**Active champion:** legal policy checkpoint `035`  
**Purpose:** obtain a genuinely independent project-level recommendation before committing to the next learning direction

## 1. How to read this packet

This is a decision review, not a request to ratify the project's current conclusions. The repository has used prospective protocols and strict conjunction gates to limit post-result tuning. Those procedures protect against overfitting, but a formally failed experiment is not always negative scientific evidence. Please distinguish:

1. **Integrity failure:** the evidence cannot answer the intended question.
2. **Learning or gameplay failure:** the candidate was validly tested and performed inadequately.
3. **Threshold-sensitive rejection:** the evidence was favorable or neutral, but one prospectively frozen numerical condition failed narrowly.
4. **Architectural infeasibility:** scaling the tested mechanism is unlikely to repair the problem.

The history uses some branch-local numbers as well as chronological experiment numbers. Refer to experiment names, dates, and artifacts rather than assuming every repeated number belongs to one global sequence.

The long authoritative record is `AGENT_CURRENT.md`; `AGENT_HANDOFF.md` contains older operational detail. This packet deliberately compresses both and may contain omissions. Reviewers should inspect the cited artifacts when a conclusion depends on a precise metric.

## 2. Objective and practical constraints

The target is a strong AI for the base two-player game of Splendor, with:

- exact rules and deterministic replay;
- acting-player-legal observations and actions;
- correct treatment of hidden deck order and blind reserves;
- robustness to legal forced-pass deadlocks;
- a fast neural policy for ordinary play;
- an eventual iterative improvement/self-play loop, if a reliable improvement operator can be established.

The current machine has 10 physical CPU cores / 20 threads and no GPU. The user can rent a GPU if it materially changes model quality, but prefers CPU when a GPU would merely save a few hours. In most completed experiments, environment simulation or stochastic continuation search—not neural fitting—was the dominant cost.

The user played ten serious local games against `035` and won 8-2. The user estimates their strength near an ordinary competitive player but does not know their population percentile. They do not consider themselves reliable enough to label isolated positions and do not want a manual annotation project. These ten games remain calibration evidence, not training data.

## 3. Current technical baseline

### Simulator and legality

The production engine, replay system, paired-seat evaluation, legal-action encoding, and canonical public-belief representation are extensively tested. Blind reserves expose identity only to their owner; opponent blind reserves retain public tier but not identity. Search uses acting-player-visible determinizations rather than the true hidden deck.

The engine treats a position in which both players can only pass as a valid `forced_pass_deadlock`, ending by the established score/card-count rule. Such games are retained. Earlier experiments that rejected this termination as invalid remain formally rejected, but their integrity interpretation was later corrected.

The clean pre-evidence test suite for experiment `069` passed 565/565.

### Champion `035`

`035` is the strongest fast legal checkpoint:

`outputs/legal_action_scorer_035_legal_observation/legal_action_attention_scorer_best.pt`

It is the first checkpoint trained under the repaired observation contract, but it was initialized from historical `022`, whose training predates the blind-reserve legality repair. At inference the encoder masks illegal information, and `035` was retrained, but historical snapshots cannot reconstruct which old reserves were public. Thus `035` is legal in operation while its initialization and some teacher provenance are not clean-from-scratch. This is a possible representation/data limitation worth reviewing.

`035` is competent against all local fast baselines. It has not been shown to be near optimal, and the small human result suggests exploitable strategic headroom.

## 4. Original roadmap and the stalled boundary

The intended roadmap was:

1. exact simulator and legal baseline;
2. supervised/structured policy warm start;
3. establish a search, planning, value, or population operator stronger than the current policy;
4. collect complete soft targets from that stronger operator;
5. distill synchronously;
6. alternate collection and supervised learning to enter iterative self-play.

Stages 1 and 2 are complete. The unresolved boundary is stages 3-5: a stronger decision procedure has existed, but no tractable target or learning update has converted that advantage into a reproducibly stronger fast checkpoint.

## 5. Evidence by research direction

| Direction | Strongest positive evidence | Decisive limitation or failure | Closure confidence |
|---|---|---|---|
| Structured supervised policy | Coherent shallow teacher data produced `023`, which substantially improved over earlier entity policies; legality/loop repair ultimately led to `035` as the fast champion. | Larger mixed imitation improved held-out action prediction while losing badly in gameplay; DAgger and preference patches caused trajectory/distribution-shift regressions. | High for more imitation of the same local teachers; low for imitation of a genuinely stronger external teacher. |
| Actor-critic / AWR / trajectory returns | KL-targeted updates moved the policy safely; final score margin was a better return than WDL; several small parent gates were favorable. | Independent seeds produced different policy directions; balanced strong-opponent cells often contained only losses; dense score-change credit worsened critic quality; no update passed broad promotion. | High for another small same-recipe update; moderate for a materially different population objective. |
| Counterfactual action credit | Mid/late matched continuations showed real replicated action ordering; a 240-state late corpus contained strong action signal. | Early signal was unstable; the auxiliary Q model failed to generalize on confirmation; local matched-return improvements did not translate into full-game strength. | High for the tested Q/compression recipe; moderate for direct use of expensive estimates. |
| PUCT, values, and root-visit distillation | Some late n-step and clipped-leaf variants beat weaker search controls on selected blocks. | PUCT did not reliably beat its `035` prior; s512 was no better than s64; learned public-belief value models failed calibration/ordering gates; visit targets largely reproduced prior behavior. | High for local budget/leaf/prior tuning. |
| Public-belief resolver | Resolver `051` is the strongest demonstrated gameplay operator: `138-62` versus raw `035`, then 74.5%, 74.5%, and 79% shares against shallow, `016`, and `023`, with clean information-legal branch accounting. | Resolver-every-move distillation `052` halved target KL but barely improved top-action agreement and drifted far from fallback behavior. Later audits found early/mid resolver action values too variable across repeated belief samples and budgets for stable supervised labels. Runtime is very high. | High that the resolver itself is strong; high that the tested one-shot target/distillation recipe failed; lower that every possible use of the resolver is exhausted. |
| Rolling-horizon planning | One 100-pair block scored 58.5% versus `035`. | A corrected replication scored 48%; the final 500-pair robustness gate scored 50.55%, CI95 [47.75%, 53.40%]. | High: near parity under the tested planner. |
| Hierarchical plans/options | Outcome-blind diagnostics strongly confirmed multi-turn plan persistence. | An option-conditioned predictive model was significantly worse than the state-only model on fresh confirmation. | High for that explicit option representation; low-to-moderate for all possible memory/latent-intent models. |
| CFR / Deep CFR | Exact and sampled CFR converged on a reduced legal Splendor subgame after an average-policy repair. | Full terminal external traversal hit node/depth limits; outcome-sampling importance weights had catastrophic variance/ESS; bounded resolver targets were too noisy for Deep-CFR-style fitting. | High for naive full-game traversal and unbiased terminal importance weighting; moderate for all continual-resolving formulations. |
| Evolution strategies | A constrained adapter update beat its clean-clone parent. | It tied raw `035` over a fresh 200-pair bridge. | High for the tested low-rank ES adapter. |
| Selective matched rollouts | Two independent state sets showed large, stable local matched-return gains. | A deployment-realistic one-shot intervention scored 47.875% against `035` over 400 games despite excellent offline effects. | High that local rollout value is not sufficient for trajectory improvement under this operator. |
| Recurrent population learning | History-required residuals composed for multiple generations. Large experiment `069` collected 6,240 clean balanced league games; generation 2 passed KL, action movement, surrogate, history-use, reload, and a 52% immediate-parent gate. | Generation 2 normalized validation entropy was 0.294256 versus a frozen 0.30 minimum. The run stopped before generation 3 and the final broad gate. Earlier broad panels were favorable or neutral overall but statistically inconclusive and seat-heterogeneous. | High that `069` formally failed; low that the candidate is proven weaker; disputed whether the entire outcome-only recurrent family should close. |

## 6. The strongest unresolved tension: resolver strength versus target quality

The project did not merely fail to find a stronger policy. Resolver `051` demonstrably played much more strongly than `035` and three broad opponents. This is the closest analogue to the search-improvement prerequisite in AlphaGo.

However, the resolver's estimates were expensive and unstable at individual early/mid states. Distillation `052` learned the soft target distribution in aggregate but barely improved top-action agreement, damaged fallback-anchor behavior, and did not earn gameplay. Subsequent fixed-state convergence studies found that increasing belief breadth helped value accuracy more than action-label repeatability; early positions remained particularly noisy.

An external reviewer should decide whether this branch was closed too broadly. Possible interpretations include:

- the resolver is a strong trajectory-level policy whose per-state numerical targets are not stable enough for ordinary supervised distillation;
- the canonical model lacks the memory/latent belief capacity needed to compress it;
- the collection distribution was too off-policy or too small;
- the resolver's advantage comes from correlated sequences, making independent root labels inappropriate;
- or the compute cost makes the operator unsuitable regardless of model changes.

The repository's current interpretation favors the last two explanations, but this is not settled beyond reasonable dispute.

Key artifacts:

- `outputs/paired_public_belief_resolver_gameplay_050/result.json`
- `outputs/broad_public_belief_resolver_confirmation_051/result.json`
- `outputs/resolver_selfplay_distillation_052/result.json`
- `outputs/resolver_distillation_disagreement_audit_052a/result.json`
- `outputs/resolver_residual_ranking_distillation_053/result.json`
- `outputs/resolver_public_history_representation_054/result.json`
- `outputs/resolver_target_repeatability_audit_055/result.json` and later resolver convergence artifacts documented near the top of `AGENT_CURRENT.md`

## 7. Current experiment `069`: what it does and does not establish

Experiment `069` was a prospectively frozen scale-or-stop test of history-required recurrent population succession.

### Generation 1

- 2,880 fresh balanced games and 90,290 actor decisions;
- validation KL 0.003935;
- 3.483% greedy action movement;
- positive overall and mid/late surrogate;
- normalized entropy 0.309344;
- 49.5% immediate-parent score share and +0.28 paired margin;
- all frozen conditions passed.

### Generation 2

- 3,360 fresh balanced games and 105,087 actor decisions;
- 3,358 ordinary completions and two legitimate forced-pass deadlocks;
- validation KL 0.005131;
- 3.634% action movement;
- positive overall and mid/late surrogate;
- real-versus-rotated-history TV 0.036045;
- 52% immediate-parent score share, mean paired margin -0.10;
- normalized validation entropy 0.294256, below the frozen 0.30 minimum.

The runner correctly stopped before generation 3 and the planned 1,600-game broad gate. No rule was changed after evidence.

Therefore:

- `069` did **not** pass;
- generation 2 is not promotable under its protocol;
- the 0.30 floor cannot honestly be changed retroactively;
- but the result is not evidence of gameplay regression or policy collapse;
- and it does not reveal whether generation 3 or the broad gate would have succeeded.

The project declared the exact outcome-only recurrent recipe closed. The reviewer should assess whether that closure is scientifically proportionate, and whether a fresh structurally entropy-constrained objective would be a genuinely new justified experiment or merely post-result threshold repair.

Primary artifact:

`outputs/recurrent_scale_or_stop_069/result.json`  
SHA-256: `0970851f468ffae65ac68846a8aba439f27b7815985915c0f69542c86e7c8f3f`

## 8. Evaluation and gate-design concerns

The project deliberately favored false negatives over false promotions. That may now be creating path dependence.

Examples worth auditing:

- Rolling-horizon `041` passed every strength metric at 58.5% but was formally rejected because one legitimate forced-pass deadlock was not listed as an allowed terminal. Corrected and larger replications later showed near parity, so the original strictness prevented a false promotion but for the wrong immediate reason.
- Selective rollout `004` passed strong independent action-quality gates but exceeded its maximum switch rate by 3 states: 30.625% versus 30%. Its separately designed deployment test later failed gameplay, again vindicating caution without validating the original threshold.
- Recurrent seat-0 confirmation `068` was essentially neutral, but its confidence lower bound was -0.06 versus a -0.05 noninferiority requirement.
- Recurrent `069` missed normalized entropy by 0.005744 while every other condition passed.

Questions for the reviewer:

1. Are fixed conjunctive pass/fail gates still the correct decision rule after dozens of sequential branches?
2. Are thresholds such as 55% score share, CI lower bound above 50%, and hard per-stratum floors calibrated to realistic effect sizes and sample variance?
3. Should integrity, safety, offline-learning, and gameplay evidence be combined by a hierarchical decision rule rather than a flat conjunction?
4. Is the project underestimating non-transitivity by treating direct `035` performance as the principal progression axis?
5. Should a league rating or population exploitability criterion replace some pairwise gates?
6. Has repeated reuse of `035` as prior, opponent, continuation policy, and benchmark made improvement targets self-referential?

## 9. Data and representation concerns

The reviewer should independently consider:

- whether acting-player-visible public state plus action history is sufficient, or whether the model needs an explicit public belief distribution over deck/reserve possibilities;
- whether long-term reserve targets and opponent denial require recurrence, latent options, or a different entity/message-passing architecture;
- whether final score margin and WDL provide enough policy-learning information in a game with delayed purchases and nonlocal denial effects;
- whether the current opponent league is broad enough;
- whether a competent but imperfect `035` can bootstrap beyond itself without a demonstrably stronger teacher;
- whether the ten human games reveal a genuine strategic gap despite being too few for training.

## 10. Candidate next directions for independent ranking

These are options, not conclusions.

### A. External-teacher qualification

Search for an open-licensed or expressly authorized Splendor agent/replay source, adapt it to the local legal engine, and require it to beat `035` before generating labels. [Board Game Arena's replay documentation](https://en.doc.boardgamearena.com/Game_replay) describes replays as exact recordings, but [BGA's current terms](https://en.boardgamearena.com/legal?section=legal) prohibit automated access and unapproved extraction; use would require an official export or written permission. No scraper should be built.

This direction changes the source of supervision rather than reprocessing `035` outcomes. Its risk is that no compatible external agent is actually stronger and authorized human data may be unavailable.

### B. Reopen resolver compression with a new sequence-level formulation

Treat resolver `051` as a proven strong teacher but stop demanding stable independent scalar action values. Possible alternatives include complete trajectory imitation, recurrent policy distillation from teacher rollouts, ranking-only targets, or a student trained on teacher state distributions with explicit belief/history representations.

This has the strongest internal teacher evidence, but substantial compute and several negative distillation/target-quality results argue against casual reopening.

### C. Fresh structurally entropy-constrained recurrent population experiment

Design a genuinely new objective in which entropy/support is constrained during optimization rather than checked only afterward. Use fresh data and a new prospective protocol; do not reinterpret `069` as passed.

This is closest to the promising current candidate but risks converting a near-threshold failure into an endless sequence of post-result repairs. It still relies on noisy outcome-only credit.

### D. Replace pairwise succession with a population/league objective

Evaluate and select by a fixed population rating or exploitability proxy, perhaps retaining multiple complementary policies rather than requiring every generation to dominate its immediate parent and every stratum.

This may address non-transitivity and seat/opponent heterogeneity but needs careful safeguards against rating overfit.

### E. Pause learning and focus on benchmark validity

Run an independent rules, population-strength, and human-calibration review before further optimization. This is appropriate if `035`'s apparent plateau might be an evaluation artifact rather than a learning limitation.

## 11. Requested reviewer output

Please provide:

1. **A one-paragraph diagnosis** of the true current blocker.
2. **A gate audit:** which formal failures are scientifically decisive, which are inconclusive, and which closures appear too broad.
3. **A ranked top three next directions**, including one recommended first experiment.
4. **A concrete first protocol:** data source, model/operator, sample size, primary metric, stopping conditions, estimated CPU/GPU needs, and what result would change your mind.
5. **A kill list:** directions that should not receive more resources.
6. **Missing evidence or questions for the user**, especially Splendor-specific issues that could change the recommendation.
7. **Confidence and cruxes:** state confidence numerically and name the assumptions on which the recommendation depends.

Do not assume that more compute is useful without identifying the bottleneck it addresses. Do not recommend lowering a failed threshold retroactively. It is acceptable to recommend a fresh experiment that changes the scientific question, provided the distinction is explicit.

## 12. Minimum files and artifacts for verification

Start with:

- `AGENT_CURRENT.md`, especially lines 1-430 and the named branch sections;
- `AGENT_HANDOFF.md` only when operational history is needed;
- `outputs/recurrent_scale_or_stop_069/result.json`;
- `outputs/broad_public_belief_resolver_confirmation_051/result.json`;
- `outputs/resolver_selfplay_distillation_052/result.json`;
- `outputs/rolling_horizon_robustness_043/gate_final.json`;
- `outputs/full_game_deep_cfr_feasibility_047/result.json`;
- `outputs/outcome_sampling_cfr_feasibility_048/result.json`;
- `outputs/one_shot_matched_rollout_005/gate_final.json`.

When an artifact path differs, use the exact path recorded in the corresponding `AGENT_CURRENT.md` section. Recompute reported metrics from raw paired rows where practical rather than relying solely on narrative summaries.
