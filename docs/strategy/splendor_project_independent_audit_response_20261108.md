# Splendor AI: independent audit of the research direction

Response to `splendor_project_independent_audit_handoff_20261108.md`. Written 8 November 2026.

This is advisory. I read files and receipts only. I changed no running process, frozen source, setting, clock or evidence, and I opened no held-out prediction or loss result from the running study.

**Labels used below:** **[F]** fact, checked in the repository or stated in an accepted report. **[I]** interpretation: a reasonable reading of the facts. **[S]** speculation: my prior, to be tested.

**What I checked beyond the handoff:** the 7 November execution protocol; the fit and loss source (`tools/splendorust_independent_data_20261107/fit.py`, `tools/splendorust_policy_head_20261104/training.py`); the original pilot postmortem (`outputs/splendorust_selfplay_postmortem_20261104_01/report.json`, `targets.json`); the one-pass strength summary; the cached upstream `CHAMPION.md`, `README.md` and `STRENGTH_CHAMPION.json`; and the public upstream `EXPERIMENTS.md` and `STRATEGY.md` at the pinned commit, read through a fetched summary rather than verbatim.

---

## 1. Bottom line

1. **Keep the foundation, change how learning is tested.** SplendoRust plus search is clearly the right base: the search-scaling gains are robust. The learning programme needs adjusting, not replacing.
2. **The three learning nulls tell us little about whether learning works.** **[F]** The interventions closed only about 2–3% of the measurable gap between the raw network and its 512-search teacher (§3.1). **[F]** A 400-game live gate needs a true gain of about +35 Elo to have even odds of passing. **[I]** A null was the expected result whether or not the recipe can improve the network.
3. **Value-target noise is the most neglected suspect.** **[F]** The joint loss weights a value regression toward ±1 game outcomes equally with the policy loss. These outcomes come from only a few hundred games, and every position in a game shares the same one. **[F]** In the pilot, value outputs moved by 0.093 on average with no held-out value gain (32 games better, 32 worse). **[I]** Under a Q-driven search with cpuct 0.4, that drift may cancel policy gains.
4. **The running study's primary endpoint is the one least likely to show a breadth effect.** **[I]** The study asks whether data from more independent games helps. But it is value targets that are fully correlated within a game. Policy targets are separate per-position search results. Value MSE is the secondary endpoint.
5. **The most important unknown is how far away the goal is.** **[F]** There is no human or independent-strong-engine anchor. **[S]** EF8A at 512 simulations may already be at or above expert-human level, or far below it. Each case implies a very different programme, and finding out costs little compute.
6. **Next, at most two experiments:** (A) one adequately sized learning cycle on the corpus that has already been collected, with value anchoring, judged first by a calibrated decision-quality proxy and then by a concurrent, adequately powered live test. (B) A small human-calibration pilot. Both are specified in §6.

---

## 2. Where the programme stands

### Strong evidence

- **[F] Search scaling with fixed weights works and keeps working:** 16→32→64→128→256→512 each produced a significant gain over the previous budget (56–64% per step; 256→512 = 56.5% [51.9–61.1]). Taken as a chain, 512 is very roughly **+120 Elo** over 128 (assuming the steps add up approximately).
- **[F] SplendoRust is far stronger than the previous local planner** (60–0, with the provenance amendments disclosed) and than the two PPO exports (0–60 each).
- **[F] The corrected-rules pipeline is unusually well verified.** Corrected catalogue, nobles, payments, privacy, value orientation, exact replays and independent acceptance are all in place. No result is likely to be explained by rule or harness bugs.
- **[F] Small fine-tunes improve held-out policy cross-entropy slightly and reproducibly** (about 0.6–0.8%).

### Weak assumptions

- **That a 400-game gate at 55% can detect what a 3,072-example, 96-step update could plausibly deliver.** See the power table in §3.2.
- **That lower policy cross-entropy means better 128-search decisions.** **[F]** In the pilot, CE improved about 0.7%, but agreement of the raw network's top move with the teacher's top move *fell* from 70.26% to 69.54%. No measurement links offline change to decision change.
- **That outcome-regression value targets from roughly 200–1,000 games add information to a strong pretrained value head.** **[I]** More likely they mainly add noise.
- **That the parent comparison chain approaches the goal.** It shows direction within one model family. It shows neither absolute level nor transfer to humans.
- **That the learning problem is new to this project.** **[F]** The upstream SplendoRust records say the first one-hot update improved its screen, and that later target and iterative controls did not establish further gains. Upstream also lists the lesson "prediction loss alone should not select an agent." The project may be repeating upstream's plateau without knowing why upstream stalled.

### The three most consequential gaps

1. **No external anchor:** no human results, and no strong independent engine under our rules. Without one, the goal cannot be measured, and effort cannot be split rationally between learning, speed and demonstration.
2. **No calibrated bridge from offline metrics to play.** Every live test costs about half a day to two days and can resolve only large effects. A cheap decision-level proxy, validated against the known search-budget gains, would turn most live tests into confirmations rather than searches.
3. **Value-target design and the EF8A training recipe are unexamined.** We do not know upstream's data scale, teacher, learning-rate schedule or value targets. So we cannot judge whether LR 1e-5, 96 steps, 3,072 examples or ±1 outcome targets are a sensible scale or kind of intervention.

---

## 3. Two quantitative observations that drive this audit

### 3.1 The interventions were tiny relative to the available signal

From the pilot postmortem's per-game figures **[F]**:

- On held-out games, the parent's policy cross-entropy against the 512-search teacher is about **1.1–1.25 nats**. The teacher's own entropy is about **0.78–0.97 nats**. The reducible gap (the KL divergence) is therefore roughly **0.3 nats per position**.
- The pilot's roughly 0.7% CE improvement is about **0.008 nats**: around **3% of that gap**. The one-pass fits (0.634%) are similar. Not all of the gap is learnable, since some of the search's information may not be representable without lookahead, so the share of the *learnable* gap closed is larger but unknown.
- Other pilot measures: whole-network relative L2 displacement was **0.71%** after 668 updates. The raw top move changed on **9.7%** of positions. The teacher is sharp: average top-move probability **0.71**, effective support **2.6** of about 21 legal moves. The raw network's top move differs from the teacher's on about **30%** of positions, so there is real signal to distil.
- **[F]** AdamW at 1e-5 moves each parameter by at most about 1e-5 per step. The 96-step fits are therefore roughly 7× smaller perturbations than the 668-step pilot.

**[I]** These studies tested whether a near-infinitesimal nudge toward the teacher changes play detectably. They did not test whether distilling 512-search targets into EF8A works.

### 3.2 The live gate needs a large effect

The reported paired interval at 400 games (±4.5 points) implies a standard error of about **2.3 points**.

| True score vs EF8A | Elo | P(passing the 55% gate at 400 games) | Games for 80% power to get lower bound > 50% |
|---:|---:|---:|---:|
| 52% | +14 | ~10% | ~4,100 |
| 53% | +21 | ~19% | ~1,850 |
| 55% | +35 | ~50% | ~660 |
| 57% | +49 | ~81% | ~420 |

(Normal approximation, using per-game variance implied by the observed paired interval. The seat criterion lowers pass rates slightly further.)

**[I]** +35 Elo is about 78% of the entire 256→512 search doubling. A one-pass update on 5% of a corpus should not be expected to deliver that. The gate mixes two questions: *is there an effect?* and *is it big enough to adopt?* Separate them, and size the sample for the smallest effect worth knowing about. I suggest about +20 Elo (53%).

---

## 4. Bottleneck ranking

| Rank | Hypothesis | Evidence for | Evidence against or missing | Observation that would discriminate |
|---|---|---|---|---|
| 1 | **Intervention size and measurement sensitivity** (a design problem, not a learning limit) | Only ~3% of KL closed; live gate resolves only ≥~5-point effects; all three nulls fall inside intervals that also include worthwhile gains | Lack of a gain at 668 steps is weak evidence against | Full-corpus fit: does KL closure rise substantially (≥15%)? Does a calibrated proxy move? |
| 2 | **Value-target noise and policy/value interference through the shared trunk** | Value drifted 0.093 with no MSE gain; equal loss weighting; the effective value sample size is the number of games, not positions; the multi-pass diagnostic degraded joint held-out losses | Policy-only (value frozen) was also null, but that ablation trained only 0.44% of parameters | Same full-corpus fit with the value output anchored to the parent's value predictions, compared with outcome-MSE value |
| 3 | **Weak link from CE to decisions under search** | Top-move agreement fell while CE improved; at 128 simulations, search may absorb prior changes | Not measured | How often does EF8A at 128 simulations choose a different move from 512 (or deeper)? Is the disagreement concentrated in near-ties? |
| 4 | **Target quality** (raw visit counts at cpuct 0.4 with dynamic FPU; outcome values) | Visit counts carry search artefacts; KataGo-style target pruning, Gumbel "completed-Q" improved-policy targets and z/q value mixing exist for this reason | Not tested | After rank 2, compare against improved-policy or Q-mixed targets on the same corpus |
| 5 | **Limited independent games (the current study)** | The diagnostic showed breadth reduced deterioration in a multi-pass regime | At one pass, policy targets are already per-position; any breadth effect should show mainly in value | The running study and a data-scaling curve on the collected corpus |
| 6 | **Capacity or representation** | Nothing direct | The network still has a ~0.3-nat gap to the teacher and was barely moved | Only if training-set KL plateaus well above the teacher's entropy at a high learning rate and long training |

---

## 5. The running study

### Value for cost

- **[F] Its main asset is the corpus:** 1,024 independent training games, about **55–57k** search-labelled positions, plus 128 held-out games, all under corrected rules. That asset outlasts the study, and the study itself trains each fit on about 5% of it.
- **[I] The primary contrast (192×16 vs 1,024×3 at a fixed 3,072 examples) does not match a real allocation decision.** Taking 3 or 16 positions from a game that has already been played costs almost nothing. In practice you would always train on every position. The real decisions are *how many games to collect* and *how to use them*. A data-scaling curve on the collected corpus (128/256/512/1,024 games, all positions, fixed passes) answers that directly, offline, at near-zero marginal cost.
- **[I] The endpoint choice matters.** Policy CE is the primary endpoint. If "independent experience" is a constraint, value MSE is where it should show most clearly. Read the secondary value result with that in mind.
- **The missing live narrow arm:** this does not materially weaken the decision. The live arm is underpowered for a broad-versus-narrow difference anyway, and the offline contrast is where breadth could show.
- **What a null can separate:** very little. It is consistent with "breadth does not matter" and with "the update is too small for anything to matter".

### My prior, stated in advance for calibration **[S]**

- Policy CE broad-minus-narrow: small, under 0.3% of CE in either direction. About a 60% chance that broad is nominally better.
- Value MSE: more likely than policy to favour broad.
- Live: about an 80% chance of failing the gate, with a score most likely in the 48–54% range.

### Conditional interpretation (none of these should trigger promotion or more samples automatically)

| Outcome | Reading | Next step |
|---|---|---|
| **Broad better offline and a live gain** (≥55%, lower bound >50%) | Surprising given the power. Suggests within-game correlation genuinely limited learning and that small updates matter more than expected. Still does not show broad beats narrow in live play. | Confirm with a larger concurrent sample before any promotion. Run experiment A, which becomes more promising. Check the gain at 512 descriptively. |
| **Offline gain, no live gain** (most likely) | Breadth helps generalisation per example, but the offline change is too small to see in 400 games. | Do not collect more games yet. Run experiment A: use the whole corpus and measure through a calibrated proxy. |
| **Inconclusive offline** | Breadth does not matter at this tiny update size. This is uninformative about data scaling. | Experiment A, plus the offline data-scaling curve. |
| **Deterioration** (broad worse offline, or live upper bound <50%) | Implicates value noise or interference (more distinct noisy outcomes, larger drift). | Experiment A with value anchoring as the primary arm. Add trunk-plus-policy training with the value anchored and the value head frozen. |

---

## 6. Recommendation and the next two experiments

**Adjust; do not replace.** Keep SplendoRust, the corrected-rules chain, EF8A as the reference, and the running study to completion. Change the learning programme from small controlled perturbations judged by underpowered 400-game gates to **one properly sized learning cycle, with its value target controlled, judged by a validated proxy and then a powered live test**. Get an external anchor in parallel.

**Alternatives considered:**

- **More collection or scaling now:** rejected until one cycle shows a gain.
- **An iterative AlphaZero loop now:** premature for the same reason.
- **A wholesale pivot to search and efficiency:** this is the strongest fallback (see §7.6).
- **An architecture change:** no evidence that it is the constraint.
- **Resuming external-engine work:** low priority. The one known near-par independent engine is the lyquentxy AlphaZero, which upstream beat 58.85% at equal simulations, but it uses a different ruleset.

### Experiment A: full-corpus distillation with value anchoring, using a calibrated proxy

**Hypothesis.** Training on all of the roughly 55k training-split positions (one pass is about 1,700–1,800 updates at batch 32), with the value output anchored so it cannot drift on noisy outcomes, closes a materially larger share of the teacher KL (≥15%). It also improves 128-search decisions enough to produce a live gain of at least +20 Elo.

**Stage A0: validate the proxy (no training, few hours).**

- Take about 2,000 positions from the 128 held-out games, using the existing outcome-blind selection.
- Build a reference by running EF8A with a deeper search than the teacher: 2,048 simulations. Use 6,400 if affordable. Record the reference's chosen move and per-action Q values.
- Proxy metrics for any player: (i) agreement with the reference's move; (ii) mean reference-Q regret of the chosen move.
- **Positive controls:** score EF8A at 64, 128, 256 and 512 simulations. The proxy is valid only if it orders them correctly and its gaps roughly track the known live gains (56.1%, 61.0%, 56.5% for successive doublings). This gives an exchange rate from proxy change to expected score change.
- If the controls fail, the proxy is invalid. Learning that is cheap and itself informative.
- Caveat: an EF8A-based reference grades with EF8A's own value function. Deeper search dilutes this, and live games remain the final arbiter.

**Stage A1: fits (training split only; nothing tuned on the test split).**

- (a) **Joint, outcome value:** full corpus, current loss, one pass. Learning rate chosen on the **64-game validation split** from {1e-5, 3e-5, 1e-4}.
- (b) **Joint, anchored value:** same, but the value term regresses toward the parent's own value predictions on each position, or a fixed mix such as 0.25·z + 0.75·v_parent. Choose the variant before running.
- Use two seeds per arm at the selected learning rate.
- Measure test-split KL closure, value MSE, and the proxy at 128 simulations (average over 2–3 search seeds).
- Contrast (b) vs EF8A is primary. (a) vs (b) isolates the value-target effect.
- Using a validation split for the learning rate is a legitimate prospective design for a *new* study. It does not alter the running protocol.

**Stage A2: live test (only if the proxy passes).**

- The arm chosen in advance, (b) unless (a) is pre-declared, plays EF8A at matched 128 simulations.
- **Run games concurrently**, after a short qualification showing that responses under load stay well inside the 9–10 s guards. At fixed simulation counts, outcomes do not depend on wall-clock time; only timing statistics do.
- 2,000 games in 1,000 seat-swapped pairs, with pre-registered group-sequential looks at 1,000 and 2,000 (for example, O'Brien–Fleming boundaries for success or futility).
- Run a descriptive check at 512 vs 512 (about 400 games) for any passing candidate.

**Success criteria (independent and pre-registered):**

- KL closure ≥15% of the reducible gap.
- Proxy gain ≥25% of the 128→256 proxy gap (an expected score of about 53% or better).
- Live: 95% lower bound >50%, both seats ≥50%, and a point estimate ≥53%.

**Rough cost:**

- Fits: under an hour each, extrapolated from "minutes" for 96 steps; about 8–10 fits in total.
- Proxy: about 2–3 hours for a 2,048-simulation reference on 2,000 positions with 8 clients. Each candidate takes minutes.
- Live: about 8–15 hours for 2,000 games if concurrency gives 4–6× the current sequential rate of about 40 games per hour. That rate is unmeasured, so measure it first.
- About **1–1.5 machine-days with no new collection**.

**Decision rule:**

| Result | Meaning | Next |
|---|---|---|
| All three criteria pass | First working cycle | Run a second cycle with the new network as teacher (fresh games); require ≥+20 Elo per cycle to continue |
| KL closure ≥15% but proxy flat | CE does not transfer to decisions | Redesign targets (improved-policy or Q-based) and test on the same corpus. Not more games. |
| KL closure <10% even at 1e-4 over a full pass | The network struggles to fit the targets | Check how much of the gap is irreducible (fit on training positions and look for a plateau). Only then consider capacity. |
| Proxy passes, live fails at 2,000 games | The proxy is miscalibrated | Re-weight toward live tests and investigate where the proxy and play diverge |
| (b) clearly beats (a) | Value noise or interference was a real constraint | Adopt value anchoring or bootstrapped value targets as standard |

**What would change my recommendation:** an informative result from the running study that contradicts §3 (for example, a clear live gain from the 96-step broad fits). In that case the update-size argument is wrong, and data breadth deserves priority.

### Experiment B: human-calibration pilot

**Hypothesis.** EF8A at a declared practical setting (128 simulations, or 512 with the time disclosed) already scores ≥60% against strong human players.

**Design.**

- Recruit 3–5 consenting strong players: high-rated Board Game Arena players or tournament-level players, reached through community channels, possibly paid.
- Play about 40–60 games in total, seat-balanced, under corrected base two-player rules, at the players' normal pace.
- Use the existing GUI (`run_gui.py` / `src/splendor_ai/gui.py`) after verifying that it runs the current SplendoRust player under the corrected rules. Otherwise build a minimal interface.
- Separately, ask the upstream author whether the public web bot ("Beat the Bot!") has collected human results. The README invites reports.
- **Do not** run a bot against unknowing people on any platform. That breaks platform terms and the consent that the evidence depends on.

**Criteria and decisions.** About 50 games gives roughly a ±14-point interval. That is enough to separate "dominant" from "competitive" from "behind", not to prove superhuman play.

| Score vs strong humans | Decision |
|---|---|
| ≥70% | Main remaining work is formal demonstration (larger expert panel) plus 1-second engineering. Learning research becomes optional. |
| 45–70% | Strength work remains the priority, now with a human-calibrated target. Study the human wins. |
| <45% | The model family has blind spots that self-play does not expose. Analyse losses for exploitable patterns, and add diverse or adversarial opponents before more self-distillation. |

**Cost.** Mainly organisational: about 25–40 hours of human play and a few days of setup. Negligible compute.

---

## 7. Answers to the remaining questions

### 7.1 Operational definition of the goal (Q1)

**Proposal:**

- **Rules:** corrected base two-player rules, bound to the catalogue hash. Lawful observation only.
- **Compute:** declared in two tiers: practical (≤1 s mean, ≤3 s max on the stated machine) and research (stated budget).
- **Opponents:** a panel of ≥5 consenting experts (for example, top-100 Board Game Arena Splendor ratings or championship finalists), each playing ≥20–30 seat-balanced games, and allowed repeated games to adapt and probe for exploits.
- **Criterion:** pooled score ≥60% with a 95% lower bound above 50%, and no expert with a point estimate above 50%.

**[S]** Because Splendor has substantial luck, a clearly superhuman program may score only around 65–75% against top humans. About 150 games can show that. Showing a marginal edge (55%) would take about 600 games, which is not feasible with humans. In practice, superhuman play can be shown only with a clear margin.

**How much a chain of parent comparisons can justify:** direction, not level. Gains against a parent can be partly non-transitive, and Elo only adds up approximately. Re-anchor periodically against a fixed panel: EF8A at 128, EF8A at 512, the strongest independent engine available, and humans.

### 7.2 Measuring the strength/cost trade-off (Q6)

- Use matched simulations for learning experiments; it isolates network quality.
- Use matched wall-clock for deployment claims.
- **[S]** Network gains at 128 often persist at higher budgets but can shrink. A descriptive 512 check for any passing candidate is cheap insurance.
- Further search scaling (1,024 vs 512) is predictable and not deployable. It is worth running only to size a stronger teacher.

### 7.3 Generalisation (Q7)

- A league of historical local agents would expose little: they are far weaker, and non-transitivity matters mainly between similarly strong agents.
- Humans (experiment B) are the most informative external test.
- Bring back independent-engine work only for an opponent that is near-par under our rules or adaptable to them.

### 7.4 Engineering opportunity cost (Q8)

**[F]** The current study is forecast at 48–60 hours, with up to 3,456 collection reproductions, a 12-hour independent reader and a historical evidence forecast of 75–90 GiB. **[I]** That level of verification fits a promotion decision, not a screen whose live arm has about a 20% pass chance even at a true +21 Elo.

**Prospective tiers** (keep the corrected-rules requirements in every tier; do not change the running protocol):

| Tier | Contents |
|---|---|
| 0 (screen) | Offline metrics and the proxy. Rule-source verification, legality of every saved move, hashes. Hours. |
| 1 (live screen) | Concurrent matched-simulation games; full legality and outcome checks; sampled exact replays (about 1–2% of decisions); automated acceptance. |
| 2 (confirmation and promotion) | The full current protocol. |

The largest single throughput gain is probably **concurrent strength games**. I found no documented scientific reason in the protocol for running them sequentially, beyond timing statistics.

### 7.5 Game-model limitations (Q9)

None is a demonstrated correctness problem for current results. All are untested strength opportunities:

- **Atomic pre-reveal choices:** first measure how often a payment, return or noble choice with more than one distinct option comes before a reveal. If it is rare, park it.
- **Three-world shared-tree search:** six worlds gave 50.5%, so this stays parked.
- **Deadlock conventions:** rare; no evidence they matter.

### 7.6 When to change course (Q10)

- **Continue learning** if experiment A passes. Then require ≥+20 Elo per cycle at confirmation to keep iterating.
- **Revise targets and optimisation** if KL closure improves but the proxy and live play do not.
- **Prioritise search and efficiency** if experiment A fails all criteria, or if learning has had about one week of machine time without a confirmed live gain. Search gains are proven: making 512 simulations fit about 1 s (now 2.37 s mean) means a 2.4× speed-up, worth about +45 Elo at the practical budget. Candidate routes: profiling, batched or quantised inference, GPU inference latency rather than collection throughput, or distillation into a smaller, faster network.
- **Revisit architecture** only if the network cannot fit the teacher on training positions despite a high learning rate and long training.
- **If experiment B shows ≥70% against strong humans**, move the main effort to demonstration and speed, whatever the learning results.

---

## 8. Missing information

### Essential to this recommendation

1. **Any human-strength anchor**, and whether the owner can reach strong human players at all.
2. **The upstream EF8A training recipe:** data scale, teacher search and budget, learning-rate schedule and final learning rate, value targets. Also the details of upstream's failed "later target and iterative controls", to avoid repeating them.
3. **Whether the corpus stores root search values (Q).** I did not find a stored root value in the collection source. If it is absent, parent-value anchoring is the available value option without recollecting.
4. **Measured throughput and guard margins for concurrent 128-simulation strength games.** This sets the cost of experiment A's live stage.

### Useful but not essential

- How often 128-search and 512-search choose different moves, and whether the disagreements fall on near-ties (the headroom for policy distillation at 128).
- A latency breakdown at 128 and 512 simulations (inference, tree work, overhead).
- How often pre-reveal auxiliary choices are non-trivial.
- The cause of the 35→62 games/hour throughput change.

---

## Appendix: sources for figures not in the handoff

| Figure | Source |
|---|---|
| KL arithmetic: per-game parent CE ~1.1–1.25 nats, teacher entropy ~0.78–0.97 | `outputs/splendorust_selfplay_postmortem_20261104_01/targets.json` |
| Top-move agreement 70.26%→69.54%; value shift 0.093; 32/32 value games; L2 displacement 0.71%; top-move change 9.7%; teacher top probability 0.713; effective support 2.57 of 21.1; all 668 updates clipped | `outputs/splendorust_selfplay_postmortem_20261104_01/report.json` |
| Loss: masked log-softmax CE on visit targets plus MSE of two tanh signed values against ±1/0 outcomes, equal weight; AdamW 1e-5 | `tools/splendorust_policy_head_20261104/training.py` (`loss_components`, `make_optimizer`) |
| Champion search settings (PUCT, cpuct 0.4, root noise 0, dynamic FPU, three worlds); upstream 20,000-game confirmation | `outputs/external_opponent_search_20261018_02/splendorust/research/STRENGTH_CHAMPION.json` |
| Upstream "later controls did not establish further gains"; "prediction loss alone should not select an agent" | Upstream `EXPERIMENTS.md` and `STRATEGY.md` at commit `1828507`, read through a fetched summary, not verbatim. Verify before relying on them. |
