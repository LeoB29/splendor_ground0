# Independent Review Response (`070`)

**Reviewer:** independent agent review requested by `docs/project_review_prompt_070.md`
against `docs/project_review_packet_070.md`
**Date:** 2026-08-03, repository at commit `2fa2944` plus experiment `071`
**Status:** advisory. This document changes no gate, passes no failed experiment,
and promotes no checkpoint.

This file preserves the review's derivations. `AGENT_CURRENT.md` carries only the
conclusions. A future agent that wants to disagree with a conclusion should
re-derive the number here rather than trusting the summary.

---

## 1. Blocker diagnosis

The blocker is not credit assignment, compute, or the absence of a stronger
operator. It is that **every learning experiment was constrained to an effect
size below its own measurement resolution, and several gates demanded more
agreement than the teacher signal itself possesses.**

The safety gates cap candidates at 1--4% changed decisions -- roughly one changed
decision per game -- while the strength gates demand +2.5 to +5 points of score
share against a league whose entire spread is about 90 Elo and whose
paired-cluster noise is +/-5.8 points at 400 clusters.

Dose-versus-outcome across the program:

| operator | dose (changed decisions) | result vs `035` | reading |
|---|---|---|---|
| resolver `050`/`051` | ~62% | 69.0% share, +3.595 margin | large, real |
| rolling horizon `043` | ~every candidate turn | 50.55%, CI95 [47.75,53.40] | genuine null |
| one-shot rollout `005` | **0.89%** (115 / 12,864 turns) | 47.875%, CI95 [44.6,51.1] | uninformative |
| clean LOO actor `041` | 1.68% | 49.125% | low dose |
| ES adapter `038`/`002` | small drift | 49.375% | low dose |
| recurrent `069` gen 2 | 3.63% argmax | 52% parent, ~neutral broad | low dose |
| deck-reserve mask `071` | 7.32% | 50.575%, CI95 [48.50,52.65] | powered null |

`005` is the clearest mis-test. Scaling the resolver's dose-response predicts
about +0.3 points at a 0.89% dose; resolving that needs roughly 26,000 paired
seeds. The experiment ran 200 and reported a refutation.

---

## 2. Why `069` could not have passed

### 2.1 The entropy floor was arithmetically unreachable

The gate is absolute: `update.validation_normalized_entropy >= 0.30`
(`run_recurrent_scale_or_stop.py:291`), on a game-weight-weighted mean of
per-decision `entropy / log(legal_action_count)`.

Measured trajectory across the whole recurrent lineage:

| run | gen 1 | gen 2 | delta per generation |
|---|---|---|---|
| `062` (early weight 1.0) | 0.38640 | 0.36830 | -0.0181 |
| `064` (early weight 0.25) | 0.34288 | 0.32058 | -0.0223 |
| `069` (early weight 0.25) | 0.30934 | 0.29426 | -0.0151 |

`069` initialised from `064` generation 2 at `0.32058` and planned **three**
generations against a `0.30` floor, with two prior generations of the same
architecture each losing about `0.02`. Roughly `0.02` of headroom for three
`0.02` steps. Generation 2 breached it on schedule.

Two compounding design errors:

1. **The floor came from a different measurement regime.** It was set in `062`,
   where early decisions carried full weight and entropy read `0.386`. `064`
   introduced 4x early down-weighting (`early_turn_weight=0.25`). Mid/late
   normalized entropy runs 0.02--0.03 below the overall figure in every reported
   generation, so the reweighting alone consumed roughly a quarter of the
   headroom with no implied change in policy sharpness. The threshold was never
   re-derived.
2. **The regulariser does not target the gated quantity, twice over.** The loss
   adds `- entropy_weight * entropy` at `entropy_weight = 0.002`, computed on
   `behavior_probabilities(...)`: temperature 1.0 **plus a 5% uniform mixture**
   (`recurrent_league.py:230-234`), and **un-normalized** in nats, so 22-action
   states dominate 3-action states by about 3x. The gate measures the **raw
   masked softmax**, **width-normalized**, per decision. The uniform mix floors
   the regularised quantity, so the bonus barely constrains what is gated.

The parent gate contributes too: recomputed from the stored paired rows, pair
SD was `0.305` (gen 1) and `0.349` (gen 2), so SE `0.043`/`0.049` and about 86%
pass probability per generation for a neutral candidate, or `0.86^3 ~= 0.63`
across three. Stacked with the entropy tripwire, P(reaching the final gate) was
effectively zero.

Recomputation check: parent-gate shares recomputed from raw rows are exactly
`0.4950` and `0.5200` as reported. The artifact arithmetic is sound.

### 2.2 The real negative result in `069` is surrogate exhaustion

Held-out surrogate collapsed monotonically across four accepted generations
while movement stayed constant:

| generation | validation surrogate | validation KL | surrogate / KL |
|---|---|---|---|
| `064` gen 1 | +0.0037879 | 0.0031585 | 1.199 |
| `064` gen 2 | +0.0013962 | 0.0038272 | 0.365 |
| `069` gen 1 | +0.0006807 | 0.0039347 | 0.173 |
| `069` gen 2 | +0.0003932 | 0.0051313 | 0.0766 |

A 15.7x decay in learning efficiency while data per generation grew about 9x.
Independently, the `064`/`065`/`066`/`068` panels put the candidate at parity
with epoch zero four separate times. **That, not the entropy gate, is what
closes the outcome-only recurrent branch.** Advantages are z-scored per
(opponent, seat, split) cell and game weights sum to 1 per game, so the
surrogate is approximately scale-comparable across corpora; KL and TV are also
comparable, so the decay reflects a falling advantage/movement correlation
rather than a units artifact.

### 2.3 Power of the `069` final gate

Bootstrapping the real 200 `066` clusters up to the planned 400-cluster panel and
evaluating the exact nine sampling-noise sub-gates:

| true score-share delta | P(conjunction passes) |
|---|---|
| 0.000 | 0.093 |
| 0.010 | 0.200 |
| 0.020 | 0.344 |
| **0.025** (the gate's own minimum) | **0.431** |
| 0.030 | 0.513 |
| 0.050 | 0.815 |
| 0.080 | 0.976 |

Type-I error about 9% is fine. 43% power at its own declared minimum detectable
effect is not. The binding sub-gates at that effect size are the per-stratum
ones (3-of-4 opponents nonnegative: 0.670; no opponent below -0.05 and no CI
upper below 0: 0.791), not the aggregate. Hard per-stratum floors on 100-cluster
strata (SD about 5.8 points) are the main power sink.

---

## 3. Gates that were unachievable in principle

Measured from the `052` corpus: frozen `035`'s top-1 agreement with the
resolver's chosen action.

| split | resolved rows | `035` agreement (game-clustered CI95) | early | mid | late |
|---|---|---|---|---|---|
| train | 4,397 | 0.3181 [0.3068, 0.3293] | 0.252 | 0.300 | 0.473 |
| validation | 1,135 | 0.3269 [0.3027, 0.3511] | 0.245 | 0.307 | 0.509 |
| confirmation | 1,093 | 0.3304 [0.3060, 0.3549] | 0.285 | 0.287 | 0.509 |

The resolver's own **test-retest** agreement with itself (`055`, two independent
namespaces, same states, same budget) is **45.8%** overall: 37.5% early, 37.5%
mid, 62.5% late. So the learnable band above `035` is about **13 points**, and no
student can score above ~46% against single-realization labels.

- `054` required **40% validation best-group accuracy** -- 55% of the available
  band, first attempt, frozen features, value objective. It reached 35.42%,
  about +2.8 points or 21% of the band. Recorded as a learnability failure; at
  least as consistent with a reliability-ceiling artifact. `AGENT_CURRENT.md`
  flagged this at the time and the conclusion was not revised.
- `052` required **+5 points of target-top agreement** *while* requiring
  fallback-anchor KL <= 0.02: move decisively on resolver states, do not move at
  all elsewhere, from a warm start with logit gap <= -1 on 57% of disagreements.
  Close to mutually exclusive for one softmax head. It achieved +0.64 points,
  about 5% of the band.
- Conversely `056` rejecting 8x averaging on an early-phase floor was correct as
  written but drew the wrong lesson: 8x raised split-half agreement 44.1% ->
  63.9% and cut regret 85.4%. Evidence the *averaged* teacher is good.

---

## 4. The resolver inherits `035`'s strategic blind spot

Structural, not incidental. In
`run_paired_public_belief_resolver_gameplay.py:250-255` the resolver's four root
candidates are `ranked[:4]`, i.e. **`035`'s top four actions**. Continuations are
`035` at eps=0.10, tau=0.50, **top-k 3**. So the resolver cannot choose an action
outside `035`'s top four, and cannot observe the payoff of any line requiring
follow-through `035` will not play in rollout.

Action mix over the ten held-out human games, 271 decisions each:

| action | human | `035` |
|---|---|---|
| TAKE_TOKENS | 51.3% | 22.5% |
| BUY_VISIBLE | 6.6% | 46.1% |
| BUY_RESERVED | 18.8% | 11.1% |
| RESERVE_VISIBLE | 23.3% | 18.8% |
| RESERVE_DECK | 0% | 1.5% |

Reserve tiers:

| | tier 1 | tier 2 | **tier 3** |
|---|---|---|---|
| human, all turns | 14 | 21 | **28** |
| human, turns 0-19 | 8 | 6 | **18** |
| `035` visible reserves, all turns | 33 | 18 | **0** |

`035` reserved a tier-3 card **zero** times in 271 deterministic decisions.
Generalisation check on `data/corpus_generation1_policy035_selfplay_038`
(1,000 games, 63,622 decisions): of 8,483 reserves on turns 0-19, exactly
**2** were tier-3 visible.

Early-phase behaviour, resolver measured on its own 6,625 resolved `052`
decisions:

| agent | early tier-3 visible reserves | early deck reserves |
|---|---|---|
| human | 18 of 32 early reserves (56%) | 0 |
| resolver | **3 of 2,227 early decisions (0.13%)** | 441 of 2,227 (**19.8%**) |

The resolver reproduces the blind spot and *amplifies* deck reserving: 9.4% of
all its decisions versus `035`'s deterministic 1.5% in the human games, roughly
6x. That increase has a name -- **strategy fusion over determinizations**: each
determinization treats the drawn card as known, so averaging makes a blind pick
look informed. The sampling is information-legal; the averaging is still biased.

Consequence: resolver distillation can only produce the best local refinement of
`035`'s existing strategic mode. Approximate Elo-equivalents treating score share
as a win probability: resolver +139 over `035`; the human's ten games +241.
**Distilling the resolver perfectly would not reach the human sample.**

Caveat on the human evidence: one player, ten games, non-randomised, aware of
playing a bot. Wilson 95% on 8/10 is roughly [49%, 94%]. The *behavioural*
asymmetry (18 versus 0) is not in doubt; the *causal* claim that the tier-3
strategy is what wins is untested. That is what a scripted-teacher A/B answers.

Counter-evidence, reported honestly: the two outcome-grounded audits do **not**
confirm the tier-3 hypothesis. Extracting every resolved mistake and the tier of
the rollout-preferred alternative: ten human games, 8 mistakes, target tier up 2
/ same 2 / down 4, one transition to tier 3; disjoint 60 non-human states, 12
mistakes, up 5 / same 4 / down 3, one to tier 3. Both audits use `035`-guided
continuations, so they cannot convert a tier-3 reserve and are biased against
finding it better. 2/20 is a lower bound, not an estimate. Neither reading is
established.

---

## 5. Answers to the packet's gate-design questions

1. **Flat conjunctions after dozens of branches?** No. A conjunction of 9--22
   conditions compounds false negatives regardless of individual calibration.
   Split it: integrity stays a hard conjunction (near-deterministic, correctly
   so); learning-signal conditions become one pre-declared primary endpoint with
   the rest as diagnostics; per-stratum conditions become collapse guardrails
   (e.g. "no stratum CI upper below -0.10"), not sign requirements.
2. **Are thresholds calibrated?** No, and it is checkable. The
   55%-share / CI-lower-above-50% pattern needs >=200 paired seeds for real
   power; several gates ran at 50--100. Hard +/-0.025 per-stratum floors on
   100-cluster strata sit inside one SE.
3. **Hierarchical rather than flat?** Yes: integrity -> offline learnability ->
   one primary gameplay endpoint, with an explicit stopping rule per level.
4. **Non-transitivity underweighted?** Partly, but the deeper problem is
   **compression**. Elo-equivalents from the resolver's shares: `035` +139,
   shallow/`016` +186, `023` +230 below the resolver. The league spans about 90
   Elo. A +2.5-point gate is about +17 Elo.
5. **League rating or exploitability instead?** Only after fixing diversity. A
   rating over four correlated opponents re-measures one axis.
6. **Has `035` reuse made this self-referential?** Yes, and worse than the packet
   states. Every league member's strategic knowledge traces to one hand-written
   evaluator: `035` <- `022` <- PGS/shallow corpora; `016` same lineage; `023`
   trained from scratch but on `022`-corpus depth-1-vs-depth-1 shallow self-play;
   shallow search **is** `GreedyHeuristicBot`'s leaf evaluation
   (`search_bot.py:478`, no checkpoint in `SHALLOW_CONTRACT`). The resolver's
   prior and continuation policy are `035`; the recurrent models embed frozen
   `035`. There is no independent opponent anywhere in the ladder.

Additional structural point on champion provenance: `035` is not only
provenance-contaminated, it is built on the **295-wide legacy tensor the project
has itself deprecated**, and it is only **48-52 against `022`** -- the legality
repair produced no measurable strength change. Yet `061`--`069` freeze every
`035` parameter and add a GRU on top, and `063` then found the residual "learned
almost entirely from frozen current-state/action features" (95.84% of KL survived
history ablation). A frozen encoder that never encoded history-relevant structure
cannot supply features a history residual could exploit. The recurrent program
spent its budget on a representation that could not host the mechanism being
tested.

---

## 6. Closure classification

| direction | verdict | basis |
|---|---|---|
| resolver as **operator** | established, not closed | `050` 138-62, 69.0% CI95 [62.5,75.0]; `051` 74.5/74.5/79.0% |
| resolver -> **policy** distillation | **worth reopening** | no from-scratch student was ever trained; all three attempts warm-started from and anchored to `035`; `052a` diagnosed that mechanism; gates exceeded label reliability |
| mixed-teacher supervised imitation | decisively closed | `020` better accuracy, 13-47 in games |
| outcome-only recurrent population learning | decisively closed **on surrogate exhaustion**, not on the entropy gate | section 2.2 |
| actor-critic / AWR / trajectory returns | provisionally closed | `007` vs `008` non-replication; all low dose; `069` supplied the large-sample answer |
| counterfactual action credit (mid/late) | provisionally closed; signal real | `018` strong replicated signal; `019` Q-model failed; reusable as labels, not as a value function |
| selective / one-shot matched rollouts | **mis-tested, not closed** | `005` at 0.89% dose could not detect its own hypothesis |
| PUCT, values, visit distillation | decisively closed | s512 = s64; `036` calibration; `060` independent-search JSD only 0.00147 below search-to-prior JSD |
| rolling horizon | decisively closed | 500 pairs, 50.55%, 2/5 blocks above 50% |
| hierarchical explicit options | decisively closed | `039` significant regression, NLL delta +0.01790 CI95 [+0.00132,+0.03451] |
| full-game CFR (traversal, outcome sampling) | decisively closed | `047` node caps; `048` ESS 1.63e-4 vs 1e-2 |
| ES low-rank adapter | decisively closed | parent-relative only |
| reserve-target defect | diagnosis stands; consumer family closed | `042`/`043`/`044` closed; 20% resolved rate, +0.836, Wilson [11.8,31.8] survives |
| deck-reserve sub-class | **closed by `071`** | powered null at 7.32% dose |
| tier-3 reserve absence | **open** | untested by `071`; sections 4 and 7 |

Threshold-sensitive rejections (`041` deadlock formality, `004` 30.625% vs 30%,
`068` -0.06 vs -0.05, `066` seat-0 -0.0575 vs -0.05) should all stay failed. But
the packet over-credits them: in three of four the later evidence was itself low
dose or underpowered, and `067` showed the `066` seat effect reversed sign across
panels. Refusing to act on noise is correct; it is not evidence the direction is
wrong.

---

## 7. Ranked directions

1. **Coherent scripted teacher for the human's strategy, then behaviour-clone
   it.** A deterministic bot that takes tokens toward the highest-value
   soon-affordable tier-2/3 card, reserves high-value visible cards, never
   reserves from the deck, and converts. No search, so free to run. 200 paired
   games versus `035`. A win yields three things at once: an existence proof that
   the strategy is strong *in this engine*, the first genuinely independent league
   opponent, and a coherent teacher for the `021`/`023` recipe -- the only
   supervised recipe that ever produced a large gain here (`021`: top-1 37.1% ->
   50.2%, beat parent 67.5%). A loss is equally informative and costs an
   afternoon.
2. **Ceiling measurement: human versus resolver.** Ten games, about 3 s per
   resolver move on 10 cores, so interactively playable. Tests the sharp
   prediction in section 4. Decides whether the resolver-distillation ceiling is
   above or below human strength, which is the largest single crux.
3. **From-scratch resolver behaviour cloning.** Still the only demonstrated
   stronger operator and the from-scratch/no-anchor variant is untried, but
   demoted: do not spend the 20--30 h of collection before the ten human games,
   because section 4 predicts the ceiling may be insufficient.

Protocol reform worth more than any single algorithm: **stop gating candidates
whose movement is capped below detectability.** Any future learner must either
change >=15% of decisions and be tested at >=200 paired seeds, or declare its
expected effect size in advance and size the panel for >=80% power against it.
That is what separated `043` (real answer) from `005` (no answer).

### Sketch of the deferred resolver-BC protocol

Stage 1, no new games, about 2 h CPU: from-scratch `CanonicalEntityActionScorer`
width 64 on the existing `052` rows (4,397 / 1,135 / 1,093, game-disjoint),
label = the resolver's chosen `action_index`, hard one-hot, no warm start, no
anchor, no soft targets. Learning curve at 25/50/100% of training *games*.
Gates: beat frozen `035` agreement by >=+8.0 points with clustered paired CI95
lower bound > +3.0 (relative, so the label-reliability ceiling cancels); monotone
across data fractions; no phase regression; normalized entropy >= 0.30 **added to
the loss as the same functional that is gated** (the `069` design fix, not a
relaxed re-run). Failing Stage 1 closes the branch for about 2 h of cost.

Stage 2, conditional, 400 fresh resolver self-play games under the frozen
`049b`/`050`/`051` contract, about 24,000 rows, 170--260 CPU-hours or 20--30 h
wall on 10 workers. Primary endpoint: 200 fresh paired seeds, both seats, 400
games versus raw `035`; pass iff share >= 0.55 and paired-cluster CI95 lower
bound > 0.50. From `043`'s real per-pair SD of about 0.32, SE about 2.3 points at
200 pairs, so about 90% power against a true +8-point effect. Secondary and
diagnostic only: 100 paired seeds student-versus-resolver to report *fraction of
teacher strength recovered*, a quantity no prior experiment measured.

Falsification: Stage 1 gate 1 or 2 fails -> not learnable at pilot scale, close
permanently. Stage 2 offline passes but gameplay <= 0.52 with CI95 upper below
0.55 -> the advantage is not representable as a stationary policy at this scale,
close; do not add games. Share in (0.52, 0.55) with CI upper above 0.55 ->
inconclusive by design, one pre-declared replication only.

---

## 8. Kill list

1. Any recurrent scale, confirmation, or entropy-repaired re-run of the `069`
   family. Killed by surrogate exhaustion and four parity panels, not by the
   entropy floor. Relaxing the floor would answer nothing.
2. PUCT in every tested form: budget, leaf targets, learned values, root priors,
   visit distillation, s512, clipped n-step.
3. Rolling-horizon planning.
4. Full-game CFR by external traversal or unbiased terminal importance weighting.
5. Any further warm-start-from-`035`, anchored-to-`035` distillation.
6. One-step matched-return imitation (`042`/`043`/`044`).
7. Explicit hierarchical option representations.
8. Low-rank ES adapters.
9. Manual state annotation and training on the ten human games. Playing *more*
   games is not on this list.
10. The `071` resolver-arm ablation, despite `071`'s decision rule authorizing
    it. See `AGENT_CURRENT.md` for the reasoning.
11. GPU rental. No measured bottleneck is neural fitting; all of it is engine
    simulation and rollouts.
12. Any BGA scraping or automated access. Terms prohibit it; nothing in the
    roadmap depends on it.

---

## 9. Confidence and cruxes

**65%** that from-scratch resolver BC produces a checkpoint beating raw `035` by
>=+5 points, conditional on Stage 1 passing. **~80%** that Stage 1 passes.
**~90%** that the gate-design arithmetic above is correct. **~40%** that the
resulting checkpoint would also beat the human player -- the weakest link.

Cruxes, descending leverage:

1. Is the resolver's advantage carried by correlated multi-move sequences? If so
   a memoryless student loses it. `038` (1,136/1,278 anchors resolve within the
   next own turn) and `039` (option variable adds nothing beyond state) argue
   against, which is why this is 65% and not 45%.
2. Is the human above or below the resolver? Determines whether direction 3 can
   ever suffice. Cheapest available measurement.
3. Does about 28,000 labelled rows suffice? `023` needed 31,080 for a weaker
   teacher. Stage 1's learning curve is the early warning.
4. Is the canonical entity representation strong enough? `023` is the weakest
   league member. Mitigation: a secondary arm on the frozen `035` encoder.
5. Is the label-reliability argument right? BC over many noisy labels should
   approximate the 8x-averaged teacher (`056`). If the noise is *biased* toward
   whatever `035` seeded the belief samples with, averaging converges to a biased
   target. `058` (61.7% of early variance is the belief-batch main effect) is
   consistent with either.

---

## 10. Expert answers on record

An experienced player (the repository owner, 8-2 versus `035`) answered the
review's questions on 2026-08-03. Verbatim substance, not paraphrase:

1. The forensic diagnosis matches their experience. They won "mostly by
   reserving high value cards early in the game, with a delayed payoff in terms
   of points." **Not** cheap-tier engine building, **not** noble targeting,
   **not** denial. They "mostly ignored what `035` was doing."
2. Willing to play 10 games against the resolver.
3. No strong confidence on whether early moves are genuinely close in value, but
   considers it not unlikely.
4. Never reserved from the deck, and judges that "reserving from the deck is an
   action that a good Splendor player is very unlikely to take."
5. Turn counts: see below, computed from the replays rather than asked.
6. Cannot say precisely, but rates depth-1 shallow search "substantially below
   an experienced player," reasoning that they beat `035` convincingly while not
   being very experienced.
7. Rules confirmed: 15 points, 4 tokens per colour plus 5 gold, 3 nobles in
   two-player.
8. Agrees `035`'s early play is the weak part and its endgame is not, without
   having tracked it precisely.

Computed from all 542 rows of `outputs/human_games/replays.jsonl`: human wins
averaged **53.2 turns** (range 48--64); its two wins were both **58**. So the
human's strategy was *faster*, not a slow value grind. Observed |margin| spans
3--13, which under `clip(margin/20)` maps to 0.15--0.65 with no saturation -- the
`/20` return scaling is well calibrated, contrary to an earlier reviewer
speculation that it compressed the signal.

---

## 11. Reviewer errors corrected in the course of the review

Recorded so a future agent does not inherit them.

- Claimed the `/20` score-margin scaling compressed the training signal about
  3x. Wrong; see section 10.
- Treated the 3-of-12 `RESERVE_DECK`->`RESERVE_VISIBLE` audit transitions as
  independent support for the deck class being bad. Too strong. `071` bounds the
  population-average substitution at <=0.10 margin points per changed decision
  against the audit's +0.836 on *screened* states. Both are true; the audit
  selected suspicious states.
- Initially read `071`'s per-seat shares (0.578 / 0.4335) as a large seat
  interaction. They are algebraically the same single paired estimate restated
  and carry no independent seat information. What they do show is a first-player
  advantage of about 57% in `035` mirror play.
- Asked the expert for turn-count data that was already in the repository.
