# Independent review (2026-10-16): generalization failure, instrument calibration, and throughput

Status: **unbound review note.** Advice only. This document authorizes no experiment, training,
simulation, promotion, arena or change to any frozen protocol, code, model, data, setting or
process. It is written to be self-contained for an agent working on the project.

Date: 2026-10-16 (written ~13:40 Paris). Reviewer: independent read-only scientific review commissioned
by the owner while `structured_proposer_128` was running. A separate, earlier review note from the
same day exists at `docs/strategy/independent_review_after_large_corpus_optimization_128.md`; where
this note disagrees with it, the disagreement and the reason are stated in Section 9.

**Revision 2 (2026-10-16, afternoon), after peer review by the project model.** That model
independently reproduced the central averages from the 1,440 saved positions. This revision
softens claims that were stated too precisely or too generally: conversion ratios are a calibration
concern, not a formula; failed tests are not impossibility claims; no hard cutoff replaces the
+0.5pp gate. It also corrects the memory/disk note, adds the measured batching time breakdown,
and limits what reusing the upcoming fresh positions can establish. The disagreements that remain
are listed in Section 9.3.

Evidence tags used throughout:

- **[O]** observed directly in a primary artifact (result/report/code), with the path given;
- **[N]** new descriptive reanalysis by this reviewer of already-saved outputs (no model inference,
  no simulation, no training); scripts are reproduced verbatim in Appendix B;
- **[I]** interpretation; **[S]** speculation.

All [N] quantities are post hoc on development panels. They are hypothesis-generating, not
confirmatory, and must not be reported as prospective results.

---

## 0. Bottom line

1. The working puzzle ("why do planning improvements not become useful learned guidance on new
   positions?") is now most clearly a **cross-state generalization failure of the tested learning
   recipes**. Every learned fourth-proposer tested improves on the anchor's outsider-ranking regret
   by only about **-6% to +11%** on fresh positions, while capturing **10-98%** on TRAIN. In these
   recipes fresh skill did not move with TRAIN fit, optimization dose, teacher label precision or
   the explicit-relation features; it rose only weakly with the number of independent training
   positions. A 64-world rollout estimate captures about half of the same regret, so
   position-specific signal exists. The cause (data variety, representation, objective, or a
   combination) is not identified. [N, high confidence in the description]
2. **There is still room for a better fourth suggestion.** A fourth proposal chosen with independent
   rollout knowledge adds about **+1.3 to +1.6pp** conditional score at equal selection precision,
   matching the eight-candidate teacher. That knowledge costs extra simulations, so this shows an
   opportunity, not that a cheap learned model can capture it. [N, moderate-high]
3. **The +0.5pp conditional gate is a poor universal rejection rule for fourth-slot learners.**
   The anchor-versus-random fourth-slot swap, worth **+15.8pp in whole games** (65.796%), measures
   only about **+0.5pp** on fast-161 conditional panels (about +1.0pp on planner-arena states, one
   comparison). Other proposer comparisons tell the same story: +0.42pp conditional went with
   56.97% in games. Small conditional gains can therefore matter, and individual ±0.4pp panels
   may miss practically useful gains. [N + O, moderate; a calibration concern, not a formula]
4. The few available comparisons suggest that conversion from conditional score to whole-game score
   **differs by intervention class**: proposal swaps convert far better than breadth, equal-budget
   breadth converted about nil, and 086 root expansion converted negatively. Opponents, positions,
   budgets and estimates differ across these comparisons, so no reliable multiplier follows.
   [O + N, moderate on direction, low on magnitude]
5. **Compute scaling is not flat** in whole games: 4x worlds = 65.5% (about +111 Elo), 2x candidates
   = 56.2% (about +43 Elo). [O]
6. **Simulation throughput constrains every line**: about 215 simulated moves/s per worker, 60-90h
   arenas, 10^4-row learning sets, 224-256-game AlphaZero generations, and studies run on 1-4 of 14
   cores because desktop applications hold most of the RAM. The achievable speedup is unmeasured. [O]
7. **Recommendation:**
   - Let `structured_proposer_128` finish with its agreed criteria unchanged.
   - Report the fresh-ranking diagnostics separately, labelled as supporting evidence.
   - Do not use +0.5pp as a universal rejection rule. Weigh the size of the improvement, its
     uncertainty, its relevance to actual play and the cost of confirmation together.
   - Before the running study's fresh results exist, define and freeze a small alternative-model
     test (Section 8.1) so those positions can serve as an untouched ranking test.
   - Make the next engineering step a **bounded investigation of where simulation time goes**,
     followed by one targeted speedup prototype (input preparation first) only if measurements
     justify it. No major engine rewrite without evidence of worthwhile savings.
   - None of this argues for abandoning learning.

---

## 1. Operational alert (reported, not acted on)

[O] "Committed memory" is memory reserved by programs and backed by physical RAM (31.8 GB) plus the
page file `C:\pagefile.sys`; it is not disk capacity. Around 13:40-14:00 Paris, committed memory was
**79.1 GB of an 80.4 GB limit**, with 0.7-1.3 GB of physical RAM available. The page file is
system-managed and was growing automatically: the limit rose from 79.4 to 80.4 GB within the hour,
and the page file was about 49.8 GB. The largest consumers were desktop applications (multiple
Firefox processes, OneDrive, ChatGPT, Memory Compression), not research processes. The running
study's fitting worker (PID 4208) used about 1.3 GB working set and 2.0 GB private memory.

Consequences:

- A hard out-of-memory failure is less likely than first stated, because Windows can grow the page
  file. But physical RAM is nearly exhausted, so the system pages to disk, which slows every
  process, including the running study.
- Disk is the more concrete risk. Drive C: had 1,838 GB used and only about 24 GB free. The study
  projected about 12.5 GB of additional storage and holds itself if free space falls below 5 GiB.
  Continued page-file growth competes for that same space and could trigger the study's
  storage-floor hold.
- Memory pressure is why recent studies were frozen with 1 fitting worker and 1-4 simulation
  workers on an i5-13600KF with 14 cores / 20 threads (no GPU). The 2026-10-12 speedup review
  already measured only 35-44% CPU use with eight workers.
- Closing memory-heavy applications and freeing some space on C: are owner actions. They would
  protect the running study and would likely allow more workers for future studies, with no
  scientific change.

Do not stop, restart or modify the running study because of this note.

---

## 2. New measurements [N]

### 2.1 Data and method

Sources (saved root records only; each contains, per root, 128 selection-world WDL outcomes for every
union candidate, up to 128 disjoint assessment-world outcomes for the chosen actions, each learned
arm's full-legal canonical group scores, and the teacher's ranked/uniform extras):

| Panel | Roots | Arms with saved group scores |
|---|---:|---|
| `outputs/large_corpus_optimization_128/main/roots/` (LCO) | 720 | anchor, control (= saved large-7136), continued (28544) |
| `outputs/independent_game_scaling_128/main/roots/` (scaling) | 720 | anchor, small, medium, large (7136), control (saved BROAD) |
| `outputs/teacher_precision_512/main/roots/` | 360 | anchor, control (128-world labels), precision (512) |
| `outputs/action_card_relations_128/main/roots/` (frozen-base) | 360 | anchor, raw, relations |

Conventions:

- Common first three candidates = original 035 top-1, 035 top-2 and one uniformly random canonical
  action (`src/splendor_ai/alphazero/run_afterstate_advantage.py:149-167`, via
  `src/splendor_ai/alphazero/run_rollout_selector_gameplay.py:94`); the learned arm replaces only
  slot four (`scripts/analysis/learned_proposal_arena.py:101`).
- Teacher outsiders **O** = teacher candidate indices minus the first three = anchor fourth + the
  anchor model's next two ranked groups + two hash-uniform groups (`tools/large_corpus_optimization_128/engine.py:108-147`).
  This is the same composition as the TRAIN "recorded outsiders".
- Score = (WDL + 1) / 2; all regret and gains in percentage points (pp) of score.
- Bootstrap: resample roots (each from a distinct source game) within the 18 fit x seat x phase cells,
  2,000-4,000 draws. Simpler than the frozen protocols' 50,000-draw procedure; descriptive only.

### 2.2 A. Fresh-position outsider ranking ("fresh capture")

For each root with |O| >= 2, each model picks argmax over O of its own `learned_group_scores`.
Regret = max over O of the 128-world selection mean - the chosen candidate's 128-world mean.
Fresh capture = 1 - mean model regret / mean anchor regret. This is the same metric family as the
TRAIN acquisition guide (TRAIN anchor regret was about 6.25pp per added row; 2,489-row validation
anchor regret about 6.0pp in `outputs/full_teacher_fit_128`), so TRAIN and fresh values are comparable.
A split-half variant selects on worlds 0-63 and scores on 64-127 (unbiased for every row, including
a 64-world rollout "oracle" that chooses on worlds 0-63).

| Panel | Model | TRAIN capture [O] | Fresh capture, 128-world | Fresh capture, half-2 |
|---|---|---:|---:|---:|
| LCO | anchor regret | - | 6.418pp | 7.503pp |
| LCO | large-7136 ("control") | 80.1-83.9% | 6.5% | 6.4% |
| LCO | continued-28544 | 97.7-98.1% | 6.0% | 6.8% |
| LCO | 64-world oracle | - | - | 53.6% |
| scaling | anchor regret | - | 6.329pp | 7.403pp |
| scaling | small (720 added) | ~99% | 0.0% | 1.5% |
| scaling | medium (2,880 added) | ~97% | 5.2% | 5.6% |
| scaling | large (11,520 added) | 80.1-83.9% | 11.1% | 12.4% |
| scaling | saved BROAD control | - | 7.5% | 8.2% |
| scaling | 64-world oracle | - | - | 53.3% |
| precision | 128-world labels | - | -5.1% | -6.5% |
| precision | 512-world labels | 96.7% added | -6.0% | -5.1% |
| precision | 64-world oracle | - | - | 47.0% |
| frozen relations | raw | ~10-15% added | 5.3% | 6.2% |
| frozen relations | relations | ~10-15% added | 5.7% | 6.1% |
| frozen relations | 64-world oracle | - | - | 53.0% |

Selected paired contrasts (pp; positive = better than anchor):

- LCO restricted outsider value gain: control +0.42 [-0.19, +1.10]; continued +0.39 [-0.32, +1.10].
- Scaling restricted outsider value gain: large +0.70 [+0.08, +1.35]; medium +0.33 [-0.34, +1.01]; small +0.00 [-0.67, +0.69].
- Anchor minus 64-world-oracle half-2 regret: +4.02 [+3.29, +4.81] (LCO), +3.95 [+3.21, +4.67] (scaling).
- Random choice among O minus anchor, half-2 regret: +0.64 [+0.10, +1.16] (LCO), +0.46 [-0.07, +0.97] (scaling),
  +1.54 [+0.75, +2.31] (precision), +1.07 [+0.34, +1.85] (relations).

Readings [I]:

- In these recipes, TRAIN fit beyond about 10-15% did not improve fresh ranking: the frozen-base
  relations models (10-15% TRAIN) did as well on fresh positions as the 98% continued models. This
  is an empirical pattern for the tested recipes, not a general boundary.
- More precise labels on the same 720 states did not help; more independent states helped a little
  (small -> large: about 0% -> 9% pooled across the two panels that contain the same large-7136 model).
- The ranking signal is real at the position level (about 50% capturable by a 64-world rollout
  estimate) but the tested models did not predict it from the state/action encoding.
- How to read the percentages: they are improvements relative to the anchor's regret, among at
  most five recorded outsiders, judged against noisy 128-world estimates. They are not fractions
  of all remaining learnable skill, and they say nothing about actions outside the recorded set.

### 2.3 B. Fourth-slot headroom (three disjoint world blocks)

Proposer block A = worlds 0-41, selection block B = 42-84 (43 worlds), evaluation block C = 85-127.
Each arm selects among its candidates on B and is scored on C. "Oracle(A)" fills slot four with the
best recorded non-first-three candidate on block A (independent of B and C, so unbiased).
"Oracle(C)" uses the evaluation block and is a biased, infeasible upper reference only.

| Arm (gain vs anchor, pp) | LCO | scaling |
|---|---:|---:|
| first three only | -1.16 [-1.65, -0.72] | -0.94 [-1.34, -0.55] |
| teacher's first uniform extra as 4th ("random1") | -0.43 [-0.95, +0.07] | -0.19 [-0.65, +0.27] |
| teacher's first ranked extra as 4th | +0.12 [-0.32, +0.58] | +0.43 [-0.08, +0.98] |
| control / large-7136 | +0.41 [-0.17, +0.98] | +0.37 [-0.10, +0.87] |
| continued-28544 | +0.005 [-0.50, +0.49] | - |
| medium / small | - | +0.17 / +0.01 |
| **oracle(A) fourth among recorded** | **+1.26 [+0.77, +1.79]** | **+1.58 [+1.03, +2.16]** |
| teacher K<=8 (same 43-world selector) | +0.87 [+0.41, +1.34] | +1.60 [+1.05, +2.14] |
| all recorded candidates K<=10 | +1.24 [+0.66, +1.80] | +2.07 [+1.47, +2.63] |
| oracle(C), biased upper reference | +2.70 | +3.14 |

Reading [I]: one well-chosen fourth candidate carries as much conditional value as the whole
eight-candidate teacher at the same selection precision, which meaningfully weakens the hypothesis
that the fourth slot is exhausted. Caveats: oracle(A) uses about 6 x 42 extra rollouts of
knowledge, so it is breadth-like in cost and shows an opportunity, not that a cheap learned model
can capture it; whether its gain would convert to whole games like a proposal swap or like breadth
(Section 2.4) is unknown.

### 2.4 C. How conditional gains convert to whole games

Selection on worlds [0, N), evaluation on disjoint worlds; "random4" = first three + the teacher's
first hash-uniform extra.

| Selection N | random4 minus anchor, LCO | random4 minus anchor, scaling |
|---:|---:|---:|
| 32 | -0.76 [-1.20, -0.34] | -0.22 [-0.61, +0.16] |
| 43 | -0.76 [-1.21, -0.28] | -0.33 [-0.74, +0.10] |
| 64 | -0.78 [-1.22, -0.36] | -0.39 [-0.78, +0.00] |
| 85 | -0.63 [-1.12, -0.18] | -0.37 [-0.82, +0.10] |
| 96 | -0.69 [-1.20, -0.22] | -0.31 [-0.81, +0.20] |
| 128, official assessment (partial coverage, biased toward 0) | -0.38 [-0.71, -0.06], n=626 | -0.01 [-0.28, +0.27], n=651 |

[O] Independent official estimate on planner-arena states (`outputs/proposal_diagnostic_128/report.md`,
201 positions, 128 selection + 128 assessment worlds): random_minus_anchor **-0.997 [-2.171, -0.058]**;
treatment(gen-2)_minus_anchor +0.142 [-0.163, +0.480].

Caveat: "random4" excludes the anchor's top three outsiders, so it is slightly worse than a true
uniform fourth; the anchor-vs-random effect is therefore if anything overstated here, and the
conversion ratios below understated.

Conversion table (conditional score gain -> whole-game score gain):

| Intervention | Conditional (pp) | Whole game | Ratio | Sources |
|---|---|---|---:|---|
| 32-world selector vs raw 035 | +2.44 (+0.0489 WDL) | 95.5% (+45.5pp, ceiling-limited) | ~19 | `docs/engineering/rollout_selector_154_result.md`, `rollout_selector_gameplay_155_result.md` |
| learned vs random 4th, 32 worlds | +0.85 [+0.35, +1.42] | 64.43% (+14.4pp) | ~17 | `learned_proposal_screen_result.md`, `learned_proposal_arena_result.md` |
| learned vs 035-ranked 4th, 32 worlds | +0.42 [+0.02, +0.84] | 56.97% (+7.0pp) | ~17 | `learned_proposal_attribution_result.md` |
| learned vs random 4th, 128 worlds | ~+0.5 fast-161 [N]; +1.0 arena states [O] | 65.80% (+15.8pp) | ~16-30 | `outputs/learned_proposal_128/report.md` |
| gen-2 learner vs gen-1, 128 worlds | +0.14 [-0.16, +0.48] arena states | 51.49% (+1.5pp) [47.3, 55.7] | ~10 (n.s.) | `outputs/proposal_iteration_128/report.md` |
| 8x128 vs 4x128 (2x compute) | +1.23 [+0.66, +1.86] | 56.22% (+6.2pp) | ~5 | `outputs/proposal_breadth_128/report.md`, `outputs/proposal_breadth_arena_128/report.md` |
| 8x64 vs 4x128 (equal compute) | +0.63 [+0.02, +1.25] | 49.75% (-0.25pp) [44.9, 54.6] | ~0 (CI about -8 to +7) | `outputs/equal_budget_breadth_128/report.md`, `outputs/equal_budget_breadth_arena_128/recovery_03/report.md` |
| 086 expanded root vs 084 | positive on fresh worlds (+0.021 margin [+0.002, +0.040], 139) | 31.875% (-18.1pp) | negative | `docs/engineering/independent_confirmation_root_policy_139_result.md`, `outputs/expanded_root_resolver_fresh_champion_comparison_134/result.json` |

Also [O]: at matched compute, precision (original four candidates extended to 128-256 worlds) scored
only +0.149 [-0.094, +0.398] conditional versus breadth's +1.23 (`outputs/proposal_breadth_128/report.md`),
yet at equal compute breadth was no better than the 4x128 anchor in whole games (49.75%). And 4x more
worlds (32 -> 128) won 65.5% [60.875, 70.125] in whole games (`outputs/world_budget_160/report.md`).

Readings [I]:

- These ratios come from a handful of comparisons with different opponents, positions, simulation
  budgets and uncertain estimates. They establish a calibration concern, not a calibrated formula.
  Do not convert a specific conditional gain into a predicted win rate with them.
- For fourth-slot proposal swaps, conditional gains of a few tenths of a point have repeatedly gone
  with large whole-game effects (+0.42pp with 56.97%; about +0.5pp with 65.8%). Individual 720-root
  panels (half-width about ±0.4pp) may therefore miss practically useful whole-game gains for this
  class, and their nulls should not be read as evidence of no useful effect.
- The pooled five-panel learner-minus-anchor estimate in the earlier review note (+0.005pp,
  fixed-effect 95% CI about ±0.15pp; wider with realistic inflation for shared fits/sources) argues
  against large average gains for those recipes, but by the same reasoning it may not exclude a
  practically useful one.
- In the available comparisons, the conditional instrument favoured "more candidates" over
  "better candidates" and "more precision" relative to whole-game outcomes.
- [S] Mechanism candidate: selecting the maximum over more candidates under the shared evaluator
  (greedy-035 continuations, which assume both players then play like 035) exploits that evaluator's
  systematic errors; an endpoint that uses the same evaluator on independent worlds cannot see this.
  Alternatives (state-distribution effects, replanning) are not excluded.

---

## 3. Where the programme stands

### 3.1 Demonstrated [O]

| Result | Score | Approx. Elo |
|---|---|---:|
| 4x32 rollout planner (155) vs raw 035 | 95.5% [93.5, 97.25] | +520 (inflated: the planner simulates 035) |
| 155 vs exact 084 (156) | 65.125% [60.375, 69.75] | +108 |
| 128 vs 32 worlds, same candidates (160) | 65.5% [60.875, 70.125] | +111 |
| learned 4th vs random 4th at 32 worlds | 64.43% [59.70, 69.15] | +103 |
| learned 4th vs 035-ranked 4th at 32 worlds | 56.97% [52.49, 61.44] | +49 |
| learned 4th vs random 4th at 128 worlds | 65.80% [60.95, 70.52] | +114 |
| 8x128 vs 4x128 (2x compute) | 56.22% [51.62, 60.70] | +43 |

"Learning never worked" is false: one search -> learning -> stronger-search cycle is established.

### 3.2 Unresolved

- Repeatable learning: the only whole-game test of a second generation was 51.49% [47.26, 55.72].
- Absolute strength: no external opponent since the ten-game human panel of 2026-08-03
  (`docs/experiments/human_resolver_ceiling_calibration_074.md`; owner 8-2 vs 035, 7-3 vs the
  050/051 resolver).
- No declared compute budget, so "superhuman under a declared budget" is not yet falsifiable.
- Ceiling of the current paradigm (one-step improvement over greedy-035 rollouts).

[S] A transitive Elo chain (084 about +200 over 035; then +108, +111, +114, +43) puts the current
best configuration roughly +150 to +330 Elo above the owner. Self-play chains usually overstate, so
this is speculative, but it suggests an owner-only calibration would mostly saturate.

### 3.3 Readings currently in use that are misleading

1. **TRAIN acquisition as evidence of fresh value.** In the tested recipes it did not predict fresh
   skill (2.2). It remains a useful diagnostic, because failure to learn and failure to generalize
   are different diagnoses, but passing it is not evidence of transfer.
2. **+0.5pp conditional as a universal boundary between useful and useless proposer learners.**
   Smaller conditional gains have gone with meaningful whole-game gains (2.4). It is a practical
   investment hurdle, not a proven conversion to playing strength.
3. **"Teacher retains +1.208pp of available improvement" as learnable proposer headroom.** It is a
   breadth-type quantity, which the instrument appears to over-credit; the eight-candidate teacher's
   measured whole-game value is 56.2% at 2x compute.
4. **Fast-161 source panels as the only evaluation population.** Planner-generated positions better
   match deployment. One comparison suggests the fourth-slot signal may be larger there; that is
   uncertain, and those positions cost more to generate.

---

## 4. Hypotheses, ranked by evidence and decision relevance

### H1. The ranker does not generalize across positions at about 10^4 rows (data-starved and/or wrong inductive bias)

- Support: Section 2.2; `docs/engineering/shared_cards_151_result.md` (final training MSE about
  0.0002 vs validation 0.104-0.124); `outputs/focused_teacher_fit_128` (TRAIN 100%, validation outsider
  gain -1.22pp); `docs/engineering/fast_selector_transfer_157_result.md` (student 57.58% vs 035 while
  its 155 teacher scored 95.5%); 149/152/153 development failures.
- Counter: the first generation did learn something worth +15.8pp; large-7136 shows a small data effect.
- Discriminators: the regularized-ranker probe (representation vs signal; Section 8.1); a fresh-capture
  learning curve using many more independent positions with cheaper full-legal labels.

### H2. The instrument is miscalibrated for the intervention class

- Support: Section 2.4.
- Counter: the gen-2 arena was 51.49% [47.3, 55.7]; ratios rest on few points with wide intervals.
- Discriminator: one whole-game arena of large-7136 (best fresh capture) vs the anchor. Prediction from
  about 9% fresh capture and about +0.06pp conditional: roughly 52%.

### H3. [S] Breadth exploits systematic errors of the greedy-035 evaluator

- Support: breadth conversion about 5x and 0x; 134 vs 139; precision vs breadth conditional mismatch.
- Counter: 8x128 still gained +6.2pp; distribution/replanning alternatives.
- Discriminator: whole-game precision vs breadth at matched compute (e.g. 4x256 vs 8x128); or a powered
  continuation crossover of breadth-chosen vs anchor-chosen moves under a stronger opponent continuation
  (140 tried with 36 states and was inconclusive).

### H4. Throughput constrains every line

- [O] 155: 51,321,818 simulated moves in 8h18m04s main collection on eight workers, about 215 moves/s
  per worker, about 4.7 ms per move. 035 is a 256/128-hidden, two-layer attention scorer evaluated at
  batch 1 in PyTorch with Python feature encoding on every rollout move
  (`src/splendor_ai/bots/model_bot.py:579-700`, `src/splendor_ai/training/model.py:59-71`).
- [O] 4x128 about 66 s/decision, 8x128 about 128 s/decision single-core; learned128 arena main
  collection 63.1h; breadth arena 80.9h plus 6.1h audit.
- [O] CPU batch-16 inference prototype: collection speed 1.245x (`docs/engineering/batching_benchmark_result.md`).
  Its saved per-block metrics (`outputs/batching_benchmark_v3/timing_01/block_*/batch16/batch_metrics.json`,
  summed over 48 blocks) split the 618.6 s batch-16 collection into **313.0 s model-input
  preparation (51%)**, **180.3 s model forward (29%)** and about 125 s engine and everything else
  (20%), over 193,976 simulated positions (about 3.2 ms per position). This was a 32-world
  workload; today's 128-world workload is unprofiled.
- [I] Arithmetic on that breakdown: eliminating almost all input-preparation time would give
  roughly 2x; merely making it faster gives less. Larger gains would also require cheaper engine
  stepping and better-batched inference, which are unmeasured. A 5-10x figure is an aspiration,
  not an established opportunity.
- [O] AlphaZero line: 256 self-play games per generation (`docs/engineering/alphazero_scaled_continuation_104.md`),
  224-game training corpora (`alphazero_high_budget_distillation_114.md`, `alphazero_capacity_pilot_116.md`).

### H5. Selector noise

Real (split-half decision agreement about 56% per the earlier note; precision helps in games, 160),
but it does not explain the label-space failure in 2.2, which is independent of selection.

### H6. Distribution shift (fast-161 states vs planner-arena states)

One comparison suggests the fourth-slot signal may be larger on arena states (about -1.0pp vs about
-0.5pp for the random-fourth swap, with a wide interval). Lower priority as a cause of learning
failure (the scaling study's added rows are in-distribution for its evaluation and still generalize
weakly), but relevant to how well evaluations match deployment.

### H7. Private information / blind reserves

Low. `outputs/deck_reserve_mask_ablation_071` was powered and neutral (50.575% [48.5, 52.65]). Only
adversarial exploitation after an opponent's blind reserve, and the agent's 13x human blind-reserve
frequency gap (`docs/human_resolver_feedback_074.md`), remain open.

---

## 5. What the evidence rules out, lowers, or leaves open

Lowered:

- More optimization on the same data (fresh capture unchanged 7136 -> 28544) — strongly lowered.
- Teacher label precision on the same states (128 vs 512 worlds) — lowered.
- Loss/target/weighting variants on the same data as the main lever — lowered.
- Frozen-base explicit-relation add-ons — lowered.
- Root breadth at matched compute as a strength lever — lowered (49.75%).
- Blind reserves as the primary architecture driver — lowered.

Not ruled out:

- Learned proposals as a deployment role (first generation +15.8pp; instrument insensitive).
- Data-scaled learning with many cheap, full-legal labels.
- Representations with a genuinely different inductive bias.
- Precision scaling above 128 worlds in whole games.
- Evaluator / policy-iteration improvements (158 tested one weak student; AlphaZero was data-starved).

Not evidence either way:

- 137 is mechanism-invalid (`stratified_refill_mechanism_invalid_no_search_conclusion`).
- 134's formal "invalid" label is runtime telemetry; its 127W/1D/272L strength result stands.

---

## 6. The running experiment: `structured_proposer_128`

State [O] at 13:40 Paris: hidden supervisor PID 6312, launched 10:46 Paris, binding
`57cd1786992c123e2e8fdedf6e4cf567196588efc8bd1ab9d2bbb24b7908f454`; `progress.json` stage
`learning_fits`, 1 of 6 fits complete at 12:57 Paris. Remaining: five fits, six independent replays,
the acquisition gate, then (if passed) 54 + 2,160 fresh source games and 18 + 720 fresh positions.
Team forecast 50-80h total.

What it can establish: whether twelve hand-computed resource-relation channels change a fully
trainable proposer relative to a matched raw control with identical information, capacity, data and
schedule.

This is a legitimate, scientifically worthwhile test. Its structured features deliberately express
information already available in the raw inputs; the question is whether making those relationships
explicit helps the model learn transferable patterns. Its acquisition check is also useful, because
failure to learn and failure to generalize are different diagnoses.

Limits on what its outcomes can show:

1. **It tests inductive bias, not information or capacity.** Colour deficit = ReLU(cost - tokens -
   bonus) is a single first-layer unit over channels both arms have (card costs are in the common
   card features; tokens+discounts are the raw arm's contrast channels). The gold-adjusted shortfall
   is a second-layer ReLU over those deficits minus gold (`tools/structured_proposer_128/model.py:93-128`).
   A null therefore speaks about this explicit encoding, not about representation learning generally.
2. **Training regime.** 642,530 trainable parameters, about 79 exposures per added row and a
   95%/90% acquisition gate. In earlier recipes, fresh skill did not track TRAIN fit (2.2).
3. **Gate sensitivity.** Small conditional gains have gone with meaningful whole-game gains for this
   class (2.4), so a failed +0.5pp gate does not by itself show a useless model. Relations minus raw
   is the contrast that isolates the treatment.
4. **Evaluation population.** Fast-161 source states rather than planner-generated positions.
5. **Operational risk** from the memory and disk situation in Section 1.

Predictions [I]: both arms land at roughly 0-12% fresh capture; P(joint pass) about 3-5%;
P(relations-minus-raw paired lower bound > 0) about 10%.

Recommendation: let it finish unchanged, with its agreed success criteria. **Before its fresh
assessment begins**, record in an unbound note a prespecified, separately labelled supporting
analysis: the fresh-ranking table of Section 2.2 for anchor/raw/relations on its 720 main roots,
the half-split oracle reference, and the three-block analysis of 2.3. Fix in advance the
estimands, the paired comparisons (relations minus raw first), the bootstrap and the reporting
format. Interpret the results by weighing effect size, uncertainty and relevance to actual play,
not with a hard cutoff. The Appendix B scripts compute it from
`outputs/structured_proposer_128/main/roots/state_*.json` once they exist (arm names come from the
record's `proposals` keys). Those roots exist only if the study passes its acquisition gate.

Interpreting outcomes:

- Null with both arms acquired (expected): weighs against this explicit encoding; says nothing about
  other representations or about data-scaled learning.
- Relations convincingly above raw on fresh ranking while the planning gate fails: evidence that the
  explicit relations help transfer; worth combining with more independent positions, and worth an
  arena if the improvement is convincing.
- Joint pass: confirm with a 402-game whole-game arena before building on it.
- Acquisition failure: unlikely; this would be a failure-to-learn diagnosis and limits transfer
  conclusions.

---

## 7. Recommendation

### 7.1 Immediate, near-zero cost

1. Owner closes memory-heavy desktop applications and frees some space on C: (Section 1).
2. Prespecify the supporting fresh-ranking analysis for the running study (Section 6), before its
   fresh results exist.
3. Define and freeze the small alternative-model test (Section 8.1) before the running study's
   fresh results exist, so those positions can serve as an untouched ranking test.
4. Declare the per-move budget (seconds per move, hardware, thread count) before choosing which
   configuration to measure against it. If it is wall-time on this 14-core machine, the existing
   planner can already use many more rollouts per move through world-level parallelism.
5. For future proposer learners, do not treat +0.5pp conditional as a universal rejection rule, and
   do not replace it with a hard cutoff on the fresh-ranking measure either. A cheap first look is
   to run a new model's inference on the saved fresh panels and compute fresh ranking (no
   simulation). Decide whether to fund an arena by weighing the measured improvement, its
   uncertainty, its relevance to actual play and the cost of confirmation; a convincing smaller
   improvement can justify an arena.

### 7.2 Proposer compression: keep learning in play, do not make the same recipe the main investment

The fresh-ranking evidence (not the pooled conditional null) is the reason: in the tested recipes,
fresh skill stayed around 5-11% regardless of dose, precision or the tested features, and its
slope with independent positions was shallow. This argues against more of the same recipe, not
against learning. Keep the cheap offline screen so the line can be reopened quickly if a
representation, objective or data-scale change moves fresh ranking. [S] The prize at fixed compute
is probably bounded by something like the 8x128 teacher's whole-game value (+6pp) up to the first
proposer's (+15.8pp), but this rests on uncertain conversions.

### 7.3 Next engineering step: bounded profiling, then one targeted prototype

Design (new namespace; current engine kept as the reference implementation):

- **Stage 0, about one CPU-hour.** cProfile the unchanged `trajectory()` on three saved roots of
  today's 128-world workload (one per phase), 2 candidates x 8 worlds, one Torch thread, confirming
  recorded trace hashes and WDL. Report shares for legal-action generation/step/state copy,
  observation and action input preparation, 035 forward, other; and moves/second. Compare with the
  batching benchmark's 51% / 29% / 20% split (Section 4, H4).
- **Stage 1, only if Stage 0 shows a worthwhile target.** One targeted prototype, most likely faster
  model-input preparation, the largest measured share. A larger lockstep or compiled-engine effort
  needs its own evidence of worthwhile savings first; no major rewrite on current evidence.
- **Acceptance for any prototype.** Zero mismatches in wdl, termination reason, turns,
  `trace_sha256` and `final_state_sha256` on a frozen sample of saved continuations from at least
  three studies, including forced-pass deadlocks, repetition/no-progress cutoffs and own/opponent
  private reserves; per-worker memory no higher than now. (The batching v3 benchmark achieved
  identical traces at batch 16, so exact equality is plausible.)
- **Decision.** Adopt only if exact and the measured end-to-end saving is worth the maintenance cost.
  On the old benchmark, removing almost all input preparation would give roughly 2x, merely making it
  faster gives less, so set expectations accordingly.
- **Later, separately authorized uses if savings materialise.** (a) Under the declared budget, a
  fixed-wall-time arena of a configuration the savings afford against the current 4x128; (b) a
  data-scaled learning test with many more independent positions at lower world counts with
  full-legal labels, judged by fresh ranking and then by an arena.

Why this sequencing. Efficiency and generalization should be investigated together. A data-scaled
learning test is the most direct test of the leading hypothesis, but at current throughput it can
only extend the number of positions modestly; knowing what speedup is realistic first tells us how
large that test can be. The profile is cheap and informs every future cost. Other alternatives: an
arena of large-7136 vs anchor (expected around 52% given its weak fresh ranking, 60-80h now); an
owner-only calibration panel (worth doing if owner time allows; strong humans or independent bots
would be the more informative external test).

### 7.4 Contingency on the running result

- Null (expected): as above.
- Relations convincingly above raw on fresh ranking: the next learning study combines that
  representation with more independent positions, screened offline, confirmed by arena.
- Joint pass: first a 402-game whole-game arena of relations vs anchor.
- No fresh positions (acquisition gate failed): the alternative-model test falls back to the 2,489
  original validation rows, which no training run has optimized on (they were read only for
  diagnostics), and to whatever untouched positions a later study generates.

---

## 8. Cheapest observations that could change this recommendation, and missing evidence

### 8.1 Small alternative-model test (minutes of CPU, no simulation)

Fit a pairwise/listwise gradient-boosted ranker and an L2 linear ranker on the exact 9,461 old +
11,520 added TRAIN rows with original 128-world outsider targets (excluding the 2,489 validation rows
and all panels), using the lawful observation295 + action79 encodings plus simple afterstate features
(points after, bonuses after, per-colour shortfall to visible/reserved cards, noble distance, tokens
after, reserve count); regularization chosen only by grouped CV on TRAIN source pairs. Score fresh
ranking on the existing LCO, scaling and relations panels exactly as in 2.2; those are development
evidence.

Using the running study's upcoming fresh positions as an untouched test:

- **Freeze first.** Fix the models, features, comparison rules, statistics and reporting format in a
  written note before anyone inspects the new outcomes. Freezing the weights and deciding later
  which statistic looks encouraging would forfeit the untouched test.
- **What it can show.** Only whether the frozen models rank the *already-recorded* alternatives
  better than the anchor on new positions. A new model may prefer a move outside that shortlist,
  which has no simulated outcome. So this reuse saves simulation for a ranking diagnostic; it
  cannot establish the model's planner performance, and certainly not its full-game win rate.
- **Availability.** Those 720 positions exist only if the running study passes its acquisition gate
  (fallback in Section 7.4).

Interpretation, by weighing size, uncertainty and relevance rather than a hard cutoff:

- A clear, well-measured improvement over the neural models suggests representation or inductive
  bias is a material part of the problem, and justifies a planner-level confirmation (fresh
  positions with the new model's own full-legal proposal simulated, then an arena if warranted).
- Results in the neural models' range would add to the evidence that these encodings and data do
  not transfer the signal, without proving it unlearnable.

### 8.2 The Stage 0 profile (about one hour)

If today's 128-world workload shows a split like the old benchmark (about half in input
preparation), the targeted prototype is justified but should be expected to give at most about 2x
on its own. If the 035 forward or engine stepping dominates instead, the prototype target changes,
and the case for any large rewrite weakens further.

### 8.3 Missing evidence

- A declared compute budget (a number).
- Any external opponent since 2026-08-03: strong humans under the declared budget, or independent bots.
- A whole-game test of any learner trained after the second generation.
- Whole-game precision scaling above 128 worlds.
- An independent rules implementation or audited rule cases: every audit shares
  `src/splendor_ai/engine/env.py`. [O] A read-through found standard two-player rules (4 gems per
  colour, 5 gold, take-two only from piles of at least 4, a claimable noble is mandatory with one
  chosen, round completes after a player reaches 15, tiebreak fewest purchased cards then draw); no
  bug was found, but this is not validation.
- The fresh-capture learning curve beyond 11,520 independent positions.

---

## 9. Disagreements

### 9.1 With the current team strategy (`AGENT_CURRENT.md`, `AGENT_HANDOFF.md`)

- Reading full-TRAIN acquisition as evidence for fresh value (it remains a useful diagnostic).
- Treating +0.5pp conditional as a universal boundary for fourth-slot learners, on fast-161 panels only.
- Treating "teacher +1.208pp" as learnable proposer headroom.
- Agreement: optimization alone is a weak remedy; do not run local sweeps; the running study is a
  legitimate inductive-bias test and should finish unchanged.

### 9.2 With the earlier same-day note `docs/strategy/independent_review_after_large_corpus_optimization_128.md`

Agree:

- One of the common first three candidates is uniformly random (verified in code, Section 2.1).
- About half the teacher's gain comes from its uniform extras.
- Declare the budget; obtain external anchoring.

Disagree:

- "The single learned-proposal slot is saturated" (its H2, 0.85): Section 2.3 shows +1.26/+1.58pp
  oracle-fourth headroom; the learners fail to generalize (2.2), the slot does not fail to carry value.
- "Compute scaling in this architecture is flat": 160 (4x worlds) won 65.5% in whole games; the
  conditional metric understates precision gains.
- "+0.5pp is excluded at z >= 2.7, so close the line on pooled evidence": correct in conditional units,
  but for this class small conditional gains have gone with large whole-game gains, so the pooled
  bound may not exclude a practically useful average effect.
- "The conditional endpoint failed its only external validation": the 8x64 arena was gated at 55%,
  which was ambitious for a +0.63pp breadth gain given how weakly breadth converted at 2x compute,
  and its interval [44.9, 54.6] does not sharply contradict a small true gain. The larger concern is
  that conversion appears to differ by intervention class in both directions, including likely
  under-crediting of proposal swaps.

### 9.3 After peer review by the project model (Revision 2)

Accepted from the peer review:

- Conversion ratios are a calibration concern, not a formula; no win-rate predictions from them.
- Failed tests in specific recipes are not impossibility claims ("memorization above 10%", "not
  learnable at this data scale", "no generalization change" were overstated and are withdrawn).
- Fresh-ranking percentages are not fractions of all remaining learnable skill.
- The running study is a legitimate test; its criteria stay unchanged; acquisition is a useful
  diagnostic.
- No hard cutoff should replace +0.5pp; weigh improvement, uncertainty, relevance and confirmation cost.
- Speedup estimates were optimistic; the measured batching split implies about 2x at most from input
  preparation alone; profile first, then one targeted prototype, no major rewrite without evidence.
- Reusing the upcoming fresh positions supports a ranking diagnostic only, and only if the test is
  frozen before the outcomes are inspected.
- The "twice the signal on planner positions" claim rests on one uncertain comparison.

Still held by this reviewer:

- Individual 720-root conditional nulls are weak evidence against practically useful fourth-slot
  learners, because small conditional gains have gone with large whole-game gains for this class.
- The tested recipes' failure to transfer, not slot exhaustion, is the clearest observed bottleneck.
- Efficiency matters scientifically, not only operationally: it sets how large the most direct test
  of the leading hypothesis (more independent positions) can be.

---

## 10. Follow-up tasks proposed to the owner (none started)

1. Freeze the supporting fresh-ranking analysis and the small alternative-model test before the
   running study's fresh results exist (Sections 6 and 8.1). Documentation plus a minutes-long fit on
   TRAIN rows; no simulation.
2. Stage 0 profile of today's 128-world workload and a scoped prototype plan (Section 7.3). Run only
   with adequate memory headroom and never during the running study's simulation stage.

For all three: never touch, signal or share files with supervisor PID 6312 or its workers; edit no
frozen file; write only to new namespaces.

---

## Appendix A. Definitions

- **Conditional score:** win = 1, draw = 0.5, loss = 0 after one root decision, with greedy-035
  continuations for both players in observer-legal determinized worlds. Not prestige points, not a
  whole-game win rate.
- **Whole-game score:** score share in complete games between two planners (draws half).
- **Anchor:** original 161 fit's learned fourth proposal on the 4x128 planner.
- **Outsiders:** recorded candidates outside the common first three.
- **Fresh capture:** 1 - (model outsider regret / anchor outsider regret) on fresh positions, using
  saved 128-world selection means (Section 2.2).
- **pp:** percentage points of score. Elo approximations use 400 log10(p / (1 - p)).

## Appendix B. Reproduction scripts (read-only; no inference or simulation)

Run from the repository root with the project interpreter in isolated mode, at below-normal priority,
one panel at a time; each streams one root JSON at a time (memory stays small). Example:

```
.venv/Scripts/python.exe -I extract.py outputs/large_corpus_optimization_128 lco_summary.json
.venv/Scripts/python.exe -I analyze.py lco_summary.json
.venv/Scripts/python.exe -I conversion.py outputs/large_corpus_optimization_128 control,continued
```

Write the summary JSON outside the repository (for example a temp directory). `extract.py` fails on
panels whose assessment outcomes contain unresolved (null) WDL, e.g. `planner_state_transfer_128`;
that panel was not analysed. `conversion.py` assumes 720 roots. These are the exact scripts that
produced every [N] number above (verified by re-running `analyze.py` on the LCO summary with
identical output).

### B.1 `extract.py`

```python
"""Read-only descriptive reanalysis of saved fresh-panel roots.

No model inference, no simulation. Streams one root JSON at a time and keeps
only per-candidate block means. Writes a compact JSON of per-root summaries.
"""
import json, sys, os

BASE = sys.argv[1]
OUT = sys.argv[2]
rows = []
NROOT = len([f for f in os.listdir(os.path.join(BASE, "main", "roots")) if f.startswith("state_")])
for i in range(NROOT):
    with open(os.path.join(BASE, "main", "roots", f"state_{i:04d}.json")) as fh:
        d = json.load(fh)
    union = d["union"]["keys"]
    n = len(union)
    sel = d["selection"]["worlds"]
    assert len(sel) == 128
    W = [[sel[w]["outcomes"][j]["wdl"] for j in range(n)] for w in range(128)]
    def block(lo, hi):
        return [sum(W[w][j] for w in range(lo, hi)) / (hi - lo) for j in range(n)]
    teacher = d["candidates"]["teacher"]["indices"]
    first3 = d["candidates"]["anchor"]["indices"][:3]
    props = {}
    for arm in [a for a in d["proposals"]]:
        p = d["proposals"][arm]
        gk = p["group_keys"]
        sc = p["learned_group_scores"]
        m = dict(zip(gk, sc))
        props[arm] = [m[k] for k in union]
    ass_idx = d["assessment"]["candidate_indices"]
    ass = d["assessment"]["worlds"]
    ass_means = {idx: sum(ass[w]["outcomes"][k]["wdl"] for w in range(len(ass))) / len(ass)
                 for k, idx in enumerate(ass_idx)}
    rows.append({
        "index": i, "fit": d["spec"]["fit"], "seat": d["spec"]["seat"], "phase": d["spec"]["phase"],
        "support": d["spec"]["canonical_support"], "n": n,
        "first3": first3, "teacher": teacher,
        "proposed": d["proposed_indices"], "choices": d["summary"]["choices"],
        "arm_scores": d["summary"]["arm_scores"],
        "ranked": d["teacher_added_indices"]["ranked"], "random": d["teacher_added_indices"]["random"],
        "v128": block(0, 128), "h1": block(0, 64), "h2": block(64, 128),
        "bA": block(0, 42), "bB": block(42, 85), "bC": block(85, 128),
        "scores": props, "ass_means": {str(k): v for k, v in ass_means.items()},
    })
    del d, W, sel
with open(OUT, "w") as fh:
    json.dump(rows, fh)
print("wrote", len(rows))
```

### B.2 `analyze.py` (Sections 2.2 and 2.3)

```python
"""Descriptive analysis on compact root summaries."""
import json, sys, random, statistics, collections

rows = json.load(open(sys.argv[1]))
S = lambda wdl: (wdl + 1) / 2  # WDL -> score

def argmax(idx, val):
    # first maximum in the given index order (matches first-winner convention)
    best = None
    for j in idx:
        if best is None or val[j] > val[best]:
            best = j
    return best

cells = collections.defaultdict(list)
for r in rows:
    cells[(r["fit"], r["seat"], r["phase"])].append(r["index"])

def boot(per, B=4000, seed=1):
    rng = random.Random(seed)
    byc = [[per[i] for i in ids if i in per] for ids in cells.values()]
    out = []
    for _ in range(B):
        s = n = 0
        for v in byc:
            m = len(v)
            for _ in range(m):
                s += v[rng.randrange(m)]; n += 1
        out.append(s / n)
    out.sort()
    return out[int(.025 * B)], out[int(.975 * B)]

def mean(d):
    return sum(d.values()) / len(d)

# ---------------------------------------------------------------- A. label space
print("== A. Fresh-state outsider ranking (same metric family as TRAIN acquisition) ==")
models = tuple(rows[0]["scores"].keys())
regret128 = {m: {} for m in models}
regretB = {m: {} for m in models + ("oracle64", "random")}
gain4 = {m: {} for m in models if m != "anchor"}
osz = collections.Counter()
for r in rows:
    O = [j for j in r["teacher"] if j not in r["first3"]]
    osz[len(O)] += 1
    if len(O) < 2:
        continue
    v, h1, h2 = r["v128"], r["h1"], r["h2"]
    mx = max(v[j] for j in O); mxB = max(h2[j] for j in O)
    for m in models:
        c = argmax(O, r["scores"][m])
        regret128[m][r["index"]] = (mx - v[c]) / 2 * 100       # pp of score
        regretB[m][r["index"]] = (mxB - h2[c]) / 2 * 100
    cA = argmax(O, h1)
    regretB["oracle64"][r["index"]] = (mxB - h2[cA]) / 2 * 100
    regretB["random"][r["index"]] = sum(mxB - h2[j] for j in O) / len(O) / 2 * 100
    a4 = argmax(O, r["scores"]["anchor"])
    for m in [x for x in models if x != "anchor"]:
        c = argmax(O, r["scores"][m])
        gain4[m][r["index"]] = (v[c] - v[a4]) / 2 * 100
print("outsider-set sizes:", dict(sorted(osz.items())))
print("roots with >=2 outsiders:", len(regret128["anchor"]))
for m in models:
    print(f"  128-world regret {m:9s}: {mean(regret128[m]):.4f} pp  "
          f"capture vs anchor {1-mean(regret128[m])/mean(regret128['anchor']):+.4f}")
for m in [x for x in models if x != "anchor"]:
    d = gain4[m]
    print(f"  restricted outsider value gain {m}-anchor: {mean(d):+.4f} pp CI{boot(d)}")
print(" half-B evaluation (choice independent of half B for every row below):")
for m in models + ("oracle64", "random"):
    print(f"  {m:9s} regret on h2 {mean(regretB[m]):.4f} pp  "
          f"capture vs anchor {1-mean(regretB[m])/mean(regretB['anchor']):+.4f}")
d = {k: regretB["anchor"][k] - regretB["oracle64"][k] for k in regretB["anchor"]}
print(f"  anchor - oracle64 regret: {mean(d):+.4f} CI{boot(d)}")
for m in [x for x in models if x != "anchor"]:
    d = {k: regretB["anchor"][k] - regretB[m][k] for k in regretB["anchor"]}
    print(f"  anchor - {m} regret (h2): {mean(d):+.4f} CI{boot(d)}")
d = {k: regretB["random"][k] - regretB["anchor"][k] for k in regretB["anchor"]}
print(f"  random - anchor regret (h2): {mean(d):+.4f} CI{boot(d)}")

# ---------------------------------------------------------------- B. 4th-slot ceiling
print("\n== B. Fourth-slot headroom: proposer block A(0-41), selection block B(42-84), eval block C(85-127) ==")
arms = collections.defaultdict(dict)
for r in rows:
    f3 = r["first3"]
    bA, bB, bC = r["bA"], r["bB"], r["bC"]
    allout = [j for j in range(r["n"]) if j not in f3]
    def play(cands):
        c = argmax(cands, bB)
        return S(bC[c]) * 100
    arms["K3 first3 only"][r["index"]] = play(f3)
    p = r["proposed"]
    if p["anchor"] is None:
        for k in list(models) + ["ranked1", "random1", "oracle(A) among recorded", "oracle(C) among recorded"]:
            arms[k][r["index"]] = play(f3)
    else:
        for m in models:
            arms[m][r["index"]] = play(f3 + [p[m]])
        arms["ranked1"][r["index"]] = play(f3 + ([r["ranked"][0]] if r["ranked"] else [p["anchor"]]))
        arms["random1"][r["index"]] = play(f3 + ([r["random"][0]] if r["random"] else [p["anchor"]]))
        o = argmax(allout, bA)
        arms["oracle(A) among recorded"][r["index"]] = play(f3 + [o])
        # infeasible upper reference: oracle sees the evaluation block (biased upward)
        o2 = argmax(allout, bC)
        arms["oracle(C) among recorded"][r["index"]] = play(f3 + [o2])
    arms["teacher K<=8"][r["index"]] = play(r["teacher"])
    arms["all recorded K<=10"][r["index"]] = play(list(range(r["n"])))
base = arms["anchor"]
for k, v in arms.items():
    d = {i: v[i] - base[i] for i in base}
    lo, hi = boot(d, B=2000, seed=7)
    print(f"  {k:28s} score {mean(v):.4f}%  minus anchor {mean(d):+.4f} pp [{lo:+.4f},{hi:+.4f}]")
```

### B.3 `conversion.py` (Section 2.4)

```python
"""Conditional size of 4th-slot swaps at several selector precisions (saved data only).

Re-streams the root JSONs (one at a time). Selection on worlds [0, N), evaluation
on worlds [E0, 128) disjoint from selection. Also uses the official assessment block
where the counterfactual choice is covered (partial coverage -> biased toward 0).
"""
import json, os, sys, random, collections

BASE = sys.argv[1]
ARMS = sys.argv[2].split(",")          # learned arms present in this panel
splits = [(32, 64), (43, 85), (64, 64), (85, 85), (96, 96)]
per = collections.defaultdict(dict)    # (label, N) -> {root: score pp}
cov = collections.Counter()
cells = collections.defaultdict(list)
for i in range(720):
    with open(os.path.join(BASE, "main", "roots", f"state_{i:04d}.json")) as fh:
        d = json.load(fh)
    n = len(d["union"]["keys"])
    W = [[w["outcomes"][j]["wdl"] for j in range(n)] for w in d["selection"]["worlds"]]
    cells[(d["spec"]["fit"], d["spec"]["seat"], d["spec"]["phase"])].append(i)
    f3 = d["candidates"]["anchor"]["indices"][:3]
    p = d["proposed_indices"]
    rnd = d["teacher_added_indices"]["random"]
    sets = {"anchor": d["candidates"]["anchor"]["indices"],
            "random4": (f3 + [rnd[0]]) if (p["anchor"] is not None and rnd) else d["candidates"]["anchor"]["indices"],
            "first3": f3,
            "teacher": d["candidates"]["teacher"]["indices"]}
    for a in ARMS:
        sets[a] = d["candidates"][a]["indices"]
    def choose(idx, lo, hi):
        best = None; bv = None
        for j in idx:
            v = sum(W[w][j] for w in range(lo, hi))
            if bv is None or v > bv:
                best, bv = j, v
        return best
    for N, E0 in splits:
        for lab, idx in sets.items():
            c = choose(idx, 0, N)
            per[(lab, N)][i] = (sum(W[w][c] for w in range(E0, 128)) / (128 - E0) + 1) / 2 * 100
    ass_idx = d["assessment"]["candidate_indices"]
    A = d["assessment"]["worlds"]
    am = {idx: (sum(A[w]["outcomes"][k]["wdl"] for w in range(len(A))) / len(A) + 1) / 2 * 100
          for k, idx in enumerate(ass_idx)}
    for lab, idx in sets.items():
        c = choose(idx, 0, 128)
        if c in am:
            per[(lab, 128)][i] = am[c]
        else:
            cov[lab] += 1
    del d, W

def boot(diff, B=3000, seed=11):
    rng = random.Random(seed)
    byc = [[diff[i] for i in ids if i in diff] for ids in cells.values()]
    byc = [v for v in byc if v]
    out = []
    for _ in range(B):
        s = m = 0
        for v in byc:
            k = len(v)
            for _ in range(k):
                s += v[rng.randrange(k)]; m += 1
        out.append(s / m)
    out.sort()
    return out[int(.025 * B)], out[int(.975 * B)]

print("uncovered roots at N=128 (official assessment block):", dict(cov))
labels = ["first3", "random4"] + ARMS + ["teacher"]
for N in [s[0] for s in splits] + [128]:
    base = per[("anchor", N)]
    print(f"-- selection N={N}  (anchor score {sum(base.values())/len(base):.3f}%)")
    for lab in labels:
        v = per[(lab, N)]
        common = [k for k in v if k in base]
        d = {k: v[k] - base[k] for k in common}
        lo, hi = boot(d)
        print(f"   {lab:10s} - anchor: {sum(d.values())/len(d):+.4f} pp [{lo:+.4f},{hi:+.4f}] n={len(d)}")
```

## Appendix C. Primary sources consulted

- State and history: `AGENT_CURRENT.md`, `AGENT_HANDOFF.md` (newest entries, plus 2026-10-08 focused/full teacher-fit entries).
- Latest completed study: `docs/engineering/large_corpus_optimization_128.md`; `outputs/large_corpus_optimization_128/{report.md,result.json,fits/*/training_28544.json,main/roots/}`.
- Running study: `docs/engineering/structured_proposer_128.md`; `tools/structured_proposer_128/model.py`; `outputs/structured_proposer_128/progress.json`.
- Learning cycle and repetition: `docs/engineering/learned_proposal_{screen,arena,attribution}_result.md`; `outputs/learned_proposal_128/report.md`; `docs/strategy/next_after_learned128.md`; `outputs/proposal_iteration_128/report.md`; `docs/strategy/next_after_proposal_iteration_128.md`; `outputs/proposal_diagnostic_128/report.md`.
- Breadth and allocation: `outputs/proposal_breadth_128/report.md`; `outputs/proposal_breadth_arena_128/report.md`; `outputs/equal_budget_breadth_128/report.md`; `outputs/equal_budget_breadth_arena_128/recovery_03/report.md`; `docs/strategy/equal_budget_breadth_arena_128.md`; `docs/engineering/saved_halving_screen_result.md`; `outputs/saved_adaptive_breadth_128/report.md`; `outputs/world_budget_160/report.md`.
- Learning variants: `outputs/independent_game_scaling_128/`; `outputs/teacher_precision_512/report.md`; `outputs/action_card_relations_128/report.md`; `docs/engineering/shared_cards_151_result.md`; `docs/engineering/fast_selector_transfer_157_result.md`; `docs/strategy/diagnosis_after_158.md`; `docs/engineering/proposal_breadth_learning_128.md`.
- Programme history: `docs/strategy/independent_superhuman_review_after_134_137.md`; `docs/strategy/superhuman_information_set_search_roadmap_after_132.md`; `docs/strategy/stochastic_refill_vs_blind_reserve_architecture_correction.md`; `outputs/expanded_root_resolver_fresh_champion_comparison_134/result.json`; `outputs/alphazero_stratified_root_refill_search_137/result.json`; `docs/engineering/independent_world_root_validation_138_result.md`; `docs/engineering/independent_confirmation_root_policy_139_result.md`; `docs/engineering/rollout_selector_gameplay_155_result.md`; `docs/engineering/rollout_selector_champion_156_result.md`.
- Verification, throughput, rules: `docs/engineering/verification_policy.md`; `docs/engineering/batching_benchmark_result.md`; `docs/strategy/parallel_work_during_162.md`; `src/splendor_ai/bots/model_bot.py`; `src/splendor_ai/training/model.py`; `src/splendor_ai/engine/env.py`; `src/splendor_ai/alphazero/run_afterstate_advantage.py`; `src/splendor_ai/alphazero/run_rollout_selector_gameplay.py`; `scripts/analysis/learned_proposal_arena.py`; `tools/large_corpus_optimization_128/engine.py`.
- External calibration: `docs/experiments/human_resolver_ceiling_calibration_074.md`; `docs/human_resolver_feedback_074.md`.
- Earlier same-day review note: `docs/strategy/independent_review_after_large_corpus_optimization_128.md`.

Read-only attestation: apart from creating this file at the owner's request, no repository file was
created, edited or deleted, and no process was started, stopped or signalled. All new analysis ran
from a temporary directory outside the repository against saved outputs only.
