# Independent Programme Review After `large_corpus_optimization_128`

Date: 2026-10-16
Status: **unbound review note.** Read-only findings and recommendations.
This document is **not execution authorization**. It starts no experiment, promotes
nothing, and changes no frozen protocol, code, model, data or setting.

Reviewer scope: an external scientific review of the whole programme against its
stated objective — demonstrable superhuman playing strength under a declared
computational budget — commissioned while `structured_proposer_128` was running.

---

## 0. Provenance, method and read-only attestation

### 0.1 What was done

- Read `AGENT_CURRENT.md`, `AGENT_HANDOFF.md`, the completed
  `large_corpus_optimization_128` protocol/result/report, the running
  `structured_proposer_128` protocol and its live progress artifacts, the
  134/137 review, the superhuman roadmap after 132, the stochastic-refill
  correction, the verification policy, and the results of the learned-proposal,
  breadth, equal-budget, teacher-precision, shared-card, relations,
  control-variate, continuation-crossover and human-calibration lines.
- Verified every headline number quoted below against primary artifacts
  (`result.json`, `report.md`, `final_status.json`) rather than handoff prose.
- Ran **new descriptive analysis on already-saved outcomes only**. No model
  inference, no simulation, no new rollouts, no training, no bootstrap over new
  data, no test run.

### 0.2 Read-only attestation

No file in the repository was edited or deleted. No process was started, stopped
or signalled. The intentionally dirty worktree and all frozen predecessors are
preserved. The only file created is this review note, in `docs/strategy/`, which
by project convention is unbound.

At the time of review the running study was healthy: supervisor PID 6312,
`outputs/structured_proposer_128/progress.json` = `{"stage":"learning_fits",
"completed":0,"total":6}`, `fits/fit_161001/raw/checkpoint_07136.pt` present at
11:51 Paris against a 10:46 launch — i.e. roughly one quarter of the first of six
fits. `runtime/stderr.txt` empty, no failure/abort/hold marker.

### 0.3 Data source for the new analysis

All quantities marked **[new]** come from
`outputs/large_corpus_optimization_128/main/roots/state_NNNN.json`
(720 roots). Each root record contains, for every union candidate:

- `selection.worlds[w].outcomes[j]` — WDL in {-1, 0, +1} for all 128 selection
  worlds and all 8-10 union candidates;
- `assessment.worlds[w].outcomes[k]` — WDL for all 128 **disjoint** assessment
  worlds, for the (at most four) distinct actions the arms chose;
- `candidates.<arm>.indices`, `teacher_added_indices.{ranked,random}`,
  `proposals.<arm>.learned_group_scores`, `summary.*`.

That is enough to reconstruct every arm's decision, to re-score counterfactual
candidate sets, and to measure the evaluator's sampling noise directly — all
without re-simulating anything. A consolidated verification script is in
**Appendix B**.

### 0.4 Reconstruction check

My reconstruction of the sealed teacher arm reproduces
`teacher_minus_anchor` as **+1.2033 pp** against the official **+1.2077 pp**
(`outputs/large_corpus_optimization_128/result.json`). The 0.004 pp residual is
my simplified ordered-tie rule. All other new quantities use the same machinery,
so they inherit that accuracy.

### 0.5 Statistical caveats that apply throughout

1. Counterfactual arms re-scored from the sealed panel are **post-hoc**
   reanalyses of a confirmatory panel. They are hypothesis-generating. They are
   not prospective confirmations and must not be reported as such.
2. Where a counterfactual arm's chosen action was *not* among the four assessed
   actions, its score is unavailable. Coverage is stated for every such
   quantity; several are exactly covered (0 uncomputable) by construction and
   this is proved in §1.2 note (a).
3. Bootstrap intervals below resample **whole source games within the 18 fixed
   fit x seat x phase cells**, matching the frozen protocol, 20,000 draws.
4. Conditional scores are win=1, draw=0.5, loss=0 from a single decision
   followed by greedy-035 continuations in observer-legal determinized worlds.
   They are **not** Splendor prestige, Elo, or full-game win rates, and §1.2(c)
   shows they do not reliably predict them.

---

## 1. Where the programme stands

### 1.1 What is demonstrated

| Result | Primary evidence | Assessment |
|---|---|---|
| Planner >> raw 035 | 155: 95.5% over 400 games | solid |
| 155 > 084 | 156: 65.125% [60.375, 69.75] | solid |
| Learned 4th proposal > **random** 4th, 32 worlds | `docs/engineering/learned_proposal_arena_result.md`: 64.4279% [59.7015, 69.1542] | solid |
| Learned 4th > **035-ranked** 4th, 32 worlds | `docs/engineering/learned_proposal_attribution_result.md`: 56.9652% [52.4876, 61.4428] | solid |
| Learned 4th > random 4th, **128** worlds | `outputs/learned_proposal_128`: 65.7960% [60.9453, 70.5224] | solid |
| 8 candidates > 4 at **2x** compute | `outputs/proposal_breadth_arena_128`: 56.2189% [51.6169, 60.6965]; 11,313,152 vs 5,952,256 continuations; 127.87 s vs 66.05 s per decision | solid |
| Breadth > precision at equal continuation count (conditional) | `outputs/proposal_breadth_128`: breadth-precision +1.0829 [+0.5089, +1.7166]; breadth-anchor +1.2316 [+0.6608, +1.8620]; precision-anchor +0.1487 [-0.0944, +0.3982] | solid as a conditional result; see §1.2(c) |

**"Learning never worked" is false.** One learned-proposal generation produced a
large, replicated, full-game improvement, at two separate world budgets, against
two separate controls.

### 1.2 Three claims in the current framing that the evidence does not support

#### (a) The "+1.208 pp of available improvement" is not proposal-quality headroom

**[new]** Decomposing the sealed panel by which teacher candidate actually won
its 128-world selection (exact additive decomposition of the official total):

| Teacher's chosen action came from | roots | mean gain when it wins | contribution |
|---|---:|---:|---:|
| same as anchor (no change) | 457 | - | +0.00000 pp |
| its two **policy-ranked** extras | 154 | +3.0616 pp | **+0.65484 pp** |
| its two **uniform-random** extras | 109 | +3.6518 pp | **+0.55284 pp** |
| total | 720 | | **+1.20768 pp** |

**46% of the headroom is delivered by candidates sampled uniformly at random
from the legal set**, and conditional on winning, random extras score *better*
than ranked ones. Per-candidate selection win rates **[new]**: ranked
154/1375 = 11.20%, random 109/1313 = 8.30%.

This is a candidate-count effect under a noisy selector, not a ranking-quality
effect. It is also delivered at 2x the selection search (8x128 vs 4x128).
Quoting "+1.208 pp retained by the broader teacher" as the target that a single
better-trained 4th proposal should capture is misleading, and that framing has
driven experiment selection since 2026-10-02.

*Exactness note.* The additive K=5 counterfactual below has **zero**
uncomputable roots. Proof: if the extra candidate `e` beats all four of the
anchor's candidates on the selection block, then it also beats the common three,
so the learned arm itself chose `e`, so `e` is in the assessed set; otherwise
the K=5 arm's pick equals the anchor's pick, which is assessed by definition.
Taking the max over 5 noisy estimates rather than 4 *increases* the winner's
curse, so these additive estimates are if anything conservative.

#### (b) The deployed candidate set already contains a uniformly random action

`candidates()` in `src/splendor_ai/alphazero/run_afterstate_advantage.py:149`
returns `[035-top1, 035-top2, random, random]` (the last two via
`rng.sample(rest, min(2, len(rest)))` over the remaining canonical groups).
`fourth_slot()` in `scripts/analysis/learned_proposal_arena.py:101` keeps
`reference[:3]` and replaces slot 4.

Therefore **"the unchanged common first three candidates" = 035-top1, 035-top2,
and one uniformly random canonical action.** No handoff entry states this.

**[new]** Which slot wins the anchor's own 128-world selection:

| slot | wins | share | mean assessed score of that choice |
|---|---:|---:|---:|
| 035-top1 | 268 | 37.2% | 52.622% |
| learned 4th | 161 | 22.4% | 58.283% |
| **uniform random** | 150 | 20.8% | 58.016% |
| 035-top2 | 141 | 19.6% | 55.992% |

**[new]** Mean 128-world selection value by slot (WDL): 035-top1 +0.0223,
035-top2 -0.0087, uniform random -0.0122, anchor 4th +0.0136, continued 4th
+0.0146, control 4th +0.0220, teacher ranked +0.0165, teacher random -0.0179.

The random slot is as productive as the learned slot. This is a material fact
about the agent under study and it reframes every "breadth" result: the
programme is not adding diversity to a deliberate candidate set, it is adding
diversity to a set that is already one-quarter random.

#### (c) The conditional endpoint failed its only external validation, at the gate magnitude

- `outputs/equal_budget_breadth_128`: uniform 8x64 minus anchor 4x128 =
  **+0.631510 pp [+0.021701, +1.254340]**, classification
  `positive_conditional_equal_budget_breadth`, **both prespecified gates passed**
  (mean >= +0.5 pp and lower bound > 0).
- The authorized full-game confirmation,
  `outputs/equal_budget_breadth_arena_128/recovery_03/report.md`: 198W / 4D /
  200L over 402 games = **49.7512% [44.9005, 54.6020]**,
  `below_material_target`, the 55% target **excluded**.

A conditional gain that cleared the +0.5 pp practical gate produced exactly zero
full-game strength. For contrast, +0.4209 pp conditional produced 56.9652%
full-game and +0.8522 pp produced 64.4279%. The implied conversion ranges from
~0 to ~17 pp of win rate per pp of conditional score.

**The +0.5 pp conditional gate therefore has no demonstrated monotone
relationship to full-game strength.** Thirteen-plus experiments since 2026-10-02
have been selected and judged by an instrument that failed the one test of it
that was ever run.

### 1.3 The diagnosis I would offer instead

The working puzzle — "why don't planning improvements become learned guidance on
new positions?" — conflates two distinct facts.

#### Fact 1: the learned-proposal slot is saturated, not broken

- Replacing a *random* 4th with a *learned* 4th: **+15.8 pp of full-game win
  rate** (`learned_proposal_128`, 65.7960%).
- Replacing a *learned* 4th with a *better-trained* learned 4th: zero, across
  nineteen consecutive studies.

**[new]** Fixed-effect pooling of the five comparable fresh-panel primary
contrasts (learner minus anchor):

| study | estimate (pp) | 95% CI | SE | weight |
|---|---:|---|---:|---:|
| `planner_state_transfer_128` | +0.049913 | [-0.190972, +0.307075] | 0.1271 | 61.95 |
| `action_card_relations_128` | -0.019531 | [-0.277778, +0.234375] | 0.1307 | 58.58 |
| `teacher_precision_512` | -0.160590 | [-0.665170, +0.346140] | 0.2580 | 15.02 |
| `independent_game_scaling_128` | +0.114475 | [-0.269111, +0.533312] | 0.2047 | 23.87 |
| `large_corpus_optimization_128` | -0.047743 | [-0.423177, +0.324978] | 0.1909 | 27.45 |

Pooled estimate **+0.0051 pp, SE 0.0732, 95% CI [-0.1383, +0.1485]**.
These panels share the three 161 fits and the fast-161 source population, so the
true SE is larger than the fixed-effect SE. Inflating it:

| SE inflation | 95% CI | z against +0.5 pp |
|---:|---|---:|
| x1.5 | [-0.2099, +0.2202] | 4.51 |
| x2.0 | [-0.2816, +0.2919] | 3.38 |
| x2.5 | [-0.3533, +0.3636] | 2.71 |

Even at 2.5x inflation the +0.5 pp practical target is excluded at p ~ 0.007.
Caveat: the pooled quantity is "the average effect of this family of recipes",
which is the decision-relevant quantity, not a single intervention's effect.

The nulls are individually underpowered. Jointly they are not. The first
deliberate candidate captured essentially all of the value a single learned
candidate can carry.

#### Fact 2: the root evaluator is noise-limited, so small quality differences are undecidable

**[new]** Split-half measurement inside the sealed 128 selection worlds
(worlds 0-63 vs 64-127; all candidates, all roots):

| quantity | value | in score pp |
|---|---:|---:|
| SE of one candidate's 128-world mean | 0.07273 WDL | 3.64 pp |
| SE of a **paired** (common-world) 128-world difference | 0.06782 WDL | 3.39 pp |
| implied cross-candidate correlation within a world | rho ~ 0.57 | - |
| observed within-root spread of 128-world means (SD) | 0.16265 WDL | 8.13 pp |
| **true** between-candidate value SD within a root | 0.14548 WDL | 7.27 pp |
| reliability of a 128-world candidate mean | 0.800 | - |
| true paired-difference SD | 0.23619 WDL | 11.81 pp |
| SNR of one pairwise comparison at 128 worlds | 3.48 | - |

Common random numbers are working (worlds are shared across candidates) but
recover only ~43% of the variance. The remainder is the belief x action
interaction, consistent with experiment 058's attribution: 61.741% of balanced
variation from the sampled public belief, 9.225% from continuation conditions,
29.034% from their interaction. The shared-belief main effect cancels in paired
differences; the interaction does not.

**[new]** Decidability of the actual decision:

| | median top1-top2 margin | > 1 SE | > 2 SE | > 3 SE |
|---|---:|---:|---:|---:|
| anchor K=4 | 0.0625 WDL = 3.12 pp = **0.92 SE** | 49.4% | **27.0%** | 15.1% |
| teacher K=8 | 0.0469 WDL = 2.34 pp = **0.69 SE** | 36.5% | 17.6% | 8.8% |

**[new]** Split-half decision agreement (64 vs 64 worlds, same root):
**anchor K=4 = 56.3%** (403/716), **teacher K=8 = 43.7%** (313/716).
Among the 2-4 independently assessed actions, the selection block's argmax
equals the assessment block's argmax only **60.2%** (213/354) of the time.

This replicates experiment 055 (independent resolver calls agreed on the exact
best action 45.83%) and the 158 saved-evidence diagnostic (16/16 world splits
disagreed 70.35% for learned continuations, 68.16% for greedy 035) at the
current budget. **The planner chooses between its top two candidates close to a
coin flip in the majority of positions.**

#### The two facts combine

The proposer's marginal product is bounded by a selector that cannot resolve the
differences the proposer is trained to produce. A better-*ranked* 4th candidate
is invisible at a 3.4 pp comparison SE; a *diverse* candidate occasionally has a
true advantage large enough to be seen. Hence:

**[new]** marginal value of **one** added candidate on top of the anchor's four,
with paired bootstrap:

| added candidate | gain (pp) | 95% CI / bounds | coverage |
|---|---:|---|---|
| continued's extra (K=5) | **+0.6228** | [+0.3619, +0.9001] | exact, 0 uncomputable |
| control's extra (K=5) | **+0.7595** | [+0.4281, +1.1686] | exact, 0 uncomputable |
| both learned extras (K=6) | **+1.0238** | [+0.6413, +1.4741] | exact, 0 uncomputable |
| teacher's two ranked (K=6) | +0.6750 | [+0.4081, +0.9474] | n=691, 29 uncomputable |
| teacher's two random (K=6) | +0.6662 | [+0.3980, +0.9600] | n=686, 34 uncomputable |
| teacher K=8 (reconstruction check) | +1.2033 | [+0.8605, +1.5592] | official +1.2077 |
| single ranked extra (mean of the two) | - | bounds [+0.3605, +0.7206] | partial |
| single random extra (mean of the two) | - | bounds [+0.3591, +0.6375] | partial |

Per candidate the learned, policy-ranked and **uniformly random** extras occupy
the same band. Bounds credit uncomputable roots 0 (lower) and the teacher's
realized gain (upper).

**[new]** Why the headline contrast is zero — exact decomposition of the
substitutive design:

| arm | identical choice | own extra selected | contribution | anchor's 4th displaced | contribution | net |
|---|---:|---:|---:|---:|---:|---:|
| continued | 508 | 143 | +0.4199 pp | 69 | -0.4677 pp | **-0.0477 pp** |
| control | 525 | 125 | +0.5018 pp | 70 | -0.4888 pp | **+0.0130 pp** |

The learned arm does not *add* a candidate; it *replaces* the anchor's own
learned 4th, which is itself good (it wins selection 22.4% of the time). Gains
and displacement losses cancel almost exactly.

**[new]** The trained models are *not* redundant with the anchor. Rank of the
proposed action under the original 161 anchor's own logits (0 = its top action):

| proposer | n | median rank | mean rank | rank <= 3 | rank >= 10 |
|---|---:|---:|---:|---:|---:|
| anchor's own 4th | 705 | 1.0 | 1.30 | 100% | 0% |
| continued's 4th | 705 | 7.0 | 8.60 | 31.2% | 33.0% |
| control's 4th | 705 | 6.0 | 8.26 | 31.3% | 30.2% |
| teacher ranked extras, when they win | 154 | 3.0 | 2.87 | 68.2% | 0% |
| teacher random extras, when they win | 109 | 9.0 | 11.81 | 0% | 49.5% |

This matches the September screen's descriptive note (learned proposals averaged
rank 6.57-6.77, the ranked control 3.05, the random 4th 14.32). Training moved
the proposals a long way from the anchor's ordering, and they are still worth
the same. Conditional on winning selection, continued's extras gain +2.1143 pp
and control's +2.8906 pp, versus +3.0616 pp for ranked and +3.6518 pp for random
teacher extras.

#### Fact 3: compute does not buy its way out

**[new]** An equal-budget frontier computed entirely from saved outcomes:
choose using the first *N* selection worlds per candidate, score the chosen
action on a **common held-out block of the final 32 selection worlds**. Disjoint
world subsets, so unbiased; full coverage for every arm, 720 roots.

Held-out conditional score (%), budget = K x N continuations:

| budget | K=4 | K=5 | K=6 | K=8 |
|---:|---:|---:|---:|---:|
| 96 | 54.7678 (N=24) | - | 55.0629 (N=16) | 55.0347 (N=12) |
| 120-128 | 55.2517 (N=32) | 55.3798 (N=24) | - | 55.3407 (N=16) |
| 144-160 | - | 55.7161 (N=32) | 55.7639 (N=24) | - |
| 192 | 55.5556 (N=48) | - | 55.9115 (N=32) | 55.9918 (N=24) |
| 240-256 | 55.7661 (N=64) | 56.1979 (N=48) | - | 56.2131 (N=32) |
| 320 | - | 56.2891 (N=64) | - | - |
| 380-384 | 56.0720 (N=95) | - | 56.3194 (N=64) | 56.3542 (N=48) |

Matched-budget contrasts, paired bootstrap within the 18 cells:

| contrast | budget | estimate | 95% CI |
|---|---:|---:|---|
| K8x24 - K4x48 | 192 | +0.4362 pp | [-0.1845, +1.0503] |
| K8x32 - K4x64 | 256 | +0.4470 pp | [-0.1671, +1.0742] |
| K8x16 - K4x32 | 128 | +0.0890 pp | [-0.5729, +0.7704] |
| K5x51 - K4x64 | ~256 | +0.3451 pp | [-0.0825, +0.7769] |
| K5x38 - K4x48 | ~192 | +0.5729 pp | [+0.1562, +1.0091] |

Unmatched (extra compute) references:

| contrast | budget | estimate | 95% CI |
|---|---:|---:|---|
| K8x64 - K4x64 | 2.0x | +1.0373 pp | [+0.5425, +1.5430] |
| K5x64 - K4x64 | 1.25x | +0.5230 pp | [+0.1910, +0.8789] |
| K4x95 - K4x24 | 4.0x | +1.3043 pp | - |

Readings:

1. **A 4x increase in root search buys +1.30 pp of conditional score.**
2. At **matched** budget, breadth versus precision is a near-wash whose CIs
   mostly span zero. This independently predicts, and explains, the 49.75% arena
   null, and it contradicts the dedicated equal-budget screen's clean pass. The
   frontier recovers that answer for *zero* new compute where the dedicated
   route cost a full study plus a ~90 h arena.
3. Adding candidates at **extra** compute does work, and matches the official
   results (+1.21 pp conditional, 56.22% full-game at 2x).
4. **[new]** Selector-precision sweep (same instrument, gain relative to the
   same-N anchor K=4, pp): the learned-vs-learned swap is flat at every
   precision, while breadth value grows with precision.

   | N worlds | continued K4 swap | control K4 swap | +continued K5 | +2 ranked K6 | +2 random K6 | teacher K8 |
   |---:|---:|---:|---:|---:|---:|---:|
   | 16 | +0.0316 | +0.2982 | +0.4768 | +0.5909 | +0.3937 | +0.8166 |
   | 32 | +0.0962 | +0.4579 | +0.6590 | +0.8181 | +0.5165 | +1.0135 |
   | 64 | +0.0966 | +0.3581 | +0.6207 | +0.6565 | +0.7433 | +1.0753 |
   | 95 | -0.0977 | +0.2409 | +0.5664 | +0.7747 | +0.6706 | +1.1697 |

   Absolute anchor K=4 held-out score: 54.1673% (N=16), 55.0195% (32),
   55.5990% (64), 56.0655% (95).

Externally, experiment 148 found 4x PUCT compute worth 53.75% [49.5, 58.0]
(not significant) at a measured 3.96x time ratio, with mean maximum depths
3.226 vs 2.404. **Compute scaling in this architecture is flat.** That, not the
proposer puzzle, is the fact most hostile to the superhuman objective.

### 1.4 The measurement gap that dominates everything

The objective is superhuman strength **under a declared computational budget**.
A grep of the strategy corpus finds the phrase as a requirement in four
documents (`superhuman_information_set_search_roadmap_after_132.md:11`,
`independent_superhuman_review_after_134_137.md:286`,
`whole_programme_direction_review_during_134.md:379`,
`parallel_work_during_162.md:102`) and as a **number in none**. The objective is
currently unfalsifiable.

The only external calibration in the project is `074`
(`docs/experiments/human_resolver_ceiling_calibration_074.md`): ten games,
2026-08-03, in which the owner beat the then-agent 7-3 (descriptive point-share
60% under a half-credit tie convention). Everything since is a chain of pairwise
self-comparisons against ancestors, which cannot establish superhuman strength.

`074`'s follow-up (`docs/human_resolver_feedback_074.md`) remains the single most
productive diagnostic the project has produced: a tiebreak-blind rollout
objective, a numerical-dust tie pathology (41 of 6,625 resolved decisions with a
chosen-minus-parent gap <= 1e-12), and a 13x gap in blind-reserve frequency
(agent 27/278 = 9.71%, human 2/278 = 0.72%, 24 of the agent's in the opening).
The blind-reserve frequency gap has never been explained.

---

## 2. Strongest remaining hypotheses, ranked

### H1 - The root evaluator's comparison noise, not the proposer, binds the planner
**Confidence: high (0.8).**

*Supporting:* 3.39 pp paired SE against a 0.92 SE median decision margin; only
27% of K=4 decisions exceed 2 SE; 56.3% / 43.7% split-half decision agreement
**[new]**; experiment 055's 45.83%; experiment 058's variance attribution;
precision-anchor only +0.1487 [-0.0944, +0.3982]; the flat frontier **[new]**.

*Counter-evidence:* `learned_proposal_128` shows a 15.8 pp full-game effect *is*
achievable by changing one candidate, so the selector is not useless - it cannot
resolve *sub-SE* differences. The one targeted variance-reduction study,
`control_variate_128`, returned variance x CPU = 1.00351 [1.00330, 1.00372],
`efficiency_not_established`. But it tested exactly one recipe - prefix
truncation at 4/8/16 plies with the frozen 143 public evaluator and nested
calibration - and all TRAIN horizon-best recipes were worse than baseline. That
failure is attributable to that evaluator, not to variance reduction generally.

*Discriminating test:* any intervention that lowers the paired comparison SE at
fixed continuation count - antithetic or stratified determinization, or
conditioning on refill statistics whose expectation is exactly known from the
remaining deck - should raise the *held-out* value of the arm's chosen action.
This is measurable on the Appendix B instrument before committing to an arena.

### H2 - The single learned-proposal slot is saturated
**Confidence: high (0.85).**

*Supporting:* pooled +0.0051 pp [-0.1383, +0.1485] **[new]**; flat at every
selector precision from N=16 to N=95 **[new]**; TRAIN acquisition of 97.7-98.1%
with zero transfer; failure survives changes in data volume (720 -> 2,880 ->
11,520 added rows), teacher precision (128 -> 512 worlds), objective, weighting,
representation, optimizer schedule and dose (7,136 -> 28,544 updates).

*Counter-evidence:* none identified.

*Discriminating test:* already performed. As close to settled as the
programme's instruments allow.

### H3 - The conditional endpoint is not a valid screen at the +/-0.5 pp scale
**Confidence: moderate-high (0.7).**

*Supporting:* 8x64 passed both gates and produced 49.7512%; the conditional ->
full-game conversion varies from ~0 to ~17 pp of win rate per pp across four
observations.

*Counter-evidence:* two learned-proposal comparisons transferred in the right
direction and roughly the right proportion. The 8x64 case differs structurally:
it reallocates precision, and lower precision compounds over ~24 repeated
decisions per game, whereas the conditional test penalises that noise once.

*Discriminating test:* a formal calibration against the arenas already run; see
§6.2.

### H4 - The evaluation design nets out a real effect by substituting rather than adding
**Confidence: moderate (0.55). My most tentative claim.**

*Supporting:* the exact decomposition and the additive K=5 counterfactual in
§1.3 (+0.6228 pp [+0.3619, +0.9001] for `continued`, +0.7595 pp for `control`,
both with zero uncomputable roots).

*Counter-evidence, and why this ranks only fourth:*
(i) it is a post-hoc reanalysis of a confirmatory panel;
(ii) it costs ~17-21% more selection compute, so it is a breadth result in
disguise, and matched-budget K=5 points are not consistently significant
(+0.3451 [-0.0825, +0.7769] at budget 256 versus +0.5729 [+0.1562, +1.0091] at
~192);
(iii) the programme already tested an additive variant on a different panel -
`saved_additive_proposals_128`, additive-broad minus anchor = +0.161675 pp
[-0.039063, +0.392795], `additive_broad_not_established` - and got much less,
because that learner proposed a distinct action in only 119/360 roots against
589/720 here.

Nonetheless the substitutive design answers "is this learned ranking better than
that learned ranking", and the handoffs read it as "can learning improve the
planner". Those are different questions and the second has not been asked of
these models.

### H5 - The architecture has a strength ceiling well below superhuman
**Confidence: moderate (0.5), high decision relevance.**

*Supporting:* flat compute scaling (4x -> +1.30 pp conditional **[new]**; 4x
PUCT -> 53.75%, n.s., in 148); depth-1 candidate evaluation with a deterministic
greedy-035 rollout to terminal
(`src/splendor_ai/alphazero/run_afterstate_advantage.py:103`); AlphaZero mean
maximum depth 3.33 / 4.10 at 256 / 1,024 simulations (137); a strictly stronger
continuation policy made the planner *worse* (158: provisional 48.00995%
[43.28358, 52.73632], all four gates fail) even though that controller beats 035
by +7.7292 pp in direct play (157: 57.5833% [54.7708, 60.3958]); exact-084
continuations attenuated the measured planner gain by about half (140:
stochastic-035 gain +0.043981 [-0.010185, +0.105093] versus exact-084
+0.021296 [-0.007407, +0.054630], underpowered at 36 positions).

*Counter-evidence:* 113 found substantial deeper-search headroom; the human
panel is ten games against a far weaker agent; no exploitability or multi-ply
study has been run on the current agent.

*Discriminating test:* an external calibration plus forensic audit (§5). If the
agent loses to a strong human on identifiable *plan-level* errors rather than
noise, H5 is live.

### H6 - Private-information handling limits strength
**Confidence: low (0.15).**

`docs/strategy/stochastic_refill_vs_blind_reserve_architecture_correction.md` is
well-argued and experiment 071 is genuinely powered (mask changed 4,641/63,424
decisions; paired score share 50.575% [48.5, 52.65],
`deck_reserve_mask_neutral_at_measured_dose`). I agree with its conclusion and
its guardrail. Two pieces remain open: adversarial exploitation *after an
opponent's* blind reserve, and the unexplained 13x blind-reserve frequency gap
versus the human.

---

## 3. What the evidence rules out, lowers, and does not rule out

### Reasonably ruled out, at the programme's own practical threshold

- Improving the single 4th-proposal slot by better supervision of any kind
  tried: data volume, teacher precision (128 -> 512 worlds), soft and finite
  targets, opportunity weighting, loss variants, learning rate, dose, coverage,
  frozen representation, explicit action-card relations. Pooled
  [-0.1383, +0.1485] pp; +0.5 pp excluded at z >= 2.7 under generous SE
  inflation **[new]**.
- Buying strength with more of the *same* search: precision at K=4 (+0.1487 pp,
  CI spans zero); breadth at matched budget in full games (49.7512%, 55%
  excluded); 4x PUCT (53.75%, n.s.).
- Replacing the rollout policy with a stronger learned controller (158).
- Hierarchical action-intent representation (135: 7.778% of positions had a
  multi-variant core intent, 2.888% of visits, +0.804 pp agreement improvement).
- A broad refill-identity coverage failure (136: 98.635% priority-edge and
  99.951% visit-winner expected identity coverage at 1,024 simulations).
- Blind deck-reserve as a *primary* architectural driver (071).

### Lowered in priority but not excluded

- Shared-card / attention representations. 151 failed on *different, smaller*
  data (shared MSE 0.095864 vs flat 0.096116 = 0.2625% reduction against a 3%
  gate; selected WDL minus 035 +0.000190 against a +0.02 gate). The frozen-base
  relations add-on acquired only 10-15% of its added training signal. Neither is
  a clean test of the running fully-trainable design.
- Control variates: one recipe, one weak evaluator.
- Learned value functions: 143-147 all failed, but all on the same label
  pipeline.

### Explicitly NOT ruled out

- **Variance reduction in the determinized root evaluation at fixed compute.**
  Barely attempted. 058 says 62% of the variance is belief sampling; common
  worlds recover only ~43% of it **[new]**.
- **Breadth at extra compute.** This is the programme's one confirmed strength
  gain (56.2189%) and it is **unpromoted**.
- Adding a learned candidate rather than substituting one (H4).
- Multi-ply or opponent-response planning on the current agent.
- That 137 says anything about chance-aware search. Its formal decision is
  `stratified_refill_mechanism_invalid_no_search_conclusion` (priority coverage
  rose 86.927% -> 95.861%, below the +10-point mechanism gate). **A
  mechanism-invalid experiment is not a scientific negative and must not be
  cited as one.** Similarly, 134's panel is formally invalid on a runtime
  counter, though its 31.875% [27.375, 36.5] gameplay signal stands on its own
  terms.

---

## 4. Assessment of the running `structured_proposer_128`

### What it can establish

A clean, well-controlled, well-verified comparison of twelve explicit relation
channels against matched raw channels at full trainability on the exact existing
corpus. If `relations - raw` has a positive 95% lower endpoint, that is real
evidence that explicit resource relations help a trainable proposer.

The protocol is careful and the execution discipline is exemplary: paired
initialisation with a zero-output branch preserving the acquisition denominator,
identical batch stream and optimizer settings, prospective gates with no
selection freedom, 181-check integrated preflight, a frozen outcome-blind
resource receipt, and a 96 h ceiling fixed before freeze. I found nothing to
criticise operationally.

### What it cannot establish

Nothing about whether the 4th-proposal *role* is worth improving - which §1.3
and H2 identify as the actual question. It also cannot separate "representation"
from "selector noise ceiling", because both arms are measured through the same
noise-limited selector.

### Material design weaknesses (reported, not acted on)

1. **The primary gate is very probably unreachable by construction.** Gate 1
   requires `relations - anchor >= +0.5 pp`. That is the same one-candidate
   substitution whose pooled effect across five fresh panels is +0.0051 pp
   [-0.1383, +0.1485] **[new]**. Nothing in twelve feature channels changes the
   selector's 3.39 pp comparison SE or the 0.92 SE median decision margin.
   Estimated P(joint pass) well under 10%.
2. **The acquisition gate will almost certainly pass**, since the frozen-base
   version already fits and the fully trainable version has 642,530 trainable
   policy parameters on 20,981 rows at 28,544 updates. The full 50-80 h,
   including the 24 h evaluation reserve, will therefore be spent.
3. **The informative contrast is demoted.** `relations - raw` is the only
   contrast the design actually isolates, yet it is gate 3, a necessary
   condition behind an unreachable gate 1. A `relations - raw` effect of, say,
   +0.2 pp would be scientifically interesting and would still be reported as
   `practical_progress_not_established`.
4. **The protocol's own exit clause is the realistic outcome, and it is the
   right one**: "A well-acquired null weighs against this particular local
   structural remedy." Read that way the study is a reasonable *closing* move
   that buys a decisive reason to stop. The disagreement is about price: 50-80 h
   and the remaining research block to buy information the pooled prior already
   supplies.

### Recommendation on the run itself

**Let it finish.** Stopping now destroys a frozen, nearly-complete comparison and
buys nothing not already derivable from saved data. Do not wait for it before
acting on §5. Prespecify now that a well-acquired null closes the
proposer-compression line permanently rather than motivating a nearby sweep.

---

## 5. Single highest-value next step

> **Declare the per-move compute budget, pick the strongest confirmed
> configuration at that budget, and re-run the `074` external calibration
> against it with a full forensic audit of every agent decision. Do not fund
> another conditional screen.**

### Why this rather than an experiment

Every machine alternative is judged by an instrument that failed its one
external validation at exactly the gate magnitude (§1.2c), on an agent whose
compute-scaling curve is flat (§1.3). Expected information per CPU-hour from
another conditional screen is now very low. Meanwhile the programme cannot
answer the only question its objective poses, and has not asked it for ten
weeks, across a span in which the agent improved by roughly +540, +108, +103 and
+43 win-rate-equivalent steps against successive ancestors with no external
anchoring at any point.

### Minimal concrete design

1. **Declare the budget.** A decision, not an experiment; zero cost. State
   seconds per move and hardware. On current evidence the honest candidates are
   4x128 (~66 s/move measured) or 8x128 (~128 s/move measured). Write it into
   the roadmap so "superhuman under a declared budget" becomes falsifiable.
2. **Fix the agent**: the strongest *confirmed* configuration at that budget. At
   ~128 s/move that is the 8x128 breadth planner, which beat the learned-128
   anchor 56.2189% [51.6169, 60.6965] over 402 games and is currently
   unpromoted. This is a promotion decision the programme has already earned and
   not taken.
3. **Calibration panel**: re-run the frozen `074` protocol against the current
   agent - exactly ten games, fresh seeds, alternating seats, no mid-panel
   inspection, no strategy adaptation, Stage A exact-execution-equivalence check
   first.
4. **Forensic audit**, prespecified before play, of every agent decision:
   (a) actions dominated in immediate prestige;
   (b) blind-reserve frequency against the human's 0.72% baseline;
   (c) every endgame decided by the fewest-development-cards tiebreak, and
   whether the terminal-WDL objective now handles it correctly;
   (d) the fraction of decisions whose chosen action had a sub-1-SE selection
   margin, i.e. coin flips.
   Infrastructure exists: `run_gui.py`, `run_human_game_forensic_audit.py`,
   `scripts/analysis/`.

### Decision criteria

Deliberately **not** a significance gate - ten games give roughly a +/-30 pp
interval. Prespecify instead:

- agent scores >= 7/10 with no identified dominated-move class -> declare the
  human calibration saturated and commission a multi-human, budget-declared
  panel as the next milestone;
- agent scores <= 5/10 -> the forensic audit's largest error class becomes the
  next research target and the proposer line is formally closed;
- in between -> report the error classes and treat direction as unresolved.

### Cost

A few CPU-hours (~10 games x ~30 agent decisions x 8 candidates x 128 worlds)
plus the owner's time. Compare 50-80 h for the running study and 60-90 h for any
arena.

### Strongest alternative and why it loses

A root-evaluator variance-reduction study (H1): antithetic or stratified
determinization, or refill statistics with known expectation used as a control
variate, screened on the Appendix B instrument before any arena. It targets the
mechanism I believe binds and is the machine experiment I would fund if one is
wanted. It loses on sequencing: it would be screened by the same conditional
endpoint whose validity is in question, and its payoff is bounded by a compute
curve that buys +1.30 pp per 4x. Fix the measurement chain first; then this
becomes the obvious second step, and interpretable.

### On the proposer-compression line

**Leave it. Not "pause" - close it.** The pooled bound, the selector-noise
measurement, and the saturation pattern (random -> learned = +15.8 pp; learned ->
better-learned = 0) together make it the best-characterised dead end in the
project. Its remaining value is as a settled negative result, worth preserving
and citing.

---

## 6. Cheapest additional observations, and missing evidence

Ordered by cost. The first three require no new inference at all.

### 6.1 Verify the saved-data frontier against the official assessment block
Minutes, zero inference. Re-run the selector-precision sweep with the selection
and assessment blocks swapped, and check the additive K=5 counterfactual against
the official 128-world assessment wherever coverage permits. If the frontier does
not reproduce, discount §1.3 and H4 accordingly. Appendix B is the instrument.

### 6.2 Calibrate the conditional endpoint against the arenas already run
An afternoon, zero inference. Four to five (conditional gain, arena score) pairs
exist: +0.8522 -> 64.4279%; +0.4209 -> 56.9652%; +1.2316 -> 56.2189% at 2x;
+0.6315 -> 49.7512% at 1x. Fit the conversion and its uncertainty. If conditional
gains predict full-game score well *except* when the intervention changes search
allocation, the +0.5 pp gate can be rehabilitated with a stated scope
restriction and H3 drops sharply.

### 6.3 Replicate the additive contrast on other saved panels
Re-score `planner_state_transfer_128`, `action_card_relations_128` and
`teacher_precision_512` with each arm restricted to the anchor's four candidates
*plus* one added learned candidate. If +0.62 pp replicates across panels the
additive role deserves prospective confirmation; if it collapses toward
`saved_additive_proposals_128`'s +0.16 pp, drop H4.

### 6.4 The running experiment's own result
1-2 days, already paid for. A `relations - anchor` >= +0.5 pp with a positive
lower endpoint would overturn the pooled bound and justify continuing the line;
estimated at under 10%. A positive `relations - raw` inside an overall null is
worth recording as a representation finding and still not worth continuing on.

### 6.5 Top-2 margin distribution at 256 and 512 worlds
A few CPU-hours on ~50 saved roots. If the fraction of decisions with margin
> 2 SE rises steeply with world count, precision is undervalued and H1's remedy
is simply "more worlds". If it stays near 27%, the margins are genuinely small
and either variance reduction or a different decision procedure is required.

### 6.6 Missing evidence

- Any external opponent that is not a project ancestor: a published Splendor
  engine, a rules-based reference, or a second human.
- A declared compute budget (a number).
- A measurement of the agent's exploitability after an *opponent's* blind
  reserve, and an explanation of the 13x blind-reserve frequency gap.
- One replication of a conditional -> full-game transfer at a +0.5 pp effect
  size, in either direction. This is the entire basis on which the last six
  weeks of experiments were selected, and it rests on four unreplicated points.

---

## 7. Substantive disagreements with the existing strategy

1. **"+1.208 pp retained teacher headroom" should stop being quoted as learnable
   proposal quality.** It is largely a candidate-count effect, 46% of it
   delivered by uniformly random candidates, at 2x the selection search.
2. **The +0.5 pp practical gate should not be used again until it is
   calibrated.** It has passed an intervention that produced nothing (8x64,
   49.75%) and been missed by interventions that produced a great deal.
3. **The proposer-compression line should be closed on pooled evidence**, not
   retired after one more individually-underpowered null.
4. **The handoffs should state that one of the three common candidates is a
   uniformly random legal action.** Its absence has distorted the reading of
   every breadth and proposal result.
5. **The programme's biggest unaddressed risk is not a learning question.** It
   is that a flat compute curve and ten weeks without an external anchor leave
   "superhuman under a declared budget" currently unmeasurable.

---

## Appendix A - All new quantities in one place

Panel: `outputs/large_corpus_optimization_128`, 720 main roots, 18 cells of 40
games, 128 selection + 128 disjoint assessment worlds per root.

**Official sealed results (for reference).** continued-anchor -0.047743
[-0.423177, +0.324978]; continued-control -0.060764 [-0.500217, +0.315213];
control-anchor +0.013021 [-0.427517, +0.494792]; teacher-anchor +1.207682
[+0.866428, +1.566840]; teacher-continued +1.255425 [+0.849609, +1.675347].
Conditional means: anchor 55.671658%, control 55.684679%, continued 55.623915%,
teacher 56.879340%. TRAIN acquisition added 97.7230-98.1031%, old ~97.5%.

**Decision changes.** Chosen action differs from the anchor's: continued 212,
control 195, teacher 263. Learned 4th identical to the anchor's 4th: continued
131/720, control 151/720.

**Teacher decomposition.** ranked extras win 154 roots (+3.0616 pp each,
+0.65484 pp total); random extras win 109 roots (+3.6518 pp each, +0.55284 pp
total); unchanged 457.

**Substitutive decomposition.** continued: 143 wins (+0.4199 pp), 69
displacements (-0.4677 pp), net -0.0477 pp. control: 125 wins (+0.5018 pp), 70
displacements (-0.4888 pp), net +0.0130 pp.

**Additive counterfactuals (paired bootstrap, 20,000 draws).** K=5 continued
+0.6228 [+0.3619, +0.9001]; K=5 control +0.7595 [+0.4281, +1.1686]; K=6 both
learned +1.0238 [+0.6413, +1.4741]; K=6 anchor+2 ranked +0.6750
[+0.4081, +0.9474]; K=6 anchor+2 random +0.6662 [+0.3980, +0.9600]; K=8 teacher
reconstruction +1.2033 [+0.8605, +1.5592].

**Single-candidate bounds (missing credited 0 / teacher gain).** ranked1
[+0.2552, +0.6284]; ranked2 [+0.4659, +0.8128]; random1 [+0.2074, +0.5261];
random2 [+0.5108, +0.7488].

**Evaluator noise.** 64-world candidate SE 0.10286 WDL (5.14 pp); 128-world
candidate SE 0.07273 WDL (3.64 pp); 128-world paired difference SE 0.06782 WDL
(3.39 pp); true between-candidate SD 0.14548 WDL (7.27 pp); reliability 0.800;
true paired-difference SD 0.23619 WDL (11.81 pp); pairwise SNR 3.48.

**Decidability.** anchor K=4 median top1-top2 margin 3.12 pp = 0.92 SE, >1 SE
49.4%, >2 SE 27.0%, >3 SE 15.1%; teacher K=8 median 2.34 pp = 0.69 SE, >2 SE
17.6%. anchor top1-worst spread: median 12.50 pp, mean 14.95 pp.

**Repeatability.** split-half (64/64) decision agreement anchor 403/716 = 56.3%,
teacher 313/716 = 43.7%. Selection argmax equals assessment argmax 213/354 =
60.2%.

**Winner's curse (chosen action, selection mean minus assessment mean, WDL).**
anchor +0.03984, continued +0.04200, control +0.04491, teacher +0.05433.

**Slot win shares (anchor K=4).** 035-top1 268 (37.2%), learned 4th 161 (22.4%),
uniform random 150 (20.8%), 035-top2 141 (19.6%).

**Proposal ranks under the anchor's own logits.** anchor 4th mean 1.30;
continued 8.60; control 8.26; teacher ranked wins 2.87; teacher random wins
11.81.

**Panel shape.** canonical support median 18, mean 21.8, min 1, max 256; 25
roots with support <= 4; union sizes 8-10 for 650 of 720 roots; distinct
assessed actions 1 in 366 roots, 2 in 207, 3 in 116, 4 in 31.

**Equal-budget frontier and precision sweep.** See the tables in §1.3 Fact 3.

---

## Appendix B - Consolidated verification script

Read-only. Reads only saved JSON under
`outputs/large_corpus_optimization_128/main/roots/`. Runs no model, no engine,
no simulation. Reproduces every **[new]** quantity above. Run with
`.venv/Scripts/python.exe -I review_check.py` from the repository root.

Cost: it parses 720 JSON files (~430 MB on disk) one at a time; the retained
per-world outcome tensor is only ~5.5 MB, so resident memory stays small and it
will not compete with a running study. Expect a few minutes, dominated by the
bootstrap (lower `BOOT` to 2000 for a quick pass).

Provenance note: during the review each block below was executed separately and
produced exactly the figures reported above. This consolidated form has been
syntax-checked but not re-executed end to end as a single file; if any figure
fails to reproduce, trust the artifacts over this script and say so.

```python
"""Read-only reanalysis of outputs/large_corpus_optimization_128. No inference."""
import json, math, random, statistics, collections

BASE = "outputs/large_corpus_optimization_128"
N_ROOTS = 720
HOLD = 32          # held-out selection worlds used by the frontier
BOOT = 20000

recs = []
for i in range(N_ROOTS):
    d = json.load(open(f"{BASE}/main/roots/state_{i:04d}.json"))
    n = len(d["union"]["keys"])
    recs.append({
        "index": i, "fit": d["spec"]["fit"], "seat": d["spec"]["seat"],
        "phase": d["spec"]["phase"], "support": n,
        "sel": [[w["outcomes"][j]["wdl"] for j in range(n)]
                for w in d["selection"]["worlds"]],
        "ass": [[w["outcomes"][k]["wdl"]
                 for k in range(len(d["assessment"]["candidate_indices"]))]
                for w in d["assessment"]["worlds"]],
        "assess_idx": d["assessment"]["candidate_indices"],
        "sel_means": d["selection_lock"]["selection_statistics"]["means"],
        "choices": dict(d["summary"]["choices"]),
        "proposed": dict(d["summary"]["proposed_indices"]),
        "arm_scores": dict(d["summary"]["arm_scores"]),
        "cand": {a: d["candidates"][a]["indices"]
                 for a in ("anchor", "continued", "control", "teacher")},
        "tr": d["teacher_added_indices"]["ranked"],
        "tn": d["teacher_added_indices"]["random"],
        "anchor_gk": d["proposals"]["anchor"]["group_keys"],
        "anchor_lg": d["proposals"]["anchor"]["learned_group_scores"],
        "union_keys": d["union"]["keys"],
    })

A = lambda r: {idx: r["ass"] and sum(w[k] for w in r["ass"]) / len(r["ass"])
               for k, idx in enumerate(r["assess_idx"])}
best = lambda r, s: max(s, key=lambda i: (r["sel_means"][i], -i))
cells = collections.defaultdict(list)
for r in recs:
    cells[(r["fit"], r["seat"], r["phase"])].append(r["index"])

def boot_ci(per_root, seed=20261016):
    rng, rows = random.Random(seed), collections.defaultdict(list)
    for c, ids in cells.items():
        for i in ids:
            if i in per_root:
                rows[c].append(per_root[i])
    out = []
    for _ in range(BOOT):
        s = n = 0
        for v in rows.values():
            m = len(v)
            for _ in range(m):
                s += v[rng.randrange(m)]; n += 1
        out.append(s / n)
    out.sort()
    return out[int(.025 * BOOT)], out[int(.975 * BOOT)]

# --- 1. teacher decomposition -------------------------------------------------
print("== teacher gain by source ==")
agg = collections.defaultdict(list)
for r in recs:
    d = r["arm_scores"]["teacher"] - r["arm_scores"]["anchor"]
    c = r["choices"]["teacher"]
    k = ("same" if c == r["choices"]["anchor"] else
         "ranked" if c in r["tr"] else "random" if c in r["tn"] else "other")
    agg[k].append(d)
for k, v in agg.items():
    print(f"  {k:7s} n={len(v):4d} mean={statistics.mean(v)*100:+.4f}pp "
          f"contribution={sum(v)/len(recs)*100:+.5f}pp")

# --- 2. substitutive decomposition -------------------------------------------
print("== substitutive decomposition ==")
for a in ("continued", "control"):
    win = loss = 0.0; nw = nl = 0
    for r in recs:
        d = r["arm_scores"][a] - r["arm_scores"]["anchor"]
        if r["choices"][a] == r["choices"]["anchor"]:
            continue
        if r["choices"][a] == r["proposed"][a]:
            win += d; nw += 1
        else:
            loss += d; nl += 1
    print(f"  {a}: wins {nw} {win/len(recs)*100:+.4f}pp | "
          f"displaced {nl} {loss/len(recs)*100:+.4f}pp | "
          f"net {(win+loss)/len(recs)*100:+.4f}pp")

# --- 3. additive counterfactuals ---------------------------------------------
print("== additive counterfactual arms (vs anchor) ==")
variants = {
    "K5 +continued":  lambda r: [r["proposed"]["continued"]],
    "K5 +control":    lambda r: [r["proposed"]["control"]],
    "K6 +both":       lambda r: [r["proposed"]["continued"], r["proposed"]["control"]],
    "K6 +2ranked":    lambda r: r["tr"],
    "K6 +2random":    lambda r: r["tn"],
    "K8 teacher":     lambda r: r["tr"] + r["tn"],
}
for name, fn in variants.items():
    per, miss = {}, 0
    for r in recs:
        am = A(r); anch = r["cand"]["anchor"]
        cand = list(anch) + [i for i in fn(r) if i is not None and i not in anch]
        p = best(r, cand)
        if p not in am:
            miss += 1; continue
        per[r["index"]] = (am[p] + 1) / 2 - r["arm_scores"]["anchor"]
    lo, hi = boot_ci(per)
    print(f"  {name:14s} n={len(per):3d} miss={miss:3d} "
          f"mean={statistics.mean(per.values())*100:+.4f}pp "
          f"CI[{lo*100:+.4f},{hi*100:+.4f}]")

# --- 4. evaluator noise and decidability -------------------------------------
print("== evaluator noise ==")
hd, within, pd_half, pdiff = [], [], [], []
for r in recs:
    n, sel = r["support"], r["sel"]
    if n < 2:
        continue
    a = [sum(sel[w][j] for w in range(64)) / 64 for j in range(n)]
    b = [sum(sel[w][j] for w in range(64, 128)) / 64 for j in range(n)]
    f = [(a[j] + b[j]) / 2 for j in range(n)]
    hd += [a[j] - b[j] for j in range(n)]
    mu = sum(f) / n
    within += [f[j] - mu for j in range(n)]
    for x in range(n):
        for y in range(x + 1, n):
            pdiff.append(f[x] - f[y])
            pd_half.append((a[x] - a[y]) - (b[x] - b[y]))
s64 = statistics.pvariance(hd) / 2; s128 = s64 / 2
vw = statistics.pvariance(within)
sp = math.sqrt(statistics.pvariance(pd_half) / 8)
print(f"  SE128 candidate = {math.sqrt(s128):.5f} WDL ({math.sqrt(s128)/2*100:.3f}pp)")
print(f"  SE128 paired    = {sp:.5f} WDL ({sp/2*100:.3f}pp)")
print(f"  true between-candidate SD = {math.sqrt(max(vw-s128,0)):.5f} WDL; "
      f"reliability = {(vw-s128)/vw:.3f}")

print("== decidability and repeatability ==")
for arm in ("anchor", "teacher"):
    gaps, ag, tot = [], 0, 0
    for r in recs:
        idx = sorted(set(r["cand"][arm]))
        if len(idx) < 2:
            continue
        v = sorted((r["sel_means"][i] for i in idx), reverse=True)
        gaps.append(v[0] - v[1])
        x = max(idx, key=lambda j: (sum(r["sel"][w][j] for w in range(64)), -j))
        y = max(idx, key=lambda j: (sum(r["sel"][w][j] for w in range(64, 128)), -j))
        ag += (x == y); tot += 1
    med = statistics.median(gaps)
    print(f"  {arm:8s} median margin {med/2*100:.2f}pp = {med/sp:.2f} SE | "
          f">2SE {sum(1 for g in gaps if g > 2*sp)/len(gaps):.3f} | "
          f"split-half agreement {ag}/{tot} = {ag/tot:.3f}")

# --- 5. equal-budget frontier -------------------------------------------------
print("== equal-budget frontier (choose on first N, score on last 32) ==")
def armset(r, nm):
    a4 = r["cand"]["anchor"]
    return {"K4": a4,
            "K5": list(dict.fromkeys(a4 + ([r["proposed"]["continued"]]
                                           if r["proposed"]["continued"] is not None else []))),
            "K6": list(dict.fromkeys(a4 + r["tr"])),
            "K8": r["cand"]["teacher"]}[nm]

def per_root(nm, N):
    out = {}
    for r in recs:
        s = [i for i in armset(r, nm) if i < r["support"]]
        if not s:
            continue
        p = max(s, key=lambda i: (sum(r["sel"][w][i] for w in range(N)), -i))
        out[r["index"]] = (sum(r["sel"][w][p] for w in range(128 - HOLD, 128)) / HOLD + 1) / 2
    return out

for nm, N in [("K4",24),("K4",32),("K4",48),("K4",64),("K4",95),
              ("K5",24),("K5",32),("K5",48),("K5",64),
              ("K6",16),("K6",24),("K6",32),("K6",48),("K6",64),
              ("K8",12),("K8",16),("K8",24),("K8",32),("K8",48)]:
    v = per_root(nm, N)
    print(f"  {nm} N={N:3d} budget={int(nm[1])*N:4d} "
          f"score={statistics.mean(v.values())*100:.4f}%")

for (x, nx), (y, ny) in [(("K4",48),("K8",24)), (("K4",64),("K8",32)),
                         (("K4",32),("K8",16)), (("K4",64),("K5",51)),
                         (("K4",48),("K5",38)), (("K4",64),("K8",64)),
                         (("K4",64),("K5",64))]:
    a, b = per_root(x, nx), per_root(y, ny)
    d = {k: (b[k] - a[k]) * 100 for k in a if k in b}
    lo, hi = boot_ci(d, seed=777)
    print(f"  {y}x{ny} - {x}x{nx}: {statistics.mean(d.values()):+.4f}pp "
          f"CI[{lo:+.4f},{hi:+.4f}]")
```

---

## Sources

Primary artifacts verified for this review:

- `outputs/large_corpus_optimization_128/{result.json,report.md,final_status.json,acquisition_gate.json}`
- `outputs/large_corpus_optimization_128/main/{roots,selections}/` (720 roots)
- `outputs/equal_budget_breadth_arena_128/recovery_03/report.md`
- `outputs/structured_proposer_128/{progress.json,training_feature_audit.json,fits/}`
- `docs/engineering/large_corpus_optimization_128.md`
- `docs/engineering/structured_proposer_128.md`
- `docs/engineering/learned_proposal_{screen,arena,attribution}_result.md`
- `docs/engineering/shared_cards_151_result.md`
- `docs/engineering/verification_policy.md`
- `docs/strategy/{equal_budget_breadth_arena_128,independent_superhuman_review_after_134_137,superhuman_information_set_search_roadmap_after_132,stochastic_refill_vs_blind_reserve_architecture_correction}.md`
- `docs/experiments/human_resolver_ceiling_calibration_074.md`,
  `docs/human_resolver_feedback_074.md`
- `src/splendor_ai/alphazero/run_afterstate_advantage.py:103,149`
- `src/splendor_ai/alphazero/run_rollout_selector_gameplay.py:86,94`
- `src/splendor_ai/alphazero/world_budget_160.py:60,80,98`
- `scripts/analysis/learned_proposal_arena.py:101`
- `tools/large_corpus_optimization_128/engine.py`
- `AGENT_CURRENT.md`, `AGENT_HANDOFF.md`
