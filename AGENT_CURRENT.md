# Agent Current State

Use this file first in a fresh chat. `AGENT_HANDOFF.md` is the long historical log.

## Goal

Build a strong 2-player Splendor AI with exact rules, legal observations/actions, structured legal-action scorers, search/PGS diagnostics, and disciplined promotion gates.

## Current Active Baselines

- Active legal-observation prior: `outputs/legal_action_scorer_035_legal_observation/legal_action_attention_scorer_best.pt`
- Frozen historical strength baseline: `outputs/legal_action_scorer_022_attention_distill/legal_action_attention_scorer_best.pt`
- Conservative strong baseline: `outputs/legal_action_scorer_016_policy_guided_mix/legal_action_scorer_best.pt`
- Rejected diagnostic candidate: `outputs/legal_action_scorer_028_early_pgs_distill/legal_action_attention_scorer_best.pt`
- Rejected constrained candidate: `outputs/legal_action_scorer_030_constrained_pref/legal_action_attention_scorer_best.pt`
- Rejected trajectory candidate: `outputs/legal_action_scorer_031_trajectory_ppo/legal_action_attention_scorer.pt`
- Rejected group-relative candidate: `outputs/legal_action_scorer_032_group_relative_ppo/legal_action_attention_scorer.pt`
- Rejected clean LOO actor candidate: `outputs/legal_action_scorer_040_clean_rloo_actor/legal_action_attention_scorer.pt`
- Rejected clean LOO magnitude correction: `outputs/legal_action_scorer_041_clean_rloo_actor_lr1e5/legal_action_attention_scorer.pt`
- Rejected reserve counterfactual calibration candidate: `outputs/legal_action_scorer_042_reserve_counterfactual/legal_action_attention_scorer.pt`
- Rejected reserve hard-target head diagnostic: `outputs/legal_action_scorer_043_reserve_hardtop_head/legal_action_attention_scorer.pt`
- Rejected reserve hard-target full-model diagnostic: `outputs/legal_action_scorer_044_reserve_hardtop_fullmodel/legal_action_attention_scorer.pt`
- Rejected rollout-credit candidate: `outputs/legal_action_scorer_034_rollout_credit_anchored/legal_action_attention_scorer.pt`
- Current legal late-game value candidate, not promoted: `outputs/value_model_009_puct035_legal_late40/value_model.pt`
- Historical contaminated value model, diagnostics only: `outputs/value_model_006_late40_outcome_currentgen/value_model_best.pt`
- Frozen clean entity actor parent: `outputs/canonical_entity_scale_004/canonical_entity_action_scorer_best.pt`
- Current clean actor-critic smoke candidate, not promoted: `outputs/canonical_entity_actor_005_smoke/canonical_entity_actor.pt`
- Rejected clean actor-critic no-movement generation: `outputs/canonical_entity_actor_006_seed90000/canonical_entity_actor.pt`
- Current clean actor-critic movement candidate, not promoted: `outputs/canonical_entity_actor_007_kl_target/canonical_entity_actor.pt`

## Current Conclusions

- Foundation reset is underway. `canonical_public_belief_key` is now the
  versioned exact observer-relative contract for new search/training work: it
  preserves public card identities (including purchased cards), preserves the
  public tier of a blind reserve without exposing its identity, and ignores
  incidental board/reserve/noble/deck ordering. The 295-wide tensor remains
  frozen exclusively for legacy checkpoints; do not extend new work through it.
- PUCT information-state keys and edges now use canonical public-belief and
  successor-action keys. Equivalent take/return encodings with the same net
  state transition share one edge and are scored through one representative
  action. This repairs a search branching artifact but is not an
  engine-strength promotion; the next task is a clean entity/action model on
  this contract.
- A clean `CanonicalEntityActionScorerModel` is now available for that next
  training line. It consumes variable-length canonical state entities and
  legal-action entities directly, uses no positional embeddings, and has
  permutation-equivalence and legal-hidden-reserve tests. Checkpoint inference
  recognizes `canonical_entity_action_scorer`; it is randomly initialized and
  no historical checkpoint or historical 295-vector is used to initialize it.
  The first clean CPU pilot is `outputs/canonical_entity_pilot_003`: eight
  fresh Greedy-vs-depth1-information-legal-ShallowSearch games, 518 snapshot
  positions, zero stalls/timeouts, and no learned checkpoint in collection.
  A randomly initialized 32-hidden model trained for three epochs achieved
  held-out policy loss 3.773 versus a uniform 3.846 (60 positions from one
  held-out game). This passes a representation/learning viability gate only;
  it is far too small for promotion or strength comparison. The greedy-only
  pilot `002` was rejected (7/20 timeouts), and depth2/two-determinization
  search was rejected as a CPU corpus configuration after exceeding 15 minutes
  for 30 games without completing. Scale the verified depth1/one-
  determinization corpus to a game-held-out block before considering a GPU.
- The scaled clean block is `outputs/canonical_entity_scale_004`: 50 fresh
  rule-based legal games, 3,180 positions, zero stalls/timeouts, and 8
  game-held-out validation games. The random 32-hidden entity model trained
  four epochs reached aggregate held-out policy loss 2.953 versus uniform
  3.406; all eight held-out games beat their own uniform baseline. Its first
  strength readout is deliberately non-promotional: 15-5 versus Greedy, 5-15
  versus the legal depth1 shallow-search teacher, and 3-17 versus legal `035`
  on 20 games each (the `035` block used paired seeds). It is a viable clean
  behavioral apprentice, not a champion. Do not scale pure imitation alone.
  `entity_scale_004` is now frozen as the clean actor parent.
- The first clean actor-critic smoke generation is
  `outputs/canonical_entity_actor_005_smoke`. Its actor sees only canonical
  public entities; a separate training-only critic sees hidden deck cards and
  opponent blind-reserve identities. The 12-game rollout league included the
  frozen parent, Greedy, legal shallow depth 1, `016`, `035`, and PUCT035 s8.
  Full-support sampling, exact stored behavior probabilities, GAE(lambda 0.95),
  clipped ratios, entropy, KL, and paired parent gates are implemented. All
  smoke gates passed, including zero rollout failures, behavior reconstruction
  error 6.9e-7, max ratio 1.016, mean KL 2.0e-5, and normalized entropy 0.841.
  The candidate beat its parent 12-8 in 20 paired games. This validates the
  learning mechanism but is too small for promotion; keep `004` frozen and
  require independent larger generations plus broad league confirmation.
- The independent larger generation `entity_actor_006_seed90000` resolves the
  smoke ambiguity. It used 72 fresh games (12 per league opponent), 2,364 actor
  decisions, and the unchanged `005` optimizer settings. The update remained
  clean but was extremely small: mean KL 6.88e-5, max ratio 1.037, normalized
  entropy 0.834, and policy loss -5.0e-6. It scored 46-54 against frozen `004`
  over 100 paired games. Same-seed controls show essentially no broad-policy
  movement: `006`/`004` scored 7/8 wins versus `016`, 2/1 versus `035`, and
  54/55 versus Greedy on 60 games each. `006` scored 9-51 versus legal shallow
  depth 1, the same 15% share as `004`'s prior 5-15 result. Reject `006` and do
  not repeat the identical one-epoch, 1e-5 actor update. Before more expensive
  games, add an effective-movement gate and test a KL-targeted update schedule
  that is materially larger while retaining the frozen-parent safety bound.
- `entity_actor_007_kl_target` implements that movement gate and KL-targeted
  schedule. On 36 fresh league games, actor learning rate 3e-5 reached mean KL
  0.00241 in the first epoch and stopped automatically. Mean probability TV was
  0.0271, 4.93% of rollout-state argmaxes changed, centered-logit movement was
  0.0424, and relative parameter movement was 0.000654; max ratio remained
  1.194 and entropy 0.842. The initial 60-game parent gate scored 37-22-1, but
  an independent 100-game paired block scored only 52-48. Keep `004` frozen and
  retain `007` as proof that bounded effective movement is possible, not as a
  promoted parent. Do not launch expensive broad/search confirmation until the
  same recipe produces a repeatable gain from another independent rollout seed.
- The previous observation encoded every opponent reserved-card identity. This was
  illegal for cards reserved blind from a deck. Reserve provenance is now tracked:
  visible reserves remain public, while deck reserves are visible only to their
  owner. Old snapshots deserialize conservatively as unknown/private.
- `022` remains the frozen historical strength baseline, but it was trained under
  the leaked observation contract. Do not use it as the prior for legal MCTS or
  new training.
- `016`, `022`, and `value006` all store 295-wide inputs, as does `035`; shape
  does not identify legality because masking preserved tensor width. Artifact
  chronology and training provenance establish that the first three predate the
  repair. At inference the fixed encoder masks them, but their learned feature
  semantics remain contaminated.
- `035` is the first corrected-observation prior. It was initialized from `022`
  and retrained on the original six-corpus mix. Historical snapshots cannot
  recover which reserves were public, so `035` is conservatively under-informed
  on those rows.
- `035` vs `022` under the fixed encoder is 48-52 over 100 games. Treat them as
  tied in strength; use `035` for legal inference, while recognizing that its
  initialization and historical teacher labels are not clean-from-scratch.
- Historical shallow search and PGS consumed the true hidden-deck order during
  simulated refills. Search, PGS, benchmark, and corpus construction now support
  observer-relative determinization, and generic benchmark/corpus search
  defaults are information-legal. Oracle search requires an explicit legacy
  flag where available.
- Evaluation-only information-set PUCT is implemented with a fresh
  determinization per simulation, acting-player observation keys, legal `035`
  priors, bounded heuristic leaves, optional checkpoint value leaves, and paired
  benchmark telemetry.
- PUCT now supports variable-length batched prior/value inference, Q-preserving
  virtual visits, detailed phase timing, and bounded information-state tree
  retention. Cross-move tree reuse is opt-in because it improved reuse rate but
  has not passed a direct strength A/B.
- Profiling removed two concrete bottlenecks from legal-action feature encoding:
  pre-action affordability is computed once per action set, and post-action
  affordability is memoized by resulting token holdings. Batched and individual
  feature/model outputs are covered by equality tests.
- Do not start MCTS self-play training yet. At 64 simulations, PUCT is tied within
  wide confidence intervals against `035`, plain search, and PGS016, and trails
  frozen `022` directionally. Doubling to 128 simulations did not improve the
  readout and exceeded one second per searched move.
- `value_model_006` is superseded for live MCTS because its training observations
  predate the legality repair. Its historical 9-11 leaf A/B remains diagnostic
  only.
- New legal PUCT outcome models `007`-`009` do not yet pass the leaf gate.
  Unrestricted value007 was unstable across simulation budgets; the legal
  late-turn value009 tied heuristic leaves 10-9-1 at s64. Keep value009 as a
  diagnostic candidate and do not use it for self-play target generation.
- Fresh 100-game legal strength confirmations show PUCT035 is competitive but
  still not a clean engine for self-play target generation:
  - s64 vs policy035: 50-50, score share 0.50, CI95 [0.43, 0.57];
  - s128 vs policy035: 54-46, score share 0.54, CI95 [0.46, 0.62];
  - s64 vs legal shallow search d2/d4: 52-48, score share 0.52, CI95 [0.41, 0.63];
  - s64 vs legal PGS035 d2/d4: 59-41, score share 0.59, CI95 [0.51, 0.68].
  This beats legal PGS on the tested block, but does not clearly beat the
  underlying policy or plain legal search.
- Late-only n-step PUCT leaves are now the first promising leaf/backup-quality
  improvement. With depth 2, width 1, candidate limit 4, and turn >= 40 gating,
  PUCT035 s64 scored 120-80 over two 100-game paired blocks vs policy035, score
  share 0.60, CI95 [0.54, 0.66], mean margin +0.905. Plain heuristic-leaf PUCT
  on the same seed blocks scored 104-96, score share 0.52, CI95 [0.465, 0.575],
  mean margin 0.0. Treat this as a candidate engine improvement that deserves
  search/PGS confirmation before self-play target generation.
- Search/PGS confirmation was mixed. On the same legal shallow-search seed block
  where plain PUCT was 52-48, late-n-step PUCT scored 62-37-1, score share
  0.625, CI95 [0.52, 0.725]. On the same legal PGS035 seed block where plain
  PUCT was 59-41, late-n-step PUCT scored only 52-48, score share 0.52, CI95
  [0.44, 0.60]. Do not promote late-n-step PUCT to self-play engine yet.
- The optimized engine can now run 128 simulations in approximately the old
  64-simulation time budget, but the added search still does not beat its legal
  prior. On the same 10 paired seeds, sequential s128 scored 9-11 against
  policy035 and corrected batch-8 s128 scored 7-13. Performance engineering is
  no longer the primary blocker; leaf/backup signal quality is.
- Do not promote `028`.
- Do not continue value-guided live PGS right now.
- Do not continue early token/reserve PGS imitation as the main recipe.
- `028` improved PGS agreement but regressed against search.
- The regression is not late seat1, not loop/fallback related, and not simply worse immediate search agreement.
- The likely issue is conversion efficiency: `028` turns early decisions into fewer total buys, fewer scoring buys, and fewer bought points than `022`.
- A conversion diagnostic now confirms that `028` shifts the 10 regression seeds toward more reserved buys and fewer visible/scoring buys; do not train again until this pattern is interpreted.
- `029` tested a conservative filter of the `028` PGS corpus. It partially recovered vs search and conversion, but lost badly to both `022` and `016`; do not promote it.
- `030` tested the strongest defensible version of this line: high-margin conversion-filtered pairwise corrections, explicit `022` regression anchors, low learning rate, one epoch, and pre-benchmark drift gates.
- `030` passed drift gates and looked promising in the first 100 search games, but failed fresh 200-game direct/search confirmation and still regressed conversion on the original 10 seeds.
- Retain `022` and stop supervised PGS-imitation/patch variants. Small top-action drift can still produce large trajectory drift, so another `031` weight/filter sweep is not justified.
- Genuine stochastic trajectory training is now implemented. `031` used signed terminal returns and `032` used matched-seed group-relative returns.
- Both trajectory candidates stayed close to `022`, but both failed the sensitive-seed conversion gate. Do not promote either.
- Do not respond by sweeping PPO learning rates or KL weights. The remaining problem is temporal/action credit assignment, not optimizer stability.
- Complete-rollout action credit is now implemented and tested. `033` showed that target-state KL does not constrain global drift; `034` added broad anchor KL and passed drift/target gates.
- `034` still failed conversion badly. Complete rollouts from a sparse root set inherit bias from the continuation policy and are not sufficient as direct policy targets.

## Latest Important Results

### Legal Observation Repair and `035`

Implemented reserve provenance across state transitions, replay snapshots,
diagnostics, and observation encoding. New tests cover:

- own blind reserves remaining visible;
- opponent blind reserves being hidden;
- opponent visible reserves remaining public;
- old snapshots defaulting to private/unknown provenance;
- determinization invariance to the simulator's true hidden-deck order.

`035` training:

- Parent: `022`.
- Data: the original `022` six-corpus mix.
- Two epochs, batch size 256, learning rate `1e-5`.
- Best epoch: 2.
- Validation loss: 1.0684.
- Validation accuracy: 65.84%.

### Evaluation-Only PUCT Gate

Implementation:

- `src/splendor_ai/bots/puct_bot.py`
- `src/splendor_ai/eval/run_puct_benchmark.py`
- `run_puct_benchmark.py`
- `tests/test_puct_bot.py`
- `tests/test_run_puct_benchmark.py`

The runner uses the same seed twice with seats swapped and reports paired
bootstrap confidence intervals. At 64 simulations PUCT delivered about 110-113
simulations/second and 0.56-0.58 seconds per searched move on CPU.

Initial 10-pair elimination results:

- PUCT035 s64 vs policy035: 10-9-1, score share 0.525,
  CI95 [0.35, 0.70].
- PUCT035 s64 vs plain search: 11-9, score share 0.55,
  CI95 [0.35, 0.75].
- PUCT035 s64 vs PGS016: 11-9, score share 0.55,
  CI95 [0.35, 0.75].
- PUCT035 s64 vs frozen policy022: 7-13, score share 0.35,
  CI95 [0.15, 0.55].
- PUCT035 s128 vs policy035: 11-9, score share 0.55,
  CI95 [0.35, 0.75], but 1.16 seconds per searched move.
- PUCT035 + value006 vs heuristic-leaf PUCT at s32: 9-11,
  score share 0.45, CI95 [0.25, 0.65].

Artifacts are under `outputs/benchmarks/puct035_*.json`.

Interpretation: PUCT is operational and competitive, but it has not passed the
engine-strength gate. The performance work recommended by this first gate is
documented below. Do not generate self-play targets from this version.

### Batched PUCT and Profiling

Implemented:

- variable-length batched policy scoring with legal-action padding masks;
- batched value-model evaluation;
- batch leaf collection with Q-preserving virtual visits;
- optional cross-move information-state tree retention with new-game/seat reset;
- phase telemetry for determinization, traversal, inference, batch calls, and
  retained-root reuse;
- exact action-set feature caching for repeated state-level and token-level
  affordability calculations.

Profile changes on an initial 128-simulation search:

- prior launches: 128 sequential calls to about 18 batch calls at batch size 8;
- model forward time fell to roughly 8% of profiled runtime;
- remaining dominant cost is legal-action feature construction, followed by
  legal-action generation/tree traversal;
- complete-game throughput is approximately 224-299 simulations/second, compared
  with 110-113 before this work;
- 128 simulations now average roughly 0.43-0.57 seconds per searched move,
  compared with 1.16 seconds before optimization.

Fresh paired elimination results:

- batched s128 vs frozen policy022, seed 41200: 14-6, score share 0.70,
  CI95 [0.55, 0.85];
- batched s128 with reuse vs policy035, seed 41300: 5-15;
- batched s128 without reuse on the same seeds: 6-14;
- sequential optimized s128 without reuse on the same seeds: 9-11;
- batch-4 s128 without reuse on the same seeds: 8-12;
- Q-preserving batch-8 s128 without reuse on the same seeds: 7-13.

The 022 result is not sufficient to promote the engine because the corrected
legal policy035 is the relevant prior baseline, and PUCT fails to add value
against it. Do not spend more compute tuning batch size on 20-game readouts.
Keep batch size 8 for throughput experiments, keep tree reuse opt-in, and address
leaf/backup quality before repeating larger strength gates.

### Information-Legal Search and Value Re-baseline

Audit:

- `016`, `022`, `035`, and `value006` all declare `observation_size=295`; this
  cannot distinguish leaked and masked encoders.
- `016`, `022`, and `value006` predate the masking repair.
- `035` is the corrected masked fine-tune and remains the legal prior.
- Shallow search and PGS previously called `env.step` against the simulator's
  true hidden deck, making historical search teachers and opponents mild
  deck-order oracles.

Implemented:

- shared `determinize_for_observer` hidden-card sampling;
- information-legal shallow search/PGS that average action values across
  determinizations;
- true-deck-order and opponent-private-reserve invariance tests;
- legal search defaults in generic benchmark and replay-corpus runners;
- PUCT replay-corpus support with manifests and full state snapshots;
- turn-gated PUCT value leaves.

Legal PUCT outcome corpora:

- `data/corpus_puct035_legal_selfplay_036`: 100 games at s64, 6,370 states;
- `data/corpus_puct035_legal_selfplay_037`: 200 games at s32, 12,822 states;
- total: 300 games, 19,192 states, zero stalls/timeouts.

Value experiments:

- `value007`: value006 fine-tuned on the first 100 games. Whole-corpus fit
  improved, but game-held-out performance was weak and unrestricted s64 leaves
  lost 7-13 to heuristic leaves.
- `value008`: trained from scratch on all 300 legal games. It strongly overfit
  trajectories; early-state terminal outcomes remain weak labels.
- `value009`: trained from scratch on turn 40+ states only. Final checkpoint
  reached held-out late-state MSE 0.1685, sign accuracy 0.576, and Pearson 0.354.
  With heuristic leaves before turn 40 and value009 afterward, the direct s64
  result was 10-9-1, score share 0.525, CI95 [0.375, 0.675].

Clean elimination benchmarks:

- PUCT035 s64 vs legal two-determinization shallow search: 8-12.
- PUCT035 s64 vs legal two-determinization PGS035: 8-12.
- Policy035 vs policy022 under the fixed encoder: 48-52.

Conclusion: the contamination recommendations were meaningful and are now
enforced in code. The legal value data improved offline late-state prediction,
but not enough to improve PUCT reliably. Do not run a 100-200 game value
confirmation or start policy self-play training from these leaves.

### Clean Legal Strength Table and Rollout Leaf Probe

Fresh 100-game paired-seat confirmations:

- `outputs/benchmarks/puct035_s64_vs_policy035_paired100_seed44600.json`:
  PUCT035 s64 vs policy035 was 50-50, score share 0.50, CI95 [0.43, 0.57],
  mean margin +0.04.
- `outputs/benchmarks/puct035_s128_vs_policy035_paired100_seed44900.json`:
  PUCT035 s128 vs policy035 was 54-46, score share 0.54, CI95 [0.46, 0.62],
  mean margin +0.21.
- `outputs/benchmarks/puct035_s64_vs_legal_search_d2d4_paired100_seed44700.json`:
  PUCT035 s64 vs information-legal shallow search depth 2 with 4
  determinizations was 52-48, score share 0.52, CI95 [0.41, 0.63], mean margin
  -0.06.
- `outputs/benchmarks/puct035_s64_vs_legal_pgs035_d2d4_paired100_seed44800.json`:
  PUCT035 s64 vs information-legal PGS035 depth 2 with 4 determinizations was
  59-41, score share 0.59, CI95 [0.51, 0.68], mean margin +1.39.

Interpretation: PUCT035 is a useful legal search opponent and may now be better
than legal PGS035, but it still fails the most important engine gate: clearly
improving on the policy prior it is supposed to strengthen. Do not start
AlphaZero-style policy self-play from this engine yet.

Implemented a gated rollout-leaf evaluator in PUCT:

- `PuctConfig.rollout_leaf_count`, `rollout_leaf_max_depth`,
  `rollout_leaf_top_k`, `rollout_leaf_temperature`, and `rollout_leaf_policy`.
- Benchmark/replay CLIs expose and record the rollout-leaf config.
- `rollout_leaf_policy=random` is the default rollout mode when rollouts are
  enabled; `heuristic` remains available for diagnostics.

Rollout-leaf probes:

- Heuristic-guided rollout leaves at s64, one rollout, depth 8 timed out before
  finishing 20 games in one hour. This is too expensive for current CPU gates.
- Random rollout leaves at s64, one rollout, depth 8 vs policy035 on seed 45100
  scored 9-11 and took 223 seconds.
- Plain heuristic leaves on the same seed block scored 10-10 and took 100
  seconds.

Conclusion: keep rollout leaves as an experimental option, but do not promote
random rollout leaves or spend a larger gate on them. Naive heuristic rollouts
need major optimization before they are viable.

### Late N-Step Leaf Candidate

Implemented a separate optimized n-step leaf evaluator in PUCT:

- `PuctConfig.n_step_leaf_depth`, `n_step_leaf_width`,
  `n_step_leaf_candidate_limit`, `n_step_leaf_min_turn_index`, and
  `n_step_leaf_max_turn_index`.
- Value checkpoints still take precedence over n-step leaves; n-step leaves take
  precedence over rollout leaves; otherwise PUCT falls back to the original
  heuristic leaf.
- The evaluator prunes legal actions before scoring, follows a narrow
  heuristic-guided line/beam, and is exposed in benchmark/replay CLI metadata.

Early probes:

- Depth 2, width 1, candidate limit 4 from all turns on seed 45200: 13-7 vs
  policy035, identical to plain PUCT on the same seeds, but about 3x slower.
- Depth 1, width 1, candidate limit 4 from all turns on seed 45300: 10-10,
  identical to plain PUCT on the same seeds, but about 2x slower.
- Per-move leaf-value memoization produced no useful cache hits; leaf states are
  mostly unique under the current traversal.

Useful candidate:

- Late-only n-step, depth 2, width 1, candidate limit 4, turn >= 40:
  - seed 45300, 20 games: 13-7 vs policy035, compared with 10-10 for plain PUCT
    on the same seeds;
  - seed 45400, 100 games: 56-44, score share 0.56, CI95 [0.47, 0.65], mean
    margin +0.43;
  - same seed 45400 plain PUCT: 49-51, score share 0.49, CI95 [0.42, 0.57],
    mean margin -0.56;
  - seed 45500, 100 games: 64-36, score share 0.64, CI95 [0.56, 0.72], mean
    margin +1.38;
  - same seed 45500 plain PUCT: 55-45, score share 0.55, CI95 [0.48, 0.63],
    mean margin +0.56.

Combined 200-game policy gate:

- Late n-step PUCT035 s64: 120-80, score share 0.60, CI95 [0.54, 0.66], mean
  margin +0.905.
- Plain PUCT035 s64 on the same seeds: 104-96, score share 0.52, CI95
  [0.465, 0.575], mean margin 0.0.

Interpretation: this is the first leaf-quality change with a credible positive
policy-gate signal. It is not yet enough to start self-play: confirm against
legal shallow search and legal PGS first, because the earlier clean table showed
plain PUCT was only tied with legal search and merely ahead of legal PGS.

### Late N-Step Search/PGS Confirmation

Confirmed the active late-n-step candidate against the two legal search
baselines:

- `outputs/benchmarks/puct035_nstepd2w1c4_late40_s64_vs_legal_search_d2d4_paired100_seed44700.json`:
  late-n-step PUCT035 s64 vs information-legal shallow search depth 2 with 4
  determinizations was 62-37-1, score share 0.625, CI95 [0.52, 0.725], mean
  margin +1.62, margin CI95 [0.09, 3.09].
- Same seed/block plain baseline:
  `outputs/benchmarks/puct035_s64_vs_legal_search_d2d4_paired100_seed44700.json`
  was 52-48, score share 0.52, CI95 [0.41, 0.63], mean margin -0.06.
- `outputs/benchmarks/puct035_nstepd2w1c4_late40_s64_vs_legal_pgs035_d2d4_paired100_seed44800.json`:
  late-n-step PUCT035 s64 vs information-legal PGS035 depth 2 with 4
  determinizations and prior weight 3000 was 52-48, score share 0.52, CI95
  [0.44, 0.60], mean margin +0.60, margin CI95 [-0.53, 1.78].
- Same seed/block plain baseline:
  `outputs/benchmarks/puct035_s64_vs_legal_pgs035_d2d4_paired100_seed44800.json`
  was 59-41, score share 0.59, CI95 [0.51, 0.68], mean margin +1.39.

Interpretation: late-n-step is not a simple promotion. It clearly improves the
legal shallow-search matchup and the policy035 gate, but it weakens the legal
PGS matchup relative to plain PUCT. Keep it as the active diagnostic candidate,
but do not generate self-play/root-visit targets from it until the PGS regression
is understood or a hybrid leaf rule passes all three gates.

### Hybrid N-Step Probes

Implemented extra optional n-step controls:

- `n_step_leaf_min_max_score`: only enable n-step leaves when either player has
  reached at least this score.
- `n_step_leaf_weight`: blend n-step value with the original heuristic leaf
  value; default 1.0 preserves previous behavior.

The goal was to keep the late-n-step policy/search gains while restoring the
plain-PUCT PGS matchup. Probes used the same 20-game seed blocks as the earlier
diagnostics:

- Turn >= 50 vs legal PGS035: 10-10.
- Turn >= 60 vs legal PGS035: 13-7, but the 100-game confirmation was only
  52-47-1, still weaker than plain PUCT's 59-41 on the same block.
- Turn >= 60 vs policy035: 11-9.
- Turn >= 60 vs legal shallow search: 12-8.
- Turn >= 40 and max score >= 10 vs legal PGS035: 9-11.
- Turn >= 40 and max score >= 13 vs legal PGS035: 12-8, but policy035 and legal
  shallow search probes were both 10-10.
- Turn >= 40, weight 0.5 vs policy035: 12-8; vs legal PGS035: 9-11.
- Turn >= 40, weight 0.25 vs policy035: 11-9; vs legal PGS035: 10-10.

Conclusion: simple turn gating, score gating, and scalar blending do not yet
produce a robust hybrid. The PGS regression is not just excessive late-game
usage; it appears tied to which root/trajectory contexts n-step changes. Do not
run a larger hybrid gate from these settings.

### PUCT Root-Decision Diff Audit

Implemented:

- `PuctSearchStats` now records the selected root action and per-action visits,
  Q, and prior.
- `run_puct_decision_diff.py` evaluates plain and candidate PUCT on the exact
  same roots while separately playing matched plain-driven and candidate-driven
  trajectories against an information-legal opponent.
- The runner skips the plain search exactly when the candidate performed zero
  n-step evaluations, records determinization variance and root rankings for
  changed choices, and summarizes action transitions and outcome flips.

Initial s64 audit against legal PGS035 used three fresh paired seeds, six seat
conditions, and twelve full games:

- 381 comparable PUCT roots; 26 changed, a 6.8% change rate; all changes were
  late-game roots.
- Outcomes: candidate improved two conditions, regressed one, and left three
  unchanged. Mean candidate-minus-plain score-margin delta was +0.33. This is a
  diagnostic sample, not a strength estimate.
- 15/26 changes were `BUY_VISIBLE -> BUY_VISIBLE`; the issue is mostly card
  selection within an action type, not a broad reserve/take phase error.
- In 23/26 changed roots, n-step selected an action with lower policy prior than
  plain PUCT's choice. The selected action's Q rose by +0.20 on average under
  n-step.
- In the one regressed condition, all three changed roots overrode a higher-prior
  choice and the selected-action Q uplift averaged +0.38. The earliest changed
  candidate-trajectory root, turn 43, moved a tier-1 visible buy from prior
  0.698 to prior 0.077 because n-step raised its Q from -0.434 to +0.014.

Interpretation: the PGS regression is consistent with narrow n-step producing
occasional optimistic low-prior outliers, not with n-step being broadly harmful.
Do not add another turn/score gate. The next candidate should robustify the leaf
delta itself, for example by clipping n-step's deviation from the baseline
heuristic or aggregating multiple plausible short lines, then use the diff audit
before any full strength gate.

### Clipped N-Step Leaf Candidate

Implemented `n_step_leaf_delta_clip` across PUCT, benchmark metadata, replay
corpus configuration, and the root-decision audit. When configured, the leaf
value is:

`heuristic + weight * clamp(n_step - heuristic, -clip, +clip)`

The default is `None`, preserving the previous engine exactly. Telemetry records
raw/applied absolute leaf deltas and the number of clipped evaluations.

Selected one threshold, `0.20`, from the root audit: it is just above the mean
Q uplift on improved changed roots (+0.18) and below the regressed-root mean
(+0.38). Twenty-game paired-seat probes reused the first ten seed pairs of each
historical 100-game block:

- vs policy035, seed 45400: clipped 12-8, score share 0.60, mean margin +1.75;
  plain was 11-9/+0.15 and uncapped late n-step was 13-7/+1.35.
- vs legal shallow search d2/d4, seed 44700: clipped 15-5, score share 0.75,
  mean margin +4.30; plain was 13-7/+1.75 and uncapped was 18-1-1/+6.05.
- vs legal PGS035 d2/d4, seed 44800: clipped 12-8, score share 0.60, mean margin
  +3.20; plain was 12-8/+1.65 and uncapped was 9-11/-0.35.

Across the three probes, clipping activated on 39-41% of evaluated n-step
leaves. Mean absolute raw delta was 0.206-0.214 and mean applied absolute delta
was 0.107-0.109. The cap therefore materially changed the signal rather than
passing through almost all leaves.

Interpretation: `clip=0.20` is the first simple hybrid to preserve a positive
profile on policy, legal search, and legal PGS slices. The samples are only 20
games each with wide intervals, and the seed blocks are reused diagnostics; do
not promote yet. Give this exact fixed configuration fresh paired confirmation
before considering self-play targets. Runtime remains expensive because the
n-step line is evaluated before clipping.

Fresh 100-game paired confirmations froze the exact configuration and used
previously unused seed blocks:

- vs policy035, seed 46200: 58-42, score share 0.58, paired CI95
  [0.49, 0.67], mean margin +1.34, margin CI95 [+0.27, +2.41]. There were no
  draws, timeouts, or non-completed games; PUCT wins were balanced by seat
  (30 as seat 0, 28 as seat 1).
- vs legal PGS035 d2/d4, seed 46300: 50-49-1, score share 0.505, paired CI95
  [0.405, 0.605], mean margin +0.90, margin CI95 [-0.47, +2.25]. There were no
  timeouts or non-completed games; PUCT wins were balanced by seat (26 as seat
  0, 24 as seat 1).
- Clipping remained stable rather than seed-specific: 41.2% of evaluated
  n-step leaves were clipped against policy and 41.8% against PGS. Mean raw
  absolute deltas were 0.216/0.221 and applied deltas were 0.109/0.109.

Interpretation: the candidate has a credible positive margin signal against its
policy prior, but it is tied with legal PGS on the fresh confirmation. The PGS
repair observed on the reused 20-game slice did not establish a fresh advantage.
Do not promote to self-play/root-visit target generation. The legal-search
confirmation was correctly skipped because it was conditional on both policy
and PGS passing. Do not sweep the clip threshold.

### `028` Training

`028` was trained from `022` using broad anchors plus focused early token/reserve PGS distillation:

- Focused corpus: `data/corpus_pgs016_early_token_reserve_028/replays.jsonl`
- Rows: 2,053
- PGS-changed rows: 789
- Best checkpoint: `outputs/legal_action_scorer_028_early_pgs_distill/legal_action_attention_scorer_best.pt`

Diagnostics improved:

- all-phase PGS agreement:
  - `016`: 71.50%
  - `022`: 70.67%
  - `028`: 72.00%
- early PGS agreement:
  - `022`: 60.48%
  - `028`: 63.71%

But game strength did not improve:

- `028` vs `022`: 24-25-1
- `028` vs `016`: 25-25-0
- vs search on seed block 28200:
  - `022`: 18-11-1
  - `028`: 12-18-0
  - `016`: 12-17-1

### Regression Audit

Differential seeds where `022` beat search and `028` lost search:

`28200, 28204, 28206, 28207, 28208, 28212, 28215, 28218, 28220, 28227`

Artifacts:

- `data/corpus_regression_022_vs_search_seed28200_snapshots/`
- `data/corpus_regression_028_vs_search_seed28200_snapshots/`
- `outputs/analysis/policy_diff_022_to_028_on_022_search_wins_diffseeds.json`
- `outputs/analysis/policy_diff_022_to_028_on_028_search_losses_diffseeds.json`
- `outputs/analysis/policy_diff_022_028_search_regression_derived_summary.json`

Main regression signal on those 10 seeds:

- First buy turn: both average 7.5
- First scoring buy: `028` slightly earlier, 32.1 vs `022` 34.5
- But conversion is much worse:
  - `022`: 21.1 buys, 7.4 scoring buys, 16.3 bought points
  - `028`: 18.8 buys, 5.2 scoring buys, 9.6 bought points

### Conversion Diagnostic

New tooling:

- `src/splendor_ai/training/run_conversion_diagnostics.py`
- root wrapper: `run_conversion_diagnostics.py`
- tests: `tests/test_conversion_diagnostics.py`

Artifacts:

- all rows in the snapshot corpora: `outputs/analysis/conversion_diagnostics_022_028_diffseeds.json`
- exact 10 regression seeds: `outputs/analysis/conversion_diagnostics_022_028_diff10.json`

Command for the exact 10-seed report:

```powershell
$env:PYTHONPATH='src'; python run_conversion_diagnostics.py --replay-path data/corpus_regression_022_vs_search_seed28200_snapshots/replays.jsonl --label 022 --replay-path data/corpus_regression_028_vs_search_seed28200_snapshots/replays.jsonl --label 028 --output-path outputs/analysis/conversion_diagnostics_022_028_diff10.json --model-turns-only --game-seed-filter 28200 28204 28206 28207 28208 28212 28215 28218 28220 28227 --early-turn-max 20 --window-turns 20 35 50 --log-every 0
```

Exact 10-seed model-seat readout:

- `022`: win rate 1.0, final score 16.3, score margin +4.7, 21.1 buys, 15.7 visible buys, 5.4 reserved buys, 6.5 scoring buys, 11.5 bought card points, 7.4 reserves, 2.0 non-converting reserves.
- `028`: win rate 0.0, final score 9.6, score margin -6.0, 18.8 buys, 12.4 visible buys, 6.4 reserved buys, 4.6 scoring buys, 6.9 bought card points, 8.1 reserves, 1.7 non-converting reserves.
- Paired delta `028 - 022`: -2.3 total buys, -3.3 visible buys, +1.0 reserved buys, -1.9 scoring buys, -4.6 bought card points, -2.1 noble points, -6.7 final score, -10.7 score margin.
- First buy is still equal at 7.5; first scoring buy is earlier for `028` (32.5 vs 34.9), so the problem is not delayed first conversion.
- Biggest changed early action-type buckets by count are mostly negative:
  - `RESERVE_VISIBLE -> RESERVE_DECK`: count 11, -7.27 bought card points, -3.18 buys, -3.09 scoring buys.
  - `RESERVE_VISIBLE -> RESERVE_VISIBLE`: count 9, -7.0 bought card points, -2.89 buys, -2.67 scoring buys.
  - `BUY_RESERVED -> BUY_RESERVED`: count 9, -4.89 bought card points, -2.11 buys, -1.67 scoring buys.
  - `TAKE_TOKENS -> TAKE_TOKENS`: count 10, -4.1 bought card points, -0.5 buys, -1.5 scoring buys.

Interpretation: `028` is not mainly missing affordable point-card opportunities in the logged states; it has fewer missed affordable point cards. The stronger signal is a reserve/deck-reserve and reserved-buy path that converts into fewer visible buys, fewer scoring buys, fewer card points, and fewer noble points.

### `029` Conservative PGS Filter

New tooling:

- `src/splendor_ai/training/run_pgs_corpus_audit.py`
- root wrapper: `run_pgs_corpus_audit.py`
- tests: `tests/test_pgs_corpus_audit.py`

Audit artifact:

- `outputs/analysis/pgs_corpus_audit_028.json`

Filtered corpus:

- `data/corpus_pgs016_early_token_reserve_029_conservative/replays.jsonl`
- `data/corpus_pgs016_early_token_reserve_029_conservative/summary.json`

Audit result on the 2,053-row `028` focused corpus:

- 789 changed rows, 1,264 unchanged anchors.
- 288 risky early changed rows.
- 284 early reserve-to-reserve changes.
- 160 early selected blind deck reserves.
- 130 early `RESERVE_VISIBLE -> RESERVE_DECK` changes.
- Changed-row search margin median: 34.
- Risky-row search margin median: 21.
- Conservative profile output:
  - 1,893 rows written;
  - 160 rows dropped;
  - 128 rows downweighted to 0.3 of their original weight.

`029` training:

- Parent: `outputs/legal_action_scorer_022_attention_distill/legal_action_attention_scorer_best.pt`
- Candidate: `outputs/legal_action_scorer_029_conservative_early_pgs/legal_action_attention_scorer_best.pt`
- Same broad corpus mix as `028`.
- Replaced the duplicated `028` focused corpus with the duplicated conservative filtered corpus.
- Best epoch: 3.

Action-level result on the original `028` focused corpus:

- `022` stored-teacher match: 52.75%.
- `028` stored-teacher match: 56.65%.
- `029` stored-teacher match: 57.09%.

Promotion gates:

- `029` vs `022`, 50 games seed 28300: 18-32.
- `029` vs `016`, 50 games seed 28400: 21-29.
- `029` vs search, 30 games seed 28200: 15-14-1.
  - Better than `028` at 12-18-0.
  - Worse than `022` at 18-11-1.

Artifacts:

- `outputs/benchmarks/legal_action_scorer_029_vs_022_seed28300_50.json`
- `outputs/benchmarks/legal_action_scorer_029_vs_016_seed28400_50.json`
- `outputs/benchmarks/legal_action_scorer_029_vs_search_seed28200_30.json`
- `data/corpus_regression_029_vs_search_seed28200_snapshots/`
- `outputs/analysis/conversion_diagnostics_022_029_diff10.json`
- `outputs/analysis/policy_diff_022_to_029_on_pgs028_corpus.json`
- `outputs/analysis/policy_diff_028_to_029_on_pgs028_corpus.json`

Exact 10-seed conversion comparison:

- `029`: win rate 0.3, final score 11.4, 19.2 buys, 12.9 visible buys, 6.3 reserved buys, 5.4 scoring buys, 8.1 bought card points.
- `029 - 022`: -1.9 buys, -2.8 visible buys, +0.9 reserved buys, -1.1 scoring buys, -3.4 bought card points, -1.5 noble points, -4.9 final score.
- This is a partial recovery from `028`, but the same reserve-heavy conversion defect remains.

### `030` Constrained Preference Experiment

New reusable tooling:

- `src/splendor_ai/training/run_constrained_preference_corpus.py`
  - selects only changed PGS rows where the stored search advantage over `022` is at least 50;
  - rejects blind deck reserves and token corrections that replace buys;
  - only retains scoring buys, visible scoring reserves replacing blind reserves, or token choices that improve distance to a visible scoring card;
  - extracts exact `022` model-turn anchors from the 10 differential seeds.
- `src/splendor_ai/training/run_policy_drift.py`
  - reports checkpoint top-action agreement by phase, prior action type, and transition on arbitrary snapshot corpora.
- Harness config: `configs/candidate_harness_030_existing.json`
- Harness result: `outputs/harness/candidate_harness_030_existing_summary.json`

Corpus:

- Preferences: `data/corpus_030_constrained/preference_rows.jsonl`
- Anti-regression anchors: `data/corpus_030_constrained/anti_regression_022_rows.jsonl`
- Summary: `data/corpus_030_constrained/summary.json`
- 2,053 PGS rows produced 49 preference corrections:
  - 45 `TAKE_TOKENS -> TAKE_TOKENS`;
  - 4 `RESERVE_DECK -> RESERVE_VISIBLE`.
- Extracted 340 `022` anchors from the 10 differential seeds, weighted 2.0.

Training:

- Parent: `022`.
- Exact sanitized `022` training mix plus the 340 anchors and 49 preference rows.
- Preference rows were pairwise-only: no CE or search-distillation contribution.
- One CPU epoch, learning rate `1e-5`, pairwise loss weight `0.05`.
- Candidate: `outputs/legal_action_scorer_030_constrained_pref/legal_action_attention_scorer_best.pt`

Pre-benchmark diagnostics:

- Broad 2,053-row PGS slice top-action agreement with `022`: 95.76%.
- Protected 340-row regression-anchor agreement: 98.24%.
- On the 49 selected corrections, `030` changed only three actions and adopted two preferred actions.
- All configured diagnostic gates passed.

Benchmarks:

- vs `022`, seed 28500, 100 games: 50-50-0.
- vs `016`, seed 28600, 100 games: 56-43-1.
- vs `022`, fresh confirmation seed 28900, 200 games: 94-105-1.
- Combined direct vs `022`: 144-155-1 over 300 games.
- Paired search seed 28700, 100 games:
  - `030`: 50-50-0, average margin +0.14;
  - `022`: 43-57-0, average margin -0.86.
- Paired search confirmation seed 29100, 200 games:
  - `030`: 83-115-2, average margin -0.645;
  - `022`: 86-112-2, average margin -0.305.
- Across all 300 paired search seeds, the `030 - 022` score-margin delta was only +0.107 with 95% CI approximately `[-0.521, +0.734]`; 63 seeds improved, 172 tied, and 65 worsened.

Conversion confirmation:

- Snapshot corpus: `data/corpus_regression_030_vs_search_seed28200_snapshots/`
- Report: `outputs/analysis/conversion_diagnostics_022_030_diff10.json`
- On the original 10 differential seeds:
  - `030`: win rate 0.6, final score 12.4, 19.1 buys, 13.5 visible buys, 5.3 scoring buys, 7.9 bought card points.
  - `030 - 022`: -0.4 win rate, -3.9 final score, -3.8 score margin, -2.0 buys, -2.2 visible buys, -1.2 scoring buys, -3.6 bought card points.
  - Reserve count was not inflated (`7.2` vs `7.4`), but first scoring buy moved later (`39.1` vs `34.9`).
- Interpretation: filtering removed the obvious reserve pathology, but very small policy drift still compounds into weaker trajectories. This invalidates the assumption that anchor-state top-action retention is enough to make PGS patching safe.

### `031` / `032` Trajectory Policy Improvement

New reusable infrastructure:

- `LegalActionAttentionSamplingBot`
  - top-k epsilon exploration around an attention checkpoint;
  - records the exact behavior probability for every selected action.
- Replay export now preserves behavior probability/log-probability and exploration metadata.
- `run_trajectory_advantage_corpus.py`
  - builds signed outcome/score-margin trajectory returns.
- `run_group_relative_trajectory_corpus.py`
  - centers returns across repeated stochastic rollouts of the same board seed, player role, and opponent.
- `run_trajectory_policy_gradient.py`
  - clipped PPO ratio against recorded behavior probabilities;
  - frozen-parent KL regularization;
  - entropy regularization and gradient clipping.

Verification:

- End-to-end smoke behavior-log-probability reconstruction error: `6.3e-8`.
- Full suite: 179 passed.

`031` plain trajectory PPO:

- Data:
  - 400 stochastic `022` self-play games;
  - 100 stochastic `022` vs `016` games;
  - 60 stochastic `022` vs search games;
  - 30,589 policy actions across 560 completed games.
- Training:
  - one epoch from `022`, learning rate `3e-6`;
  - PPO clip `0.1`, parent KL weight `1.0`;
  - mean KL `3.9e-5`, clip fraction `0.17%`.
- Drift:
  - broad PGS slice agreement 97.27%;
  - protected regression agreement 99.12%;
  - on-policy trajectory agreement 99.08%.
- Initial gates:
  - vs `022`: 47-53;
  - vs `016`: 48-52;
  - paired 100-seed search: `031` 50-50, `022` 49-51.
- Conversion gate on the original 10 seeds:
  - `031`: win rate 0.4, final score 12.0, 18.9 buys, 5.5 scoring buys, 8.7 bought card points.
  - `031 - 022`: -4.3 final score, -5.9 margin, -2.2 buys, -1.0 scoring buys, -2.8 bought card points.
- Result: reject before larger confirmation.

`032` group-relative PPO:

- Data:
  - three stochastic rollouts for every matched condition;
  - 100 self-play seeds, 50 `016` seeds, and 20 search seeds;
  - 270 groups, all size 3; 269 groups had return variation;
  - 25,836 action rows from 807 informative trajectories.
- Training used the same settings as `031`; only target construction changed.
- Drift:
  - broad PGS slice agreement 96.88%;
  - protected regression agreement 99.12%;
  - grouped-trajectory agreement 98.84%.
- Conversion gate:
  - `032`: win rate 0.6, final score 12.9, 19.2 buys, 5.9 scoring buys, 9.3 bought card points.
  - `032 - 022`: -3.4 final score, -3.4 margin, -1.9 buys, -0.6 scoring buys, -2.2 bought card points.
- Group-relative targets improved over `031`, but still failed the conversion thresholds.
- Harness: `configs/candidate_harness_032_existing.json`
- Result: diagnostic failure; broad benchmarks correctly skipped.

### `033` / `034` Complete-Rollout Action Credit

New tooling:

- `run_rollout_action_credit.py`
  - phase-balanced snapshot sampling;
  - parent-policy top actions plus strongest available buy candidates;
  - complete stochastic game rollouts after forcing each root action;
  - common rollout seeds across root actions;
  - paired action-advantage uncertainty;
  - adaptive refinement: initially changed decisions receive 16 additional rollouts, for 24 total per action.
- `run_rollout_credit_training.py`
  - soft rollout-return action distributions;
  - frozen-parent KL on target states;
  - optional separate broad anchor stream with parent KL.

Rollout corpus:

- Search-derived states: `data/corpus_rollout_credit_033_search/`
- Self-play states: `data/corpus_rollout_credit_033_selfplay/`
- 90 states evaluated; 68 passed the minimum action-return range.
- 62 initially changed states received adaptive refinement.
- Final rollout best differed from `022` on 46 written states.
- 37 changed states had positive paired 95% lower bounds.
- All confident `BUY_* -> TAKE_TOKENS` changes involved zero-point tier-1 buys, not scoring buys.

`033` target-only KL:

- Improved rollout-distribution loss but did not improve rollout-best top-action match.
- Broad top-action agreement fell to 91.09%; protected agreement fell to 95.29%.
- Rejected before game evaluation.

`034` broad-anchor KL:

- Parent: `022`.
- Training: three epochs, learning rate `1e-5`, target-state KL weight 2.0, broad-anchor KL weight 5.0.
- Rollout-best match improved from 32.35% to 33.82%.
- Broad PGS-slice agreement: 96.83%.
- Protected regression agreement: 98.53%.
- Passed target-uptake and drift gates.
- Conversion on the original 10 seeds:
  - `034`: win rate 0.4, final score 12.3, 19.3 buys, 5.6 scoring buys, 8.1 bought card points.
  - `034 - 022`: -4.0 final score, -5.6 margin, -1.8 buys, -0.9 scoring buys, -3.4 bought card points.
- Harness: `configs/candidate_harness_034_existing.json`
- Result: conversion diagnostic failure; broad benchmarks skipped.

Interpretation:

- Complete rollout outcomes are better grounded than shallow-search heuristic labels.
- They are still conditional on the current stochastic continuation policy and sparse root candidate set.
- Small rollout-target changes continue to alter downstream trajectories in ways that the local return estimates do not generalize.
- Do not tune `034`; move to tree-search visit distributions with repeated state evaluation and policy/value backup.

### Generation-1 Root-Visit Pilot

Implemented the first legal search-distillation pilot without changing the
active baseline:

- PUCT root telemetry now exports prior entropy, normalized entropy, top-1
  probability, effective branching, visit entropy, visited-action count, and a
  normalized root-visit distribution for every replayed search action.
- Replay/dataset/training now preserve those distributions and apply a direct
  masked cross-entropy root-visit objective. `run_legal_action_supervised.py`
  exposes `--root-visit-loss-weight`.
- `run_benchmark.py` now supports `--paired-seeds` for direct
  checkpoint-vs-checkpoint gates: each seed is played twice with seats reversed.

Data:

- Broad legal policy data: 2,000 completed stochastic games, 127,336 steps:
  1,000 policy035 self-play, 500 policy035 vs historical `016`, and 500
  policy035 vs historical `022`.
- Search data: 300 completed clipped-PUCT035 self-play games, 19,030 steps,
  all with normalized root-visit targets, zero stalls, and zero timeouts.
- The clipped corpus used the frozen candidate configuration: s64, legal `035`
  priors, depth-2/width-1 late n-step leaves, candidate limit 4,
  `min_turn_index=40`, and `delta_clip=0.20`.

Policy `036`:

- `outputs/legal_action_scorer_036_generation1_root_visit_pilot/`
- Initialized from legal `035`; two epochs, learning rate `1e-5`, game-level
  validation split, ordinary policy/value loss plus root-visit loss weight 2.0.
- 146,366 total snapshot rows; best validation epoch 2, loss 0.8826 and legal
  action accuracy 90.99%.
- It is not promoted. Fresh paired 100-game gates, all legal:
  - vs policy035: 48-51-1; policy036 score share 0.485, CI95 [0.40, 0.57].
  - vs legal PGS035: 45-54-1; share 0.455, CI95 [0.375, 0.54].
  - vs clipped PUCT035: 41-59; share 0.41, CI95 [0.33, 0.49], mean paired
    margin -1.49, CI95 [-2.63, -0.38].
  - vs plain PUCT035: 45-55; share 0.45, CI95 [0.37, 0.53].
- Clipped PUCT telemetry in that held-out gate: mean root prior effective
  actions 4.41, visit effective actions 3.55, and visited actions 5.34. The
  exported targets therefore contain a real but narrow search preference, not
  a degenerate one-action label.

Value `010`:

- `outputs/value_model_010_generation1_mixed_wdl/`
- Trained on the same mixed corpus with margin regression and WDL loss weight
  0.5. Best validation epoch was 1: MSE 0.2103, outcome-sign accuracy 59.24%,
  Pearson 0.2661.
- Keep it diagnostic only. It has not earned a learned-leaf A/B or live search
  integration.

Interpretation:

- The corpus/export/training/gating infrastructure works end to end, but one
  root-visit target per visited PUCT state plus broad behavior cloning did not
  turn the policy into a stronger standalone agent.
- Do not promote `036`, do not replace `035`, and do not use `010` in live
  PUCT/PGS. This is a bounded negative result, not evidence to discard legal
  PUCT or the root-visit data path.

### `036` Root-Visit Target-Uptake Audit

Implemented `run_root_visit_target_audit.py` and
`src/splendor_ai/training/run_root_visit_target_audit.py`. The audit recreates
the exact mixed-corpus, game-level split used by `036`, isolates the PUCT source
on either side of that split, and reports game-bootstrap CIs for target entropy,
cross-entropy/KL, top-action agreement, model drift, phase, and target action
family. Artifacts:

- `outputs/diagnostics/root_visit_target_uptake_036_vs035_train.json`
- `outputs/diagnostics/root_visit_target_uptake_036_vs035_heldout.json`

Results:

- The held-out PUCT slice is 28 games / 1,816 rows. Targets are not degenerate:
  3.52 effective actions, 5.32 visited actions, 21.07 legal actions, and target
  top probability 0.639.
- They are nevertheless weak as policy-improvement labels. Legal `035` already
  matches the PUCT visit argmax on 93.85% of held-out rows and has target KL
  0.0547. The tree mostly refines probabilities among actions `035` already
  ranks first, rather than supplying new decisions.
- `036` did not learn the complete visit distribution. On the training PUCT
  rows, it improves CE by only 0.0017 nats. On held-out rows it regresses by
  0.0022 nats (CI95 [-0.0009, 0.0052]) and target KL rises from 0.0547 to
  0.0569.
- This is not simple inactivity or ordinary overfit. `036` substantially moves
  the distribution (candidate-to-035 policy KL 0.0561) and raises probability
  on the target top action from 0.601 to 0.686, but it lowers policy entropy
  from 1.180 to 0.850 and loses the target visit tail.
- The distortion is stable in train and validation: it improves buy-visible and
  buy-reserved visit fit, but worsens early-phase fit and, especially,
  token-taking (+0.0392 held-out CE) and reserve-deck (+0.0121) targets.
- The direct source of that collapse is measured: 98.73% of held-out PUCT replay
  behavior actions equal the visit argmax. The ordinary one-hot policy loss on
  PUCT rows therefore conflicts with their soft visit target; the PUCT source
  is only about 13% of the mixed corpus, so broad one-hot behavior cloning also
  dominates the intended calibration signal.

Conclusion:

- `036` is neither a clean full-distribution uptake nor evidence that more of
  the same PUCT games are needed. The current objective is structurally biased
  toward an overconfident argmax approximation, and its target has too little
  top-action novelty over `035` to justify a broad policy replacement.

### `037` Conflict-Free Root-Visit Correction

Implemented `run_root_visit_correction.py` and
`src/splendor_ai/training/run_root_visit_correction.py`. This runner recreates
the exact `036` split, trains only the 272-game PUCT training partition against
soft root visits, and takes a separate batch from the 127,336-row broad legal
stream for frozen-`035` parent KL. It deliberately applies no one-hot replay CE
or value loss to the PUCT target rows.

- Candidate: `outputs/legal_action_scorer_037_root_visit_conflict_free/`
- Fixed recipe: parent `035`, two epochs, learning rate `1e-5`, root target
  batches 128, anchor batches 128, anchor KL weight 2.0, game split seed 36.
- Best validation epoch: 1. The internal held-out root CE was 1.01862 and
  entropy 1.13033. The second epoch did not improve validation CE.
- Independent held-out uptake audit:
  `outputs/diagnostics/root_visit_target_uptake_037_vs035_heldout.json`.
  `037` improves visit CE from 1.0252 (`035`) to 1.0156, delta -0.0096 with
  CI95 [-0.0107, -0.0084], and lowers target KL from 0.0547 to 0.0451.
- The correction is broad rather than a buy-only artifact: every phase and each
  target action family improves CE, including early positions (-0.0128),
  reserve deck (-0.0178), and token taking (-0.0047). Its policy entropy is
  1.1265, close to `035`'s 1.1804 and far above `036`'s collapsed 0.8498.
- It fails the decision-novelty gate. `037` changes `035`'s top action on only
  0.86% of held-out rows (CI95 [0.48%, 1.28%]); target-top agreement is 93.74%
  versus `035`'s 93.85%. The improved distribution fit therefore does not
  create a plausible standalone policy-strength change.
- Do not run any game benchmark or promote `037`. The requested pre-game gate
  did its job: correct soft-target uptake is possible, but the present PUCT
  visit targets have too little decision novelty to be a useful policy-imitation
  engine.

### Counterfactual Leaf-Quality Audit

Implemented `run_leaf_quality_audit.py` and
`src/splendor_ai/training/run_leaf_quality_audit.py`. It samples late-game PUCT
replay states, applies the same observer-relative determinization and
continuation seeds to every legal candidate action, and compares root-player
heuristic, exact clipped n-step, value009 scalar/outcome, and value010
scalar/outcome signals against matched policy035 continuation returns.

- Primary confirmation artifact:
  `outputs/diagnostics/leaf_quality_audit_puct035_late_counterfactual_confirm60.json`.
- 60 phase-stratified states were evaluated from the existing 300-game clipped
  PUCT corpus; 46 had a rollout action-return range of at least 0.05 and were
  retained for state-level ranking metrics. Each candidate used three legal
  determinizations and ten common-seed continuations per determinization.
- Clipped n-step is the only clear leaf candidate: mean rollout regret 0.266
  versus heuristic 0.370, delta -0.105 with CI95 [-0.216, -0.016]. Its mean
  Spearman action-ranking correlation is 0.168, CI95 [0.014, 0.318], versus the
  heuristic's -0.112. Top-one accuracy is directionally higher (30.4% versus
  23.9%) but not independently decisive.
- No learned value signal earns another training or live-leaf experiment.
  Value010 scalar has a small regret improvement but no robust rank-correlation
  or top-one improvement; value009 is weaker, particularly after turn 60.
  Do not train another value model from these data.

Fresh evaluation-only strength gate for the audited clipped rule:

- New runner support: `run_puct_benchmark.py --opponent-plain-puct` disables
  n-step/rollout leaves only for the PUCT opponent, allowing a fair ablation.
- s64 clipped PUCT035 vs policy035: 57-42-1, paired share 0.575, CI95
  [0.51, 0.645]; mean margin +0.83, CI95 [-0.02, +1.64].
- s64 clipped PUCT035 vs legal PGS035: 56-43-1, share 0.565, CI95
  [0.48, 0.65]; margin +1.08, CI95 [-0.28, +2.38].
- s64 clipped PUCT035 vs independently configured plain PUCT035: 56-43-1,
  share 0.565, CI95 [0.485, 0.645]; margin +0.65, CI95 [-0.39, +1.72].
- All three 100-game paired blocks completed with zero timeouts.

Conclusion:

- The audit rehabilitates clip `0.20` as the best current leaf/backup rule and
  explains why it can beat policy035. It does not establish a self-play engine:
  the fresh legal-PGS and plain-PUCT intervals still include a tie. Keep it
  evaluation-only and do not generate self-play targets or train another policy
  from it.

### Final Clipped-PUCT Promotion Gate

Completed the preregistered fresh 200-game paired confirmations without any
configuration changes. Both use legal `035` priors, s64, `c_puct=1.5`,
batch-8 inference, no reuse, and the frozen late depth-2/width-1/candidate-4
clip-0.20 rule. Artifacts:

- `outputs/benchmarks/puct035_clip020_finalconfirm_vs_legal_pgs035_paired200.json`
- `outputs/benchmarks/puct035_clip020_finalconfirm_vs_plainpuct035_paired200.json`

Results:

- vs legal PGS035: 100-98-2, paired score share 0.505, CI95 [0.435, 0.575],
  mean margin +0.755, CI95 [-0.11, +1.65].
- vs independently configured plain PUCT035: 104-96, paired share 0.52, CI95
  [0.46, 0.58], mean margin +0.51, CI95 [-0.19, +1.235].
- All 400 games completed normally with zero timeouts.

Decision:

- Do not promote clipped n-step PUCT to a self-play engine. It fails the
  declared rule requiring both legal-PGS and plain-PUCT lower score-share bounds
  to exceed 0.50. The counterfactual leaf advantage is real but does not survive
  full-game confirmation with enough certainty.

### Counterfactual Action-Value Pilot

Implemented a separate action-conditioned Q pipeline rather than changing the
policy trainer:

- `run_leaf_quality_audit.py` can now export its informative matched-return
  rows as a legal-action replay corpus.
- `run_action_value_training.py` and
  `src/splendor_ai/training/run_action_value_training.py` train a fresh score
  head on counterfactual action returns with Smooth L1 and pairwise ranking
  losses. The shared legal `035` encoders are initialized from the frozen prior;
  no one-hot policy CE, root-visit CE, or terminal WDL target is used.
- Held-out evaluation compares Q against the heuristic on top-one selection,
  regret, per-state Spearman ranking correlation, and disagreement with the
  `035` prior. The offline acceptance rule requires both a negative upper
  confidence bound for Q-minus-heuristic regret and a positive lower confidence
  bound for Q-minus-heuristic Spearman before any PUCT integration is allowed.

Corpus and audit:

- `outputs/diagnostics/action_value_corpus_001_audit.json` evaluated 300
  sampled late PUCT states from the existing clipped-PUCT corpus. Each state
  used six legal candidate actions selected from the frozen `035` policy, three
  legal determinizations, and eight matched policy035 continuations per
  action/determinization.
- 206 states had a counterfactual return range of at least 0.05 and were
  exported to `data/corpus_action_value_late_puct035_001/replays.jsonl`.
  They cover 127 games and turns 40--73.
- This larger audit reproduces the useful leaf diagnostic: clipped n-step
  regret is 0.257 versus the heuristic's 0.312, delta -0.055 with CI95
  [-0.110, -0.004]. This validates the return target quality but does not
  promote clipped PUCT, whose final full-game gate remains closed.

Q-model result:

- Candidate: `outputs/action_value_scorer_001_counterfactual_late/`.
- Game-level held-out split: 159 train rows and 47 validation rows from 25
  games. Best epoch 29 of 30.
- Q top-one rate is directionally higher (38.3% versus 36.2%) but inconclusive,
  delta +2.1 points, CI95 [-17.0, +21.3]. Its mean regret is worse: 0.341
  versus 0.297, delta +0.044, CI95 [-0.073, +0.163]. Spearman is likewise
  inconclusive: +0.037, CI95 [-0.182, +0.257].
- Q changes the `035` top action on 53.2% of held-out states, but this is not
  useful demonstrated novelty because the regret gate fails.
- Decision: do not integrate this Q checkpoint into PUCT and do not run PUCT
  strength games. The pilot is a clean negative/inconclusive result, not
  evidence for another action-Q hyperparameter sweep.

### Residual PUCT Error Attribution Audit

Implemented `run_residual_error_audit.py` and
`src/splendor_ai/training/run_residual_error_audit.py`. This is a diagnostic
only: it does not modify a checkpoint or run a promotion game gate. For every
sampled root state, it evaluates **every legal action** under common
determinizations and continuation seeds, then compares the matched-return
oracle with the `035` prior top action and the recorded clipped-PUCT root-visit
choice. It reports policy top-six coverage separately from the PUCT attribution,
because root PUCT itself expands all legal actions.

- Artifact: `outputs/diagnostics/residual_error_audit_puct035_clip020_late80.json`.
- The fixed, phase-stratified sample contains 80 late states: 27 turns 40--49,
  27 turns 50--59, and 26 turn 60+. Each legal action used three legal
  determinizations and six common-seed policy035 continuations per
  determinization.
- Only 13/80 states have a statistically resolved best root action under the
  conservative paired criterion (best-vs-runner-up mean margin at least 0.05
  and greater than 1.96 paired standard errors). The other 67 are explicitly
  labelled unresolved; do not use them to claim a search failure.
- Of the 13 resolved states, PUCT matches the pooled matched-return oracle once
  and chooses the wrong `035` top-prior action 11 times. The remaining error is
  one backup/search miss where PUCT diverges from both the prior and the
  oracle. Thus PUCT equals the `035` prior on 12/13 resolved states, while the
  prior equals the oracle on only 1/13. This is strong diagnostic evidence of
  root-prior dominance, but still a small resolved sample rather than a
  promotion-quality estimate.
- The policy top-six candidate set omits the oracle on 8/13 resolved states
  (61.5%, CI95 [30.8%, 84.6%]). This independently explains why an action-Q
  rule restricted to that candidate set is not a credible next fix.
- The per-determinization best action changes on 11/13 resolved states (84.6%,
  CI95 [61.5%, 100%]). This is expected information-set variation, not an error
  label: PUCT should optimize the mean over determinizations, and should not be
  simplified to one determinization per move.

Decision:

- Do not change the leaf, train another Q/value model, or run a PUCT strength
  gate from this result. The audit rules those directions out more clearly.
- The next testable hypothesis is narrowly about root control: can an
  evaluation-only PUCT with a less dominant root prior recover more of the
  already resolved matched-return actions without degrading the rest of the
  pool?

### Root-Prior Control Ablation

Implemented a root-only prior control in `PuctConfig`:
`root_prior_mix=1.0` preserves the frozen neural prior, `0.0` makes only the
root prior uniform, and intermediate values linearly mix the two. Child-node
priors are unchanged. Added `run_root_prior_control_ablation.py` and
`src/splendor_ai/training/run_root_prior_control_ablation.py`, which re-runs
legal PUCT on the 80 residual-audit snapshots and scores selected actions using
the already computed matched-return table. It generates no new continuations
and runs no games.

- Artifact:
  `outputs/diagnostics/root_prior_control_ablation_puct035_clip020_late80.json`.
- Fixed variants: frozen root prior (`1.0`), 50/50 root mix (`0.5`), and uniform
  root prior (`0.0`). All retain s64, `c_puct=1.5`, FPU 0, batch-8 inference,
  and the frozen late clip-0.20 leaf rule.
- The freshly rerun frozen control matches the source corpus root choice on
  92.5% of all 80 states, validating the replay setup.
- The 50/50 mix changes only 8.8% of choices and fails both offline gates:
  resolved regret delta -0.038, CI95 [-0.107, 0.000], and oracle-agreement
  delta +7.7 points, CI95 [0.0, +23.1].
- Uniform root control changes 56.3% of choices, increases root visit effective
  actions from 2.64 to 19.07, and improves resolved mean regret from 0.926 to
  0.568. Its paired regret delta is -0.358, CI95 [-0.681, -0.073]. Oracle
  agreement rises from 1/13 (7.7%) to 4/13 (30.8%), delta +23.1 points, but its
  CI95 is [0.0, +46.2] and therefore does not establish a strictly positive
  improvement.

Decision:

- Do not run continuation or paired game evaluations yet. The predeclared gate
  required both a negative regret upper bound and a positive oracle-agreement
  lower bound; uniform clears the first only.
- Uniform root control is the first result since clip-0.20 with a robustly
  favorable offline regret signal. The next justified work is a preregistered,
  larger all-legal matched-continuation confirmation of **only** frozen versus
  uniform root control. It should target enough resolved states to decide the
  agreement gate; do not add another mix value or change child priors.

### Disjoint Uniform Root-Control Confirmation

Completed the preregistered confirmation using
`outputs/diagnostics/residual_error_audit_puct035_clip020_late320_confirm.json`
and
`outputs/diagnostics/root_prior_control_ablation_puct035_clip020_late320_confirm.json`.

- The new all-legal audit sampled 320 phase-stratified states (107 turns
  40--49, 107 turns 50--59, 106 turn 60+), with the original 80 states
  explicitly excluded. The artifacts record all 80 exclusions and zero state
  overlap.
- It retains exactly the fixed protocol: three legal determinizations, six
  common-seed policy035 continuations per legal action/determination, the same
  paired resolution rule, and s64 clipped-PUCT root selections.
- 32/320 states have a resolved matched-return root action. This is fewer than
  the rough planning estimate but more than twice the exploratory resolved
  sample; the predeclared gate is evaluated on the actual resolved pool.
- The frozen rerun reproduces the source root choice on 97.8% of all states.
  On resolved states it has mean regret 0.598 and oracle agreement 31.25%.
- Uniform root control changes 60.9% of all choices but is worse: mean regret
  0.660, paired delta +0.063, CI95 [-0.120, +0.241], and oracle agreement
  28.13%, delta -3.13 points, CI95 [-18.75, +12.5]. It fails both the regret
  and agreement gates.

Decision:

- Retire uniform root control and the root-prior-mixing line. The 80-state
  apparent gain was an underpowered exploratory result and does not replicate
  on the disjoint confirmation.
- Do not run PUCT continuation or paired game tests, add another mix setting,
  change child priors, or increase simulations from this result.

### Strategic Review Guide and CPU High-Budget Gate

The following principles from the external strategic review are retained as
future guidance, with the listed constraints:

1. Training-time search may use a larger budget than the final playing agent;
   the one-second online target does not itself rule out an offline search
   operator.
2. A larger-budget operator must first demonstrate action-quality improvement
   and then paired-game strength before it generates self-play targets. Search
   budget alone is not evidence that its visits contain useful policy novelty.
3. Keep value work dormant until a stronger, demonstrably useful search corpus
   exists. Playout-cap randomization, forced-playout target pruning, and opening
   exploration are downstream corpus mechanisms, not substitutes for an engine
   strength gate.
4. Preserve paired promotion gates and the distinction between diagnostics and
   vetoes. CPU remains the default while experiments are realistically feasible;
   revisit CUDA only when corpus scale or a redesigned batched search path makes
   inference a material runtime bottleneck.

Implemented `run_puct_budget_ablation.py` and
`src/splendor_ai/training/run_puct_budget_ablation.py` to test that high-budget
hypothesis on CPU without new continuations. It reuses the 320-state all-legal
matched-return audit and compares only frozen-prior s64 and s512 PUCT, with the
same clip-0.20 leaf rule, `c_puct=1.5`, batch-8 inference, and root prior mix
1.0.

- Artifact:
  `outputs/diagnostics/puct035_clip020_cpu_s512_action_quality_late320.json`.
- CPU performance is practical for this gate: s64 averages 0.533 seconds/root
  and 268 simulations/second; s512 averages 4.158 seconds/root and 320
  simulations/second. The complete 320-state comparison took about 25 minutes.
- On the 32 resolved states, s64 has mean regret 0.598 and oracle agreement
  31.25%. s512 is slightly worse: regret 0.609, delta +0.012 with CI95
  [-0.101, +0.120], and oracle agreement 28.13%, delta -3.13 points with CI95
  [-9.38, 0.0]. It changes only 14.7% of root choices.

Decision:

- Do not run s512 paired games or generate a high-budget self-play corpus.
  s512 fails both preregistered action-quality gates on the existing
  all-legal labels.
- The project has now tested the budget-change hypothesis directly. Do not
  reinterpret the rejected s64 line as merely CPU-starved, and do not implement
  KataGo-style target machinery on this engine until a new, independently
  justified search operator exists.

### Mid-Game Headroom Audit

Implemented `run_midgame_headroom_audit.py` and
`src/splendor_ai/training/run_midgame_headroom_audit.py`. It is a resumable,
two-stage all-legal matched-return audit: a cheap screen of 20 states in each
of turns 15--24, 25--32, and 33--40, followed by the full three-determinization,
six-continuation refinement of the ten highest provisional-range states per
band. A checkpoint JSON is written after every completed state so interrupted
CPU runs resume without changing the fixed sample or seeds.

- Artifact: `outputs/diagnostics/midgame_headroom_puct035_two_stage_001.json`.
- Screening covers 60 phase-stratified states; refinement covers 30 states,
  ten in each band. Raw action return ranges are large (band means 1.29--1.51),
  but this is not the relevant confidence measure.
- After full refinement, only 3/30 states have a best-versus-runner-up margin
  above both 0.05 and 1.96 paired standard errors. The resolved states occur
  once in turns 25--32 and twice in turns 33--40; none occur in turns 15--24.
  The policy and recorded s64 PUCT choice miss the oracle on each of those
  three states, but three resolved examples do not establish phase-level
  headroom or support an action-quality promotion gate.

Decision:

- Do not implement or run the policy-guided terminal-rollout leaf pilot yet.
  The proposed sequence required meaningful mid-game resolved headroom first;
  this audit finds high raw variance, not enough reliable action separation.
- The strategic-review claim that mid-game likely supplies substantially more
  usable headroom than late game is not supported by this first controlled
  screen. Outcome-grounded leaves remain a conceptually distinct future
  evaluator class, but are not justified for a live PUCT pilot from these data.

## Best Next Step

Retain `022` as the frozen historical baseline and `035` as the active legal
prior. Keep clip `0.20` frozen as the best tested PUCT leaf diagnostic, but do
not call it a promotion-ready self-play engine.

Recommended next direction:

1. Stop root-only PUCT visit imitation, terminal WDL value training, and this
   first direct action-Q recipe. `037`, the value models, and action-value 001
   do not justify further same-family sweeps.
2. Stop optimizing clipped n-step leaves. The final 200-game gate has answered
   the promotion question; do not add simulations, sweep clip settings, or
   generate self-play from this engine.
3. Close the current PUCT-improvement line. Root visits, learned value leaves,
   candidate-limited action Q, clipped-leaf promotion, root-prior control, and
   frozen-prior s512 have each received an information-legal offline or paired
   confirmation and do not justify further local sweeps. The initial mid-game
   screen also does not justify an outcome-rollout leaf pilot. Retain `035` as
   the active legal policy and `022` as the historical baseline. Any future
   improvement proposal must start from a new independently justified
   algorithmic direction, not another PUCT parameter, budget, or leaf-target
   variation.
4. Keep all future direct gates paired-seed and information-legal. The
   10-game conversion diagnostic remains diagnostic only, never a standalone
   promotion veto.

The human calibration and clean actor experiment below supersede the earlier
claim that no independently justified non-PUCT direction was available. The
game-level actor family remains conceptually open, but candidate `040` is a
measured near no-op and must not receive strength games.

## Independent ES Gate (Retired)

The PUCT line remains closed. A separate, explicitly gated game-outcome ES
experiment was implemented and evaluated; it does not justify an iterative ES
run.

- New `src/splendor_ai/eval/parallel_paired.py` and
  `run_parallel_paired_benchmark.py` run legal attention checkpoints in
  parallel over seed/seat pairs. Every completed pair is atomically persisted
  and the runner reports paired score share, final margin, and the ES utility
  `0.75 * outcome + 0.25 * tanh(margin / 5)`. It retries atomic progress-file
  replacement briefly on Windows file-lock races. It passed a real `035` versus
  itself one-pair smoke test at
  `outputs/benchmarks/parallel_paired_smoke_035_self.json`.
- New `src/splendor_ai/training/es_adapter.py` constrains ES to a deterministic
  rank-4, fixed-A adapter over four matrices in final attention layer 1. This
  is 3,584 trainable B parameters, not the 98k score head.
- New `src/splendor_ai/training/run_es_stage0.py` and `run_es_stage0.py`
  implement the corrected test: behavior-only sigma calibration on stored
  states; 16 antithetic directions with paired-game fitness; one aggregate
  normalized update; then a fresh paired-seed gate. It does not select the
  best individual direction. A calibration veto now prevents it from spending
  fitness games if its perturbations do not change at least 1% of sampled legal
  actions. Stage 0 otherwise passes only if its fresh bounded-utility lower
  confidence bound exceeds zero with no stall/timeout.
- The clean clone trained from random attention weights on 63,622 legal `035`
  self-play states for three policy-CE-only epochs. Its best held-out action
  accuracy was 86.1%. It tied `035` over a fresh 100-pair gate (46.75% share,
  CI95 [40.25%, 53.25%]) and therefore cleared only the viability prerequisite.
- The first ES Stage 0 used sub-behavioral sigmas `0.25` and `0.75`; both
  changed 0% of sampled actions, so its game result is invalid and ignored.
  A behavior-only scan found mean action-change rates of 1.2%, 5.1%, and 19.1%
  at sigmas 3, 10, and 30. The corrected Stage 0 selected sigma 30 and passed
  against its clean-clone parent: fresh 100-pair utility +0.141, CI95
  [+0.0058, +0.272], with 57.25% score share and no failures.
- This local pass did **not** transfer. The preregistered fresh 200-pair bridge
  against frozen legal `035` is at
  `outputs/benchmarks/es_stage0_cleanclone038_002_aggregate_vs_policy035_paired400.json`:
  score share 49.375%, CI95 [44.375%, 54.375%]; bounded utility -0.0091,
  CI95 [-0.1025, +0.0854]; zero stalls/timeouts. Do not run ES iterations,
  create an ES corpus, sweep sigma/rank, or run a final PGS gate. This adapter
  direction is retired because its only demonstrated gain was parent-relative.

## Human Calibration And Clean LOO Actor

The GUI replay is adequate for a first human calibration. It stores every
pre-action legal set, public observation, full state snapshot, chosen action,
seat, checkpoint metadata, outcome, and final scores. The ten completed games
in `outputs/human_games/replays.jsonl` are balanced across seats. The human won
8/10, 4/5 from each seat, with mean final-score margin +5.6. This is a small,
non-random human sample and not a statistical bot benchmark, but it is strong
evidence that legal `035` is not yet near the practical human-strength goal.
These rows remain held out from all training and promotion gates.

A fresh legal actor-only group-relative experiment was implemented from the
clean-from-random clone `038`, with no critic and no human data:

- `run_group_relative_trajectory_corpus.py` now supports true leave-one-out
  group baselines. Each trajectory is compared with the other three runs of
  the same environment seed and role, not with a baseline containing its own
  outcome.
- New `run_group_relative_signal_audit.py` verifies complete group structure,
  informative return variation, stochastic decision uptake, exact behavior
  probability reconstruction from the frozen parent, and clean termination
  before training.
- The first epsilon-0.20 pilot was rejected before training because only 1.91%
  of written rows were non-greedy versus the preregistered 2.00% floor. The
  threshold was not relaxed. One fixed correction at epsilon 0.30 used fresh
  seeds 81100--81199 and four independent sampling seeds.
- Accepted corpus:
  `data/corpus_trajectory_040_clean_rloo_vs035_e030_grouped/trajectory_advantages.jsonl`.
  It contains 100 complete four-trajectory groups, 86 informative groups, and
  11,035 actor rows. Non-greedy uptake is 2.43%; behavior reconstruction MAE is
  `9.1e-8`. The full audit passes at
  `outputs/analysis/group_relative_signal_audit_040_clean_vs035_e030.json`.
- Exactly one actor epoch was trained from clean clone `038`, with learning
  rate `3e-6`, PPO clip 0.10, KL weight 1.0, and no critic. Training behavior
  reconstruction MAE is `1.1e-7`, mean ratio 1.000006, clip fraction 0.0033,
  and mean exploration KL `5.1e-5`.
- New `run_actor_movement_audit.py` measures greedy action changes and both raw
  and behavior-policy KL on independent 5,000-row samples from the accepted
  on-policy stream and broad legal `035` self-play. Its bounds were fixed
  before inspection: on-policy movement >=1%, broad movement >=0.5%, both
  <=10%, and mean exploration KL <=0.01.
- Candidate `040` fails the movement gate at
  `outputs/analysis/actor_movement_audit_040_clean_rloo.json`: 20/5,000
  on-policy decisions change (0.40%) and 23/5,000 broad decisions change
  (0.46%). Mean exploration KL is only about `5.9e-5` on both streams. This is
  under-movement, not damaging drift.

Decision:

- Reject candidate `040` before games. Do not run the conditional paired gates
  versus `035` or legal PGS and do not reinterpret unplayed gates as ties.
- Do not regenerate trajectories or alter the LOO target; that signal audit
  passed. If this family gets one correction, change only update magnitude from
  clean clone `038` using the same accepted corpus, with a preregistered single
  setting and the same movement/KL gate. Do not sweep learning rate/KL or relax
  the movement floor after observing `040`.
- Only a correction that clears the movement audit earns fresh paired games
  against `035`, followed conditionally by legal PGS. Conversion diagnostics
  remain descriptive and human replays remain held out.

The one allowed magnitude correction has now been run. Candidate `041` reused
the exact accepted corpus, clean `038` initialization, seed, one-epoch schedule,
PPO clip, KL regularization, entropy term, and audit bounds. The only change
from `040` was learning rate `3e-6` -> `1e-5`.

- Training artifact:
  `outputs/legal_action_scorer_041_clean_rloo_actor_lr1e5/legal_action_attention_scorer.pt`.
- Movement artifact:
  `outputs/analysis/actor_movement_audit_041_clean_rloo_lr1e5.json`.
  It passes all fixed gates: 84/5,000 changed decisions (1.68%) on both the
  on-policy and broad streams, with broad mean exploration KL `2.17e-4`.
- Fresh direct gate:
  `outputs/benchmarks/actor041_clean_rloo_lr1e5_vs_policy035_paired400.json`.
  Over 200 paired seeds / 400 games, `041` has score share 49.125%, CI95
  [44.5%, 53.75%], mean outcome -0.0175, bounded utility -0.0071 with CI95
  [-0.0960, +0.0825], and zero stalls/timeouts.

Decision:

- `041` solved the under-movement defect in `040`, but failed the preregistered
  direct superiority gate against legal `035`. Treat it as tied, not promoted.
- Do not run the conditional legal-PGS gate. It was downstream of a positive
  `035` lower confidence bound and that condition was not met.
- Close this single-update clean LOO recipe. Do not sweep learning rate, KL,
  epochs, exploration, or movement thresholds on the same corpus. Any future
  game-outcome actor proposal needs a materially different credit/update design
  and a new justification, not another magnitude correction.

## Held-Out Human-Game Forensic Audit

Implemented `run_human_game_forensic_audit.py` and
`src/splendor_ai/training/run_human_game_forensic_audit.py`. The audit keeps all
human rows held out from training and performs two distinct stages:

- Every `035` turn is reconstructed and classified for robust forced wins,
  immediate opponent-winning replies and available blocks, same-target
  immediate dominance, available scoring conversions, reserve outcomes, and
  repeated non-buy streaks. Tactical claims must agree across eight
  observer-relative hidden-card determinizations.
- Only suspicious states receive expensive outcome-grounded evaluation. The
  recorded action is compared with the best available action from every legal
  action type plus tactical, scoring, dominance, and policy-top alternatives.
  Each action receives 3 determinizations x 8 common-seed policy-guided terminal
  continuations. A mistake resolves only when paired return delta is at least
  0.10, exceeds 1.96 standard errors, and is positive in every determinization.

Final artifacts:

- Machine-readable audit:
  `outputs/diagnostics/human_game_forensic_policy035_full_v2.json`.
- Concise review:
  `outputs/diagnostics/human_game_forensic_policy035_full_v2.md`.
- Crash-resumable state-level progress:
  `outputs/diagnostics/human_game_forensic_policy035_full_v2.progress.json`.

Results:

- All 271 `035` turns across the ten games reconstruct exactly; recorded and
  reloaded policy-top actions agree 100%.
- There are zero robust missed forced wins, zero failures to block an immediate
  opponent win, zero same-target dominance cases, and zero turns where `035`
  declined an already-affordable scoring purchase. The replay does not support
  an endgame tactical-failure diagnosis or the proposed tactical override.
- Screening flags 29/271 turns: 26 reserve suspicions and three four-turn
  non-buy streaks. All 29 receive matched terminal continuations with complete
  action-class coverage and 24 samples per action.
- Eight states resolve as high-confidence mistakes. All eight occur in games
  the human won; seven are early and one mid-game. Every chosen action is a
  reserve. Better action transitions are four different visible reserves,
  three token takes, and one reserved-card purchase.
- Seven resolved chosen reserves were never converted; one reserved an already
  affordable zero-point card. Mean paired continuation improvement is +0.621.
  The policy nevertheless favors the worse chosen actions by a mean +2.709
  score units over the better alternatives. This is a large ranking error, not
  a near-tie at the policy head.

Decision:

- Do not build an endgame tactical wrapper from these games. It would address
  none of the eight resolved errors.
- Classify the observed gap as early reserve/target planning, not immediate
  conversion and not terminal tactics. The relevant credit horizon is whole
  trajectory: four errors prefer a different reserve target and four prefer a
  different action class whose benefit only appears downstream.
- Do not train on these ten human games. First run the same outcome-grounded
  action-class audit on a disjoint sample of early `035` reserve states from
  non-human legal games. This tests whether the pattern is a general policy
  defect or specific to openings the human exploited.
- If the reserve-target error generalizes, use those non-human matched outcomes
  to design a reserve-focused multi-generation/league learner with broad
  opponent coverage. If it does not, the next learner needs explicit opponent
  diversity rather than a generic reserve correction. In either case, do not
  return to a single local update against frozen `035`.

## Non-Human Reserve Generalization Audit

Implemented `run_reserve_generalization_audit.py` and
`src/splendor_ai/training/run_reserve_generalization_audit.py`. The experiment
was fixed before outcome inspection:

- Source corpus:
  `data/corpus_generation1_policy035_selfplay_038/replays.jsonl`, containing
  1,000 non-human legal-observation policy035 self-play games at seeds
  47000-47999. Human replay rows are explicitly rejected.
- The runner screened all 20,000 turns 0-19 and audited only states where the
  frozen deterministic policy035 argmax itself was a reserve. This prevents
  the source corpus's epsilon exploration from manufacturing policy errors.
- Sixty states from 60 unique games were sampled without looking at rollout
  outcomes: 15 each from turns 0-4, 5-9, 10-14, and 15-19.
- Every candidate action received 3 determinizations x 8 paired terminal
  continuations. Selection and completed state outcomes are crash-resumable,
  and the raw matched returns are retained for later analysis.
- The preregistered generalization gate required complete continuations, at
  least six strict per-state resolutions, a Wilson 95% rate lower bound above
  3%, resolved states in at least two turn bands, and mean resolved paired
  delta at least +0.20.

Final artifacts:

- `outputs/diagnostics/reserve_generalization_policy035_full.json`
- `outputs/diagnostics/reserve_generalization_policy035_full.md`
- `outputs/diagnostics/reserve_generalization_policy035_full.progress.json`

Results:

- The preregistered gate passes: 12/60 resolved mistakes (20.0%, Wilson 95% CI
  11.8%-31.8%), present in all four bands. The rates are 2/15, 3/15, 6/15,
  and 1/15 respectively, peaking at 40% on turns 10-14.
- Mean paired return improvement on resolved states is +0.836. Policy035 ranks
  the worse chosen reserve above the rollout-preferred alternative by a mean
  1.976 score units.
- Ten of the 12 resolutions remain significant under a post-audit Bonferroni
  robustness check correcting for selection of the best candidate alternative.
- Nine better alternatives are another reserve action and three are token
  takes; none is an immediate buy. The defect is mainly reserve target/tier
  selection, not a blanket tendency to reserve when another action class is
  obviously preferable.
- The recorded stochastic corpus action matches the audited deterministic
  policy action on 85% of all selected states and 11/12 resolved states. The
  result is therefore not driven by exploratory source actions.

Decision:

- The human reserve-planning diagnosis generalizes to disjoint non-human
  trajectories. Do not build a blanket reserve penalty or an endgame tactical
  override; both would target the wrong failure mode.
- Keep these 60 states as an audit/validation block. The next justified learning
  line is a fresh, larger reserve-choice corpus from disjoint games, weighted
  toward turns 5-14 and preserving comparisons among reserve targets as well as
  reserve-versus-take choices.
- Use that signal inside an iterative multi-generation learner evaluated against
  a broad league, not another one-shot correction against frozen `035`. Require
  fresh held-out reserve-ranking improvement before any paired game promotion
  gates, then test against `035`, legal PGS, and at least one held-out league
  opponent.

### Reserve Counterfactual Calibration Candidate `042`

The completed 500-state `035`-vs-`016` corpus was compiled by
`run_reserve_counterfactual_corpus.py` into a frozen, game-disjoint training
contract at `data/corpus_reserve_counterfactual_042`:

- 400 train games and 100 internal held-out games, with 100/25 games in each
  of the turn bands 0-4, 5-9, 10-14, and 15-19.
- The earlier 60-state policy035 self-play audit is a separate external check
  and is never optimized on.
- A separate 5,000-row legal policy035 parent-KL anchor stream was sampled from
  a different `035`-vs-`022` league block. It supplies no action targets.

`run_reserve_counterfactual_training.py` trained exactly one candidate from
legal `035`. It used the soft matched terminal-return distribution over every
evaluated candidate action, no one-hot replay CE, no behavior cloning, and
only the final score head (98,817 parameters) was trainable. State/action
encoders and action-set attention were frozen. The fixed schedule was four
epochs at learning rate `1e-5`, target temperature 0.35, parent-KL weight 2.0,
and CPU execution.

Artifacts:

- Candidate and metrics:
  `outputs/legal_action_scorer_042_reserve_counterfactual/`.
- Fixed uptake/drift gate:
  `outputs/legal_action_scorer_042_reserve_counterfactual/reserve_counterfactual_uptake_audit.json`.

Results:

- Soft return calibration improves decisively on the 100 held-out games:
  counterfactual CE delta `042 - 035` is -0.1246, CI95 [-0.1436, -0.1071].
  It also improves on the held-out 60-state external audit: -0.1128, CI95
  [-0.1334, -0.0940].
- Broad drift is bounded: 15/2,000 changed top actions (0.75%) and mean parent
  KL 0.00423.
- It fails the preregistered decisive-action gate. Held-out top-one return
  accuracy is only 20% versus `035`'s 18%, delta +2 points with CI95 [0, +5].
  Spearman improvement is likewise inconclusive (+0.005, CI95 [-0.01, +0.02]).
  The external 60-state best-action accuracy is unchanged at 20%.

Decision:

- Reject `042` before games. Soft target calibration alone is not a promotion
  criterion and must not be mistaken for improved play.
- Do not tune its epoch count, learning rate, KL weight, target temperature, or
  score-head scope. That would be a response to its failed top-action gate.
- The 500-state corpus remains valuable evidence that the policy has a real
  early reserve-target defect, but this specific soft-distribution imitation
  objective does not convert that evidence into better discrete decisions.

### Reserve Objective/Capacity Diagnostic

The bounded follow-up tested exactly two offline-only explanations for `042`'s
failure on the same frozen 400/100/60 split and 5,000-row parent-KL anchors.
Neither candidate was eligible for games.

- `043` changes only the target objective: exact highest-return candidate CE
  instead of the soft return distribution, while retaining the 98,817-parameter
  score-head-only scope and all `042` optimizer settings.
- `044` keeps that hard top target but unfreezes the full 666,114-parameter
  model. The corpus, four epochs, learning rate, anchor-KL weight, and gates
  remain unchanged.

The comparison artifact is
`outputs/diagnostics/reserve_counterfactual_objective_capacity_diagnostic.json`
and its concise Markdown report is `.md`.

Results:

- `043` does not improve top-one accuracy even on the 400 training games
  (-0.5 points, CI95 [-1.25, 0]) or the 100 held-out games (0 points). It
  therefore rejects soft-target dilution as the explanation.
- `044` improves held-out distributional CE by -0.560, but its training top-one
  gain is inconclusive (+2.5 points, CI95 [-0.25, +5.75]) and held-out top-one
  accuracy regresses by 3 points, CI95 [-10, +4]. Its broad parent KL is 0.0586,
  far above the fixed 0.01 ceiling.

Decision:

- Close this one-step matched-return policy-imitation family: soft versus hard
  target sharpness and score-head versus full-model capacity do not produce a
  generalizable discrete-action improvement under bounded drift.
- Do not run paired games, generate another corpus, or tune epochs, learning
  rate, KL, target temperature, or scope within this family.
- Any next learner must change the credit or decision process itself, rather
  than trying to fit the same offline return rankings more aggressively.

### Clean Entity Actor-Critic Smoke `005`

Implementation:

- `src/splendor_ai/training/entity_actor_critic.py`
- `src/splendor_ai/training/run_entity_actor_critic.py`
- `run_entity_actor_critic.py`
- `tests/test_entity_actor_critic.py`

The public actor checkpoint is initialized from frozen `entity_scale_004` and
retains the 13-type public entity contract. The separate privileged critic has
two additional entity types for actual hidden-deck cards and opponent blind
reserves; privileged state is never passed to the actor. Every canonical legal
successor has positive behavior probability through a 5% uniform mixture.
Rollouts retain the exact old action distribution and use terminal score outcome
with GAE/TD-lambda credit, clipped PPO updates, and explicit KL, entropy,
importance-ratio, exploration, action-coverage, and rollout-integrity gates.

CPU smoke artifacts are under `outputs/canonical_entity_actor_005_smoke`:

- 12 fresh league games, 375 actor decisions, zero stalls/timeouts;
- opponents evenly split across `entity004`, Greedy, legal shallow depth 1,
  `016`, `035`, and PUCT035 s8;
- minimum behavior probability 0.000734, five action types, 304/375 sampled
  actions different from the actor argmax, and rollout entropy 0.841;
- behavior log-probability reconstruction error 6.89e-7, max post-update
  importance ratio 1.0158, mean KL 2.04e-5, and post-update entropy 0.841;
- paired no-regression smoke: candidate 12, frozen parent 8, no draws or
  timeouts; every preregistered runner gate passed.

Decision:

- The architecture, legality boundary, stochastic rollout path, credit update,
  and conservative gates work end to end on CPU. A GPU is not justified by
  this workload yet.
- Do not promote `005` from a 20-game readout. The independent larger `006`
  generation below supersedes its apparent 12-8 edge.

### Clean Entity Actor-Critic Generation `006`

`outputs/canonical_entity_actor_006_seed90000` was trained from immutable `004`
with the same conservative hyperparameters as `005`, but on 72 fresh league
games at seed 90000 (12 each against `004`, Greedy, legal shallow depth 1,
`016`, `035`, and PUCT035 s8). Its 2,364 actor decisions had five action types,
minimum behavior probability 0.000541, 1,818 non-argmax samples, and zero
stalls/timeouts. All implementation gates passed.

Update diagnostics:

- behavior reconstruction error 7.84e-7;
- max importance ratio 1.0375 and mean ratio 0.99987;
- mean KL 6.88e-5 and normalized entropy 0.8335;
- policy loss -5.03e-6 and critic loss 0.1296.

Fresh strength gates:

- paired `006` versus frozen `004`, 100 games: 46-54, one timed-out game;
- same seeds versus `016`, 60 games: `006` 7-53 and `004` 8-52, with one
  timeout in each block;
- same seeds versus legal `035`, 60 games: `006` 2-58 and `004` 1-59;
- same seeds versus Greedy, 60 games: `006` 54-6 and `004` 55-5;
- `006` versus legal shallow depth 1, 60 games: 9-51, matching the 15% score
  share of `004`'s earlier 5-15 block. All rule-opponent games completed with
  no stalls/timeouts or loop fallbacks.

Artifacts:

- `outputs/canonical_entity_actor_006_seed90000/actor_critic_metrics.json`
- `outputs/benchmarks/entity_actor006_vs016_seed92000_g60.json`
- `outputs/benchmarks/entity004_vs016_seed92000_g60.json`
- `outputs/benchmarks/entity_actor006_vs035_seed93000_g60.json`
- `outputs/benchmarks/entity004_vs035_seed93000_g60.json`
- `outputs/benchmarks/entity_actor006_rules_seed94000_g60.json`
- `outputs/benchmarks/entity004_greedy_seed94000_g60.json`

Decision:

- Reject `006` as a no-movement generation and retain `004` as the frozen
  parent. The `005` 12-8 result was small-sample noise, not replicated progress.
- Do not spend another full league/search gate on the identical update recipe.
  Add an offline effective-movement gate first: report gradient norm, actor
  parameter/logit movement, advantage-action covariance, and target-state KL.
  Then test a KL-targeted multi-epoch or higher-learning-rate update that aims
  for measurable but bounded movement before launching games.
- CPU remains appropriate. The expensive portion is serial simulation/search,
  not the small actor/critic optimization; a GPU would not fix this blocker.

### KL-Targeted Entity Actor-Critic Generation `007`

The actor-critic implementation now records pre-clipping gradient norms,
parameter movement, centered-logit movement, probability total variation,
argmax changes, per-epoch rollout-state KL, advantage/log-probability
covariance, and mean raw advantage by sampled action type. Actor training can
run for multiple epochs but stops after the first epoch whose exact rollout-
state KL reaches `target_kl`. The runner can skip parent games unless all
offline movement and safety gates pass.

`outputs/canonical_entity_actor_007_kl_target` used 36 fresh league games at
seed 100000, six maximum actor epochs, actor learning rate 3e-5, target KL
0.002, minimum effective KL 0.0005, centered-logit movement floor 0.005, and
hard KL ceiling 0.01. The six league opponents were evenly represented.

Results:

- 1,156 actor steps, zero rollout stalls/timeouts, and five action types;
- target reached after one epoch: mean KL 0.002415, max importance ratio 1.194,
  normalized entropy 0.842, and policy loss -0.01356;
- first-batch/mean/max pre-clipping gradient norms 0.937/1.355/6.339;
- parameter delta L2 0.02473, relative delta 0.000654;
- mean/max centered-logit absolute movement 0.04244/0.2562;
- mean/max probability TV 0.02714/0.07068 and 4.93% argmax changes;
- advantage/log-probability correlation +0.254. Mean raw advantages were
  +0.104 for visible buys, +0.070 for blind reserves, -0.0075 for reserved
  buys, -0.0225 for visible reserves, and -0.0961 for token takes.

Strength:

- initial paired parent gate, 60 games at seed 101000: 37-22-1 for `007`, with
  one repetition cutoff;
- independent paired confirmation, 100 games at seed 102000: 52-48 for `007`,
  with two repetition cutoffs;
- aggregate readout 89-70-1, score share 0.559, but the larger independent block
  does not establish a repeatable advantage.

Decision:

- The no-movement blocker is fixed: the higher learning rate creates measurable
  policy movement and the KL stop bounds it automatically. This is a successful
  learning-mechanism experiment.
- Do not promote `007`; retain frozen `004`. Do not run the expensive broad or
  shallow-search gates after a 52-48 independent confirmation.
- The next eligible experiment is the same fixed KL-target recipe on another
  disjoint rollout seed. Require another independently trained candidate to
  clear a preregistered parent threshold before broad-league confirmation.
- CPU remains sufficient; optimization and diagnostics completed quickly
  relative to rollout/search simulation.

Artifacts:

- `outputs/canonical_entity_actor_007_kl_target/actor_critic_metrics.json`
- `outputs/benchmarks/entity_actor007_vs004_seed102000_g100.json`

### Active Overnight Reserve Corpus

A 500-state matched-return corpus job was launched at 2026-07-14 01:28 local
time through `run_reserve_training_500_overnight.ps1`.

- Source: `data/corpus_generation1_policy035_vs016_038_retry/replays.jsonl`,
  seeds 48000-48499. These 500 games are disjoint from the 60-state validation
  source at seeds 47000-47999 and add opponent diversity through baseline 016.
- Selection completed successfully: all 5,000 policy035 turns 0-19 were
  screened; 500 deterministic policy035 reserve states were selected, 125 per
  turn band and exactly one from each of 500 unique games.
- Evaluation is fixed at 3 determinizations x 8 paired terminal continuations
  per candidate. Human rows are forbidden and selection never uses outcomes.
- Progress and source snapshots:
  `data/corpus_reserve_choice_policy035_vs016_500/reserve_matched_returns.progress.json`.
- Final outputs, when complete:
  `data/corpus_reserve_choice_policy035_vs016_500/reserve_matched_returns.json`
  and `.md`.
- Logs and PID are in the same directory as `overnight.stdout.log`,
  `overnight.stderr.log`, and `overnight.pid`. At launch the wrapper PID was
  45760 and Python PID was 46652; treat those as stale if the job has completed.
- The wrapper creates a temporary Windows system-awake request and releases it
  on exit. It is crash-resumable after every completed state. Do not launch a
  second copy while the recorded process is active. If interrupted, rerun the
  same wrapper; cached selection and completed states will be reused.

## Useful Commands

Full test command:

```powershell
$env:PYTHONPATH='src'; python -m pytest -q --basetemp .codex_pytest_tmp_all
```

Current full test status:

- 278 passed after legal hidden-information search, PUCT batching, root-visit
  replay/training telemetry, paired direct checkpoint gates, the generation-1
  pilot, the held-out uptake audit, conflict-free root-visit correction,
  counterfactual leaf-quality audit, action-value corpus export/training,
  residual PUCT error attribution, disjoint root-prior confirmation,
  CPU high-budget action-quality gating, resumable mid-game headroom auditing,
  plain-PUCT ablation support, true leave-one-out trajectory construction,
  pre-training signal gating, actor movement/KL auditing, and held-out
  human-game forensic action-quality auditing, and disjoint non-human reserve
  generalization auditing with resumable matched terminal continuations, and
  reserve counterfactual corpus compilation, constrained training, uptake
  gating, fixed objective/capacity diagnosis, canonical entity modeling, and
  the public-actor/privileged-critic smoke generation.

## Files To Read Next

- `AGENT_CURRENT.md`
- `AGENT_HANDOFF.md` only if historical detail is needed
- `src/splendor_ai/training/run_policy_diff_against_teacher.py`
- `src/splendor_ai/training/run_policy_search_disagreement.py`
- `src/splendor_ai/training/run_conversion_diagnostics.py`
- `src/splendor_ai/training/run_pgs_corpus_audit.py`
- `src/splendor_ai/training/run_constrained_preference_corpus.py`
- `src/splendor_ai/training/run_policy_drift.py`
- `src/splendor_ai/training/run_candidate_harness.py`
- `src/splendor_ai/training/run_trajectory_advantage_corpus.py`
- `src/splendor_ai/training/run_group_relative_trajectory_corpus.py`
- `src/splendor_ai/training/run_trajectory_policy_gradient.py`
- `src/splendor_ai/training/run_rollout_action_credit.py`
- `src/splendor_ai/training/run_rollout_credit_training.py`
- `src/splendor_ai/training/improve_replay_actions.py`
- `src/splendor_ai/training/run_legal_action_supervised.py`
- `src/splendor_ai/bots/search_bot.py`
- `src/splendor_ai/bots/puct_bot.py`
- `src/splendor_ai/eval/run_puct_benchmark.py`
- `src/splendor_ai/training/run_root_visit_target_audit.py`
- `src/splendor_ai/training/run_root_visit_correction.py`
- `src/splendor_ai/training/run_leaf_quality_audit.py`
- `src/splendor_ai/training/run_action_value_training.py`
- `src/splendor_ai/training/run_residual_error_audit.py`
- `src/splendor_ai/training/run_root_prior_control_ablation.py`
- `src/splendor_ai/training/run_puct_budget_ablation.py`
- `src/splendor_ai/training/run_midgame_headroom_audit.py`
- `src/splendor_ai/training/run_group_relative_signal_audit.py`
- `src/splendor_ai/training/run_actor_movement_audit.py`
- `src/splendor_ai/training/run_human_game_forensic_audit.py`
- `src/splendor_ai/training/run_reserve_generalization_audit.py`
- `src/splendor_ai/training/entity_actor_critic.py`
- `src/splendor_ai/training/run_entity_actor_critic.py`
