# Scaled Public-Belief Value Bootstrap 036

This protocol was frozen before tensor compilation, training, fresh confirmation
collection, or gameplay evidence.

## Hypothesis

The previous legal canonical value result (`029`) was underpowered: it trained on
240 games and improved error without learning useful ordering. The combined
fresh `032`/`033` leagues contain 4,096 opponent/seat-balanced games and 129,482
acting-035 positions. A calibrated distributional value model, warm-started from
the strongest compatible canonical representation, may now be accurate enough
to make information-legal PUCT a genuine policy-improvement operator.

## Frozen data and model contract

- Sources: `data/league_generation_032` and
  `data/trust_region_generation_033`.
- Preserve each source's original group boundary: groups 0-447 train and
  448-511 validation. This yields 3,584 train games and 512 validation games,
  with 56/8 groups in every generation x opponent x acting-seat cell.
- Cycle all 64 generation x opponent x acting-seat x phase strata equally in
  every training epoch. Phases are turns 0-14, 15-29, 30-44, and 45+.
- The model receives only `canonical_public_belief_v1` entities visible to the
  acting player. Hidden deck order and opponent blind-reserve identity are never
  model inputs.
- Initialize all compatible entity embeddings, input layers, and transformer
  tensors from promoted clean representation `023`; initialize both new heads
  randomly. The stronger overall policy `035` is not a compatible canonical
  representation and remains the policy prior and champion.
- Predict a 41-class clipped integer final-score-margin distribution over
  [-20, 20] and a three-class loss/draw/win distribution. Optimize margin CE +
  0.5 WDL CE for eight epochs.
- Select exactly once on the lowest validation macro-stratum calibration NLL
  minus 0.25 times within-stratum ordering concordance.

## Untouched confirmation and offline gates

Only after checkpoint selection, generate 128 new seed groups (512 games) at
seeds 378000-378127: 16 groups in each opponent/acting-seat cell and four
stochastic-035 trajectories per group. Promotion to gameplay requires all of:

- at least 15% lower macro opponent/seat MAE than the handcrafted heuristic;
- paired-game MAE bootstrap CI95 entirely below zero;
- ordering concordance at least 60% and at least three points above heuristic;
- higher macro-stratum ordering correlation;
- no opponent or phase aggregate MAE more than 5% worse than heuristic;
- turns 15-44 ordering at least 58% and at least two points above heuristic;
- WDL ECE10 at most 10%;
- positive policy/value successor rank correlation, policy-top successor mean
  value percentile at least 55%, and value-best successor in policy top three at
  least 50%.

## Conditional gameplay

If and only if every offline gate passes:

1. Run 100 paired seeds at 379000-379099: learned-value PUCT64 versus the same
   policy-prior PUCT64 with the handcrafted leaf. Require zero failures, at
   least 55% score share, and bootstrap CI95 lower bound above 50%.
2. Only an A/B pass unlocks 100 paired seeds at 380000-380099: learned-value
   PUCT64 versus raw `035`, with the same gates.
3. A direct pass authorizes—but does not itself promote—a separately
   checkpointed synchronous loop: search on every move, retain full visit
   distributions from wins and losses, distill them, then collect again using
   the newly frozen network. `035` remains champion until broader promotion.

CPU is appropriate for this falsification. A GPU becomes justified for repeated
synchronous search/distillation iterations, not for fixing an invalid value
target or weak search operator.

## Result

Completed on CPU in 754 seconds. The tensor ceiling was raised losslessly from
64 to 96 after compilation encountered a legal 65-entity position; this happened
before any tensor artifact or model evidence and did not alter the protocol.

- Compiled 113,248 training and 16,234 validation positions. Exact phase
  balancing yielded 2,205 rows per stratum and 141,120 samples per epoch.
- Epoch 3 was selected. Checkpoint:
  `outputs/scaled_public_belief_value_036/canonical_entity_distributional_value_width64_best.pt`,
  SHA256 `537b3100980d18093042bf377a009da11a90f4fa63ad4beeee5c9cd5b95c0fa6`.
- The fresh confirmation corpus contained all 128 groups / 512 games, exactly
  16 groups per opponent/seat cell, 0 stalls, 0 timeouts, and 16,164 acting-035
  positions. Observed nongreedy rate was 14.54%.
- Candidate macro opponent/seat MAE was 0.28768 versus heuristic 0.36015, a
  20.1% improvement; paired-game MAE CI95 upper bound was -0.06068.
- Ordering concordance was 66.38% versus 61.47%; macro ordering correlation was
  0.446 versus 0.299. Turns 15-29 reached 61.50% versus 53.98%, and turns 30-44
  reached 69.81% versus 55.58%. WDL ECE10 was 4.51%.
- Successor Spearman was +0.202 and the 035-top successor's mean value percentile
  was 70.35%.

Decision: `reject_offline_keep_035`. Two frozen gates failed. Turn 0-14 MAE was
0.32 versus heuristic 0.30, a 6.7% regression exceeding the 5% limit. The
value-best successor was in 035's policy top three on 48.05% of sampled states,
below the 50% limit. Thresholds were not relaxed after evidence. No PUCT games
ran, no search teacher was authorized, and `035` remains champion.
