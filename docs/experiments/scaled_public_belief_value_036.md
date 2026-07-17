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
