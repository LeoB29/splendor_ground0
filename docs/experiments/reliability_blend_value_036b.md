# Reliability-Gated Value Hybrid 036b

Frozen before validation selection, fresh confirmation generation, or gameplay.
This is a new experiment motivated by `036`; it does not waive or reinterpret
either failed `036` gate.

## Fixed inputs and candidate grid

- Freeze the selected `036` neural checkpoint byte-for-byte. No gradient update,
  target change, new representation, or distributional-head recalibration is
  allowed.
- For a state, compute `alpha * learned_value + (1-alpha) * heuristic_value`.
  The learned value remains the equal blend of score-margin and WDL expectations.
- `alpha` comes from one of four fixed phase profiles for turns
  `(0-14, 15-29, 30-44, 45+)`:
  - opening heuristic: `(0, 1, 1, 1)`;
  - ramp: `(0, 0.5, 0.75, 1)`;
  - fast ramp: `(0, 0.75, 1, 1)`;
  - soft opening: `(0.25, 0.75, 1, 1)`.
- Cross those profiles with maximum-WDL-probability floors `0`, `0.45`, and
  `0.55`. Below the floor, `alpha` becomes zero and the heuristic is used.
  This produces exactly 12 candidates.

## Validation-only selection

- Scalar metrics use the existing game/seed-group-disjoint `032`/`033`
  validation tensors only.
- Successor consistency uses a deterministic reservoir of four states in each
  generation x opponent x actor-seat x phase stratum from validation groups
  448-511. Outcomes do not influence state selection.
- A candidate is eligible only if it passes every unchanged `036` offline gate,
  including the 15% macro error improvement, ordering, midgame, phase/opponent
  no-collapse, calibration, and all three successor-consistency requirements.
- Among eligible candidates, select the minimum
  `macro_opponent_seat_MAE - 0.1 * ordering_concordance`. If none is eligible,
  stop without generating confirmation data.

## Fresh confirmation and conditional games

After selection only, collect 128 new groups / 512 games at seeds
381000-381127, four stochastic-035 trajectories per group, exactly 16 groups in
each opponent/actor-seat cell. Reapply the identical `036` offline gates on this
untouched corpus and a deterministic eight-state-per-cell/phase successor sample.

Only a complete offline pass unlocks:

1. 100 paired seeds at 382000-382099, reliability-value PUCT64 versus the same
   PUCT64 with handcrafted leaves; require zero failures, score share at least
   55%, and bootstrap CI95 lower bound above 50%.
2. Only that pass unlocks 100 paired seeds at 383000-383099 versus raw `035`,
   with identical gates.

No threshold may be relaxed after evidence. A direct search pass authorizes the
synchronous search-every-move AlphaGo-style loop but does not itself promote a
new champion. CPU is sufficient for this experiment.
