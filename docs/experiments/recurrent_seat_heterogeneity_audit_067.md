# Recurrent Seat-Heterogeneity Audit (`067`)

Frozen on 2026-08-02 after `066` failed only its recurrent-seat guardrail and
before computing any pooled or cross-panel uncertainty. This is a CPU-only,
no-game, no-training audit. It cannot retroactively pass `066` or promote a
checkpoint.

## Question

Is the `066` seat-0 score-share delta of `-0.0575` evidence of a stable
seat-dependent weakness, or an unstable split that reverses in the disjoint
`065` panel and disappears when both panels are clustered correctly?

## Frozen prerequisites

- `065` result:
  `outputs/recurrent_broad_failure_diagnosis_065/result.json`
- `065` SHA-256:
  `616e0f6eb0ff7b968cfc0f9d60c51c2c186557c179c07a52586b978d784e0b4d`
- `066` result:
  `outputs/recurrent_broad_confirmation_066/result.json`
- `066` SHA-256:
  `dbbdedbb74b59ea5d0d6562eced9c08afdd43795107e2e3567e9dddb60ff162c`

Require exact decisions:

- `065`:
  `recurrent_broad_result_inconclusive_authorize_confirmation_preregistration`;
- `066`: `recurrent_broad_confirmation_failed_no_scale`.

Require passed `065` replay integrity, all `066` integrity gates, exactly one
false `066` strength gate named `both_seats_noninferior`, and exact disjoint
paired records: 40 seed clusters in `065`, 200 in `066`, with the four frozen
opponents and no duplicate `(panel, opponent, seed)` key.

## Exact recomputation

From each stored paired-seed row, use the already recorded:

- seat-0 and seat-1 candidate-minus-control score-point deltas;
- seat-0 and seat-1 score-margin deltas;
- opponent, seed, and seed ordinal.

Recompute and require exact agreement within `1e-12` with each panel's stored
aggregate, opponent, seat, and split-half summaries. Perform no simulation,
model loading, action evaluation, training, filtering, or seed replacement.
Any mismatch yields `recurrent_seat_audit_invalid_no_inference`.

## Clustered inference

Pool all 240 disjoint seed clusters, preserving the two seat outcomes from one
seed as a joint row. Use a deterministic 100,000-draw stratified cluster
bootstrap, independently resampling within each exact `(panel, opponent)`
stratum, seed `670067`. On every common draw report CI95 for:

- pooled seat-0 and seat-1 score-share deltas;
- pooled seat-0 and seat-1 margin deltas;
- seat interaction `seat1 - seat0` for score share and margin;
- each panel's seat estimates;
- cross-panel difference `066 - 065` for each seat.

Also report pooled opponent-by-seat point estimates and 50,000-draw CI95 values
using seeds `670100--670103`, plus panel and opponent contribution counts.
Seed clusters, not individual games or decisions, are the inferential unit.

## Interpretation flags

The frozen practical floor remains `-0.05`. Report:

- `seat0_mean_below_floor`, `seat1_mean_below_floor`;
- `seat0_systematically_below_floor`, `seat1_systematically_below_floor` when
  the corresponding CI95 upper bound is below `-0.05`;
- `seat0_proven_noninferior`, `seat1_proven_noninferior` when the CI95 lower
  bound is at least `-0.05`;
- `score_seat_interaction_systematic` when the seat-interaction CI excludes
  zero;
- `seat0_panel_difference_systematic` and
  `seat1_panel_difference_systematic` when cross-panel CIs exclude zero;
- `seat_signs_reverse_across_panels` when both seat point estimates change
  sign from `065` to `066`;
- `pooled_both_seats_within_floor` when both pooled point estimates are at
  least `-0.05`.

## Decisions

1. Integrity failure:
   `recurrent_seat_audit_invalid_no_inference`.
2. Either seat systematically below the floor:
   `recurrent_seat_failure_systematic_change_direction`.
3. Both panel seat signs reverse, both pooled means are within the floor, and
   neither seat is systematically below it:
   `recurrent_seat_failure_not_replicated_authorize_targeted_confirmation_preregistration`.
4. A pooled mean is below the floor, a panel difference is systematic, or the
   seat interaction is systematic without condition 2:
   `recurrent_seat_effect_heterogeneous_authorize_targeted_confirmation_preregistration`.
5. Otherwise:
   `recurrent_seat_audit_complete_no_scale_authorized`.

Any targeted-confirmation authorization permits only a separately frozen,
fresh seat-0-focused matched gate. It does not pass `066`, authorize Stage C,
promotion, retraining, objective changes, or GPU rental.
