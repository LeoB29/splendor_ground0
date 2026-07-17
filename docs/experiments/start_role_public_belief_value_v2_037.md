# Start-Role Public-Belief Value v2 (`037`)

Frozen before v2 training, validation selection, fresh confirmation, or games.
This is the final focused value/PUCT experiment. Failure closes the current value
bootstrap and moves the project to hierarchical plans/options.

## Objective representation repair

`canonical_public_belief_key` contains `state.start_player == player_id`, but the
v1 canonical entity tensor does not. A regression test demonstrates that two
states with different legal start-player roles have different public-belief keys
but identical v1 entity encodings. In the engine, seat 0 always starts; v1 asks a
model to recover this role indirectly from continuous turn index.

All v1 encoders and checkpoints remain byte-compatible. The new v2 value model:

- adds a dedicated two-value `observer_is_start_player` embedding only to the
  global entity before the transformer;
- initializes that embedding to exactly zero;
- transfers every v1 `036` tensor, producing exactly identical initial logits;
- derives the deployment flag from `state.start_player == observer_id`.

## Frozen training and selection

- Reuse the exact `032`/`033` group splits and tensors: groups 0-447 train,
  448-511 validation, with exact generation/opponent/seat/phase cycling.
- Retain 41-way score-margin CE + 0.5 three-way WDL CE.
- Train six epochs. Existing parameters use learning rate `3e-5`; only the new
  start-role embedding uses `3e-4`. Weight decay is `1e-4` except on the new
  embedding.
- Include zero-update epoch 0 as a candidate. Select the lowest validation macro
  calibration NLL minus `0.25 * ordering concordance`.
- Deploy through the already validated reliability ramp `(0, 0.5, 0.75, 1)`;
  do not retune it.
- Before fresh data, require every original `036` offline gate, at least 0.5%
  selection-score improvement over epoch 0, at least 1% lower MAE than the v1
  ramp separately for starting and second player, and no per-seat correlation
  regression. Successor consistency uses four deterministic reservoir states in
  every generation/opponent/seat/phase validation stratum.

## Fresh confirmation and games

Only a validation pass generates 128 groups / 512 games at seeds
384000-384127, with four stochastic-035 trajectories per group and exact
opponent/actor-seat balance. Reapply all original gates plus the separate
per-seat v1-improvement and correlation gates on untouched confirmation.

Only a complete offline pass unlocks 100 paired seeds / 200 games at
385000-385099: v2 reliability-value PUCT64 versus heuristic-leaf PUCT64. Require:

- zero stalls and timeouts;
- overall score share at least 55%;
- paired bootstrap CI95 lower bound above 50%;
- candidate score share at least 50% from each seat independently.

Only that pass unlocks the same gate versus raw `035` at seeds 386000-386099.
A direct pass authorizes synchronous search-every-move visit-distribution
distillation but does not automatically promote a champion. Thresholds may not
be relaxed or blocks extended post hoc. CPU is sufficient.

## Result

Rejected at the frozen validation gate on 2026-07-17. Epoch 1 was selected:

- epoch-0 score: `3.4039124805`;
- selected score: `3.3968630308`, a `0.2071%` improvement versus the required
  `0.5%`;
- v2 aggregate MAE: `0.2837901023` versus v1 `0.2834403618`;
- starting-player MAE: `0.2764297573` versus v1 `0.2733698104` (worse by
  `1.12%`);
- second-player MAE: `0.2912160264` versus v1 `0.2936006396` (better by
  `0.81%`, below the `1%` floor).

The three failed gates were
`selection_improves_at_least_0.5pct_vs_epoch0`,
`seat0_mae_improves_at_least_1pct_vs_v1`, and
`seat1_mae_improves_at_least_1pct_vs_v1`. Every original offline gate and both
seat-correlation gates passed. Successor consistency was also healthy on its
256-state reservoir (mean Spearman `0.1171`, policy-top value percentile
`0.7143`, value-best in policy top three `57.42%`).

Per protocol, no confirmation corpus or gameplay was generated and no
deployment checkpoint was retained. Decision:
`reject_v2_validation_close_value_puct`. The next direction is hierarchical
plans/options, not another public-belief leaf-value adjustment.
