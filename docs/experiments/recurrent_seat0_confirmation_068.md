# Recurrent Seat-0 Confirmation (`068`)

Frozen on 2026-08-02 after `067` found that the `066` seat-0 failure did not
replicate across disjoint panels and before running any fresh seat-targeted
game. This is a CPU-only, no-training confirmation. It cannot directly promote
a checkpoint.

## Question

On a fresh panel where the recurrent policy always occupies seat 0, can
history-required generation 2 establish statistical noninferiority to
epoch-zero `035` at the frozen `-0.05` score-share floor, without opponent or
split-half collapse?

## Frozen prerequisites

- `067` audit:
  `outputs/recurrent_seat_heterogeneity_audit_067/result.json`
- `067` SHA-256:
  `03bb2d7386ab71d93302d02f4da266859c7e472b7facefeaa94f81c02c264662`
- Epoch-zero checkpoint SHA-256:
  `3dd945ba0c525e936603042040b2c33ab70ad8552e007b6c8f2dc4619813290a`
- Generation-2 checkpoint SHA-256:
  `3ed0bd7e2dcda21646c7ac4f68e2d0ffcfe0bdacf0c151c453c52ea6a08fe09e`
- Frozen raw legal `035` SHA-256:
  `4422a82b593340eecd02af3a0941f24f0318cc7453e6a0739fd9f524539af2cd`
- Frozen `016` SHA-256:
  `876ac23a005d367ac48b3619bb7cdcb6e0230a3e9e97943d7eecdc6d07d1b056`
- Frozen clean canonical `023` SHA-256:
  `29ef7c13c5553ec88c8bcb98f7378c925f0e600e2fcf3d944df5926efed7a070`

Require exact `067` decision
`recurrent_seat_failure_not_replicated_authorize_targeted_confirmation_preregistration`
and every `067` integrity gate true.

## Fresh fixed panel

Use 100 fresh matched seeds against each static opponent:

- raw `035`: `680000--680099`;
- `016`: `680200--680299`;
- clean canonical `023`: `680400--680499`;
- legal shallow search: `680600--680699`.

For every opponent/seed, play exactly two games:

1. generation 2 in seat 0 versus the opponent in seat 1;
2. epoch-zero control in seat 0 versus the opponent in seat 1.

This gives 400 paired-seed clusters and 800 games. Do not play the recurrent
policy in seat 1. Seed bands are disjoint from all prior experiment,
collection, validation, gate, and diagnostic namespaces.

Use maximum turn 400, repetition cutoff 4, no-progress cutoff 60, and the
unchanged information-legal shallow contract. `completed` and
`forced_pass_deadlock` are valid. Do not replace, retry, include, or exclude a
seed based on its result. Any other ending invalidates the fixed panel.

## Paired inference

For each opponent/seed cluster, record candidate-minus-control:

- seat-0 score point;
- seat-0 final score margin;
- candidate and control outcome/score summaries.

Report aggregate and per-opponent values, alternating seed-ordinal halves,
leave-one-opponent-out values, and leave-one-seed-out ranges.

Use a deterministic opponent-stratified paired-cluster bootstrap for aggregate
score and margin CI95 (`100,000` draws, seed `680068`). Use 50,000 paired-seed
draws per opponent, seeds `680100--680103`. Report deterministic 100,000-draw
two-sided sign-flip p-values for score and margin using seeds `680168` and
`680169`. Never treat games, turns, or decisions as independent observations.

## Integrity gates

Require:

- all prerequisite hashes and the exact `067` authorization;
- exact four opponents, 100 rows and exact seed band per opponent;
- exactly 800 games, all with the recurrent policy in seat 0;
- all endings valid, zero illegal actions, and finite metrics;
- exact recomputation of candidate/control shares and paired deltas;
- checkpoint hashes unchanged;
- no training, model selection, seed replacement, or adaptive stopping.

Any failure yields
`recurrent_seat0_confirmation_invalid_no_strength_decision`.

## Strength gates

The frozen practical seat-0 score-share floor is `-0.05`. Require all:

1. aggregate candidate-minus-control score-share delta at least `-0.05`;
2. aggregate score-share CI95 lower bound at least `-0.05`;
3. aggregate mean paired score-margin delta nonnegative;
4. aggregate margin CI95 lower bound at least `-1.0`;
5. at least three of four opponent score-share deltas at least `-0.05`;
6. no opponent score-share delta below `-0.10`;
7. no opponent score-share CI95 upper bound below `-0.05`;
8. both even/odd aggregate score-share deltas at least `-0.05`.

Also report, without requiring, whether the aggregate score-share CI95 lower
bound is above zero (`strong_positive_evidence`).

## Decisions

- Integrity failure:
  `recurrent_seat0_confirmation_invalid_no_strength_decision`.
- All strength gates pass:
  `recurrent_seat0_confirmation_pass_authorize_stage_c_preregistration`.
- Aggregate score CI95 upper bound below `-0.05` or margin CI95 upper bound
  below `-1.0`:
  `recurrent_seat0_confirmation_systematic_failure_change_direction`.
- Any other strength failure:
  `recurrent_seat0_confirmation_failed_no_scale`.

A pass resolves only the outstanding seat guardrail and authorizes a
separately frozen Stage C recurrent-league scale protocol. It does not
retroactively pass `066`, promote generation 2, replace champion `035`,
authorize changing gates, or automatically justify GPU rental.
