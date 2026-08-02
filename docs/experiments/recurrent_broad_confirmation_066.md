# Recurrent Broad Confirmation (`066`)

Frozen on 2026-08-02 after `065` classified the `064` broad result as
inconclusive and before running any fresh confirmation game. This is a
CPU-only, no-training strength confirmation. It cannot directly promote a
checkpoint.

## Question

Across a materially larger fresh matched sample, is history-required
generation 2 broadly improving and practically noninferior to epoch-zero
`035`, without hiding an opponent, seat, or split-half collapse?

## Frozen prerequisites

- `065` diagnostic:
  `outputs/recurrent_broad_failure_diagnosis_065/result.json`
- `065` SHA-256:
  `616e0f6eb0ff7b968cfc0f9d60c51c2c186557c179c07a52586b978d784e0b4d`
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

Require the exact `065` decision
`recurrent_broad_result_inconclusive_authorize_confirmation_preregistration`
and its passed 160-game replay integrity.

## Fresh matched panel

Compare frozen generation 2 and frozen epoch zero independently against:

- raw `035`: seeds `660000--660049`;
- `016`: seeds `660100--660149`;
- clean canonical `023`: seeds `660200--660249`;
- legal shallow search: seeds `660300--660349`.

For every opponent/seed, play candidate and control in both recurrent seats.
This produces 50 paired-seed clusters and 200 games per opponent, 200 clusters
and 800 games overall. These namespaces are disjoint from all `061--065`
collection, validation, parent-gate, diagnostic, and broad-gate seeds.

Use maximum turn 400, repetition cutoff 4, no-progress cutoff 60, and the
unchanged information-legal shallow contract. `completed` and
`forced_pass_deadlock` are valid. Do not replace or exclude a seed based on
its result. Any other ending makes the confirmation invalid rather than
silently changing the panel.

## Paired inference

Keep both seats and candidate/control outcomes from one opponent/seed together
as one cluster. Report:

- aggregate and per-opponent score-share and score-margin deltas;
- recurrent-seat deltas;
- alternating even/odd seed-ordinal halves;
- leave-one-opponent-out and leave-one-seed-out ranges;
- opponent-stratified paired-cluster bootstrap CI95 for aggregate score share
  and margin (`100,000` draws, seed `660066`);
- paired-cluster bootstrap CI95 per opponent (`50,000` draws, seeds
  `660100--660103`);
- deterministic `100,000`-draw two-sided sign-flip tests for score and margin
  (seeds `660166` and `660167`).

Decisions are never treated as independent observations. Score share is the
primary strength metric; score margin is a guardrail.

## Integrity gates

Require:

- every prerequisite path/hash and the `065` authorization decision;
- exact opponent labels, seed bands, 50 rows and 200 games per opponent;
- exactly 800 games total;
- all endings valid and zero illegal actions;
- finite metrics and exact recomputation of all stored score shares;
- no training, checkpoint mutation, seed replacement, or post-result model
  selection.

Any integrity failure yields
`recurrent_broad_confirmation_invalid_no_strength_decision`.

## Strength gates

Require all of the following:

1. aggregate candidate-minus-control score-share delta strictly positive;
2. aggregate score-share CI95 lower bound at least `-0.05`;
3. aggregate mean paired score-margin delta strictly positive;
4. aggregate margin CI95 lower bound at least `-1.0`;
5. at least three of four opponent score-share deltas nonnegative;
6. no opponent score-share delta below `-0.05`;
7. both recurrent-seat score-share deltas at least `-0.05`;
8. both even/odd aggregate score-share deltas at least `-0.05`.

Also report, but do not require, `strong_positive_evidence` when the aggregate
score-share CI95 lower bound is strictly above zero.

## Decisions

- Integrity failure:
  `recurrent_broad_confirmation_invalid_no_strength_decision`.
- All strength gates pass:
  `recurrent_broad_confirmation_pass_authorize_stage_c_preregistration`.
- Aggregate score or margin CI95 upper bound below zero:
  `recurrent_broad_confirmation_systematic_failure_change_direction`.
- Any other strength failure:
  `recurrent_broad_confirmation_failed_no_scale`.

A pass authorizes only a separately frozen Stage C recurrent-league scale
protocol. It does not promote generation 2, replace champion `035`, authorize
GPU rental automatically, or permit changing these gates after observing the
confirmation.
