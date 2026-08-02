# Recurrent Broad-Failure Diagnosis (`065`)

Frozen on 2026-08-02 after `064` rejected Stage C scale and before replaying
the broad games or computing any uncertainty or action-localization statistic.
This is a CPU-only, no-training, no-new-seed diagnostic. It cannot change the
`064` decision or promote a checkpoint.

## Question

Did `064` reveal a systematic broad regression, a localized failure against
`016`/clean `023`, or ordinary uncertainty from only ten paired seeds per
opponent? Where and how does generation 2 depart from epoch-zero `035` in the
games responsible for the observed broad deltas?

## Frozen prerequisites

- Corrected `064` result:
  `outputs/history_required_recurrent_repair_064/result.json`
- Result SHA-256:
  `7a3094d37e6d37961b44f74029330c7083f5da3af7a9a03f8050b62bc59cf2e6`
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

Require the `064` formal decision
`history_required_repair_failed_diagnose_before_scale`, two accepted
generations, a valid 160-game final population gate, and
`aggregate_delta_positive` as its only false final gate.

## Exact instrumented replay

Replay only the already used final broad seeds and contracts:

- raw `035`: `643000--643009`;
- `016`: `643100--643109`;
- clean `023`: `643200--643209`;
- legal shallow search: `643300--643309`;
- candidate generation 2 and epoch-zero control;
- both recurrent seats for every model/opponent/seed combination;
- maximum turn 400, repetition cutoff 4, and no-progress cutoff 60.

This reconstructs the same 160 games and is not fresh gameplay evidence.
Require exact equality with every stored `064` outcome field after normalizing
only the tracing wrapper's class name. Require all endings to remain valid and
the recomputed opponent and aggregate score shares/deltas to agree within
`1e-12`. Any mismatch yields
`recurrent_broad_diagnosis_invalid_replay_mismatch` and no interpretation.

## Paired-seed uncertainty

For every opponent/seed cluster, keep candidate and control outcomes for both
recurrent seats together. Compute:

- candidate-minus-control score-point share and score-margin delta;
- opponent and recurrent-seat breakdowns;
- even/odd seed-ordinal split halves;
- leave-one-opponent-out and leave-one-seed-out aggregate ranges;
- stratified paired-cluster bootstrap CI95 for the aggregate, resampling ten
  seeds independently within every opponent (`100,000` draws, seed `650065`);
- paired-cluster bootstrap CI95 per opponent (`50,000` draws, seeds
  `650100--650103`);
- deterministic `100,000`-draw sign-flip two-sided p-values for aggregate
  score-share and margin deltas (seeds `650165` and `650166`).

Decisions are clustered by seed, never treated as independent samples.

## Policy-change localization

During each replayed candidate and control trajectory, evaluate both frozen
generation 2 and epoch zero on the acting recurrent player's exact legal
public state and transcript. Record:

- legal-action count, phase (`0--19`, `20--49`, `50+`), opponent, recurrent
  seat, trajectory source, and history length;
- both greedy legal action rows and action types;
- raw-softmax total variation and whether the greedy action differs;
- the action-type transition on disagreement states.

Summarize decision- and game-balanced disagreement rates and mean TV overall,
by opponent, seat, phase, trajectory source, and paired outcome-delta sign.
Report action-type transitions specifically for `016` and clean `023`.
These are descriptive localizations; no trajectory is relabeled as a training
target and no causal value is inferred from a win or loss.

## Interpretation and decisions

Use score-share as the primary strength signal and margin as a guardrail.
Report the following nonexclusive flags:

- `aggregate_score_uncertain`: aggregate score-share CI95 includes zero;
- `aggregate_score_systematically_negative`: its upper bound is below zero;
- `aggregate_margin_systematically_negative`: margin CI95 upper bound below
  zero;
- `016_loss_systematic` and `023_loss_systematic`: respective score-share
  CI95 upper bound below zero;
- `losses_localized`: either loss opponent has a systematic negative interval
  while the aggregate interval includes zero;
- `policy_movement_opponent_concentrated`: the maximum opponent
  game-balanced disagreement rate is at least twice the minimum, with at least
  `0.02` absolute separation;
- `policy_movement_phase_concentrated`: the analogous phase condition.

Formal decision priority:

1. Integrity failure:
   `recurrent_broad_diagnosis_invalid_replay_mismatch`.
2. Aggregate score or margin systematically negative:
   `recurrent_broad_regression_systematic_change_objective`.
3. Aggregate uncertain but `016` or `023` systematically negative:
   `recurrent_broad_regression_localized_change_objective`.
4. Aggregate score interval includes zero without a systematic opponent loss:
   `recurrent_broad_result_inconclusive_authorize_confirmation_preregistration`.
5. Otherwise:
   `recurrent_broad_diagnosis_complete_no_scale_authorized`.

Even an inconclusive result authorizes only design of a larger, fresh,
separately preregistered confirmation gate. It does not retroactively pass
`064`, authorize training, Stage C scale, promotion, or GPU rental.
