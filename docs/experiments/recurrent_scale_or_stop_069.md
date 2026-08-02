# Recurrent Scale-or-Stop Learning Experiment (`069`)

Frozen after the valid but inconclusive `068` seat-0 confirmation and before
implementing, collecting, or training any scaled generation. This is a new
learning-feasibility question. It does not reinterpret `066` or `068` as a
pass, and it cannot directly promote a checkpoint.

## Question

Does approximately 10,000 fresh, balanced on-policy league games turn the
history-required recurrent policy's reproducible near-tie with epoch zero into
a meaningful broad improvement? If not, stop outcome-only recurrent learning
as the project's primary direction rather than tuning or collecting again.

## Frozen prerequisites

- `068` result:
  `outputs/recurrent_seat0_confirmation_068/result.json`
- `068` SHA-256:
  `d4d366e6e77de3af9dba78f51b2341442198ca3752dd464361b1888e7d81d9b1`
- `064` result SHA-256:
  `7a3094d37e6d37961b44f74029330c7083f5da3af7a9a03f8050b62bc59cf2e6`
- Archived `064` generation 1 SHA-256:
  `0c42c04df97114bf79147dfee80e8f296485b15742d6248c8f37ff7cee78756e`
- Base `064` generation 2 SHA-256:
  `3ed0bd7e2dcda21646c7ac4f68e2d0ffcfe0bdacf0c151c453c52ea6a08fe09e`
- Raw legal `035` SHA-256:
  `4422a82b593340eecd02af3a0941f24f0318cc7453e6a0739fd9f524539af2cd`
- `016` SHA-256:
  `876ac23a005d367ac48b3619bb7cdcb6e0230a3e9e97943d7eecdc6d07d1b056`
- Clean canonical `023` SHA-256:
  `29ef7c13c5553ec88c8bcb98f7378c925f0e600e2fcf3d944df5926efed7a070`

Require the exact `068` decision
`recurrent_seat0_confirmation_failed_no_scale`, all eleven `068` integrity
gates true, and exactly one failed `068` strength gate:
`aggregate_score_ci_proves_noninferiority`. This acknowledges the failed
gate rather than treating it as authorization inherited from `068`.

## Fixed scaled succession

Initialize from frozen `064` generation 2. Preserve the exact
`history_required_interaction` architecture, frozen embedded `035`, legal
public-history schema, behavior distribution, WDL/final-margin return,
opponent/seat-cell normalization, phase weighting, clipped four-epoch update,
optimizer, entropy coefficient, gradient cap, and validation-KL projection
recipe from `064`.

Run at most three conditional scaled generations. Every opponent-seat cell has
exactly 240 eligible games: the first 192 train and last 48 validate.

1. Scale generation 1 uses raw `035`, `016`, clean `023`, shallow search,
   archived `064` generation 1, and base `064` generation 2: 12 cells and
   2,880 games. Collection starts at seed `690000`; at most 3,600 attempts.
2. Scale generation 2 conditionally adds accepted scale generation 1: 14
   cells and 3,360 games. Collection starts at seed `700000`; at most 4,200
   attempts.
3. Scale generation 3 conditionally adds accepted scale generations 1 and 2:
   16 cells and 3,840 games. Collection starts at seed `710000`; at most 4,800
   attempts.

Total eligible collection is exactly 10,080 games. The three reserved attempt
bands are disjoint. Eligibility is outcome-blind: only `completed` and
`forced_pass_deadlock` enter either split. Use maximum turn 400, repetition
cutoff 4, and no-progress cutoff 60. Never replace or exclude a valid locked
game based on outcome.

Completed generations are written atomically. A process interruption may
resume only after verifying the protocol, prerequisite, parent, candidate,
and generation-record hashes; it may not recollect or retrain a completed
generation.

## Per-generation learning and integrity gates

Require every `064` repaired-generation contract, scaled to 192/48 games per
opponent-seat cell:

- exact collection, split, opponent, seat, seed/attempt range, and eligibility;
- zero illegal actions and no ineligible game in training or validation;
- behavior reconstruction error at most `1e-5`;
- finite metrics, nonzero first gradient, frozen embedded parent, trainable
  history/residual parameters, and exact checkpoint reload;
- validation immediate-parent raw KL in `[0.0001,0.0100]`;
- validation immediate-parent argmax movement in `[0.005,0.12]`;
- positive overall and combined mid/late validation surrogates;
- validation normalized entropy at least `0.30`;
- exact empty-history fallback to raw `035`;
- real-versus-within-stratum-rotated-history mean TV at least `0.002`.

After every offline pass, play 100 untouched paired-seat games against the
immediate parent using 50 seeds: `720000--720049`, `721000--721049`, and
`722000--722049`. Require all endings valid, candidate score share at least
`0.45`, and mean paired margin at least `-1.5`. Stop before constructing the
next generation on any failure.

## Fixed final broad gate

Only after all three generations pass, compare scale generation 3 with frozen
epoch-zero `035` on 100 fresh matched seeds per static opponent, both recurrent
seats:

- raw `035`: `730000--730099`;
- `016`: `731000--731099`;
- clean `023`: `732000--732099`;
- shallow search: `733000--733099`.

This is 400 opponent/seed clusters and 1,600 games. Use an
opponent-stratified paired-cluster bootstrap with 100,000 draws and seed
`690069` for score and margin intervals. Report opponent, seat, alternating
seed-half, and leave-one-opponent-out breakdowns. Never treat individual
games, turns, or actions as independent observations.

Require all:

1. exact panel and all valid endings;
2. aggregate candidate-minus-epoch-zero score delta at least `+0.025`;
3. aggregate score CI95 lower bound at least `-0.025`;
4. aggregate paired margin delta strictly positive;
5. aggregate margin CI95 lower bound at least `-0.50`;
6. at least three of four opponent score deltas nonnegative;
7. no opponent score delta below `-0.05` and no opponent CI upper bound below
   zero;
8. both recurrent-seat score deltas at least `-0.025`;
9. both alternating seed halves at least `-0.025`;
10. on final validation states, scale generation 3 versus the base generation
    2 moves at least 1% and at most 25% of greedy actions, with raw KL at most
    `0.04`; versus epoch zero, movement is at most 30% and raw KL at most
    `0.05`.

These are learning-signal gates, not promotion gates. A passing model still
requires a separately preregistered untouched promotion confirmation.

## Decisions

- Prerequisite or evidence-integrity failure:
  `recurrent_scale_or_stop_invalid_no_learning_decision`.
- Any generation learning/parent-retention failure:
  `recurrent_scale_learning_contract_failed_stop_outcome_only_branch`.
- Three generations complete but the final broad conjunction fails without a
  systematic aggregate regression:
  `recurrent_scale_remains_neutral_stop_outcome_only_branch`.
- Aggregate score CI upper below `-0.025` or margin CI upper below `-0.50`:
  `recurrent_scale_systematic_regression_stop_outcome_only_branch`.
- Every gate passes:
  `recurrent_scale_learning_signal_authorize_confirmation_preregistration`.

Do not adapt generation counts, sample sizes, seeds, opponents, thresholds, or
the update after observing partial results. Do not promote any `069` model,
declare `068` passed, or rent a GPU from this protocol alone.
