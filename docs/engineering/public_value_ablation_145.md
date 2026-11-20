# Experiment 145: public versus neutral value in the same search

2026-09-08. The owner authorized the proposed 800-game comparison with
"ok do i t". This authorizes this arena and its necessary engineering/audit,
not training, promotion, an extension or a successor. Preserve all dirty work
and all earlier frozen code/data. No change to the 084/035 champions.

## Question and frozen treatment

Does the frozen 143 public score/resource evaluator add playing strength to
035-prior real PUCT under a .4-second decision allowance?

Public arm: exact Experiment 144 evaluator and search. Neutral arm: identical
inference and search work, but replace the scalar returned by leaf inference
with 0. Both arms compute the original public prediction before this switch,
so disabling the evaluator does not save its arithmetic. Neither changes any
policy logits, legal-attention context, canonical grouping, search exploration,
terminal recognition, backup sign, tree key or information sampling.
Completed games and two-sole-PASS deadlocks retain exact outcome values in
the existing search, before calling leaf inference. Zero means neutral only
for unfinished positions, including depth-limited network evaluations.

Both share frozen 035 checkpoint SHA256
4422a82b593340eecd02af3a0941f24f0318cc7453e6a0739fd9f524539af2cd
and 143 simple.json SHA256
a5c87ab3fb0f3f22a0e0e25a5adbfcad1bb3422c7f40cdc060d33a0d3a9b720c.
Same selected 15 public actor-minus-opponent coefficients, zero intercept,
float32 feature rounding then float64 dot product, clipping [-1,1]. No refit.
Use full concrete legal-action attention; priors are the maximum 035 logit
in each canonical successor class, softmaxed without any root action cap.

Search: BudgetSearch real mode, c_puct 1.5, max_depth 96, fresh tree each move,
no noise or move temperature, no self-play. .4 seconds includes root inference
and expansion; caller legal enumeration and trace logging are outside timing
for both arms. Complete at least one simulation and start none at/after deadline.
One simulation plus result assembly may overrun. The config's simulations=64
is not a timed ceiling. Eight CPU worker processes, one Torch thread each.
Matched time does not imply identical simulation counts, depths or visited
states; those may change because of the value treatment and are recorded.

## Schedule, estimand and precision

800 games, 400 starting seeds 1450000000--1450000399. Index 2k uses public
in seat 0; index 2k+1 uses public in seat 1. Exact same initial environment
seed within each pair. Search RNG is treatment independent:
game_seed*10000 + player_seat*1000 + turn + 145013.
No opening randomness. No prior experiment's game outcomes enter selection.
No repeated 144 game, replacement, optional stopping or sample extension.

Primary: public score share, win 1, draw .5, loss 0. Resample whole seed pairs
50,000 times, NumPy default_rng(145999), percentile 95% interval. Pair means
are the independent statistical units. The larger sample aims for greater
precision than 144, but the different opponent means its variance is unknown
and 800 games cannot guarantee a conclusive result.

Success requires CI lower > .5 and public point share >= .55. This establishes
a positive effect with a practical point estimate, not a 95% lower bound of
five percentage points. Label `material_public_value_benefit`.
If upper < .5, label `public_value_worse_than_neutral`.
If lower > .5 but point < .55, label
`positive_value_benefit_below_material_threshold`.
Otherwise label `value_benefit_not_established`.
Separately flag a valid upper CI < .55 as excluding the practical target.
Failure to show a gain is not equivalence or proof of no useful effect.
Seat shares, score margins, timing, CPU consumption, simulations and maximum
depth are descriptive and cannot supply alternative passing criteria.

## Integrity and time gates

Retain every game. Normal completion and existing two-sole-PASS deadlock are
valid, with separate counts. Repetition limit 4, no-progress limit 60, maximum
turns 400. Any arbitrary cutoff/stall has no valid winner and receives .5 only
in an operational summary. Preserve the diagnostic adjudication separately.
Any such game blocks a scientific strength conclusion; do not replace it.

For EACH arm require mean decision seconds <= .44 and p95 <= .50, and every
search must run at least its .4-second allowance. Require ratio of public to
neutral mean decision times in [.95,1.05]. Record all overshoot and CPU/wall
costs. A failure yields `invalid_integrity_or_time_match`, before interpreting
win rates. Any source/model identity, trajectory/root or statistics audit
failure likewise prevents a valid scientific conclusion. Never tune the
allowance/gates after protected outcomes exist.

## Preparation and verification

Tests cover the scalar-only switch with equal inference calls and exact priors,
terminal recognition/backup signs, information legality, timed/fixed search
equivalence for both arms, pairing, cutoff/budget/time-match gates, treatment
identity and full JSON serialization. Run the entire repository suite before
launch. Two full off-schedule probes use seed 1459000000 with swapped seats
and the actual .4-second allowance. They are excluded from protected outcomes.

Freeze protocol/config/all source and wrapper/test/verifier/input hashes before
the first protected game. Successful preflight must bind to that exact identity.
Only progress and integrity may be inspected during collection. Reveal outcomes
after all 800 games complete; no outcome-dependent changes or early conclusions.

A separate verifier reconstructs every legal trajectory and state hash, both
arms' full root canonical sets, original 035 priors, correct public/neutral
root values, visit accounting, selected moves and timing. Reproduce the first
and last decision of EACH arm in every game with saved search RNG and completed
simulation counts: normally 3,200 fixed-count search reproductions. Every root
gets structural/prior/value checks; middle trees are not fully regenerated.
Reconstruct final states/outcomes and independently recompute paired statistics,
check game hashes and every current/prior frozen dependency. This shares the
repository game engine and PUCT implementation, so is not an independent
implementation of the rules or a replay of OS wall-clock scheduling.

## Scope of conclusions

This tests the evaluator's contribution to this search under matched decision
allowances. It does not directly test raw 035, champion 084, greater depth,
different evaluators or a learning cycle. A head-to-head result versus neutral
cannot be combined transitively with 144 into a claimed win rate against 035.
No automatic policy training, promotion or further experiment follows.
