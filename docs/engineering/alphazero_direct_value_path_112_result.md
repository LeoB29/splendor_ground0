# AlphaZero Direct Value-Path Duel 112 Result

Date completed: 2026-08-25

Experiment 112 completed normally in 2h49m. It held the complete high-dose
policy computation fixed and changed only the complete value computation,
including its checkpoint-specific trunk, on 200 fresh paired seeds.

The full high-dose candidate versus the high-dose-policy/gen11-value baseline
finished 194-7-199: 49.375% score share, mean margin -0.14, seat shares 54.50%
and 44.25%, and paired CI95 44.625%--54.125%. All 400 games terminated
normally. The candidate failed score-share, margin, and second-seat checks.

Decision:
`high_dose_value_path_causal_gameplay_not_confirmed_no_architecture_authorization_no_install`.

The large direction seen retrospectively inside 111 did not reproduce on fresh
seeds. It therefore does not justify separating policy and value networks.
Nothing was trained, installed, or promoted; exact generation 2 remained
installed.

Raw result SHA-256:
`bb42a38826990f94ecedd75b6f05649ac7461183a56bc1c6871502c33ed13251`.
