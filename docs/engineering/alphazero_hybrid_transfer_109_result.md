# AlphaZero Hybrid Transfer 109 Result

Date completed: 2026-08-21

Experiment 109 tested the last replay-consolidation hypothesis: full 107 policy
with exact gen11 value. The raw full-policy control finished 102-1-97 (51.25%,
margin -0.01, seats 64.5%/38%, paired CI95 `[44.5%,58%]`). The searched hybrid
finished 102-2-96 (51.5%, margin +0.34, seats 51%/52%, paired CI95
`[44.5%,58.5%]`). All 400 games completed normally in 5 hours 32 minutes with
empty stderr.

The hybrid was balanced and directionally positive but practically flat. It
missed the predeclared 55% gate. Relative to lambda-0.25 in 108, its searched
score improved only +1.25 percentage points with an independent CI95 spanning
`[-8.25,+10.75]`; raw improved +2 points with CI95 `[-6.25,+10]`.

Final decision:
`hybrid_internal_transfer_failed_close_consolidation_branch_no_install`.
The replay-consolidation/interpolation branch is closed. No checkpoint was
trained or installed and no `035` game occurred. Raw result SHA-256:
`ba4318190b84775a097fbc3a656b823447f05edd45618fde5be6800e3d0e81a1`.

The next substantive direction is a controlled higher dose of fresh gen11
self-play, not another replay-loss, interpolation, or hybrid variant.
