# AlphaZero Search-Teacher Headroom 113 Result

Date completed: 2026-08-25

Experiment 113 completed normally in 4h39m. With the exact generation-11
checkpoint and every search setting except compute held fixed, 1,024-simulation
search defeated 256-simulation search 229-1-170 over 200 fresh paired seeds.

The candidate achieved 57.375% score share, mean margin +0.965, seat shares
62.00% and 52.75%, and paired CI95 52.50%--62.25%. All 400 games terminated
normally. Every frozen gate passed, including the strict confidence-interval
gate.

Decision:
`deeper_search_teacher_headroom_confirmed_review_high_budget_distillation_no_install`.

This establishes that the current network can produce a materially stronger
self-play teacher through additional search compute. It does not establish that
the network can absorb that teacher. The next bounded test is one high-budget
self-play/distillation generation before any larger-network rebuild. Nothing
was trained, installed, or promoted; exact generation 2 remained installed.

Raw result SHA-256:
`aea9e5a69fa91669ff3ee05d66d9d912b4d5920c90ca2944f5de7003dab18f8d`.
