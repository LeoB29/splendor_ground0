# AlphaZero Identity-Preserving Medium-Capacity Pilot 116 Result

Date completed: 2026-08-27

Experiment 116 completed normally in 3h04m. The frozen exact-gen11 parent plus
zero-output residual tower passed the epoch-zero identity audit with exact
logit and value equality. Total capacity was 991,828 parameters, of which the
635,378-parameter residual tower was trainable and the 356,450-parameter parent
remained frozen.

Parent-safe training selected epoch 3. On the untouched 32-game holdout,
candidate-minus-parent policy loss was -0.031705, value loss was -0.001153,
and weighted objective was -0.032281. Every frozen offline gate passed.

At 256 simulations the candidate versus exact gen11 finished 205-3-192 over
400 games: 51.625% score share, margin +0.3725, seats 58.75%/44.5%, paired CI95
46.875%--56.25%, and 400/400 normal terminations. The gameplay gate failed
score share, second-seat, and confidence-bound checks.

Decision:
`identity_capacity_offline_gain_not_converted_review_representation_design_no_install`.

Added capacity again improved teacher-target fit without establishing gameplay
strength. Do not widen further or generate another expensive corpus before a
targeted representation audit. Nothing was installed or promoted; exact
generation 2 remained installed.

Candidate checkpoint SHA-256:
`bb3bf65b102e5ec4c46a351c3958dba250c0f4c94f240847b7c3bc5871a57565`.

Raw result SHA-256:
`5c50ca7989381ad3c55b32f42e97a9ab67b4992085c53e1578227d932bde5718`.
