# AlphaZero High-Budget Distillation Pilot 114 Result

Date completed: 2026-08-26

Experiment 114 completed normally in 12h43m. All 256 high-budget self-play
games terminated normally, yielding 15,916 positions. The exact whole-game
partition contained 224 training games and 32 untouched holdout games.

Parent-safe training selected epoch 1 after four epochs. On the untouched
holdout, candidate-minus-parent policy loss was -0.036725, value loss was
-0.010119, and weighted objective was -0.041785. Policy target-top agreement
rose from 62.47% to 64.67%. Every offline qualification check passed.

At 256 simulations, the candidate versus exact gen11 finished 207-3-190 over
400 games: 52.125%, margin +0.2875, seats 57.0%/47.25%, paired CI95
47.5%--56.75%. The primary compression gate failed.

At 1,024 simulations, it finished 111-1-88 over 200 games: 55.75%, margin
+0.54, seats 57.0%/54.5%, paired CI95 49.0%--62.5%. This passed the frozen
practical diagnostic, but its interval still included 50% and it was not an
independent strength claim.

Decision:
`high_budget_gain_requires_expensive_search_review_representation_capacity_no_install`.

The candidate learned the stronger targets but did not establish compression
at the ordinary search budget. Its high-budget result merits a fresh powered
replication before either a continuation block or network rebuild. Nothing was
installed or promoted; exact generation 2 remained installed.

Candidate checkpoint SHA-256:
`93e548a35f2a42194c70602928fa1ae9398fe042dbad2b2c876694fa97a9c4bd`.

Raw result SHA-256:
`dd8bbaa7876b881db69d44fb4242624e1f227981b4628da392b698491c231ad5`.
