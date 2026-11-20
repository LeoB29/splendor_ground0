# AlphaZero Deep-Search Teacher-Refresh Continuation 127 Result

Date completed: 2026-08-29

Experiment 127 completed normally in 14h14m. It generated two fresh
256-game, 1,024-simulation teacher corpora and reached the frozen generation-14
endpoint. Nothing was installed or promoted, and no automatic continuation
ran.

## Teacher refresh and training

The generation-12 actor produced 256 clean games and 15,912 positions. One
original game ended in `forced_pass_deadlock`; the frozen outcome-blind rule
excluded it and admitted the next predetermined clean replacement. Parent-safe
generation-13 training selected epoch zero, so generation 13 was exactly the
generation-12 parent.

The generation-13 actor then produced 256 clean games and 15,798 positions
with no abnormal ending. Generation-14 training selected epoch 1. On its
whole-game validation split, candidate-minus-parent policy loss was
-0.013236, value loss was -0.014906, and weighted objective was -0.020689.
Mean raw-policy total variation was 0.054192 and 7.727% of top actions changed.

Thus the second refresh produced a valid offline improvement even though the
first refresh found no parent-safe update.

## Powered endpoint

At the ordinary 256-simulation budget, searched generation 14 versus searched
generation 11 finished 186-2-212 over 200 paired seeds / 400 games: 46.75%
score share, -0.6075 mean margin, seat shares 56.5% and 37.0%, and paired CI95
42.0%--51.5%. All games completed normally.

The endpoint failed score share, margin, second-seat, and confidence-bound
checks. Offline target improvement therefore did not convert into cumulative
gameplay strength.

Decision:
`deep_search_refresh_endpoint_failed_close_current_distillation_recipe_no_install`.

Do not add another unchanged teacher-refresh generation, enlarge the same
compact recipe, consult `035`, install generation 14, or promote it. The
stronger 1,024-simulation teacher established by experiment 113 remains valid;
the closed component is the present terminal-outcome distillation interface.

Generation-13 checkpoint SHA-256:
`0bf74520a16f8dc413ddf72f67ecbbfb911231e4b826f2abfad0c4e9c695f1fd`.

Generation-14 checkpoint SHA-256:
`42a7d6e8289b668328c68bf77df13f4e39fcc7463f0b232570cfe102d095fc1c`.

Raw result SHA-256:
`ba7b76de3b201a2222d43e59c997d8b4ae8d743ff3bc59f1a0fd9f995fcda44d`.
