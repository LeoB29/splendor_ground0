# AlphaZero Historical-State Reanalysis Pilot 129 Result

Date completed: 2026-08-29

Experiment 129 completed normally in 3h31m. It collected all 256 frozen fresh
historical-state continuations, trained one parent-safe candidate, passed the
ordinary offline gate, and stopped at the historical-start gameplay gate.
Nothing was installed or promoted, and the conditional ordinary initial-state
arena did not open.

## Collection and training

Exact generation 11 completed all 256 fresh 1,024-simulation continuations:
64 each from generation-2, -5, -8, and -11 source states, with no abnormal
attempt or replacement. The sealed corpus contains 8,165 fresh positions. No
historical action, policy target, value target, or outcome entered the replay.

The matched mixed replay contains 13,931 rows, only 29 fewer than the 13,960-row
experiment-114 control. It combines the 8,165 historical-continuation rows with
5,766 ordinary generation-11 rows from 93 whole games.

Parent-safe training selected epoch 1 and stopped after four epochs. On its
whole-game validation split, candidate-minus-generation-11 policy loss was
-0.032329, value loss was -0.003085, and weighted objective was -0.033872.
Mean raw-policy total variation was 0.07581 and 10.84% of top actions changed.

## Ordinary holdout

The candidate passed every frozen check on the sealed 32-game / 1,956-row
ordinary holdout:

| Model | Policy CE | Terminal MSE | Combined objective |
|---|---:|---:|---:|
| generation 11 | 1.77663 | 0.91004 | 2.23165 |
| experiment-114 terminal control | **1.73991** | **0.89992** | **2.18987** |
| experiment-129 historical candidate | 1.74149 | 0.90891 | 2.19594 |

The historical candidate improved generation 11 and remained within every
allowed tolerance of the stronger experiment-114 control. This opened the
historical-start gameplay panel.

## Historical-start gameplay

Against the exact experiment-114 terminal control, the candidate finished
67-2-59 over 64 paired states / 128 games: 53.125% score share, +0.578 mean
margin, seat shares 55.47% and 50.78%, and paired CI95 46.09%--60.94%. Every
game completed normally.

The positive point estimate and both non-losing seats are suggestive, but the
candidate missed the preregistered 55% score-share threshold and the strict
confidence threshold. The conditional 400-game ordinary initial-state arena
therefore remained closed.

Exploratorily, score share was 50.0% on early states, 60.0% on middle states,
and 50.0% on late states. By source generation it was 53.125%, 56.25%,
53.125%, and 50.0% for generations 2, 5, 8, and 11. These opened subgroups are
hypothesis-generating only and do not override the frozen overall failure.

Decision:
`historical_reanalysis_failed_diverse_state_gameplay_no_install`.

The result does not prove that population-anchored state diversity has zero
effect; its gameplay estimate was mildly positive. It does show that this
matched-size intervention did not establish a reliable gain, including no
point gain on the most current generation-11 source states. Do not run the
skipped ordinary arena, repeat the same candidate, or train a post-hoc
middle-only variant. Together with experiments 110--128, this substantially
closes unchanged compact policy/value distillation and motivates a bounded
audit of whether scalar terminal-WDL values discard important outcome
uncertainty before any further training.

Candidate checkpoint SHA-256:
`099a1ee88a094b99fc0946e7af1b25c0396319e94b5cecb907794ec976742c5f`.

Historical replay SHA-256:
`459938b1925524f3a8ed305d69ef4bbcb113c6e08dfe22323385c359b4d3b0a8`.

Raw result SHA-256:
`75cf5857ae833ba1c5609002b6d93f76660fadd3be20d2ecc98c30146c20f4fa`.
