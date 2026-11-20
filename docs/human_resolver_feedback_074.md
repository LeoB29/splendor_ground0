# Owner Feedback After Human--Resolver Calibration `074`

Recorded 2026-08-03 after all ten games and after the preregistered primary
result had been computed. These are qualitative expert observations and post-hoc
diagnostics. They do not alter `074`'s frozen inconclusive decision and are not
training or selection data.

## Owner observations

1. Although the 7-3 record was close to the previous 8-2 against `035`, the
   resolver games felt substantially closer. At least one game tied on prestige
   points and was decided by the development-card tiebreak. The binary 70% win
   rate may therefore overstate the human's relative strength.
2. The resolver reserved blindly from decks very often. The owner, while not
   claiming perfect play, considers that surprising and believes the action is
   poor in most configurations.
3. The owner recalls a virtually useless resolver move near turn 56, probably in
   seed `800006`: buying a tier-1 ruby card that served no purpose. They believe
   another action probably would have let the resolver win on its next turn.
4. The owner did not inspect every resolver move closely and may have missed
   other errors.

## Checks against the recorded data

- The closeness observation is supported descriptively. Against `035`, human
  margins were `[11, 11, -3, -8, 5, 9, 13, 5, 4, 9]`, mean `+5.6`, median `+7`,
  with eight positive margins. Against the resolver they were
  `[-5, 4, 8, -5, 5, 9, -1, 0, 0, 13]`, mean `+2.8`, median `+2`: five positive,
  two tied, and three negative. Treating equal prestige as half rather than
  applying the official card-count tiebreak gives a descriptive point-share of
  60%, versus the preregistered rule-correct game score share of 70%.
- The deck-reserve observation is also supported. The resolver made 27 deck
  reserves in 278 decisions (`9.71%`), 24 during turns 0--19. The human made two
  in 278 decisions (`0.72%`), one during the opening. This is descriptive class
  frequency, not proof that all 27 actions were mistakes.
- The seed clarification does not line up literally with the replay. In seed
  `800006` the resolver won 16--15, and its late purchases were green tier 1,
  black tier 2, blue tier 1, and red tier 2; its only tier-1 red purchase was much
  earlier, at replay turn 21. The closest recorded match is seed `800007`, replay
  turn 58: the resolver bought the free zero-point red/ruby card `T1_56` while
  trailing 12--13. A free one-point tier-2 card, `T2_46`, was also legal and was
  the raw `035` policy's first-ranked action. The game ended 15--15, with the
  resolver losing the fewest-development-cards tiebreak 19 cards to 7.
- Exact re-execution of seed `800007`, resolver decision 29, reproduced `T1_56`.
  The four resolved root values were `-0.11666666666666667` for `T2_46` and
  `-0.11666666666666665` for each of three free zero-point tier-1 red cards. The
  difference is only `1.39e-17`: numerical dust defeated the resolver's intended
  preference for the earlier raw-policy action on a value tie.
- A corpus-wide post-hoc check found 41 such switches with chosen-minus-parent
  value gap at most `1e-12`, out of 6,625 resolved decisions (`0.619%`) and 4,498
  switches (`0.912%`). This exact numerical pathology is real but rare.
- More importantly, the terminal rollout utility uses normalized prestige-score
  margin only. It ignores `state.winner` and therefore the official
  fewest-development-cards tiebreak. Consequently a 15--15 tiebreak loss and a
  true draw both have rollout value zero. That objective mismatch is directly
  relevant to buying a needless extra card in the eventual 15--15 loss.
- It is not valid to claim that `T2_46` would certainly have won. Buying it changes
  which market card is replenished, so the recorded human reply is no longer
  legal in the counterfactual trajectory. The move is demonstrably dominated in
  immediate points and selected for the wrong numerical reason, but the causal
  game outcome remains unproved.

## Research use

The observations reinforce three cautions: binary win rate is coarse at ten
games, the resolver's strategy-fusion-like deck-reserve rate is behaviorally
unlike the human's play, and score-margin rollouts are blind to the official
tiebreak. None changes the formal `074` result. A tolerance fix would be a small
correctness repair, not a plausible programme-level breakthrough at its observed
`0.619%` decision dose. Before altering the frozen resolver objective, quantify
how frequently its terminal branches tie on prestige and how often a
tiebreak-aware value changes the selected root. Do not tune from this held-out
human game or retroactively rescore the panel.
