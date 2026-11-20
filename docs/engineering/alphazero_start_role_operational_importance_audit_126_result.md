# AlphaZero Start-Role Operational Importance Audit 126 Result

Date completed: 2026-08-28

## Outcome

Experiment 126 completed normally without training. Start role was not
operationally material across evenly sampled reachable positions, including
the combined final two octiles, at the operative 256-simulation gameplay
budget.

Decision:
`start_role_not_operationally_material_close_representation_branch`.

This does not mean that start role never matters. It means the effect is too
rare and too concentrated at the very end of games to justify continuing the
general start-role representation programme.

## Reachable panel

All 32 untouched games and all 1,956 replay rows reconstructed exactly from
their initial seeds and legal recorded actions. Eight octile-midpoint states
were selected from every game, giving 256 reachable original positions spanning
normal play. None was already in `pending_round_end`.

Each original was paired with a controlled copy changing only `start_player`.
The copy preserved the v1 entity tensor, raw generation-11 prediction, and
semantic legal action set while changing the public-belief identity. Both
sides ran exact coupled searches at 64 and 256 simulations. Every integrity
check passed.

## Operative 256-simulation result

Across all 256 positions:

- mean visit-distribution TV was 0.006836;
- the preferred action changed in 1.1719% of positions;
- mean absolute search-root-value change was 0.005207;
- 2.3438% of positions had any frozen material event;
- the game-cluster CI95 lower bounds were 0.001709 for mean TV and 0.000400
  for mean absolute value change.

These are far below the frozen overall requirements of mean TV 0.05, preferred
action changes 10%, and mean absolute value change 0.05.

The combined final two octiles were larger but still not material:

- mean TV 0.027344;
- preferred action changes 4.6875%;
- mean absolute value change 0.020830;
- any material event in 9.375% of positions;
- CI95 lower bounds 0.006714 for TV and 0.001632 for value.

All six material-event positions at this budget occurred in octile 8. There
were none in octiles 1--7. In octile 8 alone, mean TV was 0.054688, preferred
actions changed 9.375%, mean absolute value change was 0.041660, and 18.75% of
positions had an event. This narrow 32-position descriptive stratum nearly
reaches some thresholds, but it was not the frozen decision stratum and still
missed both preferred-action and value magnitude requirements.

## Lower-budget diagnostic

At 64 simulations the effect was smaller overall: mean TV 0.003540, preferred
action changes 0.3906%, mean absolute value change 0.003900, and any event in
1.5625% of positions. Again, every event occurred in octile 8.

The increase from 64 to 256 simulations suggests that deeper search discovers
the end-of-round distinction as it approaches relevance. It does not show a
broad weakness that a role-aware network must repair.

## Interpretation

Experiment 117 was correct: start role can strongly affect carefully selected
states immediately before the end-of-round rule becomes decisive. Experiment
126 adds the missing prevalence evidence: that effect is absent through the
first seven octiles of these normal games and remains below the frozen
materiality gate over the final quarter.

Therefore, do not run the proposed experiment-125 role-magnitude audit, amplify
the role channel, train another start-role architecture, lower the gates, or
promote any role-aware candidate. Close the general start-role representation
branch. If a future independent failure is isolated specifically to terminal
round handling, prefer a direct rule-derived endgame feature or exact search
handling over reopening a general role-learning programme.

Return the main research programme to broader gameplay and multi-generation
learning bottlenecks rather than start-role encoding.

## Guard behavior

- No optimizer was instantiated and no parameter changed.
- No training target, candidate checkpoint, self-play, or gameplay arena was
  generated.
- The retained installed checkpoint remained unchanged.
- No installation, promotion, external `035` comparison, or automatic
  continuation ran.
- Focused 117, 125, and 126 tests pass 11/11. The full repository suite passes
  828/828 in 230.47 seconds.

## Artifacts

- Frozen protocol SHA-256:
  `27d53e38c8dee0665da5eac95bf358a10c94900bf827d2b9fa4fe1ff0e5520c2`.
- Raw result SHA-256:
  `f2043a8b86925081dca058b0643c1906ce9ac69fcda48c181364713e0b71136e`.
