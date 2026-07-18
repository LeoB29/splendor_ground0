# CFR Strategic-Equivalence Diagnostic (`045`)

Frozen before implementing any best-response or action-equivalence analyzer and
before generating any `045` result. This is a CPU-only diagnostic of the two
failed raw-policy-shape gates in `044`. It performs no training, collects no
gameplay corpus, and cannot promote a gameplay checkpoint.

## Fixed inputs

- Use the unchanged reduced production-engine Splendor game and legal
  information-state history contract from `044`.
- Reconstruct exact alternating CFR from zero regrets for exactly 10,000
  iterations.
- Reconstruct external-sampling MCCFR from zero regrets for exactly 50,000
  iterations with seed `440044`.
- Require the canonical payload hashes to equal the hashes stored in the
  immutable `044` result:
  - exact: `b8184ec7c9dbfa3611d014f98c674efbeae27728429b2fa90c7939e1bbe043ab`;
  - external: `3c6644db39998800690a2adfa6efc8b64c7f69cd55be99d678ca2ea7ebadbf78`.
- Define the discrepancy set exactly as in the post-result `044` inspection:
  information sets shared by both solvers, with at least 100 external average-
  strategy observations and raw total variation strictly above `0.30`.

No `044` threshold is changed and `044` remains formally failed.

## Exact best-response contract

Implement an information-set-consistent best response, never a
perfect-information/determinization best response:

- group all full histories belonging to one acting-player information set;
- require one stable semantic legal-action set and one chosen action across
  every history in that information set;
- process information sets from the deepest player decision backward;
- for each action, aggregate continuation utility over histories using chance
  reach and the fixed opponent's behavioral reach, excluding the responding
  player's own reach as required for counterfactual best response;
- use deterministic semantic-action order to break exact ties;
- evaluate the resulting pure response from the chance root against the fixed
  opponent policy.

For a profile `(sigma0, sigma1)`, record profile value, each player's best-
response value, deviation incentives, NashConv (the sum of the two incentives),
and exploitability (`NashConv / 2`). Analyze both the exact average profile and
the external average profile. Run the analyzer twice and require identical
canonical results.

## Action-value and reach contract

For every discrepant information set:

- compute action continuation values with both players returning to the exact
  average policy after the candidate action;
- aggregate hidden histories with exact counterfactual chance/opponent reach;
- record the value range, exact-mixture value, external-mixture value, and the
  positive external-minus-exact loss from the acting player's perspective;
- record the probability of reaching the information set under both the exact
  and external profiles;
- use the larger of those two reach probabilities as the conservative strategic
  weight;
- call a discrepancy resolved when its conservative reach is below `0.0001` or
  its external-mixture loss is at most `0.02` utility.

Aggregate conservative reach-weighted positive loss separately by acting
player. Also record the two cross-play values: external player 0 versus exact
player 1, and exact player 0 versus external player 1.

## Frozen gates

All of the following are required:

1. Both reconstructed policy hashes match `044`.
2. The best-response/action-value analysis is byte-deterministic.
3. Every best response selects exactly one legal semantic action consistently
   at every information set, with no unstable action set or missing selection.
4. Exact-profile exploitability is at most `0.005`.
5. External-profile exploitability is at most `0.020`.
6. External exploitability exceeds exact exploitability by at most `0.015`.
7. Every raw-TV-above-`0.30` discrepancy is resolved by the fixed reach-or-loss
   rule above.
8. Conservative reach-weighted positive action-value loss is at most `0.002`
   for each acting player.
9. Each cross-play value differs from exact-profile self-play value by at most
   `0.02` utility.
10. All values are finite, both players' deviation incentives are nonnegative
    within `1e-12`, and the game remains zero-sum.

## Decision

A complete pass reclassifies the two failed `044` raw-TV gates as harmless
equilibrium-policy non-uniqueness for this reference game. It authorizes only a
separately preregistered full-game neural Deep CFR feasibility pilot. That pilot
must still establish memory/runtime feasibility, neural regret-fit quality,
held-out advantage ordering, and then gameplay strength before any promotion.

Any failed `045` gate keeps full-game neural CFR and GPU rental locked. A
failure must be diagnosed as either a best-response analyzer defect or an
external-MCCFR strategic defect; do not tune these thresholds after the result.
BGA/expert replay acquisition remains a separate future option and is not used
in `045`.
