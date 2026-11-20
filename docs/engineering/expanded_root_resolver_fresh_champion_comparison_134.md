# Expanded-Root Resolver Fresh Champion Comparison 134

Date frozen: 2026-08-31

## Question

Does the exact expanded-root public-belief resolver validated offline in
experiment `086` produce a material, reproducible full-game improvement over
the exact four-root `050`/`051` research champion confirmed in experiment
`084`?

This is the oracle-calibration step selected by the mechanism review after
experiment `133`. It performs no training and changes neither resolver.

## Frozen prerequisites

- `084` result SHA-256:
  `1ed98024072b92baf3c30bc896b9dae2484f7d80b727959307dfd83bbfe2cd0f`.
- `086` result SHA-256:
  `8ae985f6747b5f3bbbcf4997357daf299ef0f46cc23534d55b6de91367910eb5`.
- `133` result SHA-256:
  `7f74736b4557a1b01b3e6febbf6a4f4cec2fa7643a78655c0f4adc529dde872c`.
- Mechanism-review SHA-256:
  `2ba4e5c60788b080d517ae8f783186efec1ee9a55f348fb8cb047a28efaa80c2`.
- Legal-action scorer `035` SHA-256:
  `4422a82b593340eecd02af3a0941f24f0318cc7453e6a0739fd9f524539af2cd`.
- Reused exact `087` resolver implementation SHA-256:
  `bcc63bcb9b753e0387aa57115facaea8c9eab13f28ade26ae9f7b80b838f2a28`.

The incomplete owner-waived `087` strength progress is outcome-sealed. The
runner must not read, summarize, complete, or pool it. Only the already public
formal `087` smoke failure is used to correct the operational gate.

## Exact agents

Candidate: the unchanged `086` expanded resolver. At each eligible decision it
retains the champion's top four roots plus every individually stable
non-`RESERVE_DECK` public root. It evaluates three observer-legal belief
particles, four shared opponent responses, and four stochastic continuations
per leaf, then applies the exact shared information-set minimum.

Comparator: the unchanged exact `050`/`051` four-root public-belief resolver
confirmed by `084`, using the same three particles, four opponent responses,
four continuations, continuation policy, utility, and terminal handling.

Both use frozen raw legal `035` scores for ranking and continuation play:
epsilon `0.1`, temperature `0.5`, top-k `3`, score-margin scale `20`, maximum
400 continuation turns, and loop fallback limit `64`.

## Fresh namespaces and execution

- deterministic smoke engine seed: `1,340,000`;
- strength engine seeds: `1,340,100--1,340,299`;
- belief root-index start: `1,340,000,000`;
- continuation seed start: `1,340,000,000,000,000,000`;
- bootstrap seed: `1,340,100`, 20,000 paired bootstrap samples;
- ten CPU pair workers, one Torch thread and serial leaf work per game;
- maximum game turns `400`, repetition limit `4`, no-progress limit `60`;
- strength active-compute ceiling: 72 hours.

The smoke runs one paired seed and an independent from-zero repeat concurrently.
The strength block contains exactly 200 paired seeds / 400 games, with the
candidate playing both seats for every seed. The paired seed is the independent
unit.

Progress may contain completed pairs solely for identity-matched crash or
user-requested pause recovery. Console output reports counts and ETA only, not
interim outcomes. A stopped run may resume the same identity and only the same
unfinished seeds. It may not replace seeds, add games, or change worker count.
An interrupted in-flight pair may be recomputed; completed pairs are retained.

## Corrected smoke and operational contract

Experiment `087` incorrectly required zero rollout fallback triggers while
also allowing bounded loop adjudication. Its champion smoke produced exactly
two valid loop-adjudicated continuations and therefore failed despite being
inside the established `084` limit.

Experiment `134` removes only that contradictory zero-trigger gate. It still
requires:

1. exact frozen provenance, configuration, candidate, and comparator contracts;
2. a byte-identical from-zero smoke repeat, four clean smoke games, both
   resolvers exercised, exact branch accounting, all continuations resolved,
   and zero hard caps;
3. exactly 400 clean strength games, with no timeout, stall, illegal action,
   invalid cutoff, or unexpected resolver failure;
4. candidate continuation count equal to the sum of
   `retained_roots * 3 * 4 * 4` over completed searches;
5. champion continuation count equal to `192` per completed search;
6. all continuations resolved, zero hard caps, and loop-adjudicated fractions
   at most `0.1%` for each resolver and jointly;
7. the strength block within 72 active wall hours.

Fallback-trigger counts remain mandatory telemetry. They are not separately
zero-gated because a valid loop-adjudicated continuation necessarily reaches
64 triggers under the frozen rollout contract.

## Frozen strength gates

All are conjunctive after operational validity:

1. optional expanded roots are retained and expansion changes the champion-
   prefix decision at least once in early (`0--19`), middle (`20--49`), and
   late (`50+`) play;
2. expansion switch rate is at least `15%` of completed candidate searches;
3. paired candidate score share is at least `55%`;
4. paired-bootstrap score-share CI95 lower bound is strictly above `50%`;
5. mean paired candidate final-score margin is strictly positive;
6. candidate score share from each seat is at least `50%`.

The expected alternative remains `56.5%`. With 400 games the test has moderate,
not high, power for the 55% point gate. There is no optional extension.

## Decisions

- Smoke or operational failure:
  `expanded_root_gameplay_134_invalid_no_strength_decision`.
- Valid strength block failing any causal gate:
  `retain_050_051_research_champion_close_exact_expansion_after_fresh_134`.
- Every gate passes:
  `promote_expanded_root_public_belief_resolver_134_as_research_champion`.

No outcome authorizes training, corpus generation, installation as the fast
agent, threshold relaxation, extra games, or an automatic successor. On pass,
raw legal `035` remains the fast champion and the grouped resolver becomes the
search frontier for a separately frozen efficiency programme. On valid failure,
the next planner must add horizon while retaining information-set grouping.

## Expected compute

The known `087` engineering estimate is 40--60 wall hours on ten CPU workers;
the active-compute ceiling is 72 hours. The machine remains available for light
work but CPU-heavy gaming may suffer. A user-requested pause must preserve all
completed pairs and the frozen identity.
