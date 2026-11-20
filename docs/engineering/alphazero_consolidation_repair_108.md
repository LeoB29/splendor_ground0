# AlphaZero Consolidation Repair Audit 108

Date frozen: 2026-08-20

## Purpose

Experiment 107's consolidated update improved held-out policy loss, especially
on old gen7/gen8 replay, but worsened held-out value MSE enough to fail every
combined-objective retention check. Experiment 108 tests two narrow explanations
without generating replay or fitting another model:

1. the coherent gen11-to-107 update was useful but too large;
2. the consolidated policy is useful, while its value update is the blocker.

## Frozen inputs

- Completed 107 result SHA-256:
  `d0bba71d37eb6883096f7c532f5abd25cd0bbcf5623c9fc10980dbb24a52e157`.
- Exact gen11 checkpoint SHA-256:
  `c97e7e06910639c0edf61fdd878fd665e390298790830e0e792061d9107354d0`.
- Exact 107 consolidation checkpoint SHA-256:
  `c21c0ed3765e09820eb415c7369d0be0733913915e29d288db825cbcf9461c89`.
- Clean replay SHA-256 values:
  - gen7: `06ee156543da64de3126eed00cb363ed5f2d4c0e49e093550024fffc2ace5a1c`;
  - gen8: `96f0d346743f30bd29e329f4bf3e2a7a361e778391c6d6e5833ca2a06af9be10`;
  - gen9: `1b13ef038ccc6b3f695d7c7860a2284272ae18c8e75d44028e4cbb6f69914f0c`;
  - gen10: `09279d9593acfd944a32571716a6527c0be40e41055b6fa1f5bc25c88f43a60c`.
- Retained installed gen2 SHA-256:
  `60ca410b7d0d42c7beda66165df161cdc239a5e39a606d967e79adcf620534dc`.

The runner verifies all hashes, exact 107 failure metrics and decision, exact
gen11/gen12 lineage, the 107 validation-game identity, and retained gen2 before
evaluation.

## Stage 1: streaming common-holdout audit

Use exactly the 205 whole games / 12,834 rows held out by 107 and never shown
to its optimizer. Stream only those rows from the frozen corpora; do not load
the complete training set.

Evaluate identical states and policy/value targets for:

- exact gen11 (`lambda=0`) and the full 107 candidate (`lambda=1`) as anchors;
- coherent parameter interpolations
  `theta = (1-lambda)*theta_gen11 + lambda*theta_107` at prospectively fixed
  `lambda` values `0.0625, 0.125, 0.25, 0.5, 0.75`;
- a diagnostic hybrid using the full 107 network for policy logits and exact
  gen11 for value.

For each variant and replay generation, report policy cross-entropy, value MSE,
and objective `policy + 0.5*value`. Reproduce the exact parent and child metrics
stored by 107 within `1e-5`; otherwise stop as an audit mismatch.

Apply 107's unchanged retention gate to every repair variant: aggregate, gen7,
and gen8 objective deltas versus gen11 must be below zero, and gen10 delta must
be at most `+0.005`. Gen9 is reported but does not gate.

## Prospective selection

If one or more coherent interpolations pass, select the one with the lowest
aggregate objective; ties select the smaller `lambda`. Save it as a derived
generation-12 checkpoint. The hybrid cannot displace a passing coherent model.

Only if no coherent interpolation passes may the hybrid qualify as the
diagnostic fallback. If neither pathway passes, stop before gameplay. Anchor
variants are never eligible.

## Stage 2: fresh internal gameplay

On fresh seeds `1970000..1970099`, both seats, run both panels regardless of
the raw result:

1. selected raw policy versus raw exact gen11, 200 games;
2. selected searched agent versus searched exact gen11, 200 games.

For a coherent selection, both policy and value use its saved checkpoint. For
the hybrid fallback, raw uses the 107 policy (raw play has no value search) and
searched play uses 107 policy plus gen11 value. Search is fixed at 256
simulations, `c_puct=2.25`, real mode, and no root noise. Report W-D-L, share,
margin, both seats, terminations, and a 50,000-sample paired bootstrap.

The searched panel passes only at >=55% share, positive margin, both seats
>=50%, and all normal terminations. Arenas commit after every game and resume
only an exact schedule prefix.

## Scope and interpretation

- No self-play, gradient update, `035` use, installation, or promotion.
- A coherent interpolation pass supports a smaller, parent-safe consolidation
  update. A hybrid-only gameplay pass causally implicates value learning and
  supports value anchoring or differentiated policy/value optimization next.
- Failure of the hybrid to transfer means the offline policy improvement is
  insufficient for gameplay; do not redesign value training on loss alone.
- Even a gameplay pass authorizes design of the next coherent training test,
  not deployment or an external benchmark.
