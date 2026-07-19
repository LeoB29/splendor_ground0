# MCCFR Average-Policy Repair and Multi-Seed Gate (`046`)

Frozen before implementing the repaired solver or producing any `046` result.
This is a CPU-only tabular correctness experiment. It changes neither the
reduced Splendor game nor regret updates from `044`; it corrects only average-
strategy accumulation and tests the extracted policy adversarially.

## Motivation and fixed repair

`045` showed that the legacy external solver's final regret-matched policy was
unexploitable on the reduced game, while its stored average policy had
exploitability `0.293314`. Replacing only average strategies with fewer than
100 observations reduced exploitability to `0.000401`. The defect is therefore
average-policy extraction, not regret learning.

Implement the standard two-player external-sampling MCCFR simple average used
by the OpenSpiel reference implementation:

- during player `i`'s regret-update traversal, enumerate player `i`'s actions;
- sample chance and the opponent's actions exactly as in `044`;
- update regrets only at player `i`'s information sets, unchanged from `044`;
- accumulate the current strategy at opponent information sets reached during
  player `i`'s pass, once per visit and without an additional reach multiplier;
- never accumulate the traverser's strategy on that same pass;
- use the existing uniform fallback only for a never-observed information set.

In a two-player external traversal, visit probability supplies the acting
opponent's own reach weighting. This matches the published/simple reference
rule. Preserve `ExternalSamplingMCCFRSolver` unchanged as the reproducible
legacy `044` implementation; add a separately named corrected solver.

Primary references are Lanctot et al., *Monte Carlo Sampling for Regret
Minimization in Extensive Games* (NeurIPS 2009), the OpenSpiel
`ExternalSamplingSolver` simple-average implementation, and Brown et al.,
*Deep Counterfactual Regret Minimization* (ICML 2019).

## Frozen runs

- Reconstruct the exact 10,000-iteration `044` reference and require canonical
  hash `b8184ec7c9dbfa3611d014f98c674efbeae27728429b2fa90c7939e1bbe043ab`.
- Run the corrected external solver for exactly 50,000 iterations from zero
  regrets at each seed: `440046`, `440047`, `440048`, `440049`, `440050`.
- Repeat seed `440046` from zero and require byte-identical canonical payloads.
- Use the unchanged exact information-set-consistent best-response analyzer
  from `045` on every corrected average profile.
- Also report current regret-matched-policy exploitability, exact cross-play,
  information-set coverage, regret, and observation counts. These do not
  substitute for average-policy exploitability.

## Rare-information-set audit

For every seed, partition exact-reference information sets by corrected average
observations: missing/zero, `1-9`, `10-99`, `100-999`, and at least `1000`.
Construct two diagnostic hybrids:

- replace only missing corrected strategies with exact-reference strategies;
- replace every corrected strategy with fewer than 100 observations with its
  exact-reference strategy.

Record each hybrid's exploitability and its absolute change from the unmodified
corrected average. The hybrids are diagnostic only; they can never be the
deployed policy.

## Frozen gates

All are required:

1. The reconstructed exact payload hash matches `044`.
2. Corrected seed `440046` repeats byte-identically.
3. Every run has zero illegal actions and unstable action sets, finite metrics,
   at least 95% exact-information-set coverage, and normalized positive regret
   below `0.05` for both players.
4. Corrected average-policy exploitability is at most `0.020` for every seed.
5. Median corrected average-policy exploitability across the five seeds is at
   most `0.010`.
6. Corrected exploitability exceeds exact exploitability by at most `0.015` for
   every seed.
7. Each player's best-response deviation incentive is at most `0.040` in every
   run; this prevents one-sided failures from hiding behind the averaged metric.
8. Both exact/corrected cross-play values differ from exact self-play by at most
   `0.020` for every seed.
9. Filling only missing information sets changes exploitability by at most
   `0.001` for every seed.
10. Replacing all sub-100-observation strategies changes exploitability by at
    most `0.005` for every seed, demonstrating that rare states no longer carry
    the result.

## Decision

A complete pass establishes a correct tabular external-sampling average-policy
path and authorizes a separately preregistered full-game neural Deep CFR
feasibility pilot. The pilot must mirror the repaired strategy-sampling
semantics and gate advantage fitting, strategy-memory fitting, held-out
information-state ordering, runtime/memory, and exploitability proxies before
gameplay.

Any failure keeps the neural pilot and GPU rental locked. Diagnose estimator
correctness or sampling variance without loosening thresholds, switching to the
non-guaranteed final current policy, or adding iterations after seeing results.
`044` and `045` remain formally failed. Raw legal `035` remains champion, and
BGA/expert data is not used in `046`.
